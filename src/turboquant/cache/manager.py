import torch
import math
from typing import Dict, List, Optional, Tuple

from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting, QuantizationStrategy

try:
    from transformers.cache_utils import Cache
    HAS_TRANSFORMERS = True
except ImportError:
    # Shim if transformers is not installed
    class Cache: pass
    HAS_TRANSFORMERS = False

class TurboQuantKVCache(Cache):
    """
    Paged KV Cache Manager for TurboQuant++.
    Orchestrates memory allocation across KVBlockPool and handles 
    hybrid precision/quantization routing.
    """
    def __init__(
        self,
        layer_idx: int,
        pool: KVBlockPool,
        routing: Optional[LayerRouting] = None,
        k_quantizer: Optional[torch.nn.Module] = None,
        v_quantizer: Optional[torch.nn.Module] = None,
    ):
        self.layer_idx = layer_idx
        self.pool = pool
        self.device = pool.device
        self.dtype = pool.dtype
        self.n_kv_heads = pool.n_heads
        self.n_heads = pool.n_heads # Default for standard attention
        self.head_dim = pool.head_dim
        self.tokens_per_block = pool.tokens_per_block
        self.group_size = 1 # Default for standard attention
        
        # Strategy assignment
        self.routing = routing or LayerRouting(pool.num_blocks) # Dummy routing if none
        self.strategy = self.routing.get_strategy(layer_idx)
        
        # Quantizers Initialized automatically if needed
        if self.strategy != QuantizationStrategy.FP16:
            if k_quantizer is None:
                from turboquant.quant.key_quantizer import TurboQuantProd
                self.k_quantizer = TurboQuantProd(
                    self.head_dim, 
                    bits=self.pool.k_bits
                ).to(self.device)
            else:
                self.k_quantizer = k_quantizer
                
            if v_quantizer is None:
                from turboquant.quant.value_quantizer import TurboQuantValue
                # V bits mapping: pool.v_bits is the target (e.g. 4)
                self.v_quantizer = TurboQuantValue(
                    self.head_dim, 
                    bits=self.pool.v_bits
                ).to(self.device)
            else:
                self.v_quantizer = v_quantizer
        else:
            self.k_quantizer = None
            self.v_quantizer = None
        
        # Paged Attention State
        self.block_table: List[int] = [] # sequence of physical block IDs
        self.block_ids = self.block_table # Alias for test compatibility
        self.num_tokens = 0
        self.layers = [] # Transformers Cache shim
        
    # --- Transformers Cache Interface Sim ---
    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        return self.num_tokens

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        return self.num_tokens

    def get_mask_size(self, layer_idx: Optional[int] = None) -> int:
        # For causal masking, mask size is the total sequence length
        return self.num_tokens
    
    def get_max_length(self) -> Optional[int]:
        return self.pool.num_blocks * self.tokens_per_block

    @property
    def is_compileable(self) -> bool:
        return False
        
    def append(self, k: torch.Tensor, v: torch.Tensor):
        """
        Append new KV tokens to the paged pool.
        k/v shape: (batch=1, n_heads, seq_len=1, head_dim)
        """
        batch, n_heads, seq_len, head_dim = k.shape
        assert batch == 1, "TurboQuant Paged Cache currently supports batch=1 append"
        
        # Ensure inputs are on the same device as the pool
        k = k.to(self.device)
        v = v.to(self.device)
        
        # Check if we need a new block
        if self.num_tokens % self.tokens_per_block == 0:
            new_block_id = self.pool.allocate_block()
            self.block_table.append(new_block_id)
            
        current_block_id = self.block_table[-1]
        slot_offset = self.num_tokens % self.tokens_per_block
        
        # Prepare Input View (Standardized across paths)
        k_in = k.reshape(self.n_heads, self.head_dim)
        v_in = v.reshape(self.n_heads, self.head_dim)
        
        # 1. Quest Summary Update (SOTA Sparsity - Universal)
        # Update dimension-wise Min/Max for the current block (applied to both FP16/Quant paths)
        if slot_offset == 0:
            self.pool.k_summaries[current_block_id, :, 0, :] = k_in
            self.pool.k_summaries[current_block_id, :, 1, :] = k_in
        else:
            self.pool.k_summaries[current_block_id, :, 0, :] = torch.min(self.pool.k_summaries[current_block_id, :, 0, :], k_in)
            self.pool.k_summaries[current_block_id, :, 1, :] = torch.max(self.pool.k_summaries[current_block_id, :, 1, :], k_in)
            
        if self.strategy == QuantizationStrategy.FP16:
            # Native FP16 Path: No compression, straight to pool
            self.pool.k_fp16[current_block_id, :, slot_offset] = k_in
            self.pool.v_fp16[current_block_id, :, slot_offset] = v_in
        else:
            # Quantized Path (TURBO_4BIT)
            # 1. Quantize with Mandatory Bit-Packing
            k_res = self.k_quantizer.quantize(k_in, pack=True)
            k_q = k_res if (hasattr(k_res, 'indices') or hasattr(k_res, 'mse_indices')) else k_res[0]
            v_q = self.v_quantizer.quantize(v_in, pack=True)
            
            # 2. Store to Pool
            k_raw_indices = getattr(k_q, "indices", getattr(k_q, "mse_indices", None))
            self.pool.k_indices[current_block_id, :, slot_offset] = k_raw_indices.to(torch.uint8)
            
            qjl_raw = getattr(k_q, "qjl_signs", None)
            if qjl_raw is not None:
                self.pool.k_qjl[current_block_id, :, slot_offset] = qjl_raw.to(torch.uint8)
            
            # Metadata: Norm, Scale, and Residual Norm (Ensure device match)
            self.pool.k_metadata[current_block_id, :, slot_offset, 0] = k_q.norms.flatten().to(self.device)
            if hasattr(k_q, 'scales') and k_q.scales is not None:
                self.pool.k_metadata[current_block_id, :, slot_offset, 1] = k_q.scales.flatten().to(self.device)
            if hasattr(k_q, 'residual_norms') and k_q.residual_norms is not None:
                self.pool.k_metadata[current_block_id, :, slot_offset, 2] = k_q.residual_norms.flatten().to(self.device)
            
            self.pool.v_indices[current_block_id, :, slot_offset] = v_q.indices.to(torch.uint8)
            self.pool.v_metadata[current_block_id, :, slot_offset, :, 0] = v_q.scales.reshape(self.n_heads, -1).to(self.device)
            self.pool.v_metadata[current_block_id, :, slot_offset, :, 1] = v_q.zero_points.reshape(self.n_heads, -1).to(self.device)
                
        self.num_tokens += seq_len

    def get_paged_ptrs(self) -> Dict[str, any]:
        """Return metadata for Triton execution."""
        return {
            "block_table": torch.tensor(self.block_table, dtype=torch.int32, device=self.device),
            "tokens_per_block": self.tokens_per_block,
            "strategy": self.strategy,
            "pool": self.pool,
            "num_tokens": self.num_tokens,
        }

    def attention_score(
        self, 
        query: torch.Tensor, 
        scale: Optional[float] = None,
        quest_threshold: float = 1e-4
    ) -> torch.Tensor:
        """Convenience wrapper for Paged Attention execution."""
        from turboquant.kernels.fused_attention import paged_turboquant_attention
        
        # SOTA: Fetch dynamic bits and scales for the dispatcher
        # Use mse_bits (bits-1) for Key to match ProdQuantized centroids
        k_bits = getattr(self.k_quantizer, "mse_bits", getattr(self.k_quantizer, "bits", 8))
        v_bits = getattr(self.v_quantizer, "bits", 4)
        qjl_scale = getattr(self.k_quantizer, "qjl_scale", 1.0)
        
        if scale is None:
            scale = 1.0 / (query.shape[-1] ** 0.5)
            
        return paged_turboquant_attention(
            query, 
            self, 
            k_bits=k_bits, 
            v_bits=v_bits, 
            qjl_scale=qjl_scale, 
            sm_scale=scale,
            quest_threshold=quest_threshold
        )

    def evict_idle_blocks(self, threshold: float = 1e-4, min_keep: int = 1):
        """
        H2O (Heavy Hitter Oracle): Evict blocks with low importance scores.
        threshold: Cumulative importance threshold below which a block is considered for eviction.
        min_keep: Minimum number of most recent blocks to always keep (for local context).
        """
        if len(self.block_table) <= min_keep:
            return # Nothing to evict
            
        # Get importance scores for blocks assigned to this layer
        # scores shape: (num_blocks_in_table, n_heads)
        physical_ids = torch.tensor(self.block_table[:-min_keep], dtype=torch.long, device=self.device)
        importance_scores = self.pool.block_importance[physical_ids].mean(dim=-1) # Average importance across heads
        
        # Identify indices to evict (Score < threshold)
        to_evict_mask = importance_scores < threshold
        indices_to_evict = torch.where(to_evict_mask)[0].cpu().tolist()
        
        if not indices_to_evict:
            return
            
        # Perform eviction (Reverse order to maintain indexing)
        for idx in sorted(indices_to_evict, reverse=True):
            block_id = self.block_table.pop(idx)
            self.pool.free_block(block_id)
            self.num_tokens -= self.tokens_per_block
            
        # Reset importance after eviction to allow new blocks to prove themselves
        self.pool.block_importance[physical_ids].zero_()
