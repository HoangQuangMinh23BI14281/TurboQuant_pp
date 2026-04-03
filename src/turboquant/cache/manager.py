import math
import torch
import torch.nn.functional as F
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
            
        # SOTA: Lazy FP16 Storage (Allocated only for FP16-routed layers)
        self.k_fp16: Dict[int, torch.Tensor] = {}
        self.v_fp16: Dict[int, torch.Tensor] = {}
        
        # Paged Attention State
        self.block_table = []
        self.block_ids = self.block_table # Alias for test compatibility
        self.num_tokens = 0
        self.layers = [] # Transformers Cache shim
        self.layer_class_to_replicate = None # SOTA Transformers compat
        self.offloading = False # SOTA Transformers compat
        
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
        
        # SOTA: Prefill Path (Handle chunking BEFORE allocation)
        if seq_len > 1:
            for i in range(0, seq_len):
                self.append(k[:, :, i:i+1, :], v[:, :, i:i+1, :])
            return

        # Ensure inputs are on the same device as the pool
        k_in = k.to(self.device).reshape(self.n_heads, self.head_dim)
        v_in = v.to(self.device).reshape(self.n_heads, self.head_dim)
        
        # Check if we need a new block
        slot_offset = self.num_tokens % self.tokens_per_block
        if slot_offset == 0:
            current_block_id = self.pool.allocate_block()
            self.block_table.append(current_block_id)
        else:
            current_block_id = self.block_table[-1]
        
        # 1. Quest Summary Update (SOTA Sparsity - Rotated Domain Sync)
        # SOTA: Compute Rotated Domain (WHT) before MSE for importance summary
        if self.strategy != QuantizationStrategy.FP16:
            k_unit = k_in / (torch.norm(k_in, p=2, dim=-1, keepdim=True) + 1e-10)
            # SOTA: transform_query handles padding (64 -> 128) and multi-head logic
            k_rotated = self.k_quantizer.mse_quantizer.transform_query(k_unit.float())
        else:
            # FP16 layers use original domain for summary, but must PAD to padded_head_dim
            if self.pool.padded_head_dim != self.head_dim:
                k_rotated = F.pad(k_in.float(), (0, self.pool.padded_head_dim - self.head_dim))
            else:
                k_rotated = k_in.float()

        if slot_offset == 0:
            self.pool.k_summaries[current_block_id, :, 0, :] = k_rotated.float()
            self.pool.k_summaries[current_block_id, :, 1, :] = k_rotated.float()
        else:
            self.pool.k_summaries[current_block_id, :, 0, :] = torch.min(self.pool.k_summaries[current_block_id, :, 0, :], k_rotated.float())
            self.pool.k_summaries[current_block_id, :, 1, :] = torch.max(self.pool.k_summaries[current_block_id, :, 1, :], k_rotated.float())
            
        if self.strategy == QuantizationStrategy.FP16:
            # Native FP16 Path: Allocated only when Manager identifies FP16 strategy
            if current_block_id not in self.k_fp16:
                self.k_fp16[current_block_id] = torch.zeros(
                    (self.n_heads, self.tokens_per_block, self.head_dim),
                    dtype=self.dtype, device=self.device
                )
                self.v_fp16[current_block_id] = torch.zeros(
                    (self.n_heads, self.tokens_per_block, self.head_dim),
                    dtype=self.dtype, device=self.device
                )
            self.k_fp16[current_block_id][:, slot_offset] = k_in
            self.v_fp16[current_block_id][:, slot_offset] = v_in
        else:
            # 2. Key Cache with SOTA Pillar 2: Sticky Metadata
            if slot_offset == 0:
                k_q = self.k_quantizer.quantize(k_in, pack=True)
                self.pool.k_metadata[current_block_id, :, 0] = k_q.norms.flatten().to(self.device).to(torch.float32)
                self.pool.k_metadata[current_block_id, :, 1] = k_q.scales.flatten().to(self.device).to(torch.float32)
                self.pool.k_metadata[current_block_id, :, 2] = k_q.residual_norms.flatten().to(self.device).to(torch.float32)
                
                self.pool.k_metadata[current_block_id, :, 2] = k_q.residual_norms.flatten().to(self.device).to(torch.float32)
            else:
                pre_norms = self.pool.k_metadata[current_block_id, :, 0].reshape(self.n_heads).to(self.device)
                pre_scales = self.pool.k_metadata[current_block_id, :, 1].reshape(self.n_heads, 1).to(self.device)
                pre_res_norms = self.pool.k_metadata[current_block_id, :, 2].reshape(self.n_heads).to(self.device)
                
                k_q = self.k_quantizer.quantize(
                    k_in, pack=True, 
                    precomputed_norms=pre_norms, 
                    precomputed_scales=pre_scales,
                    precomputed_res_norms=pre_res_norms
                )
                
                # 3. Value Cache with SOTA Pillar 2: Block-wide scaling
            if slot_offset == 0:
                v_q = self.v_quantizer.quantize(v_in, pack=True)
                self.pool.v_metadata[current_block_id, :, :, 0] = v_q.scales.reshape(self.n_heads, -1).to(self.device).to(torch.float32)
                self.pool.v_metadata[current_block_id, :, :, 1] = v_q.zero_points.reshape(self.n_heads, -1).to(self.device).to(torch.float32)
                v_indices = v_q.indices
            else:
                # Fetch block-aligned metadata
                b_scale = self.pool.v_metadata[current_block_id, :, :, 0].reshape(self.n_heads, -1)
                b_zero = self.pool.v_metadata[current_block_id, :, :, 1].reshape(self.n_heads, -1)
                
                # Quantize relative to STICKY block metadata
                v_grouped = v_in.reshape(self.n_heads, -1, self.v_quantizer.group_size)
                v_scale = b_scale.reshape(self.n_heads, -1, 1)
                v_zero = b_zero.reshape(self.n_heads, -1, 1)
                
                vi = torch.round((v_grouped - v_zero) / (v_scale + 1e-10)).clamp(0, self.v_quantizer.n_levels - 1).to(torch.uint8)
                v_indices = vi.reshape(self.n_heads, -1)
                from turboquant.quant.quant_base import pack_indices
                if self.v_quantizer.bits <= 4:
                    v_indices = pack_indices(v_indices, self.v_quantizer.bits)

            # 4. Final Store to Pool
            self.pool.k_indices[current_block_id, :, slot_offset] = k_q.mse_indices
            self.pool.k_qjl[current_block_id, :, slot_offset] = k_q.qjl_signs
            self.pool.v_indices[current_block_id, :, slot_offset] = v_indices.to(torch.uint8)
            
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
        # Use total bits; the kernel dispatcher will solve for mse_bits (bits-1)
        k_bits = self.k_quantizer.bits if self.k_quantizer else 4
        v_bits = self.v_quantizer.bits if self.v_quantizer else 4
        qjl_scale = self.k_quantizer.qjl_scale if self.k_quantizer else 1.0
        
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

    # def evict_idle_blocks(self, threshold: float = 1e-4, min_keep: int = 1):
    #     """
    #     H2O (Heavy Hitter Oracle): Frozen to prevent logical-physical mapping corruption.
    #     """
    #     pass
