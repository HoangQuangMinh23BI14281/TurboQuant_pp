import torch
import math
from typing import Dict, List, Optional, Tuple

from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting, QuantizationStrategy

class TurboQuantKVCache:
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
        self.n_heads = pool.n_heads
        self.head_dim = pool.head_dim
        self.tokens_per_block = pool.tokens_per_block
        
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
        
    def append(self, k: torch.Tensor, v: torch.Tensor):
        """
        Append new KV tokens to the paged pool.
        k/v shape: (batch=1, n_heads, seq_len=1, head_dim)
        """
        batch, n_heads, seq_len, head_dim = k.shape
        assert batch == 1, "TurboQuant Paged Cache currently supports batch=1 append"
        
        # Check if we need a new block
        if self.num_tokens % self.tokens_per_block == 0:
            new_block_id = self.pool.allocate_block()
            self.block_table.append(new_block_id)
            
        current_block_id = self.block_table[-1]
        slot_offset = self.num_tokens % self.tokens_per_block
        
        if self.strategy == QuantizationStrategy.FP16:
            # Native FP16 Path: No compression, straight to pool
            self.pool.k_fp16[current_block_id, :, slot_offset] = k.reshape(self.n_heads, self.head_dim)
            self.pool.v_fp16[current_block_id, :, slot_offset] = v.reshape(self.n_heads, self.head_dim)
        else:
            # Quantized Path (TURBO_4BIT)
            # 1. Quantize with Mandatory Bit-Packing
            k_in = k.reshape(self.n_heads, self.head_dim)
            v_in = v.reshape(self.n_heads, self.head_dim)
            
            # Robust unpacking: handle both Prod (mse_indices) and MSE (indices) NamedTuples
            k_res = self.k_quantizer.quantize(k_in, pack=True)
            # If it has indices/mse_indices, it's a TurboQuant NamedTuple object
            k_q = k_res if (hasattr(k_res, 'indices') or hasattr(k_res, 'mse_indices')) else k_res[0]
            
            v_q = self.v_quantizer.quantize(v_in, pack=True)
            
            # 2. Store to Pool (Dynamic shapes handled by Pool allocation)
            k_raw_indices = getattr(k_q, "indices", getattr(k_q, "mse_indices", None))
            self.pool.k_indices[current_block_id, :, slot_offset] = k_raw_indices.to(torch.uint8)
            
            qjl_raw = getattr(k_q, "qjl_signs", None)
            if qjl_raw is not None:
                self.pool.k_qjl[current_block_id, :, slot_offset] = qjl_raw.to(torch.uint8)
            
            # Metadata: Norm, Scale, and Residual Norm
            self.pool.k_metadata[current_block_id, :, slot_offset, 0] = k_q.norms.flatten()
            if hasattr(k_q, 'scales') and k_q.scales is not None:
                self.pool.k_metadata[current_block_id, :, slot_offset, 1] = k_q.scales.flatten()
            if hasattr(k_q, 'residual_norms') and k_q.residual_norms is not None:
                self.pool.k_metadata[current_block_id, :, slot_offset, 2] = k_q.residual_norms.flatten()
            
            self.pool.v_indices[current_block_id, :, slot_offset] = v_q.indices.to(torch.uint8)
            self.pool.v_metadata[current_block_id, :, slot_offset, :, 0] = v_q.scales.reshape(self.n_heads, -1)
            self.pool.v_metadata[current_block_id, :, slot_offset, :, 1] = v_q.zero_points.reshape(self.n_heads, -1)
            
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

    def attention_score(self, query: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
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
            sm_scale=scale
        )
