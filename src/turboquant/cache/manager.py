import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import math

from ..quant.key_quantizer import TurboQuantProd
from ..quant.value_quantizer import TurboQuantValue
from .block_pool import KVBlockPool
from .routing import LayerRouting, QuantizationStrategy

class TurboQuantKVCache:
    """
    Paged KV Cache Manager for a single Transformer Layer.
    Implementation of the SOTA Logical-to-Physical Block Mapping.
    
    Architecture:
    - Slot Allocation: Manages token positions within a block.
    - Block Table: Provides the mapping for Unified Attention Kernels.
    """
    def __init__(
        self,
        layer_idx: int,
        pool: KVBlockPool,
        routing: Optional[LayerRouting] = None,
        k_bits: int = 8,
        v_bits: int = 3,
    ):
        self.layer_idx = layer_idx
        self.pool = pool
        self.block_size = pool.block_size
        self.n_heads = pool.n_heads
        self.head_dim = pool.head_dim
        
        # Bits configuration
        self.k_bits = k_bits
        self.v_bits = v_bits
        
        # Quantizers
        self.k_quantizer = TurboQuantProd(self.head_dim, bits=k_bits).to(pool.device)
        self.v_quantizer = TurboQuantValue(self.head_dim, bits=v_bits).to(pool.device)
        
        # Sequence State
        self.block_ids: List[int] = []
        self.num_tokens = 0
        
        # Pre-allocated block table (for Triton)
        # We start with a small table and expand (Logical Mapping Only)
        self.block_table = torch.zeros(0, dtype=torch.int32, device=pool.device)

    def _allocate_new_block(self):
        """Request a physical block from the global pool."""
        bid_tensor = self.pool.allocate(1)
        bid = bid_tensor.item()
        self.block_ids.append(bid)
        
        # Update block table for Triton
        self.block_table = torch.tensor(self.block_ids, dtype=torch.int32, device=self.pool.device)
        return bid

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """
        Compress and store new KV tokens into the paged pool.
        k, v: (batch=1, n_heads, seq_len=1, d)
        """
        # Ensure block availability
        if len(self.block_ids) == 0 or (self.num_tokens % self.block_size == 0):
            self._allocate_new_block()
            
        current_block_id = self.block_ids[-1]
        slot_offset = self.num_tokens % self.block_size
        
        # 1. Quantization (SOTA Hybrid Precision)
        # Input shape: (1, n_heads, 1, d) -> squeeze to (n_heads, d)
        k_in = k.view(self.n_heads, self.head_dim)
        v_in = v.view(self.n_heads, self.head_dim)
        
        k_q = self.k_quantizer.quantize(k_in)
        v_q = self.v_quantizer.quantize(v_in)
        
        # 2. Store Key to Pool (Paged Access)
        # Using [block_id, head, slot] indexing
        self.pool.k_indices[current_block_id, :, slot_offset] = k_q.mse_indices.view(self.n_heads, -1)
        self.pool.k_qjl[current_block_id, :, slot_offset] = k_q.qjl_signs.view(self.n_heads, -1)
        self.pool.k_metadata[current_block_id, :, slot_offset, 0] = k_q.residual_norms.view(-1)
        self.pool.k_metadata[current_block_id, :, slot_offset, 1] = (k_q.norms * k_q.scales.view(-1))
        
        # 3. Store Value to Pool
        self.pool.v_indices[current_block_id, :, slot_offset] = v_q.indices.view(self.n_heads, -1)
        
        # SOTA FIX: Store V metadata for EVERY token
        self.pool.v_metadata[current_block_id, :, slot_offset, 0] = v_q.scales.view(-1)
        self.pool.v_metadata[current_block_id, :, slot_offset, 1] = v_q.zero_points.view(-1)
            
        self.num_tokens += 1

    def attention_score(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores using the Paged Triton Kernel.
        query: (batch=1, n_heads, 1, head_dim)
        Returns: (batch=1, n_heads, 1, num_tokens)
        """
        if self.num_tokens == 0:
            return torch.zeros((1, self.n_heads, 1, 0), device=query.device)
            
        from ..kernels.fused_attention import paged_turboquant_attention
        
        # Unified Paged Dispatch
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        qjl_scale = math.sqrt(math.pi / 2.0) / 128
        
        return paged_turboquant_attention(
            query, self, self.k_bits, self.v_bits,
            qjl_scale, sm_scale
        )

    def get_paged_ptrs(self):
        """Returns the necessary metadata for the Paged Triton Kernel."""
        return {
            "block_table": self.block_table,
            "context_len": self.num_tokens,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits
        }

    def clear(self):
        """Free all physical blocks back to pool."""
        if self.block_ids:
            self.pool.free(torch.tensor(self.block_ids))
            self.block_ids = []
            self.block_table = torch.zeros(0, dtype=torch.int32, device=self.pool.device)
            self.num_tokens = 0
