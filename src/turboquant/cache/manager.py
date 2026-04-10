import torch
import math
from typing import Dict, Any, List, Optional, Tuple

from turboquant.cache.block_pool import KVBlockPool
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.kernels.cache_ops import fused_cache_update  # Module-level for CUDA Graph safety
from transformers.cache_utils import Cache

class TurboQuantKVCache:
    """
    SOTA Persistent Paged Cache Manager for TurboQuant++.
    Orchestrates block-level indexing, quantization, and recovery.
    """
    def __init__(self, layer_idx: int, pool: KVBlockPool):
        self.layer_idx = layer_idx
        self.pool = pool
        self.device = pool.device
        self.n_heads = pool.n_heads
        self.n_kv_heads = pool.n_heads # SOTA Alias for dispatcher
        self.head_dim = pool.head_dim
        self.is_compileable = False
        
        # Block Management (CUDA Graph optimized)
        self.block_table: Optional[torch.Tensor] = None
        self.num_tokens_ptr = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.num_tokens = 0 # Python shadow for metadata/prefill
        
        # SOTA: Persistent Metadata Buffers
        self.k_quantizer: Optional[TurboQuantProd] = None
        self.v_quantizer: Optional[Any] = None
        
        # SOTA v8.6: Centroid Caching (Zero-Sync Dispatch)
        self.k_centroids: Optional[torch.Tensor] = None
        self.v_centroids: Optional[torch.Tensor] = None
        
        # SOTA Pillar 3: FP16 Hybrid Support (Exempt Strategy)
        self.k_fp16: Dict[int, torch.Tensor] = {}
        self.v_fp16: Dict[int, torch.Tensor] = {}
        self.block_ids = [] # SOTA: Maintain list for regular Python paths

    def _append_decode_fused(self, k_q: Any, v_q: Any):
        """SOTA: Fused cache update for decoding (seq_len=1)."""
        if self.block_table is None:
            # SOTA: Auto-allocate a modest slab for decoding
            self.block_table = self.pool.allocate_blocks_bulk(128) 

        # SOTA: Fused Cache Update (No Python Dispatch)
        fused_cache_update(self, k_q, v_q)
        
        # SOTA: In-place increment (Async)
        self.num_tokens_ptr.add_(1)
        self.num_tokens += 1

    def _append_prefill_vectorized(self, k_q: Any, v_q: Any, seq_len: int, k_rotated: Optional[torch.Tensor] = None):
        """SOTA: Vectorized block-level copy for prefill (seq_len > 1)."""
        tokens_per_block = self.pool.tokens_per_block
        li = self.layer_idx
        
        start_token = self.num_tokens
        end_token = self.num_tokens + seq_len
        
        # ... logic for allocation (same as before) ...
        current_blocks = len(self.block_ids)
        total_blocks_needed = math.ceil(end_token / tokens_per_block)
        if total_blocks_needed > current_blocks:
            new_blocks = self.pool.allocate_blocks_bulk(total_blocks_needed - current_blocks)
            self.block_ids.extend(new_blocks.tolist())
            if self.block_table is None:
                self.block_table = torch.zeros(1024, dtype=torch.int32, device=self.device)
            self.block_table[current_blocks:total_blocks_needed] = new_blocks

        # 2. Block-Level Vectorized Copy
        for b_idx in range(start_token // tokens_per_block, total_blocks_needed):
            block_start_abs = b_idx * tokens_per_block
            block_end_abs = (b_idx + 1) * tokens_per_block
            
            seq_start = max(0, block_start_abs - start_token)
            seq_end = min(seq_len, block_end_abs - start_token)
            
            slot_start = max(0, start_token - block_start_abs)
            slot_end = min(tokens_per_block, end_token - block_start_abs)
            
            p_block = self.block_ids[b_idx]
            
            self.pool.k_indices[li, p_block, :, slot_start:slot_end, :].copy_(k_q.mse_indices[0, :, seq_start:seq_end, :])
            self.pool.k_qjl[li, p_block, :, slot_start:slot_end, :].copy_(k_q.qjl_signs[0, :, seq_start:seq_end, :])
            self.pool.k_metadata[li, p_block, :, slot_start:slot_end, :].copy_(k_q.meta[0, :, seq_start:seq_end, :])
            self.pool.v_indices[li, p_block, :, slot_start:slot_end, :].copy_(v_q.indices[0, :, seq_start:seq_end, :])
            self.pool.v_metadata[li, p_block, :, slot_start:slot_end, :].copy_(v_q.meta[0, :, seq_start:seq_end, :])

            # 3. Vectorized Summaries (Quest Sparsity)
            if k_rotated is not None:
                k_rot_slice = k_rotated[0, :, seq_start:seq_end, :]
                if slot_start == 0:
                    self.pool.k_summaries[li, p_block, :, 0].copy_(k_rot_slice.amin(dim=1))
                    self.pool.k_summaries[li, p_block, :, 1].copy_(k_rot_slice.amax(dim=1))
                else:
                    # Partial block update
                    torch.min(self.pool.k_summaries[li, p_block, :, 0], k_rot_slice.amin(dim=1), out=self.pool.k_summaries[li, p_block, :, 0])
                    torch.max(self.pool.k_summaries[li, p_block, :, 1], k_rot_slice.amax(dim=1), out=self.pool.k_summaries[li, p_block, :, 1])

        self.num_tokens += seq_len
        self.num_tokens_ptr.fill_(self.num_tokens)

    def append(self, k: torch.Tensor, v: torch.Tensor, k_q: Optional[Any] = None, v_q: Optional[Any] = None, k_rotated: Optional[torch.Tensor] = None):
        batch, n_heads, seq_len, head_dim = k.shape
        assert batch == 1, "TurboQuant++ currently supports batch_size=1"
        
        if self.k_quantizer is not None:
            if k_q is None:
                k_q = self.k_quantizer.quantize(k)
            if v_q is None:
                v_q = self.v_quantizer.quantize(v)
            
            if seq_len == 1:
                self._append_decode_fused(k_q, v_q)
            else:
                self._append_prefill_vectorized(k_q, v_q, seq_len, k_rotated=k_rotated)
        else:
            # FP16 Protected Path
            tokens_per_block = self.pool.tokens_per_block
            for i in range(seq_len):
                current_slot = self.num_tokens
                slot_offset = current_slot % tokens_per_block
                
                if slot_offset == 0:
                    self.block_ids.append(self.pool.allocate_block())
                    
                curr_block = self.block_ids[-1]
                
                if curr_block not in self.k_fp16:
                    self.k_fp16[curr_block] = torch.zeros((n_heads, tokens_per_block, head_dim), dtype=k.dtype, device=self.device)
                    self.v_fp16[curr_block] = torch.zeros((n_heads, tokens_per_block, head_dim), dtype=k.dtype, device=self.device)
                    
                self.k_fp16[curr_block][:, slot_offset] = k[0, :, i]
                self.v_fp16[curr_block][:, slot_offset] = v[0, :, i]
                self.num_tokens += 1
            self.num_tokens_ptr.fill_(self.num_tokens)


    def init_static_fp16_workspace(self, n_heads: int, head_dim: int, max_seq_len: int):
        """Pre-allocate a flat buffer for protected layers (CUDA Graph safe)."""
        self.static_k_fp16 = torch.zeros((1, n_heads, max_seq_len, head_dim), device=self.device, dtype=torch.float16)
        self.static_v_fp16 = torch.zeros((1, n_heads, max_seq_len, head_dim), device=self.device, dtype=torch.float16)
        # Mask starts as -1e4 (attended nothing) - FP16 safe (min is -65504)
        self.static_attn_mask = torch.full((1, 1, 1, max_seq_len), -10000.0, device=self.device, dtype=torch.float16)

    def get_paged_ptrs(self) -> Dict[str, Any]:
        """Return metadata for Triton execution (CUDA Graph friendly)."""
        return {
            "num_tokens": self.num_tokens_ptr, # Passing the tensor pointer
            "tokens_per_block": self.pool.tokens_per_block,
            "block_table": self.block_table,
            "pool": self.pool,
        }

class TurboQuantCacheContainer(Cache):
    """
    HuggingFace-compatible Cache Container.
    Wraps multiple TurboQuantKVCache layers.
    Implements the 'Cache' interface for transformers>=4.36 compatibility.
    """
    def __init__(self, num_layers: int, pool: KVBlockPool):
        self.layers = [TurboQuantKVCache(i, pool) for i in range(num_layers)]
        self.pool = pool
        self._current_seq_len = 0 # GPS Tracker

    def __getitem__(self, idx: int) -> TurboQuantKVCache:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._current_seq_len

    def get_usable_length(self, seq_len: int, layer_idx: Optional[int] = 0) -> int:
        return self._current_seq_len

    def get_max_length(self) -> Optional[int]:
        return self.pool.config.max_seq_len

    def update_seq_length(self, increment: int = 1):
        self._current_seq_len += increment

    def get_mask_sizes(self, cache_position: torch.LongTensor, layer_idx: int = 0) -> Tuple[int, int]:
        kv_length = self._current_seq_len
        kv_offset = 0
        return kv_length, kv_offset

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_position: Optional[torch.LongTensor] = None, **kwargs):
        return key_states, value_states