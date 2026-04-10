import torch
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

    def append(self, k: torch.Tensor, v: torch.Tensor):
        batch, n_heads, seq_len, head_dim = k.shape
        assert batch == 1, "TurboQuant++ currently supports batch_size=1"
        
        tokens_per_block = self.pool.tokens_per_block
        
        if self.k_quantizer is not None and seq_len == 1:
            if self.block_table is None:
                # SOTA: Auto-allocate a reasonable slab for decoding if not pre-allocated
                # Usually benchmark_e2e will pre-allocate this for graphs
                self.block_table = self.pool.allocate_layer_blocks(128) 

            # SOTA: Quantize first
            k_q = self.k_quantizer.quantize(k)
            v_q = self.v_quantizer.quantize(v)

            # SOTA: Fused Cache Update (No Python Dispatch)
            fused_cache_update(self, k_q, v_q)
            
            # Increment counters (Both GPU and CPU shadow)
            # Increment counters (Both GPU and CPU shadow)
            self.num_tokens_ptr += 1
            self.num_tokens += 1
            return
        
        if self.k_quantizer is not None:
            # ========================================================
            # PREFILL PATH (seq_len > 1) - Optimizations for large chunks
            # ========================================================
            k_q = self.k_quantizer.quantize(k) 
            v_q = self.v_quantizer.quantize(v)
            k_idx_bytes = k_q.mse_indices.shape[-1]
            k_qjl_bytes = k_q.qjl_signs.shape[-1]
            v_idx_bytes = v_q.indices.shape[-1]
            
            bsz = self.k_quantizer.mse_quantizer.block_size
            k_rotated = self.k_quantizer.mse_quantizer.rotation(k.float().contiguous().view(-1, bsz))
            k_rotated = k_rotated.view(batch, n_heads, seq_len, head_dim)
            
            for i in range(seq_len):
                current_slot = self.num_tokens
                slot_offset = current_slot % tokens_per_block
                
                if slot_offset == 0:
                    new_block = self.pool.allocate_block()
                    self.block_ids.append(new_block)
                    # Sync with Tensor-based block_table for Attention kernel
                    if self.block_table is None:
                        self.block_table = self.pool.allocate_layer_blocks(128)
                    self.block_table[len(self.block_ids)-1] = new_block
                
                curr_block = self.block_ids[-1]
                li = self.layer_idx
                
                # SOTA metadata access (Using original loop for prefill precision)
                self.pool.k_indices[li, curr_block, :, slot_offset, :k_idx_bytes].copy_(k_q.mse_indices[0, :, i, :])
                self.pool.k_qjl[li, curr_block, :, slot_offset, :k_qjl_bytes].copy_(k_q.qjl_signs[0, :, i, :])
                self.pool.k_metadata[li, curr_block, :, slot_offset, :].copy_(k_q.meta[0, :, i, :])
                
                self.pool.v_indices[li, curr_block, :, slot_offset, :v_idx_bytes].copy_(v_q.indices[0, :, i, :])
                self.pool.v_metadata[li, curr_block, :, slot_offset, :].copy_(v_q.meta[0, :, i, :])
                
                # Summaries for Prefill
                k_rot_h_d = k_rotated[0, :, i, :]
                if slot_offset == 0:
                    self.pool.k_summaries[li, curr_block, :, 0].copy_(k_rot_h_d)
                    self.pool.k_summaries[li, curr_block, :, 1].copy_(k_rot_h_d)
                else:
                    torch.min(self.pool.k_summaries[li, curr_block, :, 0], k_rot_h_d, out=self.pool.k_summaries[li, curr_block, :, 0])
                    torch.max(self.pool.k_summaries[li, curr_block, :, 1], k_rot_h_d, out=self.pool.k_summaries[li, curr_block, :, 1])
                    
                self.num_tokens += 1
            if torch.cuda.is_current_stream_capturing() or 'cuda' in str(self.device):
                self.num_tokens_ptr.add_(1) # Update GPU tracker
        else:
            # ========================================================
            # NHÁNH FP16 NGUYÊN BẢN (Protected Layers)
            # ========================================================
            for i in range(seq_len):
                current_slot = self.num_tokens
                slot_offset = current_slot % tokens_per_block
                
                if slot_offset == 0:
                    self.block_ids.append(self.pool.allocate_block())
                    
                curr_block = self.block_ids[-1]
                
                if curr_block not in self.k_fp16:
                    self.k_fp16[curr_block] = torch.zeros((n_heads, tokens_per_block, head_dim), dtype=k.dtype, device=self.device)
                    self.v_fp16[curr_block] = torch.zeros((n_heads, tokens_per_block, head_dim), dtype=k.dtype, device=self.device)
                    
                self.k_fp16[curr_block][:, slot_offset] = k[:, :, i].squeeze(0)
                self.v_fp16[curr_block][:, slot_offset] = v[:, :, i].squeeze(0)
                self.num_tokens += 1
            if torch.cuda.is_current_stream_capturing() or 'cuda' in str(self.device):
                self.num_tokens_ptr.add_(1) # Update GPU tracker

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