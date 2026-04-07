import torch
from typing import Dict, Any, List, Optional, Tuple

from turboquant.cache.block_pool import KVBlockPool
from turboquant.quant.key_quantizer import TurboQuantProd
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
        
        # Block Management
        self.block_ids: List[int] = []
        self.num_tokens = 0
        
        # SOTA: Persistent Metadata Buffers
        self.k_quantizer: Optional[TurboQuantProd] = None
        self.v_quantizer: Optional[Any] = None
        
        # SOTA v8.6: Centroid Caching (Zero-Sync Dispatch)
        self.k_centroids: Optional[torch.Tensor] = None
        self.v_centroids: Optional[torch.Tensor] = None
        
        # SOTA Pillar 3: FP16 Hybrid Support (Exempt Strategy)
        self.k_fp16: Dict[int, torch.Tensor] = {}
        self.v_fp16: Dict[int, torch.Tensor] = {}

    def append(self, k: torch.Tensor, v: torch.Tensor):
        batch, n_heads, seq_len, head_dim = k.shape
        assert batch == 1, "TurboQuant++ currently supports batch_size=1"
        
        tokens_per_block = self.pool.tokens_per_block
        
        if self.k_quantizer is not None:
            # ========================================================
            # NHÁNH LƯỢNG TỬ HÓA (Batch Quantization - SOTA Speed)
            # ========================================================
            k_q = self.k_quantizer.quantize(k) 
            v_q = self.v_quantizer.quantize(v)
            
            # SOTA FIX: Lấy số lượng bytes nén thực tế để tránh lỗi Mismatch
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
                    self.block_ids.append(self.pool.allocate_block())
                
                curr_block = self.block_ids[-1]
                li = self.layer_idx
                
                # Chép dữ liệu nén của Key (Dùng slice dynamic để an toàn tuyệt đối)
                self.pool.k_indices[li, curr_block, :, slot_offset, :k_idx_bytes].copy_(k_q.mse_indices[0, :, i, :])
                self.pool.k_qjl[li, curr_block, :, slot_offset, :k_qjl_bytes].copy_(k_q.qjl_signs[0, :, i, :])
                
                k_meta = torch.stack([k_q.norms[0,:,i,:], k_q.scales[0,:,i,:], k_q.residual_norms[0,:,i,:]], dim=-1)
                self.pool.k_metadata[li, curr_block, :, slot_offset, :].copy_(k_meta.reshape(n_heads, -1))
                
                # Chép dữ liệu nén của Value
                self.pool.v_indices[li, curr_block, :, slot_offset, :v_idx_bytes].copy_(v_q.indices[0, :, i, :])
                
                v_meta = torch.stack([v_q.norms[0,:,i,:], v_q.scales[0,:,i,:]], dim=-1)
                self.pool.v_metadata[li, curr_block, :, slot_offset, :].copy_(v_meta.reshape(n_heads, -1))
                
                # Cập nhật Quest Summaries (Min/Max)
                k_rot_h_d = k_rotated[0, :, i, :]
                if slot_offset == 0:
                    self.pool.k_summaries[li, curr_block, :, 0].copy_(k_rot_h_d)
                    self.pool.k_summaries[li, curr_block, :, 1].copy_(k_rot_h_d)
                else:
                    self.pool.k_summaries[li, curr_block, :, 0].copy_(
                        torch.min(self.pool.k_summaries[li, curr_block, :, 0], k_rot_h_d)
                    )
                    self.pool.k_summaries[li, curr_block, :, 1].copy_(
                        torch.max(self.pool.k_summaries[li, curr_block, :, 1], k_rot_h_d)
                    )
                    
                self.num_tokens += 1
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

    def get_paged_ptrs(self) -> Dict[str, Any]:
        """Return metadata for Triton execution."""
        return {
            "num_tokens": self.num_tokens,
            "tokens_per_block": self.pool.tokens_per_block,
            "block_table": torch.tensor(self.block_ids, device=self.device, dtype=torch.int32),
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