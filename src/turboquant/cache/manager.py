import torch
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

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
        """
        Appends new KV tokens to the paged pool.
        Standardized to rank-4 inputs (batch=1, heads, seq, dim).
        """
        # 1. Shape Normalization
        # k, v are (batch, heads, seq, dim)
        batch, n_heads, seq_len, head_dim = k.shape
        assert batch == 1, "TurboQuant++ currently supports batch_size=1 (Streaming SOTA)."
        
        # 2. Block Allocation (Hardened Dispatch)
        tokens_per_block = self.pool.tokens_per_block
        for i in range(seq_len):
            current_slot = self.num_tokens
            slot_offset = current_slot % tokens_per_block
            
            if slot_offset == 0:
                # Need a new block
                new_block_id = self.pool.allocate_block()
                self.block_ids.append(new_block_id)
            
            # SOTA Pillar 3: FP16 Fallback (Always stored for protected layers)
            if self.k_quantizer is None:
                current_block_id = self.block_ids[-1]
                if current_block_id not in self.k_fp16:
                    self.k_fp16[current_block_id] = torch.zeros((n_heads, tokens_per_block, head_dim), dtype=k.dtype, device=self.device)
                    self.v_fp16[current_block_id] = torch.zeros((n_heads, tokens_per_block, head_dim), dtype=k.dtype, device=self.device)
                
                self.k_fp16[current_block_id][:, slot_offset] = k[:, :, i].squeeze(0)
                self.v_fp16[current_block_id][:, slot_offset] = v[:, :, i].squeeze(0)

            # 2. Append to Physical Pool with Layer Isolation (Quantized Path)
            if self.k_quantizer is not None:
                current_block_id = self.block_ids[-1]

                # SOTA Pillar 1: Key Quantization (MSE + Sign)
                k_in = k[:, :, i:i+1, :]
                v_in = v[:, :, i:i+1, :]

                k_q = self.k_quantizer.quantize(k_in)

                # Physical Store (Layer-Aware Mapping)
                li = self.layer_idx
                indices_flat = k_q.mse_indices.reshape(n_heads, -1)
                packed_bytes = indices_flat.shape[-1]
                self.pool.k_indices[li, current_block_id, :, slot_offset, :packed_bytes].copy_(indices_flat)

                signs_flat = k_q.qjl_signs.reshape(n_heads, -1)
                qjl_bytes = signs_flat.shape[-1]
                self.pool.k_qjl[li, current_block_id, :, slot_offset, :qjl_bytes].copy_(signs_flat)    

                # 3. SOTA Pillar 2: Key Metadata (Sub-block Interleaving v8.6)
                res_norms = k_q.residual_norms if hasattr(k_q, "residual_norms") else torch.zeros_like(k_q.norms)
                k_meta_interleaved = torch.stack([k_q.norms, k_q.scales, res_norms], dim=-1) # (H, n_sub, 3)

                li = self.layer_idx
                self.pool.k_metadata[li, current_block_id, :, slot_offset, :].copy_(k_meta_interleaved.reshape(n_heads, -1))

                # 4. SOTA Pillar 2: SRHT-MSE Value Quantization (Sub-block v8.6)
                v_q = self.v_quantizer.quantize(v_in)

                v_indices_flat = v_q.indices.reshape(n_heads, -1)
                v_bytes = v_indices_flat.shape[-1]
                self.pool.v_indices[li, current_block_id, :, slot_offset, :v_bytes].copy_(v_indices_flat)

                v_meta_interleaved = torch.stack([v_q.norms, v_q.scales], dim=-1) # (H, n_sub, 2)       
                self.pool.v_metadata[li, current_block_id, :, slot_offset, :].copy_(v_meta_interleaved.reshape(n_heads, -1))
                
                # 5. SOTA: Update block-level summaries for Quest
                # FIX 128 vs 64: Ép shape về đúng block_size trước khi xoay
                bsz = self.k_quantizer.mse_quantizer.block_size
                n_subblocks = self.k_quantizer.mse_quantizer.n_subblocks
                
                temp_k = k_in.float() # (1, H, 1, D)
                temp_k_reshaped = temp_k.contiguous().view(-1, bsz)
                
                k_rotated_reshaped = self.k_quantizer.mse_quantizer.rotation(temp_k_reshaped).clone()
                k_rotated = k_rotated_reshaped.view(1, n_heads, 1, n_subblocks, bsz)
                
                # SOTA FIX: Tính max của trị tuyệt đối (đỉnh WHT) TRÊN TỪNG SUBBLOCK
                k_rotated_max_sub = torch.max(torch.abs(k_rotated), dim=-1).values # (1, H, 1, n_sub)
                k_rotated_summary = k_rotated_max_sub.squeeze(0).squeeze(-2) # (H, n_sub)
                
                # Cập nhật vào Pool (giờ pool chứa đúng số n_sub thay vì head_dim)
                if slot_offset == 0:
                    self.pool.k_summaries[li, current_block_id, :, 0].copy_(k_rotated_summary)
                    self.pool.k_summaries[li, current_block_id, :, 1].copy_(k_rotated_summary)
                else:
                    self.pool.k_summaries[li, current_block_id, :, 0].copy_(
                        torch.min(self.pool.k_summaries[li, current_block_id, :, 0], k_rotated_summary)
                    )
                    self.pool.k_summaries[li, current_block_id, :, 1].copy_(
                        torch.max(self.pool.k_summaries[li, current_block_id, :, 1], k_rotated_summary)
                    )

            # Update count
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
        # Note: Do NOT call super().__init__() to avoid ValueError from Cache base
        self.layers = [TurboQuantKVCache(i, pool) for i in range(num_layers)]
        self.pool = pool
        self._current_seq_len = 0 # GPS Tracker

    def __getitem__(self, idx: int) -> TurboQuantKVCache:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """SOTA: Returns the sequence length relative to the given layer."""
        _ = layer_idx # Multi-layer sync in container
        return self._current_seq_len

    def get_usable_length(self, seq_len: int, layer_idx: Optional[int] = 0) -> int:
        """SOTA: Core 'transformers.Cache' method to determine mask expansion."""
        _ = (seq_len, layer_idx) # TurboQuant is non-streaming but handles paged growth
        return self._current_seq_len

    def get_max_length(self) -> Optional[int]:
        """TurboQuant++ Paged Attention is dynamically unbounded."""
        return self.pool.config.max_seq_len

    def update_seq_length(self, increment: int = 1):
        """Bắt buộc phải tự tăng biến này sau mỗi bước để đánh lừa HF"""
        self._current_seq_len += increment

    def get_mask_sizes(self, cache_position: torch.LongTensor, layer_idx: int = 0) -> Tuple[int, int]:
        """Calculates (kv_length, kv_offset) for mask preprocessing."""
        kv_length = self._current_seq_len
        kv_offset = 0 # TurboQuant performs global retrieval from block 0
        return kv_length, kv_offset

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_position: Optional[torch.LongTensor] = None, **kwargs):
        """
        Standard 'Cache.update' bridge.
        Note: Actual update happens in attention_layer.py via hijack.
        Returns the passed states as per protocol.
        """
        return key_states, value_states