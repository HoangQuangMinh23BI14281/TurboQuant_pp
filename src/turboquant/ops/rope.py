import torch
import torch.nn as nn
from typing import Optional, Tuple

class RotaryPositionalEmbeddings(nn.Module):
    """
    Standard Rotary Positional Embeddings (RoPE).
    Optimized for TurboQuant++ Paged Attention (Hugging Face Interface).
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d: int, base: int = 1_000_000): 
        super().__init__()
        self.d = d
        self.base = base
        # Register buffers for persistence and device handling
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Dynamically builds the cosine and sine caches.
        """
        if self.cos_cached is not None and self.cos_cached.shape[2] >= seq_len:
            if self.cos_cached.device == device and self.cos_cached.dtype == dtype:
                return

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d, 2, device=device).float() / self.d))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # SOTA Layout: (1, 1, seq_len, d)
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        [x1, x2] -> [-x2, x1]
        """
        x1 = x[..., : self.d // 2]
        x2 = x[..., self.d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """
        x: (batch, heads, seq, dim)
        position_ids: (batch, seq) - used for generative indexing
        """
        batch, heads, seq_len, dim = x.shape
        
        # 1. Grow Cache if needed
        max_pos = seq_len
        if position_ids is not None:
             max_pos = position_ids.max().item() + 1
             
        self._build_cache(max_pos, x.device, x.dtype)
        
        # 2. Select Cos/Sin based on position_ids or seq_len
        if position_ids is not None:
             # Ensure position_ids are on the correct device
             pos_ids = position_ids.long().to(x.device)
             
             # Standard HF/Qwen RoPE selection:
             # cos_cached is (1, 1, max_pos, d)
             # Indexing into (max_pos, d) first -> (batch, seq, d)
             # Then unsqueeze(1) -> (batch, 1, seq, d) for broadcasting over heads
             cos = self.cos_cached.squeeze(0).squeeze(0)[pos_ids].unsqueeze(1)
             sin = self.sin_cached.squeeze(0).squeeze(0)[pos_ids].unsqueeze(1)
        else:
             cos = self.cos_cached[:, :, :seq_len, :]
             sin = self.sin_cached[:, :, :seq_len, :]
             
        # Apply RoPE (batch, heads, seq, d)
        return (x * cos) + (self._rotate_half(x) * sin)

def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10_000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional interface to apply RoPE to Query and Key tensors.
    """
    d_head = q.shape[-1]
    rope_module = RotaryPositionalEmbeddings(d_head, base=base).to(q.device, q.dtype)
    return rope_module(q), rope_module(k)
