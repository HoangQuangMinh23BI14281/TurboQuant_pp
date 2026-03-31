import torch
import torch.nn as nn
from typing import Optional, Tuple

class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        Args:
            d (int): Dimension of the head (must be even).
            base (int): Base for the exponential frequency increment.
        """
        super().__init__()
        self.d = d
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Precomputes the cos and sin caches for a given sequence length.
        """
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        # inv_freq = 1.0 / (base ** (arange(0, d, 2) / d))
        # shape: (d / 2)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.d, 2, device=device).float() / self.d))
        
        # t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        
        # freqs = einsum(t, inv_freq) -> (seq_len, d / 2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Emb = [freqs, freqs] -> (seq_len, d)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin: (seq_len, 1, 1, d) to support broadcasting with (seq, batch, heads, d) 
        # or (batch, heads, seq, d) depending on layout. 
        # Note: We use [:, None, None, :] to match notebook's (seq, batch, head, d) assumption.
        self.cos_cached = emb.cos()[:, None, None, :].to(dtype)
        self.sin_cached = emb.sin()[:, None, None, :].to(dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        [x1, x2] -> [-x2, x1]
        """
        x1 = x[..., : self.d // 2]
        x2 = x[..., self.d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch, num_heads, d_head)
        Returns:
            torch.Tensor: Tensor with RoPE applied.
        """
        seq_len = x.shape[0]
        self._build_cache(seq_len, x.device, x.dtype)
        
        # BROADCASTING: self.cos_cached is (seq_len, 1, 1, d)
        # x is (seq_len, batch, heads, d)
        return (x * self.cos_cached[:seq_len]) + (self._rotate_half(x) * self.sin_cached[:seq_len])

def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10_000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional interface to apply RoPE to Query and Key tensors.
    Assumes shape (seq_len, batch, num_heads, d_head).
    """
    d_head = q.shape[-1]
    rope_module = RotaryPositionalEmbeddings(d_head, base=base).to(q.device, q.dtype)
    return rope_module(q), rope_module(k)
