import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from turboquant.layers.config import TurboQuantConfig
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.kernels.fused_attention import turboquant_attention
from turboquant.kernels.fused_attention import paged_turboquant_attention
from turboquant.cache.routing import QuantizationStrategy

class TurboQuantAttention(nn.Module):
    """
    Standard TurboQuant++ Attention Layer with Architectural Branching.
    Routes between Native FP16 (Exempt) and Dual-LUT Quantized (Active) paths.
    """
    def __init__(
        self,
        tq_config: TurboQuantConfig,
        layer_idx: int,
        total_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.tq_config = tq_config
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        
        # 1. Standard Projections (Crucial: dim -> dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // (num_heads // num_kv_heads), bias=False)
        self.v_proj = nn.Linear(dim, dim // (num_heads // num_kv_heads), bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # 2. Strategy-based Quantization
        strategy = tq_config.get_strategy(layer_idx, total_layers)
        self.is_protected = (strategy == QuantizationStrategy.FP16)
        self.k_bits = tq_config.k_bits if not self.is_protected else 16
        self.v_bits = tq_config.v_bits if not self.is_protected else 16
        
        # 3. Local Quantizers (Conditional)
        if not self.is_protected:
            self.k_quantizer = TurboQuantProd(self.head_dim, bits=self.k_bits)
            self.v_quantizer = TurboQuantValue(self.head_dim, bits=self.v_bits)
        else:
            self.k_quantizer = None
            self.v_quantizer = None
        # SOTA: RoPE Support (Injected by Patcher)
        self.rotary_emb = None
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[TurboQuantKVCache] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input query shape: (batch, seq, dim)
        """
        batch, seq_q, _ = query.shape
        _, seq_k, _ = key.shape
        
        # 1. Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. Reshape to head-split format: (batch, n_heads, seq, head_dim)
        q = q.view(batch, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply SOTA RoPE
        if self.rotary_emb is not None and position_ids is not None:
            # Note: Many modern HF models use self.rotary_emb(v_proj_output, position_ids) 
            # to get cos/sin, then apply it to q/k.
            cos, sin = self.rotary_emb(v, position_ids)
            
            # Helper to rotate 2D part
            def _rotate_half(x):
                x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)

            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)

        if kv_cache is not None:
            # Update quantizers in cache manager
            kv_cache.k_quantizer = self.k_quantizer
            kv_cache.v_quantizer = self.v_quantizer
            
            # Paged Attention (Architecture-Agnostic Entry Point)
            k_bits = self.k_quantizer.bits - 1 if self.k_quantizer else 4
            v_bits = self.v_quantizer.bits if self.v_quantizer else 4
            qjl_scale = getattr(self.k_quantizer, "qjl_scale", 1.0)
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)
            out = paged_turboquant_attention(q, kv_cache, k_bits, v_bits, qjl_scale, sm_scale)
            
            # Response handling from kernels/paged_fused.py wrapper
            # Triton wrapper returns (batch * heads, head_dim) or (batch, heads, seq, head_dim)
            # Standardize output to (batch, seq, dim)
            if out.dim() == 2:
                # out is (batch * heads, head_dim) (case for seq=1)
                out = out.view(batch, self.num_heads, 1, self.head_dim)
            
            # Now it's (batch, heads, seq, head_dim)
            # Recombine heads
            out = out.transpose(1, 2).contiguous().view(batch, seq_q, self.dim)
            return self.o_proj(out), None
            
        else:
            # Standard Attention (Fallback/Prefill)
            if self.num_heads != self.num_kv_heads:
                # GQA: Repeat KV heads
                k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            # Reshape back to (batch, seq, dim) before projection
            out = out.transpose(1, 2).contiguous().view(batch, seq_q, self.dim)
            return self.o_proj(out), None
