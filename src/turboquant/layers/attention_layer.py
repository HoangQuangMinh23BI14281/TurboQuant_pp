import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import TurboQuantConfig
from ..quant.key_quantizer import TurboQuantProd
from ..quant.value_quantizer import TurboQuantValue
from ..kernels.fused_attention import turboquant_attention

class TurboQuantAttention(nn.Module):
    """
    Production-grade attention layer with Hybrid Precision & Boundary Protection.
    Designed as a drop-in component for Llama/Qwen architectures.
    """
    def __init__(
        self,
        config: TurboQuantConfig,
        layer_idx: int,
        total_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        # Determine if this layer is protected (FP16) or quantized
        self.is_protected = config.is_protected(layer_idx, total_layers)
        self.k_bits, self.v_bits = config.get_bits(layer_idx)

        if not self.is_protected:
            # Initialize specialized quantizers for this layer
            self.k_quantizer = TurboQuantProd(self.head_dim, bits=self.k_bits)
            self.v_quantizer = TurboQuantValue(self.head_dim, bits=self.v_bits, group_size=config.group_size)
        else:
            self.k_quantizer = None
            self.v_quantizer = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with automated routing logic.
        """
        # Case 1: Boundary Protection (FP16 Path)
        if self.is_protected:
            # Standard attention path
            if scale is None:
                scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if causal_mask is not None:
                scores = scores.masked_fill(~causal_mask, float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, value)
            return output, weights

        # Case 2: Hybrid Precision Path (TurboQuant++ Port)
        # 1. Quantize Key and Value
        # Note: In a real KV-cache system, this would happen per-token or block-wise.
        # Here we demonstrate the per-layer routing logic.
        q_key = self.k_quantizer.quantize(key)
        q_value = self.v_quantizer.quantize(value)

        # 2. Dispatch to optimized kernels
        return turboquant_attention(
            query, q_key, q_value, 
            quantizer=self.k_quantizer,
            scale=scale,
            causal_mask=causal_mask,
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            group_size=self.config.group_size
        )
