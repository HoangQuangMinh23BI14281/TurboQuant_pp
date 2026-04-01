"""
TurboQuant Fused Attention — PyTorch Reference Implementation.

Provides attention score computation using TurboQuant-compressed keys,
combining the MSE reconstruction with the QJL unbiased inner product estimator.

This is the PyTorch reference; a Triton fused kernel will replace the
hot path (see triton_attention.py when implemented).

Inner product estimate from TurboQuant Algorithm 2:
  <q, k> ≈ <q, k̃_mse> + ||r||₂ * sqrt(π/2)/D² * <SRHT(q_rot), sign(SRHT(r))>

Where:
  - k̃_mse = dequantized key from MSE stage
  - r = k_rot - k̃_mse_rot (residual in rotated domain)
  - SRHT = Sign * WHT (matching llama.cpp)
  - D = block_size (WHT domain size)
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from ..quant.quantizer import TurboQuantProd, TurboQuantMSE, ProdQuantized, MSEQuantized, ValueQuantized
from ..quant.lloyd_max import LM_CENTROIDS
from .fused_ref import attention_score_mse, attention_score_prod

try:
    from .triton_attention import turboquant_mse_score, turboquant_qjl_score, turboquant_fused_decode
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def attention_score_prod_dispatch(
    query: torch.Tensor,
    quantized_key: ProdQuantized,
    quantizer: TurboQuantProd,
    scale: Optional[float] = None,
    k_bits: int = 4, # Fallback, usually from config
) -> torch.Tensor:
    """
    High-level dispatcher for attention scores (Key part only).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)

    # Fast path: Triton GPU
    if HAS_TRITON and query.is_cuda:
        centroids = LM_CENTROIDS[k_bits].to(query.device, query.dtype)
        q_rot, q_sketch = quantizer.transform_query(query)
        
        # Shape adjustment
        if q_rot.dim() > 2:
            q_rot = q_rot.reshape(-1, q_rot.shape[-1])
            q_sketch = q_sketch.reshape(-1, q_sketch.shape[-1])

        qjl_factor = math.sqrt(math.pi / 2.0) / math.sqrt(quantizer.block_size)

        scores = turboquant_mse_score(
            q_rot, quantized_key.mse_indices, quantized_key.norms, quantized_key.scales,
            centroids, k_bits, mse_scale=1.0
        )
        scores = turboquant_qjl_score(
            q_sketch, quantized_key.qjl_signs, quantized_key.residual_norms, quantized_key.norms, 
            qjl_factor, out=scores
        )
        return scores.reshape(*query.shape[:-1], -1) * scale

    # Fallback path: PyTorch Reference
    return attention_score_prod(query, quantized_key, quantizer, scale)


def turboquant_attention(
    query: torch.Tensor,
    quantized_key: ProdQuantized,
    value: torch.Tensor, # Can be torch.Tensor (FP16) or ValueQuantized (3-bit)
    quantizer: TurboQuantProd,
    scale: Optional[float] = None,
    causal_mask: Optional[torch.Tensor] = None,
    k_bits: int = 4,
    v_bits: int = 4,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified Attention API with Asymmetric Hybrid Precision Support.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)

    # CASE A: Full Fused Path (Triton)
    # Both K and V are quantized, and we are on GPU
    if (HAS_TRITON and query.is_cuda and 
        isinstance(value, ValueQuantized) and 
        causal_mask is None): # Only fused decode (N=1) for now
        
        q_rot, q_sketch = quantizer.transform_query(query)
        centroids = LM_CENTROIDS[k_bits].to(query.device, query.dtype)
        qjl_factor = math.sqrt(math.pi / 2.0) / math.sqrt(quantizer.block_size)
        
        output = turboquant_fused_decode(
            q_rot, q_sketch, quantized_key, value,
            centroids, k_bits, v_bits, qjl_factor, scale, group_size
        )
        # Unpad from block_size back to head_dim if necessary
        if output.shape[-1] > quantizer.dim:
            output = output[..., :quantizer.dim]
            
        # Reshape back to query shape (Batch, Head, Dim) or (Batch, Seq, Head, Dim)
        return output.reshape(query.shape).to(query.dtype), None  # No weights returned in fused path

    # CASE B: Standard Path (De-coupled Scores + Softmax)
    # 1. Compute scores
    scores = attention_score_prod_dispatch(query, quantized_key, quantizer, scale, k_bits=k_bits)

    # 2. Apply masking
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    # 3. Softmax & Output projection
    weights = F.softmax(scores, dim=-1)
    
    # 4. Handle possibly-quantized value for matmul
    if isinstance(value, ValueQuantized):
        # We need a value dequantizer instance (placeholder, or dequantize on the fly)
        # Note: In production, we'll use a specialized V-Dequant kernel here if not using fused path.
        from ..quant.value_quantizer import TurboQuantValue
        v_dequantizer = TurboQuantValue(quantizer.dim, bits=v_bits, group_size=group_size).to(query.device)
        v_tensor = v_dequantizer.dequantize(value)
    else:
        v_tensor = value

    output = torch.matmul(weights, v_tensor.float())

    return output.to(query.dtype), weights
