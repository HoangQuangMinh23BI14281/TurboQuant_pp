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

from ..quant.quantizer import TurboQuantProd, TurboQuantMSE, ProdQuantized, MSEQuantized
from ..quant.lloyd_max import lloyd_max_dequantize, LM_CENTROIDS
from ..ops.wht import fwht
from ..ops.sign_array import apply_sign_array

try:
    from .triton_attention import turboquant_attention_score, turboquant_fused_decode, turboquant_mse_score, turboquant_qjl_score
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def attention_score_mse(
    query: torch.Tensor,
    quantized_key: MSEQuantized,
    quantizer: TurboQuantMSE,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention scores using MSE-dequantized keys.

    Simple path: fully dequantize keys, then matmul with query.

    Args:
        query: (..., n_q, dim) query vectors
        quantized_key: MSEQuantized from TurboQuantMSE.quantize()
        quantizer: the TurboQuantMSE instance (needed for rotation inverse)
        scale: attention scale factor (default: 1/sqrt(dim))

    Returns:
        scores: (..., n_q, n_k) attention logits
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)

    # Dequantize all keys
    keys_hat = quantizer.dequantize(quantized_key)  # (..., n_k, dim)

    # Standard dot product attention
    scores = torch.matmul(query.float(), keys_hat.float().transpose(-2, -1))

    return scores * scale


def attention_score_prod(
    query: torch.Tensor,
    quantized_key: ProdQuantized,
    quantizer: TurboQuantProd,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention scores using the TurboQuant asymmetric QJL estimator.

    Two terms:
      Term 1 (MSE): <q, k̃_mse>
      Term 2 (QJL): ||r|| * sqrt(π/2)/D² * <SRHT_qjl(q_rotated), sign_bits>

    This computes the QJL correction in the WHT domain for efficiency,
    matching the llama.cpp implementation in fattn-common.cuh.

    Args:
        query: (n_q, dim) or (..., n_q, dim) query vectors
        quantized_key: ProdQuantized from TurboQuantProd.quantize()
        quantizer: the TurboQuantProd instance
        scale: attention scale factor (default: 1/sqrt(dim))

    Returns:
        scores: (..., n_q, n_k) attention logits
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)

    dim = quantizer.dim
    block_size = quantizer.block_size
    # Term 1: MSE contribution

    # ── Term 1: MSE contribution ──
    # Dequantize keys using MSE stage only, then dot with query
    mse_q = MSEQuantized(
        indices=quantized_key.mse_indices,
        norms=quantized_key.norms,
        scales=quantized_key.scales,
        bits=quantized_key.mse_bits,
        packed=quantized_key.packed,
    )
    keys_mse = quantizer.mse_quantizer.dequantize(mse_q)  # (..., n_k, dim)
    
    # PyTorch dequantize is now isometric, so base scores are already in original domain
    scores_mse = torch.matmul(query.float(), keys_mse.float().transpose(-2, -1))

    # ── Term 2: QJL correction ──
    # The QJL inner product correction estimates <q, residual_in_original_domain>
    #
    # During quantize:
    #   1. key was normalized: k_unit = k / ||k||
    #   2. k_unit was padded and rotated (Isometrically): k_rot = Rot(pad(k_unit))
    #   3. Scale = ||k_rot|| / sqrt(D)
    #   4. Quantized: k_hat_rot = centroids * Scale
    #   5. Residual (Rotated): r_rot = k_rot - k_hat_rot
    #   6. QJL Signs: sign_bits = sign(WHT(qjl_signs * r_rot))
    #
    # Final Estimator (Isometry):
    #   <q, k_res> = ||k|| * <q_rot, r_rot>
    #   ≈ ||k|| * ||r_rot|| * (sqrt(pi/2) / D) * <q_sketch, sign_bits>
    #

    # Rotate query using quantizer helper
    q_rotated = quantizer.mse_quantizer.transform_query(query)  # (..., block_size)

    # Step 2: Apply QJL SRHT to rotated query (same as applied to residual)
    qjl_signs_array = quantizer.qjl_signs  # (block_size,)
    q_signed = apply_sign_array(q_rotated, qjl_signs_array)
    q_qjl_projected = fwht(q_signed)  # (..., n_q, block_size)

    # Step 3: Dot with stored sign bits
    qjl_signs_packed = quantized_key.qjl_signs
    if quantized_key.packed:
        from ..quant.quantizer import unpack_indices
        k_qjl_signs = unpack_indices(qjl_signs_packed, 1, block_size)
    else:
        k_qjl_signs = qjl_signs_packed
    
    sign_float = k_qjl_signs.float() * 2.0 - 1.0  # (..., n_k, block_size)
    qjl_dot = torch.matmul(q_qjl_projected, sign_float.transpose(-2, -1).to(q_qjl_projected.dtype))  # (..., n_q, n_k)

    # 1. Orthonormal WHT is isometric: no D^n growth.
    # 2. QJL bit-sum normalization: 1/sqrt(D) (for orthonormal WHT)
    # 3. Sign-to-inner-product scale: sqrt(pi/2)
    qjl_factor = math.sqrt(math.pi / 2.0) / math.sqrt(block_size)
    
    # residual_norms is already in FULL domain per mandatory formula.
    residual_norms = quantized_key.residual_norms
    d_qjl = residual_norms

    scores_qjl = qjl_factor * qjl_dot * d_qjl.unsqueeze(-2)

    # Combined score
    if HAS_TRITON and query.is_cuda:
        # Use Triton for score computation if on GPU
        # Extract centroids
        centroids = LM_CENTROIDS[quantizer.mse_bits].to(query.device, query.dtype)
        
        # We need the rotation matrix Pi and QJL matrix S for the dispatcher
        # However, our dispatcher currently handles query transformation internally
        # but the Triton kernels in triton_attention.py expect pre-transformed queries.
        # Let's use the high-level dispatcher from triton_attention.
        
        # Wait, the triton_attention_score dispatcher in triton_attention.py
        # expects Pi and S. In our case, Pi and S are sequences of ops.
        # I should probably update the dispatcher in triton_attention.py 
        # to accept the quantizer object OR perform the transformations here.
        
        # Pre-transform query for Triton
        q_rot, q_sketch = quantizer.transform_query(query)
        q_rot = q_rot.squeeze(-2) if q_rot.dim() > query.dim() else q_rot
        q_sketch = q_sketch.squeeze(-2) if q_sketch.dim() > query.dim() else q_sketch
        
        # No energy growth in Triton matmul with orthonormal WHT. 
        mse_scale = 1.0
        block_size = quantizer.mse_quantizer.block_size
        
        # Calculate QJL factor (includes sqrt(pi/2) and D isometry)
        qjl_factor = math.sqrt(math.pi / 2.0) / math.sqrt(block_size)

        # Score calculation in Triton
        scores = turboquant_mse_score(
            q_rot, quantized_key.mse_indices, quantized_key.norms, quantized_key.scales,
            centroids, quantizer.mse_bits,
            mse_scale=mse_scale
        )
        scores = turboquant_qjl_score(
            q_sketch, quantized_key.qjl_signs, quantized_key.residual_norms, quantized_key.norms, 
            qjl_factor, out=scores
        )
        return scores.reshape(*query.shape[:-1], -1) * scale

    return (scores_mse + scores_qjl) * scale


def turboquant_attention(
    query: torch.Tensor,
    quantized_key: ProdQuantized,
    value: torch.Tensor,
    quantizer: TurboQuantProd,
    scale: Optional[float] = None,
    causal_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full attention computation with TurboQuant-compressed keys.

    Args:
        query: (batch, n_q, dim) query vectors
        quantized_key: ProdQuantized from quantizer.quantize()
        value: (batch, n_k, dim) value vectors (dequantized or original)
        quantizer: TurboQuantProd instance
        scale: attention scale (default: 1/sqrt(dim))
        causal_mask: optional (n_q, n_k) boolean mask (True = attend)

    Returns:
        (output, weights):
            output: (batch, n_q, dim) attention output
            weights: (batch, n_q, n_k) attention weights
    """
    # Dispatch to fused Triton kernel if possible
    if HAS_TRITON and query.is_cuda and value.dim() <= 3:
        # Note: requires ValueQuantized for full fusion.
        # If value is raw FP16, we can still use Triton for scores but need softmax separately.
        # For now, let's stick to the score-only Triton path for maximum compatibility.
        pass

    # Compute scores using QJL estimator (uses Triton internally if available)
    scores = attention_score_prod(query, quantized_key, quantizer, scale)

    # Apply causal mask
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    # Softmax
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values (using float32 for accumulation)
    output = torch.matmul(weights, value.float())

    return output.to(query.dtype), weights
