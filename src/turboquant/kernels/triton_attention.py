import torch
from .triton_mse import turboquant_mse_score
from .triton_qjl import turboquant_qjl_score
from .triton_fused import turboquant_fused_decode, _turboquant_fused_decode_kernel
from typing import Optional

def _get_packing_params(bits: int):
    """
    Determines packing configuration for quantized indices.
    """
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return bits, 8 // bits
    else:
        return 8, 1

# This file acts as a legacy gateway and central dispatcher for Triton kernels.
# All actual kernel logic has been moved to:
# - triton_mse.py
# - triton_qjl.py
# - triton_fused.py
# - triton_utils.py

def turboquant_attention_score(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    quantized_key,
    mse_bits: int,
    qjl_scale: float,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    """
    Dispatcher for attention score computation.
    Requires pre-rotated/sketched queries.
    """
    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms
    scales = quantized_key.scales
    
    # centroids fetched based on bit precision
    from ..quant.lloyd_max import LM_CENTROIDS
    centroids = LM_CENTROIDS[mse_bits].to(q_rot.device, q_rot.dtype)

    # Combined score calculation
    scores = turboquant_mse_score(q_rot, mse_packed, norms, scales, centroids, mse_bits, block_n=block_n, num_warps=num_warps)
    scores = turboquant_qjl_score(q_sketch, qjl_signs, res_norms, norms, qjl_scale, out=scores, block_n=block_n, num_warps=num_warps)

    return scores
