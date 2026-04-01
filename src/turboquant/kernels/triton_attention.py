import torch
from .triton_mse import turboquant_mse_score
from .triton_qjl import turboquant_qjl_score
from .triton_fused import turboquant_fused_decode, _turboquant_fused_decode_kernel
from .triton_utils import _get_packing_params

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

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)
        scales = scales.reshape(BH_actual, -1)

    # Note: centroids should be fetched from the quantizer/config context
    # Usually handled by the high-level fused_attention.py dispatcher
    from ..quant.lloyd_max import LM_CENTROIDS
    centroids = LM_CENTROIDS[mse_bits].to(q_rot.device, q_rot.dtype)

    # Combined score calculation
    scores = turboquant_mse_score(q_rot, mse_packed, norms, scales, centroids, mse_bits)
    scores = turboquant_qjl_score(q_sketch, qjl_signs, res_norms, norms, qjl_scale, out=scores)

    return scores
