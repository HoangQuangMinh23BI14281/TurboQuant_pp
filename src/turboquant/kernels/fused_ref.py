import math
import torch
from ..quant.quant_base import MSEQuantized, ProdQuantized, unpack_indices
from ..ops.wht import fwht
from ..ops.sign_array import apply_sign_array

def attention_score_mse(
    query: torch.Tensor,
    quantized_key: MSEQuantized,
    quantizer,
    scale: float = None,
) -> torch.Tensor:
    """
    Standard PyTorch reference for MSE-based attention scores.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)
    
    keys_hat = quantizer.dequantize(quantized_key)
    scores = torch.matmul(query.float(), keys_hat.float().transpose(-2, -1))
    return scores * scale

def attention_score_prod(
    query: torch.Tensor,
    quantized_key: ProdQuantized,
    quantizer,
    scale: float = None,
) -> torch.Tensor:
    """
    Standard PyTorch reference for Asymmetric QJL estimator scores.
    SOTA v8.6.11: Sub-block aware reference path.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)
 
    dim = quantizer.dim
    block_size = quantizer.block_size
    n_subblocks = quantizer.n_subblocks
    
    # Term 1: MSE part
    mse_q = MSEQuantized(
        indices=quantized_key.mse_indices,
        norms=quantized_key.norms,
        scales=quantized_key.scales,
        bits=quantized_key.mse_bits,
        packed=quantized_key.packed,
    )
    # TurboQuantMSE.dequantize is already sub-block aware
    keys_mse = quantizer.mse_quantizer.dequantize(mse_q)
    scores_mse = torch.matmul(query.float(), keys_mse.float().transpose(-2, -1))
 
    # Term 2: QJL correction
    q_rot, q_qjl_projected = quantizer.transform_query(query)
    
    k_qjl_signs = quantized_key.qjl_signs
    if quantized_key.packed:
        k_qjl_signs = unpack_indices(k_qjl_signs, 1, block_size * n_subblocks)
    
    sign_float = k_qjl_signs.float() * 2.0 - 1.0 # (..., n_k, d)
    
    # Reshape for sub-block dot products: (..., n_subblocks, block_size)
    q_reshaped = q_qjl_projected.float().view(*q_qjl_projected.shape[:-1], n_subblocks, block_size)
    k_signs_reshaped = sign_float.float().view(*sign_float.shape[:-1], n_subblocks, block_size)
    
    # Calculate dot product per sub-block: (..., n_q, n_k, n_subblocks)
    # We want q_reshaped[..., q_idx, sub_idx, :] dot k_signs_reshaped[..., k_idx, sub_idx, :]
    # (..., n_q, sub, block) @ (..., n_k, sub, block).T
    # Easier: (..., n_q, sub, block) * (..., n_k, sub, block) and sum over block
    q_unsqueezed = q_reshaped.unsqueeze(-3) # (..., n_q, 1, sub, block)
    k_unsqueezed = k_signs_reshaped.unsqueeze(-4) # (..., 1, n_k, sub, block)
    
    qjl_dot_sub = (q_unsqueezed * k_unsqueezed).sum(dim=-1) # (..., n_q, n_k, n_subblocks)
    
    # SOTA: Factor = sqrt(2.0/pi) / sqrt(block_size). 
    qjl_factor = math.sqrt(2.0 / math.pi) / math.sqrt(block_size)
    
    # Apply sub-block residual norms correctly: (..., n_k, n_subblocks)
    res_norms = quantized_key.residual_norms # (..., n_k, n_subblocks)
    
    # Scores QJL: qjl_factor * sum_over_subblocks(dot_sub * res_norms)
    scores_qjl_sub = qjl_factor * qjl_dot_sub * res_norms.unsqueeze(-3)
    scores_qjl = scores_qjl_sub.sum(dim=-1)
 
    return (scores_mse + scores_qjl) * scale
