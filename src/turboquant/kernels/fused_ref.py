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
    """
    if scale is None:
        scale = 1.0 / math.sqrt(quantizer.dim)

    dim = quantizer.dim
    block_size = quantizer.block_size
    
    # Term 1: MSE part
    mse_q = MSEQuantized(
        indices=quantized_key.mse_indices,
        norms=quantized_key.norms,
        scales=quantized_key.scales,
        bits=quantized_key.mse_bits,
        packed=quantized_key.packed,
    )
    keys_mse = quantizer.mse_quantizer.dequantize(mse_q)
    scores_mse = torch.matmul(query.float(), keys_mse.float().transpose(-2, -1))

    # Term 2: QJL correction
    q_rotated = quantizer.mse_quantizer.transform_query(query)
    q_signed = apply_sign_array(q_rotated, quantizer.qjl_signs)
    q_qjl_projected = fwht(q_signed)

    k_qjl_signs = quantized_key.qjl_signs
    if quantized_key.packed:
        k_qjl_signs = unpack_indices(k_qjl_signs, 1, block_size)
    
    sign_float = k_qjl_signs.float() * 2.0 - 1.0
    qjl_dot = torch.matmul(q_qjl_projected, sign_float.transpose(-2, -1).to(q_qjl_projected.dtype))
    
    qjl_factor = math.sqrt(math.pi / 2.0) / math.sqrt(block_size)
    # SOTA: Broadcast residual norms correctly across query tokens.
    # qjl_dot shape: (..., n_q, n_k)
    # residual_norms shape: (n_k, n_heads, num_groups)
    res_norms = quantized_key.residual_norms.mean(dim=-1) # (n_k, n_heads)
    
    # Standardize to (n_heads, 1, n_k) for clean broadcasting with (n_heads, n_q, n_k)
    if res_norms.ndim == 2:
        res_norms = res_norms.transpose(0, 1).unsqueeze(-2)
        
    scores_qjl = qjl_factor * qjl_dot * res_norms

    return (scores_mse + scores_qjl) * scale
