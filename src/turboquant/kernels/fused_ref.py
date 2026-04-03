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
    q_rotated, q_qjl_projected = quantizer.transform_query(query)
    
    k_qjl_signs = quantized_key.qjl_signs
    if quantized_key.packed:
        k_qjl_signs = unpack_indices(k_qjl_signs, 1, block_size)
    
    sign_float = k_qjl_signs.float() * 2.0 - 1.0 # (n_k, d) or (batch, heads, n_k, d)
    
    # Matching Triton accumulation: Dot(Sketch_Q, Signs_K)
    # q_qjl_projected: (..., n_q, d)
    # sign_float: (..., n_k, d)
    qjl_dot = torch.matmul(q_qjl_projected.to(torch.float32), sign_float.transpose(-1, -2).to(torch.float32))
    
    # SOTA: Factor = sqrt(2.0/pi) / sqrt(block_size). 
    # This factor matches TurboQuantProd and the Triton kernel exactly.
    qjl_factor = math.sqrt(2.0 / math.pi) / math.sqrt(block_size)
    
    # Broadcast residual norms correctly
    # quantized_key.residual_norms: (..., n_k)
    res_norms = quantized_key.residual_norms.unsqueeze(-2) # (..., 1, n_k)
    
    scores_qjl = qjl_factor * qjl_dot * res_norms

    return (scores_mse + scores_qjl) * scale
