import torch
import math
import pytest
from turboquant.quant.quantizer import TurboQuantProd, MSEQuantized, ProdQuantized
from turboquant.kernels.fused_attention import attention_score_prod

@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("bits", [3, 4])
def test_qjl_correlation_improvement(d, bits):
    """
    Verify that the QJL stage (Term 2) actually improves correlation 
    over the MSE-only stage (Term 1).
    """
    torch.manual_seed(42)
    n_q, n_k = 10, 100
    q_module = TurboQuantProd(d, bits)
    
    query = torch.randn(n_q, d)
    key = torch.randn(n_k, d)
    scale = 1.0 / math.sqrt(d)
    
    # Ground Truth
    true_scores = (query @ key.T) * scale
    
    # Quantize
    quantized = q_module.quantize(key)
    
    # Term 1 Only (MSE)
    # We reconstruct keys manually to isolate Term 1
    mse_q = MSEQuantized(
        indices=quantized.mse_indices, 
        norms=quantized.norms, 
        scales=quantized.scales, 
        bits=quantized.mse_bits, 
        packed=quantized.packed
    )
    keys_mse = q_module.mse_quantizer.dequantize(mse_q)
    scores_mse = (query @ keys_mse.T) * scale
    
    # Term 1 + Term 2 (Full Prod)
    est_scores = attention_score_prod(query, quantized, q_module)
    
    # Correlation Check
    corr_mse = torch.corrcoef(torch.stack([true_scores.flatten(), scores_mse.flatten()]))[0, 1].item()
    corr_prod = torch.corrcoef(torch.stack([true_scores.flatten(), est_scores.flatten()]))[0, 1].item()
    
    # Assert improvement
    assert corr_prod > corr_mse, f"QJL stage failed to improve correlation: {corr_prod:.4f} <= {corr_mse:.4f}"
    # Target SOTA level: > 0.99 for 4-bit
    if bits >= 4:
        assert corr_prod > 0.99, f"Product correlation {corr_prod:.4f} too low for {bits}-bits"
    
    # ADVANCED METRICS FROM DEBUG_QJL
    assert q_module.mse_quantizer.n_rotation_passes >= 2, "SOTA requires at least 2 rotation passes"
    assert quantized.norms.mean() > 0, "Key norms should be positive"
    assert quantized.residual_norms.mean() < quantized.norms.mean(), "Residual should be smaller than total norm"

def test_qjl_correction_scale():
    """Verify QJL correction is within expected numerical bounds."""
    d, bits = 128, 4
    q_module = TurboQuantProd(d, bits)
    query = torch.randn(5, d)
    key = torch.randn(20, d)
    
    quantized = q_module.quantize(key)
    
    # Term 1
    mse_q = MSEQuantized(indices=quantized.mse_indices, norms=quantized.norms, scales=quantized.scales, bits=quantized.mse_bits, packed=quantized.packed)
    keys_mse = q_module.mse_quantizer.dequantize(mse_q)
    scores_mse = (query @ keys_mse.T) * (1.0 / math.sqrt(d))
    
    # Full
    scores_full = attention_score_prod(query, quantized, q_module)
    
    correction = scores_full - scores_mse
    
    # Mean correction should be small relative to mean score
    assert correction.abs().mean() < scores_mse.abs().mean(), "QJL correction dominates the attention score!"
    assert q_module.block_size == 128, "SOTA Block Size must be 128"
    
    # Range check
    assert scores_full.max() < 20.0, "Score explosion detected!"
    assert scores_full.min() > -20.0, "Score collapse detected!"
