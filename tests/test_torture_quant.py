import torch
import pytest
import math
from src.turboquant.quant.key_quantizer import TurboQuantProd
from src.turboquant.quant.value_quantizer import TurboQuantValue

@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("bits", [4])
def test_outlier_hell_key(dim, bits):
    """
    TORTURE TEST: Extreme outliers in Key Cache.
    Tests if Lloyd-Max Gaussian LUT can handle values that are 1000x the mean.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantizer = TurboQuantProd(dim, bits=bits + 1).to(device) # +1 because Prod = MSE(bits) + QJL(1)
    
    # Create a vector with a massive outlier
    x = torch.randn((1, dim), device=device)
    x[0, 0] = 1000.0  # The "Hell" outlier
    
    # Quantize and dequantize
    q_data = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q_data)
    
    # Check if the outlier is preserved with reasonable relative error
    # (Lloyd-Max will clip, but the refined_gamma should help minimize MSE)
    cos_sim = torch.nn.functional.cosine_similarity(x.float(), x_hat.float(), dim=-1)
    
    print(f"Outlier Sim: {cos_sim.item():.4f}")
    assert cos_sim.item() > 0.95, f"Quantization collapsed under extreme outlier (Sim: {cos_sim.item()})"


@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("group_size", [64])
def test_zero_variance_hell_value(dim, group_size):
    """
    TORTURE TEST: Zero-variance groups in Value Cache.
    Tests if asymmetric Min-Max avoids division by zero or NaN.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantizer = TurboQuantValue(dim, bits=4, group_size=group_size).to(device)
    
    # Create a vector where one group is all zeros, and another is all same values
    x = torch.randn((1, dim), device=device)
    x[0, 0:group_size] = 0.0          # All zeros group
    x[0, group_size:2*group_size] = 42.0  # Constant value group
    
    # Quantize and dequantize
    q_data = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q_data)
    
    assert not torch.isnan(x_hat).any(), "NaN detected in dequantized output"
    assert not torch.isinf(x_hat).any(), "Inf detected in dequantized output"
    
    # Precision check for the constant group
    const_recon = x_hat[0, group_size:2*group_size]
    torch.testing.assert_close(const_recon, x[0, group_size:2*group_size], atol=1e-2, rtol=1e-2)


def test_wht_precision_hell():
    """
    TORTURE TEST: Deep WHT Precision.
    Tests the cumulative error of nested FWHT transforms in Float16.
    """
    from src.turboquant.ops.wht import fwht, ifwht
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2048
    
    x = torch.randn((1, dim), device=device, dtype=torch.float16)
    
    # Apply FWHT many times (simulating multi-pass rotation)
    y = x.clone()
    for _ in range(10):
        y = fwht(y)
        y = ifwht(y)
        
    # Check drift from original
    error = torch.mean(torch.abs(x - y))
    print(f"Cumulative WHT Drift (10 passes): {error.item():.6f}")
    assert error.item() < 1e-2, "WHT Drift exceeded hell-level tolerance"

if __name__ == "__main__":
    # Quick manual burn
    test_outlier_hell_key(128, 4)
    test_zero_variance_hell_value(128, 64)
    test_wht_precision_hell()
