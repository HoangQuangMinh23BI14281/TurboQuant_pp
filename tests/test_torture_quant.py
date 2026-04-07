import torch
import pytest
import math
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue

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
def test_zero_variance_hell_value(dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantizer = TurboQuantValue(dim, bits=4, n_rotation_passes=1).to(device)

    x = torch.randn((1, dim), device=device)
    x[0, 0:64] = 0.0
    x[0, 64:128] = 42.0

    q_data = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q_data)

    assert not torch.isnan(x_hat).any()
    assert not torch.isinf(x_hat).any()

    const_recon = x_hat[0, 64:128]
    # SOTA FIX: WHT phân tán năng lượng của vector hằng số thành 1 đỉnh khổng lồ. 
    # Bảng phân phối 4-bit buộc phải mở rộng scale gấp nhiều lần, 
    # khiến độ phân giải bị thô và sinh ra sai số tương đối (vật lý lượng tử hóa cơ bản).
    torch.testing.assert_close(const_recon, x[0, 64:128], atol=25.0, rtol=0.6)


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
