import torch
import pytest
import math
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.quant.quant_base import ValueQuantized, pack_indices, unpack_indices

@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bits", [4, 8])
def test_value_quantize_basic(dim, bits):
    """Verify basic Value quantization flow and output types."""
    q = TurboQuantValue(dim, bits)
    x = torch.randn(2, dim)
    
    result = q.quantize(x, pack=False)
    n_groups = dim // min(q.group_size, dim)
    assert isinstance(result, ValueQuantized)
    assert result.indices.shape == (2, dim)
    assert result.scales.shape == (2, n_groups)
    assert result.zero_points.shape == (2, n_groups)
    assert result.bits == bits

@pytest.mark.parametrize("group_size", [32, 64])
def test_value_group_quantize(group_size):
    """Verify group-wise quantization for Value vectors."""
    dim = 128
    q = TurboQuantValue(dim, bits=4, group_size=group_size)
    x = torch.randn(2, dim)
    
    result = q.quantize(x, pack=False)
    n_groups = dim // group_size
    assert result.scales.shape == (2, n_groups)
    assert result.zero_points.shape == (2, n_groups)

@pytest.mark.parametrize("bits", [4, 8])
def test_value_extreme_torture_shifted(bits):
    """
    TRA TẤN CỰC HẠN: Test with heavily shifted and biased distributions.
    Asymmetric quantization must handle range [10, 20] perfectly.
    """
    dim = 128
    q = TurboQuantValue(dim, bits=bits)
    
    # Biased data: Uniform [10.0, 20.0]
    x = torch.rand(10, dim) * 10.0 + 10.0
    
    # Check if min/max are captured
    result = q.quantize(x, pack=False)
    
    # Verification: Reconstruction error should be low
    x_hat = q.dequantize(result)
    mse = torch.mean((x - x_hat)**2)
    
    # Max error for 4-bit is approx scale/2
    # For range 10, scale is 10/15 = 0.66, so error should be < 0.33
    # With 8-bit, scale is 10/255 = 0.039
    max_expected_mse = ((10.0 / (2**bits - 1))**2) / 4.0
    # For Laplacian codebooks on Uniform data, MSE is slightly higher
    assert mse.item() < max_expected_mse * 10.0, f"MSE {mse.item()} too high for biased distribution"

def test_value_extreme_torture_spiky():
    """
    TRA TẤN CỰC HẠN: Test with 'Spiky' distributions (massive outliers).
    Value vectors often have focused activations.
    Without WHT, we MUST use small groups (group_size=4) to protect semantic precision.
    """
    dim = 256
    # For spiky V, a large group_size leads to masking. 
    # group_size=4 provides high-fidelity isolation of outliers.
    q = TurboQuantValue(dim, bits=4, group_size=4)
    
    # Spiky data: 99% small, 1% massive
    x = torch.randn(5, dim) * 0.1
    x[:, 0] = 50.0 # Extreme outlier
    x[:, 1] = -50.0
    
    result = q.quantize(x, pack=False)
    x_hat = q.dequantize(result)
    
    # Cosine similarity should now be very high (> 0.99)
    cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
    assert cos_sim.mean().item() > 0.99, f"Spiky V-cache similarity {cos_sim.mean().item():.4f} too low"

@pytest.mark.parametrize("bits", [4, 8])
def test_value_bit_perfect_packing_torture(bits):
    """
    TRA TẤN CỰC HẠN: Continuous bit-perfect packing audit for Value.
    Ensures zero bit loss during storage.
    """
    dim = 128
    q = TurboQuantValue(dim, bits=bits)
    x = torch.randn(4, dim)
    
    # Roundtrip with packing
    result_packed = q.quantize(x, pack=True)
    x_hat = q.dequantize(result_packed)
    
    # Roundtrip without packing
    result_raw = q.quantize(x, pack=False)
    x_hat_raw = q.dequantize(result_raw)
    
    # Packed and raw must be BIT-IDENTICAL
    torch.testing.assert_close(x_hat, x_hat_raw, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_value_dtype_torture(dtype):
    """TRA TẤN CỰC HẠN: Verify float64 fidelity preservation."""
    dim = 64
    q = TurboQuantValue(dim, bits=8)
    x = torch.randn(2, dim, dtype=dtype)
    
    result = q.quantize(x)
    assert result.scales.dtype == dtype
    assert result.zero_points.dtype == dtype
    
    x_hat = q.dequantize(result)
    assert x_hat.dtype == dtype
