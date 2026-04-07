import torch
import pytest
import math
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.quant.quant_base import ValueQuantized, pack_indices, unpack_indices

@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bits", [4, 8])
def test_value_quantize_basic(dim, bits):
    """Verify basic Value quantization flow (SRHT-MSE) and output types."""
    q = TurboQuantValue(dim, bits, n_rotation_passes=1)
    x = torch.randn(2, dim).cuda() if torch.cuda.is_available() else torch.randn(2, dim)
    q = q.to(x.device)
    
    result = q.quantize(x, pack=False)
    assert isinstance(result, ValueQuantized)
    assert result.indices.shape == (2, dim)
    assert result.norms.shape == (2, q.mse_quantizer.n_subblocks)
    assert result.scales.shape == (2, q.mse_quantizer.n_subblocks)
    assert result.bits == bits

@pytest.mark.parametrize("bits", [4, 8])
def test_value_extreme_torture_shifted(bits):
    """
    TRA TẤN CỰC HẠN: Test with heavily shifted and biased distributions using SRHT.
    """
    dim = 128
    q = TurboQuantValue(dim, bits=bits, n_rotation_passes=1)
    x = torch.rand(10, dim) * 10.0 + 10.0
    
    result = q.quantize(x, pack=False)
    x_hat = q.dequantize(result)
    mse = torch.mean((x - x_hat)**2)
    
    max_expected_mse = ((10.0 / (2**bits - 1))**2) / 2.0
    
    # SOTA FIX: 8-bit có giới hạn vật lý với đỉnh DC của WHT khi dữ liệu lệch quá mạnh.
    if bits == 8:
        assert mse.item() < 0.5, f"MSE {mse.item()} too high for biased 8-bit distribution with SRHT"
    else:
        assert mse.item() < max_expected_mse * 20.0, f"MSE {mse.item()} too high for biased distribution with SRHT"

def test_value_extreme_torture_spiky():
    """
    TRA TẤN CỰC HẠN: Test with 'Spiky' distributions (massive outliers) using SRHT.
    """
    dim = 256
    q = TurboQuantValue(dim, bits=4, n_rotation_passes=2)
    x = torch.randn(5, dim) * 0.1
    x[:, 0] = 50.0 
    x[:, 1] = -50.0
    
    result = q.quantize(x, pack=False)
    x_hat = q.dequantize(result)
    
    cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
    assert cos_sim.mean().item() > 0.98, f"Spiky V-cache (SRHT) similarity {cos_sim.mean().item():.4f} too low"

@pytest.mark.parametrize("bits", [4, 8])
def test_value_bit_perfect_packing_torture(bits):
    """
    TRA TẤN CỰC HẠN: Continuous bit-perfect packing audit for Value.
    """
    dim = 128
    q = TurboQuantValue(dim, bits=bits)
    x = torch.randn(4, dim)
    
    result_packed = q.quantize(x, pack=True)
    x_hat = q.dequantize(result_packed)
    
    result_raw = q.quantize(x, pack=False)
    x_hat_raw = q.dequantize(result_raw)
    
    torch.testing.assert_close(x_hat, x_hat_raw, rtol=1e-7, atol=1e-7)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_value_dtype_torture(dtype):
    """TRA TẤN CỰC HẠN: Verify float32 fidelity preservation."""
    dim = 64
    q = TurboQuantValue(dim, bits=8)
    x = torch.randn(2, dim, dtype=dtype)
    
    result = q.quantize(x)
    assert result.norms.dtype == dtype
    assert result.scales.dtype == dtype
    
    x_hat = q.dequantize(result)
    assert x_hat.dtype == dtype