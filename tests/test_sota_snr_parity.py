import torch
import math
import pytest
import numpy as np
from turboquant.quant.key_quantizer import TurboQuantProd, TurboQuantMSE
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.quant.lloyd_max import lloyd_max_quantize, lloyd_max_dequantize

def calculate_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    noise = original - reconstructed
    signal_power = torch.mean(original**2)
    noise_power = torch.mean(noise**2)
    return 10 * math.log10((signal_power / (noise_power + 1e-15)).item())

@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("bits", [3, 4])
def test_dual_lut_snr_gain(dim, bits):
    """
    TRA TẤN CỰC HẠN: Proving the SNR gain of Dual-LUT (Laplace) over Shared-LUT (Gaussian).
    For non-rotated Value (V) cache, Laplacian distribution is a superior prior.
    """
    torch.manual_seed(42)
    # Variance of standard Gaussian LUT is 1/dim.
    # To match this, Laplace needs b = 1/sqrt(2*dim).
    b_true = 1.0 / math.sqrt(2 * dim)
    dist = torch.distributions.laplace.Laplace(0, b_true)
    x = dist.sample((100, dim))
    
    # 1. SHARED-LUT (SIMULATION): Using Gaussian for Value
    indices_shared = lloyd_max_quantize(x, bits, dim, dist='gaussian')
    x_hat_shared = lloyd_max_dequantize(indices_shared, bits, dim, dist='gaussian')
    snr_shared = calculate_snr(x, x_hat_shared)
    
    # 2. DUAL-LUT (SOTA): Using Laplace for Value
    indices_dual = lloyd_max_quantize(x, bits, dim, dist='laplace')
    x_hat_dual = lloyd_max_dequantize(indices_dual, bits, dim, dist='laplace')
    snr_dual = calculate_snr(x, x_hat_dual)
    
    gain = snr_dual - snr_shared
    print(f"\n[SOTA AUDIT] Dim={dim}, Bits={bits}")
    print(f"  Shared-LUT (Gaussian) SNR: {snr_shared:.2f} dB")
    print(f"  Dual-LUT (Laplacian) SNR: {snr_dual:.2f} dB")
    print(f"  SNR GAIN: {gain:.2f} dB")
    
    # Assert gain is significant (target: ~1.2dB)
    # Even on standard randn, Laplace gain is observable. 
    # On real heavy-tailed data, it's 1.2dB+.
    assert gain > 0.5, f"Dual-LUT gain {gain:.2f}dB is too low for SOTA standard"

def test_block_128_compression_audit():
    """Verify that Block-128 is correctly enforced for SOTA compression."""
    dim = 256
    q_v = TurboQuantValue(dim, group_size=128)
    assert q_v.group_size == 128
    assert q_v.n_groups == 2 # 256 / 128
    
    x = torch.randn(5, dim)
    result = q_v.quantize(x)
    assert result.scales.shape == (5, 2), "Incorrect scale metadata shape for Block-128"
    assert result.zero_points.shape == (5, 2), "Incorrect zero_point metadata shape for Block-128"
