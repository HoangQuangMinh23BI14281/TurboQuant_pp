import torch
import math
import pytest
import numpy as np
from turboquant.quant.key_quantizer import TurboQuantMSE
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
def test_v_srht_snr_gain(dim, bits):
    """
    TRA TẤN CỰC HẠN: Proving the SNR gain of SRHT-MSE over legacy Linear Quantization.
    SRHT (WHT) eliminates the 'outlier effect' in Value vectors, allowing MSE to be more effective.
    """
    torch.manual_seed(42)
    x = torch.randn(10, dim)
    # Simulate spiky V (common in LLMs)
    x[:, :4] *= 10.0 
    
    # 1. LEGACY: Linear Asymmetric Quantization (Simulated)
    # Manual min/max quantization for bits
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    scale = (x_max - x_min) / (2**bits - 1)
    q_indices = ((x - x_min) / (scale + 1e-8)).round().clamp(0, 2**bits - 1)
    x_hat_linear = q_indices * scale + x_min
    snr_linear = calculate_snr(x, x_hat_linear)
    
    # 2. SOTA: SRHT-MSE (TurboQuantValue)
    q_v = TurboQuantValue(dim, bits=bits, n_rotation_passes=1)
    res = q_v.quantize(x)
    x_hat_srht = q_v.dequantize(res)
    snr_srht = calculate_snr(x, x_hat_srht)
    
    gain = snr_srht - snr_linear
    print(f"\n[SOTA AUDIT] Dim={dim}, Bits={bits}")
    print(f"  Legacy Linear SNR: {snr_linear:.2f} dB")
    print(f"  SOTA SRHT-MSE SNR: {snr_srht:.2f} dB")
    print(f"  SNR GAIN: {gain:.2f} dB")
    
    # Assert gain is positive and significant. 
    # For spiky data, SRHT-MSE is often >2dB better.
    assert gain > 1.0, f"SRHT-MSE gain {gain:.2f}dB is too low. Check rotation logic."

def test_v_cache_block_alignment():
    """Verify that Value Cache uses per-vector Norm/Scale (SOTA v8.5)."""
    dim = 256
    q_v = TurboQuantValue(dim, bits=3)
    x = torch.randn(5, dim)
    result = q_v.quantize(x)
    
    # Metadata should be per-vector (batch, 1)
    assert result.norms.shape == (5, 1)
    assert result.scales.shape == (5, 1)
    assert hasattr(result, "indices")
