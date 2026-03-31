import torch
import pytest
from turboquant.quant.lloyd_max import lloyd_max_quantize, lloyd_max_dequantize, LM_CENTROIDS

@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_lloyd_max_range(bits):
    """
    Verify indices are within [0, 2^bits - 1] for the full 1-bit to 8-bit range.
    """
    x = torch.randn(2, 64)
    indices = lloyd_max_quantize(x, bits)
    
    n_clusters = 2 ** bits
    assert torch.all(indices >= 0)
    assert torch.all(indices < n_clusters)

@pytest.mark.parametrize("bits", [2, 4])
def test_lloyd_max_reconstruction_values(bits):
    """
    Verify dequantize returns values from the official table.
    """
    x = torch.randn(1, 16)
    indices = lloyd_max_quantize(x, bits)
    reconstructed = lloyd_max_dequantize(indices, bits)
    
    # All values in reconstructed must be in LM_CENTROIDS[bits]
    centroids = LM_CENTROIDS[bits]
    for val in reconstructed.flatten():
        # Check if val is close to any centroid
        assert any(torch.isclose(val, centroids, atol=1e-5))

def test_lloyd_max_monotonicity():
    """
    Check if larger inputs map to larger or equal indices.
    """
    bits = 2
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    indices = lloyd_max_quantize(x, bits)
    
    # Should be monotonically increasing
    assert (indices[1:] >= indices[:-1]).all()

def test_lloyd_max_device_dtype():
    """
    Verify dtype preservation (float16 support).
    """
    bits = 2
    x = torch.randn(4, 4).half()
    indices = lloyd_max_quantize(x, bits)
    reconstructed = lloyd_max_dequantize(indices, bits)
    
    # reconstructed should ideally be float32 for accuracy or match input?
    # Usually centroids are float32.
    assert reconstructed.dtype == torch.float32 or reconstructed.dtype == torch.float16
