import torch
import pytest
from turboquant.ops.sign_array import generate_sign_array, apply_sign

@pytest.mark.parametrize("d", [64, 128, 256])
def test_sign_values_domain(d):
    """
    Check if the sign array S only contains elements from {-1, 1}.
    """
    S = generate_sign_array(d)
    
    # Check if elements are in {-1, 1}
    assert torch.all((S == 1) | (S == -1))

@pytest.mark.parametrize("d", [64, 128, 256])
def test_deterministic_seed(d):
    """
    Same seed should produce same sign array.
    """
    S1 = generate_sign_array(d, seed=1234)
    S2 = generate_sign_array(d, seed=1234)
    S3 = generate_sign_array(d, seed=1235)
    
    assert torch.all(S1 == S2)
    assert not torch.all(S1 == S3)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_sign_inverse(d):
    """
    A sign-flip S is its own inverse (S * S = I for diagonal S).
    """
    x = torch.randn(2, d)
    S = generate_sign_array(d)
    
    # apply S twice
    x_rotated = apply_sign(x, S)
    x_roundtrip = apply_sign(x_rotated, S)
    
    # S * S * x = x
    torch.testing.assert_close(x, x_roundtrip, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_apply_sign_broadcasting(d):
    """
    Apply sign-flip to multidimensional tensors.
    """
    x = torch.randn(2, 4, 8, d)
    S = generate_sign_array(d)
    
    x_rotated = apply_sign(x, S)
    
    assert x_rotated.shape == x.shape
    assert not torch.allclose(x, x_rotated)
