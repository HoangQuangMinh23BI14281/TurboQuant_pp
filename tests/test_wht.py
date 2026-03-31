import torch
import pytest
import math
from turboquant.ops.wht import generate_hadamard, fwht, ifwht

@pytest.mark.parametrize("d", [64, 128, 256])
def test_orthogonality(d):
    """
    Test orthogonality of the Hadamard matrix: H * H^T = I
    """
    H = generate_hadamard(d, normalized=True)
    identity = torch.eye(d)
    
    # H * H^T
    res = torch.mm(H, H.t())
    
    # Check if H * H^T is approx identity
    torch.testing.assert_close(res, identity, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_fwht_vs_naive(d):
    """
    FWHT should produce the same result as matrix multiplication with Hadamard matrix.
    """
    H = generate_hadamard(d, normalized=True)
    x = torch.randn(2, 3, d) # Random tensor with some batch dims
    
    # FWHT(x)
    x_fwht = fwht(x)
    
    # Naive multiplication: x @ H^T (because x is row-major and we want x_rot = x * H_d)
    # The architecture doc says: K_rot = K' * S * H_d.
    # If S is identity, K_rot = K' * H_d.
    x_naive = torch.matmul(x, H.t())
    
    torch.testing.assert_close(x_fwht, x_naive, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_roundtrip(d):
    """
    ifwht(fwht(x)) should return x.
    """
    x = torch.randn(4, d)
    x_fwht = fwht(x)
    x_roundtrip = ifwht(x_fwht)
    
    torch.testing.assert_close(x, x_roundtrip, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_norm_preservation(d):
    """
    Orthonormal transforms preserve the L2 norm.
    """
    x = torch.randn(10, d)
    norm_orig = torch.norm(x, p=2, dim=-1)
    
    x_fwht = fwht(x)
    norm_fwht = torch.norm(x_fwht, p=2, dim=-1)
    
    torch.testing.assert_close(norm_orig, norm_fwht, rtol=1e-6, atol=1e-6)
