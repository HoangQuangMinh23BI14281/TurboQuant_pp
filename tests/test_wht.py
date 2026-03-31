import torch
import pytest
import math
from turboquant.ops.wht import fwht, ifwht, generate_hadamard

@pytest.mark.parametrize("d", [64, 128, 256])
def test_orthogonality(d):
    """
    Check if the generate_hadamard(normalized=True) creates an orthonormal matrix.
    H @ H.T should be identity.
    """
    H = generate_hadamard(d, normalized=True)
    I = torch.eye(d)
    
    # H * H^T
    HHT = torch.matmul(H, H.t())
    torch.testing.assert_close(HHT, I, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_fwht_vs_naive(d):
    """
    FWHT should produce the same result as matrix multiplication with raw Hadamard matrix.
    Matching llama.cpp scaling: fwht(x) == x @ H (non-normalized).
    """
    H = generate_hadamard(d, normalized=False).double()
    x = torch.randn(2, 3, d).double()
    
    # FWHT(x)
    x_fwht = fwht(x)
    
    # Naive multiplication: x @ H (H is symmetric, H.t() == H)
    x_naive = torch.matmul(x, H.t())
    
    # At double precision, the butterfly logic matches naive multiplication exactly
    torch.testing.assert_close(x_fwht, x_naive, rtol=1e-12, atol=1e-12)

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
    For non-normalized WHT, the L2 norm grows by sqrt(d).
    ||fwht(x)|| = sqrt(d) * ||x||
    """
    x = torch.randn(10, d)
    norm_orig = torch.norm(x, p=2, dim=-1)
    
    x_fwht = fwht(x)
    norm_fwht = torch.norm(x_fwht, p=2, dim=-1)
    
    # Expected norm: norm_orig * sqrt(d)
    torch.testing.assert_close(norm_fwht, norm_orig * math.sqrt(float(d)), rtol=1e-6, atol=1e-6)
