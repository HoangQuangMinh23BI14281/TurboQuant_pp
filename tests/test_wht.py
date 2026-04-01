import torch
import pytest
import math
from turboquant.ops.wht import fwht, ifwht, generate_hadamard

@pytest.mark.parametrize("d", [64, 128, 256, 512])
def test_orthogonality(d):
    """
    EXTREME HARDENING: Check if generate_hadamard(normalized=True) is bit-perfect orthonormal.
    H @ H.T must be identity at double precision (1e-15).
    """
    H = generate_hadamard(d, normalized=True).double()
    I = torch.eye(d, dtype=torch.float64, device=H.device)
    
    HHT = torch.matmul(H, H.t())
    torch.testing.assert_close(HHT, I, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_fwht_vs_naive(d):
    """
    EXTREME HARDENING: FWHT vs Naive Matmul at double precision.
    The butterfly logic must match naive multiplication EXACTLY (1e-15).
    """
    H = generate_hadamard(d, normalized=True).double()
    x = torch.randn(2, 3, d).double()
    
    x_fwht = fwht(x)
    x_naive = torch.matmul(x, H.t())
    
    torch.testing.assert_close(x_fwht, x_naive, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_roundtrip(d):
    """
    EXTREME HARDENING: ifwht(fwht(x)) roundtrip at double precision (1e-15).
    """
    x = torch.randn(4, d).double()
    x_roundtrip = ifwht(fwht(x))
    
    torch.testing.assert_close(x, x_roundtrip, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_norm_preservation(d):
    """
    EXTREME HARDENING: Isometry check at double precision.
    ||fwht(x)|| must equal ||x|| within machine epsilon (1e-15).
    """
    x = torch.randn(10, d).double()
    norm_orig = torch.norm(x, p=2, dim=-1)
    norm_fwht = torch.norm(fwht(x), p=2, dim=-1)
    
    torch.testing.assert_close(norm_fwht, norm_orig, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preservation(dtype):
    """
    EXTREME HARDENING: FWHT must preserve input dtype (No silent casts).
    """
    x = torch.randn(4, 64, dtype=dtype)
    out = fwht(x)
    assert out.dtype == dtype
    
    out_inv = ifwht(out)
    assert out_inv.dtype == dtype
