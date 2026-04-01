import torch
import pytest
from turboquant.ops.rotation import TurboQuantRotation

@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("n_passes", [1, 2, 3])
def test_cascaded_rotation_properties(d, n_passes):
    """
    Verify that multi-pass Cascaded WHT preserves orthogonality and norm.
    """
    rot = TurboQuantRotation(d, n_passes=n_passes)
    x = torch.randn(2, d)
    
    # 1. Norm preservation: ||H(S(x))|| == sqrt(d) * ||x|| if H is not normalized
    # But our fwht/ifwht should ideally handle scaling or we check relative norm.
    x_rot = rot(x)
    
    # Standard WHT: ||H(x)|| = sqrt(d) * ||x||
    # 2-pass WHT: ||H(S2(H(S1(x))))|| = d * ||x||
    expected_scale = d ** (n_passes / 2.0)
    
    norm_in = torch.norm(x, p=2, dim=-1)
    norm_out = torch.norm(x_rot, p=2, dim=-1)
    
    # We check if (norm_out / norm_in) is constant and match expected_scale
    torch.testing.assert_close(norm_out / norm_in, torch.full_like(norm_in, expected_scale), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("n_passes", [1, 2])
def test_cascaded_rotation_roundtrip(d, n_passes):
    """
    Verify x == rot.inverse(rot(x))
    """
    rot = TurboQuantRotation(d, n_passes=n_passes)
    x = torch.randn(4, d)
    
    x_rot = rot(x)
    x_recon = rot.inverse(x_rot)
    
    torch.testing.assert_close(x, x_recon, rtol=1e-5, atol=1e-5)

def test_cascaded_independence():
    """
    Verify that different passes use different signs.
    """
    d = 64
    rot = TurboQuantRotation(d, n_passes=2)
    # Check if S1 != S2
    assert not torch.allclose(rot.all_signs[0], rot.all_signs[1])
