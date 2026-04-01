import torch
import pytest
from turboquant.ops.rotation import TurboQuantRotation

@pytest.mark.parametrize("d", [64, 128, 256, 1024])
@pytest.mark.parametrize("n_passes", [1, 2, 3])
def test_cascaded_rotation_properties(d, n_passes):
    """
    EXTREME HARDENING: Verify that multi-pass Cascaded WHT preserves norm exactly (Isometry).
    ||rot(x)|| / ||x|| must be EXACTLY 1.0 at double precision.
    """
    rot = TurboQuantRotation(d, n_passes=n_passes).double()
    x = torch.randn(2, d).double()
    
    x_rot = rot(x)
    
    norm_in = torch.norm(x, p=2, dim=-1)
    norm_out = torch.norm(x_rot, p=2, dim=-1)
    
    # Orthonormal WHT is isometric: expected_ratio = 1.0
    torch.testing.assert_close(norm_out, norm_in, rtol=1e-15, atol=1e-15)

@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("n_passes", [1, 2])
def test_cascaded_rotation_roundtrip(d, n_passes):
    """
    EXTREME HARDENING: Verify x == rot.inverse(rot(x)) at double precision.
    """
    rot = TurboQuantRotation(d, n_passes=n_passes).double()
    x = torch.randn(4, d).double()
    
    x_rot = rot(x)
    x_recon = rot.inverse(x_rot)
    
    torch.testing.assert_close(x, x_recon, rtol=1e-15, atol=1e-15)

def test_cascaded_independence():
    """
    Verify that different passes use different signs.
    """
    d = 64
    rot = TurboQuantRotation(d, n_passes=2)
    # Check if S1 != S2
    assert not torch.allclose(rot.all_signs[0], rot.all_signs[1])
