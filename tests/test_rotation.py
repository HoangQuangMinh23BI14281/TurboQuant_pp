import torch
import pytest
from turboquant.ops.rotation import TurboQuantRotation, apply_srht
from turboquant.ops.wht import fwht, ifwht
from turboquant.ops.sign_array import generate_sign_array, apply_sign_array

@pytest.mark.parametrize("d", [64, 128, 256])
def test_rotation_roundtrip(d):
    """
    Check if rotation followed by inverse rotation returns the original tensor.
    S * H * H_inv * S_inv = I
    """
    rot = TurboQuantRotation(d, pattern='tbq')
    x = torch.randn(4, 2, d)
    
    # Forward rotation
    x_rot = rot(x)
    
    # Inverse rotation
    x_recon = rot.inverse(x_rot)
    
    torch.testing.assert_close(x, x_recon, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("pattern", ['tbq', 'qjl'])
def test_rotation_manual_consistency(d, pattern):
    """
    Check if TurboQuantRotation class matches manual steps.
    """
    rot = TurboQuantRotation(d, pattern=pattern)
    x = torch.randn(1, d)
    
    # Class forward
    x_rot_class = rot(x)
    
    # Manual forward
    signs = generate_sign_array(d, use_llama_preset=pattern)
    x_rot_manual = fwht(apply_sign_array(x, signs))
    
    torch.testing.assert_close(x_rot_class, x_rot_manual)

def test_apply_srht_functional():
    """
    Check functional interface apply_srht.
    """
    d = 128
    x = torch.randn(4, d)
    
    x_rot = apply_srht(x, pattern='qjl')
    
    # Reference
    rot = TurboQuantRotation(d, pattern='qjl')
    x_rot_ref = rot(x)
    
    torch.testing.assert_close(x_rot, x_rot_ref)

def test_rotation_device_dtype():
    """
    Test if rotation module handles casting correctly.
    """
    d = 64
    rot = TurboQuantRotation(d, pattern='tbq')
    
    # Test float16 (if available)
    x = torch.randn(1, d).half()
    rot = rot.half()
    
    # This should not crash and should correctly handle signs buffer
    x_rot = rot(x)
    assert x_rot.dtype == torch.float16
