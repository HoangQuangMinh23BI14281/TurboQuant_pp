import torch
import pytest
from turboquant.ops.sign_array import generate_sign_array, apply_sign_array, TBQ_SIGNS, QJL_SIGNS, get_llama_sign

@pytest.mark.parametrize("d", [64, 128, 256, 512])
def test_sign_values_domain(d):
    """
    Signs must be either 1.0 or -1.0.
    """
    signs = generate_sign_array(d)
    unique_vals = torch.unique(signs)
    
    # Check if only -1.0 and 1.0 are present
    for val in unique_vals:
        assert val.item() in [-1.0, 1.0]

@pytest.mark.parametrize("d", [64, 128, 256])
def test_deterministic_seed(d):
    """
    Same seed must produce same signs.
    """
    s1 = generate_sign_array(d, seed=123)
    s2 = generate_sign_array(d, seed=123)
    torch.testing.assert_close(s1, s2)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_sign_inverse(d):
    """
    Applying signs twice must return the original tensor (S * S = I).
    """
    x = torch.randn(2, d)
    signs = generate_sign_array(d)
    
    x1 = apply_sign_array(x, signs)
    x2 = apply_sign_array(x1, signs)
    
    torch.testing.assert_close(x, x2)

@pytest.mark.parametrize("d", [256])
def test_llama_presets(d):
    """
    Verify llama.cpp presets against the first few bits.
    """
    tbq = generate_sign_array(d, use_llama_preset='tbq')
    qjl = generate_sign_array(d, use_llama_preset='qjl')
    
    # TBQ_SIGNS[0] = 0xa7 (10100111) -> Bit 0 is 1 -> Sign is -1.0? 
    # Check get_llama_sign: -1.0 if ((signs[idx >> 3] >> (idx & 7)) & 1) else 1.0
    # llama.cpp logic: bit 0: (0xa7 >> 0) & 1 == 1 -> -1.0
    assert tbq[0].item() == -1.0
    assert tbq[1].item() == -1.0
    assert tbq[2].item() == -1.0
    assert tbq[3].item() == 1.0 # 0xa7 & 0x08 == 0
    
    # Periodicity check for d > 256
    tbq_large = generate_sign_array(512, use_llama_preset='tbq')
    assert torch.equal(tbq_large[:256], tbq_large[256:])

def test_apply_sign_broadcasting():
    """
    Test if apply_sign_array handles batch/head dimensions correctly.
    """
    d = 128
    x = torch.randn(2, 4, 8, d) # (batch, seq, heads, dim)
    signs = generate_sign_array(d)
    
    out = apply_sign_array(x, signs)
    assert out.shape == x.shape
    
    # Manual check for one vector
    torch.testing.assert_close(out[0, 0, 0], x[0, 0, 0] * signs)
