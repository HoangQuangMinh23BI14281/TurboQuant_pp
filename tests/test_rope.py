import torch
import pytest
from turboquant.ops.rope import apply_rotary_pos_emb

def _get_cos_sin(dim: int, seq_len: int, device="cpu"):
    """Helper to generate cos/sin for testing."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().view(1, 1, seq_len, dim), emb.sin().view(1, 1, seq_len, dim)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_rope_identity_at_zero(d):
    """
    RoPE at position 0 should be an identity (no rotation).
    """
    cos, sin = _get_cos_sin(d, 1)
    # At pos 0, cos=1, sin=0
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)
    
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    # At pos 0, it should be identical
    torch.testing.assert_close(q, q_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k, k_out, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_rope_relative_position(d):
    """
    Check if the dot product is sensitive only to relative positions.
    <f(q, m), f(k, m+n)> should be equal to <f(q, 0), f(k, n)>.
    """
    torch.manual_seed(42)
    q_vec = torch.randn(1, 1, 1, d)
    k_vec = torch.randn(1, 1, 1, d)
    
    cos, sin = _get_cos_sin(d, 10)
    
    # Apply RoPE at different positions
    # Case 1: m=2, n=5 (distance 3)
    q2_out, _ = apply_rotary_pos_emb(q_vec, q_vec, cos[:,:,2:3,:], sin[:,:,2:3,:])
    k5_out, _ = apply_rotary_pos_emb(k_vec, k_vec, cos[:,:,5:6,:], sin[:,:,5:6,:])
    dot_2_5 = torch.sum(q2_out * k5_out)
    
    # Case 2: m=0, n=3 (distance 3)
    q0_out, _ = apply_rotary_pos_emb(q_vec, q_vec, cos[:,:,0:1,:], sin[:,:,0:1,:])
    k3_out, _ = apply_rotary_pos_emb(k_vec, k_vec, cos[:,:,3:4,:], sin[:,:,3:4,:])
    dot_0_3 = torch.sum(q0_out * k3_out)
    
    # Both should be equal as they have the same relative distance
    torch.testing.assert_close(dot_2_5, dot_0_3, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_apply_rope_functional(d):
    """
    Test the functional interface apply_rotary_pos_emb.
    """
    batch, heads, seq, dim = 2, 8, 32, d
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    cos, sin = _get_cos_sin(dim, seq)
    
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    
    # Verify k_out is different from k (rotation applied)
    assert not torch.allclose(k, k_out)

@pytest.mark.parametrize("d", [128])
def test_rope_architecture_alignment(d):
    """
    Verify RoPE with (Batch=2, Heads=8, Seq=32) SOTA layout.
    """
    batch, heads, seq, dim = 2, 8, 32, d
    # Input in SOTA layout: (Batch, Heads, Seq, Dim)
    x = torch.randn(batch, heads, seq, dim)
    cos, sin = _get_cos_sin(dim, seq)
    
    x_out, _ = apply_rotary_pos_emb(x, x, cos, sin)
    
    assert x_out.shape == (batch, heads, seq, dim)
    # Verify that token 31 is actually rotated
    assert not torch.allclose(x[:, :, 31, :], x_out[:, :, 31, :])
