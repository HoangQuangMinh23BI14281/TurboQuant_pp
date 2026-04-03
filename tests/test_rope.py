import torch
import pytest
from turboquant.ops.rope import RotaryPositionalEmbeddings, apply_rope

@pytest.mark.parametrize("d", [64, 128, 256])
def test_rope_identity_at_zero(d):
    """
    RoPE at position 0 should be an identity (no rotation).
    Note: Standard RoPE implementation with caching starts rotation at pos 0.
    In the notebook implementation, pos 0 has cos(0)=1, sin(0)=0.
    """
    rope = RotaryPositionalEmbeddings(d)
    x = torch.randn(1, 1, 1, d) # (seq, batch, heads, d)
    x_out = rope(x)
    
    # At pos 0, it should be identical
    torch.testing.assert_close(x, x_out, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_rope_relative_position(d):
    """
    Check if the dot product is sensitive only to relative positions.
    <f(q, m), f(k, m+n)> should be equal to <f(q, 0), f(k, n)>.
    Requires using the same vectors for q and k at different positions.
    """
    rope = RotaryPositionalEmbeddings(d)
    
    # Use fixed vectors q and k at different positions
    q_vec = torch.randn(1, 1, 1, d)
    k_vec = torch.randn(1, 1, 1, d)
    
    # Expand to a sequence so we can apply different pos encodings
    q = q_vec.expand(10, -1, -1, -1).clone()
    k = k_vec.expand(10, -1, -1, -1).clone()
    
    q_rope = rope(q)
    k_rope = rope(k)
    
    # Case 1: m=2, n=5 (distance 3)
    dot_2_5 = torch.sum(q_rope[2] * k_rope[5])
    
    # Case 2: m=0, n=3 (distance 3)
    dot_0_3 = torch.sum(q_rope[0] * k_rope[3])
    
    # Both should be equal as they have the same relative distance
    torch.testing.assert_close(dot_2_5, dot_0_3, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("d", [64, 128, 256])
def test_apply_rope_functional(d):
    """
    Test the functional interface apply_rope.
    """
    q = torch.randn(5, 2, 8, d)
    k = torch.randn(5, 2, 8, d)
    
    q_out, k_out = apply_rope(q, k)
    
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    
    # Verify k_out is different from k (rotation applied)
    assert not torch.allclose(k, k_out)

@pytest.mark.parametrize("d", [128])
def test_rope_architecture_alignment(d):
    """
    EXTREME TORTURE: Verify RoPE with (Batch=2, Heads=8, Seq=32).
    If RoPE assumes (Seq, Batch, Heads, D), it will incorrectly use Batch=2 as SeqLen.
    """
    batch, heads, seq, dim = 2, 8, 32, d
    rope = RotaryPositionalEmbeddings(dim)
    
    # Input in SOTA layout: (Batch, Heads, Seq, Dim)
    x = torch.randn(batch, heads, seq, dim)
    
    # This should NOT fail and should apply rotation across all 32 tokens
    try:
        x_out = rope(x)
    except Exception as e:
        pytest.fail(f"RoPE failed on SOTA (B,H,S,D) layout: {e}")
        
    assert x_out.shape == (batch, heads, seq, dim)
    
    # Verify that token 31 is actually rotated (not just using identity because cache was too small)
    assert not torch.allclose(x[:, :, 31, :], x_out[:, :, 31, :])
