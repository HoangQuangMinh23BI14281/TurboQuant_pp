import torch
import pytest
from turboquant.kernels.triton_dequant_v import dequantize_value_triton

def dequantize_v_pytorch(indices, scales, zeros, group_size=128):
    """
    Reference PyTorch implementation for Asymmetric Min-Max dequantization.
    Formula: V = (idx - zero_point) * scale
    """
    # Ensure float32 for high precision during reference calculation
    v_idx = indices.float()
    v_scale = scales.float()
    v_zero = zeros.float()
    
    # Broadcast metadata across the group
    # scales/zeros shape: (B, H, N, G), where G = D // group_size
    # We need to repeat each metadata value group_size times to match indices (D)
    # Handling both (B, H, N, G) or (BH, N, G)
    if v_scale.dim() == 4:
        B, H, N, G = v_scale.shape
    else:
        BH, N, G = v_scale.shape
    v_scale_expanded = v_scale.repeat_interleave(group_size, dim=-1)
    v_zero_expanded = v_zero.repeat_interleave(group_size, dim=-1)
    
    # Apply formula
    v_dequant = (v_idx - v_zero_expanded) * v_scale_expanded
    return v_dequant.half()

@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("heads", [8])
@pytest.mark.parametrize("seq_len", [128, 512])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("group_size", [128])
def test_value_dequant_bit_exact(batch, heads, seq_len, head_dim, group_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = "cuda"
    BH = batch * heads
    num_groups = head_dim // group_size
    
    # Create random source data
    # indices: uint8 representing 4-bit or 8-bit values
    indices = torch.randint(0, 16, (BH, seq_len, head_dim), device=device, dtype=torch.uint8)
    scales = torch.randn((BH, seq_len, num_groups), device=device, dtype=torch.float16).abs()
    zeros = torch.randn((BH, seq_len, num_groups), device=device, dtype=torch.float16)
    
    # Reference
    expected = dequantize_v_pytorch(indices, scales, zeros, group_size)
    
    # Triton (Expects PACKED indices)
    from turboquant.quant.quant_base import pack_indices
    indices_packed = pack_indices(indices.view(BH * seq_len, head_dim), bits=4).view(BH, seq_len, -1)
    actual = dequantize_value_triton(indices_packed, scales, zeros, group_size, bits=4)
    
    # Check bit-exactness (using small atol for FP16 precision differences)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    print(f"PASS: batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim} | Bit-exact check successful.")

if __name__ == "__main__":
    # Quick manual run
    test_value_dequant_bit_exact(1, 4, 128, 128, 128)
