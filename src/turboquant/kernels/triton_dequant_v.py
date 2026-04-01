import torch
import triton
import triton.language as tl

@triton.jit
def _dequantize_v_kernel(
    V_IDX_ptr,        # Pointer to INT4/UINT8 indices (B, H, N, D)
    V_SCALES_ptr,     # Pointer to scales (B, H, N, N_GROUPS)
    V_ZEROS_ptr,      # Pointer to zero_points (B, H, N, N_GROUPS)
    OUT_ptr,          # Pointer to output FP16/BF16 (B, H, N, D)
    # Strides
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_o_bh, stride_o_n, stride_o_d,
    # Shapes
    N, D,
    GROUP_SIZE: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Standalone Triton Kernel for Asymmetric Value Dequantization.
    Formula: V = (idx - zero_point) * scale
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    # Offset calculations
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    n_mask = n_offs < N
    d_mask = d_offs < D
    
    # Load indices
    v_idx = tl.load(
        V_IDX_ptr + pid_bh * stride_v_bh + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
        mask=n_mask[:, None] & d_mask[None, :], other=0
    ).to(tl.float32)

    # Calculate group mapping
    g_offs = d_offs // GROUP_SIZE
    
    # Load metadata (scales and zeros)
    v_scale = tl.load(
        V_SCALES_ptr + pid_bh * stride_vs_bh + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
        mask=n_mask[:, None] & (g_offs[None, :] < (D // GROUP_SIZE)), other=1.0
    ).to(tl.float32)
    
    v_zero = tl.load(
        V_ZEROS_ptr + pid_bh * stride_vz_bh + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
        mask=n_mask[:, None] & (g_offs[None, :] < (D // GROUP_SIZE)), other=0.0
    ).to(tl.float32)

    # Dequantization: (idx - zero) * scale
    v_dequant = (v_idx - v_zero) * v_scale

    # Store result
    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + n_offs[:, None] * stride_o_n + d_offs[None, :] * stride_o_d,
        v_dequant,
        mask=n_mask[:, None] & d_mask[None, :]
    )

def dequantize_value_triton(indices, scales, zeros, group_size=128):
    """
    Python wrapper for the standalone Value dequantization kernel.
    """
    assert indices.is_cuda and scales.is_cuda and zeros.is_cuda
    
    # Dimensions: Indices (B*H, N, D), Scales (B*H, N, G), Zeros (B*H, N, G)
    if indices.dim() == 4:
        B, H, N, D = indices.shape
        indices_flat = indices.view(B * H, N, D)
        scales_flat = scales.view(B * H, N, -1)
        zeros_flat = zeros.view(B * H, N, -1)
    else:
        BH, N, D = indices.shape
        indices_flat, scales_flat, zeros_flat = indices, scales, zeros
        BH, N, D = indices_flat.shape

    out = torch.empty_like(indices_flat, dtype=torch.float16)
    
    BLOCK_N = 16
    BLOCK_D = 64
    
    grid = (indices_flat.shape[0], triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
    
    _dequantize_v_kernel[grid](
        indices_flat, scales_flat, zeros_flat, out,
        indices_flat.stride(0), indices_flat.stride(1), indices_flat.stride(2),
        scales_flat.stride(0), scales_flat.stride(1), scales_flat.stride(2),
        zeros_flat.stride(0), zeros_flat.stride(1), zeros_flat.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N, D,
        GROUP_SIZE=group_size,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4
    )
    
    return out.view(indices.shape) if indices.dim() == 4 else out
