import torch
import triton
import triton.language as tl
from .triton_utils import _get_packing_params

@triton.jit
def _turboquant_fused_decode_kernel(
    # Query (already rotated for MSE, and sketched for QJL)
    Q_ROT_ptr,       # (BH, D)
    Q_SKETCH_ptr,    # (BH, D)
    # Quantized keys
    MSE_ptr,         # (BH, N, packed_d_mse) 
    SIGNS_ptr,       # (BH, N, packed_d_signs)
    NORMS_ptr,       # (BH, N)
    RES_NORMS_ptr,   # (BH, N)
    CENTROIDS_ptr,   # (n_clusters,)
    # Values (group-quantized & packed)
    V_DATA_ptr,      # (BH, N, packed_d_v)
    V_SCALES_ptr,    # (BH, N, N_GROUPS)
    V_ZEROS_ptr,     # (BH, N, N_GROUPS)
    # Output
    OUT_ptr,         # (BH, D)
    # Strides
    stride_qr_bh, stride_qr_d,
    stride_qs_bh, stride_qs_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_o_bh, stride_o_d,
    # Dims
    N, D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    PACKED_D_V: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Quant params
    K_BITS: tl.constexpr,
    K_VALS_PER_BYTE: tl.constexpr,
    V_BITS: tl.constexpr,
    V_VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    K_BIT_MASK: tl.constexpr = (1 << K_BITS) - 1
    V_BIT_MASK: tl.constexpr = (1 << V_BITS) - 1

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    # Offsets
    d_offs = tl.arange(0, D)
    g_offs = d_offs // GROUP_SIZE

    num_blocks = tl.cdiv(N, BLOCK_N)
    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── Part 1: MSE Score (Key) ──
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for sub in range(K_VALS_PER_BYTE):
                coord_idx = byte_idx * K_VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * K_BITS)) & K_BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(Q_ROT_ptr + pid_bh * stride_qr_bh + coord_idx * stride_qr_d).to(tl.float32)
                    mse_scores += q_val * centroid_val
        
        key_norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n, mask=n_mask, other=0.0).to(tl.float32)
        mse_scores *= key_norms

        # ── Part 2: QJL Score (Key) ──
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d).to(tl.float32)
                    qjl_dot += q_val * sign_val

        res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n, mask=n_mask, other=0.0).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Part 3: Value Aggregation (Hybrid Vectorized Unpacking) ──
        for byte_idx in range(PACKED_D_V):
            # Load block of packed Value indices (1D: BLOCK_N)
            v_packed = tl.load(V_DATA_ptr + pid_bh * stride_v_bh + n_offs * stride_v_n + byte_idx * stride_v_d,
                              mask=n_mask, other=0).to(tl.int32)
            
            for sub in range(V_VALS_PER_BYTE):
                coord_idx = byte_idx * V_VALS_PER_BYTE + sub
                if coord_idx < D:
                    # Unpack Value index
                    v_idx = ((v_packed >> (sub * V_BITS)) & V_BIT_MASK).to(tl.float32)
                    
                    # Robust direct load for metadata (1D)
                    g_idx = coord_idx // GROUP_SIZE
                    v_s = tl.load(V_SCALES_ptr + pid_bh * stride_vs_bh + n_offs * stride_vs_n + g_idx * stride_vs_g,
                                 mask=n_mask, other=1.0).to(tl.float32)
                    v_z = tl.load(V_ZEROS_ptr + pid_bh * stride_vz_bh + n_offs * stride_vz_n + g_idx * stride_vz_g,
                                 mask=n_mask, other=0.0).to(tl.float32)
                    
                    # Dequantize: V = (idx - zero) * scale
                    # v_idx, v_s, v_z are all 1D (BLOCK_N)
                    v_val = (v_idx - v_z) * v_s 
                    
                    # Weighted contribution for this dimension
                    acc_val = tl.sum(p * v_val, 0)
                    
                    # Update accumulator (D) - use mask for dimension coord_idx
                    mask_d = (d_offs == coord_idx)
                    acc = tl.where(mask_d, acc + acc_val, acc)

        m_i = m_new

    # Final normalization
    acc = acc / l_i
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)

def turboquant_fused_decode(
    q_rot: torch.Tensor,
    q_sketch: torch.Tensor,
    quantized_key,
    value_quantized,
    centroids: torch.Tensor,
    k_bits: int,
    v_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Hybrid Precision Fused Decode Dispatcher.
    """
    if q_rot.dim() == 3: q_rot = q_rot.squeeze(1)
    if q_sketch.dim() == 3: q_sketch = q_sketch.squeeze(1)
    BH, D = q_rot.shape

    # Key Metadata
    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms
    
    # Value Metadata
    v_data = value_quantized.indices
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zero_points

    # Flatten Batch-Head if needed
    if mse_packed.dim() > 3:
        mse_packed = mse_packed.flatten(0, -3)
        qjl_signs = qjl_signs.flatten(0, -3)
        norms = norms.flatten(0, -2)
        res_norms = res_norms.flatten(0, -2)
        v_data = v_data.flatten(0, -3)
        v_scales = v_scales.flatten(0, -3)
        v_zeros = v_zeros.flatten(0, -3)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]
    packed_d_v = v_data.shape[2]
    N_GROUPS = D // group_size

    k_eff_bits, k_vals_per_byte = _get_packing_params(k_bits)
    v_eff_bits, v_vals_per_byte = _get_packing_params(v_bits)

    out = torch.zeros((BH, D), device=q_rot.device, dtype=torch.float32)
    BLOCK_N = 64
    grid = (BH,)

    _turboquant_fused_decode_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros, out,
        q_rot.stride(0), q_rot.stride(1),
        q_sketch.stride(0), q_sketch.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs, PACKED_D_V=packed_d_v,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        K_BITS=k_eff_bits, K_VALS_PER_BYTE=k_vals_per_byte,
        V_BITS=v_eff_bits, V_VALS_PER_BYTE=v_vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out.to(q_rot.dtype)
