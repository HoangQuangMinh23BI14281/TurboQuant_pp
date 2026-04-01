import torch
import triton
import triton.language as tl
import math
from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# Kernel 1: MSE score computation
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _turboquant_mse_score_kernel(
    # Pointers
    Q_ptr, MSE_ptr, NORMS_ptr, SCALES_ptr, CENTROIDS_ptr, OUT_ptr,
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_s_bh, stride_s_n,
    stride_o_bh, stride_o_n,
    # Shapes
    BH, N, D, PACKED_D, 
    NQ,  # Number of queries per key-cache head
    MSE_SCALE,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Decouple query index from key index
    # Each key_bh_idx head has NQ queries associated with it
    key_bh_idx = pid_bh // NQ

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Accumulate dot product
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D):
        packed = tl.load(
            MSE_ptr + key_bh_idx * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                idx = (packed >> (sub * BITS)) & BIT_MASK
                centroid_val = tl.load(CENTROIDS_ptr + idx)
                q_val = tl.load(Q_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                dot += q_val * centroid_val

    norms = tl.load(NORMS_ptr + key_bh_idx * stride_n_bh + n_offs * stride_n_n, mask=n_mask, other=0.0).to(tl.float32)
    scales = tl.load(SCALES_ptr + key_bh_idx * stride_s_bh + n_offs * stride_s_n, mask=n_mask, other=1.0).to(tl.float32)

    scores = dot * norms * scales * MSE_SCALE
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, scores, mask=n_mask)


# ──────────────────────────────────────────────────────────────────────
# Kernel 2: QJL score computation
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr,    # (BH, D) pre-sketched query (q @ S^T)
    SIGNS_ptr,       # (BH, N, packed_d_signs) packed sign bits
    RES_NORMS_ptr,   # (BH, N) residual norms
    NORMS_ptr,       # (BH, N) key norms
    OUT_ptr,         # (BH, N) output QJL scores (added to existing)
    # Strides
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,  # D // 8
    QJL_SCALE,  # qjl_factor
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Accumulate dot product: q_sketched[j] * sign[j]
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        # Load packed sign byte for this block: (BLOCK_N,)
        packed = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        # Extract 8 sign bits per byte
        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                # Convert {0,1} -> {-1, +1}
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                # Load query sketched coord
                q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d).to(tl.float32)
                # Dot
                dot += q_val * sign_val

    # Scale by residual norms, key norms, and QJL factor
    res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n, mask=n_mask, other=0.0).to(tl.float32)
    # res_norms ALREADY contains full-domain magnitude (norm * key_norm) per mandatory formula.
    qjl_scores = dot * res_norms * QJL_SCALE

    # Add to existing scores
    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, existing + qjl_scores, mask=n_mask)


# ──────────────────────────────────────────────────────────────────────
# Python Wrappers
# ──────────────────────────────────────────────────────────────────────

def _get_packing_params(bits: int):
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2
    else:
        return 8, 1

def turboquant_mse_score(
    query_rot: torch.Tensor,
    mse_packed: torch.Tensor,
    norms: torch.Tensor,
    scales: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    mse_scale: float = 1.0,  # rotation_scale / (D^n_passes)
) -> torch.Tensor:
    # Ensure 2D (BH, D)
    if query_rot.dim() == 1:
        query_rot = query_rot.unsqueeze(0)
    elif query_rot.dim() == 3:
        query_rot = query_rot.reshape(-1, query_rot.shape[-1])
    
    # Ensure 3D (BH, N, packed_d)
    if mse_packed.dim() == 2:
        mse_packed = mse_packed.unsqueeze(0)
    
    # Ensure 2D (BH, N)
    if norms.dim() == 1:
        norms = norms.unsqueeze(0)
    if scales.dim() == 1:
        scales = scales.unsqueeze(0)

    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    BH_keys = mse_packed.shape[0]
    NQ = BH // BH_keys
    
    packed_d = mse_packed.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros((BH, N), device=query_rot.device, dtype=torch.float32)
    BLOCK_N = 128
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, scales, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        BH, N, D, packed_d, NQ, mse_scale,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N
    )
    return out

def turboquant_qjl_score(
    q_sketched: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    norms: torch.Tensor,
    qjl_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Ensure 2D (BH, D)
    if q_sketched.dim() == 1:
        q_sketched = q_sketched.unsqueeze(0)
    elif q_sketched.dim() == 3:
        q_sketched = q_sketched.reshape(-1, q_sketched.shape[-1])

    # Ensure 3D (BH, N, packed_d)
    if qjl_signs.dim() == 2:
        qjl_signs = qjl_signs.unsqueeze(0)
    
    # Ensure 2D (BH, N)
    if residual_norms.dim() == 1:
        residual_norms = residual_norms.unsqueeze(0)
    if norms.dim() == 1:
        norms = norms.unsqueeze(0)

    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    BH_keys = qjl_signs.shape[0]
    NQ = BH // BH_keys
    
    packed_d_signs = qjl_signs.shape[2]

    if out is None:
        out = torch.zeros((BH, N), device=q_sketched.device, dtype=torch.float32)
    
    BLOCK_N = 128
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs, NQ=NQ,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N
    )
    return out

def turboquant_attention_score(
    query: torch.Tensor,               # (B, H, 1, D) or (BH, 1, D)
    quantized_key,                      # ProdQuantized
    Pi: torch.Tensor,                   # (D, D) rotation matrix
    S: torch.Tensor,                    # (D, D) QJL matrix
    centroids: torch.Tensor,           # (n_clusters,) codebook
    mse_bits: int,
    qjl_scale: float,
) -> torch.Tensor:
    """
    High-level: compute TurboQuant attention scores using Triton kernels.
    """
    if query.dim() == 4:
        B, H, Q, D = query.shape
        query_flat = query.reshape(B * H, Q, D)
    else:
        query_flat = query
        D = query.shape[-1]

    # Precompute rotated and sketched queries
    q_rot = torch.matmul(query_flat.squeeze(1).float(), Pi.T)      # (BH, D)
    q_sketch = torch.matmul(query_flat.squeeze(1).float(), S.T)    # (BH, D)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    # Compute combined score: MSE + QJL
    scores = turboquant_mse_score(q_rot, mse_packed, norms, centroids, mse_bits)
    scores = turboquant_qjl_score(q_sketch, qjl_signs, res_norms, qjl_scale, out=scores)

    return scores


# ──────────────────────────────────────────────────────────────────────
# Kernel 3: Fused Decode Attention (Online Softmax + Value Aggregation)
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _turboquant_fused_decode_kernel(
    # Query (already rotated for MSE, and sketched for QJL)
    Q_ROT_ptr,       # (BH, D) q @ Pi^T
    Q_SKETCH_ptr,    # (BH, D) q @ S^T
    # Quantized keys
    MSE_ptr,         # (BH, N, packed_d_mse) packed MSE indices
    SIGNS_ptr,       # (BH, N, packed_d_signs) packed QJL signs
    NORMS_ptr,       # (BH, N) key norms
    RES_NORMS_ptr,   # (BH, N) residual norms
    CENTROIDS_ptr,   # (n_clusters,) codebook
    # Values (group-quantized)
    V_DATA_ptr,      # (BH, N, D) uint8 quantized values
    V_SCALES_ptr,    # (BH, N, N_GROUPS) value scales
    V_ZEROS_ptr,     # (BH, N, N_GROUPS) value zeros
    # Output
    OUT_ptr,         # (BH, D) output
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
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Quant params
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,  # 1/sqrt(d)
    # Block
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([1], dtype=tl.float32)                   # running sum of exp
    acc = tl.zeros([D], dtype=tl.float32)                    # running weighted sum

    num_blocks = tl.cdiv(N, BLOCK_N)

    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── Compute TQ attention score for this block ──

        # Part 1: MSE score
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(Q_ROT_ptr + pid_bh * stride_qr_bh + coord_idx * stride_qr_d).to(tl.float32)
                    mse_scores += q_val * centroid_val

        key_norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # Part 2: QJL score
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

        res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        # Combined score
        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Value Fetch & Aggregate ──
        d_offs = tl.arange(0, D)
        v_quant = tl.load(
            V_DATA_ptr + pid_bh * stride_v_bh
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0
        ).to(tl.float32)
        
        # Group-based dequantization
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_bh * stride_vs_bh
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_bh * stride_vz_bh
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)
        
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    # Final normalization
    acc = acc / l_i
    d_offs = tl.arange(0, D)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


def turboquant_fused_decode(
    query: torch.Tensor,
    quantized_key,
    value_quantized,                    # NamedTuple with data, scales, zeros
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Fully fused decode attention using Triton.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        # Auto-flatten batch dims
        orig_shape = mse_packed.shape
        mse_packed = mse_packed.reshape(-1, *orig_shape[-2:])
        qjl_signs = qjl_signs.reshape(-1, *orig_shape[-2:])
        norms = norms.reshape(-1, norms.shape[-1])
        res_norms = res_norms.reshape(-1, res_norms.shape[-1])

    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros
    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, -1, D)
        v_scales = v_scales.reshape(BH, -1, v_scales.shape[-1])
        v_zeros = v_zeros.reshape(BH, -1, v_zeros.shape[-1])

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]
    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros((BH, D), device=query.device, dtype=torch.float32)
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
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out.to(query.dtype)
