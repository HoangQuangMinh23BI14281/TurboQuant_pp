import torch
import triton
import triton.language as tl
from typing import Optional

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
                dot += q_val * sign_val

    res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n, mask=n_mask, other=0.0).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    # Add to existing scores
    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, existing + qjl_scores, mask=n_mask)

def turboquant_qjl_score(
    q_sketched: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    norms: torch.Tensor,
    qjl_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q_sketched.dim() == 1:
        q_sketched = q_sketched.unsqueeze(0)
    elif q_sketched.dim() == 3:
        q_sketched = q_sketched.reshape(-1, q_sketched.shape[-1])

    if qjl_signs.dim() == 2:
        qjl_signs = qjl_signs.unsqueeze(0)
    
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
