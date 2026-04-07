import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr,    # (BH_q, D) pre-sketched query
    SIGNS_ptr,       # (BH_k, N, packed_d_signs) packed sign bits
    RES_NORMS_ptr,   # (BH_k, N, N_SUBBLOCKS) residual norms
    OUT_ptr,         # (BH_q, N) output QJL scores
    # Strides
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n, stride_rn_sub,
    stride_o_bh, stride_o_n,
    # Dims
    N, D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    NQ: tl.constexpr, 
    N_SUBBLOCKS,
    BLOCK_SIZE_DIM,
    QJL_SCALE,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    key_bh_idx = pid_bh // NQ
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    dot_total = tl.zeros([BLOCK_N], dtype=tl.float32)

    for sub_idx in range(N_SUBBLOCKS):
        dot_sub = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        # Load residual norm của sub-block
        res_norm_val = tl.load(RES_NORMS_ptr + key_bh_idx * stride_rn_bh + n_offs * stride_rn_n + sub_idx * stride_rn_sub, mask=n_mask, other=0.0).to(tl.float32)

        start_dim = sub_idx * BLOCK_SIZE_DIM
        end_dim = tl.minimum(start_dim + BLOCK_SIZE_DIM, D)
        
        start_byte = start_dim // 8
        end_byte = (end_dim + 7) // 8

        for byte_idx in range(start_byte, end_byte):
            packed = tl.load(
                SIGNS_ptr + key_bh_idx * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0
            ).to(tl.int32)

            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx >= start_dim and coord_idx < end_dim:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d).to(tl.float32)
                    dot_sub += q_val * sign_val
                    
        # Nhân dot của sub-block với residual_norm tương ứng rồi cộng dồn
        dot_total += dot_sub * res_norm_val

    qjl_scores = dot_total * QJL_SCALE
    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, existing + qjl_scores, mask=n_mask)

def turboquant_qjl_score(
    q_sketched: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    norms: torch.Tensor, 
    qjl_scale: float,
    out: Optional[torch.Tensor] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    if q_sketched.dim() == 1:
        q_sketched = q_sketched.unsqueeze(0)
    elif q_sketched.dim() == 3:
        q_sketched = q_sketched.reshape(-1, q_sketched.shape[-1])

    if qjl_signs.dim() == 2:
        qjl_signs = qjl_signs.unsqueeze(0)
    
    if residual_norms.dim() == 2:
        residual_norms = residual_norms.unsqueeze(-1)

    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    BH_keys = qjl_signs.shape[0]
    NQ = BH // BH_keys
    packed_d_signs = qjl_signs.shape[2]
    
    n_subblocks = residual_norms.shape[-1]
    import math
    block_size_dim = math.ceil(D / n_subblocks)

    if out is None:
        out = torch.zeros((BH, N), device=q_sketched.device, dtype=torch.float32)
    
    BLOCK_N = block_n if block_n else 128
    num_warps = num_warps if num_warps else 4
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1), residual_norms.stride(2),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs, NQ=NQ, 
        N_SUBBLOCKS=n_subblocks, BLOCK_SIZE_DIM=block_size_dim,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N, num_warps=num_warps
    )
    return out