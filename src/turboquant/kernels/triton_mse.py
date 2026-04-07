import torch
import triton
import triton.language as tl
from .triton_utils import _get_packing_params
from typing import Optional

@triton.jit
def _turboquant_mse_score_kernel(
    # Pointers
    Q_ptr, MSE_ptr, NORMS_ptr, SCALES_ptr, CENTROIDS_ptr, OUT_ptr,
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n, stride_n_sub, # SOTA: Thêm stride cho sub-blocks
    stride_s_bh, stride_s_n, stride_s_sub, # SOTA: Thêm stride cho sub-blocks
    stride_o_bh, stride_o_n,
    # Shapes
    BH, N, D, PACKED_D, 
    NQ,  # Number of queries per key-cache head
    N_SUBBLOCKS, # SOTA: Số lượng sub-blocks mỗi head (VD: d=128, block=64 -> 2)
    BLOCK_SIZE_DIM, # SOTA: Kích thước mỗi sub-block (VD: 64)
    MSE_SCALE,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    key_bh_idx = pid_bh // NQ
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # SOTA v8.7: Tính dot product riêng cho từng sub-block trước khi cộng tổng
    dot_total = tl.zeros([BLOCK_N], dtype=tl.float32)

    for sub_idx in range(N_SUBBLOCKS):
        dot_sub = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        # Load Norm và Scale của sub-block hiện tại
        norm_val = tl.load(NORMS_ptr + key_bh_idx * stride_n_bh + n_offs * stride_n_n + sub_idx * stride_n_sub, mask=n_mask, other=0.0).to(tl.float32)
        scale_val = tl.load(SCALES_ptr + key_bh_idx * stride_s_bh + n_offs * stride_s_n + sub_idx * stride_s_sub, mask=n_mask, other=1.0).to(tl.float32)

        # Giới hạn số byte cần đọc cho sub-block này
        start_dim = sub_idx * BLOCK_SIZE_DIM
        end_dim = tl.minimum(start_dim + BLOCK_SIZE_DIM, D)
        
        # Vì dữ liệu packed theo chiều D, ta duyệt qua các byte thuộc sub-block này
        start_byte = start_dim // VALS_PER_BYTE
        end_byte = (end_dim + VALS_PER_BYTE - 1) // VALS_PER_BYTE

        for byte_idx in range(start_byte, end_byte):
            packed = tl.load(
                MSE_ptr + key_bh_idx * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0
            ).to(tl.int32)

            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx >= start_dim and coord_idx < end_dim:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(Q_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                    dot_sub += q_val * centroid_val
        
        # Nhân dot của sub-block với norm và scale của chính nó, rồi cộng vào tổng
        dot_total += dot_sub * norm_val * scale_val

    scores = dot_total * MSE_SCALE
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, scores, mask=n_mask)


def turboquant_mse_score(
    query_rot: torch.Tensor,
    mse_packed: torch.Tensor,
    norms: torch.Tensor,
    scales: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    mse_scale: float = 1.0,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    if query_rot.dim() == 1:
        query_rot = query_rot.unsqueeze(0)
    elif query_rot.dim() == 3:
        query_rot = query_rot.reshape(-1, query_rot.shape[-1])
    if mse_packed.dim() == 2:
        mse_packed = mse_packed.unsqueeze(0)
    
    # Đảm bảo Norm/Scale là tensor 3 chiều: [Batch_Heads, Seq, N_Subblocks]
    if norms.dim() == 2:
        norms = norms.unsqueeze(-1)
    if scales.dim() == 2:
        scales = scales.unsqueeze(-1)

    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    BH_keys = mse_packed.shape[0]
    NQ = BH // BH_keys
    packed_d = mse_packed.shape[2]
    n_subblocks = norms.shape[-1]
    import math
    block_size_dim = math.ceil(D / n_subblocks)

    eff_bits, vals_per_byte = _get_packing_params(mse_bits)
    out = torch.zeros((BH, N), device=query_rot.device, dtype=torch.float32)
    BLOCK_N = block_n if block_n else 128
    num_warps = num_warps if num_warps else 4
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, scales, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1), norms.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2),
        out.stride(0), out.stride(1),
        BH, N, D, packed_d, NQ, n_subblocks, block_size_dim, mse_scale,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N, num_warps=num_warps
    )
    return out