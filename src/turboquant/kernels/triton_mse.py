import torch
import triton
import triton.language as tl
from .triton_utils import _get_packing_params

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

def turboquant_mse_score(
    query_rot: torch.Tensor,
    mse_packed: torch.Tensor,
    norms: torch.Tensor,
    scales: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    mse_scale: float = 1.0,
) -> torch.Tensor:
    # Ensure shapes/dims are compatible (standardization)
    if query_rot.dim() == 1:
        query_rot = query_rot.unsqueeze(0)
    elif query_rot.dim() == 3:
        query_rot = query_rot.reshape(-1, query_rot.shape[-1])
    
    if mse_packed.dim() == 2:
        mse_packed = mse_packed.unsqueeze(0)
    
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
