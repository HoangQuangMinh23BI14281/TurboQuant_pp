import torch
import triton
import triton.language as tl
from .triton_utils import _get_packing_params

@triton.jit
def _turboquant_paged_fused_kernel(
    # Query (Rotated & Sketched)
    Q_ROT_ptr, Q_SKETCH_ptr,
    # Physical Block Pool (Global)
    K_INDICES_ptr, K_SIGNS_ptr, K_METADATA_ptr,
    V_INDICES_ptr, V_METADATA_ptr,
    CENTROIDS_ptr,
    # Paged Map
    BLOCK_TABLE_ptr, # (n_blocks_in_seq)
    # Output
    OUT_ptr,
    # Strides (Physical Pool)
    stride_k_index_b, stride_k_index_h, stride_k_index_s, stride_k_index_d,
    stride_k_signs_b, stride_k_signs_h, stride_k_signs_s, stride_k_signs_d,
    stride_k_meta_b, stride_k_meta_h, stride_k_meta_s, stride_k_meta_attr,
    stride_v_index_b, stride_v_index_h, stride_v_index_s, stride_v_index_d,
    stride_v_meta_b, stride_v_meta_h, stride_v_meta_s, stride_v_meta_attr,
    # Strides (Query/Output)
    stride_qr_bh, stride_qr_d,
    stride_qs_bh, stride_qs_d,
    stride_o_bh, stride_o_d,
    # Dims
    N, D: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    PACKED_D_V: tl.constexpr,
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
    head_id = pid_bh % N_HEADS

    K_BIT_MASK: tl.constexpr = (1 << K_BITS) - 1
    V_BIT_MASK: tl.constexpr = (1 << V_BITS) - 1

    # Softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    d_offs = tl.arange(0, D)

    num_logical_blocks = tl.cdiv(N, BLOCK_N)
    for b_idx in range(num_logical_blocks):
        n_start = b_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        
        # 1. Logical to Physical Mapping (Validated L2 Mirror)
        physical_id_idx = n_start // BLOCK_SIZE
        physical_block_id = tl.load(BLOCK_TABLE_ptr + physical_id_idx).to(tl.int32)
        slot_offs = n_offs % BLOCK_SIZE
        
        # Base pointers for this Physical Block & Head
        k_idx_base = K_INDICES_ptr + physical_block_id * stride_k_index_b + head_id * stride_k_index_h
        k_signs_base = K_SIGNS_ptr + physical_block_id * stride_k_signs_b + head_id * stride_k_signs_h
        k_meta_base = K_METADATA_ptr + physical_block_id * stride_k_meta_b + head_id * stride_k_meta_h
        v_idx_base = V_INDICES_ptr + physical_block_id * stride_v_index_b + head_id * stride_v_index_h
        v_meta_base = V_METADATA_ptr + physical_block_id * stride_v_meta_b + head_id * stride_v_meta_h

        # 2. Key Scores (MSE + QJL)
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(k_idx_base + slot_offs * stride_k_index_s + byte_idx * stride_k_index_d, mask=n_mask, other=0).to(tl.int32)
            for sub in range(K_VALS_PER_BYTE):
                coord_idx = byte_idx * K_VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * K_BITS)) & K_BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx).to(tl.float32)
                    q_val_ptr = Q_ROT_ptr + pid_bh * stride_qr_bh + coord_idx * stride_qr_d
                    q_val = tl.load(q_val_ptr).to(tl.float32)
                    mse_scores += q_val * centroid_val
        
        # Metadata
        token_meta_ptr = k_meta_base + slot_offs * stride_k_meta_s
        res_n = tl.load(token_meta_ptr + 0 * stride_k_meta_attr, mask=n_mask, other=0.0).to(tl.float32)
        total_n = tl.load(token_meta_ptr + 1 * stride_k_meta_attr, mask=n_mask, other=0.0).to(tl.float32)
        mse_scores *= total_n
        
        # QJL signs
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(k_signs_base + slot_offs * stride_k_signs_s + byte_idx * stride_k_signs_d, mask=n_mask, other=0).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_sketch_val_ptr = Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d
                    q_sketch_val = tl.load(q_sketch_val_ptr).to(tl.float32)
                    qjl_dot += q_sketch_val * sign_val
        
        scores = (mse_scores + qjl_dot * res_n * QJL_SCALE) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # Online Softmax update
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # 3. Value Aggregation
        token_v_meta_ptr = v_meta_base + slot_offs * stride_v_meta_s
        v_scale = tl.load(token_v_meta_ptr + 0 * stride_v_meta_attr, mask=n_mask, other=1.0).to(tl.float32)
        v_zero = tl.load(token_v_meta_ptr + 1 * stride_v_meta_attr, mask=n_mask, other=0.0).to(tl.float32)
        
        for byte_idx in range(PACKED_D_V):
            v_packed = tl.load(v_idx_base + slot_offs * stride_v_index_s + byte_idx * stride_v_index_d, mask=n_mask, other=0).to(tl.int32)
            for sub in range(V_VALS_PER_BYTE):
                coord_idx = byte_idx * V_VALS_PER_BYTE + sub
                if coord_idx < D:
                    v_idx = ((v_packed >> (sub * V_BITS)) & V_BIT_MASK).to(tl.float32)
                    # SOTA FIX: x = index * scale + zero
                    v_val = v_idx * v_scale + v_zero
                    acc_val = tl.sum(p * v_val, 0)
                    mask_d = (d_offs == coord_idx)
                    acc = tl.where(mask_d, acc + acc_val, acc)
        
        m_i = m_new

    # Final normalization
    acc = acc / l_i
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)

def turboquant_paged_fused_attention(
    q_rot, q_sketch,
    block_pool,
    block_table,
    context_len,
    centroids,
    n_heads,
    k_bits, v_bits,
    qjl_scale, sm_scale
):
    """
    Paged Attention Dispatcher for TurboQuant++ Hybrid Cache.
    """
    BH, D = q_rot.shape[0], q_rot.shape[-1]
    out = torch.zeros((BH, D), device=q_rot.device, dtype=torch.float32)
    
    k_eff_bits, k_vals_per_byte = _get_packing_params(k_bits)
    v_eff_bits, v_vals_per_byte = _get_packing_params(v_bits)
    
    grid = (BH,)
    BLOCK_N = 64
    
    k_idx_strides = block_pool.k_indices.stride()
    k_qjl_strides = block_pool.k_qjl.stride()
    k_meta_strides = block_pool.k_metadata.stride()
    v_idx_strides = block_pool.v_indices.stride()
    v_meta_strides = block_pool.v_metadata.stride()

    _turboquant_paged_fused_kernel[grid](
        q_rot, q_sketch,
        block_pool.k_indices, block_pool.k_qjl, block_pool.k_metadata,
        block_pool.v_indices, block_pool.v_metadata, 
        centroids, block_table, out,
        k_idx_strides[0], k_idx_strides[1], k_idx_strides[2], k_idx_strides[3],
        k_qjl_strides[0], k_qjl_strides[1], k_qjl_strides[2], k_qjl_strides[3],
        k_meta_strides[0], k_meta_strides[1], k_meta_strides[2], k_meta_strides[3],
        v_idx_strides[0], v_idx_strides[1], v_idx_strides[2], v_idx_strides[3],
        v_meta_strides[0], v_meta_strides[1], v_meta_strides[2], v_meta_strides[3],
        q_rot.stride(0), q_rot.stride(1),
        q_sketch.stride(0), q_sketch.stride(1),
        out.stride(0), out.stride(1),
        N=context_len, D=D, N_HEADS=n_heads, BLOCK_SIZE=block_pool.block_size,
        PACKED_D_MSE=block_pool.k_indices.shape[-1],
        PACKED_D_SIGNS=block_pool.k_qjl.shape[-1],
        PACKED_D_V=block_pool.v_indices.shape[-1],
        K_BITS=k_eff_bits, K_VALS_PER_BYTE=k_vals_per_byte,
        V_BITS=v_eff_bits, V_VALS_PER_BYTE=v_vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out.to(q_rot.dtype)
