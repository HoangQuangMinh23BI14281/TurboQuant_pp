import math
import torch
import triton
import triton.language as tl
from typing import Dict, Any, Optional

# Standard Centroid Fetching (Moved from fused_attention to break circular dependency)
def compute_centroids(bits: int, dist: str = 'gaussian'):
    from ..quant.lloyd_max import compute_lloyd_max_codebook
    return compute_lloyd_max_codebook(bits, dist=dist)['centroids']

def _get_packing_params(bits: int):
    if bits <= 4:
        return bits, 8 // bits
    else:
        return bits, 1

@triton.jit
def _turboquant_paged_fused_kernel(
    Q_ROT_ptr,
    NUM_TOKENS_ptr,          # SOTA: GPU-resident context length
    K_INDICES_ptr, K_SIGNS_ptr, K_METADATA_ptr,
    V_INDICES_ptr, V_METADATA_ptr,
    K_CENTROIDS_ptr, V_CENTROIDS_ptr,
    BLOCK_TABLE_ptr,
    K_SUMMARIES_ptr,        # Quest: Min/Max per block
    OUT_ptr,
    LAYER_IDX,
    MAX_BLOCKS, # SOTA: Boundary guard
    stride_k_index_l, stride_k_index_b, stride_k_index_h, stride_k_index_s, stride_k_index_d,
    stride_k_signs_l, stride_k_signs_b, stride_k_signs_h, stride_k_signs_s, stride_k_signs_d,
    stride_k_meta_l, stride_k_meta_b, stride_k_meta_h, stride_k_meta_s, stride_k_meta_attr,
    stride_v_index_l, stride_v_index_b, stride_v_index_h, stride_v_index_s, stride_v_index_d,
    stride_v_meta_l, stride_v_meta_b, stride_v_meta_h, stride_v_meta_s, stride_v_meta_attr,
    stride_sum_l, stride_sum_b, stride_sum_h, stride_sum_attr, stride_sum_d,
    stride_qr_bh, stride_qr_d,
    stride_o_bh, stride_o_d,
    D,
    K_BITS, K_VALS_PER_BYTE,
    V_BITS, V_VALS_PER_BYTE,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QJL_SCALE, SM_SCALE,
    QUEST_THRESHOLD, V_SPARSE_THRESHOLD,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    q_head_idx = pid_bh % N_HEADS
    kv_head_idx = q_head_idx // (N_HEADS // N_KV_HEADS)
    
    # SOTA: Load context length from GPU memory (CUDA Graph friendly)
    N = tl.load(NUM_TOKENS_ptr)
    
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    
    q_rot = tl.load(Q_ROT_ptr + pid_bh * stride_qr_bh + d_range, mask=d_mask, other=0.0).to(tl.float32)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # =====================================================================
    # SOTA HOISTING: Đưa các phép tính toán nguyên đắt đỏ ra ngoài vòng lặp
    # =====================================================================
    
    # 1. Tính toán trước vị trí byte và số bit cần dịch (shift)
    k_byte_idx = d_range // K_VALS_PER_BYTE
    k_shift = (d_range % K_VALS_PER_BYTE) * K_BITS
    k_mask_bits = (1 << K_BITS) - 1

    ks_byte_idx = d_range // 8
    ks_shift = d_range % 8

    v_byte_idx = d_range // V_VALS_PER_BYTE
    v_shift = (d_range % V_VALS_PER_BYTE) * V_BITS
    v_mask_bits = (1 << V_BITS) - 1

    # 2. Tính toán trước các địa chỉ con trỏ cơ sở (Base Pointers) cho Layer và Head hiện tại
    k_idx_head_base = K_INDICES_ptr + LAYER_IDX * stride_k_index_l + kv_head_idx * stride_k_index_h
    k_sgn_head_base = K_SIGNS_ptr + LAYER_IDX * stride_k_signs_l + kv_head_idx * stride_k_signs_h
    k_meta_head_base = K_METADATA_ptr + LAYER_IDX * stride_k_meta_l + kv_head_idx * stride_k_meta_h
    
    v_idx_head_base = V_INDICES_ptr + LAYER_IDX * stride_v_index_l + kv_head_idx * stride_v_index_h
    v_meta_head_base = V_METADATA_ptr + LAYER_IDX * stride_v_meta_l + kv_head_idx * stride_v_meta_h
    
    sum_head_base = K_SUMMARIES_ptr + LAYER_IDX * stride_sum_l + kv_head_idx * stride_sum_h

    # =====================================================================
    # SOTA: Parallel Block-Loop with Dynamic Boundary Guard (While Loop)
    # =====================================================================
    b_idx = 0
    while (b_idx * BLOCK_SIZE < N) and (b_idx < MAX_BLOCKS):
        start_n = b_idx * BLOCK_SIZE
        
        # SOTA: Recalculate n_mask for masked loads inside while loop
        n_mask = (start_n + tl.arange(0, BLOCK_N)) < N
        
        # Lấy ID của Block vật lý trong Paged Memory
        pb_id = tl.load(BLOCK_TABLE_ptr + b_idx)

        
        # 1. Quest: Query-Aware Sparsity Check
        # SOTA: Only process if within valid sequence length
        if start_n < N:
            sum_block_base = sum_head_base + pb_id * stride_sum_b
            k_min = tl.load(sum_block_base + 0 * stride_sum_attr + d_range, mask=d_mask, other=0.0)
            k_max = tl.load(sum_block_base + 1 * stride_sum_attr + d_range, mask=d_mask, other=0.0)
            
            quest_bound = tl.sum(tl.where(q_rot > 0, q_rot * k_max, q_rot * k_min)) * SM_SCALE
            
            if quest_bound >= QUEST_THRESHOLD:
                # SOTA Pillar 3: Tile-Aligned Block Optimization
                slot_indices = tl.arange(0, BLOCK_N)
                k_byte_base = k_idx_head_base + pb_id * stride_k_index_b + slot_indices[:, None] * stride_k_index_s
                k_signs_base = k_sgn_head_base + pb_id * stride_k_signs_b + slot_indices[:, None] * stride_k_signs_s
                
                # 1. Load MSE Indices
                k_idx_packed = tl.load(k_byte_base + k_byte_idx[None, :] * stride_k_index_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
                ki = ((k_idx_packed >> k_shift[None, :]).to(tl.int32) & k_mask_bits)
                
                # 2. Load Direct Signs
                k_sig_packed = tl.load(k_signs_base + ks_byte_idx[None, :] * stride_k_signs_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
                ksi = (k_sig_packed >> ks_shift[None, :]) & 1
                
                # 3. Key Metadata
                k_met_base = k_meta_head_base + pb_id * stride_k_meta_b + slot_indices[:, None] * stride_k_meta_s
                kn = tl.load(k_met_base + 0 * stride_k_meta_attr).to(tl.float32)
                ks = tl.load(k_met_base + 1 * stride_k_meta_attr).to(tl.float32)
                kr = tl.load(k_met_base + 2 * stride_k_meta_attr).to(tl.float32)
                
                k_mse = tl.load(K_CENTROIDS_ptr + ki).to(tl.float32) * kn * ks
                k_qjl = (ksi.to(tl.float32) * 2.0 - 1.0) * kr * QJL_SCALE
                
                scores = tl.sum(q_rot[None, :] * (k_mse + k_qjl), 1) * SM_SCALE
                scores = tl.where(n_mask, scores, float("-inf"))
                
                # Online Softmax
                m_new = tl.maximum(m_i, tl.max(scores, 0))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(scores - m_new)
                
                # 2. Sparse V-Fetch
                if tl.max(p) > V_SPARSE_THRESHOLD:
                    v_idx_base = v_idx_head_base + pb_id * stride_v_index_b + slot_indices[:, None] * stride_v_index_s
                    v_met_base = v_meta_head_base + pb_id * stride_v_meta_b + slot_indices[:, None] * stride_v_meta_s
                    
                    vn = tl.load(v_met_base + 0 * stride_v_meta_attr).to(tl.float32)
                    vs = tl.load(v_met_base + 1 * stride_v_meta_attr).to(tl.float32)
                    
                    v_idx_packed = tl.load(v_idx_base + v_byte_idx[None, :] * stride_v_index_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
                    vi = ((v_idx_packed >> v_shift[None, :]).to(tl.int32) & v_mask_bits)
                    
                    v_deq = tl.load(V_CENTROIDS_ptr + vi).to(tl.float32) * vn * vs
                    acc = acc * alpha + tl.sum(p[:, None] * v_deq, 0)
                else:
                    acc = acc * alpha
                
                l_i = l_i * alpha + tl.sum(p, 0)
                m_i = m_new
        
        # SOTA: Loop increment
        b_idx += 1

    # Store kết quả cuối
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_range, (acc / (l_i + 1e-10)).to(tl.float32), mask=d_mask)


def turboquant_paged_fused_attention(
    query: torch.Tensor,
    kv_cache: Any,
    k_bits: int,
    v_bits: int,
    qjl_scale: float,
    sm_scale: float,
    mask: Optional[torch.Tensor] = None,
    k_centroids: Optional[torch.Tensor] = None,
    v_centroids: Optional[torch.Tensor] = None,
    quest_threshold: float = 1e-4
) -> torch.Tensor:
    # SOTA: Derive n_heads from query shape (Dynamic Dispatch)
    if query.dim() == 4:
        # (batch, heads, seq, dim)
        n_heads = query.shape[1]
    elif query.dim() == 3:
        # (heads, seq, dim) or (batch, seq, dim)
        n_heads = query.shape[0]
    else:
        n_heads = kv_cache.n_heads
        
    head_dim = kv_cache.head_dim
    q_view = query.reshape(n_heads, head_dim)
    
    # SOTA: Transform query to Rotated Domain (WHT) only if using prod/quantizer
    if kv_cache.k_quantizer is not None:
         q_rot, _ = kv_cache.k_quantizer.transform_query(q_view)
    else:
         # FP16 Path: use original domain but PAD if necessary
         from torch.nn import functional as F
         q_rot = F.pad(q_view.float(), (0, kv_cache.pool.padded_head_dim - head_dim)) if kv_cache.pool.padded_head_dim > head_dim else q_view.float()

    D = q_rot.shape[-1]
    BH = n_heads 
    q_rot = q_rot.reshape(BH, D)
    
    # SOTA Pillar 1: Luôn trừ 1 bit cho QJL Sign!
    k_mse_bits = k_bits - 1 
    
    ptrs = kv_cache.get_paged_ptrs()
    pool = ptrs["pool"]
    block_table = ptrs["block_table"]
    context_len = ptrs["num_tokens"]
    # NOTE: Do NOT compare context_len (GPU tensor) to Python int here!
    # 'if context_len == 0:' would force a host-device sync, which is
    # ILLEGAL during CUDA Graph capture. The kernel handles N=0 via masking.

    # CUDA Graph Safety: centroids and _static_out MUST be pre-initialized
    # before graph capture. Any allocation here would be fatal.
    assert k_centroids is not None, "k_centroids must be pre-initialized before CUDA Graph capture"
    assert v_centroids is not None, "v_centroids must be pre-initialized before CUDA Graph capture"
    assert hasattr(kv_cache, '_static_out'), "_static_out must be pre-allocated before CUDA Graph capture"
    
    # Zero the static output buffer in-place (GPU op, graph-safe)
    out = kv_cache._static_out[0].view(BH, D)
    out.zero_()
    tokens_per_block = pool.tokens_per_block
    
    v_sparse_threshold = pool.config.quant.v_sparse_threshold
    BLOCK_D = 2**math.ceil(math.log2(D))
    
    # SOTA Performance Fix: Calculate MAX_BLOCKS
    # In Eager mode, we only iterate over used blocks.
    # In Graph mode, we MUST use the full table capacity for capture stability.
    is_capturing = torch.cuda.is_current_stream_capturing()
    max_blocks_limit = block_table.shape[0] if is_capturing else len(kv_cache.block_ids)
    # Ensure there is at least one block to avoid 0-iteration logic errors
    max_blocks_limit = max(1, max_blocks_limit)

    grid = (BH,)
    _turboquant_paged_fused_kernel[grid](
        Q_ROT_ptr=q_rot,
        NUM_TOKENS_ptr=context_len,
        K_INDICES_ptr=pool.k_indices,
        K_SIGNS_ptr=pool.k_qjl,
        K_METADATA_ptr=pool.k_metadata,
        V_INDICES_ptr=pool.v_indices,
        V_METADATA_ptr=pool.v_metadata, 
        K_CENTROIDS_ptr=k_centroids, 
        V_CENTROIDS_ptr=v_centroids,
        BLOCK_TABLE_ptr=block_table, 
        K_SUMMARIES_ptr=pool.k_summaries, 
        OUT_ptr=out,
        LAYER_IDX=kv_cache.layer_idx,
        MAX_BLOCKS=max_blocks_limit,
        stride_k_index_l=pool.k_indices.stride(0),
        stride_k_index_b=pool.k_indices.stride(1),
        stride_k_index_h=pool.k_indices.stride(2),
        stride_k_index_s=pool.k_indices.stride(3),
        stride_k_index_d=pool.k_indices.stride(4),
        stride_k_signs_l=pool.k_qjl.stride(0),
        stride_k_signs_b=pool.k_qjl.stride(1),
        stride_k_signs_h=pool.k_qjl.stride(2),
        stride_k_signs_s=pool.k_qjl.stride(3),
        stride_k_signs_d=pool.k_qjl.stride(4),
        stride_k_meta_l=pool.k_metadata.stride(0),
        stride_k_meta_b=pool.k_metadata.stride(1),
        stride_k_meta_h=pool.k_metadata.stride(2),
        stride_k_meta_s=pool.k_metadata.stride(3),
        stride_k_meta_attr=pool.k_metadata.stride(4),
        stride_v_index_l=pool.v_indices.stride(0),
        stride_v_index_b=pool.v_indices.stride(1),
        stride_v_index_h=pool.v_indices.stride(2),
        stride_v_index_s=pool.v_indices.stride(3),
        stride_v_index_d=pool.v_indices.stride(4),
        stride_v_meta_l=pool.v_metadata.stride(0),
        stride_v_meta_b=pool.v_metadata.stride(1),
        stride_v_meta_h=pool.v_metadata.stride(2),
        stride_v_meta_s=pool.v_metadata.stride(3),
        stride_v_meta_attr=pool.v_metadata.stride(4),
        stride_sum_l=pool.k_summaries.stride(0),
        stride_sum_b=pool.k_summaries.stride(1),
        stride_sum_h=pool.k_summaries.stride(2),
        stride_sum_attr=pool.k_summaries.stride(3),
        stride_sum_d=pool.k_summaries.stride(4),
        stride_qr_bh=q_rot.stride(0), stride_qr_d=q_rot.stride(1),
        stride_o_bh=out.stride(0), stride_o_d=out.stride(1),
        D=D,
        K_BITS=k_mse_bits, K_VALS_PER_BYTE=8 // k_mse_bits,
        V_BITS=v_bits, V_VALS_PER_BYTE=8 // v_bits,
        N_HEADS=n_heads,
        N_KV_HEADS=kv_cache.n_kv_heads,
        BLOCK_SIZE=pool.tokens_per_block,
        QJL_SCALE=qjl_scale,
        SM_SCALE=sm_scale, 
        QUEST_THRESHOLD=quest_threshold,
        V_SPARSE_THRESHOLD=pool.config.quant.v_sparse_threshold,
        BLOCK_N=pool.config.hw.triton_block_n,
        BLOCK_D=BLOCK_D,
        num_warps=pool.config.hw.triton_num_warps
    )
    
    # SOTA v8.5 Final Step: Inverse SRHT to restore original V-domain
    if kv_cache.v_quantizer is not None:
        out = kv_cache.v_quantizer.mse_quantizer.rotation.inverse(out.to(query.dtype))
    
    return out.reshape(query.shape).to(query.dtype)

def paged_turboquant_attention(
    query: torch.Tensor, 
    kv_cache: Any, 
    k_bits: int, 
    v_bits: int, 
    qjl_scale: float, 
    sm_scale: float, 
    mask: Optional[torch.Tensor] = None,
    quest_threshold: float = 1e-4
) -> torch.Tensor:
    # High-level entry point for direct testing
    return turboquant_paged_fused_attention(
        query, kv_cache, k_bits, v_bits, qjl_scale, sm_scale, 
        mask=mask, quest_threshold=quest_threshold
    )
