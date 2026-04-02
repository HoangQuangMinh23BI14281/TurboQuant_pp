import torch
import triton
import triton.language as tl
from typing import Dict, Any

# Standard Centroid Fetching (Moved from fused_attention to break circular dependency)
def compute_centroids(bits: int, dist: str = 'gaussian'):
    from ..quant.lloyd_max import compute_lloyd_max_codebook
    return compute_lloyd_max_codebook(bits, dist=dist)['centroids']

def _get_packing_params(bits: int):
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return bits, 8 // bits
    else:
        return 8, 1

@triton.jit
def _turboquant_paged_fused_kernel(
    Q_ROT_ptr, Q_SKETCH_ptr,
    K_INDICES_ptr, K_SIGNS_ptr, K_METADATA_ptr,
    V_INDICES_ptr, V_METADATA_ptr,
    K_CENTROIDS_ptr, V_CENTROIDS_ptr,
    BLOCK_TABLE_ptr,
    K_SUMMARIES_ptr,        # Quest: Min/Max per block
    BLOCK_IMPORTANCE_ptr,   # H2O: Cumulative scores
    OUT_ptr,
    stride_k_index_b, stride_k_index_h, stride_k_index_s, stride_k_index_d,
    stride_k_signs_b, stride_k_signs_h, stride_k_signs_s, stride_k_signs_d,
    stride_k_meta_b, stride_k_meta_h, stride_k_meta_s, stride_k_meta_attr,
    stride_v_index_b, stride_v_index_h, stride_v_index_s, stride_v_index_d,
    stride_v_meta_b, stride_v_meta_h, stride_v_meta_s, stride_v_meta_g, stride_v_meta_attr,
    stride_sum_b, stride_sum_h, stride_sum_attr, stride_sum_d,
    stride_imp_b, stride_imp_h,
    stride_qr_bh, stride_qr_d,
    stride_qs_bh, stride_qs_d,
    stride_o_bh, stride_o_d,
    N, D,
    PACKED_D_MSE, PACKED_D_SIGNS, PACKED_D_V,
    K_BITS, K_VALS_PER_BYTE,
    V_BITS, V_VALS_PER_BYTE,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QJL_SCALE, SM_SCALE,
    QUEST_THRESHOLD,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_h = pid_bh % N_HEADS
    pid_kv_h = pid_h // (N_HEADS // N_KV_HEADS)
    
    d_range = tl.arange(0, 128)
    d_mask = d_range < D
    
    q_rot = tl.load(Q_ROT_ptr + pid_bh * stride_qr_bh + d_range, mask=d_mask, other=0.0).to(tl.float32)
    q_sk = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + d_range, mask=d_mask, other=0.0).to(tl.float32)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([128], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n_range = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_range < N
        
        # 1. Quest: Query-Aware Sparsity Check (Block Level Skip)
        # SOTA: Evaluate if this block is even relevant before loading heavy KV data
        p_idx = start_n // BLOCK_SIZE
        pb_id = tl.load(BLOCK_TABLE_ptr + p_idx, mask=(start_n < N))
        
        # SOTA: GQA-Aware Fetching
        k_min = tl.load(K_SUMMARIES_ptr + pb_id * stride_sum_b + pid_kv_h * stride_sum_h + 0 * stride_sum_attr + d_range, mask=d_mask, other=0.0)
        k_max = tl.load(K_SUMMARIES_ptr + pb_id * stride_sum_b + pid_kv_h * stride_sum_h + 1 * stride_sum_attr + d_range, mask=d_mask, other=0.0)
        
        # Upper bound: sum(max(q_i * min_i, q_i * max_i))
        q_min = q_rot * k_min
        q_max = q_rot * k_max
        quest_bound = tl.sum(tl.maximum(q_min, q_max)) * SM_SCALE
        
        if quest_bound >= QUEST_THRESHOLD:
            block_indices = (n_range / BLOCK_SIZE).to(tl.int32)
            slot_indices = n_range - (block_indices * BLOCK_SIZE)
            physical_block_ids = tl.load(BLOCK_TABLE_ptr + block_indices, mask=n_mask, other=0)
            
            k_byte_base = K_INDICES_ptr + physical_block_ids[:, None] * stride_k_index_b + pid_kv_h * stride_k_index_h
            k_sig_base = K_SIGNS_ptr + physical_block_ids[:, None] * stride_k_signs_b + pid_kv_h * stride_k_signs_h
            k_met_base = K_METADATA_ptr + physical_block_ids[:, None] * stride_k_meta_b + pid_kv_h * stride_k_meta_h
            
            # Load packed K indices and signs
            k_byte_idx = d_range // K_VALS_PER_BYTE
            k_sub_idx = d_range % K_VALS_PER_BYTE
            k_idx_packed = tl.load(k_byte_base + slot_indices[:, None] * stride_k_index_s + k_byte_idx[None, :] * stride_k_index_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
            ki = (k_idx_packed >> (k_sub_idx * K_BITS)).to(tl.int32) & ((1 << K_BITS) - 1)

            ks_byte_idx = d_range // 8
            ks_sub_idx = d_range % 8
            k_sig_packed = tl.load(k_sig_base + slot_indices[:, None] * stride_k_signs_s + ks_byte_idx[None, :] * stride_k_signs_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
            ksi = (k_sig_packed >> ks_sub_idx) & 1
            
            kn = tl.load(k_met_base + slot_indices[:, None] * stride_k_meta_s + 0 * stride_k_meta_attr, mask=n_mask[:, None], other=1.0)
            ks = tl.load(k_met_base + slot_indices[:, None] * stride_k_meta_s + 1 * stride_k_meta_attr, mask=n_mask[:, None], other=1.0)
            kr = tl.load(k_met_base + slot_indices[:, None] * stride_k_meta_s + 2 * stride_k_meta_attr, mask=n_mask[:, None], other=0.0)
            
            k_mse = tl.load(K_CENTROIDS_ptr + ki).to(tl.float32) * kn * ks
            k_qjl = tl.where(ksi == 1, 1.0, -1.0) * kr * QJL_SCALE
            
            scores = tl.sum(q_rot[None, :] * k_mse + q_sk[None, :] * k_qjl, 1) * SM_SCALE
            scores = tl.where(n_mask, scores, float("-inf"))
            
            m_new = tl.maximum(m_i, tl.max(scores, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            
            # 2. H2O: Importance Accumulation
            block_importance = tl.sum(p)
            tl.atomic_add(BLOCK_IMPORTANCE_ptr + pb_id * stride_imp_b + pid_kv_h * stride_imp_h, block_importance)

            v_idx_base = V_INDICES_ptr + physical_block_ids[:, None] * stride_v_index_b + pid_kv_h * stride_v_index_h
            v_met_base = V_METADATA_ptr + physical_block_ids[:, None] * stride_v_meta_b + pid_kv_h * stride_v_meta_h
            
            v_byte_idx = d_range // V_VALS_PER_BYTE
            v_sub_idx = d_range % V_VALS_PER_BYTE
            v_idx_packed = tl.load(v_idx_base + slot_indices[:, None] * stride_v_index_s + v_byte_idx[None, :] * stride_v_index_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
            vi = (v_idx_packed >> (v_sub_idx * V_BITS)).to(tl.int32) & ((1 << V_BITS) - 1)
            
            v_mse = tl.load(V_CENTROIDS_ptr + vi).to(tl.float32)
            v_scale = tl.load(v_met_base + slot_indices[:, None] * stride_v_meta_s + 0 * stride_v_meta_attr, mask=n_mask[:, None], other=1.0)
            v_zeros = tl.load(v_met_base + slot_indices[:, None] * stride_v_meta_s + 1 * stride_v_meta_attr, mask=n_mask[:, None], other=0.0)
            
            v_deq = v_mse * v_scale + v_zeros
            acc = acc * alpha + tl.sum(p[:, None] * v_deq, 0)
            l_i = l_i * alpha + tl.sum(p, 0)
            m_i = m_new

    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_range, (acc / l_i).to(tl.float32), mask=d_mask)

def turboquant_paged_fused_attention(
    query: torch.Tensor,
    kv_cache: Any,
    k_bits: int,
    v_bits: int,
    qjl_scale: float,
    sm_scale: float,
    quest_threshold: float = 1e-4
) -> torch.Tensor:
    # 1. Transform Query (Sketched Attention)
    # SOTA FIX: Ensure query is reshaped to head-wise view [n_heads, head_dim] for transformation
    n_heads = kv_cache.n_heads
    head_dim = kv_cache.head_dim
    q_view = query.reshape(n_heads, head_dim)
    
    q_rot, q_sketch = kv_cache.k_quantizer.transform_query(q_view)
    D = q_rot.shape[-1]
    BH = n_heads # For batch=1
    q_rot = q_rot.reshape(BH, D)
    q_sketch = q_sketch.reshape(BH, D)
    
    # 2. Get Paged Pointers
    ptrs = kv_cache.get_paged_ptrs()
    pool = ptrs["pool"]
    block_table = ptrs["block_table"]
    context_len = ptrs["num_tokens"]
    if context_len == 0:
        return torch.zeros_like(query)

    # 3. Fetch Centroids
    k_centroids = compute_centroids(k_bits, dist='gaussian').to(query.device, query.dtype)
    v_centroids = compute_centroids(v_bits, dist='laplace').to(query.device, query.dtype)
    
    # 4. Dispatch to Kernel
    out = torch.zeros((BH, D), device=query.device, dtype=torch.float32)
    tokens_per_block = pool.tokens_per_block
    k_vpb = 8 // k_bits
    v_vpb = 8 // v_bits
    
    kis, kqs, kms = pool.k_indices.stride(), pool.k_qjl.stride(), pool.k_metadata.stride()
    vis, vms = pool.v_indices.stride(), pool.v_metadata.stride()
    ss, ims = pool.k_summaries.stride(), pool.block_importance.stride()
    
    grid = (BH,)
    _turboquant_paged_fused_kernel[grid](
        q_rot, q_sketch,
        pool.k_indices, pool.k_qjl, pool.k_metadata,
        pool.v_indices, pool.v_metadata, 
        k_centroids, v_centroids, 
        block_table, 
        pool.k_summaries, 
        pool.block_importance,
        out,
        kis[0], kis[1], kis[2], kis[3],
        kqs[0], kqs[1], kqs[2], kqs[3],
        kms[0], kms[1], kms[2], kms[3],
        vis[0], vis[1], vis[2], vis[3],
        vms[0], vms[1], vms[2], vms[3], vms[4],
        ss[0], ss[1], ss[2], ss[3],
        ims[0], ims[1],
        q_rot.stride(0), q_rot.stride(1),
        q_sketch.stride(0), q_sketch.stride(1),
        out.stride(0), out.stride(1),
        context_len, D,
        pool.k_indices.shape[-1], pool.k_qjl.shape[-1], pool.v_indices.shape[-1],
        k_bits, k_vpb, v_bits, v_vpb,
        n_heads, kv_cache.n_kv_heads, kv_cache.group_size, tokens_per_block,
        qjl_scale, sm_scale, 
        quest_threshold,
        128, num_warps=4
    )
    
    if D > kv_cache.head_dim:
        out = out[..., :kv_cache.head_dim]
        
    return out.reshape(query.shape).to(query.dtype)

def paged_turboquant_attention(query: torch.Tensor, kv_cache: Any) -> torch.Tensor:
    # High-level entry point for direct testing
    return turboquant_paged_fused_attention(
        query, kv_cache, 
        kv_cache.k_quantizer.bits - 1 if kv_cache.k_quantizer else 4,
        4, 1.0, 1.0 / (query.shape[-1]**0.5)
    )
