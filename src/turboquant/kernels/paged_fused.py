import torch
import triton
import triton.language as tl
from typing import Dict, Any

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
    K_INDICES_ptr, K_SIGNS_ptr, K_METADATA_ptr,
    V_INDICES_ptr, V_METADATA_ptr,
    K_CENTROIDS_ptr,
    BLOCK_TABLE_ptr,
    K_SUMMARIES_ptr,        # Quest: Min/Max per block
    BLOCK_IMPORTANCE_ptr,   # H2O: Cumulative scores
    OUT_ptr,
    stride_k_index_b, stride_k_index_h, stride_k_index_s, stride_k_index_d,
    stride_k_signs_b, stride_k_signs_h, stride_k_signs_s, stride_k_signs_d,
    stride_k_meta_b, stride_k_meta_h, stride_k_meta_attr, # Updated Meta K
    stride_v_index_b, stride_v_index_h, stride_v_index_s, stride_v_index_d,
    stride_v_meta_b, stride_v_meta_h, stride_v_meta_g, stride_v_meta_attr, # Updated Meta V
    stride_sum_b, stride_sum_h, stride_sum_attr, stride_sum_d,
    stride_imp_b, stride_imp_h,
    stride_qr_bh, stride_qr_d,
    stride_o_bh, stride_o_d,
    N, D,
    K_BITS, K_VALS_PER_BYTE,
    V_BITS, V_VALS_PER_BYTE,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
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
    
    q_rot = tl.load(Q_ROT_ptr + pid_bh * stride_qr_bh + d_range, mask=d_range < D, other=0.0).to(tl.float32)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([128], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n_range = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_range < N
        
        # 1. Quest: Query-Aware Sparsity Check
        p_idx = start_n // BLOCK_SIZE
        pb_id = tl.load(BLOCK_TABLE_ptr + p_idx, mask=(start_n < N))
        
        k_min = tl.load(K_SUMMARIES_ptr + pb_id * stride_sum_b + pid_kv_h * stride_sum_h + 0 * stride_sum_attr + d_range, mask=d_mask, other=0.0)
        k_max = tl.load(K_SUMMARIES_ptr + pb_id * stride_sum_b + pid_kv_h * stride_sum_h + 1 * stride_sum_attr + d_range, mask=d_mask, other=0.0)
        
        q_min = q_rot * k_min
        q_max = q_rot * k_max
        quest_bound = tl.sum(tl.maximum(q_min, q_max)) * SM_SCALE
        
        if quest_bound >= QUEST_THRESHOLD:
            block_indices = (n_range / BLOCK_SIZE).to(tl.int32)
            slot_indices = n_range - (block_indices * BLOCK_SIZE)
            physical_block_ids = tl.load(BLOCK_TABLE_ptr + block_indices, mask=n_mask, other=0)
            
            # SOTA Pillar 3: Optimized Load Paths
            k_byte_base = K_INDICES_ptr + physical_block_ids[:, None] * stride_k_index_b + pid_kv_h * stride_k_index_h
            k_sig_base = K_SIGNS_ptr + physical_block_ids[:, None] * stride_k_signs_b + pid_kv_h * stride_k_signs_h
            k_met_base = K_METADATA_ptr + physical_block_ids[:, None] * stride_k_meta_b + pid_kv_h * stride_k_meta_h
            
            k_byte_idx = d_range // K_VALS_PER_BYTE
            k_sub_idx = d_range % K_VALS_PER_BYTE
            k_idx_packed = tl.load(k_byte_base + slot_indices[:, None] * stride_k_index_s + k_byte_idx[None, :] * stride_k_index_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
            ki = (k_idx_packed >> (k_sub_idx * K_BITS)).to(tl.int32) & ((1 << K_BITS) - 1)

            ks_byte_idx = d_range // 8
            ks_sub_idx = d_range % 8
            k_sig_packed = tl.load(k_sig_base + slot_indices[:, None] * stride_k_signs_s + ks_byte_idx[None, :] * stride_k_signs_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
            ksi = (k_sig_packed >> ks_sub_idx) & 1
            
            # Key Metadata (Shared per Block - SOTA Pillar 2)
            # Actually, K Meta in my manager was still per-slot for norm? 
            # No, I changed it to per-block: self.pool.k_metadata[current_block_id, :, 0]
            kn = tl.load(k_met_base + 0 * stride_k_meta_attr)
            ks = tl.load(k_met_base + 1 * stride_k_meta_attr)
            kr = tl.load(k_met_base + 2 * stride_k_meta_attr)
            
            k_mse = tl.load(K_CENTROIDS_ptr + ki).to(tl.float32) * kn * ks
            k_qjl = (ksi.to(tl.float32) * 2.0 - 1.0) * kr * QJL_SCALE
            
            # SOTA V1.1.0: Direct Sign (Unified Query Domain)
            scores = tl.sum(q_rot[None, :] * (k_mse + k_qjl), 1) * SM_SCALE
            scores = tl.where(n_mask, scores, float("-inf"))
            
            m_new = tl.maximum(m_i, tl.max(scores, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            
            # H2O: Importance Accumulation
            block_importance = tl.sum(p)
            tl.atomic_add(BLOCK_IMPORTANCE_ptr + pb_id * stride_imp_b + pid_kv_h * stride_imp_h, block_importance)

            # SOTA Pillar 3: Vectorized Asymmetric V-Dequantization
            v_idx_base = V_INDICES_ptr + physical_block_ids[:, None] * stride_v_index_b + pid_kv_h * stride_v_index_h
            v_met_base = V_METADATA_ptr + physical_block_ids[:, None] * stride_v_meta_b + pid_kv_h * stride_v_meta_h
            
            v_byte_idx = d_range // V_VALS_PER_BYTE
            v_sub_idx = d_range % V_VALS_PER_BYTE
            v_idx_packed = tl.load(v_idx_base + slot_indices[:, None] * stride_v_index_s + v_byte_idx[None, :] * stride_v_index_d, mask=(n_mask[:,None] & d_mask[None,:]), other=0)
            vi = (v_idx_packed >> (v_sub_idx * V_BITS)).to(tl.int32) & ((1 << V_BITS) - 1)
            
            # Value Metadata: Block-Aligned Linear Scaling
            # Loaded once per head per block (No slot index!)
            v_scale = tl.load(v_met_base + 0 * stride_v_meta_g + 0 * stride_v_meta_attr) 
            v_zeros = tl.load(v_met_base + 0 * stride_v_meta_g + 1 * stride_v_meta_attr)
            
            # V = idx * scale + min (Pillar 1: Absolute Asymmetric)
            v_deq = vi.to(tl.float32) * v_scale + v_zeros
            
            acc = acc * alpha + tl.sum(p[:, None] * v_deq, 0)
            l_i = l_i * alpha + tl.sum(p, 0)
            m_i = m_new

    # SOTA: add safety epsilon to prevent NaN if all blocks are skipped
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_range, (acc / (l_i + 1e-10)).to(tl.float32), mask=d_mask)

def turboquant_paged_fused_attention(
    query: torch.Tensor,
    kv_cache: Any,
    k_bits: int,
    v_bits: int,
    qjl_scale: float,
    sm_scale: float,
    quest_threshold: float = 1e-4
) -> torch.Tensor:
    n_heads = kv_cache.n_heads
    head_dim = kv_cache.head_dim
    q_view = query.reshape(n_heads, head_dim)
    
    q_rot, _ = kv_cache.k_quantizer.transform_query(q_view)
    D = q_rot.shape[-1]
    BH = n_heads 
    q_rot = q_rot.reshape(BH, D)
    
    # SOTA Pillar 1: Luôn trừ 1 bit cho QJL Sign!
    k_mse_bits = k_bits - 1 
    
    ptrs = kv_cache.get_paged_ptrs()
    pool = ptrs["pool"]
    block_table = ptrs["block_table"]
    context_len = ptrs["num_tokens"]
    if context_len == 0:
        return torch.zeros_like(query)

    # Tạo bảng Centroid ĐÚNG SỐ BIT
    k_centroids = compute_centroids(k_mse_bits, dist='gaussian').to(query.device, query.dtype)
    
    out = torch.zeros((BH, D), device=query.device, dtype=torch.float32)
    tokens_per_block = pool.tokens_per_block
    
    # Tính bước nhảy unpack ĐÚNG SỐ BIT (Pillar 3)
    k_vpb = 8 // k_mse_bits 
    v_vpb = 8 // v_bits
    
    kis, kqs, kms = pool.k_indices.stride(), pool.k_qjl.stride(), pool.k_metadata.stride()
    vis, vms = pool.v_indices.stride(), pool.v_metadata.stride()
    ss, ims = pool.k_summaries.stride(), pool.block_importance.stride()
    
    grid = (BH,)
    _turboquant_paged_fused_kernel[grid](
        Q_ROT_ptr=q_rot,
        K_INDICES_ptr=pool.k_indices,
        K_SIGNS_ptr=pool.k_qjl,
        K_METADATA_ptr=pool.k_metadata,
        V_INDICES_ptr=pool.v_indices,
        V_METADATA_ptr=pool.v_metadata, 
        K_CENTROIDS_ptr=k_centroids, 
        BLOCK_TABLE_ptr=block_table, 
        K_SUMMARIES_ptr=pool.k_summaries, 
        BLOCK_IMPORTANCE_ptr=pool.block_importance,
        OUT_ptr=out,
        stride_k_index_b=kis[0], stride_k_index_h=kis[1], stride_k_index_s=kis[2], stride_k_index_d=kis[3],
        stride_k_signs_b=kqs[0], stride_k_signs_h=kqs[1], stride_k_signs_s=kqs[2], stride_k_signs_d=kqs[3],
        stride_k_meta_b=kms[0], stride_k_meta_h=kms[1], stride_k_meta_attr=kms[2],
        stride_v_index_b=vis[0], stride_v_index_h=vis[1], stride_v_index_s=vis[2], stride_v_index_d=vis[3],
        stride_v_meta_b=vms[0], stride_v_meta_h=vms[1], stride_v_meta_g=vms[2], stride_v_meta_attr=vms[3],
        stride_sum_b=ss[0], stride_sum_h=ss[1], stride_sum_attr=ss[2], stride_sum_d=ss[3],
        stride_imp_b=ims[0], stride_imp_h=ims[1],
        stride_qr_bh=q_rot.stride(0), stride_qr_d=q_rot.stride(1),
        stride_o_bh=out.stride(0), stride_o_d=out.stride(1),
        N=context_len, D=D,
        K_BITS=k_mse_bits, K_VALS_PER_BYTE=k_vpb,
        V_BITS=v_bits, V_VALS_PER_BYTE=v_vpb,
        N_HEADS=n_heads,
        N_KV_HEADS=kv_cache.n_kv_heads,
        BLOCK_SIZE=tokens_per_block,
        QJL_SCALE=qjl_scale,
        SM_SCALE=sm_scale, 
        QUEST_THRESHOLD=quest_threshold,
        BLOCK_N=128,
        num_warps=4
    )
    
    if D > kv_cache.head_dim:
        out = out[..., :kv_cache.head_dim]
        
    return out.reshape(query.shape).to(query.dtype)

def paged_turboquant_attention(query: torch.Tensor, kv_cache: Any, k_bits: int, v_bits: int, qjl_scale: float, sm_scale: float) -> torch.Tensor:
    # High-level entry point for direct testing
    return turboquant_paged_fused_attention(
        query, kv_cache, k_bits, v_bits, qjl_scale, sm_scale
    )
