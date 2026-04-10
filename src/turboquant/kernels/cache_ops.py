import math
import torch
import triton
import triton.language as tl
from typing import Optional, Any

def _next_pow2(n):
    return 2**math.ceil(math.log2(n))

@triton.jit
def _fused_cache_update_kernel(
    K_POOL_INDICES_ptr, K_POOL_SIGNS_ptr, K_POOL_META_ptr,
    V_POOL_INDICES_ptr, V_POOL_META_ptr,
    NEW_K_INDICES_ptr, NEW_K_SIGNS_ptr, NEW_K_META_ptr,
    NEW_V_INDICES_ptr, NEW_V_META_ptr,
    BLOCK_TABLE_ptr,
    NUM_TOKENS_ptr,
    LAYER_IDX,
    stride_k_idx_l, stride_k_idx_b, stride_k_idx_h, stride_k_idx_s, stride_k_idx_d,
    stride_k_sgn_l, stride_k_sgn_b, stride_k_sgn_h, stride_k_sgn_s, stride_k_sgn_d,
    stride_k_meta_l, stride_k_meta_b, stride_k_meta_h, stride_k_meta_s, stride_k_meta_d,
    stride_v_idx_l, stride_v_idx_b, stride_v_idx_h, stride_v_idx_s, stride_v_idx_d,
    stride_v_meta_l, stride_v_meta_b, stride_v_meta_h, stride_v_meta_s, stride_v_meta_d,
    stride_new_k_idx_h, stride_new_k_idx_d,
    stride_new_k_sgn_h, stride_new_k_sgn_d,
    stride_new_k_meta_h, stride_new_k_meta_d,
    stride_new_v_idx_h, stride_new_v_idx_d,
    stride_new_v_meta_h, stride_new_v_meta_d,
    TOKENS_PER_BLOCK,
    HEAD_IDX_START,
    BLOCK_D_K_IDX: tl.constexpr,
    BLOCK_D_K_SGN: tl.constexpr,
    BLOCK_D_K_META: tl.constexpr,
    BLOCK_D_V_IDX: tl.constexpr,
    BLOCK_D_V_META: tl.constexpr,
    N_HEADS_PER_PROGRAM: tl.constexpr,
):
    # Program ID handles multiple heads if requested
    pid = tl.program_id(0)
    head_idx = HEAD_IDX_START + pid
    
    # 1. Load current context length from GPU memory
    num_tokens = tl.load(NUM_TOKENS_ptr)
    
    # 2. Compute Paged Slot
    block_idx_in_layer = num_tokens // TOKENS_PER_BLOCK
    slot_offset = num_tokens % TOKENS_PER_BLOCK
    
    # 3. Resolve Physical Block ID
    physical_block_id = tl.load(BLOCK_TABLE_ptr + block_idx_in_layer)
    
    # 4. Compute Pool Pointers for this slot
    # K Indices
    k_idx_p = K_POOL_INDICES_ptr + LAYER_IDX * stride_k_idx_l + physical_block_id * stride_k_idx_b + \
              head_idx * stride_k_idx_h + slot_offset * stride_k_idx_s
    # K Signs
    k_sgn_p = K_POOL_SIGNS_ptr + LAYER_IDX * stride_k_sgn_l + physical_block_id * stride_k_sgn_b + \
              head_idx * stride_k_sgn_h + slot_offset * stride_k_sgn_s
    # K Meta
    k_meta_p = K_POOL_META_ptr + LAYER_IDX * stride_k_meta_l + physical_block_id * stride_k_meta_b + \
               head_idx * stride_k_meta_h + slot_offset * stride_k_meta_s
    # V Indices
    v_idx_p = V_POOL_INDICES_ptr + LAYER_IDX * stride_v_idx_l + physical_block_id * stride_v_idx_b + \
              head_idx * stride_v_idx_h + slot_offset * stride_v_idx_s
    # V Meta
    v_meta_p = V_POOL_META_ptr + LAYER_IDX * stride_v_meta_l + physical_block_id * stride_v_meta_b + \
               head_idx * stride_v_meta_h + slot_offset * stride_v_meta_s

    # 5. Load New Data and Store to Pool
    # K Indices
    d_k_idx = tl.arange(0, BLOCK_D_K_IDX)
    new_k_idx = tl.load(NEW_K_INDICES_ptr + pid * stride_new_k_idx_h + d_k_idx)
    tl.store(k_idx_p + d_k_idx * stride_k_idx_d, new_k_idx)

    # K Signs
    d_k_sgn = tl.arange(0, BLOCK_D_K_SGN)
    new_k_sgn = tl.load(NEW_K_SIGNS_ptr + pid * stride_new_k_sgn_h + d_k_sgn)
    tl.store(k_sgn_p + d_k_sgn * stride_k_sgn_d, new_k_sgn)

    # K Meta
    d_k_meta = tl.arange(0, BLOCK_D_K_META)
    new_k_meta = tl.load(NEW_K_META_ptr + pid * stride_new_k_meta_h + d_k_meta)
    tl.store(k_meta_p + d_k_meta * stride_k_meta_d, new_k_meta)

    # V Indices
    d_v_idx = tl.arange(0, BLOCK_D_V_IDX)
    new_v_idx = tl.load(NEW_V_INDICES_ptr + pid * stride_new_v_idx_h + d_v_idx)
    tl.store(v_idx_p + d_v_idx * stride_v_idx_d, new_v_idx)

    # V Meta
    d_v_meta = tl.arange(0, BLOCK_D_V_META)
    new_v_meta = tl.load(NEW_V_META_ptr + pid * stride_new_v_meta_h + d_v_meta)
    tl.store(v_meta_p + d_v_meta * stride_v_meta_d, new_v_meta)

def fused_cache_update(
    kv_cache: Any,
    k_q: Any, # ProdQuantized
    v_q: Any, # ValueQuantized
):
    """
    SOTA: Fused Cache Update for CUDA Graphs.
    Eliminates all Python indexing and slicing.
    """
    pool = kv_cache.pool
    num_tokens_ptr = kv_cache.num_tokens_ptr
    block_table = kv_cache.block_table
    layer_idx = kv_cache.layer_idx
    
    # We assume batch_size=1 and seq_len=1 for decoding
    # SOTA v9.4: Use flat views to avoid Python-side squeeze overhead
    k_idx = k_q.mse_indices.view(-1, k_q.mse_indices.shape[-1])
    k_sgn = k_q.qjl_signs.view(-1, k_q.qjl_signs.shape[-1])
    k_meta = k_q.meta.view(-1, k_q.meta.shape[-1])
    
    v_idx = v_q.indices.view(-1, v_q.indices.shape[-1])
    v_meta = v_q.meta.view(-1, v_q.meta.shape[-1])
    
    n_heads = k_q.mse_indices.shape[1]
    
    grid = (n_heads,)
    _fused_cache_update_kernel[grid](
        pool.k_indices, pool.k_qjl, pool.k_metadata,
        pool.v_indices, pool.v_metadata,
        k_idx, k_sgn, k_meta,
        v_idx, v_meta,
        block_table,
        num_tokens_ptr,
        layer_idx,
        pool.k_indices.stride(0), pool.k_indices.stride(1), pool.k_indices.stride(2), pool.k_indices.stride(3), pool.k_indices.stride(4),
        pool.k_qjl.stride(0), pool.k_qjl.stride(1), pool.k_qjl.stride(2), pool.k_qjl.stride(3), pool.k_qjl.stride(4),
        pool.k_metadata.stride(0), pool.k_metadata.stride(1), pool.k_metadata.stride(2), pool.k_metadata.stride(3), pool.k_metadata.stride(4),
        pool.v_indices.stride(0), pool.v_indices.stride(1), pool.v_indices.stride(2), pool.v_indices.stride(3), pool.v_indices.stride(4),
        pool.v_metadata.stride(0), pool.v_metadata.stride(1), pool.v_metadata.stride(2), pool.v_metadata.stride(3), pool.v_metadata.stride(4),
        k_idx.stride(0), k_idx.stride(1),
        k_sgn.stride(0), k_sgn.stride(1),
        k_meta.stride(0), k_meta.stride(1),
        v_idx.stride(0), v_idx.stride(1),
        v_meta.stride(0), v_meta.stride(1),
        pool.tokens_per_block,
        0,
        BLOCK_D_K_IDX=_next_pow2(k_idx.shape[1]),
        BLOCK_D_K_SGN=_next_pow2(k_sgn.shape[1]),
        BLOCK_D_K_META=_next_pow2(k_meta.shape[1]),
        BLOCK_D_V_IDX=_next_pow2(v_idx.shape[1]),
        BLOCK_D_V_META=_next_pow2(v_meta.shape[1]),
        N_HEADS_PER_PROGRAM=1,
        num_warps=4
    )
