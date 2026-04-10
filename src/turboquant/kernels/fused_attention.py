"""
TurboQuant Fused Attention — PyTorch Reference Implementation.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Any

from ..quant.key_quantizer import TurboQuantProd, TurboQuantMSE
from ..quant.quant_base import ProdQuantized, MSEQuantized
from ..quant.value_quantizer import ValueQuantized
from .fused_ref import attention_score_mse, attention_score_prod

try:
    from .triton_attention import turboquant_attention_score
    from .paged_fused import turboquant_paged_fused_attention
    from .triton_fused import turboquant_fused_decode
    from ..quant.lloyd_max import LM_CENTROIDS
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def attention_score_prod_dispatch(
    query: torch.Tensor,
    quantized_key: ProdQuantized,
    quantizer: TurboQuantProd,
    qjl_scale: float,
    sm_scale: float,
    tq_config: Optional[Any] = None,
) -> torch.Tensor:
    if HAS_TRITON and query.is_cuda:
        q_rot, q_sketch = quantizer.transform_query(query)
        block_n = tq_config.hw.triton_block_n if tq_config else None
        num_warps = tq_config.hw.triton_num_warps if tq_config else None
        scores = turboquant_attention_score(q_rot, q_sketch, quantized_key, quantizer.bits - 1, qjl_scale, block_n=block_n, num_warps=num_warps)
        if query.dim() > scores.dim():
            scores = scores.view(query.shape[:-1] + (scores.shape[-1],))
        return scores * sm_scale
    return attention_score_prod(query, quantized_key, quantizer, sm_scale)

def attention_score_mse_dispatch(
    query: torch.Tensor,
    quantized_key: MSEQuantized,
    quantizer: TurboQuantMSE,
    sm_scale: float,
    tq_config: Optional[Any] = None,
) -> torch.Tensor:
    if HAS_TRITON and query.is_cuda:
        from .triton_mse import turboquant_mse_score
        q_rot = quantizer.transform_query(query)
        block_n = tq_config.hw.triton_block_n if tq_config else None
        num_warps = tq_config.hw.triton_num_warps if tq_config else None
        scores = turboquant_mse_score(q_rot, quantized_key.indices, quantized_key.norms, quantized_key.scales, 
                                     centroids=None, mse_bits=quantizer.bits, block_n=block_n, num_warps=num_warps)
        if query.dim() > scores.dim():
            scores = scores.view(query.shape[:-1] + (scores.shape[-1],))
        return scores * sm_scale
    return attention_score_mse(query, quantized_key, quantizer, sm_scale)

def paged_turboquant_attention(
    query: torch.Tensor,
    kv_cache: Any,
    k_bits: int,
    v_bits: int,
    qjl_scale: float,
    sm_scale: float,
    quest_threshold: float = 1e-4,
    mask: Optional[torch.Tensor] = None,
    k_centroids: Optional[torch.Tensor] = None,
    v_centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Consolidated Dispatcher for Paged Attention Path.
    Handles Strategy-based precision branching (FP16 Fallback vs Triton Kernel).
    """
    seq_q = query.shape[-2]
    
    if kv_cache.k_quantizer is None or seq_q > 1:
        ptrs = kv_cache.get_paged_ptrs()
        # Use Python shadow for FP16 fallback path (avoids GPU tensor → Python sync)
        num_tokens = kv_cache.num_tokens
        tokens_per_block = ptrs["tokens_per_block"]
        block_table = ptrs["block_table"]
        
        if num_tokens == 0:
            return torch.zeros_like(query)
            
        k_all = []
        v_all = []
        
        if kv_cache.k_quantizer is not None:
            from ..quant.quant_base import unpack_indices
            from .paged_fused import compute_centroids
            pool = kv_cache.pool
            for i in range(num_tokens):
                block_idx = i // tokens_per_block
                slot_idx = i % tokens_per_block
                p_id = block_table[block_idx].item()
                li = kv_cache.layer_idx
                
                ki = pool.k_indices[li, p_id, :, slot_idx]
                ksi = pool.k_qjl[li, p_id, :, slot_idx]
                k_meta = pool.k_metadata[li, p_id, :, slot_idx]
                
                k_sub = pool.k_subblocks
                k_gs = pool.k_group_size
                
                # FIX: Thêm .contiguous() để xử lý phân mảnh bộ nhớ của Paged Attention
                ki_reshaped = ki.to(torch.long).contiguous().view(n_heads, k_sub, k_gs)
                k_meta_reshaped = k_meta.contiguous().view(n_heads, k_sub, 3)
                
                kn = k_meta_reshaped[..., 0].unsqueeze(-1)
                ks = k_meta_reshaped[..., 1].unsqueeze(-1) 
                kr = k_meta_reshaped[..., 2].unsqueeze(-1) 
                
                if k_centroids is None:
                    k_centroids = compute_centroids(k_bits - 1, dist='gaussian').to(query.device)
                
                k_mse_sub = k_centroids[ki_reshaped] * kn * ks 
                k_mse = k_mse_sub.reshape(n_heads, -1)
                
                ksi_unpacked = unpack_indices(ksi.unsqueeze(0), 1, pool.head_dim).squeeze(0)
                # FIX: Thêm .contiguous()
                ksi_reshaped = ksi_unpacked.float().contiguous().view(n_heads, k_sub, k_gs)
                k_qjl_sub = (ksi_reshaped * 2.0 - 1.0) * kr * qjl_scale
                k_qjl = k_qjl_sub.reshape(n_heads, -1)
                
                k_all.append((k_mse + k_qjl).to(query.dtype))
                
                # --- Dequantize V ---
                vi = pool.v_indices[li, p_id, :, slot_idx]
                v_meta = pool.v_metadata[li, p_id, :, slot_idx]
                
                v_sub = pool.v_subblocks
                v_gs = pool.v_group_size
                
                # FIX: Thêm .contiguous()
                vi_reshaped = vi.to(torch.long).contiguous().view(n_heads, v_sub, v_gs)
                v_meta_reshaped = v_meta.contiguous().view(n_heads, v_sub, 2)
                
                vn = v_meta_reshaped[..., 0].unsqueeze(-1) 
                vs = v_meta_reshaped[..., 1].unsqueeze(-1) 
                
                if v_centroids is None:
                    v_centroids = compute_centroids(v_bits, dist='gaussian').to(query.device)
                
                v_recon_unit_sub = v_centroids[vi_reshaped]
                v_rot_scaled = v_recon_unit_sub * vs
                
                from ..ops.rotation import TurboQuantRotation
                rot_v = TurboQuantRotation(v_gs, n_passes=1).to(query.device)
                v_unit_sub = rot_v.inverse(v_rot_scaled)
                
                v_deq_sub = v_unit_sub * vn
                v_all.append(v_deq_sub.reshape(n_heads, -1).to(query.dtype))
        elif hasattr(kv_cache, 'static_k_fp16'):
            # SOTA v8.8: Static Workspace Path (CUDA Graph Safe)
            # Use the pre-allocated flat buffers and the GPU-updated mask
            k_cache = kv_cache.static_k_fp16
            v_cache = kv_cache.static_v_fp16
            current_mask = kv_cache.static_attn_mask
        else:
            # FIX: Nhánh FP16 Thuần (Dành cho Protected Layers như Layer 0, Layer 23)
            # Use block_ids (Python list from prefill), NOT block_table (pre-allocated for Triton)
            for i in range(num_tokens):
                block_idx = i // tokens_per_block
                slot_idx = i % tokens_per_block
                physical_block = kv_cache.block_ids[block_idx]
                k_all.append(kv_cache.k_fp16[physical_block][:, slot_idx])
                v_all.append(kv_cache.v_fp16[physical_block][:, slot_idx])
            
            k_cache = torch.stack(k_all, dim=1).unsqueeze(0).to(query.dtype)
            v_cache = torch.stack(v_all, dim=1).unsqueeze(0).to(query.dtype)
            current_mask = mask if seq_q > 1 else None
        
        if query.shape[1] != k_cache.shape[1]:
            scale = query.shape[1] // k_cache.shape[1]
            k_cache = k_cache.repeat_interleave(scale, dim=1)
            v_cache = v_cache.repeat_interleave(scale, dim=1)
            
        real_is_causal = (seq_q > 1) and (current_mask is None)
        
        return torch.nn.functional.scaled_dot_product_attention(
            query, k_cache.to(query.dtype), v_cache.to(query.dtype), 
            attn_mask=current_mask.to(query.dtype) if current_mask is not None else None, 
            is_causal=real_is_causal
        )

    if not HAS_TRITON or not query.is_cuda:
        raise RuntimeError("Paged Attention requires Triton and CUDA.")
    
    from .paged_fused import turboquant_paged_fused_attention as paged_dispatch
    return paged_dispatch(query, kv_cache, k_bits, v_bits, qjl_scale, sm_scale, mask=mask, k_centroids=k_centroids, v_centroids=v_centroids, quest_threshold=quest_threshold)

def turboquant_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    quantizer: Optional[Any] = None,
    v_bits: int = 4,
    qjl_scale: float = 1.0,
    sm_scale: Optional[float] = None,
    kv_cache: Optional[Any] = None,
    causal_mask: Optional[torch.Tensor] = None,
    k_bits: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[-1]**0.5)
        
    if qjl_scale == 1.0:
        if quantizer is not None:
            qjl_scale = getattr(quantizer, "qjl_scale", 1.0)
        elif kv_cache is not None and kv_cache.k_quantizer is not None:
            qjl_scale = getattr(kv_cache.k_quantizer, "qjl_scale", 1.0)

    if kv_cache is not None:
        if kv_cache.strategy and kv_cache.strategy.name == "FP16":
            pass 
        else:
            k_bits_val = k_bits if k_bits is not None else (kv_cache.k_quantizer.bits if kv_cache.k_quantizer else 4)
            v_bits_val = v_bits if v_bits is not None else (kv_cache.v_quantizer.bits if kv_cache.v_quantizer else 4)
            output = paged_turboquant_attention(query, kv_cache, k_bits_val, v_bits_val, qjl_scale, sm_scale)
            return output, None

    # Triton Fused Contiguous Path (Decode Stage)
    if HAS_TRITON and query.is_cuda and query.shape[-2] == 1 and \
       isinstance(key, ProdQuantized) and isinstance(value, ValueQuantized):
        
        q_rot, q_sketch = quantizer.transform_query(query)
        centroids = LM_CENTROIDS[quantizer.bits - 1].to(query.device, query.dtype)
        
        # Use the new SOTA fused kernel
        output = turboquant_fused_decode(
            q_rot=q_rot,
            q_sketch=q_sketch,
            quantized_key=key,
            value_quantized=value,
            centroids=centroids,
            k_bits=quantizer.bits,
            v_bits=value.bits,
            qjl_scale=qjl_scale,
            sm_scale=sm_scale
        )
        return output.view(query.shape[:-1] + (query.shape[-1],)), None

    if isinstance(key, ProdQuantized):
        scores = attention_score_prod_dispatch(query, key, quantizer, qjl_scale, sm_scale)
    elif isinstance(key, MSEQuantized):
        scores = attention_score_mse_dispatch(query, key, quantizer, sm_scale)
    else:
        scores = torch.matmul(query, key.transpose(-1, -2)) * sm_scale

    if causal_mask is not None:
        if causal_mask.dtype == torch.bool:
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        else:
            scores = scores + causal_mask

    weights = torch.softmax(scores, dim=-1)

    if isinstance(value, ValueQuantized):
        from ..quant.value_quantizer import TurboQuantValue
        n_subblocks = value.norms.shape[-1]
        v_block_size = math.ceil(query.shape[-1] / n_subblocks)
        v_dequantizer = TurboQuantValue(query.shape[-1], bits=value.bits, block_size=v_block_size).to(query.device)
        v_tensor = v_dequantizer.dequantize(value)
    else:
        v_tensor = value

    output = torch.matmul(weights, v_tensor.float())
    return output.to(query.dtype), weights