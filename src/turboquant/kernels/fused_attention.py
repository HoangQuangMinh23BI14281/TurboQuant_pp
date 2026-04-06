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
        block_n = tq_config.triton_block_n if tq_config else None
        num_warps = tq_config.triton_num_warps if tq_config else None
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
        block_n = tq_config.triton_block_n if tq_config else None
        num_warps = tq_config.triton_num_warps if tq_config else None
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
) -> torch.Tensor:
    """
    Consolidated Dispatcher for Paged Attention Path.
    Handles Strategy-based precision branching (FP16 Fallback vs Triton Kernel).
    """
    # SOTA: Precision Branching (Boundary Protection Support)
    seq_q = query.shape[-2]
    
    # 1. FP16 Path or Quantized Prefill (PyTorch SDPA)
    if kv_cache.k_quantizer is None or seq_q > 1:
        ptrs = kv_cache.get_paged_ptrs()
        num_tokens = ptrs["num_tokens"]
        tokens_per_block = ptrs["tokens_per_block"]
        block_table = ptrs["block_table"]
        
        if num_tokens == 0:
            return torch.zeros_like(query)
            
        k_all = []
        v_all = []
        
        # SOTA: Hybrid Prefill (Lazy Dequantizer)
        if kv_cache.k_quantizer is not None:
            # Layer is quantized, but we are in Prefill -> Dequantize on-the-fly
            from ..quant.quant_base import unpack_indices
            from .paged_fused import compute_centroids
            pool = kv_cache.pool
            for i in range(num_tokens):
                block_idx = i // tokens_per_block
                slot_idx = i % tokens_per_block
                p_id = block_table[block_idx].item()
                li = kv_cache.layer_idx
                
                # Fetch MSE + Signs
                ki = pool.k_indices[li, p_id, :, slot_idx]
                ksi = pool.k_qjl[li, p_id, :, slot_idx]
                kn = pool.k_metadata[li, p_id, :, slot_idx, 0]
                ks = pool.k_metadata[li, p_id, :, slot_idx, 1]
                kr = pool.k_metadata[li, p_id, :, slot_idx, 2]
                
                # Dequantize K (Algorithm 2 Restoration)
                k_centroids = compute_centroids(k_bits - 1, dist='gaussian').to(query.device)
                k_mse = k_centroids[ki.to(torch.long)] * kn.unsqueeze(-1) * ks.unsqueeze(-1)
                
                # QJL Restoration (Direct Sign)
                ksi_unpacked = unpack_indices(ksi.unsqueeze(0), 1, pool.head_dim).squeeze(0)
                k_qjl = (ksi_unpacked.float() * 2.0 - 1.0) * kr.unsqueeze(-1) * qjl_scale
                
                k_all.append((k_mse + k_qjl).to(query.dtype))
                
                # Dequantize V
                v_vi = pool.v_indices[li, p_id, :, slot_idx]
                v_sc = pool.v_metadata[li, p_id, :, :, 0]
                v_zp = pool.v_metadata[li, p_id, :, :, 1]
                
                # Unpack dynamic (4-bit LSB or 8-bit)
                if v_bits == 4:
                    v_unpacked = torch.stack([v_vi & 0xF, v_vi >> 4], dim=-1).flatten(1)
                else:
                    v_unpacked = v_vi.float()
                v_deq = v_unpacked.float() * v_sc + v_zp
                v_all.append(v_deq.to(query.dtype))
        else:
            # Native FP16 Path
            for i in range(num_tokens):
                block_idx = i // tokens_per_block
                slot_idx = i % tokens_per_block
                physical_block = block_table[block_idx].item()
                k_all.append(kv_cache.k_fp16[physical_block][:, slot_idx])
                v_all.append(kv_cache.v_fp16[physical_block][:, slot_idx])
            
        k_cache = torch.stack(k_all, dim=1).unsqueeze(0).to(query.dtype)
        v_cache = torch.stack(v_all, dim=1).unsqueeze(0).to(query.dtype)
        
        # GQA Repeat Interleave for SDPA
        if query.shape[1] != k_cache.shape[1]:
            scale = query.shape[1] // k_cache.shape[1]
            k_cache = k_cache.repeat_interleave(scale, dim=1)
            v_cache = v_cache.repeat_interleave(scale, dim=1)
            
        current_mask = mask if seq_q > 1 else None
        real_is_causal = (seq_q > 1) and (current_mask is None)
        
        return torch.nn.functional.scaled_dot_product_attention(
            query, k_cache.to(query.dtype), v_cache.to(query.dtype), 
            attn_mask=current_mask, is_causal=real_is_causal
        )

    # 2. Triton Path (Decode Stage - Only 1 query token)
    if not HAS_TRITON or not query.is_cuda:
        raise RuntimeError("Paged Attention requires Triton and CUDA.")
    
    from .paged_fused import turboquant_paged_fused_attention as paged_dispatch
    return paged_dispatch(query, kv_cache, k_bits, v_bits, qjl_scale, sm_scale, quest_threshold)

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
        
    # SOTA: Auto-resolve QJL scale from quantizer if not explicitly overridden
    if qjl_scale == 1.0:
        if quantizer is not None:
            qjl_scale = getattr(quantizer, "qjl_scale", 1.0)
        elif kv_cache is not None and kv_cache.k_quantizer is not None:
            qjl_scale = getattr(kv_cache.k_quantizer, "qjl_scale", 1.0)

    # 1. Paged Attention Path
    if kv_cache is not None:
        if kv_cache.strategy and kv_cache.strategy.name == "FP16":
            pass # allow fallback to contiguous for fp16 if no use_paged property or if requested
        else:
            k_bits_val = k_bits if k_bits is not None else (kv_cache.k_quantizer.bits if kv_cache.k_quantizer else 4)
            v_bits_val = v_bits if v_bits is not None else (kv_cache.v_quantizer.bits if kv_cache.v_quantizer else 4)
            output = paged_turboquant_attention(query, kv_cache, k_bits_val, v_bits_val, qjl_scale, sm_scale)
            return output, None

    # 2. Contiguous Path
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
        from turboquant.quant.value_quantizer import TurboQuantValue
        v_dequantizer = TurboQuantValue(query.shape[-1], bits=v_bits).to(query.device)
        v_tensor = v_dequantizer.dequantize(value)
    else:
        v_tensor = value

    output = torch.matmul(weights, v_tensor.float())
    return output.to(query.dtype), weights
