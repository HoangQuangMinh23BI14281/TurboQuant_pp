import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from turboquant.layers.config import TurboQuantConfig
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.kernels.fused_attention import turboquant_attention
from turboquant.kernels.fused_attention import paged_turboquant_attention
from turboquant.kernels.paged_fused import compute_centroids  # Module-level: no runtime import during decode
from turboquant.quant.lloyd_max import harden_lloyd_max # SOTA v8.7: Guard against .to() during capture
from turboquant.cache.routing import QuantizationStrategy
from turboquant.ops.rope import apply_rotary_pos_emb

class TurboQuantAttention(nn.Module):
    """
    Standard TurboQuant++ Attention Layer with Architectural Branching.
    Routes between Native FP16 (Exempt) and Dual-LUT Quantized (Active) paths.
    """
    def __init__(
        self,
        tq_config: TurboQuantConfig,
        layer_idx: int,
        total_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        q_bias: bool = False,
        k_bias: bool = False,
        v_bias: bool = False,
        o_bias: bool = False,
    ):
        super().__init__()
        self.tq_config = tq_config
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        
        # 1. Standard Projections
        self.q_proj = nn.Linear(dim, dim, bias=q_bias)
        self.k_proj = nn.Linear(dim, (num_kv_heads * self.head_dim), bias=k_bias)
        self.v_proj = nn.Linear(dim, (num_kv_heads * self.head_dim), bias=v_bias)
        self.o_proj = nn.Linear(dim, dim, bias=o_bias)
        
        # 2. Strategy-based Quantization
        strategy = tq_config.get_strategy(layer_idx, total_layers)
        self.is_protected = (strategy == QuantizationStrategy.FP16)
        self.k_bits = tq_config.quant.k_bits if not self.is_protected else 16
        self.v_bits = tq_config.quant.v_bits if not self.is_protected else 16
        
        if not self.is_protected:
            # FIX: Bỏ tham số block_size đi. Các Quantizer giờ đã SOTA tự động nội suy (default 64)
            self.k_quantizer = TurboQuantProd(
                self.head_dim, 
                bits=self.k_bits, 
                n_rotation_passes=tq_config.quant.n_rotation_passes
            )
            self.k_quantizer.mse_quantizer.epsilon = tq_config.quant.quant_epsilon
            
            self.v_quantizer = TurboQuantValue(
                self.head_dim, 
                bits=self.v_bits,
                n_rotation_passes=tq_config.quant.n_rotation_passes
            )
            self.v_quantizer.mse_quantizer.epsilon = tq_config.quant.quant_epsilon
        else:
            self.k_quantizer = None
            self.v_quantizer = None
            
        self.rotary_emb = None # Injected by patcher

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[TurboQuantKVCache] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_q, _ = query.shape
        _, seq_k, _ = key.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. View/Transpose to (batch, heads, seq, dim) — HF STANDARD
        q = q.view(batch, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE (Standard HF sequence: AFTER Transpose)
        # SOTA: Passthrough Support (DecoderLayer passes pre-calculated cos/sin)
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
             # position_embeddings is a Tuple[cos, sin]
             cos, sin = position_embeddings
             q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None:
             # Legacy Fallback: Recalculate
             cos, sin = self.rotary_emb(v, seq_len=seq_k) # Fixed seq_len reference
             q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 4. SOTA: KV Cache Persistence (Mechanical Hijack Check)
        if kv_cache is not None:
            # === VÁ LỖI TỬ HUYỆT (Quantizer Sync) ===
            if (not self.is_protected) and (kv_cache.k_quantizer is None):
                kv_cache.k_quantizer = self.k_quantizer
                kv_cache.v_quantizer = self.v_quantizer

            # SOTA: Always use TurboQuant paged attention if cache is present
            kv_cache.append(k, v)
            
            # SOTA: The Compass Fix - Update container sequence length at Layer 0
            if self.layer_idx == 0:
                if hasattr(self, "_parent_model") and hasattr(self._parent_model, "_tq_cache_override"):
                    container = self._parent_model._tq_cache_override
                    container.update_seq_length(q.shape[2])
            
            if seq_q > 1:
                if not self.is_protected:
                    k_full = self.k_quantizer(k).to(q.dtype) if self.k_quantizer else k
                    v_full = self.v_quantizer(v).to(q.dtype) if self.v_quantizer else v
                    
                    # SOTA FIX: ĐỒNG BỘ DTYPE! Trả về đúng float16/bfloat16 của query
                    k_full = k_full.to(q.dtype)
                    v_full = v_full.to(q.dtype)
                else:
                    k_full, v_full = k, v
                
                if self.num_heads != self.num_kv_heads:
                    k_full = k_full.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v_full = v_full.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                
                is_causal = (seq_q > 1)
                mask_entry = None if is_causal else mask
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k_full, v_full, attn_mask=mask_entry, is_causal=is_causal
                )
                
                # CUDA Graph Safety: Pre-initialize centroids and _static_out
                # during the FIRST prefill pass so they're ready for decode.
                if not self.is_protected and kv_cache.k_centroids is None:
                    import math
                    kv_cache.k_centroids = compute_centroids(self.k_bits - 1, dist='gaussian').to(q.device, q.dtype)
                    kv_cache.v_centroids = compute_centroids(self.v_bits, dist='gaussian').to(q.device, q.dtype)
                    
                    # SOTA v8.7: Harden Lloyd-Max Codebooks for CUDA Graph safety
                    harden_lloyd_max(self.k_bits - 1, device=q.device, dtype=q.dtype, dist='gaussian')
                    harden_lloyd_max(self.v_bits, device=q.device, dtype=q.dtype, dist='gaussian')
                    
                    # Pre-allocate static output buffer for the Triton kernel
                    D_padded = kv_cache.pool.padded_head_dim
                    D = 2**math.ceil(math.log2(D_padded)) if D_padded > 0 else self.head_dim
                    kv_cache._static_out = torch.zeros(
                        (1, self.num_heads, 1, D), device=q.device, dtype=torch.float32
                    )

                # SOTA v8.8: Initialize Static FP16 Workspace for Protected Layers
                if self.is_protected and not hasattr(kv_cache, 'static_k_fp16'):
                    kv_cache.init_static_fp16_workspace(self.num_kv_heads, self.head_dim, kv_cache.pool.config.max_seq_len)
                    # Sync initial tokens into the workspace (batch=0, heads, seq, dim)
                    kv_cache.static_k_fp16[0, :, :seq_q] = k[0].to(torch.float16)
                    kv_cache.static_v_fp16[0, :, :seq_q] = v[0].to(torch.float16)
                    # Mark initial tokens as active in mask
                    kv_cache.static_attn_mask[0, 0, 0, :seq_q] = 0.0
            else:
                # DECODE BRANCH (seq_q == 1)
                if self.is_protected:
                    # SOTA v8.8: Graph-safe update for Protected Workspace
                    # 1. Update Workspace on GPU
                    # We use the fact that num_tokens_ptr is the EXACT index for the new token
                    idx = kv_cache.num_tokens_ptr.item() if not torch.cuda.is_current_stream_capturing() else 0 # Dummy for capture
                    
                    # During REAL capture/replay, we rely on the fact that indexing with a tensor 
                    # inside a graph must be handled carefully. 
                    # For K/V, we can use slice-based copy if we use a pre-calculated index.
                    # However, a cleaner way for CUDA Graphs is scatter_.
                    
                    # Update K/V: (1, n_heads, 1, head_dim) -> (1, n_heads, max_seq_len, head_dim)
                    # Note: k has shape (1, n_heads, 1, head_dim) after RoPE
                    k_f16 = k.to(torch.float16)
                    v_f16 = v.to(torch.float16)
                    
                    # SOTA: The GPU-Only Indexing
                    # scatter_ is fully graph-compatible. 
                    # We scatter across the 'seq_len' dimension (dim=2 for k/v, dim=3 for mask)
                    target_idx = kv_cache.num_tokens_ptr.view(1, 1, 1, 1).expand(1, self.num_kv_heads, 1, self.head_dim)
                    kv_cache.static_k_fp16.scatter_(2, target_idx, k_f16)
                    kv_cache.static_v_fp16.scatter_(2, target_idx, v_f16)
                    
                    # Update Mask: [1, 1, 1, max_seq_len]
                    mask_idx = kv_cache.num_tokens_ptr.view(1, 1, 1, 1)
                    kv_cache.static_attn_mask.scatter_(3, mask_idx, 0.0)
                
                # SOTA: Flash-Paged Attention CallKernel (Zero-Sync)
                sm_scale = self.tq_config.sm_scale if self.tq_config.sm_scale is not None else 1.0 / (self.head_dim ** 0.5)
                
                # CUDA Graph Safety: centroids MUST already be initialized from prefill
                if not self.is_protected:
                    assert kv_cache.k_centroids is not None, (
                        f"Layer {self.layer_idx}: k_centroids not initialized. "
                        "Run prefill before decode to pre-initialize."
                    )

                out = paged_turboquant_attention(
                    query=q, 
                    kv_cache=kv_cache, 
                    k_bits=self.k_bits,
                    v_bits=self.v_bits,
                    qjl_scale=self.tq_config.quant.qjl_scale,
                    sm_scale=sm_scale,
                    mask=mask,
                    k_centroids=kv_cache.k_centroids,
                    v_centroids=kv_cache.v_centroids,
                    quest_threshold=self.tq_config.quest_threshold
                )
        else:
            # 3. Fallback Path: No Cache 
            k_f, v_f = k, v
            if self.num_heads != self.num_kv_heads:
                k_f = k_f.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v_f = v_f.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k_f, v_f, attn_mask=mask, is_causal=(seq_q > 1)
            )
            
        # 4. Final Alignment and Output Projection
        if out.dim() == 4:
            out = out.transpose(1, 2).reshape(batch, seq_q, self.dim)
        else:
            out = out.reshape(batch, seq_q, self.dim)
            
        return self.o_proj(out), None