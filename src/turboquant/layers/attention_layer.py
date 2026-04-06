import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from turboquant.layers.config import TurboQuantConfig
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.kernels.fused_attention import turboquant_attention
from turboquant.kernels.fused_attention import paged_turboquant_attention
from turboquant.cache.routing import QuantizationStrategy
from turboquant.ops.rope import RotaryPositionalEmbeddings
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

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
            self.k_quantizer = TurboQuantProd(
                self.head_dim, 
                bits=self.k_bits, 
                n_rotation_passes=tq_config.quant.n_rotation_passes
            )
            self.k_quantizer.mse_quantizer.epsilon = tq_config.quant.quant_epsilon
            self.v_quantizer = TurboQuantValue(
                self.head_dim, 
                bits=self.v_bits,
                group_size=tq_config.quant.v_group_size
            )
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
             cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
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
                # 2.1 Prefill Path: Hardware-accelerated SDPA
                k_full, v_full = k, v
                if self.num_heads != self.num_kv_heads:
                    k_full = k_full.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v_full = v_full.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                
                is_causal = (seq_q > 1)
                mask_entry = None if is_causal else mask
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k_full, v_full, attn_mask=mask_entry, is_causal=is_causal
                )
            else:
                # 2.2 Decode Path: Paged Triton Kernel
                sm_scale = self.tq_config.sm_scale if self.tq_config.sm_scale is not None else 1.0 / (self.head_dim ** 0.5)
                out = paged_turboquant_attention(
                    query=q, 
                    kv_cache=kv_cache, 
                    k_bits=self.k_bits,
                    v_bits=self.v_bits,
                    qjl_scale=self.tq_config.quant.qjl_scale,
                    sm_scale=sm_scale,
                    mask=mask,
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
