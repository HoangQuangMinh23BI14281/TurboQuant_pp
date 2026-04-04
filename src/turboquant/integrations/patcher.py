import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from transformers.cache_utils import Cache

from turboquant.layers.config import TurboQuantConfig
from turboquant.layers.attention_layer import TurboQuantAttention

def patch_hf_model(model: nn.Module, tq_config: TurboQuantConfig) -> nn.Module:
    """
    Surgical Monkey-Patching for HuggingFace Transformers.
    Replaces standard Attention layers with TurboQuant++ Paged Attention.
    """
    # 1. Identify Attention Layers to Patch
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        print("WARNING: Could not find model.layers. Skipping patching.")
        return model
        
    total_layers = len(model.model.layers)
    print(f"DEBUG: Starting patch on {total_layers} layers...")
    replaced_count = 0
    current_idx = 0
    
    for layer in model.model.layers:
        if hasattr(layer, "self_attn"):
            print(f"DEBUG: Patching Layer {current_idx}...")
            old_attn = layer.self_attn
            
            # Extract standard architectural params
            dim = model.config.hidden_size
            num_heads = model.config.num_attention_heads
            num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
            
            # SOTA: Detect Biases & Shapes
            q_shape = old_attn.q_proj.weight.shape
            k_shape = old_attn.k_proj.weight.shape
            v_shape = old_attn.v_proj.weight.shape
            q_bias = old_attn.q_proj.bias is not None
            k_bias = old_attn.k_proj.bias is not None
            v_bias = old_attn.v_proj.bias is not None
            o_bias = old_attn.o_proj.bias is not None
            
            print(f"DEBUG: Layer {current_idx} | Q: {q_shape} | K: {k_shape} | V: {v_shape}", flush=True)
            
            # Create our TurboQuant Attention Layer
            new_layer = TurboQuantAttention(
                tq_config=tq_config,
                layer_idx=current_idx,
                total_layers=total_layers,
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                q_bias=q_bias,
                k_bias=k_bias,
                v_bias=v_bias,
                o_bias=o_bias
            ).to(old_attn.q_proj.weight.device).to(old_attn.q_proj.weight.dtype)
            
            # Transfer Weights from Old to New
            new_layer.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
            new_layer.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
            new_layer.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
            new_layer.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)
            
            # Transfer Biases if they exist
            if q_bias: new_layer.q_proj.bias.data.copy_(old_attn.q_proj.bias.data)
            if k_bias: new_layer.k_proj.bias.data.copy_(old_attn.k_proj.bias.data)
            if v_bias: new_layer.v_proj.bias.data.copy_(old_attn.v_proj.bias.data)
            if o_bias: new_layer.o_proj.bias.data.copy_(old_attn.o_proj.bias.data)
            
            # Inject RoPE
            if hasattr(old_attn, "rotary_emb"):
                new_layer.rotary_emb = old_attn.rotary_emb
            
            # Swap Modules
            layer.self_attn = new_layer
            
            # SOTA: Bind parent model reference for cache hijack recovery
            new_layer._parent_model = model
            
            # Inject HF-compatible Forward (The Bridge)
            _wrap_hf_forward(new_layer)
            
            current_idx += 1
            replaced_count += 1

    print(f"Successfully patched {replaced_count} layers into {type(model).__name__}.")
    return model

def _wrap_hf_forward(layer: TurboQuantAttention):
    """
    Monkey-patches our layer with a forward method that understands 
    HuggingFace's Attention input signature.
    """
    original_forward = layer.forward
    
    def hf_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        
        # 1. Bốc đúng túi Cache XỊN của chúng ta (đã được ghim sẵn vào model)
        container = getattr(layer._parent_model, "_tq_cache_override", None)
        actual_kv_cache = container.layers[layer.layer_idx] if container else None

        # 2. Chạy toán học SOTA
        try:
            # Note: actual_kv_cache handles all logic internally
            attn_output, attn_weights = original_forward(
                query=hidden_states, 
                key=hidden_states, 
                value=hidden_states, 
                kv_cache=actual_kv_cache, 
                mask=attention_mask, 
                position_ids=position_ids,
                **kwargs
            )
        except Exception as e:
            print(f"!!! CRITICAL ERROR in Layer {layer.layer_idx} !!!")
            import traceback
            traceback.print_exc()
            raise e
            
        # 3. Nuôi "Ghost Cache" của Hugging Face
        # Nếu HF truyền DynamicCache vào, ta CHỈ TĂNG BỘ ĐẾM để nó không bị lạc đường.
        # Tuyệt đối không append tensor vào đây để tiết kiệm 100% VRAM!
        if past_key_value is not None:
            # HF DynamicCache uses _seen_tokens (pre 4.40) or get_seq_length() 
            # We strictly sync _seen_tokens only at Layer 0 (the leader)
            if hasattr(past_key_value, "_seen_tokens") and layer.layer_idx == 0:
                past_key_value._seen_tokens += hidden_states.shape[1]
                
        # Trả lại ĐÚNG số lượng tham số mà môi trường này mong đợi (output, weights) 
        # Note: Cache được cập nhật IN-PLACE trong manager, nhưng HF vẫn mong nhận 2 biến này
        return attn_output, attn_weights

    layer.forward = hf_forward
