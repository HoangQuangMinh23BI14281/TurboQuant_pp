import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Type, Tuple
import logging

from turboquant.layers.attention_layer import TurboQuantAttention
from turboquant.layers.config import TurboQuantConfig
from turboquant.cache.manager import TurboQuantKVCache

logger = logging.getLogger(__name__)

def patch_hf_model(model: nn.Module, tq_config: Optional[TurboQuantConfig] = None):
    """
    Surgically replace HuggingFace Attention layers with TurboQuantAttention.
    Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM.
    """
    if tq_config is None:
        tq_config = TurboQuantConfig() # Default 4-bit SOTA
        
    # 1. Identify Attention Modules
    # We look for common HF Attention class names
    target_classes = ("LlamaAttention", "Qwen2Attention", "MistralAttention")
    
    # Track replaced layers
    replaced_count = 0
    total_layers = 0
    
    # Calculate total layers first (for strategy mapping)
    layer_map = {}
    for name, module in model.named_modules():
        if any(cls in str(type(module)) for cls in target_classes):
             total_layers += 1
             layer_map[total_layers - 1] = module

    logger.info(f"Detected {total_layers} attention layers for patching.")

    # 2. Sequential Replacement
    current_idx = 0
    for name, module in model.named_modules():
        if any(cls in str(type(module)) for cls in target_classes):
            # SOTA Pillar 4: Boundary Layer Protection
            # Skip first and last 2 layers (High importance for vocabulary/decisions)
            if current_idx < 2 or current_idx >= total_layers - 2:
                logger.info(f"Skipping Boundary Layer {current_idx} ({name}) - Running in Native FP16")
                current_idx += 1
                continue
            
            # Create our SOTA layer
            dim = getattr(module, "hidden_size", model.config.hidden_size)
            num_heads = getattr(module, "num_heads", model.config.num_attention_heads)
            num_kv_heads = getattr(module, "num_key_value_heads", getattr(model.config, "num_key_value_heads", num_heads))

            new_layer = TurboQuantAttention(
                tq_config=tq_config,
                layer_idx=current_idx,
                total_layers=total_layers,
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            ).to(module.q_proj.weight.device).to(module.q_proj.weight.dtype)
            
            # SOTA: Inject HF Model Config for ROPE and other logic compatibility
            new_layer.config = model.config
            
            # Transfer ROPE
            if hasattr(module, "rotary_emb"):
                new_layer.rotary_emb = module.rotary_emb
            
            # Transfer Weights
            new_layer.q_proj.weight.data.copy_(module.q_proj.weight.data)
            new_layer.k_proj.weight.data.copy_(module.k_proj.weight.data)
            new_layer.v_proj.weight.data.copy_(module.v_proj.weight.data)
            new_layer.o_proj.weight.data.copy_(module.o_proj.weight.data)
            
            # Replace in Parent
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, attr_name, new_layer)
            
            # Inject HF-compatible Forward (The Bridge)
            # This allows the layer to work with model.forward()
            _wrap_hf_forward(new_layer)
            
            current_idx += 1
            replaced_count += 1

    logger.info(f"Successfully patched {replaced_count} layers into {type(model).__name__}.")
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # 1. Convert past_key_value to TurboQuantKVCache if necessary
        # Note: vLLM-style serving will maintain an external cache manager.
        # Here we bridge to existing DynamicCache or our own.
        tq_cache = None
        if isinstance(past_key_value, TurboQuantKVCache):
            tq_cache = past_key_value
        
        # 2. Run SOTA Kernel
        attn_output, _ = original_forward(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            kv_cache=tq_cache,
            mask=attention_mask,
            position_ids=position_ids
        )
        
        # 3. Handle HF Return Format (Output, AttnProb)
        # Modern HF (v4.36+) expects 2 values if using Cache object
        return attn_output, None

    layer.forward = hf_forward
