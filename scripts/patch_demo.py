import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.layers.config import TurboQuantConfig
from turboquant.integrations.patcher import patch_hf_model
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.manager import TurboQuantCacheContainer

def atomic_demo():
    print("--- STARTING ATOMIC DEMO ---", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 1. Load Model with Eager Enforcement
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        attn_implementation="eager"
    ).to(device)
    torch.cuda.synchronize()
    
    # SOTA: Fidelity Verification Mode (Force FP16 for all 24 layers)
    # This isolates the "Compass" (Position ID) sync from quantization noise.
    # SOTA: Hardened Calibration Configuration
    # Uses stable 4-bit MSE (k=5) / 8-bit Val (v=8) defaults for absolute recovery.
    from turboquant.layers.config import QuantConfig, HardwareConfig
    tq_config = TurboQuantConfig(
        quant=QuantConfig(
            k_bits=5,
            v_bits=8,
            k_group_size=64, # Moved from top-level group_size
            qjl_scale=0.1,
        ),
        hw=HardwareConfig(
            num_blocks=1024,
            tokens_per_block=128,
        ),
        protect_boundaries=True,
        n_head_protected=2, 
        n_tail_protected=2,
        quest_threshold=-1e6
    )
    
    model = patch_hf_model(model, tq_config)
    torch.cuda.synchronize()
    
    # 2. Setup SOTA Paged Cache Container
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    
    pool = KVBlockPool(
        config=tq_config,
        head_dim=head_dim, 
        n_heads=n_kv_heads, 
        num_blocks=1024,
        device=device
    )
    container = TurboQuantCacheContainer(num_layers=n_layers, pool=pool)
    
    # SOTA: Ghim chặt Container vào Model để các Layer tự mò lấy
    model._tq_cache_override = container 
    print("TurboQuant Cache Container Initialized.", flush=True)
    
    # 3. Atomic Generation
    prompt = "Explain quantum physics to a 5-year-old in one sentence."
    
    # SOTA: Standard Qwen2.5 Chat Template
    messages = [
        {"role": "user", "content": prompt}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    print(f"\nPROMPT: {prompt}", flush=True)
    print("GENERATING Helpfully...", flush=True)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    input_len = inputs.input_ids.shape[1]
    generated_ids_only = output[0][input_len:]
    
    print("\n--- TOKEN-BY-TOKEN RECOVERY ---")
    recovered_text = ""
    for tid in generated_ids_only:
        token_str = tokenizer.decode([tid])
        print(f"ID {tid:6} | '{token_str}'")
        recovered_text += token_str
    
    print(f"\nFINAL RECOVERED RESPONSE:\n{recovered_text}")
    print("\n--- ATOMIC DEMO FINISHED ---", flush=True)

if __name__ == "__main__":
    atomic_demo()
