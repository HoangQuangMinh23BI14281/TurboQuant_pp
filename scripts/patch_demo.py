import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.integrations.patcher import patch_hf_model
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting

def demo_patch_and_generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" # Small & Fast for demo
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    
    # 1. Apply Surgical Patcher (Phase 1)
    print("\n--- Applying SOTA Surgical Patcher ---")
    model = patch_hf_model(model)
    
    # 2. Setup SOTA Paged Cache (needed for patched forward)
    print("\n--- Initializing TurboQuant Paged Cache Pools ---")
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    
    pool = KVBlockPool(num_blocks=1024, head_dim=head_dim, n_heads=n_kv_heads, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    cache.n_heads = n_heads
    cache.n_kv_heads = n_kv_heads
    cache.group_size = n_heads // n_kv_heads
    
    # 3. Test Generation
    prompt = "Explain quantum physics to a 5-year-old in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    # We pass 'past_key_values' = cache to trigger our patched attention logic
    with torch.no_grad():
        # NOTE: patching hf_forward means we can call model.forward or model.generate
        # though DynamicCache/Paged Cache orchestration for .generate() needs a slight wrap
        output = model.generate(
            **inputs, 
            max_new_tokens=20, 
            use_cache=True, 
            past_key_values=cache # Our custom cache object
        )
        
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    demo_patch_and_generate()
