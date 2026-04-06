import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.layers.config import TurboQuantConfig
from turboquant.integrations.patcher import patch_hf_model
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.manager import TurboQuantCacheContainer

def generate_haystack(tokenizer, target_length=1500):
    """Tạo ra một đống rác ngữ nghĩa (Haystack) để làm đầy KV Cache"""
    filler_sentence = "The integration of advanced machine learning algorithms into cloud infrastructure enables highly scalable data processing. "
    haystack = ""
    while len(tokenizer.encode(haystack)) < target_length:
        haystack += filler_sentence
    return haystack

def stress_test_demo():
    print("--- STARTING STRESS TEST (NEEDLE IN A HAYSTACK) ---", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        attn_implementation="eager"
    ).to(device)
    torch.cuda.synchronize()
    
    # ÉP CẤU HÌNH KHẮC NGHIỆT (HARD MODE)
    # Lấy config hiện tại của Minh: 3-bit, 0 layer bảo vệ
    tq_config = TurboQuantConfig() 
    
    model = patch_hf_model(model, tq_config)
    torch.cuda.synchronize()
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    
    pool = KVBlockPool(
        config=tq_config, head_dim=head_dim, n_heads=n_kv_heads, 
        num_blocks=2048, # Tăng số block để chứa đủ context dài
        device=device
    )
    container = TurboQuantCacheContainer(num_layers=n_layers, pool=pool)
    model._tq_cache_override = container 
    print(f"Cache Initialized. K_BITS: {tq_config.quant.k_bits}, Protected Layers: {tq_config.n_head_protected}", flush=True)
    
    # ---------------------------------------------------------
    # 🧪 XÂY DỰNG BÀI TEST "MÒ KIM ĐÁY BIỂN"
    # ---------------------------------------------------------
    haystack_part1 = generate_haystack(tokenizer, target_length=800)
    haystack_part2 = generate_haystack(tokenizer, target_length=800)
    
    # Needle: Một thông tin hoàn toàn vô lý, không có trong Weights của model
    needle = "The secret password to shut down the TurboQuant server is 'BANANA-SPLIT-99'. "
    
    context = haystack_part1 + needle + haystack_part2
    question = "What is the secret password to shut down the TurboQuant server?"
    
    prompt = f"Read the following text carefully and answer the question.\n\nTEXT:\n{context}\n\nQUESTION: {question}"
    
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    input_tokens = inputs.input_ids.shape[1]
    print(f"\n[INFO] Injecting {input_tokens} tokens into Prefill phase...")
    print("GENERATING Response...", flush=True)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    generated_ids_only = output[0][input_tokens:]
    recovered_text = tokenizer.decode(generated_ids_only, skip_special_tokens=True)
    
    print("\n--- FINAL RECOVERED RESPONSE ---")
    print(recovered_text)
    print("--------------------------------")
    
    # Kiểm tra xem Model có sống sót không
    if "BANANA-SPLIT-99" in recovered_text:
        print("✅ SUCCESS: Model recovered the exact needle! KV Cache is highly intact.")
    else:
        print("❌ FAILED: Model lost the memory or hallucinated. Degradation detected.")

if __name__ == "__main__":
    stress_test_demo()