import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def benchmark_perplexity(model, tokenizer, num_chunks=5, seq_len=512):
    print(f"\n📊 [BASELINE PPL] Measuring Perplexity (WikiText-2, {num_chunks} chunks)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.squeeze(0).cuda()
    
    total_loss = 0.0
    count = 0
    
    for i in tqdm(range(0, min(num_chunks * seq_len, input_ids.size(0) - seq_len), seq_len)):
        chunk = input_ids[i : i + seq_len].unsqueeze(0)
        # Baseline uses standard HuggingFace forward (no cache reset needed as no cache is persistent)
        outputs = model(chunk, use_cache=False) 
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = chunk[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += loss.item()
        count += 1
    
    avg_loss = total_loss / count
    ppl = torch.exp(torch.tensor(avg_loss))
    print(f"🎯 BASELINE PPL: {ppl.item():.4f}")
    return ppl.item()

@torch.no_grad()
def benchmark_speed(model, tokenizer, max_new_tokens=50):
    print("\n⚡ [BASELINE SPEED] Measuring Decoding Throughput...")
    prompt = "The future of AI systems depends on efficient memory management."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 1. Prefill
    out = model(**inputs, use_cache=True)
    curr_input = torch.argmax(out.logits[..., -1, :], dim=-1, keepdim=True)
    past_kv = out.past_key_values
    
    # 2. Decode Loop
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    generated_tokens = 0
    for i in range(max_new_tokens):
        out = model(curr_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        curr_input = torch.argmax(out.logits[..., -1, :], dim=-1, keepdim=True)
        generated_tokens += 1
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    t_total = end_time - start_time
    tps = generated_tokens / t_total if t_total > 0 else 0
    print(f"🚀 BASELINE Speed: {tps:.2f} tokens/s ({generated_tokens} tokens in {t_total:.4f}s)")
    return tps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading BASELINE model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Reset CUDA stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        attn_implementation="eager"
    ).to(device)
    
    benchmark_perplexity(model, tokenizer)
    benchmark_speed(model, tokenizer)
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n📊 [VRAM] Peak Memory: {peak_vram:.2f} MB")

if __name__ == "__main__":
    main()
