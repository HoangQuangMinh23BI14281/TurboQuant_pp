import sys
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# SOTA: Basic Config
from turboquant.layers.config import TurboQuantConfig
from turboquant.integrations.patcher import patch_hf_model
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.manager import TurboQuantCacheContainer

def reset_turboquant_cache(model):
    if not hasattr(model, "_tq_cache_override"):
        return
    container = model._tq_cache_override
    container.pool.reset()
    container._current_seq_len = 0
    for layer in container.layers:
        # Reset Tensor-based state
        layer.num_tokens_ptr.zero_()
        layer.num_tokens = 0
        layer.block_ids = []  # Reset Python block tracking
        layer.k_fp16 = {}     # Reset FP16 cache for protected layers
        layer.v_fp16 = {}
        layer.block_table = layer.pool.allocate_layer_blocks(128) # Pre-fill for graph stability
        layer.k_centroids = None
        layer.v_centroids = None
        # Clear any pre-allocated static output
        if hasattr(layer, '_static_out'):
            delattr(layer, '_static_out')

@torch.no_grad()
def benchmark_perplexity(model, tokenizer, num_chunks=5, seq_len=512):
    print(f"\n📊 [PPL] Measuring Perplexity (WikiText-2, {num_chunks} chunks)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.squeeze(0).cuda()
    
    total_loss = 0.0
    count = 0
    
    for i in tqdm(range(0, min(num_chunks * seq_len, input_ids.size(0) - seq_len), seq_len)):
        chunk = input_ids[i : i + seq_len].unsqueeze(0)
        reset_turboquant_cache(model)
        outputs = model(chunk, use_cache=True)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = chunk[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += loss.item()
        count += 1
    
    avg_loss = total_loss / count
    ppl = torch.exp(torch.tensor(avg_loss))
    print(f"🎯 PPL: {ppl.item():.4f}")

@torch.no_grad()
def benchmark_speed(model, tokenizer, max_new_tokens=50):
    print("\n⚡ [SPEED] Measuring Decoding Throughput...")
    prompt = "The future of AI systems depends on efficient memory management."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    reset_turboquant_cache(model)
    out = model(**inputs, use_cache=True)
    curr_input = torch.argmax(out.logits[..., -1, :], dim=-1, keepdim=True)
    past_kv = out.past_key_values
    
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
    print(f"🚀 Speed: {tps:.2f} tokens/s ({generated_tokens} tokens in {t_total:.4f}s)")
    return tps

# =========================================================================
# CUDA Graph Wrapper — Zero Python Dispatch Decoding
# =========================================================================
class TurboGraphWrapper:
    """CUDA Graph Wrapper for LLM Decoding.
    
    Requirements for graph capture:
    1. All centroids and _static_out MUST be pre-initialized (done during prefill)
    2. All block_tables MUST be pre-allocated (done in reset_turboquant_cache)
    3. No host-device syncs (no comparing GPU tensors to Python values)
    4. No GPU memory allocations (no torch.zeros, .to(), etc.)
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.graph = torch.cuda.CUDAGraph()
        self.static_input = None
        self.static_past_kv = None
        self.static_logits = None
        self.is_captured = False

    def capture(self, input_ids, past_kv):
        print("📸 [GRAPH] Capturing Decoding Graph...")
        self.static_input = input_ids.clone()
        self.static_past_kv = past_kv
        
        # Phase 1: Warmup on a side stream.
        # Forces ALL lazy initializations to complete:
        # - Triton kernel autotuning/compilation
        # - RoPE cos/sin cache expansion  
        # After this, every subsequent forward call is pure GPU ops.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.model(self.static_input, past_key_values=self.static_past_kv, use_cache=True)
            self.static_logits = out.logits.clone()
        torch.cuda.current_stream().wait_stream(s)

        # Phase 2: Full sync before capture
        torch.cuda.synchronize()
        
        # Phase 3: Capture the graph
        with torch.cuda.graph(self.graph):
            out = self.model(self.static_input, past_key_values=self.static_past_kv, use_cache=True)
            self.static_logits.copy_(out.logits)
        
        self.is_captured = True
        print("✅ [GRAPH] Capture Complete.")

    def __call__(self, input_ids):
        if not self.is_captured:
            raise RuntimeError("Graph not captured!")
        
        # Zero-Copy Replay — only update input, replay recorded GPU ops
        self.static_input.copy_(input_ids)
        self.graph.replay()
        return self.static_logits

@torch.no_grad()
def benchmark_speed_graph(model, tokenizer, max_new_tokens=50):
    print("\n⚡ [SPEED] Measuring Decoding Throughput (CUDA GRAPHS)...")
    prompt = "The future of AI systems depends on efficient memory management."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 1. Prefill — triggers centroid + _static_out initialization via attention_layer
    reset_turboquant_cache(model)
    out = model(**inputs, use_cache=True)
    curr_input = torch.argmax(out.logits[..., -1, :], dim=-1, keepdim=True)
    past_kv = out.past_key_values
    
    # 2. Verify all layers are graph-ready
    container = model._tq_cache_override
    for i, layer_cache in enumerate(container.layers):
        if layer_cache.k_quantizer is not None:
            assert hasattr(layer_cache, '_static_out'), f"Layer {i}: _static_out not initialized after prefill!"
            assert layer_cache.k_centroids is not None, f"Layer {i}: k_centroids not initialized after prefill!"
    print(f"   ✓ All {len(container.layers)} layers verified graph-ready")
    
    # 2.5. Causal mask already patched globally in main()
    pass

    # 3. Capture Graph
    wrapper = TurboGraphWrapper(model)
    wrapper.capture(curr_input, past_kv)
    
    # 4. Replay Loop — pure GPU, zero Python dispatch
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    generated_tokens = 0
    for i in range(max_new_tokens):
        logits = wrapper(curr_input)
        curr_input = torch.argmax(logits[..., -1, :], dim=-1, keepdim=True)
        generated_tokens += 1
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    t_total = end_time - start_time
    tps = generated_tokens / t_total if t_total > 0 else 0
    print(f"🚀 [GRAPH] Speed: {tps:.2f} tokens/s ({generated_tokens} tokens in {t_total:.4f}s)")
    return tps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Reset CUDA stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager").to(device)
    
    tq_config = TurboQuantConfig(protect_boundaries=True) # SOTA Accuracy: Protect head/tail layers
    model = patch_hf_model(model, tq_config)
    
    # Cache Init (This is where TurboQuant takes VRAM for the pool)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    num_blocks = 4096 # Pre-allocated pool for multi-layer CUDA Graph stability
    pool = KVBlockPool(config=tq_config, head_dim=head_dim, n_heads=n_kv_heads, num_blocks=num_blocks, device=device, n_layers=n_layers)
    model._tq_cache_override = TurboQuantCacheContainer(num_layers=n_layers, pool=pool)
    
    # PPL Measurement — NO patches, uses real HF causal mask
    benchmark_perplexity(model, tokenizer)
    
    # Choose between Graph and Eager path
    use_graph = "--use_graph" in sys.argv
    if use_graph:
        print("\n🚀 [MODE] CUDA Graph Acceleration ENABLED")
        
        # Apply mask patch ONLY for graph capture (after PPL is measured honestly)
        import transformers.models.qwen2.modeling_qwen2 as qwen2_module
        def _noop_create_causal_mask(*args, **kwargs):
            return None  # Decode-only: Graph replays seq_len=1, mask is unnecessary
        qwen2_module.create_causal_mask = _noop_create_causal_mask
        
        benchmark_speed_graph(model, tokenizer)
    else:
        benchmark_speed(model, tokenizer)
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n📊 [VRAM] Peak Memory: {peak_vram:.2f} MB")
    print(f"📦 Pool Size: {num_blocks} blocks ({num_blocks * tq_config.hw.tokens_per_block} tokens / layer)")

if __name__ == "__main__":
    main()
