import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from turboquant.layers.attention_layer import TurboQuantAttention
from turboquant.layers.config import TurboQuantConfig
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting

def debug():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Layer 0 is FP16, Layer 1 is Quantized
    config = TurboQuantConfig(n_head_protected=1) 
    
    dim, n_heads, max_seq_len = 256, 4, 128
    head_dim = dim // n_heads # 64
    tokens_per_block = 16
    n_blocks = max_seq_len // tokens_per_block
    
    print(f"--- DEBUG HYBRID PRECISION ---")
    print(f"Device: {device}, Head Dim: {head_dim}")
    
    # Correct Signature: num_blocks, head_dim, n_heads, tokens_per_block
    pool = KVBlockPool(
        num_blocks=n_blocks, 
        head_dim=head_dim, 
        n_heads=n_heads, 
        tokens_per_block=tokens_per_block, 
        device=device
    )
    
    routing = LayerRouting(num_layers=4, exempt_layers=[0])
    
    # Init Layer 1 (Quantized branch)
    layer = TurboQuantAttention(config, layer_idx=1, total_layers=4, dim=dim, num_heads=n_heads, num_kv_heads=n_heads).to(device).half()
    
    # Init Cache for Layer 1
    cache = TurboQuantKVCache(layer_idx=1, pool=pool, routing=routing)
    
    # Inputs (Batch, Seq, Dim)
    q = torch.randn(1, 1, dim, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, dim, device=device, dtype=torch.float16)
    v = torch.randn(1, 1, dim, device=device, dtype=torch.float16)
    
    print(f"Input Q shape: {q.shape}")
    print(f"Layer Strategy: {cache.strategy}")
    
    try:
        print("Running Forward Pass (Quantized Layer 1)...")
        out, _ = layer(q, k, v, kv_cache=cache)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    debug()
