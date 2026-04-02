import torch
import pytest
import math
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting, QuantizationStrategy
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.layers.attention_layer import TurboQuantAttention
from turboquant.layers.config import TurboQuantConfig

def test_fp16_paged_storage_and_gather():
    """Verify that FP16 path correctly uses raw buffers and gather logic."""
    num_blocks = 10
    tokens_per_block = 16
    head_dim = 64
    n_heads = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Signature: num_blocks, head_dim, n_heads, tokens_per_block
    pool = KVBlockPool(num_blocks=num_blocks, head_dim=head_dim, n_heads=n_heads, tokens_per_block=tokens_per_block, device=device)
    
    # Layer 0 is FP16
    routing = LayerRouting(num_layers=4, exempt_layers=[0])
    cache = TurboQuantKVCache(layer_idx=0, pool=pool, routing=routing)
    
    # Mock data (batch=1, seq_len=1, dimension=256)
    # Layer 0 is FP16
    q = torch.randn(1, 1, head_dim * n_heads, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, head_dim * n_heads, device=device, dtype=torch.float16)
    v = torch.randn(1, 1, head_dim * n_heads, device=device, dtype=torch.float16)
    
    # Simple projection simulate (since we are testing the cache storage, not the layer here)
    # But for cache.append we need the HEAD SPLIT version
    k_split = k.view(1, 1, n_heads, head_dim).transpose(1, 2)
    v_split = v.view(1, 1, n_heads, head_dim).transpose(1, 2)
    
    cache.append(k_split, v_split)
    
    # Check if stored in fp16 buffer
    bid = cache.block_table[0]
    # In pool.k_fp16 shape: (num_blocks, n_heads, tokens_per_block, head_dim)
    assert torch.allclose(pool.k_fp16[bid, :, 0], k_split.squeeze(0).squeeze(1))
    assert torch.allclose(pool.v_fp16[bid, :, 0], v_split.squeeze(0).squeeze(1))

def test_hybrid_precision_paged_attention():
    """
    Test routing between Layer 0 (FP16) and Layer 1 (Quantized).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim, n_heads = 256, 4
    head_dim = dim // n_heads # 64
    tokens_per_block = 16
    
    config = TurboQuantConfig(n_head_protected=1) # Layer 0 is FP16
    routing = LayerRouting(num_layers=4, exempt_layers=[0])
    pool = KVBlockPool(num_blocks=20, head_dim=head_dim, n_heads=n_heads, tokens_per_block=tokens_per_block, device=device)
    
    # Layer 0: FP16
    layer0 = TurboQuantAttention(config, layer_idx=0, total_layers=4, dim=dim, num_heads=n_heads, num_kv_heads=n_heads).to(device).half()
    cache0 = TurboQuantKVCache(layer_idx=0, pool=pool, routing=routing)
    
    # Layer 1: Quantized
    layer1 = TurboQuantAttention(config, layer_idx=1, total_layers=4, dim=dim, num_heads=n_heads, num_kv_heads=n_heads).to(device).half()
    cache1 = TurboQuantKVCache(layer_idx=1, pool=pool, routing=routing)
    
    # Mock Inputs (Batch, Seq, Dim)
    q = torch.randn(1, 1, dim, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, dim, device=device, dtype=torch.float16)
    v = torch.randn(1, 1, dim, device=device, dtype=torch.float16)
    
    # 1. Forward Layer 0 (Should success)
    out0, _ = layer0(q, k, v, kv_cache=cache0)
    assert cache0.strategy == QuantizationStrategy.FP16
    assert out0.shape == (1, 1, dim)
    
    # 2. Forward Layer 1 (Should success)
    out1, _ = layer1(q, k, v, kv_cache=cache1)
    assert cache1.strategy == QuantizationStrategy.TURBO_4BIT
    assert out1.shape == (1, 1, dim)
