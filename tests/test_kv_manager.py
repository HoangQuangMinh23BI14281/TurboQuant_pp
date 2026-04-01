import torch
import pytest
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting, QuantizationStrategy
from turboquant.cache.manager import TurboQuantKVCache

def test_kv_manager_paged_allocation():
    """Verify paged memory allocation and Block-128 compliance."""
    num_layers = 4
    num_blocks = 10
    block_size = 16 # tokens per block
    head_dim = 128
    n_heads = 1
    
    # Pre-allocate pool
    pool = KVBlockPool(num_blocks, block_size, head_dim, n_heads)
    routing = LayerRouting.from_percent(num_layers, percent=0.25) # 25% exempt: 0 and 3
    
    # Layer 1 (Quantized middle layer)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool, routing=routing)
    assert cache.strategy == QuantizationStrategy.TURBO_4BIT
    
    # Append 16 tokens (exactly 1 block)
    for _ in range(16):
        k = torch.randn(1, n_heads, 1, head_dim, device="cuda")
        v = torch.randn(1, n_heads, 1, head_dim, device="cuda")
        cache.append(k, v)
    
    assert len(cache.block_ids) == 1
    # Check 16th token was stored at offset 15
    bid = cache.block_ids[0]
    assert torch.any(pool.k_indices[bid, 0, 15] != 0) 
    
    # 17th token should trigger second block allocation
    k = torch.randn(1, n_heads, 1, head_dim, device="cuda")
    v = torch.randn(1, n_heads, 1, head_dim, device="cuda")
    cache.append(k, v)
    
    assert len(cache.block_ids) == 2
    assert pool.usage == 0.2 # 2/10 blocks used

def test_kv_manager_boundary_protection():
    """Verify Boundary Protection (exempt layers) logic."""
    num_layers = 10
    pool = KVBlockPool(30, 8, 64, 1)
    routing = LayerRouting(num_layers, exempt_layers=[0, 9])
    
    # Layer 0: FP16 Exempt
    cache0 = TurboQuantKVCache(0, pool, routing)
    assert cache0.strategy == QuantizationStrategy.FP16
    assert cache0.k_quantizer is None
    
    # Layer 5: Turbo4Bit Production
    cache5 = TurboQuantKVCache(5, pool, routing)
    assert cache5.strategy == QuantizationStrategy.TURBO_4BIT
    assert cache5.k_quantizer is not None

def test_kv_manager_metadata_fidelity():
    """Verify that Block-128 metadata is correctly persisted in the pool."""
    head_dim = 256 # 2 groups of Block-128
    pool = KVBlockPool(10, 8, head_dim, 1)
    routing = LayerRouting(5, exempt_layers=[])
    
    cache = TurboQuantKVCache(2, pool, routing)
    
    k = torch.randn(1, 1, 1, head_dim, device="cuda")
    v = torch.randn(1, 1, 1, head_dim, device="cuda")
    
    cache.append(k, v)
    bid = cache.block_ids[0]
    
    # K-metadata: (residual_norm, norm)
    # Norms should be positive
    assert pool.k_metadata[bid, 0, 0, 1].item() > 0
    
    # V-metadata: (Scale, Zero) per group
    # Head_dim 256 / 128 = 2 groups
    assert pool.v_metadata.shape[-2] == 2 
    assert torch.any(pool.v_metadata[bid, 0, 0, :, 0] != 0) # Scales non-zero
