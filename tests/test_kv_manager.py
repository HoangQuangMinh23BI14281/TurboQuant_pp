import torch
import pytest
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue

def test_kv_manager_paged_allocation():
    num_layers = 4
    num_blocks = 10
    tokens_per_block = 16 
    head_dim = 128
    n_heads = 8
    
    # Signature: num_blocks, head_dim, n_heads, tokens_per_block
    pool = KVBlockPool(
        num_blocks=num_blocks, 
        head_dim=head_dim, 
        n_heads=n_heads, 
        tokens_per_block=tokens_per_block
    )
    
    # Init Cache for Layer 1
    cache = TurboQuantKVCache(
        layer_idx=1,
        pool=pool,
        k_quantizer=TurboQuantProd(head_dim),
        v_quantizer=TurboQuantValue(head_dim)
    )
    
    # 1. Fill exactly one block
    for i in range(tokens_per_block):
        k = torch.randn(1, n_heads, 1, head_dim)
        v = torch.randn(1, n_heads, 1, head_dim)
        cache.append(k, v)
        
    assert len(cache.block_table) == 1
    assert cache.num_tokens == tokens_per_block
    
    # 2. Trigger new block allocation
    k = torch.randn(1, n_heads, 1, head_dim)
    v = torch.randn(1, n_heads, 1, head_dim)
    cache.append(k, v)
    
    assert len(cache.block_table) == 2
    assert cache.num_tokens == tokens_per_block + 1

def test_kv_manager_metadata_fidelity():
    num_blocks = 10
    tokens_per_block = 16
    n_heads = 8
    head_dim = 256 # 2 groups of Block-128
    
    pool = KVBlockPool(
        num_blocks=num_blocks, 
        head_dim=head_dim, 
        n_heads=n_heads, 
        tokens_per_block=tokens_per_block
    )
    
    cache = TurboQuantKVCache(
        layer_idx=1,
        pool=pool,
        k_quantizer=TurboQuantProd(head_dim),
        v_quantizer=TurboQuantValue(head_dim)
    )
    
    k = torch.randn(1, n_heads, 1, head_dim)
    v = torch.randn(1, n_heads, 1, head_dim)
    cache.append(k, v)
    
    bid = cache.block_table[0]
    
    # Check shape of K indices inside the pool (SOTA: Padded to 128, Dynamic Packing)
    bits = cache.k_quantizer.bits
    vals_per_byte = 8 // (bits - 1)
    expected_dim = max(128, head_dim) // vals_per_byte
    assert pool.k_indices.shape[-1] == expected_dim
    
    # Check K metadata (norm, scale, residual_norm)
    # k_metadata shape: (blocks, heads, tokens_per_block, 3)
    assert pool.k_metadata.shape[-1] == 3
    # Ensure norms are not zero after append
    assert torch.any(pool.k_metadata[bid, 0, 0, 0] != 0) 
    
    # Check V metadata (Scale and Zero per token per group)
    expected_groups = max(1, head_dim // 128)
    # v_metadata shape: (blocks, heads, tokens_per_block, groups, 2)
    assert pool.v_metadata.shape[-2] == expected_groups
    assert torch.any(pool.v_metadata[bid, 0, 0, 0, 0] != 0)

def test_kv_manager_boundary_protection():
    """
    Ensure the manager respects the 'Exempt' routing (no quantization).
    """
    from turboquant.cache.routing import LayerRouting, QuantizationStrategy
    
    num_blocks = 10
    tokens_per_block = 16
    head_dim = 128
    n_heads = 8
    
    pool = KVBlockPool(num_blocks, head_dim, n_heads, tokens_per_block)
    # Layer 0 is exempt (FP16)
    routing = LayerRouting(num_layers=4, exempt_layers=[0])
    
    cache = TurboQuantKVCache(layer_idx=0, pool=pool, routing=routing)
    assert cache.strategy == QuantizationStrategy.FP16
    
    # Append should go to k_fp16 pool, not k_indices
    k = torch.randn(1, n_heads, 1, head_dim)
    v = torch.randn(1, n_heads, 1, head_dim)
    cache.append(k, v)
    
    bid = cache.block_table[0]
    assert torch.any(pool.k_fp16[bid, 0, 0] != 0)
    assert torch.all(pool.k_indices[bid, 0, 0] == 0)
