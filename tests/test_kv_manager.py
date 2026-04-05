import torch
import pytest
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.layers.config import TurboQuantConfig

def test_kv_manager_paged_allocation():
    num_layers = 4
    num_blocks = 10
    tokens_per_block = 16 
    head_dim = 128
    n_heads = 8
    
    # Signature: config, head_dim, n_heads, num_blocks
    config = TurboQuantConfig(tokens_per_block=tokens_per_block)
    pool = KVBlockPool(
        config=config,
        head_dim=head_dim, 
        n_heads=n_heads, 
        num_blocks=num_blocks
    )
    
    # Init Cache for Layer 1
    cache = TurboQuantKVCache(
        layer_idx=1,
        pool=pool
    )
    # SOTA: Must match pool bits (default v=3)
    cache.k_quantizer = TurboQuantProd(head_dim, bits=pool.k_bits)
    cache.v_quantizer = TurboQuantValue(head_dim, bits=pool.v_bits)
    
    # 1. Fill exactly one block
    for i in range(tokens_per_block):
        k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
        v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
        cache.append(k, v)
        
    assert len(cache.block_ids) == 1
    assert cache.num_tokens == tokens_per_block
    
    # 2. Trigger new block allocation
    k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    cache.append(k, v)
    
    assert len(cache.block_ids) == 2
    assert cache.num_tokens == tokens_per_block + 1

def test_kv_manager_metadata_fidelity():
    num_blocks = 10
    tokens_per_block = 16
    n_heads = 8
    head_dim = 256 # 2 groups of Block-128
    
    config = TurboQuantConfig(tokens_per_block=tokens_per_block)
    pool = KVBlockPool(
        config=config,
        head_dim=head_dim, 
        n_heads=n_heads, 
        num_blocks=num_blocks
    )
    
    cache = TurboQuantKVCache(
        layer_idx=1,
        pool=pool
    )
    cache.k_quantizer = TurboQuantProd(head_dim, bits=pool.k_bits)
    cache.v_quantizer = TurboQuantValue(head_dim, bits=pool.v_bits)
    
    k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    cache.append(k, v)
    
    bid = cache.block_ids[0]
    
    # Check shape of K indices inside the pool (SOTA: Padded to 128, Dynamic Packing)
    bits = cache.k_quantizer.bits
    vals_per_byte = 8 // (bits - 1)
    expected_dim = max(128, head_dim) // vals_per_byte
    assert pool.k_indices.shape[-1] == expected_dim
    
    # Check K metadata (norm, scale, residual_norm)
    # k_metadata shape: (n_layers, blocks, heads, tokens, 3) 
    assert pool.k_metadata.shape[-1] == 3
    # Ensure norms are not zero after append
    assert torch.any(pool.k_metadata[1, bid, 0, 0, 0] != 0) 
    
    # Check V metadata (Scale and Zero per block per token per group)
    expected_groups = max(1, head_dim // 32)
    # v_metadata shape: (n_layers, blocks, heads, tokens_per_block, groups, 2)
    assert pool.v_metadata.shape[-2] == expected_groups
    assert torch.any(pool.v_metadata[1, bid, 0, 0, 0, 0] != 0)

def test_kv_manager_boundary_protection():
    """
    Ensure the manager respects the 'Exempt' routing (no quantization).
    """
    from turboquant.cache.routing import LayerRouting, QuantizationStrategy
    
    num_blocks = 10
    tokens_per_block = 16
    head_dim = 128
    n_heads = 8
    
    config = TurboQuantConfig(tokens_per_block=tokens_per_block)
    pool = KVBlockPool(config, head_dim, n_heads, num_blocks)
    
    cache = TurboQuantKVCache(layer_idx=0, pool=pool)
    cache.is_protected = True # Emulate routing
    
    # Append should go to k_fp16 local dictionary, not global pool k_indices
    k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    cache.append(k, v)
    
    bid = cache.block_ids[0]
    # Verify in LOCAL manager storage, not pool
    assert torch.any(cache.k_fp16[bid][0, 0] != 0)
    assert torch.all(pool.k_indices[0, bid, 0, 0, 0] == 0)
