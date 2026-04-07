import torch
import pytest
import math
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.layers.config import TurboQuantConfig, HardwareConfig

def test_kv_manager_paged_allocation():
    num_layers = 4
    num_blocks = 10
    tokens_per_block = 16 
    head_dim = 128
    n_heads = 8
    
    config = TurboQuantConfig(hw=HardwareConfig(tokens_per_block=tokens_per_block))
    pool = KVBlockPool(
        config=config, head_dim=head_dim, n_heads=n_heads, num_blocks=num_blocks
    )
    
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    # SOTA FIX: Phải truyền block_size để Quantizer và Pool đồng thuận
    cache.k_quantizer = TurboQuantProd(head_dim, bits=pool.k_bits, block_size=pool.k_group_size)
    cache.v_quantizer = TurboQuantValue(head_dim, bits=pool.v_bits, block_size=pool.v_group_size)
    
    for i in range(tokens_per_block):
        k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
        v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
        cache.append(k, v)
        
    assert len(cache.block_ids) == 1
    assert cache.num_tokens == tokens_per_block
    
    k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    cache.append(k, v)
    
    assert len(cache.block_ids) == 2
    assert cache.num_tokens == tokens_per_block + 1

def test_kv_manager_metadata_fidelity():
    num_blocks = 10
    tokens_per_block = 16
    n_heads = 8
    head_dim = 256 
    
    config = TurboQuantConfig(hw=HardwareConfig(tokens_per_block=tokens_per_block))
    pool = KVBlockPool(
        config=config, head_dim=head_dim, n_heads=n_heads, num_blocks=num_blocks
    )
    
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    # SOTA FIX: Truyền block_size
    cache.k_quantizer = TurboQuantProd(head_dim, bits=pool.k_bits, block_size=pool.k_group_size)
    cache.v_quantizer = TurboQuantValue(head_dim, bits=pool.v_bits, block_size=pool.v_group_size)
    
    k = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    v = torch.randn(1, n_heads, 1, head_dim, device=pool.device)
    cache.append(k, v)
    
    bid = cache.block_ids[0]
    
    n_subblocks = math.ceil(head_dim / pool.k_group_size)
    bits = cache.k_quantizer.bits
    vals_per_byte = 8 // (bits - 1)
    expected_dim = (pool.k_group_size * n_subblocks) // vals_per_byte
    
    assert pool.k_indices.shape[-1] == expected_dim
    assert pool.k_metadata.shape[-1] == 3 * n_subblocks
    assert torch.any(pool.k_metadata[1, bid, 0, 0, 0] != 0) 
    assert pool.v_metadata.shape[-1] == 2 * pool.v_subblocks
    assert torch.any(pool.v_metadata[1, bid, 0, 0, 0] != 0)

def test_kv_manager_boundary_protection():
    """
    Ensure the manager respects the 'Exempt' routing (no quantization).
    """
    num_blocks = 10
    tokens_per_block = 16
    head_dim = 128
    n_heads = 8
    
    config = TurboQuantConfig(hw=HardwareConfig(tokens_per_block=tokens_per_block))
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
