import torch
import pytest
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting
from turboquant.layers.config import TurboQuantConfig

def test_quest_summary_consistency():
    """Verify that append() correctly updates Min/Max summaries."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_heads, head_dim = 4, 128
    config = TurboQuantConfig()
    pool = KVBlockPool(config, head_dim=head_dim, n_heads=n_heads, num_blocks=2, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    from turboquant.quant.key_quantizer import TurboQuantProd
    cache.k_quantizer = TurboQuantProd(head_dim, bits=pool.k_bits, n_rotation_passes=2)
    
    # 1. First token
    k1 = torch.ones(1, n_heads, 1, head_dim, device=device) * 5.0
    v1 = torch.ones(1, n_heads, 1, head_dim, device=device)
    cache.append(k1, v1)
    
    bid = cache.block_ids[0]
    # Check if updated (not zero)
    assert torch.any(pool.k_summaries[1, bid, :, 0, :] != 0)
    assert torch.allclose(pool.k_summaries[1, bid, :, 0, :], pool.k_summaries[1, bid, :, 1, :])
    
    val1 = pool.k_summaries[1, bid, 0, 0, 0].item()
    
    # 2. Second token with different values
    k2 = torch.ones(1, n_heads, 1, head_dim, device=device) * -10.0
    v2 = torch.ones(1, n_heads, 1, head_dim, device=device)
    cache.append(k2, v2)
    
    # min should be updated, max should stay same (since val2 < val1)
    val2 = pool.k_summaries[1, bid, 0, 0, 0].item()
    # verify that min either stays same or decreases, and max stays same or increases
    assert pool.k_summaries[1, bid, 0, 0, 0] <= val1
    assert torch.any(pool.k_summaries[1, bid, :, 1, :] >= val1)

def test_quest_skip_logic():
    """Verify that the Quest threshold successfully skips zero-importance blocks."""
    pytest.skip("Simulated Attention Score is deprecated due to Paged Fused Triton kernel migration.")
    if not torch.cuda.is_available():
        pytest.skip("Quest skip logic requires CUDA/Triton")
        
    device = "cuda"
    n_heads, head_dim = 1, 128
    config = TurboQuantConfig()
    pool = KVBlockPool(config, head_dim=head_dim, n_heads=n_heads, num_blocks=4, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    
    # Block 1: Signal (Value 0.1)
    for _ in range(pool.tokens_per_block):
        cache.append(torch.ones(1, n_heads, 1, head_dim, device=device)*0.1, 
                    torch.ones(1, n_heads, 1, head_dim, device=device))
        
    # Block 2: Silence (Value 0.0000001)
    for _ in range(pool.tokens_per_block):
        cache.append(torch.ones(1, n_heads, 1, head_dim, device=device)*1e-7, 
                    torch.ones(1, n_heads, 1, head_dim, device=device))
        
    query = torch.ones(1, 1, n_heads * head_dim, device=device, dtype=torch.float16)
    
    # 1. Run with threshold=0 (No skip)
    out_all = cache.attention_score(query, quest_threshold=0.0)
    
    # 2. Run with threshold=1.0 (Should skip the silence block)
    # Block 1 bound: sum(1 * 10) = 1280. 
    # Block 2 bound: sum(1 * 1e-7) = 1.28e-5.
    out_sparse = cache.attention_score(query, quest_threshold=1e-1)
    
    # Output should be very close because Block 2 contribution is negligible
    # but the denominator (l_i) will be slightly different if Block 2 is skipped.
    diff = torch.abs(out_all - out_sparse).max()
    print(f"Quest Sparsity Diff: {diff.item()}")
    
    # Error should be small
    assert diff < 1e-3
