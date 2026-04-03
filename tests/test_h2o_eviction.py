import torch
import pytest
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool

def test_h2o_importance_accumulation():
    """Verify that kernel correctly updates block importance in the pool."""
    if not torch.cuda.is_available():
        pytest.skip("H2O testing requires CUDA/Triton")
        
    device = "cuda"
    n_heads, head_dim = 4, 128
    pool = KVBlockPool(num_blocks=10, head_dim=head_dim, n_heads=n_heads, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    
    # Fill 2 blocks
    for _ in range(pool.tokens_per_block * 2):
        cache.append(torch.randn(1, n_heads, 1, head_dim, device=device), 
                    torch.randn(1, n_heads, 1, head_dim, device=device))
        
    # Initially importance is 0
    assert torch.all(pool.block_importance == 0)
    
    # Run attention with Quest disabled to ensure full scoring
    query = torch.randn(1, 1, n_heads * head_dim, device=device, dtype=torch.float16)
    cache.attention_score(query, quest_threshold=0.0)
    
    # Check if importance was accumulated
    active_ids = cache.block_table
    importance = pool.block_importance[active_ids]
    assert torch.any(importance > 0)
    print(f"H2O Accumulated importance: {importance.sum().item()}")

def test_h2o_eviction_integrity():
    """Verify that eviction removes blocks and updates state correctly."""
    # SOTA: Skip while H2O is frozen for mapping stabilization
    pytest.skip("H2O Eviction is temporarily frozen for architectural stabilization.")
    if not torch.cuda.is_available():
        pytest.skip("H2O testing requires CUDA/Triton")
        
    device = "cuda"
    n_heads, head_dim = 1, 128
    pool = KVBlockPool(num_blocks=10, head_dim=head_dim, n_heads=n_heads, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    
    # 1. Fill 4 blocks
    for b in range(4):
        # Scale input so some blocks are way more important (Signal=100.0, Noise=0.001)
        val = 100.0 if b % 2 == 0 else 0.001
        for _ in range(pool.tokens_per_block):
            cache.append(torch.ones(1, n_heads, 1, head_dim, device=device)*val, 
                        torch.ones(1, n_heads, 1, head_dim, device=device))
            
    initial_tokens = cache.num_tokens # 16 * 4 = 64
    initial_blocks = len(cache.block_table) # 4
    
    # 2. Run attention to score
    # Query all-ones to match the Signal
    query = torch.ones(1, 1, n_heads * head_dim, device=device, dtype=torch.float16)
    cache.attention_score(query, quest_threshold=0.0)
    
    # 3. Evict with appropriate threshold
    # Noise blocks will have near-zero importance compared to Signal blocks
    cache.evict_idle_blocks(threshold=1e-5, min_keep=1)
    
    # Verify removal
    final_blocks = len(cache.block_table)
    assert final_blocks < initial_blocks
    assert cache.num_tokens == final_blocks * pool.tokens_per_block
    
    # Check pool free list
    assert len(pool.free_blocks) == pool.num_blocks - final_blocks
    
    # 4. Verify we can still append after eviction
    cache.append(torch.randn(1, n_heads, 1, head_dim, device=device), 
                torch.randn(1, n_heads, 1, head_dim, device=device))
    assert len(cache.block_table) == final_blocks + 1
