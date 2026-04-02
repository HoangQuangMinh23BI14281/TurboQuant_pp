import torch
import pytest
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting

def test_quest_summary_consistency():
    """Verify that append() correctly updates Min/Max summaries."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_heads, head_dim = 4, 128
    pool = KVBlockPool(num_blocks=2, head_dim=head_dim, n_heads=n_heads, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    
    # 1. First token
    k1 = torch.ones(1, n_heads, 1, head_dim, device=device) * 5.0
    v1 = torch.ones(1, n_heads, 1, head_dim, device=device)
    cache.append(k1, v1)
    
    bid = cache.block_table[0]
    # min/max should both be 5.0
    print(f"DEBUG Summary Slot 0: Min={pool.k_summaries[bid, 0, 0, 0].item()}, Max={pool.k_summaries[bid, 0, 1, 0].item()}")
    assert torch.allclose(pool.k_summaries[bid, :, 0, :], torch.tensor(5.0, device=device))
    assert torch.allclose(pool.k_summaries[bid, :, 1, :], torch.tensor(5.0, device=device))
    
    # 2. Second token with different values
    k2 = torch.ones(1, n_heads, 1, head_dim, device=device) * -10.0
    v2 = torch.ones(1, n_heads, 1, head_dim, device=device)
    cache.append(k2, v2)
    
    # min should be -10, max should be 5
    print(f"DEBUG Summary Slot 1: Min={pool.k_summaries[bid, 0, 0, 0].item()}, Max={pool.k_summaries[bid, 0, 1, 0].item()}")
    assert torch.allclose(pool.k_summaries[bid, :, 0, :], torch.tensor(-10.0, device=device))
    assert torch.allclose(pool.k_summaries[bid, :, 1, :], torch.tensor(5.0, device=device))

def test_quest_skip_logic():
    """Verify that the Quest threshold successfully skips zero-importance blocks."""
    if not torch.cuda.is_available():
        pytest.skip("Quest skip logic requires CUDA/Triton")
        
    device = "cuda"
    n_heads, head_dim = 1, 128
    pool = KVBlockPool(num_blocks=4, head_dim=head_dim, n_heads=n_heads, device=device)
    cache = TurboQuantKVCache(layer_idx=1, pool=pool)
    
    # Block 1: Signal (Value 10)
    for _ in range(pool.tokens_per_block):
        cache.append(torch.ones(1, n_heads, 1, head_dim, device=device)*10.0, 
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
