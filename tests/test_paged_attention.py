import torch
import math
import pytest
from src.turboquant.cache.block_pool import KVBlockPool
from src.turboquant.cache.manager import TurboQuantKVCache
from src.turboquant.kernels.fused_attention import turboquant_attention

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPagedAttention:
    """
    Verification Test for Paged Hybrid Fused Attention.
    Compares the paged memory path against the contiguous baseline.
    """
    
    @pytest.mark.parametrize("k_bits, v_bits", [(8, 3), (4, 4)])
    def test_paged_vs_contiguous_parity(self, k_bits, v_bits, seq_len=200):
        device = "cuda"
        head_dim = 128
        n_heads = 4
        # We use seq_len tokens (from parameter)
        torch.manual_seed(42)
        
        # 1. Global Pool Initialization
        pool = KVBlockPool(num_blocks=10, head_dim=head_dim, n_heads=n_heads, k_bits=k_bits, v_bits=v_bits)
        
        # 2. Per-layer Cache Manager
        kv_cache = TurboQuantKVCache(layer_idx=5, pool=pool, k_bits=k_bits, v_bits=v_bits)
        
        # 3. Fill Cache
        # We'll batch append to simulate a prefill or long decode
        k_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        v_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        
        for i in range(seq_len):
            kv_cache.append(k_full[:, :, i:i+1, :], v_full[:, :, i:i+1, :])
            
        expected_blocks = math.ceil(seq_len / 128)
        assert len(kv_cache.block_ids) == expected_blocks, f"Should occupy {expected_blocks} blocks for {seq_len} tokens"
        
        # 4. Run Paged Attention
        query = torch.randn((1, n_heads, 1, head_dim), device=device)
        paged_output = kv_cache.attention_score(query)
        
        # 5. Contiguous Baseline
        # We simulate the quantized state from the manager's quantizers
        # To strictly compare, we ensure the contiguous baseline uses 
        # the same input shapes as expected by the fused dispatcher.
        q_key_in = kv_cache.k_quantizer.quantize(k_full.view(n_heads, seq_len, head_dim))
        q_val_in = kv_cache.v_quantizer.quantize(v_full.view(n_heads, seq_len, head_dim))
        
        # Metadata
        sm_scale = 1.0 / math.sqrt(head_dim)
        qjl_scale = math.sqrt(math.pi / 2.0) / 128
        
        # Dispatch to contiguous kernel (using n_heads as batch)
        # Note: PyTorch reference handles its own qjl scaling internally.
        contig_output, _ = turboquant_attention(
            query.view(n_heads, 1, head_dim), 
            q_key_in, q_val_in, 
            quantizer=kv_cache.k_quantizer, 
            scale=sm_scale,
            k_bits=k_bits, v_bits=v_bits
        )
        # Reshape baseline back to (1, n_heads, 1, D)
        contig_output = contig_output.view(1, n_heads, 1, head_dim)
        
        # 6. Verification
        # Allow small epsilon due to possibly different summation order in Triton blocks
        # but outputs should be nearly identical.
        max_diff = torch.max(torch.abs(paged_output - contig_output)).item()
        
        print(f"Paged vs Contiguous Parity ({k_bits}/{v_bits}): Max Diff = {max_diff:.6f}")
        assert max_diff < 1e-4, f"Paged and Contiguous outputs differ significantly: {max_diff}"
        
        # Cleanup
        kv_cache.clear()
        assert pool.usage == 0.0

if __name__ == "__main__":
    # Manual execution for WSL debugging
    if torch.cuda.is_available():
        test = TestPagedAttention()
        test.test_paged_vs_contiguous_parity(8, 3)
        test.test_paged_vs_contiguous_parity(4, 4)
