import torch
import pytest
import math
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting
from turboquant.kernels.fused_attention import turboquant_attention

class TestPagedAttention:
    """
    SOTA Hardening: Verify Bit-Exact Parity between Paged and Contiguous paths.
    """
    @pytest.mark.parametrize("k_bits, v_bits", [(8, 3), (4, 4)])
    def test_paged_vs_contiguous_parity(self, k_bits, v_bits, seq_len=200):
        device = "cuda"
        head_dim = 128
        n_heads = 4
        torch.manual_seed(42)

        # 1. Global Pool Initialization
        # SOTA: Increase num_blocks to 20 to avoid exhaustion for 200 tokens (needs 13 blocks)
        pool = KVBlockPool(num_blocks=20, head_dim=head_dim, n_heads=n_heads, k_bits=k_bits, v_bits=v_bits)

        # 2. Per-layer Cache Manager
        # SOTA: Mark Layer 1 as quantized to ensure k_quantizer is initialized
        kv_cache = TurboQuantKVCache(layer_idx=1, pool=pool, routing=LayerRouting(10, [0]))

        # 3. Fill Cache
        k_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        v_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)

        for i in range(seq_len):
            kv_cache.append(k_full[:, :, i:i+1, :], v_full[:, :, i:i+1, :])

        expected_blocks = math.ceil(seq_len / kv_cache.tokens_per_block)
        assert len(kv_cache.block_ids) == expected_blocks

        # 4. Run Paged Attention
        query = torch.randn((1, n_heads, 1, head_dim), device=device)
        paged_output = kv_cache.attention_score(query)

        # 5. Contiguous Baseline
        q_key_in = kv_cache.k_quantizer.quantize(k_full.view(n_heads, seq_len, head_dim))
        q_val_in = kv_cache.v_quantizer.quantize(v_full.view(n_heads, seq_len, head_dim))

        sm_scale = 1.0 / math.sqrt(head_dim)
        mask = torch.ones((n_heads, 1, seq_len), device=device, dtype=torch.bool)
        
        contig_output, _ = turboquant_attention(
            query.view(n_heads, 1, head_dim),
            q_key_in, q_val_in,
            quantizer=kv_cache.k_quantizer,
            sm_scale=sm_scale,
            causal_mask=mask,
            k_bits=k_bits, v_bits=v_bits
        )
        contig_output = contig_output.view(1, n_heads, 1, head_dim)

        # 6. EXTREME TELEMETRY DEBUGGING
        max_diff = torch.max(torch.abs(paged_output - contig_output)).item()
        paged_norm = torch.norm(paged_output).item()
        contig_norm = torch.norm(contig_output).item()

        print(f"\n[PARITY DEBUG] Bits({k_bits}/{v_bits}):")
        print(f"  > Paged Norm   : {paged_norm:.8f}")
        print(f"  > Contig Norm  : {contig_norm:.8f}")
        print(f"  > Max Absolute Diff: {max_diff:.8f}")
        
        print(f"  > Paged (First 5): {paged_output[0,0,0,:5].tolist()}")
        print(f"  > Contig (First 5): {contig_output[0,0,0,:5].tolist()}")
        
        # SOTA: Bit-exact parity expected < 1e-4
        assert max_diff < 1e-4, f"Paged and Contiguous differ significantly: {max_diff}"
