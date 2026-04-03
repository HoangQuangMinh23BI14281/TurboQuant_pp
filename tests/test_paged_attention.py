import torch
import pytest
import math
from typing import Any
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting

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

        pool = KVBlockPool(num_blocks=20, head_dim=head_dim, n_heads=n_heads, k_bits=k_bits, v_bits=v_bits)
        kv_cache = TurboQuantKVCache(layer_idx=1, pool=pool, routing=LayerRouting(10, [0]))

        k_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        v_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        kv_cache.append(k_full, v_full)

        query = torch.randn((1, n_heads, 1, head_dim), device=device)
        
        # FIX 1: Tắt triệt để SOTA Quest Sparsity bằng -1e6 (thay vì -1.0)
        paged_output = kv_cache.attention_score(query, quest_threshold=-1e6)
        
        query_rot, query_sketch = kv_cache.k_quantizer.transform_query(query)

        # ==========================================
        # THE ULTIMATE PYTORCH BASELINE
        # Đọc trực tiếp từ Pool, không lượng tử hóa lại!
        # ==========================================
        bs = kv_cache.tokens_per_block
        k_recon_list = []
        v_recon_list = []
        k_qjl_list = []
        
        from turboquant.quant.lloyd_max import lloyd_max_dequantize
        from turboquant.quant.quant_base import unpack_indices

        for i, physical_id in enumerate(kv_cache.block_table):
            start = i * bs
            end = min(start + bs, seq_len)
            if start >= seq_len: break
            n_tokens = end - start
            
            # --- 1. KEY RECONSTRUCTION (Từ Pool) ---
            k_packed = pool.k_indices[physical_id] # (n_heads, 128, packed_bytes)
            k_unpacked = unpack_indices(k_packed, kv_cache.k_quantizer.bits - 1, head_dim) # (n_heads, 128, 128)
            
            k_deq_unit = lloyd_max_dequantize(k_unpacked, kv_cache.k_quantizer.bits - 1, dist='gaussian')
            k_norm_pool = pool.k_metadata[physical_id, :, 0].view(n_heads, 1, 1)
            k_scale_pool = pool.k_metadata[physical_id, :, 1].view(n_heads, 1, 1)
            
            k_deq_rot = k_deq_unit * k_scale_pool * k_norm_pool
            k_recon_list.append(k_deq_rot[:, :n_tokens, :])
            
            # --- 2. VALUE RECONSTRUCTION (Từ Pool) ---
            v_packed = pool.v_indices[physical_id]
            v_unpacked = unpack_indices(v_packed, kv_cache.v_quantizer.bits, head_dim)
            
            v_scale = pool.v_metadata[physical_id, :, :, 0].view(n_heads, 1, -1)
            v_zero = pool.v_metadata[physical_id, :, :, 1].view(n_heads, 1, -1)
            
            v_deq = v_unpacked.float() * v_scale + v_zero
            v_recon_list.append(v_deq[:, :n_tokens, :])

            # --- 3. QJL SIGNS RECONSTRUCTION (Từ Pool) ---
            signs_packed = pool.k_qjl[physical_id] 
            signs_unpacked = unpack_indices(signs_packed, 1, head_dim)
            signs_float = signs_unpacked.float() * 2.0 - 1.0
            k_qjl_list.append(signs_float[:, :n_tokens, :])

        k_recon = torch.cat(k_recon_list, dim=1)
        v_recon = torch.cat(v_recon_list, dim=1)
        k_qjl_total = torch.cat(k_qjl_list, dim=1)

        # --- TÍNH TOÁN SOFTMAX THEO CHUẨN TRITON ---
        q_recon = query_rot.reshape(n_heads, 1, head_dim)
        scores_mse = torch.matmul(q_recon, k_recon.transpose(1, 2))
        
        q_sk = query_sketch.reshape(n_heads, head_dim)
        qjl_dot = torch.sum(q_sk.reshape(n_heads, 1, head_dim) * k_qjl_total, dim=-1).reshape(1, n_heads, 1, -1)
        
        # Lấy Residual Norms cho toàn bộ sequence
        res_norms_list = []
        for physical_id in kv_cache.block_table:
            rn = pool.k_metadata[physical_id, :, 2].view(n_heads, 1).repeat_interleave(bs, dim=1)
            res_norms_list.append(rn)
        res_norms = torch.cat(res_norms_list, dim=1)[:, :seq_len]
        
        qjl_scale = kv_cache.k_quantizer.qjl_scale if kv_cache.k_quantizer else 1.0
        scores_qjl = qjl_dot * res_norms.reshape(1, n_heads, 1, -1) * qjl_scale
        
        sm_scale = 1.0 / math.sqrt(head_dim)
        attn_scores = (scores_mse + scores_qjl) * sm_scale
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        ref_output = torch.matmul(attn_weights, v_recon)
        contig_output = ref_output.view(1, n_heads, 1, head_dim)

        max_diff = torch.max(torch.abs(paged_output - contig_output)).item()
        print(f"DEBUG_PARITY_SOTA: Norm {torch.norm(contig_output).item()} | Diff {max_diff}")
        
        assert max_diff < 1e-4, f"Paged and Contiguous differ significantly: {max_diff}"
