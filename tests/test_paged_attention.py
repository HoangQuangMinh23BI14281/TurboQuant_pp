import torch
import pytest
import math
from typing import Any
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.layers.config import TurboQuantConfig, QuantConfig, HardwareConfig
from turboquant.quant.key_quantizer import TurboQuantProd
from turboquant.quant.value_quantizer import TurboQuantValue
from turboquant.ops.rotation import TurboQuantRotation

class TestPagedAttention:
    @pytest.mark.parametrize("k_bits, v_bits", [(8, 3), (4, 4)])
    def test_paged_vs_contiguous_parity(self, k_bits, v_bits, seq_len=200):
        device = "cuda"
        head_dim = 128
        n_heads = 4
        tokens_per_block = 16
        torch.manual_seed(42)

        config = TurboQuantConfig(
            hw=HardwareConfig(tokens_per_block=tokens_per_block, triton_block_n=tokens_per_block),
            quant=QuantConfig(k_bits=k_bits, v_bits=v_bits)
        )
        pool = KVBlockPool(config, head_dim=head_dim, n_heads=n_heads, num_blocks=20)
        kv_cache = TurboQuantKVCache(layer_idx=1, pool=pool)
        
        # SOTA FIX: Truyền block_size
        kv_cache.k_quantizer = TurboQuantProd(dim=head_dim, bits=k_bits, n_rotation_passes=1, block_size=pool.k_group_size).to(device)
        kv_cache.v_quantizer = TurboQuantValue(dim=head_dim, bits=v_bits, n_rotation_passes=1, block_size=pool.v_group_size).to(device)

        k_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        v_full = torch.randn((1, n_heads, seq_len, head_dim), device=device)
        kv_cache.append(k_full, v_full)

        query = torch.randn((1, n_heads, 1, head_dim), device=device)
        
        from turboquant.kernels.fused_attention import paged_turboquant_attention
        paged_output = paged_turboquant_attention(
            query=query, 
            kv_cache=kv_cache, 
            k_bits=k_bits, 
            v_bits=v_bits, 
            qjl_scale=kv_cache.k_quantizer.qjl_scale,
            sm_scale=1.0 / math.sqrt(head_dim),
            quest_threshold=-1e6
        )
        
        bs = kv_cache.pool.tokens_per_block
        k_recon_list = []
        v_recon_list = []
        k_qjl_list = []
        
        from turboquant.quant.lloyd_max import lloyd_max_dequantize
        from turboquant.quant.quant_base import unpack_indices
        from turboquant.kernels.paged_fused import compute_centroids

        rot_k = TurboQuantRotation(pool.k_group_size, n_passes=1).to(device)
        rot_v = TurboQuantRotation(pool.v_group_size, n_passes=1).to(device)
        k_centroids = compute_centroids(k_bits - 1, dist='gaussian').to(device)
        v_centroids = compute_centroids(v_bits, dist='gaussian').to(device)

        for i, physical_id in enumerate(kv_cache.block_ids):
            start = i * bs
            end = min(start + bs, seq_len)
            if start >= seq_len: break
            n_tokens = end - start
            li = kv_cache.layer_idx
            
            k_packed = pool.k_indices[li, physical_id, :, :n_tokens] 
            n_subblocks = math.ceil(head_dim / pool.k_group_size)
            padded_dim = pool.k_group_size * n_subblocks
            
            k_packed_contig = k_packed.clone().detach().contiguous()
            k_indices = unpack_indices(k_packed_contig, k_bits - 1, padded_dim)[..., :head_dim]
            
            k_norm = pool.k_metadata[li, physical_id, :, :n_tokens, 0].unsqueeze(-1).contiguous()
            k_scale = pool.k_metadata[li, physical_id, :, :n_tokens, 1].unsqueeze(-1).contiguous()
            
            k_recon_unit = k_centroids[k_indices.long()]
            k_recon_rot = k_recon_unit * k_scale * k_norm
            k_recon_list.append(k_recon_rot)
            
            v_packed = pool.v_indices[li, physical_id, :, :n_tokens]
            v_packed_contig = v_packed.clone().detach().contiguous()
            v_indices = unpack_indices(v_packed_contig, v_bits, padded_dim)[..., :head_dim]
            
            v_norm = pool.v_metadata[li, physical_id, :, :n_tokens, 0].unsqueeze(-1).contiguous()
            v_scale = pool.v_metadata[li, physical_id, :, :n_tokens, 1].unsqueeze(-1).contiguous()
            
            v_recon_unit = v_centroids[v_indices.long()]
            v_rotated = v_recon_unit * v_scale
            
            v_rot_blocks = v_rotated.view(*v_rotated.shape[:-1], n_subblocks, pool.v_group_size)
            v_unrot_blocks = rot_v.inverse(v_rot_blocks)
            v_unrot = v_unrot_blocks.view(v_rotated.shape)
            
            v_recon_list.append(v_unrot * v_norm)

            signs_packed = pool.k_qjl[li, physical_id, :, :n_tokens] 
            signs_packed_contig = signs_packed.clone().detach().contiguous()
            signs_unpacked = unpack_indices(signs_packed_contig, 1, padded_dim)[..., :head_dim]
            k_qjl_list.append(signs_unpacked.float() * 2.0 - 1.0)

        k_recon = torch.cat(k_recon_list, dim=1)
        v_recon = torch.cat(v_recon_list, dim=1)
        k_qjl_total = torch.cat(k_qjl_list, dim=1)

        q_rot, q_sketch = kv_cache.k_quantizer.transform_query(query)
        q_rot = q_rot.view(n_heads, 1, head_dim)
        scores_mse = torch.matmul(q_rot, k_recon.transpose(1, 2))
        
        q_sk = q_sketch.view(n_heads, 1, head_dim)
        qjl_dot = (q_sk * k_qjl_total).sum(dim=-1, keepdim=True).transpose(1, 2)
        
        res_norms_list = []
        for physical_id in kv_cache.block_ids:
            res_norms_list.append(pool.k_metadata[li, physical_id, :, :bs, 2])
        res_norms = torch.cat(res_norms_list, dim=1)[:, :seq_len].unsqueeze(1)
        
        scores_qjl = qjl_dot * res_norms * kv_cache.k_quantizer.qjl_scale
        attn_scores = (scores_mse + scores_qjl) * (1.0 / math.sqrt(head_dim))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        ref_output = torch.matmul(attn_weights, v_recon)
        contig_output = ref_output.transpose(0, 1).view(1, n_heads, 1, head_dim)

        max_diff = torch.max(torch.abs(paged_output - contig_output)).item()
        assert max_diff < 1e-2, f"Paged and Contiguous differ significantly: {max_diff}"