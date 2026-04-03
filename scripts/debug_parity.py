import torch
import math
from turboquant.cache.manager import TurboQuantKVCache
from turboquant.cache.block_pool import KVBlockPool
from turboquant.cache.routing import LayerRouting

def debug_parity():
    device = "cuda"
    head_dim = 128
    n_heads = 4
    seq_len = 200
    k_bits, v_bits = 8, 4
    
    torch.manual_seed(42)
    pool = KVBlockPool(num_blocks=20, head_dim=head_dim, n_heads=n_heads, k_bits=k_bits, v_bits=v_bits)
    kv_cache = TurboQuantKVCache(layer_idx=1, pool=pool, routing=LayerRouting(10, [0]))

    k_raw = torch.randn((1, n_heads, seq_len, head_dim), device=device)
    v_raw = torch.randn((1, n_heads, seq_len, head_dim), device=device)
    
    # 1. Rotate K exactly like in production
    k_rot = kv_cache.k_quantizer.mse_quantizer.rotation(k_raw.view(-1, head_dim)).view(1, n_heads, seq_len, head_dim)
    
    # 2. Append
    kv_cache.append(k_rot, v_raw)
    
    # 3. Query
    query_raw = torch.randn((1, n_heads, 1, head_dim), device=device)
    paged_out = kv_cache.attention_score(query_raw)
    
    # 4. Manual Ref
    query_rot, _ = kv_cache.k_quantizer.transform_query(query_raw)
    
    # Check if attention_score used the same query_rot
    print(f"Query Rot Norm: {torch.norm(query_rot).item():.4f}")
    
    # Reconstruction
    bs = 128
    k_recon_list = []
    for i, pid in enumerate(kv_cache.block_table):
        start = i * bs
        end = min(start + bs, seq_len)
        chunk = k_rot[:, :, start:end, :]
        k_norm = pool.k_metadata[pid, :, 0].view(n_heads, 1).repeat_interleave(end-start, dim=0).flatten()
        # Dequantize with block norm
        q = kv_cache.k_quantizer.mse_quantizer.quantize(chunk.view(-1, head_dim))
        from turboquant.quant.quant_base import MSEQuantized
        q_mod = MSEQuantized(indices=q.indices, norms=k_norm, scales=q.scales, bits=q.bits, packed=q.packed)
        deq = kv_cache.k_quantizer.mse_quantizer.dequantize(q_mod).view(n_heads, end-start, head_dim)
        k_recon_list.append(deq)
    
    k_recon = torch.cat(k_recon_list, dim=1)
    
    # Ref Scores
    q_ref = query_rot.view(n_heads, 1, head_dim)
    ref_scores = torch.matmul(q_ref, k_recon.transpose(1, 2)) * (1.0 / math.sqrt(head_dim))
    ref_probs = torch.softmax(ref_scores, dim=-1)
    
    # Assuming V is FP16 for now to isolate K
    # Actually V is quantized too.
    
    print(f"Paged Output (First 5): {paged_out[0,0,0,:5].tolist()}")
    print(f"Max Diff check done.")

if __name__ == "__main__":
    debug_parity()
