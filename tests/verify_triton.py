import torch
import math
import triton
from turboquant.quant.quantizer import TurboQuantProd
from turboquant.kernels.fused_attention import attention_score_prod
import torch.nn.functional as F
from turboquant.layers.config import TurboQuantConfig, QuantConfig

def verify_scores():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Triton verification.")
        return

    device = "cuda"
    d, bits = 128, 4
    n_q, n_k = 1, 128
    torch.manual_seed(42)

    q = TurboQuantProd(d, bits).to(device)
    query = torch.randn(n_q, d, device=device)
    key = torch.randn(n_k, d, device=device)

    # 1. Quantize (packed)
    quantized_key = q.quantize(key, pack=True)

    config = TurboQuantConfig(quant=QuantConfig(k_bits=8, v_bits=3))

    # 2. PyTorch Reference Score (unpacked)
    # We'll temporarily force HAS_TRITON = False for the reference check
    import turboquant.kernels.fused_attention as fa
    original_has_triton = fa.HAS_TRITON
    
    fa.HAS_TRITON = False
    quantized_key_unpacked = q.quantize(key, pack=False)
    ref_scores = attention_score_prod(query, quantized_key_unpacked, q)

    # 3. Triton Score (integrated path)
    fa.HAS_TRITON = original_has_triton
    triton_scores = attention_score_prod(query, quantized_key, q)

    # Compare
    diff = torch.abs(ref_scores.flatten() - triton_scores.flatten())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Ref range: {ref_scores.min().item():.4f} to {ref_scores.max().item():.4f}")
    print(f"Triton range: {triton_scores.min().item():.4f} to {triton_scores.max().item():.4f}")
    
    if max_diff < 1e-4:
        print("SUCCESS: Triton scores match PyTorch reference!")
    else:
        print("FAILURE: Triton scores deviate from PyTorch reference.")

if __name__ == "__main__":
    verify_scores()
