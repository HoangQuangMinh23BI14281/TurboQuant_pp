import torch
import math
from turboquant.quant.quantizer import TurboQuantProd, TurboQuantMSE, MSEQuantized
from turboquant.kernels.fused_attention import attention_score_prod
from turboquant.ops.wht import fwht
from turboquant.ops.sign_array import apply_sign_array

torch.manual_seed(42)
d, bits = 128, 4
n_q, n_k = 10, 100
q = TurboQuantProd(d, bits)
query = torch.randn(n_q, d)
key = torch.randn(n_k, d)
scale = 1.0 / math.sqrt(d)

# True scores
true_scores = (query @ key.T) * scale

# MSE-only scores (Term 1)
quantized = q.quantize(key)
mse_q = MSEQuantized(indices=quantized.mse_indices, norms=quantized.norms, bits=quantized.mse_bits)
keys_mse = q.mse_quantizer.dequantize(mse_q)
scores_mse = (query @ keys_mse.T) * scale

# Full prod scores
est_scores = attention_score_prod(query, quantized, q)

# Correlation
corr_mse = torch.corrcoef(torch.stack([true_scores.flatten(), scores_mse.flatten()]))[0, 1]
corr_prod = torch.corrcoef(torch.stack([true_scores.flatten(), est_scores.flatten()]))[0, 1]

print(f"MSE-only correlation: {corr_mse:.4f}")
print(f"Prod (MSE+QJL) correlation: {corr_prod:.4f}")
print(f"MSE scores range: [{scores_mse.min():.4f}, {scores_mse.max():.4f}]")
print(f"Prod scores range: [{est_scores.min():.4f}, {est_scores.max():.4f}]")
print(f"True scores range: [{true_scores.min():.4f}, {true_scores.max():.4f}]")
qjl_correction = est_scores - scores_mse
print(f"QJL correction range: [{qjl_correction.min():.4f}, {qjl_correction.max():.4f}]")
print(f"QJL correction abs mean: {qjl_correction.abs().mean():.6f}")
print(f"MSE scores abs mean: {scores_mse.abs().mean():.6f}")
print(f"QJL/MSE ratio: {qjl_correction.abs().mean() / scores_mse.abs().mean():.4f}")
print(f"rotation_scale: {q.mse_quantizer.rotation_scale}")
print(f"block_size: {q.block_size}")
print(f"n_passes: {q.mse_quantizer.n_rotation_passes}")
print(f"residual_norms mean: {quantized.residual_norms.mean():.6f}")
print(f"key_norms mean: {quantized.norms.mean():.6f}")
