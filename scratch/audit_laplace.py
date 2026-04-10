import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from turboquant.quant.lloyd_max import compute_lloyd_max_codebook

print("--- 4-bit Gaussian (Key) ---")
cb_k = compute_lloyd_max_codebook(4, dist='gaussian')
print("Centroids (16):", cb_k['centroids'])
print("Boundaries (15):", cb_k['boundaries'])
print("MC:", cb_k['max_centroid'])

print("\n--- 3-bit Laplace (Value) ---")
cb_v = compute_lloyd_max_codebook(3, dist='laplace')
print("Centroids (8):", cb_v['centroids'])
print("Boundaries (7):", cb_v['boundaries'])
print("MC:", cb_v['max_centroid'])
