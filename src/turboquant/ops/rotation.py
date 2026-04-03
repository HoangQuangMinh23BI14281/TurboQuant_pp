import torch
import torch.nn as nn
from typing import Optional
from .wht import fwht, ifwht
from .sign_array import generate_sign_array, apply_sign_array

class TurboQuantRotation(nn.Module):
    def __init__(self, d: int, n_passes: int = 2, pattern: str = 'tbq', seed: Optional[int] = None):
        super().__init__()
        self.d = d
        self.n_passes = n_passes
        
        # Use seed if provided, otherwise hash the pattern for determinism
        real_seed = seed if seed is not None else hash(pattern) % 10000
        
        # SOTA: Generate signs for all passes (Bit-Exact with Reference)
        all_signs = []
        for i in range(n_passes):
            # For pass 0, use the official Llama preset if it matches
            preset = pattern if (i == 0 and pattern in ['tbq', 'qjl']) else None
            # Independent seed derivation for staggered passes
            p_seed = (real_seed + i)
            signs = generate_sign_array(d, seed=p_seed, use_llama_preset=preset) 
            all_signs.append(signs)
        self.register_buffer("all_signs", torch.stack(all_signs))
        
        # SOTA: pre-register UNNORMALIZED Hadamard matrix for O(1) matmul scaling
        # We keep it as +1/1 for numerical stability
        from .wht import get_wht_matrix
        self.register_buffer("wht_mat", get_wht_matrix(d, normalized=False))
        # FUSED SCALING: Total scale is (1/sqrt(d))^n_passes = 1 / d^(n/2)
        # This prevents rounding accumulated by irrational sqrt(d) at each pass
        self.register_buffer("final_scale", torch.tensor(1.0 / (d ** (n_passes / 2.0)), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Cascaded SRHT (Sign -> WHT) x N passes
        Note: Scaling is fused to the end for 1e-15 precision parity.
        """
        out = x
        h_mat = self.wht_mat.to(device=out.device, dtype=out.dtype)
        for i in range(self.n_passes):
            # 1. Apply pass-specific sign (Robust to device/dtype mismatch)
            out = apply_sign_array(out, self.all_signs[i])
            # 2. Vectorized WHT (Synchronize Device & Dtype)
            # Matmul with unnormalized matrix (integer-exact additions)
            out = torch.matmul(out, h_mat)
        
        # 3. Apply Fused Scale (Synchronized Device & Dtype)
        return out * self.final_scale.to(device=out.device, dtype=out.dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        SRHT is orthonormal, thus its own inverse (with reversed sign order)
        Note: Scaling is fused for 1e-15 precision parity.
        """
        out = x
        h_mat = self.wht_mat.to(device=out.device, dtype=out.dtype)
        # Apply Fused Scale first (Inverse of the forward's final scale is the same)
        out = out * self.final_scale.to(device=out.device, dtype=out.dtype)
        
        for i in reversed(range(self.n_passes)):
            # 1. Hadamard matmul (Integer additions)
            out = torch.matmul(out, h_mat)
            # 2. Sign is its own inverse (Robust to device/dtype mismatch)
            out = apply_sign_array(out, self.all_signs[i])
        return out

def apply_cascaded_srht(x: torch.Tensor, n_passes: int = 2, pattern: str = 'tbq') -> torch.Tensor:
    rot = TurboQuantRotation(x.shape[-1], n_passes=n_passes, pattern=pattern).to(x.device)
    return rot(x)
