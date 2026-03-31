import torch
import torch.nn as nn
from .wht import fwht, ifwht
from .sign_array import generate_sign_array, apply_sign_array

class TurboQuantRotation(nn.Module):
    """
    Unified Rotation module for TurboQuant++ (SRHT: Subsampled Randomized Hadamard Transform).
    Fuses Sign-flip (S) and Walsh-Hadamard Transform (H).
    
    This matches Giai đoạn 2: Random Rotation in the architecture document.
    """
    def __init__(self, d: int, pattern: str = 'tbq'):
        """
        Args:
            d: dimension of vectors (must be power of 2)
            pattern: 'tbq' (default) or 'qjl' llama.cpp bit-patterns
        """
        super().__init__()
        self.d = d
        self.pattern = pattern
        # Pre-generate signs as a buffer (matching llama.cpp patterns)
        signs = generate_sign_array(d, use_llama_preset=pattern)
        self.register_buffer('signs', signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply S * H rotation to input x.
        x shape: (..., d)
        
        Formula: x_rot = H(x * S)
        """
        # 1. Apply fixed signs (S)
        x_signed = apply_sign_array(x, self.signs)
        # 2. Apply FWHT (H)
        return fwht(x_signed)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse rotation (H_inv * S_inv).
        Since H_inv = H/d and S_inv = S:
        Formula: x_orig = H_inv(x_rot) * S
        """
        # 1. Apply IFWHT
        x_iwht = ifwht(x)
        # 2. Apply signs (S is self-inverse)
        return apply_sign_array(x_iwht, self.signs)

def apply_srht(x: torch.Tensor, pattern: str = 'tbq') -> torch.Tensor:
    """
    Functional interface for SRHT rotation.
    """
    d = x.shape[-1]
    signs = generate_sign_array(d, use_llama_preset=pattern).to(x.device)
    return fwht(apply_sign_array(x, signs))
