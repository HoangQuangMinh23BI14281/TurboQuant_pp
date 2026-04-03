import torch
import math

def generate_hadamard(d: int, normalized: bool = True) -> torch.Tensor:
    """
    Generate a Hadamard matrix of size d x d using iterative construction.
    d must be a power of 2.
    """
    if (d & (d - 1)) != 0 or d <= 0:
        raise ValueError("d must be a power of 2")

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float64)
    
    current_d = 2
    while current_d < d:
        H = torch.cat((
            torch.cat((H, H), dim=1),
            torch.cat((H, -H), dim=1)
        ), dim=0)
        current_d *= 2
        
    if normalized:
        # Orthonormal normalization: 1/sqrt(d)
        # We ensure the division happens in float64 to prevent precision loss.
        H = H.to(torch.float64) / math.sqrt(d)
        
    return H

def get_wht_matrix(d: int, normalized: bool = True) -> torch.Tensor:
    """
    Returns a Hadamard matrix for performance (via matmul).
    Arg `normalized`: False returns raw +1/-1 matrix for precision stability.
    """
    return generate_hadamard(d, normalized=normalized)

def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT).
    DEPRECATED: Use get_wht_matrix and torch.matmul for performance.
    """
    d = x.shape[-1]
    h_mat = get_wht_matrix(d).to(x.device).to(x.dtype)
    return torch.matmul(x, h_mat)

def ifwht(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Fast Walsh-Hadamard Transform.
    Since FWHT is orthonormal, it is its own inverse.
    """
    return fwht(x)
