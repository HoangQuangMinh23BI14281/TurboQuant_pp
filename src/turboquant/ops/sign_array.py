import torch
from typing import Optional

def generate_sign_array(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate a 1D sign array S containing random elements from {-1, 1}.
    Represented as a 1D vector for broadcasting in element-wise multiplication.
    Args:
        d (int): Dimension size.
        seed (int, optional): Seed for reproducibility.
        device (str, optional): Target device.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    
    # Generate random bits and map [0, 1] to [-1, 1]
    # bit * 2 - 1 maps 0->-1, 1->1
    bits = torch.randint(0, 2, (d,), generator=g, device=device)
    return (bits * 2 - 1).float()

def apply_sign(x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    Apply the sign-flip array S to x.
    S is assumed to be a 1D vector of shape (d,).
    x can have shape (..., d).
    """
    # Simply element-wise multiply for diagonal matrix application
    return x * S
