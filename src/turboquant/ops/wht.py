import torch
import math

def generate_hadamard(d: int, normalized: bool = True) -> torch.Tensor:
    """
    Generate a Hadamard matrix of size d x d using iterative construction.
    d must be a power of 2.
    """
    if (d & (d - 1)) != 0 or d <= 0:
        raise ValueError("d must be a power of 2")

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32)
    
    current_d = 2
    while current_d < d:
        H = torch.cat((
            torch.cat((H, H), dim=1),
            torch.cat((H, -H), dim=1)
        ), dim=0)
        current_d *= 2
        
    if normalized:
        H = H / math.sqrt(d)
        
    return H

def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT).
    Optimized iterative implementation for PyTorch.
    Complexity: O(d log d) where d is the last dimension of x.
    x shape: (..., d) where d is a power of 2.
    """
    d = x.shape[-1]
    if (d & (d - 1)) != 0:
        raise ValueError("Last dimension of x must be a power of 2")

    # Standard iterative FWHT
    # x shape: (..., d)
    original_shape = x.shape
    res = x.float().reshape(-1, d)
    
    num_steps = int(math.log2(d))
    for i in range(num_steps):
        # Butterfly distance for this step: 2^i
        group_size = 2**i
        num_groups = d // (2 * group_size)
        
        # Reshape to (batch, num_groups, 2, group_size)
        res = res.view(-1, num_groups, 2, group_size)
        
        # Apply butterfly: [a+b, a-b]
        a = res[:, :, 0, :]
        b = res[:, :, 1, :]
        res = torch.stack([a + b, a - b], dim=2)
        
    res = res.reshape(original_shape)
    
    # Normalize by sqrt(d) for orthonormality
    return res / math.sqrt(float(d))

def ifwht(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Fast Walsh-Hadamard Transform.
    Since the normalized WHT is its own inverse (orthonormal and symmetric):
    ifwht(x) == fwht(x)
    """
    return fwht(x)
