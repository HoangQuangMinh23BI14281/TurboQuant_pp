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
        # Orthonormal normalization: 1/sqrt(d)
        # We ensure the division happens in float64 to prevent precision loss.
        H = H.to(torch.float64) / math.sqrt(d)
        
    return H

def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT).
    Optimized iterative implementation for PyTorch.
    Matching llama.cpp scaling: no normalization in forward pass.
    
    Complexity: O(d log d) where d is the last dimension of x.
    x shape: (..., d) where d is a power of 2.
    """
    d = x.shape[-1]
    if (d & (d - 1)) != 0:
        raise ValueError("Last dimension of x must be a power of 2")

    # Standard iterative FWHT
    # x shape: (..., d)
    original_shape = x.shape
    original_dtype = x.dtype
    # Use float32 for half-precision inputs, otherwise use input dtype (e.g. float64)
    calc_dtype = torch.float32 if x.dtype in [torch.half, torch.bfloat16] else x.dtype
    res = x.to(calc_dtype).reshape(-1, d)
    
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
    
    # Orthonormal normalization: divide by sqrt(d) (ensuring dtype preservation)
    res_norm = res / torch.tensor(math.sqrt(d), dtype=res.dtype, device=res.device)
    return res_norm.to(original_dtype)

def ifwht(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Fast Walsh-Hadamard Transform.
    Since FWHT is now orthonormal, it is its own inverse.
    """
    return fwht(x)
