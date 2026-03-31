import torch

# Sign patterns from llama.cpp cpy-utils.cuh (256 bits for 256 elements)
TBQ_SIGNS = bytes([
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
    0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
    0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
])

# Independent sign pattern for QJL SRHT (from llama.cpp)
QJL_SIGNS = bytes([
    0xd3, 0x4e, 0xa8, 0x17, 0x9c, 0x5b, 0xe6, 0x31,
    0x72, 0xb9, 0x0d, 0xf5, 0x43, 0x8a, 0x6e, 0xc7,
    0x58, 0x2f, 0x94, 0xe1, 0xb6, 0x3d, 0x0a, 0x7c,
    0xc5, 0x61, 0xd8, 0x4f, 0xa3, 0x97, 0x1e, 0x85,
])

def get_llama_sign(signs: bytes, idx: int) -> float:
    """Get sign (-1.0 or +1.0) at index idx from sign bytes (matching llama.cpp logic)."""
    return -1.0 if ((signs[idx >> 3] >> (idx & 7)) & 1) else 1.0

def generate_sign_array(d: int, seed: int = 42, use_llama_preset: str = None) -> torch.Tensor:
    """
    Generate a 1D tensor of signs (+1 or -1).
    Preserves determinism via seed or use_llama_preset.
    
    Args:
        d: length of the array
        seed: used if use_llama_preset is None
        use_llama_preset: 'tbq' or 'qjl' to use fixed llama.cpp patterns
    """
    if use_llama_preset == 'tbq':
        return torch.tensor([get_llama_sign(TBQ_SIGNS, i % 256) for i in range(d)])
    elif use_llama_preset == 'qjl':
        return torch.tensor([get_llama_sign(QJL_SIGNS, i % 256) for i in range(d)])
    
    # Generic random sign array
    generator = torch.Generator().manual_seed(seed)
    # Generate random bits (0 or 1) and map to -1 or 1
    bits = torch.randint(0, 2, (d,), generator=generator)
    return (bits.float() * 2.0 - 1.0)

def apply_sign_array(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiply x by signs along the last dimension.
    """
    return x * signs.to(x.device, dtype=x.dtype)
