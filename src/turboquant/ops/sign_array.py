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

def generate_sign_array(d: int, seed: int = 42, use_llama_preset: str = None, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate a 1D tensor of signs (+1 or -1) using pure PyTorch vectorization.
    Preserves determinism via seed or use_llama_preset.
    """
    if use_llama_preset in ['tbq', 'qjl']:
        # 1. Chọn mảng byte
        sign_bytes = TBQ_SIGNS if use_llama_preset == 'tbq' else QJL_SIGNS
        byte_tensor = torch.tensor(list(sign_bytes), dtype=torch.uint8, device=device)
        
        # 2. Vector hóa việc tính toán Index
        idx = torch.arange(d, device=device)
        byte_idx = (idx // 8) % len(sign_bytes)  # Xoay vòng nếu d > 256
        bit_idx = idx % 8
        
        # 3. Dịch bit song song trên GPU/CPU
        bits = (byte_tensor[byte_idx] >> bit_idx) & 1
        
        # 4. Map (0 -> 1.0) và (1 -> -1.0)
        signs = torch.where(bits == 1, -1.0, 1.0).to(dtype)
        return signs
    
    # Generic random sign array
    generator = torch.Generator(device=device)
    if device is not None and device.type == 'cuda':
        generator.manual_seed(seed)
    else:
        generator.manual_seed(seed)
        
    bits = torch.randint(0, 2, (d,), generator=generator, device=device)
    return (bits.to(dtype) * 2.0 - 1.0)

def apply_sign_array(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiply x by signs along the last dimension.
    """
    # Robust broadcasting for any number of leading dimensions
    s = signs.to(x.device, dtype=x.dtype)
    s = s.view(*((1,) * (x.dim() - 1)), -1)
    return x * s
