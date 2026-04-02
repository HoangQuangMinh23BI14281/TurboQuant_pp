import torch
import math
from typing import Optional

class KVBlockPool:
    """
    Paged Memory Pool for TurboQuant++ Paged Attention.
    Standardized to Block-128 quantization groups.
    """
    def __init__(
        self,
        num_blocks: int,
        head_dim: int,
        n_heads: int,
        tokens_per_block: int = 16, # number of tokens per paged block
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        k_bits: int = 8,
        v_bits: int = 3,
        v_vals_per_byte: int = 2 # 3-bit packed as 4-bit (2 per byte)
    ):
        self.num_blocks = num_blocks
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.tokens_per_block = tokens_per_block
        self.device = device
        self.dtype = dtype
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.v_vals_per_byte = v_vals_per_byte
        
        # SOTA: TurboQuant++ always operates on 128-element blocks (padding target)
        self.padded_head_dim = max(128, head_dim)
        
        # 1. Calculated Buffer Sizes (Dynamic based on packing logic)
        # Key Indices (MSE): vals_per_byte = 8 // (bits-1)
        k_mse_bits = k_bits - 1
        k_vals_per_byte = 8 // k_mse_bits
        self.k_idx_bytes = self.padded_head_dim // k_vals_per_byte
        
        # Key QJL Signs: 1-bit packed (8 items per byte)
        self.k_qjl_bytes = self.padded_head_dim // 8
        
        # Value Indices: vals_per_byte = 8 // bits
        v_vals_per_byte = 8 // v_bits
        self.v_idx_bytes = head_dim // v_vals_per_byte
        
        # 2. Static Pre-allocation
        # Key Physical Storage
        self.k_indices = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, self.k_idx_bytes), 
            dtype=torch.uint8, device=self.device
        )
        self.k_qjl = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, self.k_qjl_bytes), 
            dtype=torch.uint8, device=self.device
        )
        
        # Key Metadata: (norm, scale, residual_norm) for Sketched Attention reconstruction
        self.k_metadata = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, 3), 
            dtype=torch.float32, device=self.device
        )
        
        # Value Physical Storage
        self.v_indices = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, self.v_idx_bytes),
            dtype=torch.uint8, device=self.device
        )
        
        # SOTA FIX: Per-token metadata for Value (Multi-group support)
        self.num_v_groups = max(1, head_dim // 128)
        self.v_metadata = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, self.num_v_groups, 2), # (Scale, Zero)
            dtype=torch.float32, device=self.device
        )
        
        # NATIVE Path Storage: Paged FP16 Buffers (used when quantizer is None)
        self.k_fp16 = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, head_dim),
            dtype=self.dtype, device=self.device
        )
        self.v_fp16 = torch.zeros(
            (num_blocks, n_heads, tokens_per_block, head_dim),
            dtype=self.dtype, device=self.device
        )
        
        # Free list management
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = 0
        
    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("KVPool exhaustion: no free blocks available.")
        self.allocated_blocks += 1
        return self.free_blocks.pop(0)

    def free_block(self, block_id: int):
        self.free_blocks.append(block_id)
        self.allocated_blocks -= 1

    @property
    def usage(self) -> float:
        return self.allocated_blocks / self.num_blocks

    def reset(self):
        self.free_blocks = list(range(self.num_blocks))
        self.allocated_blocks = 0
        self.k_indices.zero_()
        self.k_qjl.zero_()
        self.k_metadata.zero_()
        self.v_indices.zero_()
        self.v_metadata.zero_()
        self.k_fp16.zero_()
        self.v_fp16.zero_()
