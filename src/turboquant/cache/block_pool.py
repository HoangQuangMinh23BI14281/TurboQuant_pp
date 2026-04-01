import torch
from typing import Dict, List, Optional, Tuple, NamedTuple
import math

class KVBlockPool:
    """
    SOTA Physical Memory Pool for Hybrid Quantized KV Blocks.
    
    Architecture:
    - Block Size: 128.
    - Value Metadata: Per-token Scales/ZeroPoints for maximum fidelity.
    """
    def __init__(
        self,
        num_blocks: int,
        head_dim: int,
        n_heads: int,
        k_bits: int = 8,
        v_bits: int = 3,
        block_size: int = 128,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.device = torch.device(device)
        self.dtype = dtype
        
        # 1. Packing Calculations
        self.k_vals_per_byte = 8 // k_bits if k_bits <= 8 else 1
        self.k_idx_bytes = head_dim // self.k_vals_per_byte
        self.qjl_bytes = head_dim // 8
        self.v_vals_per_byte = 8 // v_bits if v_bits <= 8 else 1
        self.v_idx_bytes = math.ceil(head_dim / self.v_vals_per_byte)
        
        # 2. Static Pre-allocation
        # Key Physical Storage
        self.k_indices = torch.zeros(
            (num_blocks, n_heads, block_size, self.k_idx_bytes),
            dtype=torch.uint8, device=self.device
        )
        self.k_qjl = torch.zeros(
            (num_blocks, n_heads, block_size, self.qjl_bytes),
            dtype=torch.uint8, device=self.device
        )
        self.k_metadata = torch.zeros(
            (num_blocks, n_heads, block_size, 2), # (residual_norm, norm)
            dtype=torch.float32, device=self.device
        )
        
        # Value Physical Storage
        self.v_indices = torch.zeros(
            (num_blocks, n_heads, block_size, self.v_idx_bytes),
            dtype=torch.uint8, device=self.device
        )
        
        # SOTA FIX: Per-token metadata for Value
        self.v_metadata = torch.zeros(
            (num_blocks, n_heads, block_size, 2), # (Scale, Zero) PER TOKEN
            dtype=torch.float32, device=self.device
        )
        
        self.free_blocks = list(range(num_blocks))
        self.allocated_count = 0

    def allocate(self, num_needed: int = 1) -> torch.Tensor:
        if len(self.free_blocks) < num_needed:
            raise MemoryError(f"TurboQuant KV Pool OOM: Free={len(self.free_blocks)}, Needed={num_needed}")
        
        block_ids = []
        for _ in range(num_needed):
            bid = self.free_blocks.pop(0)
            block_ids.append(bid)
            self.allocated_count += 1
        return torch.tensor(block_ids, dtype=torch.int32, device=self.device)

    def free(self, block_ids: torch.Tensor):
        bids = block_ids.tolist()
        for bid in bids:
            if bid not in self.free_blocks:
                self.free_blocks.append(bid)
                self.allocated_count -= 1

    @property
    def usage(self) -> float:
        return self.allocated_count / self.num_blocks
