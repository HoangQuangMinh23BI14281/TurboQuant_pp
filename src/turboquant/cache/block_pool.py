import torch
from typing import Dict, List, Optional, Tuple, NamedTuple

class PhysicalBlock(NamedTuple):
    block_id: int
    device: torch.device

class KVBlockPool:
    """
    Physical memory pool for quantized KV blocks.
    Standardized for PagedAttention-style indexing.
    
    Each block holds 'block_size' tokens.
    Memory is pre-allocated as a single large tensor for each quantized component.
    """
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        head_dim: int,
        n_heads: int,
        key_bits: int = 4,
        value_bits: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Standards: Block-128 alignment
        # K (TurboQuantProd) components:
        # 1. MSE Indices (packed)
        # For 4-bit, we pack 2 per byte. head_dim // 2
        # For 3-bit, we might pack differently, but 4-bit is the baseline for the manager.
        self.k_idx_bytes = head_dim // 2
        self.k_indices = torch.zeros(
            (num_blocks, n_heads, block_size, self.k_idx_bytes),
            dtype=torch.uint8, device=self.device
        )
        
        # 2. QJL Signs (packed)
        # head_dim bits per token, packed 8 per byte
        self.qjl_bytes = head_dim // 8
        self.k_qjl = torch.zeros(
            (num_blocks, n_heads, block_size, self.qjl_bytes),
            dtype=torch.uint8, device=self.device
        )
        
        # 3. Norms (residual_norms + norms)
        self.k_metadata = torch.zeros(
            (num_blocks, n_heads, block_size, 2), # 2 floats per token
            dtype=dtype, device=self.device
        )
        
        # V (TurboQuantValue) components:
        # Asymmetric Block-128 quantization
        # 1. V Indices (packed)
        self.v_idx_bytes = head_dim // 2 # 4-bit
        self.v_indices = torch.zeros(
            (num_blocks, n_heads, block_size, self.v_idx_bytes),
            dtype=torch.uint8, device=self.device
        )
        
        # 2. V Metadata: Scale and ZeroPoint per group (Block-128)
        self.n_groups = (head_dim + 127) // 128
        self.v_metadata = torch.zeros(
            (num_blocks, n_heads, block_size, self.n_groups, 2), # (Scale, Zero) per group
            dtype=dtype, device=self.device
        )
        
        # Free list
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks: List[int] = []

    def allocate(self, num_needed: int = 1) -> List[int]:
        """Allocate physical blocks from the pool."""
        if len(self.free_blocks) < num_needed:
            raise MemoryError(f"KVBlockPool OOM: needed {num_needed}, free {len(self.free_blocks)}")
        
        allocated = []
        for _ in range(num_needed):
            bid = self.free_blocks.pop(0)
            self.allocated_blocks.append(bid)
            allocated.append(bid)
        return allocated

    def free(self, block_ids: List[int]):
        """Return blocks to the pool."""
        for bid in block_ids:
            if bid in self.allocated_blocks:
                self.allocated_blocks.remove(bid)
                self.free_blocks.append(bid)

    def get_k_block(self, block_id: int):
        return {
            "indices": self.k_indices[block_id],
            "qjl": self.k_qjl[block_id],
            "metadata": self.k_metadata[block_id]
        }

    def get_v_block(self, block_id: int):
        return {
            "indices": self.v_indices[block_id],
            "metadata": self.v_metadata[block_id]
        }

    @property
    def usage(self) -> float:
        return len(self.allocated_blocks) / self.num_blocks
