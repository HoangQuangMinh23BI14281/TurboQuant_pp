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
        config, 
        head_dim: int,
        n_heads: int,
        num_blocks: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        n_layers: int = 24
    ):
        self.config = config
        self.num_blocks = num_blocks if num_blocks is not None else config.hw.num_blocks
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.tokens_per_block = config.hw.tokens_per_block
        self.device = device
        self.dtype = dtype
        self.k_bits = config.quant.k_bits
        self.v_bits = config.quant.v_bits
        self.n_layers = n_layers
        
        # FIX TỐI THƯỢNG: ÉP cứng group_size bằng 64 để đồng bộ tuyệt đối với logic bên trong TurboQuantMSE (trừ khi nó bị force bé hơn)
        self.k_group_size = min(64, int(2 ** math.ceil(math.log2(head_dim)))) if head_dim > 0 else 1
        self.v_group_size = self.k_group_size 
        
        self.padded_head_dim = head_dim
        
        k_mse_bits = max(1, self.k_bits - 1)
        k_vals_per_byte = 8 // k_mse_bits
        
        self.k_subblocks = math.ceil(self.head_dim / self.k_group_size)
        self.v_subblocks = math.ceil(self.head_dim / self.v_group_size)
        
        k_padded_dim = self.k_subblocks * self.k_group_size
        v_padded_dim = self.v_subblocks * self.v_group_size
        
        self.k_idx_bytes = k_padded_dim // k_vals_per_byte
        self.k_qjl_bytes = k_padded_dim // 8
        
        v_vals_per_byte = 8 // self.v_bits
        self.v_idx_bytes = v_padded_dim // v_vals_per_byte
        
        self.k_indices = torch.zeros((self.n_layers, self.num_blocks, self.n_heads, self.tokens_per_block, self.k_idx_bytes), dtype=torch.uint8, device=self.device)
        self.k_qjl = torch.zeros((self.n_layers, self.num_blocks, self.n_heads, self.tokens_per_block, self.k_qjl_bytes), dtype=torch.uint8, device=self.device)
        
        # Cấp phát chắc chắn 3 tham số nhân với số subblocks
        self.k_metadata = torch.zeros((self.n_layers, self.num_blocks, self.n_heads, self.tokens_per_block, 3 * self.k_subblocks), dtype=torch.float32, device=self.device)
        
        self.v_indices = torch.zeros((self.n_layers, self.num_blocks, self.n_heads, self.tokens_per_block, self.v_idx_bytes), dtype=torch.uint8, device=self.device)
        self.v_metadata = torch.zeros((self.n_layers, self.num_blocks, self.n_heads, self.tokens_per_block, 2 * self.v_subblocks), dtype=torch.float32, device=self.device)
        
        self.k_summaries = torch.zeros((self.n_layers, self.num_blocks, self.n_heads, 2, self.padded_head_dim), dtype=torch.float32, device=self.device)
        
        self.free_blocks = list(range(self.num_blocks))
        self.allocated_blocks = 0
        
    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("KVPool exhaustion: no free blocks available.")
        self.allocated_blocks += 1
        return self.free_blocks.pop(0)

    def allocate_layer_blocks(self, num_blocks: int) -> torch.Tensor:
        """SOTA: Pre-allocate a slab of blocks for a specific layer.
        Essential for CUDA Graphs to avoid Python allocation in loop.
        """
        if len(self.free_blocks) < num_blocks:
            raise MemoryError(f"KVPool exhaustion: requested {num_blocks} but only {len(self.free_blocks)} free.")
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(self.free_blocks.pop(0))
            self.allocated_blocks += 1
            
        return torch.tensor(blocks, dtype=torch.int32, device=self.device)

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
        self.k_summaries.zero_()