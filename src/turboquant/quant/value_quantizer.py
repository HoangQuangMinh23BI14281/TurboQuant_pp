import torch
import torch.nn as nn
from .key_quantizer import TurboQuantMSE
from .quant_base import ValueQuantized, pack_indices, unpack_indices
from typing import Optional

class TurboQuantValue(nn.Module):
    """
    TurboQuant MSE-optimal quantizer for Value (V) cache.
    """
    # SOTA FIX: Thêm block_size và **kwargs để miễn nhiễm với TypeError
    def __init__(self, dim: int, bits: int = 8, n_rotation_passes: int = 1, block_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.dim = dim
        self.bits = bits
        
        # Tự động scale block_size = dim cho các test di sản nếu không truyền vào
        if block_size is None:
            import math
            block_size = int(2 ** math.ceil(math.log2(dim))) if dim > 0 else 1
            
        self.mse_quantizer = TurboQuantMSE(dim, bits, n_rotation_passes, dist='laplace', block_size=block_size)
        self.block_size = self.mse_quantizer.block_size

    def quantize(self, x: torch.Tensor, pack: bool = True) -> ValueQuantized:
        mse_q = self.mse_quantizer.quantize(x, pack=pack)
        # SOTA Fast-Path: Pre-pack metadata for single-dispatch copy
        # For Value, we have [norms, scales]
        meta = torch.cat([
            mse_q.norms.float(), 
            mse_q.scales.float()
        ], dim=-1)

        return ValueQuantized(
            indices=mse_q.indices,
            norms=mse_q.norms,
            scales=mse_q.scales,
            bits=self.bits,
            packed=pack,
            meta=meta
        )

    def dequantize(self, q: ValueQuantized) -> torch.Tensor:
        from .quant_base import MSEQuantized
        mse_q = MSEQuantized(
            indices=q.indices,
            norms=q.norms,
            scales=q.scales,
            bits=q.bits,
            packed=q.packed
        )
        return self.mse_quantizer.dequantize(mse_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))