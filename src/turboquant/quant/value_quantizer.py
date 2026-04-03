import math
import torch
import torch.nn as nn
from .quant_base import ValueQuantized, pack_indices, unpack_indices
from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize

class TurboQuantValue(nn.Module):
    """
    TurboQuant asymmetric quantizer for Value (V) cache.
    Formula: V = (idx - zero_point) * scale
    """
    def __init__(self, dim: int, bits: int = 4, group_size: int = 128):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.group_size = min(group_size, dim)
        self.n_levels = 2**bits
        assert dim % self.group_size == 0
        self.n_groups = dim // self.group_size

    def quantize(self, x: torch.Tensor, pack: bool = True) -> ValueQuantized:
        shape = x.shape[:-1]
        x_grouped = x.view(*shape, self.n_groups, self.group_size)
        
        v_min = x_grouped.min(dim=-1, keepdim=True).values
        v_max = x_grouped.max(dim=-1, keepdim=True).values
        
        # SOTA: Linear Asymmetric Quantization
        # V = (idx * scale) + min
        v_scale = (v_max - v_min) / (self.n_levels - 1 + 1e-10)
        
        # indices = round((x - min) / scale)
        indices = torch.round((x_grouped - v_min) / (v_scale + 1e-10)).clamp(0, self.n_levels - 1)
        
        indices = indices.view(*shape, self.dim).to(torch.uint8)
        scales = v_scale.view(*shape, self.n_groups)
        zero_points = v_min.view(*shape, self.n_groups)

        if pack:
            indices = pack_indices(indices, self.bits)
            
        return ValueQuantized(indices=indices, scales=scales, zero_points=zero_points, bits=self.bits, packed=pack)

    def dequantize(self, q: ValueQuantized) -> torch.Tensor:
        indices = q.indices
        shape = q.scales.shape[:-1]
        
        if q.packed:
            indices = unpack_indices(indices, q.bits, self.dim)
            
        indices_grouped = indices.view(*shape, self.n_groups, self.group_size)
        scales_grouped = q.scales.view(*shape, self.n_groups, 1)
        zp_grouped = q.zero_points.view(*shape, self.n_groups, 1)
        
        # SOTA: V = idx * scale + min
        reconstructed = indices_grouped.float() * scales_grouped + zp_grouped
        
        return reconstructed.view(*shape, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing)."""
        return self.dequantize(self.quantize(x))
