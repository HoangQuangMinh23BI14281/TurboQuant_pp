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
        x_centered = x_grouped - v_min
        
        v_scale = torch.norm(x_centered, dim=-1, keepdim=True) / math.sqrt(self.group_size)
        x_unit = x_centered / (v_scale + 1e-10)
        
        indices = lloyd_max_quantize(x_unit, self.bits, dist='laplace')
        unit_recon = lloyd_max_dequantize(indices, self.bits, dist='laplace')
        
        numerator = (x_centered * unit_recon).sum(dim=-1, keepdim=True)
        denominator = (unit_recon * unit_recon).sum(dim=-1, keepdim=True) + 1e-10
        refined_gamma = numerator / denominator
        
        indices = indices.view(*shape, self.dim).to(torch.uint8)
        scales = refined_gamma.view(*shape, self.n_groups)
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
        
        unit_recon = lloyd_max_dequantize(indices_grouped.long(), q.bits, dist='laplace')
        reconstructed = unit_recon * scales_grouped + zp_grouped
        
        return reconstructed.view(*shape, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing)."""
        return self.dequantize(self.quantize(x))
