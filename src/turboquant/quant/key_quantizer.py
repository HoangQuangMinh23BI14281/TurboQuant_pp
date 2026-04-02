import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .quant_base import MSEQuantized, ProdQuantized, pack_indices, unpack_indices
from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize
from ..ops.rotation import TurboQuantRotation
from ..ops.wht import fwht, ifwht
from ..ops.sign_array import generate_sign_array, apply_sign_array

class TurboQuantMSE(nn.Module):
    """
    TurboQuant MSE-optimal quantizer (Algorithm 1) for K or V.
    Standardized to 128 block size with cascaded SRHT.
    """
    def __init__(self, dim: int, bits: int = 8, n_rotation_passes: int = 2):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits
        self.n_rotation_passes = n_rotation_passes

        bs = 128
        while bs < dim:
            bs *= 2
        self.block_size = bs
        self.padded = (self.block_size != self.dim)
        self.rotation = TurboQuantRotation(self.block_size, n_passes=n_rotation_passes, pattern='tbq')

    def transform_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.shape[-2] == 1:
            query = query.squeeze(-2)
        
        if self.padded:
            query = F.pad(query.float(), (0, self.block_size - self.dim))
        else:
            query = query.float()
        
        return self.rotation(query)

    def quantize(self, x: torch.Tensor, pack: bool = False) -> MSEQuantized:
        shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype

        vec_norms = torch.norm(x, p=2, dim=-1)
        x_unit = x / (vec_norms.unsqueeze(-1) + 1e-10)

        if self.padded:
            padding = torch.zeros((*shape, self.block_size - self.dim), device=device, dtype=dtype)
            x_padded = torch.cat([x_unit, padding], dim=-1)
        else:
            x_padded = x_unit

        x_rotated = self.rotation(x_padded)
        rms_scales = torch.norm(x_rotated, dim=-1, keepdim=True) / math.sqrt(self.block_size)
        x_normalized = x_rotated / (rms_scales + 1e-10)

        indices = lloyd_max_quantize(x_normalized, self.bits, dist='gaussian')
        reconstructed_unit = lloyd_max_dequantize(indices, self.bits, dist='gaussian')
        
        numerator = (x_rotated * reconstructed_unit).sum(dim=-1, keepdim=True)
        denominator = (reconstructed_unit * reconstructed_unit).sum(dim=-1, keepdim=True) + 1e-10
        refined_gamma = numerator / denominator

        if pack:
            indices = pack_indices(indices, self.bits)

        return MSEQuantized(indices=indices, norms=vec_norms, scales=refined_gamma, bits=self.bits, packed=pack)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = q.indices
        if q.packed:
            indices = unpack_indices(indices, q.bits, self.block_size)

        x_reconstructed = lloyd_max_dequantize(indices, q.bits)
        
        if x_reconstructed.shape[-1] < self.block_size:
            x_reconstructed = F.pad(x_reconstructed, (0, self.block_size - x_reconstructed.shape[-1]))
        elif x_reconstructed.shape[-1] > self.block_size:
            x_reconstructed = x_reconstructed[..., :self.block_size]

        x_rotated = x_reconstructed * q.scales
        x_padded = self.rotation.inverse(x_rotated)
        x_hat = x_padded[..., :self.dim] * q.norms.unsqueeze(-1)

        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing roundtrip quality)."""
        return self.dequantize(self.quantize(x))

    def quantize_and_residual(self, x: torch.Tensor, pack: bool = True) -> Tuple[MSEQuantized, torch.Tensor]:
        shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype

        vec_norms = torch.norm(x, p=2, dim=-1)
        x_unit = x / (vec_norms.unsqueeze(-1) + 1e-10)

        if self.padded:
            padding = torch.zeros((*shape, self.block_size - self.dim), device=device, dtype=dtype)
            x_padded = torch.cat([x_unit, padding], dim=-1)
        else:
            x_padded = x_unit

        x_rotated = self.rotation(x_padded)
        rms_scales = torch.norm(x_rotated, dim=-1, keepdim=True) / math.sqrt(self.block_size)
        x_normalized = x_rotated / (rms_scales + 1e-10)
        
        indices = lloyd_max_quantize(x_normalized, self.bits, dist='gaussian')
        reconstructed_unit = lloyd_max_dequantize(indices, self.bits, dist='gaussian')
        
        numerator = (x_rotated * reconstructed_unit).sum(dim=-1, keepdim=True)
        denominator = (reconstructed_unit * reconstructed_unit).sum(dim=-1, keepdim=True) + 1e-10
        refined_gamma = numerator / denominator
        
        reconstructed_rotated = reconstructed_unit * refined_gamma
        residual_rotated = x_rotated - reconstructed_rotated

        if pack:
            indices = pack_indices(indices, self.bits)

        mse_q = MSEQuantized(indices=indices, norms=vec_norms, scales=refined_gamma, bits=self.bits, packed=pack)
        return mse_q, residual_rotated


class TurboQuantProd(nn.Module):
    """
    TurboQuant inner-product-optimal quantizer (Algorithm 2) for K.
    """
    def __init__(self, dim: int, bits: int = 8, n_rotation_passes: int = 2):
        super().__init__()
        assert bits >= 2
        self.dim = dim
        self.bits = bits
        self.mse_bits = bits - 1
        self.mse_quantizer = TurboQuantMSE(dim, self.mse_bits, n_rotation_passes)
        self.block_size = self.mse_quantizer.block_size

        qjl_signs = generate_sign_array(self.block_size, use_llama_preset='qjl')
        self.register_buffer('qjl_signs', qjl_signs)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / math.sqrt(self.block_size)

    def transform_query(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_rot = self.mse_quantizer.transform_query(query)
        q_sketch = fwht(apply_sign_array(q_rot, self.qjl_signs))
        return q_rot, q_sketch

    def quantize(self, x: torch.Tensor, pack: bool = True) -> ProdQuantized:
        mse_q, residual_rotated = self.mse_quantizer.quantize_and_residual(x, pack=False)
        residual_signed = apply_sign_array(residual_rotated, self.qjl_signs)
        residual_projected = fwht(residual_signed)

        qjl_sign_bits = (residual_projected >= 0).to(torch.uint8)
        residual_norms = torch.norm(residual_rotated, dim=-1) * mse_q.norms

        mse_indices = mse_q.indices
        if pack:
            mse_indices = pack_indices(mse_indices, self.mse_bits)
            qjl_sign_bits = pack_indices(qjl_sign_bits, 1)

        return ProdQuantized(
            mse_indices=mse_indices,
            qjl_signs=qjl_sign_bits,
            scales=mse_q.scales,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=self.mse_bits,
            packed=pack,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        mse_indices = q.mse_indices
        qjl_signs = q.qjl_signs

        if q.packed:
            mse_indices = unpack_indices(mse_indices, q.mse_bits, self.block_size)
            qjl_signs = unpack_indices(qjl_signs, 1, self.block_size)

        reconstructed_normalized = lloyd_max_dequantize(mse_indices, q.mse_bits)
        reconstructed_mse_rot = reconstructed_normalized * q.scales

        signs_float = qjl_signs.float() * 2.0 - 1.0
        residual_direction = ifwht(signs_float)
        residual_direction = apply_sign_array(residual_direction, self.qjl_signs)

        dir_norm = torch.norm(residual_direction, dim=-1, keepdim=True) + 1e-10
        unit_res_norms = q.residual_norms / (q.norms + 1e-10)
        residual_estimated_unit = residual_direction * (unit_res_norms.unsqueeze(-1) / dir_norm)

        combined_rotated = reconstructed_mse_rot + residual_estimated_unit
        x_padded = self.mse_quantizer.rotation.inverse(combined_rotated)
        return x_padded[..., :self.dim] * q.norms.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing roundtrip quality)."""
        return self.dequantize(self.quantize(x))
