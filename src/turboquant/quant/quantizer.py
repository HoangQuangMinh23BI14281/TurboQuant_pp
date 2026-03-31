import torch
import math
from typing import Dict, Optional, Tuple
from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize
from ..ops.rotation import TurboQuantRotation

class TurboQuantizer:
    def __init__(self, d: int, bits: int, use_qjl: bool = True):
        self.d = d
        self.bits = bits
        self.use_qjl = use_qjl
        if use_qjl:
            self.mse_bits = bits - 1
            self.qjl_bits = 1
        else:
            self.mse_bits = bits
            self.qjl_bits = 0
        self.block_size = 256
        if d > 256:
            self.block_size = ((d + 255) // 256) * 256
        self.rotation = TurboQuantRotation(self.block_size, pattern='tbq')

    def quantize_key(self, key: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = key.device
        dtype = key.dtype
        shape = key.shape[:-1]
        vec_norms = torch.norm(key, p=2, dim=-1, keepdim=True)
        key_norm = key / (vec_norms + 1e-10)
        if self.block_size > self.d:
            padding = torch.zeros((*shape, self.block_size - self.d), device=device, dtype=dtype)
            key_padded = torch.cat([key_norm, padding], dim=-1)
        else:
            key_padded = key_norm
        key_rotated = self.rotation(key_padded)
        indices = lloyd_max_quantize(key_rotated, self.mse_bits)
        reconstructed_rotated = lloyd_max_dequantize(indices, self.mse_bits)
        result = {'norm': vec_norms.squeeze(-1), 'indices': indices.to(torch.uint8)}
        if self.use_qjl:
            residual = key_rotated - reconstructed_rotated
            qjl_indices = (residual >= 0).to(torch.uint8)
            res_abs_mean = torch.mean(torch.abs(residual), dim=-1)
            residual_norm = res_abs_mean * math.sqrt(math.pi / 2.0)
            result['qjl_indices'] = qjl_indices
            result['residual_norm'] = residual_norm
        return result

    def dequantize_key(self, quant_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        norm = quant_data['norm'].unsqueeze(-1)
        indices = quant_data['indices']
        reconstructed_rotated = lloyd_max_dequantize(indices, self.mse_bits)
        if self.use_qjl and 'qjl_indices' in quant_data:
            qjl_indices = quant_data['qjl_indices']
            residual_norm = quant_data['residual_norm'].unsqueeze(-1)
            qjl_recon = (qjl_indices.float() * 2.0 - 1.0) * residual_norm
            reconstructed_rotated = reconstructed_rotated + qjl_recon
        reconstructed_padded = self.rotation.inverse(reconstructed_rotated)
        return reconstructed_padded[..., :self.d] * norm

    def pack_indices(self, indices: torch.Tensor) -> torch.Tensor:
        shape = indices.shape
        indices_flat = indices.reshape(-1, shape[-1])
        n_vectors, d = indices_flat.shape
        if self.mse_bits == 8: return indices.to(torch.uint8)
        vals_per_byte = 8 // self.mse_bits
        packed_d = d // vals_per_byte
        packed = torch.zeros((n_vectors, packed_d), device=indices.device, dtype=torch.uint8)
        for i in range(vals_per_byte):
            shift = 8 - (i + 1) * self.mse_bits
            packed |= (indices_flat[:, i::vals_per_byte].to(torch.uint8) << shift)
        return packed.reshape(*shape[:-1], packed_d)

    def unpack_indices(self, packed: torch.Tensor, original_d: int) -> torch.Tensor:
        shape = packed.shape
        packed_flat = packed.reshape(-1, shape[-1])
        n_vectors, packed_d = packed_flat.shape
        if self.mse_bits == 8: return packed.to(torch.long)
        vals_per_byte = 8 // self.mse_bits
        unpacked = torch.zeros((n_vectors, original_d), device=packed.device, dtype=torch.long)
        mask = (1 << self.mse_bits) - 1
        for i in range(vals_per_byte):
            shift = 8 - (i + 1) * self.mse_bits
            unpacked[:, i::vals_per_byte] = ((packed_flat >> shift) & mask).to(torch.long)
        return unpacked.reshape(*shape[:-1], original_d)
