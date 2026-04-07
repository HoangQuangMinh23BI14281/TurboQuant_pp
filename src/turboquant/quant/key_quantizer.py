import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .quant_base import MSEQuantized, ProdQuantized, ValueQuantized, pack_indices, unpack_indices
from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize
from ..ops.rotation import TurboQuantRotation
from ..ops.wht import fwht, ifwht
from ..ops.sign_array import generate_sign_array, apply_sign_array
from ..cache.routing import QuantizationStrategy

class TurboQuantMSE(nn.Module):
    def __init__(self, dim: int, bits: int = 8, n_rotation_passes: int = 1, dist: str = 'gaussian', block_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.dist = dist
        self.n_levels = 2 ** bits
        self.n_rotation_passes = n_rotation_passes
        self.strategy = QuantizationStrategy.TURBO_MSE
        self.epsilon = 1e-8 

        if block_size is not None:
            self.block_size = block_size
        else:
            self.block_size = int(2 ** math.ceil(math.log2(dim))) if dim > 0 else 1
        
        self.n_subblocks = math.ceil(self.dim / self.block_size)
        self.padded = (self.block_size * self.n_subblocks != self.dim)
        self.rotation = TurboQuantRotation(self.block_size, n_passes=n_rotation_passes, pattern='tbq')

    def transform_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.shape[-2] == 1:
            query = query.squeeze(-2)
        
        if self.padded:
            query = F.pad(query.float(), (0, self.block_size * self.n_subblocks - self.dim))
        else:
            query = query.float()
        
        # SOTA FIX: Đảm bảo contiguous trước khi view để xử lý paged memory
        q_reshaped = query.contiguous().view(-1, self.block_size)
        q_rot_reshaped = self.rotation(q_reshaped)
        return q_rot_reshaped.view(query.shape)

    def quantize(self, x: torch.Tensor, pack: bool = False, precomputed_norms: torch.Tensor = None, precomputed_scales: torch.Tensor = None) -> MSEQuantized:
        shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype
        
        if self.padded:
            x_padded_init = F.pad(x.float(), (0, self.block_size * self.n_subblocks - self.dim))
        else:
            x_padded_init = x.float()
            
        # SOTA FIX: .contiguous() là bắt buộc khi dữ liệu bị slice từ KV Cache
        x_reshaped = x_padded_init.contiguous().view(-1, self.block_size)
        
        if precomputed_norms is not None:
            vec_norms = precomputed_norms.contiguous().view(-1, 1)
        else:
            vec_norms = torch.norm(x_reshaped, p=2, dim=-1, keepdim=True) + self.epsilon
            
        x_unit_reshaped = x_reshaped / vec_norms
        x_rot_reshaped = self.rotation(x_unit_reshaped)
        
        if precomputed_scales is not None:
            refined_gamma = precomputed_scales.contiguous().view(-1, 1)
            x_normalized = x_rot_reshaped / (refined_gamma + self.epsilon)
            indices = lloyd_max_quantize(x_normalized, self.bits, dist=self.dist)
        else:
            x_rot_max = torch.max(torch.abs(x_rot_reshaped.float()), dim=-1, keepdim=True).values
            
            # SOTA FIX: Nội suy max_c từ Codebook thực tế (Gaussian vs Laplace)
            from .lloyd_max import compute_lloyd_max_codebook
            cb = compute_lloyd_max_codebook(self.bits, d=1, dist=self.dist)
            max_c = cb['centroids'].max().item()
            
            rms_scales = (x_rot_max / max_c).to(dtype) 
            x_normalized = x_rot_reshaped / (rms_scales + self.epsilon)
            
            x_rot_f32 = x_rot_reshaped.float()
            
            # Pass 1: Lấy indices sơ bộ
            indices_tmp = lloyd_max_quantize(x_normalized, self.bits, dist=self.dist)
            recon_u = lloyd_max_dequantize(indices_tmp, self.bits, dist=self.dist).float()
            
            num = (x_rot_f32 * recon_u).sum(dim=-1, keepdim=True)
            den = (recon_u * recon_u).sum(dim=-1, keepdim=True) + self.epsilon
            gamma_pass1 = (num / den).to(dtype)
            
            # Pass 2: Tinh chỉnh tinh vi
            x_normalized_2 = x_rot_reshaped / (gamma_pass1 + self.epsilon)
            indices_pass2 = lloyd_max_quantize(x_normalized_2, self.bits, dist=self.dist)
            
            recon_f = lloyd_max_dequantize(indices_pass2, self.bits, dist=self.dist).float()
            num_f = (x_rot_f32 * recon_f).sum(dim=-1, keepdim=True)
            den_f = (recon_f * recon_f).sum(dim=-1, keepdim=True) + self.epsilon
            gamma_pass2 = (num_f / den_f).to(dtype)
            
            # SOTA FIX: Spiky Guard (Bảo vệ hằng số/outliers cực hạn)
            x_rot_mean = torch.mean(torch.abs(x_rot_reshaped.float()), dim=-1, keepdim=True) + self.epsilon
            is_spiky = (x_rot_max / x_rot_mean) > 5.0
            
            # CỰC KỲ QUAN TRỌNG: Nếu spiky, phải dùng CẢ scale gốc VÀ indices gốc
            refined_gamma = torch.where(is_spiky, rms_scales, gamma_pass2)
            indices = torch.where(is_spiky, indices_tmp, indices_pass2)
            
        if pack:
            indices = pack_indices(indices.view(shape + (-1,)), self.bits)
        else:
            indices = indices.view(shape + (-1,))
            
        meta_shape = shape + (self.n_subblocks,)
        return MSEQuantized(
            indices=indices, 
            norms=vec_norms.view(meta_shape), 
            scales=refined_gamma.view(meta_shape), 
            bits=self.bits, 
            packed=pack
        )

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        shape = q.indices.shape[:-1 if q.packed else -1]
        dtype = q.norms.dtype
        indices = q.indices
        
        n_subblocks = q.norms.shape[-1]
        block_size = self.block_size
        if n_subblocks * block_size < self.dim:
             block_size = math.ceil(self.dim / n_subblocks)
             
        if q.packed:
            indices = unpack_indices(indices, q.bits, block_size * n_subblocks)

        x_reconstructed = lloyd_max_dequantize(indices, q.bits, dist=self.dist)
        recon_reshaped = x_reconstructed.view(-1, block_size)
        
        scales_flat = q.scales.contiguous().view(-1, 1)
        norms_flat = q.norms.contiguous().view(-1, 1)
        
        x_rot_scaled = recon_reshaped * scales_flat
        x_unit_blocks = self.rotation.inverse(x_rot_scaled)
        
        x_hat_blocks = x_unit_blocks * norms_flat
        x_hat = x_hat_blocks.view(shape + (-1,))[..., :self.dim]
        return x_hat.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

    def quantize_and_residual(self, x: torch.Tensor, pack: bool = True, precomputed_norms: torch.Tensor = None, precomputed_scales: torch.Tensor = None) -> Tuple[MSEQuantized, torch.Tensor]:
        mse_q = self.quantize(x, pack=pack, precomputed_norms=precomputed_norms, precomputed_scales=precomputed_scales)
        
        indices = unpack_indices(mse_q.indices, self.bits, self.block_size * self.n_subblocks) if pack else mse_q.indices
        reconstructed_unit = lloyd_max_dequantize(indices, self.bits, dist='gaussian')
        recon_reshaped = reconstructed_unit.view(-1, self.block_size)
        
        scales_flat = mse_q.scales.contiguous().view(-1, 1)
        norms_flat = mse_q.norms.contiguous().view(-1, 1)
        
        reconstructed_rotated_blocks = recon_reshaped * scales_flat
        
        x_fp32 = x.float()
        if self.padded:
            x_padded_init = F.pad(x_fp32, (0, self.block_size * self.n_subblocks - self.dim))
        else:
            x_padded_init = x_fp32
            
        x_reshaped = x_padded_init.contiguous().view(-1, self.block_size)
        x_unit_blocks = x_reshaped / (norms_flat + self.epsilon)
        
        x_rotated_blocks = self.rotation(x_unit_blocks.to(x.dtype))
        
        residual_rotated_blocks = x_rotated_blocks.float() - reconstructed_rotated_blocks.float()
        residual_rotated = residual_rotated_blocks.view(x_padded_init.shape)
        
        return mse_q, residual_rotated.to(x.dtype)


class TurboQuantProd(nn.Module):
    def __init__(self, dim: int, bits: int = 8, n_rotation_passes: int = 1, block_size: Optional[int] = None, **kwargs):
        super().__init__()
        assert bits >= 2
        self.dim = dim
        self.bits = bits
        self.mse_bits = bits - 1
        
        if block_size is None:
            block_size = int(2 ** math.ceil(math.log2(dim))) if dim > 0 else 1
            
        self.mse_quantizer = TurboQuantMSE(dim, self.mse_bits, n_rotation_passes, dist='gaussian', block_size=block_size)
        self.block_size = self.mse_quantizer.block_size
        self.n_subblocks = self.mse_quantizer.n_subblocks
        self.strategy = QuantizationStrategy.TURBO_PROD
        self.qjl_scale = math.sqrt(2.0 / math.pi) / math.sqrt(self.block_size)

    def transform_query(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_rot = self.mse_quantizer.transform_query(query)
        return q_rot, q_rot

    def quantize(self, x: torch.Tensor, pack: bool = True, precomputed_norms=None, precomputed_scales=None, precomputed_res_norms=None) -> ProdQuantized:
        mse_q, residual_rotated = self.mse_quantizer.quantize_and_residual(x, pack=False, precomputed_norms=precomputed_norms, precomputed_scales=precomputed_scales)
        
        qjl_sign_bits = (residual_rotated >= 0).to(torch.uint8)
        
        if precomputed_res_norms is not None:
            residual_norms = precomputed_res_norms
        else:
            res_reshaped = residual_rotated.float().view(-1, self.block_size)
            res_norms_unit = torch.norm(res_reshaped, p=2, dim=-1, keepdim=True)
            residual_norms = (res_norms_unit * mse_q.norms.contiguous().view(-1, 1)).view(mse_q.norms.shape)

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
        shape = q.mse_indices.shape[:-1 if q.packed else -1]
        dtype = q.norms.dtype
        
        mse_indices = q.mse_indices
        qjl_signs = q.qjl_signs

        if q.packed:
            # SOTA FIX: Dùng đúng self.block_size
            mse_indices = unpack_indices(mse_indices, q.mse_bits, self.block_size * self.n_subblocks)
            qjl_signs = unpack_indices(qjl_signs, 1, self.block_size * self.n_subblocks)

        reconstructed_normalized = lloyd_max_dequantize(mse_indices, q.mse_bits)
        
        # SOTA FIX: Giữ nguyên shape block để tính toán với residual
        recon_reshaped = reconstructed_normalized.view(-1, self.block_size)
        
        scales_flat = q.scales.contiguous().view(-1, 1)
        norms_flat = q.norms.contiguous().view(-1, 1)
        res_norms_flat = q.residual_norms.contiguous().view(-1, 1)

        # Tính MSE part trong không gian block_size
        reconstructed_mse_rot_blocks = recon_reshaped * scales_flat

        signs_float = qjl_signs.float() * 2.0 - 1.0
        res_dir_reshaped = signs_float.view(-1, self.block_size)
        
        dir_norm = torch.norm(res_dir_reshaped, p=2, dim=-1, keepdim=True) + self.mse_quantizer.epsilon
        
        unit_res_norms_flat = res_norms_flat / (norms_flat + self.mse_quantizer.epsilon)
        res_est_unit_blocks = res_dir_reshaped * (unit_res_norms_flat / dir_norm)

        # Gộp 2 thành phần lại khi vẫn còn ở shape (-1, block_size)
        combined_reshaped = (reconstructed_mse_rot_blocks + res_est_unit_blocks).to(dtype)
        
        # Xoay ngược (Hàm inverse mới ở trên sẽ lo vụ shape 64 vs 128)
        x_unit_blocks = self.mse_quantizer.rotation.inverse(combined_reshaped)
        
        x_hat_blocks = x_unit_blocks * norms_flat
        
        # Cắt padding và trả về
        return x_hat_blocks.view(shape + (-1,))[..., :self.dim].to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))