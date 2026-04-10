import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .quant_base import MSEQuantized, ProdQuantized, ValueQuantized, pack_indices, unpack_indices
from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize, compute_lloyd_max_codebook
from ..ops.rotation import TurboQuantRotation
from ..ops.wht import fwht, ifwht
from ..ops.sign_array import generate_sign_array, apply_sign_array
from ..cache.routing import QuantizationStrategy
from ..kernels.quant_fused import fused_quantize

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
        # V10.14: 64-byte Aligned Boundaries (16 elements)
        cb = compute_lloyd_max_codebook(self.bits, d=1, dist=self.dist)
        # Pad 15 finite boundaries with 1 large value to reach 16
        bounds = cb['boundaries'][1:-1].float()
        padded_bounds = torch.cat([bounds, torch.tensor([1e10], device=bounds.device, dtype=bounds.dtype)])
        self.register_buffer("triton_boundaries", padded_bounds.contiguous())
        self.max_centroid = cb['max_centroid']
        self.register_buffer("rot_final_scale", self.rotation.final_scale.float())
        self.register_buffer("wht_mat_f32", self.rotation.wht_mat.float().contiguous())
        self.final_scale_val = self.rot_final_scale.item()

    def transform_query(self, query: torch.Tensor) -> torch.Tensor:
        # SOTA v9.4: Zero-Dispatch Transform
        orig_shape = query.shape
        if orig_shape[-2] == 1:
            query = query.squeeze(-2)
        
        if self.padded:
            query = F.pad(query.float(), (0, self.block_size * self.n_subblocks - self.dim))
        else:
            query = query.float()
        
        # SOTA: In-place-ish reshape with minimalist contiguous call
        # We only call contiguous() once for the whole pipeline
        q_flattened = query.reshape(-1, self.block_size)
        if not q_flattened.is_contiguous():
            q_flattened = q_flattened.contiguous()
            
        q_rot = self.rotation(q_flattened)
        return q_rot.reshape(orig_shape)

    def quantize(self, x: torch.Tensor, pack: bool = False, precomputed_norms: torch.Tensor = None, precomputed_scales: torch.Tensor = None) -> MSEQuantized:
        # SOTA v10.1: Optimized Fused Triton Path
        if x.is_cuda and x.shape[-1] == self.block_size:


            orig_shape = x.shape
            x_flat = x.float().reshape(-1, self.dim)
            
            orig_shape = x.shape
            x_flat = x.float().reshape(-1, self.dim)
            
            # SOTA v12.5: Zero-Allocation Cache for CUDA Graphs
            if not hasattr(self, "_static_out_indices") or self._static_out_indices.shape[0] < x_flat.shape[0]:
                vals_per_byte = 2 if pack else 1
                packed_d = self.dim // vals_per_byte
                self._static_out_indices = torch.empty((x_flat.shape[0], packed_d), dtype=torch.uint8, device=x.device)
                self._static_out_scales = torch.empty((x_flat.shape[0], 1), dtype=torch.float32, device=x.device)
                self._static_out_norms = torch.empty((x_flat.shape[0], 1), dtype=torch.float32, device=x.device)
                self._static_out_rotated = torch.empty((x_flat.shape[0], self.dim), dtype=torch.float32, device=x.device)

            indices_raw, scales_raw, norms_raw, rotated_raw = fused_quantize(
                x_flat, 
                self.rotation.all_signs,
                self.wht_mat_f32,
                self.bits, 
                self.max_centroid, 
                self.final_scale_val,
                dist_type=self.dist,
                pack=pack,
                out_indices=self._static_out_indices,
                out_scales=self._static_out_scales,
                out_norms=self._static_out_norms,
                out_rotated=self._static_out_rotated
            )
            
            # SOTA v12.6: Slice static buffers to current token count (Zero-Copy)
            n_tokens = x_flat.shape[0]
            indices = indices_raw[:n_tokens]
            scales = scales_raw[:n_tokens]
            norms = norms_raw[:n_tokens]
            rotated = rotated_raw[:n_tokens]




            
            meta_shape = orig_shape[:-1] + (self.n_subblocks,)
            packed_d = indices.shape[-1]
            out_shape = orig_shape[:-1] + (packed_d,)
            
            return MSEQuantized(
                indices=indices.reshape(out_shape),
                norms=norms.reshape(meta_shape),
                scales=scales.reshape(meta_shape),
                bits=self.bits,
                packed=pack,
                rotated_tensor=rotated.reshape(orig_shape)
            )

        # Legacy/CPU Fallback
        shape = x.shape[:-1]
        dtype = x.dtype
        
        # 1. Flatten into blocks with a single contiguous check
        x_fp32 = x.float()
        if self.padded:
            x_fp32 = F.pad(x_fp32, (0, self.block_size * self.n_subblocks - self.dim))
        
        x_reshaped = x_fp32.reshape(-1, self.block_size)
        if not x_reshaped.is_contiguous():
            x_reshaped = x_reshaped.contiguous()
        
        # 2. Vector Norms
        if precomputed_norms is not None:
            vec_norms = precomputed_norms.reshape(-1, 1)
        else:
            vec_norms = torch.norm(x_reshaped, p=2, dim=-1, keepdim=True) + self.epsilon
            
        x_unit_reshaped = x_reshaped / vec_norms
        x_rot_reshaped = self.rotation(x_unit_reshaped)
        
        if precomputed_scales is not None:
            refined_gamma = precomputed_scales.reshape(-1, 1)
            x_normalized = x_rot_reshaped / (refined_gamma + self.epsilon)
            indices = lloyd_max_quantize(x_normalized, self.bits, dist=self.dist)
        else:
            from .lloyd_max import compute_lloyd_max_codebook
            cb = compute_lloyd_max_codebook(self.bits, d=1, dist=self.dist)
            max_c = cb['max_centroid']
            
            x_rot_max = torch.max(torch.abs(x_rot_reshaped), dim=-1, keepdim=True).values
            rms_scales = (x_rot_max / max_c).to(dtype) 
            x_normalized = x_rot_reshaped / (rms_scales + self.epsilon)
            
            if x.shape[0] == 1 or x_reshaped.shape[0] <= 32:
                indices = lloyd_max_quantize(x_normalized, self.bits, dist=self.dist)
                refined_gamma = rms_scales
            else:
                x_rot_f32 = x_rot_reshaped.float()
                indices_tmp = lloyd_max_quantize(x_normalized, self.bits, dist=self.dist)
                recon_u = lloyd_max_dequantize(indices_tmp, self.bits, dist=self.dist).float()
                num = (x_rot_f32 * recon_u).sum(dim=-1, keepdim=True)
                den = (recon_u * recon_u).sum(dim=-1, keepdim=True) + self.epsilon
                gamma_pass1 = (num / den).to(dtype)
                indices = lloyd_max_quantize(x_rot_reshaped / (gamma_pass1 + self.epsilon), self.bits, dist=self.dist)
                x_rot_mean = torch.mean(torch.abs(x_rot_f32), dim=-1, keepdim=True) + self.epsilon
                is_spiky = (x_rot_max / x_rot_mean) > 5.0
                refined_gamma = torch.where(is_spiky, rms_scales, gamma_pass1)
                if is_spiky.any():
                    indices = torch.where(is_spiky, indices_tmp, indices)

        # Final packaging for fallback
        x_rot_out = x_rot_reshaped.view(shape + (-1,))
        meta_shape = shape + (self.n_subblocks,)
        indices_out = indices.view(shape + (-1,))
        if pack:
            from .quant_base import pack_indices
            indices_out = pack_indices(indices_out, self.bits)
            
        return MSEQuantized(
            indices=indices_out, 
            norms=vec_norms.view(meta_shape), 
            scales=refined_gamma.view(meta_shape), 
            bits=self.bits, 
            packed=pack,
            rotated_tensor=x_rot_out
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
        # SOTA v12.7: Leverage zero-allocation path for CUDA Graphs
        mse_q = self.quantize(x, pack=pack, precomputed_norms=precomputed_norms, precomputed_scales=precomputed_scales)
        
        # 1. Zero-Allocation Reconstruction
        # SOTA v12.7: Pre-allocate static recon buffer
        n_tokens = x.reshape(-1, self.block_size).shape[0]
        if not hasattr(self, "_static_recon") or self._static_recon.shape[0] < n_tokens:
            self._static_recon = torch.empty((self._static_out_indices.shape[0], self.block_size), dtype=torch.float32, device=x.device)
            self._static_residual = torch.empty((self._static_out_indices.shape[0], self.block_size), dtype=x.dtype, device=x.device)

        # Unpack indices for reconstruction (Matches v12.6 slicing)

        indices = unpack_indices(mse_q.indices, self.bits, self.block_size * self.n_subblocks) if pack else mse_q.indices
        
        # SOTA: In-place Reconstruction using pre-allocated centroids (lloyd_max)
        # We assume lloyd_max_dequantize is being called on the sliced indices
        reconstructed_unit = lloyd_max_dequantize(indices, self.bits, dist='gaussian')
        
        # 2. Reconstruct rotated blocks using metadata
        scales = mse_q.scales.float().reshape(-1, 1)
        recon_rotated = self._static_recon[:n_tokens]
        torch.mul(reconstructed_unit.view(-1, self.block_size), scales, out=recon_rotated)
        
        # 3. Get actual rotated blocks (Zero-Copy slicing from v12.6)
        x_rotated_blocks = mse_q.rotated_tensor.view(-1, self.block_size)
        
        # 4. In-place Residual Calculation (CRITICAL for CUDA Graphs)
        residual = self._static_residual[:n_tokens]
        torch.sub(x_rotated_blocks, recon_rotated.to(x.dtype), out=residual)
        
        return mse_q, residual


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
        
        # SOTA v12.7: Ensure correct 4D shapes for Cache Manager
        orig_heads_seq = x.shape[:-1]
        n_tokens = residual_rotated.shape[0]
        
        # SOTA v12.7: Zero-Allocation static sign bits
        if not hasattr(self, "_static_sign_bits") or self._static_sign_bits.shape[0] < n_tokens:
            self._static_sign_bits = torch.empty((self.mse_quantizer._static_out_indices.shape[0], self.block_size), dtype=torch.uint8, device=x.device)
            # Pre-allocate meta buffer: [norms, scales, residual_norms] -> size 3
            self._static_meta = torch.empty((self.mse_quantizer._static_out_indices.shape[0], 3), dtype=torch.float32, device=x.device)

        # In-place Sign Check
        qjl_sign_bits_raw = self._static_sign_bits[:n_tokens]
        torch.ge(residual_rotated, 0, out=qjl_sign_bits_raw)
        
        if precomputed_res_norms is not None:
            residual_norms = precomputed_res_norms
        else:
            res_reshaped = residual_rotated.float()
            res_norms_unit = torch.norm(res_reshaped, p=2, dim=-1, keepdim=True)
            residual_norms = res_norms_unit * mse_q.norms.contiguous().view(-1, 1)

        mse_indices = mse_q.indices
        final_qjl_signs = qjl_sign_bits_raw
        
        if pack:
            from .quant_base import pack_indices
            mse_indices = pack_indices(mse_indices.view(-1, self.block_size), self.mse_bits).view(orig_heads_seq + (-1,))
            final_qjl_signs = pack_indices(qjl_sign_bits_raw, 1).view(orig_heads_seq + (-1,))
        else:
            final_qjl_signs = qjl_sign_bits_raw.view(orig_heads_seq + (-1,))

        # SOTA v12.7: Pre-pack metadata into static buffer (No Allocation)
        meta = self._static_meta[:n_tokens].view(orig_heads_seq + (-1,))
        meta[..., 0:1].copy_(mse_q.norms.float().view(orig_heads_seq + (1,)))
        meta[..., 1:2].copy_(mse_q.scales.float().view(orig_heads_seq + (1,)))
        meta[..., 2:3].copy_(residual_norms.float().view(orig_heads_seq + (1,)))

        return ProdQuantized(
            mse_indices=mse_indices,
            qjl_signs=final_qjl_signs,
            scales=mse_q.scales.view(orig_heads_seq + (-1,)),
            residual_norms=residual_norms.view(orig_heads_seq + (-1,)),
            norms=mse_q.norms.view(orig_heads_seq + (-1,)),
            mse_bits=self.mse_bits,
            packed=pack,
            meta=meta,
            rotated_tensor=mse_q.rotated_tensor
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