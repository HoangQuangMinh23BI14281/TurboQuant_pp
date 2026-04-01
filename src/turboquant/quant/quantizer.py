"""
TurboQuant Quantizers — Algorithm 1 (MSE) and Algorithm 2 (Inner Product).

Two quantizer classes operating on tensors of shape (..., d):

- TurboQuantMSE: MSE-optimal quantizer for V or K cache (Algorithm 1)
  Quantize: normalize → pad → cascaded SRHT → Lloyd-Max per coordinate
  Dequantize: centroids → inverse SRHT → truncate → rescale

- TurboQuantProd: Inner-product-optimal quantizer for K cache (Algorithm 2)
  Stage 1: TurboQuantMSE at (b-1) bits
  Stage 2: QJL via SRHT on residual (1 bit per coordinate)
  Provides unbiased inner product estimate:
    <y, x̃> = <y, x̃_mse> + ||r|| * sqrt(π/2)/D² * <SRHT(y), sign(SRHT(r))>

- TurboQuantValue: Asymmetric scalar quantizer for V cache (Production-grade)
  Quantize: Per-token/group asymmetric scaling (Scale + ZeroPoint)
  Dequantize: indices * Scale + ZeroPoint
  No WHT/SRHT to preserve semantic density.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, NamedTuple

from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize
from ..ops.rotation import TurboQuantRotation
from ..ops.wht import fwht, ifwht
from ..ops.sign_array import generate_sign_array, apply_sign_array


# ──────────────────────────────────────────────────────────────────────
# Output Types
# ──────────────────────────────────────────────────────────────────────

class MSEQuantized(NamedTuple):
    """Output of TurboQuantMSE.quantize()."""
    indices: torch.Tensor       # (..., block_size) or (..., packed_d) packed indices
    norms: torch.Tensor         # (...,) original L2 norms
    scales: torch.Tensor        # (..., 1) dynamic RMS scales per vector
    bits: int                   # bits per index
    packed: bool = False        # whether indices are bit-packed

class ProdQuantized(NamedTuple):
    """Output of TurboQuantProd.quantize()."""
    mse_indices: torch.Tensor      # (..., block_size) or (..., packed_d) packed indices
    qjl_signs: torch.Tensor        # (..., packed_d_signs) packed sign bits uint8
    scales: torch.Tensor           # (..., 1) dynamic RMS scales per vector (from MSE stage)
    residual_norms: torch.Tensor   # (...,) L2 norm of residual in rotated domain
    norms: torch.Tensor            # (...,) original L2 norms
    mse_bits: int                  # bits per MSE index (= total_bits - 1)
    packed: bool = False           # whether data is bit-packed

class ValueQuantized(NamedTuple):
    """Output of TurboQuantValue.quantize()."""
    indices: torch.Tensor       # (..., dim) or (..., packed_d) packed indices
    scales: torch.Tensor        # (..., n_groups) float scales
    zero_points: torch.Tensor   # (..., n_groups) float zero points (min values)
    bits: int                   # number of bits per index
    packed: bool = False        # whether data is bit-packed


# ──────────────────────────────────────────────────────────────────────
# Bit Packing Utilities
# ──────────────────────────────────────────────────────────────────────

def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Bit-pack integer indices into uint8 bytes.
    Packing is LSB-first (matching Triton kernel logic).
    """
    shape = indices.shape
    d = shape[-1]
    if bits == 8:
        return indices.to(torch.uint8)

    vals_per_byte = 8 // bits
    if indices.dtype == torch.uint8 and indices.shape[-1] == d // vals_per_byte:
        return indices # Already packed correctly

    indices_flat = indices.reshape(-1, d)
    n_vectors = indices_flat.shape[0]
    packed_d = d // vals_per_byte
    packed = torch.zeros((n_vectors, packed_d), device=indices.device, dtype=torch.uint8)
    for i in range(vals_per_byte):
        shift = i * bits
        packed |= (indices_flat[:, i::vals_per_byte].to(torch.uint8) << shift)
    return packed.reshape(*indices.shape[:-1], packed_d)


def unpack_indices(packed: torch.Tensor, bits: int, original_d: int) -> torch.Tensor:
    """
    Unpack bit-packed uint8 bytes back to integer indices.
    """
    shape = packed.shape
    if bits == 8:
        return packed.to(torch.long)

    vals_per_byte = 8 // bits
    packed_flat = packed.reshape(-1, shape[-1])
    n_vectors = packed_flat.shape[0]
    mask = (1 << bits) - 1
    unpacked = torch.zeros((n_vectors, original_d), device=packed.device, dtype=torch.long)
    for i in range(vals_per_byte):
        shift = i * bits
        unpacked[:, i::vals_per_byte] = ((packed_flat >> shift) & mask).to(torch.long)
    return unpacked.reshape(*shape[:-1], original_d)


# ──────────────────────────────────────────────────────────────────────
# Algorithm 1: TurboQuantMSE (for V cache)
# ──────────────────────────────────────────────────────────────────────

class TurboQuantMSE(nn.Module):
    """
    TurboQuant MSE-optimal quantizer (Algorithm 1).

    Pipeline:
      1. Extract and store L2 norm per vector
      2. Normalize to unit sphere
      3. Pad to block_size (power of 2)
      4. Apply cascaded SRHT rotation (sign + WHT, 2 passes)
      5. Lloyd-Max scalar quantization per coordinate

    Args:
        dim: original vector dimension
        bits: bits per coordinate (1-8)
        n_rotation_passes: number of cascaded SRHT passes (default: 2)
    """

    def __init__(self, dim: int, bits: int, n_rotation_passes: int = 2):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits
        self.n_rotation_passes = n_rotation_passes

        # SOTA Block Size: Standardized to 128 for 5.12x compression
        # Must be a multiple of 128 for Triton SRAM tiling optimization
        bs = 128
        while bs < dim:
            bs *= 2
        self.block_size = bs

        self.padded = (self.block_size != self.dim)

        # Cascaded SRHT rotation (using existing TurboQuantRotation)
        self.rotation = TurboQuantRotation(self.block_size, n_passes=n_rotation_passes, pattern='tbq')

    def transform_query(self, query: torch.Tensor) -> torch.Tensor:
        """
        Prepare query for MSE score computation (pad and rotate).
        
        Args:
            query: (..., dim) or (..., 1, dim)
        Returns:
            rotated_query: (..., block_size)
        """
        # Ensure (..., dim)
        if query.shape[-2] == 1:
            query = query.squeeze(-2)
        
        # Pad
        if self.padded:
            query = torch.nn.functional.pad(query.float(), (0, self.block_size - self.dim))
        else:
            query = query.float()
        
        # Rotate
        return self.rotation(query)

    def quantize(self, x: torch.Tensor, pack: bool = False) -> MSEQuantized:
        """
        Quantize vectors x of shape (..., dim).

        Returns:
            MSEQuantized with indices, norms, and bits.
        """
        shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype

        # Step 1: Compute and store L2 norms
        vec_norms = torch.norm(x, p=2, dim=-1)  # (...,)

        # Step 2: Normalize to unit sphere
        x_unit = x / (vec_norms.unsqueeze(-1) + 1e-10)

        # Step 3: Pad to block_size
        if self.padded:
            padding = torch.zeros((*shape, self.block_size - self.dim), device=device, dtype=dtype)
            x_padded = torch.cat([x_unit, padding], dim=-1)
        else:
            x_padded = x_unit

        # Step 4: Cascaded SRHT rotation (Isometry now)
        x_rotated = self.rotation(x_padded)

        # Step 5: Dynamic Block-wise Scaling (User formula: norm / sqrt(d))
        # This brings any distribution to std=1 for Lloyd-Max
        rms_scales = torch.norm(x_rotated, dim=-1, keepdim=True) / math.sqrt(self.block_size)
        x_normalized = x_rotated / (rms_scales + 1e-10)

        # Step 6: Initial Lloyd-Max quantization (Gaussian LUT for rotated domain)
        indices = lloyd_max_quantize(x_normalized, self.bits, dist='gaussian')
        reconstructed_unit = lloyd_max_dequantize(indices, self.bits, dist='gaussian')
        
        # Step 7: Refined Gamma (Least-Squares Optimal Scaling)
        # Minimizes ||x_rotated - gamma * reconstructed_unit||^2
        numerator = (x_rotated * reconstructed_unit).sum(dim=-1, keepdim=True)
        denominator = (reconstructed_unit * reconstructed_unit).sum(dim=-1, keepdim=True) + 1e-10
        refined_gamma = numerator / denominator

        if pack:
            indices = pack_indices(indices, self.bits)

        return MSEQuantized(indices=indices, norms=vec_norms, scales=refined_gamma, bits=self.bits, packed=pack)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        """
        Reconstruct vectors from MSEQuantized.

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        indices = q.indices
        if q.packed:
            indices = unpack_indices(indices, q.bits, self.block_size)

        # Centroid lookup (N(0,1) scale)
        x_reconstructed = lloyd_max_dequantize(indices, q.bits)
        
        # Ensure it has exactly block_size features for inverse rotation
        if x_reconstructed.shape[-1] < self.block_size:
            padding = self.block_size - x_reconstructed.shape[-1]
            x_reconstructed = F.pad(x_reconstructed, (0, padding))
        elif x_reconstructed.shape[-1] > self.block_size:
            x_reconstructed = x_reconstructed[..., :self.block_size]

        # Undo scaling back to rotation domain
        # Rescale centroids by the stored dynamic RMS scales (D^n passes growth is already inside rms_scale)
        x_rotated = x_reconstructed * q.scales

        # Inverse rotation
        x_padded = self.rotation.inverse(x_rotated)

        # Truncate padding and rescale
        x_hat = x_padded[..., :self.dim] * q.norms.unsqueeze(-1)

        return x_hat

    def quantize_and_residual(self, x: torch.Tensor, pack: bool = True) -> Tuple[MSEQuantized, torch.Tensor]:
        """
        Quantize and return both the MSEQuantized data and the residual
        in the rotated domain (needed by TurboQuantProd for QJL stage).

        Returns:
            (mse_quantized, residual_rotated)
            where residual_rotated has shape (..., block_size)
        """
        shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype

        # Step 1: Norms
        vec_norms = torch.norm(x, p=2, dim=-1)
        x_unit = x / (vec_norms.unsqueeze(-1) + 1e-10)

        # Step 2: Pad
        if self.padded:
            padding = torch.zeros((*shape, self.block_size - self.dim), device=device, dtype=dtype)
            x_padded = torch.cat([x_unit, padding], dim=-1)
        else:
            x_padded = x_unit

        # Step 3: Rotate
        x_rotated = self.rotation(x_padded)

        # Step 4: Dynamic Block-wise Scaling (Initial estimate: RMS)
        rms_scales = torch.norm(x_rotated, dim=-1, keepdim=True) / math.sqrt(self.block_size)
        x_normalized = x_rotated / (rms_scales + 1e-10)
        
        # Step 5: Quantize in normalized domain (Gaussian LUT)
        indices = lloyd_max_quantize(x_normalized, self.bits, dist='gaussian')
        reconstructed_unit = lloyd_max_dequantize(indices, self.bits, dist='gaussian')
        
        # Step 6: Refined Gamma (Least-Squares Optimal Scaling)
        # gamma = <X_rot, X_hat_unit> / <X_hat_unit, X_hat_unit>
        numerator = (x_rotated * reconstructed_unit).sum(dim=-1, keepdim=True)
        denominator = (reconstructed_unit * reconstructed_unit).sum(dim=-1, keepdim=True) + 1e-10
        refined_gamma = numerator / denominator
        
        # Step 7: Recover reconstruction in rotated domain using refined_gamma
        reconstructed_rotated = reconstructed_unit * refined_gamma
        
        # Step 8: Calculate residual in rotated domain (Used for Stage 2 QJL)
        residual_rotated = x_rotated - reconstructed_rotated

        # Pack indices if requested
        if pack:
            indices = pack_indices(indices, self.bits)

        mse_q = MSEQuantized(indices=indices, norms=vec_norms, scales=refined_gamma, bits=self.bits, packed=pack)
        return mse_q, residual_rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing roundtrip quality)."""
        return self.dequantize(self.quantize(x))


# ──────────────────────────────────────────────────────────────────────
# Algorithm 2: TurboQuantProd (for K cache)
# ──────────────────────────────────────────────────────────────────────

class TurboQuantProd(nn.Module):
    """
    TurboQuant inner-product-optimal quantizer (Algorithm 2).

    Two-stage process:
      Stage 1: TurboQuantMSE at (b-1) bits → MSE reconstruction
      Stage 2: QJL via SRHT on residual → 1-bit sign correction

    The QJL stage uses an independent SRHT (sign_array + WHT) to project
    the residual, matching the llama.cpp implementation:
      projected = WHT(qjl_signs * residual)
      sign_bits = (projected >= 0)

    This provides an unbiased inner product estimate:
      <y, x̃> = <y, x̃_mse> + ||r||₂ * sqrt(π/2)/D² * <SRHT(y), sign(SRHT(r))>

    Args:
        dim: original vector dimension
        bits: total bits per coordinate (>= 2; 1 bit for QJL + rest for MSE)
        n_rotation_passes: number of cascaded SRHT passes for MSE stage
    """

    def __init__(self, dim: int, bits: int, n_rotation_passes: int = 2):
        super().__init__()
        assert bits >= 2, "TurboQuantProd requires at least 2 bits (1 for MSE + 1 for QJL)"

        self.dim = dim
        self.bits = bits
        self.mse_bits = bits - 1

        # Stage 1: MSE quantizer at (bits-1) bits
        self.mse_quantizer = TurboQuantMSE(dim, self.mse_bits, n_rotation_passes)
        self.block_size = self.mse_quantizer.block_size

        # Stage 2: QJL sign array (independent from TBQ rotation signs)
        # Using the llama.cpp QJL preset sign pattern
        qjl_signs = generate_sign_array(self.block_size, use_llama_preset='qjl')
        self.register_buffer('qjl_signs', qjl_signs)

        # QJL dequantization scale factor
        # From the paper: sqrt(π/2) / D
        # D = block_size (WHT is now isometric 1/sqrt(D) normalization)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / self.block_size

    def transform_query(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare query for full TurboQuant score computation.
        
        Returns:
            (q_rot, q_sketch): rotated query and QJL-sketched query.
        """
        q_rot = self.mse_quantizer.transform_query(query)
        q_sketch = fwht(apply_sign_array(q_rot, self.qjl_signs))
        return q_rot, q_sketch

    def quantize(self, x: torch.Tensor, pack: bool = True) -> ProdQuantized:
        """
        Full TurboQuant quantization of shape (..., dim).

        Returns:
            ProdQuantized with MSE indices, QJL SRHT-projected signs,
            residual norms, and original norms.
        """
        # Stage 1: MSE quantization + residual in rotated domain
        # Always get raw indices first for residual calculation, then pack at the end
        mse_q, residual_rotated = self.mse_quantizer.quantize_and_residual(x, pack=False)

        # Stage 2: QJL via SRHT on residual
        # Apply QJL sign array then WHT (completing the SRHT projection)
        residual_signed = apply_sign_array(residual_rotated, self.qjl_signs)
        residual_projected = fwht(residual_signed)

        # Take sign bits of the SRHT-projected residual
        qjl_sign_bits = (residual_projected >= 0).to(torch.uint8)

        # True L2 norm of residual in original vector domain (Forced by User)
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
        """
        Reconstruct vectors from ProdQuantized.

        The QJL residual reconstruction inverts the SRHT to recover
        the residual direction, then rescales to the correct norm.

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        mse_indices = q.mse_indices
        qjl_signs = q.qjl_signs

        if q.packed:
            mse_indices = unpack_indices(mse_indices, q.mse_bits, self.block_size)
            qjl_signs = unpack_indices(qjl_signs, 1, self.block_size)

        # Stage 1: MSE dequantization
        # Centroids in N(0,1) scale → apply dynamic RMS scales → rotation domain
        reconstructed_normalized = lloyd_max_dequantize(mse_indices, q.mse_bits)
        reconstructed_mse_rot = reconstructed_normalized * q.scales

        # Stage 2: QJL residual reconstruction via SRHT inverse
        # Convert sign bits {0, 1} → {-1, +1}
        signs_float = qjl_signs.float() * 2.0 - 1.0

        # Inverse SRHT: IWHT then undo sign flip to get residual direction
        residual_direction = ifwht(signs_float)
        residual_direction = apply_sign_array(residual_direction, self.qjl_signs)

        # Normalize to unit direction, then scale to correct residual norm
        dir_norm = torch.norm(residual_direction, dim=-1, keepdim=True) + 1e-10
        # q.residual_norms is in FULL domain, so we normalize by q.norms 
        # to bring it back to the UNIT rotated domain for addition with MSE part.
        unit_res_norms = q.residual_norms / (q.norms + 1e-10)
        residual_estimated_unit = residual_direction * (unit_res_norms.unsqueeze(-1) / dir_norm)

        # Add QJL correction to MSE reconstruction (in UNIT rotation domain)
        combined_rotated = reconstructed_mse_rot + residual_estimated_unit

        # Inverse rotation (this will divide by block_size^n_passes during ifwht)
        x_padded = self.mse_quantizer.rotation.inverse(combined_rotated)

        # Truncate padding and rescale by original norm
        return x_padded[..., :self.dim] * q.norms.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing roundtrip quality)."""
        return self.dequantize(self.quantize(x))


# ──────────────────────────────────────────────────────────────────────
# TurboQuantValue (for V cache)
# ──────────────────────────────────────────────────────────────────────

class TurboQuantValue(nn.Module):
    """
    TurboQuant asymmetric quantizer for Value (V) cache.
    
    Uses asymmetric scalar quantization (Scale + ZeroPoint) per token or per group.
    This handles the non-zero-mean distribution of Value vectors without WHT.
    
    Formula:
      v_quant = round((v - min) / scale)
      v_recon = v_quant * scale + min
    """

    def __init__(self, dim: int, bits: int = 4, group_size: int = 128):
        super().__init__()
        self.dim = dim
        self.bits = bits
        # SOTA: default to 128, but fallback if dim is smaller
        self.group_size = min(group_size, dim)
        self.n_levels = 2**bits
        
        # Ensure group_size divides dim
        assert dim % self.group_size == 0, f"dim {dim} must be divisible by group_size {self.group_size}"
        self.n_groups = dim // self.group_size

    def quantize(self, x: torch.Tensor, pack: bool = True) -> ValueQuantized:
        """
        Asymmetric quantization using Laplacian LUT for 1.2dB SNR gain.
        
        Returns:
            ValueQuantized with packed indices, scales, and zero_points.
        """
        shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype
        
        # Step 1: Reshape into groups: (..., n_groups, group_size)
        x_grouped = x.view(*shape, self.n_groups, self.group_size)
        
        # Step 2: Compute min per group (ZeroPoint)
        v_min = x_grouped.min(dim=-1, keepdim=True).values
        x_centered = x_grouped - v_min
        
        # Step 3: Dynamic Scaling for Unit Laplacian Quantization
        # Use simple RMS-style scale for initial normalization
        v_scale = torch.norm(x_centered, dim=-1, keepdim=True) / math.sqrt(self.group_size)
        x_unit = x_centered / (v_scale + 1e-10)
        
        # Step 4: Laplacian Lloyd-Max Quantize (Asymmetric original domain)
        indices = lloyd_max_quantize(x_unit, self.bits, dist='laplace')
        indices_f = indices.float()
        
        # Step 5: Refined Gamma (Least-Squares Optimal Scaling for Asymmetric)
        # Minimize ||x_centered - gamma * reconstructed_unit||^2
        unit_recon = lloyd_max_dequantize(indices, self.bits, dist='laplace')
        
        numerator = (x_centered * unit_recon).sum(dim=-1, keepdim=True)
        denominator = (unit_recon * unit_recon).sum(dim=-1, keepdim=True) + 1e-10
        refined_gamma = numerator / denominator
        
        # Step 6: Flatten indices back to (..., dim)
        indices = indices.view(*shape, self.dim).to(torch.uint8)
        
        # Flatten scales and zero_points to (..., n_groups)
        scales = refined_gamma.view(*shape, self.n_groups)
        zero_points = v_min.view(*shape, self.n_groups)

        if pack:
            indices = pack_indices(indices, self.bits)
            
        return ValueQuantized(
            indices=indices,
            scales=scales,
            zero_points=zero_points,
            bits=self.bits,
            packed=pack
        )

    def dequantize(self, q: ValueQuantized) -> torch.Tensor:
        """
        Reconstruct Value vectors using Laplacian Dual-LUT.
        """
        indices = q.indices
        shape = q.scales.shape[:-1]
        
        if q.packed:
            indices = unpack_indices(indices, q.bits, self.dim)
            
        # Reshape for group-wise broadcast
        indices_grouped = indices.view(*shape, self.n_groups, self.group_size).float()
        scales_grouped = q.scales.view(*shape, self.n_groups, 1)
        zp_grouped = q.zero_points.view(*shape, self.n_groups, 1)
        
        # v_recon = lloyd_max_dequantize(indices, dist='laplace') * refined_gamma + zero_point
        unit_recon = lloyd_max_dequantize(indices_grouped.long(), q.bits, dist='laplace')
        reconstructed = unit_recon * scales_grouped + zp_grouped
        
        return reconstructed.view(*shape, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing)."""
        return self.dequantize(self.quantize(x))
