"""
TurboQuant Quantizers — Algorithm 1 (MSE) and Algorithm 2 (Inner Product).

Two quantizer classes operating on tensors of shape (..., d):

- TurboQuantMSE: MSE-optimal quantizer for V cache (Algorithm 1)
  Quantize: normalize → pad → cascaded SRHT → Lloyd-Max per coordinate
  Dequantize: centroids → inverse SRHT → truncate → rescale

- TurboQuantProd: Inner-product-optimal quantizer for K cache (Algorithm 2)
  Stage 1: TurboQuantMSE at (b-1) bits
  Stage 2: QJL via SRHT on residual (1 bit per coordinate)
  Provides unbiased inner product estimate:
    <y, x̃> = <y, x̃_mse> + ||r|| * sqrt(π/2)/D² * <SRHT(y), sign(SRHT(r))>
"""

import math
import torch
import torch.nn as nn
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
    bits: int                   # bits per index
    packed: bool = False        # whether indices are bit-packed

class ProdQuantized(NamedTuple):
    """Output of TurboQuantProd.quantize()."""
    mse_indices: torch.Tensor      # (..., block_size) or (..., packed_d) packed indices
    qjl_signs: torch.Tensor        # (..., packed_d_signs) packed sign bits uint8
    residual_norms: torch.Tensor   # (...,) L2 norm of residual in rotated domain
    norms: torch.Tensor            # (...,) original L2 norms
    mse_bits: int                  # bits per MSE index (= total_bits - 1)
    packed: bool = False           # whether data is bit-packed


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
    indices_flat = indices.reshape(-1, d)
    n_vectors = indices_flat.shape[0]
    packed_d = d // vals_per_byte
    packed = torch.zeros((n_vectors, packed_d), device=indices.device, dtype=torch.uint8)
    for i in range(vals_per_byte):
        shift = i * bits
        packed |= (indices_flat[:, i::vals_per_byte].to(torch.uint8) << shift)
    return packed.reshape(*shape[:-1], packed_d)


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

        # Block size: next power of 2 >= dim, minimum 256
        if dim <= 256:
            self.block_size = 256
        else:
            bs = 256
            while bs < dim:
                bs *= 2
            self.block_size = bs

        self.padded = (self.block_size != self.dim)

        # Cascaded SRHT rotation (using existing TurboQuantRotation)
        self.rotation = TurboQuantRotation(self.block_size, n_passes=n_rotation_passes, pattern='tbq')

        # After n passes of unnormalized WHT, each component has:
        #   std ≈ block_size^((n_passes - 1) / 2)
        # We scale to std≈1 so the exact N(0,1) Lloyd-Max static tables apply.
        self.rotation_scale = float(self.block_size ** ((n_rotation_passes - 1) / 2.0))

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

        # Step 4: Cascaded SRHT rotation
        x_rotated = self.rotation(x_padded)

        # Step 5: Scale to std≈1 for optimal Lloyd-Max quantization
        x_scaled = x_rotated / self.rotation_scale

        # Step 6: Lloyd-Max quantization using exact N(0,1) static tables
        indices = lloyd_max_quantize(x_scaled, self.bits)

        if pack:
            indices = pack_indices(indices, self.bits)

        return MSEQuantized(indices=indices, norms=vec_norms, bits=self.bits, packed=pack)

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
        x_scaled = lloyd_max_dequantize(indices, q.bits)

        # Undo scaling back to rotation domain
        x_rotated = x_scaled * self.rotation_scale

        # Inverse rotation
        x_padded = self.rotation.inverse(x_rotated)

        # Truncate padding and rescale
        x_hat = x_padded[..., :self.dim] * q.norms.unsqueeze(-1)

        return x_hat

    def quantize_and_residual(self, x: torch.Tensor) -> Tuple[MSEQuantized, torch.Tensor]:
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

        # Steps 1-4: same as quantize
        vec_norms = torch.norm(x, p=2, dim=-1)
        x_unit = x / (vec_norms.unsqueeze(-1) + 1e-10)
        if self.padded:
            padding = torch.zeros((*shape, self.block_size - self.dim), device=device, dtype=dtype)
            x_padded = torch.cat([x_unit, padding], dim=-1)
        else:
            x_padded = x_unit
        x_rotated = self.rotation(x_padded)

        # Scale to std≈1 for Lloyd-Max
        x_scaled = x_rotated / self.rotation_scale

        # Step 5: Quantize + dequantize in scaled domain
        indices = lloyd_max_quantize(x_scaled, self.bits)
        reconstructed_scaled = lloyd_max_dequantize(indices, self.bits)

        # Residual in scaled domain (same domain as quantized data)
        residual_scaled = x_scaled - reconstructed_scaled

        # Convert residual back to rotation domain for QJL
        residual_rotated = residual_scaled * self.rotation_scale

        mse_q = MSEQuantized(indices=indices, norms=vec_norms, bits=self.bits)
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
        # From the paper: sqrt(π/2) / D²
        # D = block_size (WHT normalization scales by 1/D in inverse)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / (self.block_size * self.block_size)

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
        mse_q, residual_rotated = self.mse_quantizer.quantize_and_residual(x)

        # Stage 2: QJL via SRHT on residual
        # Apply QJL sign array then WHT (completing the SRHT projection)
        residual_signed = apply_sign_array(residual_rotated, self.qjl_signs)
        residual_projected = fwht(residual_signed)

        # Take sign bits of the SRHT-projected residual
        qjl_sign_bits = (residual_projected >= 0).to(torch.uint8)

        # True L2 norm of residual
        residual_norms = torch.norm(residual_rotated, dim=-1)

        mse_indices = mse_q.indices
        if pack:
            mse_indices = pack_indices(mse_indices, self.mse_bits)
            qjl_sign_bits = pack_indices(qjl_sign_bits, 1)

        return ProdQuantized(
            mse_indices=mse_indices,
            qjl_signs=qjl_sign_bits,
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
        rotation_scale = self.mse_quantizer.rotation_scale

        mse_indices = q.mse_indices
        qjl_signs = q.qjl_signs
        if q.packed:
            mse_indices = unpack_indices(mse_indices, q.mse_bits, self.block_size)
            qjl_signs = unpack_indices(qjl_signs, 1, self.block_size)

        # Stage 1: MSE dequantization
        # Centroids in N(0,1) scale → multiply by rotation_scale → rotation domain
        reconstructed_scaled = lloyd_max_dequantize(mse_indices, q.mse_bits)
        reconstructed_rotated = reconstructed_scaled * rotation_scale

        # Stage 2: QJL residual reconstruction via SRHT inverse
        # Convert sign bits {0, 1} → {-1, +1}
        signs_float = qjl_signs.float() * 2.0 - 1.0

        # Inverse SRHT: IWHT then undo sign flip to get residual direction
        residual_direction = ifwht(signs_float)
        residual_direction = apply_sign_array(residual_direction, self.qjl_signs)

        # Normalize to unit direction, then scale to correct residual norm
        dir_norm = torch.norm(residual_direction, dim=-1, keepdim=True) + 1e-10
        residual_estimated = residual_direction * (q.residual_norms.unsqueeze(-1) / dir_norm)

        # Add QJL correction to MSE reconstruction (in rotation domain)
        combined_rotated = reconstructed_rotated + residual_estimated

        # Inverse rotation
        x_padded = self.mse_quantizer.rotation.inverse(combined_rotated)

        # Truncate padding and rescale by original norm
        return x_padded[..., :self.dim] * q.norms.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (for testing roundtrip quality)."""
        return self.dequantize(self.quantize(x))
