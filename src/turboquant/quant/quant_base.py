import torch
from typing import NamedTuple

# ──────────────────────────────────────────────────────────────────────
# Output Types (Data Structures)
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
    indices: torch.Tensor       # (..., block_size) or (..., packed_d) packed indices
    norms: torch.Tensor         # (...,) original L2 norms
    scales: torch.Tensor        # (..., 1) dynamic RMS scales per vector
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
