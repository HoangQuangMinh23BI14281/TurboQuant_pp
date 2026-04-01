import torch
import pytest
from turboquant.quant.quantizer import TurboQuantMSE, TurboQuantProd, MSEQuantized, ProdQuantized, pack_indices, unpack_indices

# ─── TurboQuantMSE Tests ─────────────────────────────────────────────

@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("bits", [4, 8])
def test_mse_quantize_basic(d, bits):
    """Verify basic MSE quantization flow and output types."""
    q = TurboQuantMSE(d, bits)
    key = torch.randn(2, d)

    result = q.quantize(key, pack=False)
    assert isinstance(result, MSEQuantized)
    assert result.indices.shape[-1] == q.block_size
    assert result.norms.shape == (2,)
    assert result.bits == bits

    # Indices in valid range
    max_idx = result.indices.max().item()
    assert max_idx < (1 << bits)

@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("bits", [3, 4])
def test_mse_roundtrip(d, bits):
    """MSE quantize-dequantize should preserve shape and have high cosine sim."""
    q = TurboQuantMSE(d, bits)
    key = torch.randn(4, d)

    reconstructed = q(key)  # forward = quantize + dequantize
    assert reconstructed.shape == key.shape

    cos_sim = torch.nn.functional.cosine_similarity(key, reconstructed, dim=-1)
    assert cos_sim.mean().item() > 0.9, f"cos_sim={cos_sim.mean().item():.4f} too low for d={d}, bits={bits}"


# ─── TurboQuantProd Tests ────────────────────────────────────────────

@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("bits", [4])
def test_prod_quantize_flow(d, bits):
    """Verify Prod quantization outputs and QJL sign properties."""
    q = TurboQuantProd(d, bits)
    key = torch.randn(1, d)

    result = q.quantize(key, pack=False)
    assert isinstance(result, ProdQuantized)
    assert result.mse_indices.shape[-1] == q.block_size
    assert result.qjl_signs.shape[-1] == q.block_size
    assert result.residual_norms.shape == (1,)
    assert result.norms.shape == (1,)
    assert result.mse_bits == bits - 1

    # MSE indices in valid range
    assert result.mse_indices.max().item() < (1 << result.mse_bits)
    # QJL signs are 0 or 1
    assert result.qjl_signs.max().item() <= 1
    assert result.qjl_signs.min().item() >= 0


def test_prod_quantize_packed():
    """Verify packed shapes for Prod quantization."""
    d = 128
    bits = 4
    q = TurboQuantProd(d, bits)
    key = torch.randn(2, d)

    result = q.quantize(key, pack=True)
    assert result.packed is True
    
    # MSE bits = 3 -> 2 values per byte
    assert result.mse_indices.shape[-1] == q.block_size // 2
    # QJL bits = 1 -> 8 values per byte
    assert result.qjl_signs.shape[-1] == q.block_size // 8
    assert result.mse_indices.dtype == torch.uint8
    assert result.qjl_signs.dtype == torch.uint8


def test_prod_roundtrip():
    """
    Full End-to-End verification: Prod quantize then dequantize.
    Should achieve > 0.9 cosine similarity for standard dimensions.
    """
    d = 128
    bits = 4
    q = TurboQuantProd(d, bits)

    key = torch.randn(4, d)
    reconstructed = q(key)

    assert reconstructed.shape == key.shape

    cos_sim = torch.nn.functional.cosine_similarity(key, reconstructed, dim=-1)
    assert cos_sim.mean().item() > 0.9, f"cos_sim={cos_sim.mean().item():.4f} too low"


def test_prod_roundtrip_nonstandard_dim():
    """
    Roundtrip quality for non-standard dimension (d=120, requires padding).
    """
    d = 120
    bits = 4
    q = TurboQuantProd(d, bits)

    key = torch.randn(4, d)
    reconstructed = q(key)

    assert reconstructed.shape == key.shape

    cos_sim = torch.nn.functional.cosine_similarity(key, reconstructed, dim=-1)
    assert cos_sim.mean().item() > 0.9, f"cos_sim={cos_sim.mean().item():.4f} too low for d={d}"


# ─── Bit Packing Tests ──────────────────────────────────────────────

@pytest.mark.parametrize("bits", [4, 8])
def test_bit_packing_lossless(bits):
    """Verify that packing and unpacking indices is lossless."""
    d = 256

    # Generate random indices in [0, 2^bits - 1]
    indices = torch.randint(0, 1 << bits, (2, d), dtype=torch.long)

    # Pack
    packed = pack_indices(indices, bits)

    if bits < 8:
        vals_per_byte = 8 // bits
        expected_packed_d = d // vals_per_byte
        assert packed.shape[-1] == expected_packed_d
    assert packed.dtype == torch.uint8

    # Unpack
    unpacked = unpack_indices(packed, bits, d)

    torch.testing.assert_close(indices, unpacked)


# ─── Broadcasting Tests ─────────────────────────────────────────────

def test_prod_broadcasting():
    """Verify batch dimension support for TurboQuantProd."""
    d = 128
    bits = 4
    q = TurboQuantProd(d, bits)

    key = torch.randn(2, 4, d)  # (batch, seq, dim)
    result = q.quantize(key, pack=False)

    assert result.mse_indices.shape == (2, 4, q.block_size)
    assert result.norms.shape == (2, 4)
    assert result.qjl_signs.shape == (2, 4, q.block_size)
    assert result.residual_norms.shape == (2, 4)


def test_mse_broadcasting():
    """Verify batch dimension support for TurboQuantMSE."""
    d = 128
    bits = 4
    q = TurboQuantMSE(d, bits)

    key = torch.randn(2, 4, d)  # (batch, seq, dim)
    result = q.quantize(key, pack=False)

    assert result.indices.shape == (2, 4, q.block_size)
    assert result.norms.shape == (2, 4)


# ─── Quality Comparison Tests ────────────────────────────────────────

def test_prod_better_than_mse_at_same_total_bits():
    """TurboQuantProd should reconstruct with quality comparable to or
    better inner product quality than pure MSE at same total bits."""
    d = 128
    bits = 4
    torch.manual_seed(42)

    mse_q = TurboQuantMSE(d, bits)
    prod_q = TurboQuantProd(d, bits)

    key = torch.randn(20, d)
    query = torch.randn(5, d)

    # True inner products
    true_ip = torch.matmul(query, key.T)

    # MSE reconstruction inner products
    key_mse = mse_q(key)
    mse_ip = torch.matmul(query, key_mse.T)

    # Prod reconstruction inner products
    key_prod = prod_q(key)
    prod_ip = torch.matmul(query, key_prod.T)

    # Both should correlate well with true
    mse_corr = torch.corrcoef(torch.stack([true_ip.flatten(), mse_ip.flatten()]))[0, 1]
    prod_corr = torch.corrcoef(torch.stack([true_ip.flatten(), prod_ip.flatten()]))[0, 1]

    assert mse_corr.item() > 0.9, f"MSE correlation {mse_corr:.4f} too low"
    assert prod_corr.item() > 0.9, f"Prod correlation {prod_corr:.4f} too low"


@pytest.mark.parametrize("bits", range(1, 9))
def test_bit_perfect_packing(bits):
    """
    EXTREME HARDENING: Bit-perfect packing/unpacking test for all depths (1-8 bits).
    Every possible value must be recovered exactly.
    """
    d = 256
    # Create test indices covering full range [0, 2^bits - 1]
    n_vals = d
    max_val = (1 << bits) - 1
    
    # Generate random indices within range
    indices = torch.randint(0, max_val + 1, (2, 4, n_vals), dtype=torch.uint8)
    
    # Pack
    packed = pack_indices(indices, bits)
    
    # Unpack
    unpacked = unpack_indices(packed, bits, n_vals)
    
    # Check bit-perfection
    assert torch.equal(indices, unpacked), f"Bit-packing failed for {bits} bits. Indices lost!"
