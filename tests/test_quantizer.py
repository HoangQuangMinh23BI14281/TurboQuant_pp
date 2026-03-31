import torch
import pytest
from turboquant.quant.quantizer import TurboQuantizer

@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_key_basic(d, bits):
    """
    Verify basic quantization flow and output shapes.
    """
    # MSE only
    q = TurboQuantizer(d, bits, use_qjl=False)
    key = torch.randn(2, d)
    
    result = q.quantize_key(key)
    assert 'norm' in result
    assert 'indices' in result
    
    # Indices shape matches block_size (256)
    assert result['indices'].shape[-1] == q.block_size
    assert result['norm'].shape == (2,)
    
    # Check bit range (using .item() for scalar comparison)
    max_idx = result['indices'].max().item()
    limit = (1 << q.mse_bits)
    assert max_idx < limit

@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("bits", [4])
def test_quantize_qjl_flow(d, bits):
    """
    Verify QJL output presence and sign properties.
    """
    q = TurboQuantizer(d, bits, use_qjl=True)
    key = torch.randn(1, d)
    
    result = q.quantize_key(key)
    assert 'qjl_indices' in result
    assert 'residual_norm' in result
    assert result['qjl_indices'].shape[-1] == q.block_size
    
    # QJL indices should be 0 or 1
    assert result['indices'].max().item() < (1 << q.mse_bits)
    assert result['qjl_indices'].max().item() <= 1

@pytest.mark.parametrize("bits", [4, 8])
def test_bit_packing_lossless(bits):
    """
    Verify that packing and unpacking indices is a lossless operation.
    """
    d = 256
    q = TurboQuantizer(d, bits, use_qjl=False)
    
    # Generate random indices in [0, 2^mse_bits - 1]
    indices = torch.randint(0, 1 << q.mse_bits, (2, q.block_size), dtype=torch.long)
    
    # Pack
    packed = q.pack_indices(indices)
    
    # Expected packed size: block_size * mse_bits / 8
    expected_packed_d = q.block_size * q.mse_bits // 8
    assert packed.shape[-1] == expected_packed_d
    assert packed.dtype == torch.uint8
    
    # Unpack
    unpacked = q.unpack_indices(packed, q.block_size)
    
    torch.testing.assert_close(indices, unpacked)

def test_quantization_roundtrip():
    """
    Full End-to-End verification: Quantize then Dequantize.
    """
    d = 120 # Non-standard dimension
    bits = 4
    q = TurboQuantizer(d, bits, use_qjl=True)
    
    key = torch.randn(2, d)
    quant_data = q.quantize_key(key)
    reconstructed = q.dequantize_key(quant_data)
    
    assert reconstructed.shape == key.shape
    
    # Check correlation (should be very high > 0.9 for 4-bit)
    cos_sim = torch.nn.functional.cosine_similarity(key, reconstructed, dim=-1)
    assert cos_sim.mean().item() > 0.9

def test_quantizer_broadcasting():
    """
    Verify batch dimension support.
    """
    d = 128
    bits = 4
    q = TurboQuantizer(d, bits, use_qjl=True)
    
    key = torch.randn(2, 4, d) # (batch, seq, dim)
    result = q.quantize_key(key)
    
    assert result['indices'].shape == (2, 4, q.block_size)
    assert result['norm'].shape == (2, 4)
    assert result['qjl_indices'].shape == (2, 4, q.block_size)
