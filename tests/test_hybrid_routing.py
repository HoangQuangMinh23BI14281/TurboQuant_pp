import torch
import pytest
from turboquant.layers.config import TurboQuantConfig, QuantConfig
from turboquant.layers.attention_layer import TurboQuantAttention

def test_boundary_layer_protection():
    """
    Test 1: Check if protect_boundaries=True correctly bypasses quantization 
    for head/tail layers.
    """
    config = TurboQuantConfig(
        quant=QuantConfig(k_bits=8, v_bits=3),
        protect_boundaries=True,
        n_head_protected=2,
        n_tail_protected=2,
    )
    
    total_layers = 12
    dim = 256
    num_heads = 4
    head_dim = dim // num_heads
    
    # Layer 0: Should be PROTECTED (FP16)
    layer_head = TurboQuantAttention(config, layer_idx=0, total_layers=total_layers, 
                                     dim=dim, num_heads=num_heads, num_kv_heads=num_heads)
    assert layer_head.is_protected == True
    assert layer_head.k_quantizer is None
    
    # Layer 5: Should be QUANTIZED (Hybrid K-8, V-3)
    layer_mid = TurboQuantAttention(config, layer_idx=5, total_layers=total_layers, 
                                    dim=dim, num_heads=num_heads, num_kv_heads=num_heads)
    assert layer_mid.is_protected == False
    assert layer_mid.k_bits == 8
    assert layer_mid.v_bits == 3
    assert layer_mid.k_quantizer.bits == 8
    assert layer_mid.v_quantizer.bits == 3

def test_hybrid_precision_forward():
    """
    Test 2: Check if forward pass works with K-8 and V-3.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TurboQuantConfig(quant=QuantConfig(k_bits=8, v_bits=3), protect_boundaries=False)
    
    dim = 128
    num_heads = 4
    head_dim = dim // num_heads
    layer = TurboQuantAttention(config, layer_idx=5, total_layers=10, 
                                dim=dim, num_heads=num_heads, num_kv_heads=num_heads).to(device)
    
    # Mock data (Batch, N, Dim)
    q = torch.randn((1, 1, dim), device=device)
    k = torch.randn((1, 32, dim), device=device)
    v = torch.randn((1, 32, dim), device=device)
    
    # Run forward
    output, weights = layer(q, k, v)
    
    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    print(f"Hybrid Forward Passed (K-8, V-3): Output shape {output.shape}")

if __name__ == "__main__":
    test_boundary_layer_protection()
    if torch.cuda.is_available():
        test_hybrid_precision_forward()
