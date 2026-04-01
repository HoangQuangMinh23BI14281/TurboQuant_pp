import torch
import math
import pytest
from src.turboquant.quant.key_quantizer import TurboQuantProd
from src.turboquant.quant.value_quantizer import TurboQuantValue
from src.turboquant.kernels.fused_attention import turboquant_attention

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHellFused:
    """
    HELL-LEVEL STRESS TESTS for the Fused Triton Kernel.
    Designed to break numerical stability, memory indexing, and bit-packing logic.
    """
    
    def test_softmax_stability_hell(self):
        """
        HELL 1: Numerical Stability.
        Burying the kernel in extreme scores (>1000) to test Online Softmax robustness.
        """
        device = "cuda"
        dim = 128
        n_q, n_k = 1, 1024
        bits = 4
        
        # 1. Setup quantizers
        k_quant = TurboQuantProd(dim, bits=bits).to(device)
        v_quant = TurboQuantValue(dim, bits=bits).to(device)
        
        # 2. Forge adversarial data: extreme dot products
        # We want scores = query * key.T to be very large (e.g. 5000)
        query = torch.ones((1, 1, dim), device=device) * 10.0
        key = torch.ones((1, n_k, dim), device=device) * 10.0 # 10 * 10 * 128 = 12,800!
        value = torch.randn((1, n_k, dim), device=device)
        
        # 3. Quantize
        q_key = k_quant.quantize(key)
        q_value = v_quant.quantize(value)
        
        # 4. Run Fused Attention (Triton)
        # Should NOT return NaN despite exp(12800) being inf
        output, _ = turboquant_attention(query, q_key, q_value, quantizer=k_quant, scale=1.0)
        
        assert not torch.isnan(output).any(), "Softmax stability failed: NaN detected in output!"
        assert not torch.isinf(output).any(), "Softmax stability failed: Inf detected in output!"
        print(f"Hell 1 Passed: Output is stable at score={10.0*10.0*dim}")

    def test_long_context_hell(self):
        """
        HELL 2: 128k Long-Context Support.
        Tests tiling, indexing, and memory boundaries for massive N.
        """
        device = "cuda"
        dim = 128
        n_q = 1
        n_k = 131072 # 128k tokens
        bits = 4
        
        k_quant = TurboQuantProd(dim, bits=bits).to(device)
        v_quant = TurboQuantValue(dim, bits=bits).to(device)
        
        # Low-variance data to avoid OOM in test logic
        query = torch.randn((1, 1, dim), device=device)
        key = torch.randn((1, n_k, dim), device=device)
        value = torch.randn((1, n_k, dim), device=device)
        
        q_key = k_quant.quantize(key)
        q_value = v_quant.quantize(value)
        
        # Execution
        output, _ = turboquant_attention(query, q_key, q_value, quantizer=k_quant)
        
        assert output.shape == (1, 1, dim)
        assert not torch.isnan(output).any()
        print(f"Hell 2 Passed: 128k Long-Context handled successfully.")

    def test_odd_bit_packing_hell(self):
        """
        HELL 3: Asymmetric Odd-Bit Unpacking (K=5, V=3).
        Tests sub-byte SRAM unpacking at the limits.
        """
        device = "cuda"
        dim = 128
        n_k = 256
        
        # K=5 (Lẻ), V=3 (Lẻ)
        k_quant = TurboQuantProd(dim, bits=5).to(device)
        v_quant = TurboQuantValue(dim, bits=3).to(device)
        
        q = torch.randn((1, 1, dim), device=device)
        k = torch.randn((1, n_k, dim), device=device)
        v = torch.randn((1, n_k, dim), device=device)
        
        q_key = k_quant.quantize(k)
        q_value = v_quant.quantize(v)
        
        # Run Fused Path
        output, _ = turboquant_attention(q, q_key, q_value, quantizer=k_quant, k_bits=5, v_bits=3)
        
        assert not torch.isnan(output).any()
        print(f"Hell 3 Passed: Odd-bit configuration K=5, V=3 works.")

    def test_large_dim_hell(self):
        """
        HELL 4: Large Head-Dim (D=256).
        Checks for register pressure and tile boundary issues.
        """
        device = "cuda"
        dim = 256 # Double the standard dim
        n_k = 512
        
        k_quant = TurboQuantProd(dim, bits=4).to(device)
        v_quant = TurboQuantValue(dim, bits=4).to(device)
        
        q = torch.randn((1, 1, dim), device=device)
        k = torch.randn((1, n_k, dim), device=device)
        v = torch.randn((1, n_k, dim), device=device)
        
        q_key = k_quant.quantize(k)
        q_value = v_quant.quantize(v)
        
        # Execute
        output, _ = turboquant_attention(q, q_key, q_value, quantizer=k_quant)
        
        assert output.shape == (1, 1, 256)
        print(f"Hell 4 Passed: Large dim D=256 verified.")

if __name__ == "__main__":
    test = TestHellFused()
    # Manual run for debugging
    if torch.cuda.is_available():
        test.test_softmax_stability_hell()
        test.test_long_context_hell()
        test.test_odd_bit_packing_hell()
        test.test_large_dim_hell()
