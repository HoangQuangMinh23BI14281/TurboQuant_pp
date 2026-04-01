import torch
import math
import pytest
from turboquant.quant.quantizer import TurboQuantProd, TurboQuantMSE
from turboquant.kernels.fused_attention import attention_score_prod, attention_score_mse, turboquant_attention


@pytest.mark.parametrize("d", [128, 256])
def test_attention_score_mse_shape(d):
    """Verify attention_score_mse produces correct output shape."""
    bits = 4
    n_q, n_k = 5, 20
    q = TurboQuantMSE(d, bits)

    query = torch.randn(n_q, d)
    key = torch.randn(n_k, d)

    quantized = q.quantize(key)
    scores = attention_score_mse(query, quantized, q)

    assert scores.shape == (n_q, n_k)


@pytest.mark.parametrize("d", [128, 256])
def test_attention_score_prod_shape(d):
    """Verify attention_score_prod produces correct output shape."""
    bits = 4
    n_q, n_k = 5, 20
    q = TurboQuantProd(d, bits)

    query = torch.randn(n_q, d)
    key = torch.randn(n_k, d)

    quantized = q.quantize(key)
    scores = attention_score_prod(query, quantized, q)

    assert scores.shape == (n_q, n_k)


def test_attention_score_prod_correlation():
    """
    Verify that QJL attention scores correlate very well with true scores.
    Correlation should be > 0.95 for d=128, bits=4.
    """
    d = 128
    bits = 4
    n_q, n_k = 10, 100
    torch.manual_seed(42)

    q = TurboQuantProd(d, bits)
    query = torch.randn(n_q, d)
    key = torch.randn(n_k, d)

    # True scores
    scale = 1.0 / math.sqrt(d)
    true_scores = torch.matmul(query, key.T) * scale

    # TurboQuant estimated scores
    quantized = q.quantize(key)
    est_scores = attention_score_prod(query, quantized, q)

    # Correlation must be very high
    corr = torch.corrcoef(torch.stack([true_scores.flatten(), est_scores.flatten()]))[0, 1]
    assert corr.item() > 0.95, f"Score correlation {corr:.4f} too low"


def test_turboquant_attention_output_shape():
    """Verify full turboquant_attention returns correct shapes."""
    d = 128
    bits = 3
    n_q, n_k = 4, 32
    torch.manual_seed(42)

    q = TurboQuantProd(d, bits)
    query = torch.randn(n_q, d)
    key = torch.randn(n_k, d)
    value = torch.randn(n_k, d)

    quantized = q.quantize(key)
    output, weights = turboquant_attention(query, quantized, value, q)

    assert output.shape == (n_q, d)
    assert weights.shape == (n_q, n_k)

    # Weights should sum to 1
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(n_q))


def test_attention_quality_vs_true():
    """
    Verify full attention output is close to true attention.
    Relative error should be < 0.3 for d=128, bits=4.
    """
    d = 128
    bits = 4
    n_q, n_k = 8, 64
    torch.manual_seed(42)

    q = TurboQuantProd(d, bits)
    scale = 1.0 / math.sqrt(d)

    query = torch.randn(n_q, d)
    key = torch.randn(n_k, d)
    value = torch.randn(n_k, d)

    # True attention
    true_scores = torch.matmul(query, key.T) * scale
    true_weights = torch.softmax(true_scores, dim=-1)
    true_output = torch.matmul(true_weights, value)

    # TurboQuant attention
    quantized = q.quantize(key)
    est_output, _ = turboquant_attention(query, quantized, value, q)

    # Relative error must be low
    rel_error = torch.norm(est_output.float() - true_output) / torch.norm(true_output)
    assert rel_error.item() < 0.3, f"Attention output relative error {rel_error:.4f} too high"
