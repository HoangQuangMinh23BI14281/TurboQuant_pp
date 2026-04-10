import torch
import math
import numpy as np
from typing import Tuple, Dict, Optional

# ──────────────────────────────────────────────────────────────────────
# Full Lloyd-Max Codebooks (1-8 bits) - EXACTLY AS PROVIDED BY USER
# ──────────────────────────────────────────────────────────────────────

LM_BOUNDARIES = {
    1: torch.tensor([0.]),
    2: torch.tensor([-0.9816, 0.0, 0.9816]),
    3: torch.tensor([-1.7479, -1.0500, -0.5005, 0.0000, 0.5005, 1.0500, 1.7479]),
    4: torch.tensor([-2.4008, -1.8435, -1.4371, -1.0993, -0.7995, -0.5224, -0.2582, 0.0000,
                     0.2582, 0.5224, 0.7995, 1.0993, 1.4371, 1.8435, 2.4008]),
    5: torch.tensor([-2.9760, -2.5045, -2.1733, -1.9080, -1.6818, -1.4813, -1.2991, -1.1303,
                     -0.9717, -0.8209, -0.6761, -0.5358, -0.3991, -0.2647, -0.1320, 0.0000,
                     0.1320, 0.2647, 0.3991, 0.5358, 0.6761, 0.8209, 0.9717, 1.1303,
                     1.2991, 1.4813, 1.6818, 1.9080, 2.1733, 2.5045, 2.9760]),
    6: torch.tensor([-3.5169, -3.1058, -2.8233, -2.6015, -2.4159, -2.2544, -2.1103, -1.9792,
                     -1.8584, -1.7457, -1.6397, -1.5392, -1.4433, -1.3515, -1.2630, -1.1774,
                     -1.0943, -1.0134, -0.9344, -0.8571, -0.7812, -0.7066, -0.6331, -0.5606,
                     -0.4889, -0.4178, -0.3473, -0.2773, -0.2077, -0.1383, -0.0691, 0.0000,
                     0.0691, 0.1383, 0.2077, 0.2773, 0.3473, 0.4178, 0.4889, 0.5606,
                     0.6331, 0.7066, 0.7812, 0.8571, 0.9344, 1.0134, 1.0943, 1.1774,
                     1.2630, 1.3515, 1.4433, 1.5392, 1.6397, 1.7457, 1.8584, 1.9792,
                     2.1103, 2.2544, 2.4159, 2.6015, 2.8233, 3.1058, 3.5169]),
    7: torch.tensor([-4.0871, -3.7257, -3.4813, -3.2920, -3.1356, -3.0011, -2.8824, -2.7756,
                     -2.6782, -2.5883, -2.5045, -2.4259, -2.3516, -2.2811, -2.2138, -2.1494,
                     -2.0874, -2.0276, -1.9697, -1.9136, -1.8590, -1.8057, -1.7538, -1.7029,
                     -1.6531, -1.6042, -1.5561, -1.5087, -1.4621, -1.4161, -1.3706, -1.3256,
                     -1.2812, -1.2371, -1.1934, -1.1501, -1.1071, -1.0644, -1.0219, -0.9797,
                     -0.9377, -0.8959, -0.8542, -0.8128, -0.7714, -0.7302, -0.6891, -0.6482,
                     -0.6073, -0.5665, -0.5257, -0.4851, -0.4445, -0.4039, -0.3634, -0.3229,
                     -0.2825, -0.2421, -0.2017, -0.1613, -0.1210, -0.0807, -0.0403, 0.0000,
                     0.0403, 0.0807, 0.1210, 0.1613, 0.2017, 0.2421, 0.2825, 0.3229,
                     0.3634, 0.4039, 0.4445, 0.4851, 0.5257, 0.5665, 0.6073, 0.6482,
                     0.6891, 0.7302, 0.7714, 0.8128, 0.8542, 0.8959, 0.9377, 0.9797,
                     1.0219, 1.0644, 1.1071, 1.1501, 1.1934, 1.2371, 1.2812, 1.3256,
                     1.3706, 1.4161, 1.4621, 1.5087, 1.5561, 1.6042, 1.6531, 1.7029,
                     1.7538, 1.8057, 1.8590, 1.9136, 1.9697, 2.0276, 2.0874, 2.1494,
                     2.2138, 2.2811, 2.3516, 2.4259, 2.5045, 2.5883, 2.6782, 2.7756,
                     2.8824, 3.0011, 3.1356, 3.2920, 3.4813, 3.7257, 4.0871]),
    8: torch.tensor([-4.5020, -4.1704, -3.9485, -3.7782, -3.6388, -3.5200, -3.4161, -3.3236,
                     -3.2400, -3.1636, -3.0933, -3.0280, -2.9670, -2.9098, -2.8558, -2.8047,
                     -2.7562, -2.7101, -2.6659, -2.6237, -2.5831, -2.5441, -2.5065, -2.4701,
                     -2.4350, -2.4009, -2.3679, -2.3357, -2.3044, -2.2738, -2.2440, -2.2148,
                     -2.1862, -2.1582, -2.1307, -2.1037, -2.0771, -2.0508, -2.0250, -1.9995,
                     -1.9742, -1.9493, -1.9246, -1.9002, -1.8759, -1.8519, -1.8280, -1.8042,
                     -1.7806, -1.7572, -1.7338, -1.7106, -1.6875, -1.6644, -1.6414, -1.6185,
                     -1.5956, -1.5728, -1.5500, -1.5273, -1.5046, -1.4820, -1.4593, -1.4367,
                     -1.4142, -1.3916, -1.3691, -1.3465, -1.3240, -1.3015, -1.2790, -1.2565,
                     -1.2341, -1.2116, -1.1891, -1.1667, -1.1442, -1.1218, -1.0993, -1.0769,
                     -1.0544, -1.0320, -1.0095, -0.9871, -0.9646, -0.9422, -0.9198, -0.8973,
                     -0.8749, -0.8525, -0.8300, -0.8076, -0.7852, -0.7627, -0.7403, -0.7179,
                     -0.6954, -0.6730, -0.6506, -0.6281, -0.6057, -0.5833, -0.5608, -0.5384,
                     -0.5160, -0.4935, -0.4711, -0.4487, -0.4262, -0.4038, -0.3814, -0.3589,
                     -0.3365, -0.3141, -0.2916, -0.2692, -0.2468, -0.2243, -0.2019, -0.1795,
                     -0.1570, -0.1346, -0.1122, -0.0897, -0.0673, -0.0449, -0.0224, 0.0000,
                     0.0224, 0.0449, 0.0673, 0.0897, 0.1122, 0.1346, 0.1570, 0.1795,
                     0.2019, 0.2243, 0.2468, 0.2692, 0.2916, 0.3141, 0.3365, 0.3589,
                     0.3814, 0.4038, 0.4262, 0.4487, 0.4711, 0.4935, 0.5160, 0.5384,
                     0.5608, 0.5833, 0.6057, 0.6281, 0.6506, 0.6730, 0.6954, 0.7179,
                     0.7403, 0.7627, 0.7852, 0.8076, 0.8300, 0.8525, 0.8749, 0.8973,
                     0.9198, 0.9422, 0.9646, 0.9871, 1.0095, 1.0320, 1.0544, 1.0769,
                     1.0993, 1.1218, 1.1442, 1.1667, 1.1891, 1.2116, 1.2341, 1.2565,
                     1.2790, 1.3015, 1.3240, 1.3465, 1.3691, 1.3916, 1.4142, 1.4367,
                     1.4593, 1.4820, 1.5046, 1.5273, 1.5500, 1.5728, 1.5956, 1.6185,
                     1.6414, 1.6644, 1.6875, 1.7106, 1.7338, 1.7572, 1.7806, 1.8042,
                     1.8280, 1.8519, 1.8759, 1.9002, 1.9246, 1.9493, 1.9742, 1.9995,
                     2.0250, 2.0508, 2.0771, 2.1037, 2.1307, 2.1582, 2.1862, 2.2148,
                     2.2440, 2.2738, 2.3044, 2.3357, 2.3679, 2.4009, 2.4350, 2.4701,
                     2.5065, 2.5441, 2.5831, 2.6237, 2.6659, 2.7101, 2.7562, 2.8047,
                     2.8558, 2.9098, 2.9670, 3.0280, 3.0933, 3.1636, 3.2400, 3.2798,
                     3.3674, 3.4649, 3.5751, 3.7025, 3.8540, 4.0430, 4.2978, 4.5020]),
}

LM_CENTROIDS = {
    1: torch.tensor([-0.7979, 0.7979]),
    2: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: torch.tensor([-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519]),
    4: torch.tensor([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284,
                     0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326]),
    5: torch.tensor([-3.2608, -2.6912, -2.3178, -2.0288, -1.7873, -1.5763, -1.3864, -1.2118,
                     -1.0488, -0.8946, -0.7472, -0.6050, -0.4667, -0.3314, -0.1981, -0.0659,
                     0.0659, 0.1981, 0.3314, 0.4667, 0.6050, 0.7472, 0.8946, 1.0488,
                     1.2118, 1.3864, 1.5763, 1.7873, 2.0288, 2.3178, 2.6912, 3.2608]),
    6: torch.tensor([-3.7674, -3.2664, -2.9452, -2.7015, -2.5016, -2.3302, -2.1786, -2.0419,
                     -1.9165, -1.8002, -1.6912, -1.5882, -1.4902, -1.3965, -1.3064, -1.2195,
                     -1.1352, -1.0533, -0.9735, -0.8954, -0.8188, -0.7437, -0.6696, -0.5967,
                     -0.5245, -0.4532, -0.3824, -0.3122, -0.2424, -0.1729, -0.1037, -0.0345,
                     0.0345, 0.1037, 0.1729, 0.2424, 0.3122, 0.3824, 0.4532, 0.5245,
                     0.5967, 0.6696, 0.7437, 0.8188, 0.8954, 0.9735, 1.0533, 1.1352,
                     1.2195, 1.3064, 1.3965, 1.4902, 1.5882, 1.6912, 1.8002, 1.9165,
                     2.0419, 2.1786, 2.3302, 2.5016, 2.7015, 2.9452, 3.2664, 3.7674]),
    7: torch.tensor([-4.3088, -3.8654, -3.5859, -3.3767, -3.2074, -3.0638, -2.9384, -2.8264,
                     -2.7248, -2.6315, -2.5450, -2.4640, -2.3878, -2.3155, -2.2467, -2.1809,
                     -2.1178, -2.0570, -1.9982, -1.9412, -1.8859, -1.8320, -1.7795, -1.7281,
                     -1.6778, -1.6284, -1.5799, -1.5322, -1.4853, -1.4389, -1.3932, -1.3480,
                     -1.3033, -1.2590, -1.2152, -1.1717, -1.1285, -1.0857, -1.0431, -1.0008,
                     -0.9586, -0.9167, -0.8750, -0.8335, -0.7921, -0.7508, -0.7097, -0.6686,
                     -0.6277, -0.5868, -0.5461, -0.5054, -0.4647, -0.4242, -0.3836, -0.3432,
                     -0.3027, -0.2623, -0.2219, -0.1815, -0.1412, -0.1008, -0.0605, -0.0202,
                     0.0202, 0.0605, 0.1008, 0.1412, 0.1815, 0.2219, 0.2623, 0.3027,
                     0.3432, 0.3836, 0.4242, 0.4647, 0.5054, 0.5461, 0.5868, 0.6277,
                     0.6686, 0.7097, 0.7508, 0.7921, 0.8335, 0.8750, 0.9167, 0.9586,
                     1.0008, 1.0431, 1.0857, 1.1285, 1.1717, 1.2152, 1.2590, 1.3033,
                     1.3480, 1.3932, 1.4389, 1.4853, 1.5322, 1.5799, 1.6284, 1.6778,
                     1.7281, 1.7795, 1.8320, 1.8859, 1.9412, 1.9982, 2.0570, 2.1178,
                     2.1809, 2.2467, 2.3155, 2.3878, 2.4640, 2.5450, 2.6315, 2.7248,
                     2.8264, 2.9384, 3.0638, 3.2074, 3.3767, 3.5859, 3.8654, 4.3088]),
    8: torch.tensor([-4.7062, -4.2978, -4.0430, -3.8540, -3.7025, -3.5751, -3.4649, -3.3674,
                     -3.2798, -3.2002, -3.1271, -3.0595, -2.9965, -2.9375, -2.8820, -2.8296,
                     -2.7799, -2.7326, -2.6875, -2.6444, -2.6030, -2.5632, -2.5249, -2.4880,
                     -2.4523, -2.4177, -2.3842, -2.3516, -2.3198, -2.2889, -2.2587, -2.2292,
                     -2.2004, -2.1721, -2.1443, -2.1171, -2.0903, -2.0639, -2.0378, -2.0121,
                     -1.9868, -1.9617, -1.9369, -1.9123, -1.8880, -1.8638, -1.8399, -1.8161,
                     -1.7924, -1.7689, -1.7455, -1.7222, -1.6990, -1.6759, -1.6529, -1.6299,
                     -1.6070, -1.5842, -1.5614, -1.5387, -1.5159, -1.4933, -1.4706, -1.4480,
                     -1.4254, -1.4029, -1.3803, -1.3578, -1.3353, -1.3128, -1.2903, -1.2678,
                     -1.2453, -1.2228, -1.2004, -1.1779, -1.1554, -1.1330, -1.1105, -1.0881,
                     -1.0656, -1.0432, -1.0207, -0.9983, -0.9759, -0.9534, -0.9310, -0.9086,
                     -0.8861, -0.8637, -0.8412, -0.8188, -0.7964, -0.7739, -0.7515, -0.7291,
                     -0.7066, -0.6842, -0.6618, -0.6393, -0.6169, -0.5945, -0.5720, -0.5496,
                     -0.5272, -0.5047, -0.4823, -0.4599, -0.4374, -0.4150, -0.3926, -0.3701,
                     -0.3477, -0.3253, -0.3028, -0.2804, -0.2580, -0.2355, -0.2131, -0.1907,
                     -0.1682, -0.1458, -0.1234, -0.1009, -0.0785, -0.0561, -0.0336, -0.0112,
                     0.0112, 0.0336, 0.0561, 0.0785, 0.1009, 0.1234, 0.1458, 0.1682,
                     0.1907, 0.2131, 0.2355, 0.2580, 0.2804, 0.3028, 0.3253, 0.3477,
                     0.3701, 0.3926, 0.4150, 0.4374, 0.4599, 0.4823, 0.5047, 0.5272,
                     0.5496, 0.5720, 0.5945, 0.6169, 0.6393, 0.6618, 0.6842, 0.7066,
                     0.7291, 0.7515, 0.7739, 0.7964, 0.8188, 0.8412, 0.8637, 0.8861,
                     0.9086, 0.9310, 0.9534, 0.9759, 0.9983, 1.0207, 1.0432, 1.0656,
                     1.0881, 1.1105, 1.1330, 1.1554, 1.1779, 1.2004, 1.2228, 1.2453,
                     1.2678, 1.2903, 1.3128, 1.3353, 1.3578, 1.3803, 1.4029, 1.4254,
                     1.4480, 1.4706, 1.4933, 1.5159, 1.5387, 1.5614, 1.5842, 1.6070,
                     1.6299, 1.6529, 1.6759, 1.6990, 1.7222, 1.7455, 1.7689, 1.7924,
                     1.8161, 1.8399, 1.8638, 1.8880, 1.9123, 1.9369, 1.9617, 1.9868,
                     2.0121, 2.0378, 2.0639, 2.0903, 2.1171, 2.1443, 2.1721, 2.2004,
                     2.2292, 2.2587, 2.2889, 2.3198, 2.3516, 2.3842, 2.4177, 2.4523,
                     2.4880, 2.5249, 2.5632, 2.6030, 2.6444, 2.6875, 2.7326, 2.7799,
                     2.8296, 2.8820, 2.9375, 2.9965, 3.0595, 3.1271, 3.2002, 3.2798,
                     3.3674, 3.4649, 3.5751, 3.7025, 3.8540, 4.0430, 4.2978, 4.7062]),
}

# ──────────────────────────────────────────────────────────────────────
# Native Torch Gaussian Functions (Stable erfc/erfinv)
# ──────────────────────────────────────────────────────────────────────

def gaussian_cdf(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / (sigma * math.sqrt(2))))

def gaussian_sf(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return 0.5 * torch.erfc(x / (sigma * math.sqrt(2)))

def gaussian_ppf(q: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return (sigma * math.sqrt(2)) * torch.erfinv(2 * q - 1)

def laplace_ppf(q: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """PPF of Laplace distribution: F^-1(q) = -b * sgn(q-0.5) * ln(1 - 2|q-0.5|)"""
    return -scale * torch.sign(q - 0.5) * torch.log(1 - 2 * torch.abs(q - 0.5))

def _gaussian_conditional_expectation(a: float, b: float, sigma: float = 1.0) -> float:
    a_st = a / sigma if math.isfinite(a) else a
    b_st = b / sigma if math.isfinite(b) else b
    if a_st > 0:
        prob = (gaussian_sf(torch.tensor(a_st), 1.0) - gaussian_sf(torch.tensor(b_st), 1.0)).item()
    else:
        prob = (gaussian_cdf(torch.tensor(b_st), 1.0) - gaussian_cdf(torch.tensor(a_st), 1.0)).item()
    if prob < 1e-15:
        if math.isfinite(a) and not math.isfinite(b): return a + sigma
        if not math.isfinite(a) and math.isfinite(b): return b - sigma
        return (a + b) / 2.0
    pdf_a = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a_st**2) if math.isfinite(a_st) else 0.0
    pdf_b = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * b_st**2) if math.isfinite(b_st) else 0.0
    return sigma * (pdf_a - pdf_b) / prob

def _laplace_conditional_expectation(a: float, b: float, b_param: float = 1.0) -> float:
    """E[X | a < X < b] for X ~ Laplace(0, b_param)"""
    # Safety: ensure finite bounds for math.exp
    a = max(-700 * b_param, min(700 * b_param, a))
    b = max(-700 * b_param, min(700 * b_param, b))

    def anti_xf(x):
        if x >= 0: return -0.5 * math.exp(-x / b_param) * (x + b_param)
        else: return 0.5 * math.exp(x / b_param) * (x - b_param)
    
    def cdf(x):
        if x >= 0: return 1.0 - 0.5 * math.exp(-x / b_param)
        else: return 0.5 * math.exp(x / b_param)
    
    prob = cdf(b) - cdf(a)
    if prob < 1e-18: 
        if math.isinf(a): return b - b_param
        if math.isinf(b): return a + b_param
        return (a + b) / 2.0
    return (anti_xf(b) - anti_xf(a)) / prob

# ──────────────────────────────────────────────────────────────────────
# Stable Solver and APIs
# ──────────────────────────────────────────────────────────────────────

_CODEBOOK_CACHE = {}
_DEVICE_CODEBOOK_CACHE = {} # SOTA v8.7: Guard against .to() during CUDA Graph capture

def harden_lloyd_max(bits: int, device: torch.device, dtype: torch.dtype, dist: str = 'gaussian'):
    """Force-cache codebooks on a specific device/dtype for CUDA Graph safety."""
    cb = compute_lloyd_max_codebook(bits, d=1, dist=dist)
    key = (bits, dist, str(device), dtype)
    if key not in _DEVICE_CODEBOOK_CACHE:
        _DEVICE_CODEBOOK_CACHE[key] = {
            'centroids': cb['centroids'].to(device, dtype),
            'boundaries': cb['boundaries'].to(device, dtype)
        }
    return _DEVICE_CODEBOOK_CACHE[key]

def compute_lloyd_max_codebook(bits: int, d: int = 1, dist: str = 'gaussian', max_iter: int = 40, epsilon: float = 1e-10) -> Dict:
    """
    Compute optimal Lloyd-Max centroids/boundaries for a given distribution.
    - 'gaussian': used for Key cache (rotated domain follows N(0, 1/d)).
    - 'laplace': used for Value cache (asymmetric original domain).
    """
    cache_key = (int(bits), int(d), dist)
    if cache_key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[cache_key]

    # BIN count and quantiles
    n = 1 << bits
    q = torch.linspace(1e-5, 1-1e-5, n + 1, dtype=torch.float32, device='cpu')[1:-1]
    
    if dist == 'gaussian':
        sigma_or_b = 1.0 / math.sqrt(d)
        ppf_fn = gaussian_ppf
        expectation_fn = _gaussian_conditional_expectation
        
        # Optimization: Use static llama.cpp N(0, 1) tables if d=1 and gaussian
        if d == 1 and bits in LM_BOUNDARIES and bits in LM_CENTROIDS:
            cb = {
                'centroids': LM_CENTROIDS[bits].clone().float(), 
                'boundaries': LM_BOUNDARIES[bits].clone().float()
            }
            _CODEBOOK_CACHE[cache_key] = cb
            return cb
            
    elif dist == 'laplace':
        # For Laplace(0, b), variance is 2*b^2. 
        # To match Gaussian variance (1/d), we need b = 1/sqrt(2d)
        sigma_or_b = 1.0 / math.sqrt(2 * d)
        ppf_fn = laplace_ppf
        expectation_fn = _laplace_conditional_expectation
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    # Initial boundaries using PPF of targets
    boundaries = ppf_fn(q, sigma_or_b).numpy().tolist()
    centroids = np.zeros(n)
    
    for _ in range(max_iter):
        bounds = [-float('inf')] + list(boundaries) + [float('inf')]
        for i in range(n):
            centroids[i] = expectation_fn(bounds[i], bounds[i+1], sigma_or_b)
        
        new_b = (centroids[:-1] + centroids[1:]) / 2.0
        if np.allclose(boundaries, new_b, atol=epsilon):
            break
        boundaries = new_b.tolist()
    
    cb = {
        'centroids': torch.tensor(np.sort(centroids), dtype=torch.float32),
        'boundaries': torch.tensor(boundaries, dtype=torch.float32)
    }

    _CODEBOOK_CACHE[cache_key] = cb
    return cb

def lloyd_max_quantize(x: torch.Tensor, bits: int, d: Optional[int] = None, dist: str = 'gaussian') -> torch.Tensor:
    # SOTA v8.7: Fast-path for CUDA Graphs
    key = (int(bits), dist, str(x.device), x.dtype)
    if key in _DEVICE_CODEBOOK_CACHE:
        bounds = _DEVICE_CODEBOOK_CACHE[key]['boundaries']
    else:
        # Fallback (non-graph path or first run)
        cb = compute_lloyd_max_codebook(int(bits), d=(d if d else 1), dist=dist)
        bounds = cb['boundaries'].to(x.device, x.dtype)
        if not torch.cuda.is_current_stream_capturing():
            _DEVICE_CODEBOOK_CACHE[key] = {'boundaries': bounds, 'centroids': cb['centroids'].to(x.device, x.dtype)}
            
    indices = torch.bucketize(x.contiguous(), bounds)
    return indices.clamp(0, (1 << int(bits)) - 1).long()

def lloyd_max_dequantize(indices: torch.Tensor, bits: int, d: Optional[int] = None, dist: str = 'gaussian') -> torch.Tensor:
    # SOTA v8.7: Fast-path for CUDA Graphs
    key = (int(bits), dist, str(indices.device), torch.float16) # Optimization: centroids usually f16
    if key in _DEVICE_CODEBOOK_CACHE:
        centroids = _DEVICE_CODEBOOK_CACHE[key]['centroids']
    else:
        cb = compute_lloyd_max_codebook(int(bits), d=(d if d else 1), dist=dist)
        centroids = cb['centroids'].to(indices.device)
        if not torch.cuda.is_current_stream_capturing():
             _DEVICE_CODEBOOK_CACHE[key] = {'centroids': centroids, 'boundaries': cb['boundaries'].to(indices.device)}

    centroids_view = centroids.view(-1)
    return centroids_view[indices.long().clamp(0, (1 << int(bits)) - 1).contiguous()]
