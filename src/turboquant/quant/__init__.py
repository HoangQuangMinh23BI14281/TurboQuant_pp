from .lloyd_max import lloyd_max_quantize, lloyd_max_dequantize
from .quantizer import (
    TurboQuantMSE, TurboQuantProd,
    MSEQuantized, ProdQuantized,
    pack_indices, unpack_indices,
)

# Backward compatibility alias
TurboQuantizer = TurboQuantProd

__all__ = [
    'lloyd_max_quantize', 'lloyd_max_dequantize',
    'TurboQuantMSE', 'TurboQuantProd',
    'MSEQuantized', 'ProdQuantized',
    'pack_indices', 'unpack_indices',
    'TurboQuantizer',
]
