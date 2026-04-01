import triton.language as tl

def _get_packing_params(bits: int):
    """
    Helper to determine packing configuration for quantized indices.
    """
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        # Standard TurboQuant 4-bit packing
        return 4, 2
    else:
        # No packing / 8-bit
        return 8, 1
