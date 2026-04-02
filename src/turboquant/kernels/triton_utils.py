import triton.language as tl

def _get_packing_params(bits: int):
    """
    Determines packing configuration for quantized indices.
    Synced 1:1 with quant_base.py pack_indices logic.
    """
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        # Dynamic packing for 3-bit / 4-bit
        # bits=3 -> vpb=2, bits=4 -> vpb=2
        return bits, 8 // bits
    else:
        # No packing / 8-bit
        return 8, 1
