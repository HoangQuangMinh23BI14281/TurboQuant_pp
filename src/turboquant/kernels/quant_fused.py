import torch
import triton
import triton.language as tl
import math

# ──────────────────────────────────────────────────────────────────────
# Fused FWHT + Lloyd-Max + Packing Kernel (v12.5 SPEED DEMON)
# ──────────────────────────────────────────────────────────────────────

@triton.jit
def _fused_quant_kernel(
    X_ptr,          # (N, D)
    SIGNS_ptr,      # (N_PASSES, D)
    WHT_MAT_ptr,    # (D, D)
    OUT_INDICES_ptr,# (N, PACKED_D or D)
    OUT_SCALES_ptr, # (N, 1)
    OUT_NORMS_ptr,  # (N, 1)
    OUT_ROTATED_ptr,# (N, D)
    N, D,
    N_PASSES,
    FINAL_SCALE,
    DIST_ID: tl.constexpr, 
    BITS: tl.constexpr,
    PACKED_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DO_PACK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # 1. Initialization
    d_offset = tl.arange(0, BLOCK_D)
    x = tl.load(X_ptr + pid * D + d_offset).to(tl.float32)
    
    # Norm calculation (Epsilon 1e-8)
    norm = tl.sqrt(tl.sum(x * x))
    v = (x / (norm + 1e-8)).to(tl.float32)

    # 2. Optimized Cascaded Rotation (FWHT)
    h_mat = tl.load(WHT_MAT_ptr + d_offset[:, None] * 64 + d_offset[None, :]).to(tl.float32)
    for p in range(N_PASSES):
        signs = tl.load(SIGNS_ptr + p * D + d_offset).to(tl.float32)
        v = (v * signs).to(tl.float32)
        v = tl.sum(v[None, :] * h_mat, axis=1)

    # 3. Final Scaling & Store Rotated
    v = (v * FINAL_SCALE).to(tl.float32)
    tl.store(OUT_ROTATED_ptr + pid * D + d_offset, v)
    
    # 4. Table Selection (SOTA Audit v13.0)
    if DIST_ID == 1: # Laplace (Value Cache)
        mc = 3.0755
    else: # Gaussian (Key Cache)
        mc = 2.7326
        
    v_abs = tl.abs(v)
    max_val = tl.max(v_abs)
    rms_scale = (max_val / mc).to(tl.float32)
    v_u = (v / (rms_scale + 1e-8)).to(tl.float32)
    
    # 5. Lloyd-Max Bucketize (Audit-Aligned v13.0)
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    if DIST_ID == 1: # 3-bit Laplace (Audit: 2.3697, 1.2460, 0.5300)
        idx = tl.where(v_u > -2.3697, idx + 1, idx)
        idx = tl.where(v_u > -1.2460, idx + 1, idx)
        idx = tl.where(v_u > -0.5300, idx + 1, idx)
        idx = tl.where(v_u >  0.0000, idx + 1, idx)
        idx = tl.where(v_u >  0.5300, idx + 1, idx)
        idx = tl.where(v_u >  1.2460, idx + 1, idx)
        idx = tl.where(v_u >  2.3697, idx + 1, idx)
    else: # 4-bit Gaussian (Audit: 2.4008, 1.8435, ...)
        idx = tl.where(v_u > -2.4008, idx + 1, idx)
        idx = tl.where(v_u > -1.8435, idx + 1, idx)
        idx = tl.where(v_u > -1.4371, idx + 1, idx)
        idx = tl.where(v_u > -1.0993, idx + 1, idx)
        idx = tl.where(v_u > -0.7995, idx + 1, idx)
        idx = tl.where(v_u > -0.5224, idx + 1, idx)
        idx = tl.where(v_u > -0.2582, idx + 1, idx)
        idx = tl.where(v_u >  0.0000, idx + 1, idx)
        idx = tl.where(v_u >  0.2582, idx + 1, idx)
        idx = tl.where(v_u >  0.5224, idx + 1, idx)
        idx = tl.where(v_u >  0.7995, idx + 1, idx)
        idx = tl.where(v_u >  1.0993, idx + 1, idx)
        idx = tl.where(v_u >  1.4371, idx + 1, idx)
        idx = tl.where(v_u >  1.8435, idx + 1, idx)
        idx = tl.where(v_u >  2.4008, idx + 1, idx)

    # 6. SOTA Matrix-Based Packing
    if DO_PACK and BITS < 8:
        i_idx = tl.arange(0, BLOCK_D)[:, None]
        k_idx = tl.arange(0, PACKED_D)[None, :]
        m0 = tl.where(i_idx == 2 * k_idx, 1, 0)
        m1 = tl.where(i_idx == 2 * k_idx + 1, 1 << BITS, 0)
        packed = tl.sum(idx[:, None] * (m0 + m1), axis=0)
        tl.store(OUT_INDICES_ptr + pid * PACKED_D + tl.arange(0, PACKED_D), packed.to(tl.uint8))
    else:
        tl.store(OUT_INDICES_ptr + pid * D + d_offset, idx.to(tl.uint8))

    # 7. Metadata Store
    tl.store(OUT_SCALES_ptr + pid, rms_scale)
    tl.store(OUT_NORMS_ptr + pid, norm)

def fused_quantize(x, all_signs, wht_mat, bits, max_centroid, final_scale, dist_type='gaussian', pack=True, out_indices=None, out_scales=None, out_norms=None, out_rotated=None):
    N, D = x.shape
    device = x.device
    dist_id = 1 if dist_type == 'laplace' else 0
    vals_per_byte = 2 if (bits < 8 and pack) else 1
    packed_d = D // vals_per_byte
    
    # SOTA v12.5: Zero-Allocation Path for CUDA Graphs
    if out_indices is None:
        out_indices = torch.empty((N, packed_d), dtype=torch.uint8, device=device)
    if out_scales is None:
        out_scales = torch.empty((N, 1), dtype=torch.float32, device=device)
    if out_norms is None:
        out_norms = torch.empty((N, 1), dtype=torch.float32, device=device)
    if out_rotated is None:
        out_rotated = torch.empty((N, D), dtype=torch.float32, device=device)
    
    _fused_quant_kernel[(N,)](
        X_ptr=x, SIGNS_ptr=all_signs, WHT_MAT_ptr=wht_mat,
        OUT_INDICES_ptr=out_indices, OUT_SCALES_ptr=out_scales, OUT_NORMS_ptr=out_norms, OUT_ROTATED_ptr=out_rotated,
        N=N, D=D, N_PASSES=all_signs.shape[0], FINAL_SCALE=final_scale,
        DIST_ID=dist_id, BITS=bits, PACKED_D=packed_d, BLOCK_D=D, DO_PACK=pack,
        num_warps=8 
    )
    return out_indices, out_scales, out_norms, out_rotated
