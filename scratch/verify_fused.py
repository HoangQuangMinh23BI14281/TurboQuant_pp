import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from turboquant.quant.key_quantizer import TurboQuantMSE
from turboquant.kernels.quant_fused import fused_quantize

def verify():
    device = "cuda"
    dim = 64
    bits = 4
    
    # 1. Setup Quantizer
    print("Initializing Quantizer...")
    quantizer = TurboQuantMSE(dim=dim, block_size=dim, bits=bits, n_rotation_passes=2).to(device)
    quantizer.eval()
    
    # 2. Input
    x = torch.linspace(-2, 2, dim, device=device).reshape(1, 1, 1, dim)
    
    # 3. Modular Path
    print("Running Modular Path...")
    from turboquant.quant.lloyd_max import lloyd_max_quantize, compute_lloyd_max_codebook
    cb_mod = compute_lloyd_max_codebook(bits, dist='gaussian')
    bounds_mod = cb_mod['boundaries'].to(device).float() # All 15 finite boundaries
    print(f"Modular Max Centroid: {cb_mod['max_centroid']}")
    print(f"Modular Boundaries (15): {bounds_mod.tolist()}")



    
    with torch.no_grad():
        # Step 1: Norm/Rotation
        norms = torch.norm(x.float(), p=2, dim=-1, keepdim=True)
        x_unit = x.float() / (norms + 1e-8)
        
        # Rotation
        x_rot = x_unit.to(torch.float32)
        h_mat = quantizer.rotation.wht_mat.to(torch.float32)
        for i in range(quantizer.n_rotation_passes):
            x_rot = x_rot * quantizer.rotation.all_signs[i].to(torch.float32)
            x_rot = torch.matmul(x_rot, h_mat)
        x_rot = x_rot * quantizer.rotation.final_scale.to(torch.float32)
        
        # Max/Scale
        x_max = torch.max(torch.abs(x_rot), dim=-1, keepdim=True).values
        scales = x_max / quantizer.max_centroid
        
        # Unitize matching kernel
        v_unit_mod = x_rot / (scales + 1e-8)
        # Match sum(v > bounds) logic
        # For [-inf, b1...b15, +inf], bucketize returns N in [1, 16]
        # Our bounds_mod is [b1...b15]. Sum count is [0, 15]
        diffs_mod = v_unit_mod[:, :, :, :, None] > bounds_mod[None, None, None, None, :]
        indices_mod_raw = torch.sum(diffs_mod.int(), dim=-1)

    # 4. Fused Path
    print("Running Fused Path...")
    all_signs = quantizer.rotation.all_signs
    wht_mat = quantizer.rotation.wht_mat.float()
    final_scale = quantizer.rotation.final_scale.item()
    
    q_fused_indices, q_fused_scales, q_fused_norms, q_fused_rotated = fused_quantize(
        x.float().reshape(-1, dim),
        all_signs,
        wht_mat,
        bits,
        quantizer.max_centroid,
        final_scale,
        pack=False
    )
    
    # 5. Compare
    print("\n--- Detailed Comparison ---")
    print(f"Norm Max Diff: {torch.abs(norms.flatten() - q_fused_norms.flatten()).max().item():.6e}")
    print(f"Rotated Tensor Max Diff: {torch.abs(x_rot.flatten() - q_fused_rotated.flatten()).max().item():.6e}")
    
    # Indices
    from turboquant.quant.quant_base import unpack_indices
    # Only unpack if the kernel returned a packed tensor
    if q_fused_indices.shape[-1] < dim:
        idx_fused_unpacked = unpack_indices(q_fused_indices, bits, dim)
        idx_fused = idx_fused_unpacked.flatten().cpu()
    else:
        idx_fused = q_fused_indices.flatten().cpu()
    
    idx_mod = indices_mod_raw.flatten().cpu()
    mismatch_indices = (idx_fused != idx_mod).nonzero().flatten()


    mismatch_count = mismatch_indices.numel()
    print(f"Index Mismatches: {mismatch_count} / {dim}")
    
    if mismatch_count > 0:
        first_idx = mismatch_indices[0].item()
        print(f"\n--- Debug First Mismatch (Element {first_idx}) ---")
        print(f"v_unit_mod: {v_unit_mod.flatten()[first_idx].item():.10f}")
        # Recalc fuse v_unit
        v_unit_fused = q_fused_rotated.flatten() / (q_fused_scales.flatten() + 1e-8)
        print(f"v_unit_fused: {v_unit_fused[first_idx].item():.10f}")
        print(f"Mod Index: {idx_mod[first_idx].item()}")
        print(f"Fused Index: {idx_fused[first_idx].item()}")
    else:
        print("✅ SUCCESS: ALL INDICES MATCH BIT-EXACTLY!")


if __name__ == "__main__":
    verify()
