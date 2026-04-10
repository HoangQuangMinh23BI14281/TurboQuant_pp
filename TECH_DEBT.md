# 🛠️ TurboQuant++ Technical Debt Tracker

## 🚀 Recent Breakthroughs (V10.0)
- **Status:** **PERFORMANCE RESTORED** (Goal: >50 tokens/s | Current: **52.41 tokens/s**)
- **Key Optimization:** Implemented Fused Triton Quantization (WHT + Norm + Max + Bucketize + Packing) in a single kernel.

## 🔴 Critical Issues (High Priority)
- [ ] **Multi-Batch Support**: Fused kernels are currently optimized for `batch_size=1`. Scaling to batch > 1 requires updating `fused_quant_kernel`.
- [ ] **FlashAttention-3 Integration**: The current paged attention is based on FA2 patterns. FA3 could push speed beyond 80 tps.

## 🟡 Improvement Items (Medium Priority)
- [ ] **Fused Residuals**: Move the remaining Python code in `TurboQuantProd` (Key Residuals) into a separate Triton kernel.
- [ ] **Quantization-Aware Training (QAT)**: Explore QAT to further lower PPL from 20.09 to <15.0.

## ✅ Resolved Debt
- [x] **Host-Device Sync Stalls**: Eliminated all `.item()` and codebook access syncs. (V9.2)
- [x] **Contiguous Copy Overhead**: Implemented "Contiguous Shield" to ensure 0 redundant copies. (V9.4)
- [x] **Python Dispatch BottleNeck**: Resolved by Fused v10.0 Kernel. (V10.0)
- [x] **Double Quant/Dequant in Prefill Path**: Resolved via fused quantized prefill kernel. (V10.0)
- [x] **CUDA Graph Initialization Latency**: Resolved by moving static allocation to `warmup()` call. (V10.0)

---
*Last Updated: 2026-04-10. This registry must be reviewed before each release cycle.*
