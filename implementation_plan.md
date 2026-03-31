# TurboQuant++ — Implementation Plan

## 1. Tổng quan Dự án

**TurboQuant++** là hệ thống lượng tử hóa KV Cache dựa trên phép biến đổi Walsh-Hadamard (WHT),
kết hợp chiến lược lượng tử hóa bất đối xứng (Asymmetric Quantization) để đạt mức nén 2-4 bits
mà vẫn bảo toàn Perplexity gần tương đương FP16.

Tài liệu kiến trúc gốc định nghĩa **6 giai đoạn**:

| Giai đoạn | Tên | Tóm lược |
|:-:|---|---|
| 1 | RoPE-Invariant Preprocessing | Áp dụng RoPE lên Q, K trước mọi biến đổi |
| 2 | WHT + Independent Sign Array | Phân tán outliers bằng Walsh-Hadamard + sign flip |
| 3 | Dual K/V LUT Quantization | K-Means codebook riêng biệt cho K và V |
| 4 | Recovery Branch (QJL / DirectSign) | Phục hồi lỗi Key bằng 1-bit residual sign |
| 5 | Sparse V & Boundary Protection | Giữ boundary layers 8-bit, sparse dequant V |
| 6 | Triton Fused Kernel | Hợp nhất toàn bộ pipeline vào 1 kernel |

---

## 2. Cấu trúc Thư mục Đề xuất (Folder Tree)

```
turboquant_pp/
├── pyproject.toml
├── src/
│   └── turboquant/
│       ├── ops/                     # [COMPLETED] WHT, Sign Array, RoPE
│       ├── quant/                   # [COMPLETED] Lloyd-Max, Quantizer
│       ├── kernels/                 # [CURRENT] Triton fused kernels
│       │   ├── fused_attention.py   # High-level Attention wrapper
│       │   └── triton_attention.py  # TurboQuant_prod Fused Kernel
│       └── cache/                   # [NEXT] KV Cache manager
├── tests/
└── benchmarks/
```

---

## 3. Lộ trình Milestones

### Milestone 1 — Toán Tử Cơ sở (ops/) [COMPLETED]
✅ **WHT, Sign Array, RoPE** đã được triển khai và pass 100% tests.

---

### Milestone 2 — Lloyd-Max & Dimension-Aware Quantization [COMPLETED]
✅ **Lloyd-Max Solver (Analytical)**, Bit-exact tables (1-8 bits), và **TurboQuantizer** (Padding, Bit-packing) đã hoàn tất và pass 100% tests.

---

### Milestone 3 — Triton Fused Attention Core ("TurboQuant_prod") [CURRENT]

**Covers: Giai đoạn 4 (Integrated QJL) + Giai đoạn 6 (Fused Kernel)**

> [!IMPORTANT]
> **Chiến lược Fused QJL**: Không sử dụng QJL như một bộ nhớ đệm bù lỗi rời rạc. Toàn bộ logic giải nén MSE (centroids) và QJL (signs) phải được thực hiện SONG SONG bên trong một Triton Kernel duy nhất.

| Hạng mục | Chi tiết |
|---|---|
| **Mục tiêu** | Đạt hiệu suất Dot Product tối đa với QJL bù lỗi trực tiếp trong Attention |
| **Output** | `src/turboquant/kernels/triton_attention.py` — Nhân Triton hợp nhất thực thụ. <br> `src/turboquant/kernels/fused_attention.py` — Wrapper cho Attention module. |
| **Tasks** | 1. **Parallel Stream Processing**: Tải interleaved MSE indices và QJL signs từ DRAM. <br> 2. **Register-level Unpacking**: Unpack bit QJL và tra centroids đồng thời trong cùng một vòng lặp. <br> 3. **Online Softmax**: Tích hợp Flash-Attention style softmax để tránh ghi scores trung gian. |

**Verification:**
- `test_fused_vs_ref`: So sánh output nhân Triton với PyTorch reference (từ Milestone 2) với sai số $< 10^{-4}$.
- `test_quant_prod_latency`: Đảm bảo kernel chạy nhanh hơn phương pháp 2-pass.

---

### Milestone 4 — Full Model Integration & Evaluation

**Covers: Giai đoạn 5 (Sparse V) + 100% End-to-End Pipeline**

| Hạng mục | Chi tiết |
|---|---|
| **Mục tiêu** | Tích hợp vào Llama/Gemma-2 và đo đạc Perplexity thực tế |
| **Tasks** | 1. Triển khai `Sparse V` decoding dựa trên attention weights. <br> 2. Xây dựng `TurboQuantKV` manager quản lý cache lifetime. <br> 3. Chạy đo PPL trên WikiText-2. |

---

## 8. Verification Plan (Summary)

### Automated Tests
- CI chạy toàn bộ 67+ tests hiện có + các bài test Triton mới.
- Kiểm tra tính đúng đắn trên cả d=64, 128, 256.

### Manual Verification
- Benchmarking wall-clock time trên GPU A100/H100 hoặc RTX 4090.
- So sánh dung lượng bộ nhớ KV Cache thực tế so với FP16.
