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
│       ├── ops/                     # [COMPLETED] WHT, Cascaded Rotation, RoPE
│       ├── quant/                   # [COMPLETED] Lloyd-Max, Quantizer
│       ├── kernels/                 # [CURRENT] Triton fused kernels
│       │   ├── fused_attention.py   # High-level Attention wrapper
│       │   └── triton_attention.py  # TurboQuant_prod Fused Kernel (SRAM-block-diagonal)
│       └── cache/                   # [NEXT] KV Cache manager
├── tests/
└── benchmarks/
```

---

## 2.1. Advanced Rotation Strategies (Structured Random Matrices)

Để đạt được sự hội tụ toán học về ma trận trực giao Haar-random (Google) mà vẫn giữ tốc độ $O(d \log d)$, chúng ta triển khai:

1. **Cascaded WHT (Kẹp chả)**: 
   - $K_{rot} = K \cdot S_1 \cdot \text{H}_d \cdot S_2 \cdot \text{H}_d$
   - Sử dụng 2 lớp Walsh-Hadamard xen kẽ 2 mảng dấu ngẫu nhiên độc lập.
   - Giúp phân phối dữ liệu "nhuyễn" hơn, tiệm cận phân phối Beta hoàn hảo.

2. **Block-Diagonal Random Rotation (ISOQuant style)**:
   - Thay vì $d \times d$ matmul, thực hiện nhân các khối nhỏ (32x32 hoặc 64x64) trực giao.
   - Ưu điểm: Nạp vừa SRAM và tận dụng **Tensor Cores** trong Triton kernel.

3. **Data-aware Sign Array**:
   - Sử dụng phân tích Offline (SVD) trên calibration set để xác định vị trí outlier.
   - Thiết kế $S$ để triệt tiêu các kênh (channels) có biên độ lớn nhất trước khi qua WHT.

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
> **Chiến lược Fused QJL**: Tuyệt đối không giải nén QJL ra thành vector. Việc cộng gộp được thực hiện trực tiếp ở cấp độ phép tính tích vô hướng (inner product) bên trong kernel Triton.
> - **MSE Term**: $||k|| \cdot \sum (q_{\text{rot}} \cdot \text{LM}(idx))$
> - **QJL Term**: $\text{qjl\_scale} \cdot ||r|| \cdot \sum (q_{\text{sketch}} \cdot \text{signs})$

| Hạng mục | Chi tiết |
|---|---|
| **Mục tiêu** | Đạt hiệu suất Dot Product tối đa với QJL bù lỗi trực tiếp trong Attention |
| **Output** | `src/turboquant/kernels/triton_attention.py` — Nhân Triton hợp nhất thực thụ. <br> `src/turboquant/kernels/fused_attention.py` — Wrapper cho Attention module. |
| **Tasks** | 1. **Query Pre-processing**: Rotate query for MSE and sketch query for QJL once per decode step. <br> 2. **MSE Inner Product**: Implement Triton kernel to dot rotated query with centroids fetched from packed indices. <br> 3. **QJL Inner Product**: Implement Triton kernel to dot sketched query with packed 1-bit signs. <br> 4. **Fused Attention**: Implement a single Flash-Attention style kernel that combines both terms and aggregates values. |

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
