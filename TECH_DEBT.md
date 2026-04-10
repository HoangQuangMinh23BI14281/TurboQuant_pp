# Danh sách Technical Debt Tồn Đọng (TECH_DEBT.md)

Tài liệu này liệt kê các vấn đề kỹ thuật khẩn cấp cần được giải quyết trong dự án TurboQuant++ nhằm đảm bảo hiệu năng và tính ổn định.

## 1. ⚠️ Tính toán chặn luồng trong `manager.py` (Critical Bottleneck)

- **Vị trí:** `src/turboquant/cache/manager.py` -> `TurboQuantKVCache.append`
- **Mô tả:** Hệ thống hiện tại sử dụng vòng lặp Python thuần túy `for i in range(seq_len):` để sao chép metadata và chunk tensors của Quantizer vào Block Pool. Overhead của code host (CPU) đè bẹp hoàn toàn hiệu năng GPU khi context prefill lớn (e.g. seq_len = 8192).
- **Hành động:** 
  - Gỡ bỏ vòng lặp `for`.
  - Thay thế bằng các phép toán Vectorization (`torch.scatter`, `torch.index_copy_`).
  - Tách logic thành các sub-functions (`_append_prefill`, `_append_decode`).

## 2. 🚨 Truy cập bộ nhớ không an toàn trong Triton Kernels (Security/Stability Risk)

- **Vị trí:** `src/turboquant/kernels/paged_fused.py`
- **Mô tả:** Triton scripts đọc độ dài tokens (`N = tl.load(NUM_TOKENS_ptr)`) và fetch block ID trên base memory pointer. Các kernel này thiếu Assertions hoặc cơ chế kiểm soát hard-boundary từ Python. Khi `N` vượt qua số lượng block được allocate hoặc `MAX_BLOCKS` (mặc định 1024), kernel trỏ đến địa chỉ không hợp lệ, dẫn tới CUDA Segment Violation hoặc hỏng trạng thái RAM đồ họa.
- **Hành động:**
  - Bổ sung validation của `seq_len` tại tầng Python trước khi Launch Kernel.
  - Sửa lại mask loading logic trong Triton để không đọc lố `len(BLOCK_TABLE)`.

## 3. 🐢 Overhead tính năng GPU Scatter tại Decode Phase (Performance Leak)

- **Vị trí:** `src/turboquant/layers/attention_layer.py` 
- **Mô tả:** Quá trình decode từng token một trên nhánh FP16 thực hiện lệnh `view()`, `expand()` và `scatter_()` liên tục của Torch. Các thao tác này phát sinh độ trễ launch kernels (launch latency) đôi khi còn tốn chi phí hơn cả bước Attention thông thường, khiến Decode T-1 quá chậm.
- **Hành động:**
  - Tiền phân bổ Index/Slices buffer thay vì tạo tensor index động với `.expand()`.
  - Nếu có thể, viết custom micro-kernel trên Triton để copy-paste các phần tử một cách trực tiếp tránh metadata overloads của PyTorch.

---
*(Tài liệu này được sinh tự động. Đội ngũ cần xử lý các gạch đầu dòng này trước bản release kế tiếp).*
