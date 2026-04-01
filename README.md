# 🚀 Kiến Trúc Hợp Nhất: WHT-Driven Dual KV Quantization (TurboQuant++)

Tài liệu này đặc tả kỹ thuật luồng xử lý 6 giai đoạn của hệ thống lượng tử hóa bộ nhớ KV Cache **TurboQuant++**. Hệ thống tận dụng phép biến đổi Walsh-Hadamard (WHT) kết hợp chiến lược lượng tử hóa bất đối xứng (Asymmetric Quantization) để đạt mức nén cực đại (2-4 bits) trong khi bảo toàn điểm Perplexity (PPL) gần tương đương FP16.

---

## 🛠️ Giai đoạn 1: Tiền xử lý Kháng biến (RoPE-Invariant)

Bảo vệ tính toàn vẹn của không gian vector trước khi thực hiện bất kỳ phép biến đổi (xoay/nén) nào. Việc phá vỡ trật tự không gian trước RoPE sẽ làm hỏng hoàn toàn thông tin vị trí chuỗi (sequence information).

* **Thực thi:** Áp dụng Rotary Position Embedding (RoPE) vào Query ($Q$) và Key ($K$) ở định dạng nguyên bản FP16.
* **Công thức Toán học:**
  Giả sử $x_m$ là vector embedding tại vị trí $m$, phép biến đổi RoPE được định nghĩa là một ma trận xoay khối (block-diagonal rotation matrix) $R_{\Theta, m}$:
  $$Q' = Q \cdot R_{\Theta, m}$$
  $$K' = K \cdot R_{\Theta, m}$$
  Trong đó, $R_{\Theta, m}$ bảo toàn không gian tích vô hướng theo khoảng cách vị trí tương đối:
  $$\langle Q'_m, K'_n \rangle = \langle Q_m, K_n \rangle \cos((m-n)\theta) + \dots$$
* **Mục đích:** Tuyệt đối không lượng tử hóa hay nhân ma trận xoay WHT trước bước này.

---

## 🌪️ Giai đoạn 2: Lõi Biến đổi Toán học (WHT + Independent Sign Array)

Thay thế hoàn toàn Head-Wise SVD và IsoQuant bằng phép biến đổi nguyên chuẩn. Do Key chứa các giá trị ngoại lai (outliers) cực lớn, ta cần phân tán chúng đều ra toàn bộ các chiều vector.

* **Thực thi:** Áp dụng ma trận Walsh-Hadamard Transform (WHT) kết hợp với mảng dấu ngẫu nhiên (Sign Array) lên $K'$.
* **Công thức Toán học:**
  Ta định nghĩa ma trận đường chéo chứa các dấu ngẫu nhiên độc lập $S \in \{-1, 1\}^{d \times d}$. Ma trận WHT chuẩn hóa $H_d$ kích thước $d \times d$ được định nghĩa đệ quy:
  $$H_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad H_d = H_{d/2} \otimes H_2$$
  Key sau khi xoay ($K_{rot}$) được tính bằng:
  $$K_{rot} = K' \cdot S \cdot H_d$$
  *(Phép nhân $H_d$ thực chất được tính siêu tốc qua thuật toán Fast Walsh-Hadamard Transform - FWHT với độ phức tạp $O(d \log d)$, chỉ dùng phép cộng/trừ).*
* **Lưu ý:** Value ($V$) được giữ nguyên: $V_{rot} = V$.

---

## ✂️ Giai đoạn 3: Lượng tử hóa Phân ly (Dual K/V LUT)

Phân phối của Key (sắc nhọn) và Value (phân tán đều) là hoàn toàn khác biệt. Cần dùng hai sách mã (Codebook) hoàn toàn độc lập được tối ưu bằng thuật toán K-Means (Lloyd-Max) giảm thiểu sai số toàn phương trung bình (MSE).

* **Thực thi:** Xây dựng hai bảng tra cứu $LUT_K$ và $LUT_V$ (thường là 3-bit, tương đương 8 centroids).
* **Công thức Toán học:**
  Hàm lượng tử hóa tìm centroid gần nhất:
  $$Q_{LUT}(x) = \arg\min_{c_i \in LUT} \| x - c_i \|^2$$
  Áp dụng phân ly:
  $$\hat{K} = Q_{LUT_K}(K_{rot})$$
  $$\hat{V} = Q_{LUT_V}(V_{rot})$$
* **Kết quả:** Lưu trữ các chỉ mục (indices) 3-bit thay vì các giá trị FP16 thực tế.

---

## 🛡️ Giai đoạn 4: Phân nhánh Phục hồi (QJL/DirectSign vs. Thuần MSE)

Đây là điểm dung hợp tinh hoa: Khôi phục lỗi lượng tử (Quantization Error) của Key bằng 1-bit, trong khi "buông lỏng" Value vì hàm Softmax sẽ tự động triệt tiêu sai số.

* **Bộ nhớ Key ($K$) - Chống nhiễu tích vô hướng:**
  Tính phần dư (Residual error): $E_K = K_{rot} - \hat{K}$.
  * **Trường hợp $d = 256$ (Nén MSE + QJL 1-bit):**
      Lưu thêm 1 ma trận dấu của phần dư:
      $$M_{sign} = \text{sign}(E_K) \in \{-1, 1\}$$
      Key được phục hồi bằng:
      $$K_{restored} = \hat{K} + \alpha \cdot M_{sign}$$
      *(Với $\alpha$ là hệ số tỷ lệ co giãn học được. Kết hợp WHT + QJL giúp phương sai nhiễu giảm 11.7 lần).*
  * **Trường hợp $d \le 128$:** Dùng DirectSign tương tự để tiết kiệm chi phí tính toán.
* **Bộ nhớ Value ($V$) - Tối giản hóa:**
  Không dùng bit phục hồi:
  $$V_{restored} = \hat{V}$$

---

## ⚡ Giai đoạn 5: Tối ưu Thực chiến (TurboQuant+ & Sparse V)

Tối ưu hóa luồng suy luận (Inference) dựa trên đặc tính phân bổ sức chú ý (Attention Allocation) của LLM.

* **Boundary V (Bảo vệ Biên):**
  Giữ nguyên định dạng 8-bit cho $L_{first\_2}$ (2 layer đầu) và $L_{last\_2}$ (2 layer cuối) do độ nhạy cảm lỗi cực cao.
* **Sparse V (Giải mã Thưa thớt):**
  Tính ma trận trọng số Attention $A$:
  $$A = \text{Softmax}\left(\frac{Q' \cdot (K_{restored} \cdot H_d^T \cdot S^T)^T}{\sqrt{d}}\right)$$
  Thay vì giải nén toàn bộ Value, áp dụng mặt nạ lọc thưa (Sparse Mask) $\mathcal{M}$:
  $$\mathcal{M}_{i,j} = \begin{cases} 1, & \text{nếu } A_{i,j} > 10^{-6} \\ 0, & \text{ngược lại} \end{cases}$$
  Đầu ra cuối cùng (chỉ giải mã $V$ khi $\mathcal{M}=1$):
  $$O = \sum \left( (A \odot \mathcal{M}) \cdot V_{restored} \right)$$

---

## 🔥 Giai đoạn 6: Nấu chảy bằng Triton (Fused Kernel)

Hợp nhất (Fuse) toàn bộ quy trình trên vào một nhân (Kernel) CUDA/Triton duy nhất để tối ưu hóa IOPS và phá vỡ giới hạn Memory-Bound.

**Vòng lặp trong Fused Kernel (Mỗi Block xử lý 1 Head):**
1. **Load:** Đọc chỉ mục LUT (3-bit) và bit phục hồi QJL (1-bit) của Key từ VRAM vào SRAM.
2. **Dequantize $K$:** Áp dụng công thức $K_{restored} = LUT_K[\text{idx}] + \alpha \cdot \text{sign}$.
3. **Inverse WHT:** Giải xoay siêu tốc $K' = K_{restored} \cdot H_d \cdot S$ ngay trên SRAM.
4. **Compute Attention:** Tính $P = Q' \cdot K'^T$ và Softmax.
5. **Sparse V Trigger:** Kiểm tra $P_i > 10^{-6}$.
6. **Dequantize $V$:** Nếu thỏa mãn, đọc chỉ mục LUT Value từ VRAM, tra bảng $LUT_V$ để lấy $V_{restored}$.
7. **Accumulate:** Tính tổng trọng số $O += P_i \cdot V_{restored}$ và xả (store) kết quả cuối cùng ra VRAM.



rsync -a --exclude='.venv' --exclude='__pycache__' /mnt/c/Users/ADMIN/OneDrive/Desktop/turboquant_pp/ ~/projects/turboquant_pp/
cd ~/projects/turboquant_pp
uv sync
uv run python -m pytest -v