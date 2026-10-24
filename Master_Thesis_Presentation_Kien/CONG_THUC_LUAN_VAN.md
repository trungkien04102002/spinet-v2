# Toàn bộ công thức trong luận văn, giải thích dễ hiểu

Tài liệu ôn bảo vệ (02/11/2026). Đi qua **đủ 32 công thức có đánh số** trong luận văn đã nộp
(commit `ef23e38`), theo thứ tự chương, cộng với các công thức chỉ nhắc trong lời văn hoặc trên
slide (softmax, cô-sin, GAP, GELU, SupCon, Grad-CAM), và **cách tính từng con số trong các bảng
đã chiếu**.

Mỗi công thức có bốn phần:
- **Làm gì:** một câu, đủ để trả lời khi bị hỏi nhanh.
- **Ký hiệu:** nghĩa của từng chữ.
- **Ví dụ số:** tính tay được.
- **Vì sao dùng / dễ bị hỏi:** lý do chọn và câu hỏi hay gặp.

Số trong ngoặc như (4.7) là số thứ tự công thức trong chương đó của luận văn (đếm theo thứ tự
xuất hiện, có thể lệch với số in trong PDF nếu Overleaf đánh khác).

Đọc kèm:
- `GIAI_THICH_METRIC_VA_CONG_THUC.md`: giải thích metric dài hơn, ví dụ ma trận nhầm lẫn.
- `SCRIPT_BAO_VE_LVTN.md`: khối 📘 dưới trang 9, 17, 18.

---

## Mục lục

0. Ký hiệu toán dùng chung
1. Chương 2: nền tảng (10 công thức + ReLU, softmax)
2. Chương 3: cross-entropy có trọng số của Ning Shen (1 công thức)
3. Chương 4: mô hình đề xuất (10 công thức + GAP, GELU, cô-sin, SupCon)
4. Chương 5: chỉ số đánh giá (11 công thức)
5. **Cách tính các con số trong từng bảng đã chiếu**
6. Grad-CAM (trên slide, không đánh số trong luận văn)
7. Bảng tra một dòng cho mọi công thức

---

## 0. Ký hiệu toán dùng chung

| Ký hiệu | Đọc là | Nghĩa |
|---|---|---|
| $\sum_{i=1}^{n} x_i$ | tổng | $x_1 + x_2 + \dots + x_n$ |
| $\exp(z) = e^z$ | mũ e | luôn dương; $e^0 = 1$, $e^1 \approx 2.72$, $e^2 \approx 7.39$ |
| $\log(p)$ | log tự nhiên | $\log 1 = 0$; $p$ càng nhỏ thì $\log p$ càng âm |
| $a^\top b$ | tích vô hướng | nhân từng cặp rồi cộng: $(1,2)^\top(3,4) = 3 + 8 = 11$ |
| $\|f\|_2$ | độ dài vector | $\sqrt{f_1^2 + \dots + f_n^2}$; vector $(3,4)$ dài 5 |
| $[a; b]$ | ghép | nối hai vector: $[(1,2);(3)] = (1,2,3)$ |
| $\otimes$ | nhân từng phần tử | nhân từng ô với ô tương ứng |
| $\sigma(z)$ | sigmoid | $1/(1+e^{-z})$, ép số bất kỳ về (0, 1) |
| $\arg\max_c$ | vị trí lớn nhất | trả về **vị trí** $c$ có giá trị lớn nhất, không phải giá trị đó |
| $\mathbb{R}^{C \times D \times H \times W}$ | kích thước | khối có $C$ kênh, $D$ lát, cao $H$, rộng $W$ |
| $\hat{y}$ | y mũ | giá trị **dự đoán** (khác $y$ là giá trị thật) |

---

## 1. Chương 2: công thức nền tảng

Chương 2 là kiến thức nền. Hội đồng ít hỏi sâu, nhưng có thể hỏi để kiểm tra hiểu cơ bản.

### (2.1) Perceptron: phép cộng có trọng số

$$a = \sum_{i=1}^{n} w_i x_i + w_0$$

- **Làm gì:** một nơ-ron nhân mỗi đầu vào với một trọng số rồi cộng lại, thêm một hằng số.
- **Ký hiệu:** $x_i$ đầu vào; $w_i$ trọng số học được (đầu vào nào quan trọng thì trọng số lớn);
  $w_0$ độ lệch (bias), cho phép dịch ngưỡng; $a$ là tổng.
- **Ví dụ:** $x = (1, 2)$, $w = (0.5, -1)$, $w_0 = 0.2$ → $a = 0.5 - 2 + 0.2 = -1.3$.
- **Liên hệ luận văn:** mọi lớp "Linear" trong mô hình (Fusion MLP, đầu tuyến tính của
  SpineNetV2) đều là nhiều perceptron chạy song song.

### (2.2) Perceptron: so ngưỡng

$$\hat{y} = \begin{cases} 1 & f(a) \ge \text{threshold} \\ 0 & f(a) < \text{threshold} \end{cases}$$

- **Làm gì:** đưa $a$ qua hàm kích hoạt $f$, vượt ngưỡng thì ra 1, không thì ra 0.
- **Ví dụ:** tiếp ví dụ trên, sigmoid$(-1.3) \approx 0.21$, ngưỡng 0.5 → $\hat{y} = 0$.
- **Vì sao cần hàm kích hoạt phi tuyến:** không có nó thì chồng bao nhiêu lớp tuyến tính cũng chỉ
  bằng một lớp tuyến tính, mạng không học được quan hệ phức tạp.

### ReLU (trong lời văn mục CNN)

$$\text{ReLU}(x) = \max(0, x)$$

- **Làm gì:** giữ số dương, đổi số âm thành 0. Rẻ, và không làm đạo hàm nhỏ dần như sigmoid.
- **Ví dụ:** ReLU$(-3) = 0$, ReLU$(2) = 2$.

### (2.3) RNN encoder: cập nhật trạng thái ẩn

$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t)$$

- **Làm gì:** đọc chuỗi từng phần tử một; trạng thái mới $h_t$ = trộn trạng thái cũ $h_{t-1}$
  với phần tử đang đọc $x_t$.
- **Ký hiệu:** $W_{hh}$ trọng số "nhớ cái cũ", $W_{xh}$ trọng số "đọc cái mới", $f$ hàm phi tuyến.
- **Liên hệ luận văn:** chỉ là kiến thức nền cho phần attention; mô hình của em **không** dùng
  RNN.

### (2.4) Vector ngữ cảnh

$$C = h_T$$

- **Làm gì:** sau khi đọc hết $T$ phần tử, lấy trạng thái cuối làm bản tóm tắt cả chuỗi.
- **Điểm yếu (dẫn tới attention):** nén cả chuỗi dài vào một vector cố định là "nút cổ chai".

### (2.5) Khởi tạo decoder

$$s_0 = C$$

- **Làm gì:** decoder bắt đầu từ bản tóm tắt của encoder.

### (2.6) RNN decoder: cập nhật trạng thái

$$s_t = g(W_{ss} s_{t-1} + W_{ys} y_{t-1})$$

- **Làm gì:** giống (2.3) nhưng đầu vào là **chữ vừa sinh ra ở bước trước** $y_{t-1}$ (tự hồi
  quy).

### (2.7) Decoder dự đoán chữ tiếp theo

$$P(y_t \mid y_1, \dots, y_{t-1}, C) = \text{softmax}(W_{sy} s_t)$$

- **Làm gì:** biến trạng thái thành xác suất cho từng chữ trong từ điển, chọn chữ có xác suất cao.
- **Softmax** là cùng một hàm dùng ở đầu cô-sin của Hybrid, xem (4.9).

### Softmax (dùng xuyên suốt)

$$\text{softmax}(z)_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$$

- **Làm gì:** biến một dãy điểm số bất kỳ thành xác suất: dương hết, cộng lại bằng 1, điểm cao
  hơn thì xác suất cao hơn.
- **Ví dụ:** điểm $(2, 1, 0)$ → $e^z = (7.39, 2.72, 1.00)$, tổng 11.11 → xác suất
  $(0.67, 0.24, 0.09)$.
- **Tính chất hay bị hỏi:** cộng cùng một số vào mọi điểm thì kết quả không đổi; softmax **giữ
  thứ tự**, nên argmax trước hay sau softmax là như nhau.

### (2.8) Attention trong encoder–decoder

$$c_i = \sum_{j=1}^{T} \alpha_{ij} h_j$$

- **Làm gì:** thay vì chỉ dùng $h_T$, mỗi bước giải mã lấy **trung bình có trọng số của mọi
  trạng thái** encoder; trọng số $\alpha_{ij}$ cho biết chữ đầu vào $j$ quan trọng bao nhiêu khi
  sinh chữ $i$.
- **Ví dụ:** $\alpha = (0.7, 0.2, 0.1)$ → $c = 0.7 h_1 + 0.2 h_2 + 0.1 h_3$.
- **Liên hệ luận văn:** **slice attention pooling (4.4)** dùng đúng ý tưởng này, chỉ khác là
  "chuỗi" ở đây là 3 lát ảnh.

### (2.9) CBAM 2D: chú ý theo kênh

$$M_c(F) = \sigma\big(\text{MLP}(\text{AvgPool}(F)) + \text{MLP}(\text{MaxPool}(F))\big)$$

- **Làm gì:** cho mỗi kênh đặc trưng một trọng số 0–1: kênh hữu ích giữ lại, kênh nhiễu làm mờ.
- Giải thích đầy đủ ở bản 3D (4.1), vì luận văn dùng bản 3D.

### (2.10) CBAM 2D: chú ý theo không gian

$$M_s(F') = \sigma\big(\text{Conv}_{7\times7}([\text{AvgPool}(F'); \text{MaxPool}(F')])\big)$$

- **Làm gì:** cho mỗi vị trí trên ảnh một trọng số 0–1: vùng tổn thương tô đậm, nền làm mờ.
- Bản 3D ở (4.2).

---

## 2. Chương 3: công thức của Ning Shen

### (3.1) Cross-entropy có trọng số

$$L_{WCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} w_c \, y_{i,c} \log(\hat{y}_{i,c})$$

- **Làm gì:** cross-entropy thông thường, nhưng lỗi ở lớp hiếm bị phạt nặng hơn theo hệ số $w_c$.
- **Ký hiệu:** $N$ số mẫu; $C$ số lớp; $y_{i,c} = 1$ nếu mẫu $i$ thuộc lớp $c$ (còn lại 0);
  $\hat{y}_{i,c}$ xác suất mô hình gán; $w_c$ trọng số lớp, Ning Shen dùng **1 : 2 : 4** cho
  Normal/Mild : Moderate : Severe.
- **Ví dụ:** một ca Severe mà mô hình chỉ cho xác suất 0.2: cross-entropy thường phạt
  $-\log 0.2 = 1.61$; có trọng số thì phạt $4 \times 1.61 = 6.44$.
- **So với luận văn:** Ning Shen dùng tỉ lệ cố định chọn tay; luận văn tính trọng số từ số mẫu
  thật (4.8) và cộng thêm Focal (4.7) để giảm mẫu dễ. Hai cách cùng mục đích.

---

## 3. Chương 4: công thức của mô hình đề xuất

Đây là phần **chắc chắn bị hỏi**. Thứ tự đi đúng theo dòng dữ liệu: ảnh → nhánh CBAM (4.1–4.3)
→ nhánh BiomedCLIP (4.4) → hợp nhất (4.5) → chấm nhãn (4.6, 4.9) → hàm mất mát (4.7, 4.8, 4.10).

### (4.1) CBAM-3D: chú ý theo kênh

$$M_c(F) = \sigma\big(\text{MLP}(\text{AvgPool}_{3D}(F)) + \text{MLP}(\text{MaxPool}_{3D}(F))\big) \in \mathbb{R}^{C \times 1 \times 1 \times 1}$$

- **Làm gì:** chấm điểm "kênh nào quan trọng". Ra **một số cho mỗi kênh**.
- **Từng bước:**
  1. $\text{AvgPool}_{3D}$: mỗi kênh (một khối $D \times H \times W$) gộp thành 1 số bằng cách lấy
     trung bình. $\text{MaxPool}_{3D}$: lấy giá trị lớn nhất. Ra hai vector, mỗi vector $C$ số.
  2. Cả hai đi qua **cùng một** MLP (dùng chung trọng số), lớp ẩn $C/16$ (ví dụ 512 kênh → 32 →
     512).
  3. Cộng hai kết quả, qua sigmoid $\sigma$ → mỗi kênh một số trong (0, 1).
- **Ví dụ:** 4 kênh "bờ đĩa lồi / dịch não tủy / mỡ dưới da / nhiễu" nhận 0.92 / 0.85 / 0.10 /
  0.05 → hai kênh đầu giữ, hai kênh sau gần như tắt.
- **Vì sao cả avg lẫn max:** avg thấy mức chung, max thấy điểm phản ứng mạnh nhất (thường là tổn
  thương nhỏ). Tổn thương nhỏ dễ bị avg làm loãng nên cần max.
- **Vì sao MLP thu gọn 16 lần:** ít tham số, vẫn học được quan hệ giữa các kênh.

### (4.2) CBAM-3D: chú ý theo không gian

$$M_s(F') = \sigma\big(\text{Conv}_{7\times7\times7}([\text{AvgPool}_c(F'); \text{MaxPool}_c(F')])\big) \in \mathbb{R}^{1 \times D \times H \times W}$$

- **Làm gì:** chấm điểm "vị trí nào quan trọng". Ra **một số cho mỗi vị trí 3D**.
- **Từng bước:**
  1. Tại mỗi vị trí $(d, h, w)$, gộp **theo chiều kênh**: trung bình mọi kênh và lớn nhất mọi kênh
     → 2 bản đồ 3D.
  2. Ghép thành khối 2 kênh, tích chập $7 \times 7 \times 7$ (nhìn một vùng lân cận rộng), ra 1
     kênh.
  3. Sigmoid → mặt nạ 3D, mỗi ô trong (0, 1).
- **Ví dụ:** hẹp lỗ liên hợp trái: vùng lỗ liên hợp ở các lát bên trái ≈ 0.9, ống sống lát giữa
  ≈ 0.5, nền ≈ 0.1.
- **Điểm khác 2D (đóng góp của luận văn):** mặt nạ có chiều lát $D$, **mỗi lát có trọng số
  riêng**, nên tổn thương chỉ nằm ở 1–2 lát vẫn được tô đậm.
- **Vì sao kernel 7:** theo bài gốc CBAM; kernel lớn nhìn được ngữ cảnh rộng hơn.

### (4.3) CBAM-3D: áp hai mặt nạ

$$F' = M_c(F) \otimes F, \qquad F'' = M_s(F') \otimes F'$$

- **Làm gì:** nhân mặt nạ kênh vào đặc trưng, rồi nhân tiếp mặt nạ không gian. Kênh trước, không
  gian sau.
- **Ví dụ "pha loãng tín hiệu":** tổn thương chiếm 2% khối. Gộp trung bình cuối mạng (GAP) thì
  tổn thương chỉ góp 2%. Có mặt nạ (tổn thương 1, nền 0.1): $0.02 / (0.02 + 0.98 \times 0.1)
  \approx 17\%$, mạnh hơn khoảng 8 lần.
- **Vì sao kênh trước:** bài gốc CBAM thử các cách sắp xếp, "kênh → không gian nối tiếp" tốt
  nhất. Trực giác: chọn đúng loại đặc trưng rồi mới tìm vị trí của nó.

### GAP: gộp trung bình toàn cục (lời văn)

$$f_{cbam} = \frac{1}{D H W}\sum_{d,h,w} F''_{:,d,h,w}$$

- **Làm gì:** sau tầng cuối (512 kênh), lấy trung bình mỗi kênh trên toàn khối → vector
  $f_{cbam}$ 512 chiều.

### (4.4) Slice attention pooling của nhánh BiomedCLIP

$$\alpha_i = \frac{\exp(w_a^\top z_i)}{\sum_j \exp(w_a^\top z_j)}, \qquad f_{bmc} = \sum_i \alpha_i z_i$$

- **Làm gì:** BiomedCLIP chỉ đọc ảnh 2D, nên mã hóa 3 lát giữa (lát 4, 5, 6 trên 9) thành 3
  vector, rồi gộp thành 1 vector bằng trọng số học được.
- **Ký hiệu:** $z_i$ vector 512 chiều của lát $i$; $w_a$ vector 512 chiều học được ("lát nào đáng
  chú ý"); $w_a^\top z_i$ điểm của lát $i$; $\alpha_i$ trọng số, cộng lại bằng 1 (softmax).
- **Ví dụ:** điểm 2 / 1 / 0 → $\alpha$ = 0.67 / 0.24 / 0.09 → $f_{bmc} = 0.67 z_4 + 0.24 z_5 +
  0.09 z_6$.
- **Dễ bị hỏi:**
  - $\alpha$ khác nhau ở mỗi đĩa đệm (vì $z_i$ khác), $w_a$ thì cố định sau khi học.
  - Học $w_a$ không cần nhãn lát: đạo hàm từ hàm mất mát chấm mức độ tự đẩy $w_a$.
  - Chỉ 513 tham số ($w_a$ 512 + bias 1; bias không ảnh hưởng vì softmax bất biến khi cộng hằng).
  - Trung bình đều là trường hợp riêng $\alpha = 1/3$; mô hình vẫn học ra được nếu đó là tốt nhất.
- Chi tiết hơn: `GIAI_THICH_METRIC_VA_CONG_THUC.md` mục 15.

### (4.5) Fusion MLP và chuẩn hóa

$$f_{img} = W_2\,\phi\big(W_1 [f_{cbam}; f_{bmc}] + b_1\big) + b_2, \qquad \hat{f}_{img} = \frac{f_{img}}{\|f_{img}\|_2}$$

- **Làm gì:** ghép hai vector (512 + 512 = 1024), cho qua MLP hai lớp (1024 → 768 → 512) để trộn
  thông tin hai nhánh, rồi kéo độ dài về 1.
- **Ký hiệu:** $W_1$ ($768 \times 1024$), $W_2$ ($512 \times 768$) ma trận học được; $b_1, b_2$
  bias; $\phi$ = GELU; sau lớp 1 có dropout 0.1.
- **Vì sao ra 512:** bằng số chiều vector văn bản của BiomedCLIP, để so trực tiếp ở bước sau.
- **Vì sao chuẩn hóa về 1:** để tích vô hướng **đúng bằng cô-sin**, chỉ đo hướng. Ví dụ $(3, 4)$
  dài 5 → $(0.6, 0.8)$ dài 1.
- **Vì sao ghép mà không cộng:** cộng thì chiều thứ $i$ của hai vector phải cùng nghĩa, mà hai
  vector từ hai mạng khác nhau. Ghép rồi để MLP học cách trộn thì không cần giả định đó.
- **Số tham số:** $768 \times 1024 + 512 \times 768 \approx 786\text{K} + 393\text{K} \approx$
  **1.18 triệu**.

### GELU (ký hiệu $\phi$)

- **Làm gì:** hàm kích hoạt giống ReLU nhưng "mềm": số âm không bị cắt cụt về 0 đột ngột mà giảm
  dần. Dùng vì là mặc định của Transformer (BiomedCLIP dùng ViT), học ổn định.
- Công thức: $\text{GELU}(x) = x \cdot \Phi(x)$, với $\Phi$ là hàm phân phối chuẩn tích lũy. Ví dụ
  GELU(2) ≈ 1.95, GELU(−2) ≈ −0.05.

### Độ tương đồng cô-sin (lời văn)

$$\cos(a, b) = \frac{a^\top b}{\|a\|\,\|b\|}$$

- **Làm gì:** đo hai vector "cùng hướng" đến đâu: 1 là cùng hướng, 0 là vuông góc, −1 là ngược.
- Vì $\hat{f}_{img}$ và $t_c$ đều dài 1, nên $\hat{f}_{img}^\top t_c$ chính là cô-sin.

### (4.6) Chấm nhãn bằng cô-sin

$$\hat{y} = \arg\max_c \; s \cdot \hat{f}_{img}^\top t_c$$

- **Làm gì:** so vector ảnh với vector câu mô tả của từng mức (ví dụ "severe spinal canal
  stenosis"), chọn mức giống nhất.
- **Ký hiệu:** $t_c$ vector câu mô tả mức $c$ (BiomedCLIP mã hóa, dài 1); $s$ hệ số nhiệt độ học
  được.
- **Vì sao là đóng góp zero-shot:** công thức **không có tham số nào gắn riêng với từng mức**. Mọi
  thông tin về nhãn nằm trong $t_c$. Đổi bộ câu mô tả là chấm được bộ nhãn khác, không cần head
  mới.
- **$s$ có đổi dự đoán không:** không, nhân một số dương không đổi thứ tự. $s$ chỉ quan trọng khi
  huấn luyện (qua softmax ở 4.9).

### (4.7) Focal Loss

$$\text{FL} = -\omega_y (1 - p_y)^\gamma \log(p_y)$$

- **Làm gì:** như cross-entropy, nhưng **giảm mạnh phần đóng góp của mẫu dễ** để mô hình tập
  trung vào mẫu khó (thường là Severe).
- **Ký hiệu:** $y$ mức đúng; $p_y$ xác suất mô hình gán cho mức đúng (lấy từ 4.9); $-\log p_y$
  cross-entropy; $(1 - p_y)^\gamma$ hệ số focal, $\gamma = 2$; $\omega_y$ trọng số lớp (4.8).
- **Ví dụ (bỏ qua $\omega$):**

| | Mẫu dễ $p_y = 0.9$ | Mẫu khó $p_y = 0.2$ | Khó / dễ |
|---|---|---|---|
| Cross-entropy $-\log p_y$ | 0.105 | 1.609 | ≈ 15 lần |
| Hệ số $(1 - p_y)^2$ | 0.01 | 0.64 | |
| Focal Loss | 0.001 | 1.030 | **≈ 1000 lần** |

- **Ý nghĩa:** hàng nghìn ca Normal/Mild dễ cộng lại không còn lấn át vài chục ca Severe khó.
- **Vì sao $\gamma = 2$:** giá trị mặc định của bài gốc Focal Loss (Lin 2017), đã chứng minh ổn.
  $\gamma = 0$ là cross-entropy thường.

### (4.8) Trọng số lớp theo căn bậc hai

$$\omega_c = \frac{1/\sqrt{n_c}}{\frac{1}{3}\sum_{j=1}^{3} 1/\sqrt{n_j}}$$

- **Làm gì:** lớp càng ít mẫu thì trọng số càng lớn, nhưng chỉ tăng theo căn bậc hai để không
  quá chênh.
- **Ký hiệu:** $n_c$ số mẫu lớp $c$ trong tập train; mẫu số là trung bình ba giá trị $1/\sqrt{n}$,
  để **trung bình ba trọng số bằng 1** (độ lớn chung của loss không đổi).
- **Ví dụ với tỉ lệ 81% / 15% / 4%:**

| | Normal/Mild | Moderate | Severe | Severe / Normal |
|---|---|---|---|---|
| Căn bậc hai $1/\sqrt{n}$ | 0.111 | 0.258 | 0.500 | |
| → $\omega$ (chia trung bình 0.290) | 0.38 | 0.89 | 1.73 | **≈ 4.5 lần** |
| Nghịch đảo trực tiếp $1/n$ (so sánh) | 0.012 | 0.067 | 0.250 | ≈ 20 lần |

- **Vì sao không nghịch đảo trực tiếp:** chênh 20 lần làm loss nhảy mạnh mỗi khi gặp một ca
  Severe, huấn luyện mất ổn định. Căn bậc hai là mức giữa. (Luận văn ghi "khoảng bốn lần" và
  "khoảng mười lăm lần"; tỉ lệ thật tùy nhãn, cùng bậc.)

### (4.9) Xác suất từ đầu cô-sin (softmax)

$$p_{k,c} = \frac{\exp(s\,\hat{f}_{img}^\top t_{k,c})}{\sum_{j=1}^{3}\exp(s\,\hat{f}_{img}^\top t_{k,j})}$$

- **Làm gì:** biến 3 điểm cô-sin của nhãn $k$ thành 3 xác suất cộng lại bằng 1.
- **Ký hiệu:** $k$ nhãn bệnh (ống sống, lỗ liên hợp trái, phải); $c$ mức độ; $s$ nhiệt độ.
- **Vì sao cần $s$ (ví dụ):** cô-sin với 3 câu là 0.20 / 0.25 / 0.31.

| | Điểm | Xác suất Normal/Mild / Moderate / Severe |
|---|---|---|
| Không nhân $s$ | 0.20 / 0.25 / 0.31 | 0.32 / 0.33 / 0.35 (gần đều, "lưỡng lự") |
| $s \approx 14.3$ (khởi tạo) | 2.86 / 3.58 / 4.43 | 0.13 / 0.26 / 0.61 |
| $s = 50$ | 10 / 12.5 / 15.5 | 0.004 / 0.047 / 0.949 |

- **$s$ được chọn thế nào:** học được. Mô hình học $\log s$, khởi tạo $s = 1/0.07 \approx 14.3$
  như CLIP, chặn tối đa 100 để softmax không quá "gắt" (đạo hàm gần 0, mất ổn định).
- **Cách dùng hai đầu ra:** xác suất $p$ dùng tính AUC/AUPRC; argmax của $p$ (= 4.6) ra nhãn, dùng
  tính Recall/Precision/F1.

### SupCon (lời văn, trọng số 0.1)

$$\mathcal{L}_{SupCon} = \sum_i \frac{-1}{|P(i)|}\sum_{p \in P(i)} \log\frac{\exp(z_i^\top z_p/\tau)}{\sum_{a \ne i}\exp(z_i^\top z_a/\tau)}$$

- **Làm gì:** trong một lô, kéo vector ảnh của các ca **cùng mức độ** lại gần nhau, đẩy các ca
  khác mức ra xa. Giúp không gian vector "gọn" theo mức độ.
- **Ký hiệu:** $P(i)$ các mẫu cùng nhãn với $i$ trong lô; $\tau$ nhiệt độ.
- **Nói ngắn khi bị hỏi:** "Đây là số hạng phụ với trọng số nhỏ 0.1, giúp các ca cùng mức độ nằm
  gần nhau; phần chính vẫn là Focal Loss." Luận văn không có thí nghiệm bỏ riêng SupCon, nên đừng
  nói nó đóng góp bao nhiêu điểm.

### (4.10) Uncertainty weighting (Kendall 2018)

$$\mathcal{L}_{total} = \sum_{k=1}^{3}\left(\frac{1}{2\sigma_k^2}\mathcal{L}_k + \log\sigma_k\right)$$

- **Làm gì:** gộp loss của 3 nhãn bệnh, nhưng **tự học trọng số** cho từng nhãn thay vì cộng
  thẳng hay dò tay.
- **Ký hiệu:** $\mathcal{L}_k$ = Focal + 0.1 × SupCon của nhãn $k$; $\sigma_k$ tham số học được
  ("độ bất định" của nhãn $k$); $\frac{1}{2\sigma_k^2}$ trọng số thật; $\log\sigma_k$ số hạng phạt.
- **Vì sao cần $\log\sigma_k$:** không có nó, mô hình chỉ cần cho $\sigma \to \infty$ là loss
  về 0, tức bỏ học. $\log\sigma$ tăng theo $\sigma$ nên chặn việc đó.
- **Ví dụ (tối ưu tại $\sigma_k^2 = \mathcal{L}_k$):**

| Nhãn | $\mathcal{L}_k$ | $\sigma_k$ tối ưu | Trọng số $1/(2\sigma^2)$ |
|---|---|---|---|
| Ống sống (dễ hơn) | 0.5 | 0.71 | **1.0** |
| Lỗ liên hợp (khó hơn) | 2.0 | 1.41 | **0.25** |

  Nhãn khó (loss lớn) tự được giảm trọng số, nên không lấn át nhãn dễ.
- **Vì sao dùng:** lỗ liên hợp khó hơn ống sống trên mặt cắt dọc. Cộng thẳng thì đạo hàm bị hai
  nhãn lỗ liên hợp chi phối.

---

## 4. Chương 5: công thức chỉ số đánh giá

Ký hiệu chung: **TP** đoán có, thật có; **FP** đoán có, thật không (báo nhầm); **FN** đoán không,
thật có (bỏ sót); **TN** đoán không, thật không. Với 3 mức, tính **từng mức một** theo kiểu "mức
này so với phần còn lại" (one-vs-rest).

Ví dụ dùng chung (Hybrid seed 42, ống sống, lớp Severe, tập val 1.942 đĩa đệm): 81 ca Severe
thật, mô hình bắt được 57 (TP), sót 24 (FN), báo nhầm 90 (FP), TN ≈ 1.771.

### (5.1) Dice: độ chồng lắp (phân vùng)

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

- **Làm gì:** đo vùng dự đoán $P$ và vùng chuẩn $G$ chồng nhau bao nhiêu. 1 là trùng khít, 0 là
  không chạm.
- **Ví dụ:** $|P| = 100$ voxel, $|G| = 150$, chung 80 → $160 / 250 = 0.64$.
- **Hiểu nhanh:** Dice chính là **F1 ở mức voxel** ($2TP / (2TP + FP + FN)$).

### (5.2) IoU (Jaccard)

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

- **Ví dụ:** chung 80, hợp $100 + 150 - 80 = 170$ → $80 / 170 = 0.47$.
- **Quan hệ:** IoU luôn ≤ Dice; $\text{Dice} = 2\,\text{IoU} / (1 + \text{IoU})$ (0.47 → 0.64).

### (5.3) Precision và Recall (dùng cho cả voxel lẫn ca bệnh)

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

- **Precision:** trong những gì mô hình báo có, bao nhiêu đúng. Ví dụ: $57 / 147 = 0.388$.
- **Recall:** trong những ca thật sự có, mô hình bắt được bao nhiêu. Ví dụ: $57 / 81 = 0.704$.
- **Ở phân vùng:** TotalSpineSeg Precision 97.87% = gần như mọi voxel nó khoanh đều là đốt sống;
  Recall 56.49% = chỉ khoanh được một nửa vì không tính gai sau, mỏm ngang.

### (5.4) ASSD: khoảng cách bề mặt đối xứng trung bình

$$\text{ASSD} = \frac{1}{|\partial P| + |\partial G|}\left(\sum_{p \in \partial P} d(p, \partial G) + \sum_{g \in \partial G} d(g, \partial P)\right)$$

- **Làm gì:** đo đường viền dự đoán lệch đường viền chuẩn **bao nhiêu mm** trung bình. Càng nhỏ
  càng tốt.
- **Ký hiệu:** $\partial P$, $\partial G$ tập điểm trên viền; $d(p, \partial G)$ khoảng cách từ
  điểm $p$ tới điểm gần nhất trên viền kia.
- **Vì sao "đối xứng":** đo cả hai chiều (dự đoán → chuẩn và chuẩn → dự đoán), để không bỏ sót
  trường hợp một bên khoanh thiếu.
- **Khác Dice:** Dice đếm voxel, ASSD đo hình dạng viền. Hai mô hình cùng Dice có thể khác ASSD.

### (5.5) Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Làm gì:** tỉ lệ đoán đúng trên tổng số.
- **Vì sao không dùng làm chỉ số chính:** dữ liệu 81% Normal/Mild thì đoán "Normal" hết đã được
  81%. Đó là lý do SpineNetV2 có Accuracy 0.810 cao hơn Hybrid 0.724 mà vẫn bỏ sót 88% ca nặng.

### (5.6), (5.7) Precision, Recall cho thí nghiệm chọn mô hình

- Cùng công thức (5.3), nhưng TP/FP/FN đếm theo **bệnh nhân**, sau khi quy về có/không bệnh (xem
  mục 5.2 dưới).

### (5.8) F1

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Làm gì:** trung bình điều hòa của Precision và Recall. **Chỉ cao khi cả hai cùng cao.**
- **Ví dụ:** $P = 0.388$, $R = 0.704$ → $2 \times 0.273 / 1.092 = 0.500$.
- **Vì sao điều hòa chứ không cộng chia đôi:** $P = 1.0$, $R = 0.01$ thì trung bình cộng 0.505
  (trông ổn), F1 chỉ 0.02 (đúng là mô hình vô dụng).

### (5.9) F1 macro

$$\text{F1}_{macro} = \frac{1}{K}\sum_{k=1}^{K} \text{F1}_k$$

- **Làm gì:** tính F1 cho từng lớp rồi **trung bình đều** ($K$ = số lớp, ở RSNA là 3).
- **Vì sao macro:** lớp Severe 4% vẫn chiếm 1/3 điểm. Nếu lấy trung bình theo số mẫu (weighted)
  thì Severe gần như không ảnh hưởng.
- **Ví dụ:** ống sống F1 = 0.952 / 0.390 / 0.500 → macro = 0.614.

### (5.10) AUC

$$\text{AUC} = \int_0^1 \text{TPR}\big(\text{FPR}^{-1}(t)\big)\,dt$$

- **Làm gì:** diện tích dưới đường ROC (trục ngang FPR = tỉ lệ báo nhầm, trục dọc TPR = Recall),
  vẽ khi quét ngưỡng từ cao xuống thấp.
- **Cách hiểu dễ nhớ nhất:** **xác suất một ca bệnh được chấm điểm cao hơn một ca không bệnh**
  chọn ngẫu nhiên. 0.5 là đoán bừa, 1.0 là xếp hoàn hảo.
- **Ví dụ 5 ca** (sắp theo điểm giảm dần): Bệnh 0.9, Không 0.8, Bệnh 0.7, Không 0.4, Không 0.2.
  Có 2 × 3 = 6 cặp bệnh–không; ca bệnh cao hơn ở 5 cặp → AUC = 5/6 = 0.833.
- **Không cần ngưỡng:** chỉ xét thứ tự xếp hạng.
- **Code:** `sklearn.metrics.roc_auc_score`.

### (5.11) AUPRC

$$\text{AUPRC} = \int_0^1 \text{Precision}\big(\text{Recall}^{-1}(t)\big)\,dt$$

- **Làm gì:** diện tích dưới đường Precision–Recall. Code dùng **Average Precision**: đi xuống
  danh sách xếp hạng, mỗi khi gặp một ca bệnh thì ghi Precision tại đó, rồi lấy trung bình.
- **Ví dụ 5 ca trên:** gặp ca bệnh ở hạng 1 (Precision 1/1 = 1.0) và hạng 3 (2/3 = 0.67) → AP =
  (1.0 + 0.67) / 2 = 0.83.
- **Mốc đoán bừa = tỉ lệ ca dương:** Severe chiếm khoảng 4% thì đoán bừa AUPRC ≈ **0.04** (AUC
  đoán bừa luôn 0.5). Nên Severe AUPRC 0.333 là khoảng 8 lần đoán bừa.
- **Vì sao cần cả AUPRC:** AUC tính cả TN, mà lớp hiếm thì TN rất nhiều nên AUC dễ đẹp. AUPRC chỉ
  nhìn vào ca dương nên khắt khe hơn. Đó là lý do Severe AUC 0.899 nhưng Severe AUPRC chỉ 0.333.
- **Code:** `sklearn.metrics.average_precision_score`.

---

## 5. Cách tính các con số trong từng bảng đã chiếu

Đây là phần hay bị hỏi kiểu "con số 0.527 này tính thế nào". Mọi phép kiểm tra dưới đây đã
tính lại từ file kết quả.

### 5.1 Bảng chọn mô hình phân vùng (trang 13)

- Mỗi chỉ số (Dice, Precision, Recall, ASSD; IoU chỉ có trong luận văn) tính **riêng cho từng ca**
  trên mặt nạ nhị phân "đốt sống / không phải", rồi **trung bình 23 ca**.
- Nhãn chuẩn của bộ Lumbar Spine Segmentation là **một mặt nạ chung cho cả đốt sống** (thân + gai
  sau + mỏm ngang), nên mô hình chỉ khoanh thân đốt (TotalSpineSeg) bị Recall thấp.
- Độ lệch chuẩn ±5.23% (lời văn) là độ dao động giữa 23 ca.

### 5.2 Bảng chọn mô-đun đánh giá bất thường (trang 14)

1. **Quy về có/không bệnh:** mức Moderate hoặc Severe = có bệnh (1); Normal/Mild = không (0).
2. **Lên mức bệnh nhân:** một bệnh nhân "có bệnh" ở một nhãn nếu **ít nhất 1 trong 5 tầng** (L1/L2
   → L5/S1) có bệnh.
3. **Tính Accuracy/Precision/Recall/F1** theo công thức (5.5)–(5.8) trên **150 bệnh nhân**, riêng
   cho 3 nhãn.
4. **Số trên slide = trung bình cộng 3 nhãn** của từng chỉ số. Kiểm tra:

| | Tính | Kết quả |
|---|---|---|
| Ning Shen Accuracy | (86.63 + 77.01 + 78.61) / 3 | 80.75 → **80.8%** |
| Ning Shen F1 | (81.20 + 79.43 + 81.48) / 3 | **80.70%** |
| SpineNetV2 Recall | (80.00 + 84.00 + 77.27) / 3 | **80.42%** |
| SpineNetV2 F1 | (78.79 + 77.30 + 76.40) / 3 | **77.50%** |
| MedGemma Recall | (20.87 + 8.43 + 6.02) / 3 | **11.77%** |
| MedGemma F1 | (31.17 + 13.86 + 10.75) / 3 | **18.59%** |

- **Lưu ý khi bị hỏi:** F1 trên slide là **trung bình của 3 F1**, không phải F1 tính lại từ
  Precision và Recall trung bình (hai cách cho số hơi khác).
- **Vì sao F1 ở đây 77–81% mà ở bảng RSNA chỉ 0.42–0.53:** bài toán dễ hơn nhiều (2 mức thay vì
  3; mức bệnh nhân thay vì từng đĩa đệm; gộp Moderate với Severe).

### 5.3 Bảng kết quả RSNA 2024 (trang 21): trung bình 3 lần chạy

Mỗi lần chạy (seed 42, 123, 456) được đánh giá trên tập val của nó (395 bệnh nhân, 1.942 đĩa đệm
với seed 42). Với **mỗi lần chạy**:

**Bước 1: từng nhãn bệnh, từng mức độ.** Có 3 nhãn × 3 mức = 9 ô. Mỗi ô tính Recall, Precision,
F1 (one-vs-rest), AUC, AUPRC (dùng xác suất softmax của mức đó).

**Bước 2: gộp thành con số của lần chạy đó.**

| Dòng trong bảng | Công thức cho một lần chạy |
|---|---|
| **Mean F1 (macro)** | $\frac{1}{3}\sum_{k}\Big(\frac{1}{3}\sum_{c} \text{F1}_{k,c}\Big)$: macro theo 3 mức, rồi trung bình 3 nhãn |
| **Mean Recall / Precision** | như trên, thay F1 bằng Recall / Precision |
| **Mean AUC / AUPRC** | như trên, với AUC / AUPRC từng mức |
| **Severe Recall** | $\frac{1}{3}\sum_{k} \text{Recall}_{k,\text{Severe}}$: chỉ lấy ô Severe, trung bình 3 nhãn |
| **Severe F1 / AUC / AUPRC** | tương tự, chỉ ô Severe |
| **Mean Accuracy** (lời văn) | $\frac{1}{3}\sum_{k} \text{Accuracy}_k$, accuracy 3 mức của từng nhãn |

Kiểm tra với Hybrid seed 42:

| | Ống sống | LLH trái | LLH phải | Trung bình |
|---|---|---|---|---|
| F1 ba mức | 0.952 / 0.390 / 0.500 | 0.727 / 0.408 / 0.278 | 0.797 / 0.446 / 0.252 | |
| Macro F1 từng nhãn | 0.614 | 0.471 | 0.498 | **Mean F1 = 0.528** |
| Severe Recall | 0.704 | 0.375 | 0.313 | **0.464** |
| Severe F1 | 0.500 | 0.278 | 0.252 | **0.343** |
| Severe AUC | 0.962 | 0.869 | 0.858 | **0.896** |
| Accuracy | 0.885 | 0.604 | 0.674 | **0.721** |

**Bước 3: trung bình ± độ lệch chuẩn qua 3 lần chạy.**

$$\bar{x} = \frac{x_1 + x_2 + x_3}{3}, \qquad s = \sqrt{\frac{\sum_i (x_i - \bar{x})^2}{3 - 1}}$$

- Ví dụ Severe F1 của Hybrid: 0.343 / 0.348 / 0.375 → trung bình **0.356**; độ lệch chuẩn
  $\sqrt{(0.0123^2 + 0.0073^2 + 0.0197^2)/2} \approx$ **0.017**.
- Chia cho $n - 1 = 2$ (độ lệch chuẩn mẫu) vì chỉ có 3 lần chạy, ước lượng từ mẫu nhỏ.
- ⚠️ **Riêng dòng Severe Recall** trong bảng chia cho $n = 3$: Hybrid 46.4 / 49.7 / 49.7 → ±0.016
  (chia 3) thay vì ±0.019 (chia 2); SpineNetV2 ±0.005 thay vì ±0.006. Trung bình không đổi. Nếu
  bị hỏi: "Dòng đó em tính độ lệch chuẩn theo tổng thể, các dòng khác theo mẫu; tính lại theo mẫu
  là ±0.019, kết luận không đổi."

### 5.4 Bảng chi tiết theo lớp (phụ lục trang 51)

- **Một lần chạy đại diện (seed 42)**, không phải trung bình 3 lần.
- Mỗi ô = chỉ số của mức đó, **trung bình 3 nhãn bệnh**. Ví dụ Severe Recall Hybrid 0.464 =
  (0.704 + 0.375 + 0.313) / 3.
- Vì vậy số ở đây (0.464, 0.343) khác số 3 lần chạy ở trang 21 (0.486, 0.356).

### 5.5 Bảng zero-shot SPIDER (trang 23)

- Mô hình Hybrid huấn luyện trên RSNA, **không huấn luyện thêm**. Chỉ thay câu mô tả.
- **Mỗi nhãn:** dự đoán = argmax của cô-sin với các câu mô tả của nhãn đó (không qua softmax; kết
  quả như nhau vì softmax giữ thứ tự).
  - **F1** = F1 macro theo các mức của nhãn đó (Pfirrmann 5 mức, Modic 4 mức, các nhãn có/không
    2 mức).
  - **AUC:** nhãn có/không dùng điểm = cô-sin("có") − cô-sin("không"); nhãn nhiều mức tính
    one-vs-rest rồi trung bình.
- **Dòng trung bình** = trung bình cộng 8 nhãn: F1 0.362, AUC 0.733.
- **Chỗ hay bị hỏi:** trượt đốt sống F1 0.028 nhưng AUC 0.807: xếp hạng đúng, chỉ là argmax (ngưỡng
  mặc định) gần như không bao giờ chọn "có" với nhãn chỉ 2.8% ca dương. Modic AUC 0.454 < 0.5: kém
  hơn đoán bừa, vì Modic cần chuỗi T1 mà mô hình không có.

### 5.6 Bảng SPIDER có tinh chỉnh (trang 24)

Thiết lập: mỗi cấu hình lấy checkpoint tốt nhất trên RSNA, giữ cố định phần thân, học lại phần
đầu ra trên 8 nhãn SPIDER (1.174 đĩa đệm, chia 80/20, 15 epoch, 3 seed).

| Dòng | Công thức cho một lần chạy |
|---|---|
| **Mean Accuracy (%)** | trung bình cộng accuracy của 8 nhãn |
| **Mean F1 (macro)** | trung bình cộng 8 nhãn của (F1 macro theo các mức của nhãn đó) |
| **Mean AUC / AUPRC** | trung bình cộng 8 nhãn |
| **Inference (ms/mẫu)** | tổng thời gian chạy tập val / số đĩa đệm |

Rồi trung bình ± độ lệch chuẩn qua 3 seed như 5.3.

### 5.7 Bảng SPIDER theo từng nhãn (phụ lục trang 52, luận văn Bảng 5.x)

- Mỗi ô Recall / Precision / F1 là **macro theo các mức của nhãn đó**, trung bình 3 seed.
- **Vì sao Recall 0.500 = bỏ sót mọi ca:** với nhãn có/không,
  $$\text{Recall}_{macro} = \frac{\text{Recall}_{\text{không}} + \text{Recall}_{\text{có}}}{2}$$
  Đoán "không" cho mọi ca → Recall lớp "không" = 1.0, lớp "có" = 0.0 → (1.0 + 0.0) / 2 = **0.500**.
  Đó là trường hợp của SpineNetV2 và CBAM ở thoát vị đĩa đệm, cả 3 seed.
- ⚠️ **Bảng chỉ có 7 dòng, thiếu Pfirrmann**, nhưng Mean F1 ở trang 24 vẫn tính đủ 8 nhãn. Số
  Pfirrmann (3 seed): SpineNetV2 F1 0.592, CBAM 0.540, BMC-only 0.439, **Hybrid 0.544**. Ở nhãn
  này SpineNetV2 tốt hơn Hybrid. Nếu bị hỏi, trả lời thật: "Pfirrmann có trong Mean F1; ở nhãn
  này SpineNetV2 cao hơn, Hybrid đứng thứ hai."

### 5.8 Bảng phụ lục số tham số và chi phí (trang 41, 42)

- **Số tham số** = đếm số phần tử của mọi ma trận trọng số. Hybrid chỉ học ~1.18M = $W_1$, $W_2$,
  bias, $w_a$, $s$, 3 số $\sigma_k$.
- **Thời gian huấn luyện** = thời gian thật trên RTX 4090. Hybrid "+24.8 phút" là thời gian **thêm**
  của bước 2, chưa tính bước 1 huấn luyện CBAM.
- **Suy luận ms/mẫu** = thời gian chạy hết tập val / số mẫu (~62 ms trên RSNA, ~65–67 ms trên
  SPIDER do khác máy/lần đo).

---

## 6. Grad-CAM (slide trang 22)

$$\alpha_k = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}, \qquad L^c_{\text{Grad-CAM}} = \text{ReLU}\Big(\sum_k \alpha_k A^k\Big)$$

- **Làm gì:** vẽ bản đồ nhiệt "mô hình dựa vào vùng nào để chấm lớp $c$".
- **Từng bước:**
  1. $A^k$: bản đồ đặc trưng thứ $k$ ở một tầng tích chập (slide dùng tầng 3).
  2. $\partial y^c / \partial A^k_{ij}$: nếu tăng giá trị ở ô $(i, j)$ thì điểm lớp Severe tăng
     bao nhiêu. Lấy trung bình mọi ô ($Z$ = số ô) → $\alpha_k$ = độ quan trọng của bản đồ $k$.
  3. Cộng các bản đồ theo trọng số $\alpha_k$, qua ReLU để chỉ giữ vùng **làm tăng** điểm Severe,
     phóng to về kích thước ảnh rồi tô màu.
- **Giới hạn phải nói:** một ca chỉ là **minh họa**, không phải bằng chứng. Bằng chứng là bảng số.

---

## 7. Bảng tra một dòng

| Công thức | Một câu để nói |
|---|---|
| Perceptron (2.1, 2.2) | Nhân đầu vào với trọng số, cộng lại, qua hàm kích hoạt rồi so ngưỡng |
| ReLU | Giữ số dương, đổi số âm thành 0 |
| RNN (2.3–2.7) | Đọc chuỗi từng bước, trạng thái mới trộn trạng thái cũ với phần tử mới |
| Attention (2.8) | Trung bình có trọng số các trạng thái, trọng số cho biết cái nào quan trọng |
| Softmax | Biến điểm số thành xác suất, giữ thứ tự |
| WCE Ning Shen (3.1) | Cross-entropy, lỗi lớp hiếm phạt nặng theo tỉ lệ 1:2:4 |
| CBAM kênh (4.1) | Mỗi kênh một trọng số: chọn loại đặc trưng |
| CBAM không gian (4.2) | Mỗi vị trí 3D một trọng số: chọn vùng cần nhìn |
| CBAM áp dụng (4.3) | Nhân hai mặt nạ vào đặc trưng, kênh trước không gian sau |
| Slice pool (4.4) | Gộp 3 lát BiomedCLIP bằng trọng số softmax học được |
| Fusion (4.5) | Ghép hai vector, MLP trộn, chuẩn hóa độ dài về 1 |
| Cô-sin head (4.6) | Chọn mức có câu mô tả giống ảnh nhất |
| Focal (4.7) | Cross-entropy nhưng giảm mạnh mẫu dễ |
| Trọng số căn (4.8) | Lớp ít mẫu trọng số lớn hơn, theo căn bậc hai cho đỡ chênh |
| Softmax cô-sin (4.9) | Biến 3 cô-sin thành 3 xác suất, $s$ làm rõ khác biệt |
| Uncertainty (4.10) | Tự học trọng số cho 3 nhãn, nhãn khó tự giảm trọng số |
| Dice (5.1) | Độ chồng lắp, chính là F1 ở mức voxel |
| IoU (5.2) | Giao chia hợp, chặt hơn Dice |
| Precision (5.3) | Báo có thì đúng bao nhiêu |
| Recall (5.3) | Ca thật có thì bắt được bao nhiêu |
| ASSD (5.4) | Viền lệch trung bình bao nhiêu mm |
| Accuracy (5.5) | Đoán đúng bao nhiêu phần, bị lớp đa số làm đẹp |
| F1 (5.8) | Trung bình điều hòa P và R, chỉ cao khi cả hai cao |
| F1 macro (5.9) | Trung bình đều các lớp, lớp hiếm vẫn chiếm phần ngang |
| AUC (5.10) | Xác suất ca bệnh được xếp trên ca không bệnh |
| AUPRC (5.11) | Như AUC nhưng chỉ nhìn ca dương, đoán bừa bằng tỉ lệ ca dương |
| Grad-CAM | Đạo hàm cho biết vùng nào làm tăng điểm Severe |
