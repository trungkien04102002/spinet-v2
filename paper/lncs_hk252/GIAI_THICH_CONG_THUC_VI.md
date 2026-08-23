# Giải thích toàn bộ công thức — phần Proposed Method (paper MIWAI)

Tài liệu này giải thích từng công thức trong Mục 3 (Proposed Method) của `main.tex`,
theo đúng thứ tự xuất hiện. Mỗi công thức gồm: (a) viết lại, (b) nghĩa từng ký hiệu,
(c) trực giác / vì sao thiết kế vậy, (d) lưu ý có thể bị reviewer hỏi.

Ký hiệu toán giữ nguyên tiếng Anh để khớp paper.

---

## 0. Bức tranh tổng & các ký hiệu nền

| Ký hiệu | Ý nghĩa |
|---|---|
| $V$ | một khối ảnh (volume) của 1 đĩa đệm (IVD), kích thước $1\times 9\times 112\times 224$ (1 kênh, 9 lát sagittal, 112×224 pixel) |
| $\mathcal{V}$ | không gian tất cả các volume đó, $\mathcal{V}=\mathbb{R}^{1\times 9\times 112\times 224}$ |
| $\Phi$ | **encoder** — mạng biến volume thành 1 vector đặc trưng (embedding) đã chuẩn hoá |
| $\hat f$ hay $\hat f_{\text{img}}$ | embedding ảnh sau chuẩn hoá L2, nằm trên mặt cầu đơn vị $\mathbb{S}^{d-1}$, $d=512$ |
| $c$ | **head phân loại** — biến embedding thành xác suất các lớp |
| $\mathcal{Y}$ | tập nhãn (vd RSNA: {Normal/Mild, Moderate, Severe}) |
| $\mathbb{S}^{d-1}$ | mặt cầu đơn vị trong không gian $d$ chiều (mọi vector có độ dài = 1) |
| $\Delta^{m}$ | simplex $m$ chiều = tập các vector xác suất (các phần tử ≥ 0, tổng = 1) |

**Ý tưởng cốt lõi cần nhớ:** model = "encoder dùng chung" + "head thay đổi tuỳ task".
Encoder $\Phi$ học một lần; muốn đổi sang bộ nhãn khác chỉ cần đổi head $c$ và tập nhãn $\mathcal{Y}$.
Đây chính là nền tảng cho khả năng zero-shot mở rộng nhãn.

---

## 1. Công thức (eq:composition) — mô hình tổng quát

$$h = c \circ \Phi, \qquad \Phi : \mathcal{V} \to \mathbb{S}^{d-1}, \quad c : \mathbb{S}^{d-1} \to \Delta^{|\mathcal{Y}|-1}$$

- $h = c \circ \Phi$ nghĩa là "model $h$ = chạy $\Phi$ trước, rồi đưa kết quả qua $c$": $h(V) = c(\Phi(V))$.
- $\Phi : \mathcal{V} \to \mathbb{S}^{d-1}$: nhận volume, trả về embedding **độ dài 1** (đã L2-normalize), 512 chiều.
- $c : \mathbb{S}^{d-1} \to \Delta^{|\mathcal{Y}|-1}$: nhận embedding, trả về phân bố xác suất trên $|\mathcal{Y}|$ lớp. (Nếu 3 lớp thì $\Delta^{2}$ — chú ý số mũ là $|\mathcal{Y}|-1$ vì 3 xác suất tổng bằng 1 nên chỉ có 2 bậc tự do.)

**Trực giác:** tách bạch "hiểu ảnh" ($\Phi$) khỏi "ra quyết định nhãn" ($c$). Vì $\Phi$ cố định, đổi task = đổi $c$.

**Có thể bị hỏi:** *Vì sao ép embedding lên mặt cầu (L2-normalize)?* → Để dùng **cosine similarity** với text embedding của BiomedCLIP (BiomedCLIP cũng chuẩn hoá). Trên mặt cầu, tích vô hướng $\hat f^\top t_k$ chính là cosine — đo độ "giống nghĩa" giữa ảnh và mô tả văn bản.

---

## 2. Công thức (eq:erm) — huấn luyện có giám sát trên RSNA

$$\min_{\Phi,\,\{c_t\}} \;\; \sum_{t \in \mathcal{T}} \mathbb{E}_{(V, y_t) \sim \mathcal{D}_{\text{RSNA}}} \big[\, \ell_t\big(c_t(\Phi(V)),\, y_t\big)\,\big]$$

- $\mathcal{T} = \{\text{canal},\,\text{L-foram},\,\text{R-foram}\}$: 3 bệnh lý (hẹp ống sống, hẹp lỗ liên hợp trái/phải). Mỗi bệnh có **head riêng** $c_t$.
- $\mathbb{E}_{(V,y_t)\sim \mathcal{D}_{\text{RSNA}}}[\cdot]$: kỳ vọng (trung bình) trên dữ liệu RSNA — tức trung bình loss trên toàn bộ mẫu huấn luyện.
- $\ell_t$: hàm mất mát cho task $t$ = **focal loss có trọng số lớp** (xem công thức 8).
- $\min_{\Phi,\{c_t\}}$: tối ưu **đồng thời** encoder $\Phi$ và cả 3 head $c_t$.

**Trực giác:** đây là khung ERM (Empirical Risk Minimization) chuẩn — "tìm tham số sao cho tổng loss trên dữ liệu nhỏ nhất", nhưng (1) cộng loss trên 3 bệnh (multi-task), (2) loss là focal có trọng số để chống mất cân bằng lớp.

**Có thể bị hỏi:** *Vì sao 3 head riêng thay vì 1 head 9 lớp?* → Mỗi bệnh là 1 trục phân độ độc lập (cùng 3 mức Normal/Mild–Moderate–Severe); tách head cho phép mỗi bệnh có ranh giới quyết định riêng và xử lý nhãn thiếu (-1) độc lập.

---

## 3. Zero-shot head (cosine, không tham số) — mở rộng nhãn

$$t_k = \frac{\psi(q_k)}{\|\psi(q_k)\|_2}, \qquad c^{\text{zs}}_k(\hat f) = \mathrm{softmax}_k\!\big(s\,\hat f^{\top} t_k\big)$$

- $\psi$: **text encoder của BiomedCLIP** (đóng băng). Nhận một câu mô tả nhãn $q_k$ (prompt, vd *"an MRI of severe disc herniation"*) → trả về vector văn bản.
- $t_k$: vector văn bản của nhãn $k$ sau khi chuẩn hoá L2 (gọi là **text anchor**).
- $\hat f^\top t_k$: tích vô hướng = **cosine similarity** giữa embedding ảnh và anchor văn bản (vì cả hai đều độ dài 1).
- $s$: **temperature / logit scale** — số nhân kế thừa từ BiomedCLIP, điều chỉnh độ "sắc" của softmax.
- $\mathrm{softmax}_k$: biến các điểm tương đồng thành xác suất trên các nhãn $k$.

**"Không tham số" (parameter-free):** head này **không có trọng số học được** — chỉ so embedding ảnh với embedding văn bản. Muốn thêm nhãn mới chỉ cần **viết thêm prompt**, không cần train lại.

**Trực giác — điểm bán hàng của paper:** vì $\Phi$ cho ra embedding **cùng không gian** với text encoder, ta phân loại bằng cách hỏi "ảnh này gần mô tả nào nhất?". Đổi từ bộ nhãn RSNA sang SPIDER = chỉ đổi tập prompt $\{q_k\}$ → **bộ nhãn trở thành "đầu vào" của inference**, không phải hằng số kiến trúc.

**Có thể bị hỏi:** *Khác gì CLIP zero-shot gốc?* → Về cơ chế thì **đúng là tính chất zero-shot chuẩn của CLIP** (paper thừa nhận, không claim mới). Cái mới là **áp dụng cross-dataset** (train RSNA → suy luận nhãn SPIDER) và nhánh CBAM + fusion được giám sát giúp embedding mang thông tin tốt hơn naked BiomedCLIP.

---

## 4. Công thức (eq:cbam) — khối chú ý CBAM (nhánh 1, có học)

$$F'' = M_s\!\big(M_c(F) \otimes F\big) \otimes \big(M_c(F) \otimes F\big)$$

- $F \in \mathbb{R}^{C\times D\times H\times W}$: feature map 3D sau một tầng ResNet (C kênh, D×H×W không gian).
- $M_c$: **channel attention** — học "kênh nào quan trọng". Là MLP 2 lớp (bottleneck tỉ lệ $r=16$) chạy trên thống kê average-pool và max-pool theo không gian.
- $M_s$: **spatial attention** — học "vùng không gian nào quan trọng". Là 1 conv 3D $7\times7\times7$ trên bản đồ gộp theo kênh.
- $\otimes$: nhân **element-wise** (theo từng phần tử), có broadcast theo chiều bị thiếu.
- Cả $M_c, M_s$ đi qua **sigmoid** → giá trị trong $[0,1]$ đóng vai trò "cổng" (gate) khuếch đại/giảm.

**Đọc công thức từ trong ra ngoài:**
1. $M_c(F)\otimes F$: lấy $F$ rồi nhân với trọng số kênh → "lọc theo kênh", gọi tạm là $F'$.
2. $M_s(F')$: tính trọng số không gian từ $F'$.
3. $M_s(F')\otimes F'$: nhân tiếp trọng số không gian → $F''$.

Tức **lọc 2 lần liên tiếp: theo kênh rồi theo không gian.**

**Trực giác:** giúp mạng "tập trung" vào vùng/kênh chứa tổn thương (vd vùng ống sống bị hẹp) thay vì dàn đều. Hình Grad-CAM trong paper minh hoạ điều này.

**Có thể bị hỏi:** *CBAM gốc là 2D, đây làm sao?* → Mở rộng sang 3D: pooling và conv đều thành 3D, hoạt động trên cả chiều sâu D (số lát). Sau khối cuối + global average pooling → $f_{\text{cbam}} \in \mathbb{R}^{512}$.

---

## 5. Công thức (eq:slicepool) — gộp 9 lát bằng attention (nhánh 2, BiomedCLIP)

$$\alpha_i = \frac{\exp(w_a^\top z_i)}{\sum_{j=1}^{9} \exp(w_a^\top z_j)}, \qquad f_{\text{bmc}} = \sum_{i=1}^{9} \alpha_i \, z_i$$

- $z_i = g_\theta(s_i)$: embedding của lát thứ $i$, lấy từ **BiomedCLIP ViT-B/16 đóng băng** $g_\theta$ (mỗi lát sagittal được nhân 3 kênh, resize 224×224, đưa qua encoder ảnh).
- $w_a \in \mathbb{R}^{512}$: **vector trọng số học được** — tham số học **duy nhất** của nhánh này.
- $\alpha_i$: trọng số chú ý của lát $i$, tính bằng **softmax** trên 9 điểm $w_a^\top z_i$ → các $\alpha_i \ge 0$, tổng = 1.
- $f_{\text{bmc}} = \sum_i \alpha_i z_i$: embedding nhánh BiomedCLIP = **trung bình có trọng số** của 9 lát.

**Trực giác:** thay vì lấy trung bình cộng đều 9 lát (mean pooling), mạng **học cách up-weight lát nào nhìn rõ tổn thương nhất**. $\alpha_i$ lớn = lát đó quan trọng.

**Có thể bị hỏi:** *Vì sao xử lý từng lát 2D rồi mới gộp (2.5D), không phải 3D thật?* → BiomedCLIP là model 2D (train trên ảnh-caption), nên phải đưa từng lát qua rồi gộp. Bối cảnh 3D thật do **nhánh CBAM** đảm nhiệm. Đây là một hạn chế đã ghi trong paper (chưa có VLM 3D y tế public lúc làm).

---

## 6. Công thức (eq:fusion) — hợp nhất 2 nhánh (MLP fusion, có học)

$$f_{\text{img}} = W_2\,\phi\!\big(W_1\,[f_{\text{cbam}};f_{\text{bmc}}] + b_1\big) + b_2$$

- $[f_{\text{cbam}};f_{\text{bmc}}]$: **nối (concat)** 2 embedding 512 chiều → vector 1024 chiều.
- $W_1 \in \mathbb{R}^{768\times 1024},\, b_1$: lớp tuyến tính thứ nhất, 1024 → 768.
- $\phi$: hàm kích hoạt **ReLU** (kèm dropout 0.1).
- $W_2 \in \mathbb{R}^{512\times 768},\, b_2$: lớp tuyến tính thứ hai, 768 → 512.
- Sau đó **L2-normalize**: $\hat f_{\text{img}} = f_{\text{img}} / \|f_{\text{img}}\|_2$ → đưa về mặt cầu (để dùng cosine ở công thức 7).

**Trực giác:** MLP 2 lớp học cách **trộn** thông tin hình học 3D (CBAM) với thông tin ngữ nghĩa (BiomedCLIP) thành 1 embedding 512 chiều "tốt hơn cả hai". Đây là khối "fusion" mà thí nghiệm trivial-ensemble (C2) sẽ chứng minh là có giá trị thật (so với chỉ cộng trung bình output 2 nhánh).

**Có thể bị hỏi:** *Số tham số học được?* → ~1.18M (CBAM fine-tune + $w_a$ + MLP fusion), so với ~218M tổng (gồm 2 backbone đóng băng). Tức **>99% tham số bị đóng băng**, chỉ học phần rất nhỏ → hợp với dữ liệu nhỏ.

---

## 7. Công thức (eq:cosine) — dự đoán bằng cosine (cho zero-shot)

$$\hat{y} = \arg\max_{k}\; s \cdot \hat{f}_{\text{img}}^{\top}\, t_k$$

- Giống công thức 3 nhưng viết ở dạng **dự đoán nhãn** (lấy argmax thay vì softmax).
- Chọn nhãn $k$ có **cosine similarity cao nhất** giữa embedding ảnh $\hat f_{\text{img}}$ và text anchor $t_k$.
- $s$: temperature, không đổi argmax (chỉ ảnh hưởng độ tự tin softmax) nhưng giữ cho nhất quán.

**Trực giác:** "ảnh này giống mô tả nhãn nào nhất thì gán nhãn đó". Đổi $\{t_k\}$ (đổi prompt) = đổi bộ nhãn, **không train lại** — đúng tinh thần mở rộng nhãn.

---

## 8. Công thức (eq:focal) — focal loss có trọng số lớp

$$\mathrm{FL}_t(p_t) = -\,w_{c(t)}\,(1 - p_t)^{\gamma}\, \log(p_t), \qquad \gamma = 2.0$$

- $p_t$: xác suất model gán cho **lớp đúng** (ground-truth). $p_t$ gần 1 = đoán đúng và tự tin.
- $-\log(p_t)$: cross-entropy chuẩn (phạt khi $p_t$ nhỏ).
- $(1-p_t)^{\gamma}$: **hệ số focal** — khi mẫu **dễ** ($p_t$ lớn), $(1-p_t)^\gamma$ ≈ 0 → **giảm trọng số** mẫu dễ, dồn sức học mẫu khó. $\gamma=2$ là độ "tập trung".
- $w_{c(t)}$: **trọng số lớp** của lớp đúng — lớp hiếm (Severe) được nhân trọng số lớn hơn (xem dưới).
- Tổng loss = cộng $\mathrm{FL}_t$ trên 3 bệnh; nhãn thiếu (giá trị $-1$) bị **bỏ qua** qua `ignore_index`.

**Trọng số lớp:** $w_c \propto 1/\sqrt{n_c}$ (chuẩn hoá để trung bình = 1), với $n_c$ = số mẫu lớp $c$.

- Vì sao **căn bậc hai** $1/\sqrt{n_c}$ thay vì $1/n_c$? → $1/n_c$ làm tỉ lệ gradient Severe/Normal lên ~15× (quá gắt, mất ổn định). $1/\sqrt{n_c}$ kéo xuống ~4× — vẫn ưu tiên lớp hiếm nhưng huấn luyện ổn định hơn.

**Trực giác (cốt lõi của cả paper):** 3 vũ khí chống mất cân bằng:
1. **Focal** $(1-p_t)^\gamma$: bỏ qua mẫu dễ (đa số là Normal/Mild).
2. **Class weight** $1/\sqrt{n_c}$: tăng gradient cho lớp hiếm.
3. (Ở mức sampler) **Oversampling** minority ×3–5: tăng tần suất Severe vào batch.

Ba cái bổ trợ nhau: class weight chỉnh **độ lớn gradient** khi mẫu đã vào batch; oversampling chỉnh **xác suất mẫu vào batch**.

**Có thể bị hỏi:** *Focal và class weight có trùng vai trò?* → Không. Focal phân biệt dễ/khó **theo từng mẫu** (dựa $p_t$); class weight phân biệt **theo lớp** (dựa $n_c$). Một cái theo độ khó, một cái theo tần suất lớp.

---

## 9. Sơ đồ liên kết các công thức

```
        volume V
          │
   ┌──────┴───────┐
   │              │
 CBAM 3D        9 lát → BiomedCLIP(frozen) g_θ
 (eq 4)              │
   │            slice attention pool (eq 5)
 f_cbam(512)        f_bmc(512)
   └──────┬───────┘
       concat 1024
     MLP fusion (eq 6)
          │
     L2-normalize  →  f̂_img  (đây chính là Φ(V), eq 1)
          │
   ┌──────┴────────┐
 head tuyến tính   cosine vs text anchors t_k = ψ(q_k)  (eq 3,7)
 (supervised RSNA) (zero-shot SPIDER)
   │                  │
 focal loss (eq 8)   argmax → nhãn
 → tối ưu (eq 2)
```

---

## 10. Một câu tóm tắt cho mỗi công thức (học thuộc nhanh)

1. **eq:composition** — Model = encoder dùng chung + head đổi theo task.
2. **eq:erm** — Train trên RSNA: tối thiểu tổng focal loss của 3 bệnh.
3. **zero-shot head** — Phân loại bằng cosine với embedding văn bản của prompt; thêm nhãn = thêm prompt.
4. **eq:cbam** — Lọc feature 2 lần: theo kênh rồi theo không gian, để tập trung vào tổn thương.
5. **eq:slicepool** — Gộp 9 lát bằng trung bình có trọng số học được (up-weight lát rõ bệnh).
6. **eq:fusion** — MLP trộn embedding hình học (CBAM) + ngữ nghĩa (BiomedCLIP) → 1 embedding 512.
7. **eq:cosine** — Dự đoán = nhãn có cosine cao nhất với embedding ảnh.
8. **eq:focal** — Loss giảm trọng số mẫu dễ + tăng trọng số lớp hiếm (căn bậc hai) để chống mất cân bằng.
