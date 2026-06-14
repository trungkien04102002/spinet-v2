# Q&A chuẩn bị bảo vệ — câu hỏi khó từ hội đồng

> File này để cầm theo lúc bảo vệ. Mỗi câu có: **(1) câu trả lời ngắn nói ngay**, **(2) số liệu dẫn chứng** (lấy từ log thật, đã kiểm), **(3) lật slide phụ lục nào**.
> Số dùng nhất quán với slide (single-seed RSNA, 1×RTX 4090).

---

## 1. "Làm sao em biết mô hình đã hội tụ?"

**Trả lời ngắn:** Em không lấy epoch cuối, mà **theo dõi validation và chọn checkpoint tốt nhất (early stopping)**. Dấu hiệu hội tụ là **val_loss chạm đáy rồi đi ngang/đi lên (không cải thiện thêm vài epoch liên tiếp), trong khi val severe-F1 bão hòa**. Khi đó dừng và lấy checkpoint có val tốt nhất.

**Dẫn chứng (val_loss chạm đáy / best epoch):**

| Cấu hình | val_loss đáy | Best epoch (severe-F1) | Nhận xét hội tụ |
|---|---|---|---|
| Baseline | ep25 (vẫn đang giảm chậm) | ep16–18 | hội tụ chậm, đến cuối ngân sách vẫn nhúc nhích |
| CBAM | ep20 (0.158) | ep14–23 | hội tụ giữa chừng |
| **Hybrid** | **ep9 (0.139)** | **sớm (~ep10)** | **hội tụ sớm nhất, val_loss thấp nhất** |

→ Em huấn luyện với ngân sách cố định (25–30 epoch) nhưng **báo cáo theo best-checkpoint**, không phải epoch cuối → tránh overfitting.

---

## 2. "Tốc độ hội tụ của phương pháp em so với baseline?" ⭐

**Trả lời ngắn:** **Hybrid hội tụ NHANH HƠN baseline**, và hội tụ tới điểm tốt hơn. Lý do: nhánh BiomedCLIP **frozen** đã cung cấp không gian đặc trưng y khoa mạnh sẵn, nên chỉ còn khối MLP nhỏ phải học → đường val_loss xuống nhanh và phẳng sớm.

**Dẫn chứng:**
- Hybrid: val_loss đạt đáy **ep9**, severe-F1 tốt nhất quanh **ep10**.
- Baseline: severe-F1 tốt nhất tận **ep16–18**, val_loss vẫn còn giảm tới ep25 (chưa phẳng hẳn).
- Về wall-clock: Hybrid **24.8 phút** so với Baseline 16.4 phút, **CBAM 56.0 phút** → Hybrid train nhanh hơn CBAM nhiều dù mạnh hơn.

→ *Lật slide "Phụ lục: thời gian huấn luyện & suy luận".*

---

## 3. "Computation cost cao hơn baseline nhiều không? Tăng cost nhiều mà chỉ hơn vài % thì không đáng." ⭐⭐ (câu bẫy quan trọng)

**Trả lời ngắn — 3 ý:**

1. **Chi phí train là chi phí MỘT LẦN, offline:**
   - Hybrid train **2 giai đoạn**: bước 1 train CBAM (56 phút) → checkpoint; bước 2 **train thêm 24.8 phút trên nền checkpoint CBAM (frozen) + BiomedCLIP (frozen)** → tổng ~80.8 phút. *(Đừng nói "Hybrid 24.8 phút rẻ hơn CBAM 56 phút" — không fair vì Hybrid dùng lại CBAM.)*
   - Bước 2 chỉ huấn luyện **khối hợp nhất ~0.5M** (fusion MLP + cosine head + slice attention pool + logit scale); **CBAM-ResNet 21.3M và BiomedCLIP 86.2M đều frozen**.

2. **Inference KHÔNG đổi:** ~62 ms/mẫu cho cả ba cấu hình (thêm nhánh BiomedCLIP gần như không tăng độ trễ) → vẫn thời gian thực (~16 mẫu/giây).

3. **Cải thiện ở chỉ số lâm sàng quan trọng nhất** (không phải "vài %"):
   - Severe **Recall 11.5% → 46.4%** (~4×); Severe **F1 0.149 → 0.343** (~2.3×); Mean F1 0.420 → 0.528.
   - *(Lưu ý: accuracy tổng GIẢM 0.814→0.721 — có chủ đích, vì baseline cao do thiên về lớp đa số.)*

> **Câu chốt (khung trade-off):** *"Em chấp nhận tốn thêm thời gian huấn luyện — một lần, offline — để đổi lấy: thời gian suy luận giữ nguyên ~62 ms, cải thiện mạnh lớp Severe, và thêm khả năng mở nhãn zero-shot. Chi phí tăng nằm hết ở khâu train một lần; lúc triển khai thực tế (suy luận) thì không tốn thêm gì."*

---

## 4. "Report không thấy phân tích cost và convergence?"

**Trả lời ngắn:** Báo cáo **có bảng computational cost** (thời gian huấn luyện, throughput, độ trễ suy luận — Bảng Training/Inference time). Phần hội tụ thể hiện gián tiếp qua **cơ chế chọn best-checkpoint theo validation** và cột "best epoch". Em **có đầy đủ log per-epoch (train_loss/val_loss/severe-F1)** nên nếu hội đồng cần, em **bổ sung được đường cong hội tụ ngay**. *(Đã chuẩn bị số ở mục 1–2 trên.)*

> Nếu bị truy: thành thật nhận đây là điểm có thể trình bày rõ hơn trong báo cáo, và nêu kế hoạch thêm hình loss-curve + bảng so sánh tốc độ hội tụ vào bản ĐATN.

---

## 5. "Định nghĩa robustness (tính ổn định) ở đây là gì?"

**Trả lời ngắn:** Trong phạm vi đề tài, robustness được hiểu theo 2 khía cạnh cụ thể (không chung chung):

1. **Ổn định trước mất cân bằng lớp:** mô hình **không sụp đổ trên lớp hiếm**. Baseline có severe-recall ~11% (gần như bỏ lớp Severe); Hybrid giữ severe-recall 46% → ổn định hơn nhiều ở lớp khó.
2. **Tổng quát hóa liên tập dữ liệu (cross-dataset):** chuyển từ RSNA sang **SPIDER**, kiến trúc Hybrid **giữ hiệu năng ổn định**, trong khi CBAM đơn nhánh suy giảm. Đây là bằng chứng robustness về domain.
3. *(nếu hỏi sâu)* Trong bài báo còn báo cáo **độ lệch chuẩn qua nhiều seed** — biến động nhỏ → ổn định về khởi tạo.

---

## 6. "Novelty của phương pháp là gì?" ⭐⭐

**Trả lời ngắn — KHÔNG nói "em dùng CBAM + BiomedCLIP" (đó là công cụ có sẵn). Nói theo THIẾT KẾ:**

Novelty nằm ở **thiết kế kiến trúc Hybrid hai nhánh giải quyết đồng thời hai bài toán** mà từng nhánh đơn lẻ không làm được:

1. **Kết hợp nhánh chú ý thể tích (đặc trưng 3D cục bộ vùng tổn thương nhỏ) với nhánh nền đa phương thức frozen (tri thức y khoa tổng quát học sẵn)** — nhằm trực tiếp vào **lớp Severe quá ít dữ liệu**: nhánh nền bù đặc trưng tổng quát, nhánh chú ý khu trú tổn thương.
2. **Thay đầu phân loại softmax cố định bằng đầu đối sánh cosine với prompt văn bản** → **mở rộng nhãn bệnh mới chỉ bằng đổi prompt, không huấn luyện lại (zero-shot)**. Softmax cố định không làm được điều này.

→ Điểm mới = **(a) cách phối hợp hai nguồn đặc trưng để trị mất cân bằng + (b) cơ chế mở nhãn zero-shot**, áp dụng cho **đánh giá thoái hóa cột sống thắt lưng**. Theo hiểu biết của em, đây là lần đầu áp dụng zero-shot theo prompt văn bản (BiomedCLIP) cho bài toán grading thoái hóa cột sống thắt lưng.

> Câu chốt: *"Cái mới không phải là dùng module nào, mà là thiết kế phối hợp chúng để vừa cứu lớp bệnh nặng vừa mở rộng được nhãn mà không cần train lại."*

---

## 7. "CBAM có được huấn luyện không? Hybrid frozen hết thì train cái gì?" ⭐ (câu dễ hỏi)

**Trả lời ngắn:** CBAM **có** được huấn luyện — nhưng ở **một bước riêng trước đó**. Quy trình **2 giai đoạn**:

1. **Bước 1 — train CBAM-only** (`train_rsna_attention.py`): huấn luyện đầy đủ CBAM-3D ResNet-34 (21.3M) trên RSNA, khởi tạo từ backbone ResNet pretrained. Đây **chính là cấu hình "CBAM" trong bảng ablation**. Kết quả lưu thành checkpoint **`best_model_attention.pth`**.
2. **Bước 2 — train Hybrid** (`train_rsna_hybrid.py`): **nạp `best_model_attention.pth`** rồi **frozen**; nạp BiomedCLIP (cũng frozen); **chỉ huấn luyện khối hợp nhất** (fusion MLP + cosine head, ~0.5M).

**`best_model_attention.pth` = model CBAM-only đã train ở bước 1.**

> **Câu chốt:** *"CBAM có được huấn luyện — ở bước 1 em train riêng model CBAM trên RSNA (chính là cấu hình CBAM trong ablation). Ở bước 2, Hybrid tái sử dụng trọng số CBAM đó, frozen lại, chỉ học khối hợp nhất nhỏ để kết hợp với đặc trưng BiomedCLIP. Nhờ tái dùng CBAM đã học sẵn nên Hybrid hội tụ rất nhanh."*

→ Đây cũng là lý do Hybrid train nhanh hơn CBAM-only: CBAM-only train cả backbone từ đầu (~ep20), còn Hybrid chỉ học thêm phần hợp nhất trên nền CBAM có sẵn (~ep9).

---

## 8. "Sao thời gian huấn luyện ngắn vậy? (24.8 phút)" ⭐

**Trả lời ngắn:** Vì bước 2 **không huấn luyện lại mạng lớn nào**, chỉ học một khối nhỏ. 4 lý do:

1. **Chỉ train ~0.5M tham số** (fusion MLP + cosine head), không phải hàng chục triệu — backward pass rất nhẹ.
2. **Tái dùng backbone CBAM đã pretrained** (nạp checkpoint, frozen) → không học lại từ đầu.
3. **BiomedCLIP frozen** → chỉ chạy forward một lần để lấy đặc trưng, không cập nhật gradient.
4. **Đặc trưng nền mạnh sẵn** (CBAM đã học trên RSNA + BiomedCLIP học trên 15M cặp ảnh–văn bản) nên phần hợp nhất **hội tụ sớm** (≈ epoch 9).

> **Câu chốt:** *"24.8 phút là vì bước 2 chỉ huấn luyện khối hợp nhất nhỏ (~0.5M tham số) trên nền hai backbone đã frozen — không train lại CBAM hay BiomedCLIP. Đặc trưng nền đã mạnh sẵn nên hội tụ rất nhanh. Nếu tính cả bước 1 train CBAM thì tổng là ~80.8 phút."*

*(Không phải do ít data — RSNA có ~1\,800 BN train; mà do số tham số học rất ít + tái dùng backbone pretrained.)*

---

## 9. Giải thích chi tiết kiến trúc — từng phần làm gì + vì sao model khác không làm được ⭐⭐

*(Dùng khi thầy nói "em mô tả kỹ kiến trúc đi" hoặc "model em khác gì các model trước". Lật kèm slide phụ lục "Kiến trúc Hybrid đầy đủ" / "Vị trí CBAM trên ResNet-34".)*

### 9.1. Luồng dữ liệu (đầu vào → đầu ra)

`Khối ảnh per-IVD (9×112×224)` → **hai nhánh song song** → **hợp nhất** → **đối sánh cosine với prompt** → nhãn.

**Vào gì → ra gì ở mỗi khối:**
1. **Vào:** khối MRI 1 đĩa đệm `9 lát × 112 × 224`.
2. **CBAM-3D (frozen):** trích đặc trưng cục bộ 3D → **`f_cbam` 512-d**.
3. **BiomedCLIP image encoder (frozen):** mã hoá các lát + slice-pool → **`f_bmc` 512-d**.
4. **Fusion MLP (học):** `concat=1024 → 768 → 512`, L2 → **`f_img` 512-d** (embedding ảnh).
5. **Text encoder BiomedCLIP (frozen, chạy 1 lần/cache):** mỗi prompt nhãn (vd *"severe spinal canal stenosis"*) → **`t_k` 512-d** = prototype của lớp đó.
6. **Cosine head:** `logit_k = scale·(f_img·t_kᵀ)` → softmax → **xác suất + nhãn** (Normal/Mild · Moderate · Severe).

> **Nói 1 câu:** *"Ảnh MRI đĩa đệm vào hai nhánh song song — CBAM ra một vector 512-d mô tả tổn thương cục bộ, BiomedCLIP ra một vector 512-d mang tri thức y khoa nền; Fusion MLP ghép lại thành embedding ảnh 512-d, rồi so cosine với các vector prompt văn bản (mỗi nhãn một câu, mã hoá qua text encoder), nhãn nào giống nhất là kết quả."*

| Thành phần | Làm gì | Giúp điều gì |
|---|---|---|
| **Nhánh 1: CBAM-3D ResNet-34** | Trích đặc trưng 3D cục bộ; CBAM dồn trọng số vào đúng kênh + vùng không gian tổn thương → vector `f_cbam ∈ R^512` | **Khu trú tổn thương nhỏ** của lớp Severe (vùng bệnh rất nhỏ trong khối ảnh) |
| **Nhánh 2: BiomedCLIP ViT-B/16 (frozen)** | Mã hoá ảnh (trên các lát cắt) bằng đặc trưng học sẵn từ **15M cặp ảnh–văn bản y khoa**; slice attention pool → `f_bmc ∈ R^512` | (a) **Bù dữ liệu Severe quá ít** bằng prior tổng quát; (b) đặt ảnh vào **chung không gian với văn bản** |
| **Fusion head** | `concat(f_cbam, f_bmc)=1024` → MLP `1024→768→512`, chuẩn hoá L2 → `f_img` | Kết hợp **đặc trưng cục bộ 3D** + **prior y khoa tổng quát** |
| **Cosine head + prompt** | Tính cosine giữa `f_img` và embedding của các **prompt văn bản** `{q_k}` (mã hoá bằng text encoder BiomedCLIP frozen); chọn `argmax` | **Phân loại theo mô tả văn bản** thay vì lớp cố định → đổi prompt là thêm nhãn mới |

### 9.2. Vì sao của em làm được mà model khác không?

**So với CNN phân loại thông thường (Baseline / CBAM-only):**
- Chúng dùng **đầu softmax cố định** trên một tập lớp định sẵn → **muốn thêm nhãn bệnh mới phải đổi kiến trúc đầu ra và huấn luyện lại**. Của em dùng **đầu cosine với prompt** → thêm nhãn = thêm một câu mô tả văn bản, **không cần train lại (zero-shot)**. *(Đây là điểm khác cốt lõi.)*
- Chỉ học trên RSNA nên **không có tri thức nền** → trên lớp Severe hiếm (~4%) thì sụp (recall ~11%). Của em **mượn đặc trưng BiomedCLIP học sẵn** → cải thiện mạnh lớp hiếm (recall 46%).

**So với chỉ dùng foundation model 2D (BMC-only):**
- BiomedCLIP là **2D**, thiếu **chú ý thể tích 3D cục bộ** → định vị tổn thương nhỏ kém. Của em **ghép thêm nhánh CBAM-3D** nên có cả hai: định vị 3D cục bộ + prior tổng quát.

**Tóm lại — 3 điều cộng hưởng mà không model đơn lẻ nào có đủ:**
1. **CBAM-3D** → khu trú tổn thương nhỏ (cứu lớp Severe).
2. **BiomedCLIP frozen** → bù dữ liệu hiếm + đưa ảnh vào không gian ảnh–văn bản.
3. **Đầu cosine–prompt** (thay softmax cố định) → **mở nhãn mới bằng văn bản, không train lại**.

> **Câu chốt:** *"Cái khác biệt không nằm ở việc dùng module nào, mà ở chỗ em thiết kế đầu ra theo kiểu đối sánh ảnh–văn bản thay vì softmax cố định: nhờ vậy mô hình vừa tận dụng được tri thức nền của foundation model để cứu lớp bệnh hiếm, vừa thêm được nhãn bệnh mới chỉ bằng cách đổi prompt — điều mà một bộ phân loại softmax thông thường không làm được."*

---

## 10. "Bảng 2.2 SpineNetV2 Recall 80% — sao Severe Recall chỉ 11.5%?" ⭐ (rất dễ hỏi)

**Vì hai con số đo HAI bài toán khác nhau, khác độ chi tiết:**
- **Bảng 2.2 (Recall ~80%)** = bài **phát hiện bất thường NHỊ PHÂN** ("có bất thường hay không"), trung bình 3 nhóm bệnh → câu hỏi thô, **dễ**.
- **Severe Recall 11.5%** = bài **grading 3 mức** (Normal/Mild, Moderate, Severe), đo riêng **lớp Severe hiếm nhất** (~4%) → câu hỏi tinh, **khó**.

> **Trả lời:** *"80% là recall của bài phát hiện bất thường nhị phân — chỉ cần biết 'có vấn đề hay không'. 11.5% là recall riêng lớp Severe trong bài phân mức độ 3 lớp — phải nói đúng 'nặng tới mức nào'. Phát hiện có bất thường thì dễ, phân đúng là Severe (lớp cực hiếm) thì khó hơn nhiều. Chính khoảng cách này — detect tốt nhưng grade lớp nặng kém — là động lực để em tinh chỉnh mô-đun grading."*

*(Khớp report chap05: detection-level F1 = 77.5% trong khi Severe Recall = 11.5%.)*

**Follow-up dễ bị hỏi tiếp: "Sao MedGemma lại dưới 50% (Acc 38.7%) — binary thì random đã 50% rồi mà?"**
- MedGemma **không phải bộ phân loại nhị phân huấn luyện sẵn** — nó là **VLM đa dụng chạy zero-shot** (hỏi "có bất thường không?"). Nó **thiên lệch hệ thống về 'bình thường'**: Recall chỉ 6–21% (bỏ sót 80–94% ca bệnh), Precision ~39–62%.
- Mốc 50% chỉ đúng cho **bộ đoán ngẫu nhiên cân bằng (tung đồng xu)**. MedGemma là mô hình **lệch một phía** → accuracy tụt theo tỉ lệ lớp nó hay chọn → **dưới 50% là bình thường**, không mâu thuẫn gì.

> **Trả lời:** *"MedGemma không đoán 50/50 như tung đồng xu — nó là VLM zero-shot trả lời 'bình thường' quá thận trọng nên Recall 6–21%, bỏ sót phần lớn ca bệnh. Mô hình lệch một phía thì accuracy xuống dưới mức ngẫu nhiên là chuyện thường; mốc 50% chỉ áp cho bộ đoán cân bằng. Chính vì lệch và bỏ sót nhiều nên em loại MedGemma — VLM chưa chuyên biệt hóa không đáng tin cho sàng lọc."*

## 11. "F1 tính thế nào — bình thường hay có trọng số?" ⭐

- **F1 thường** (harmonic mean của Precision & Recall), **KHÔNG** trọng số theo tần suất lớp.
- Gộp nhiều lớp/điều kiện → dùng **macro** (trung bình cộng không trọng số), mỗi lớp cân ngang nhau.
- Bảng 2.2 = F1 nhị phân lớp "bất thường", tính từng nhóm rồi trung bình 3 nhóm; F1 grading (0.343...) = macro F1 trên 3 lớp severity.

> **Trả lời:** *"F1 bình thường — trung bình điều hòa của Precision và Recall. Em dùng macro, trung bình không trọng số qua các lớp; cố ý KHÔNG dùng weighted vì weighted sẽ để lớp đa số Normal/Mild (~85%) lấn át, còn macro ép mô hình làm tốt cả lớp hiếm Severe."*

*(Chi tiết macro vs weighted + ví dụ: xem `DEFENSE_METRICS_VI.md` Phần A.)*

---

## 12. "Hybrid khác SpineNetV2 chỗ nào — cách xử lý và đầu ra (head)?" ⭐⭐

| | SpineNetV2 (baseline) | Hybrid (đề xuất) |
|---|---|---|
| Đặc trưng | 3D ResNet-34 → 1 vector | CBAM-ResNet (f_cbam) **+** BiomedCLIP (f_bmc) → concat → fusion MLP → f_img (L2) |
| **Đầu ra (head)** | **Linear + softmax** trên tập lớp cố định; trọng số lớp học cứng trong ma trận W | **Cosine** giữa f_img và vector **prompt văn bản**; "prototype lớp" = embedding của prompt |
| Thêm nhãn mới | Phải đổi head + train lại | Chỉ thêm 1 prompt (zero-shot) |
| Tri thức nền | Không | Có (BiomedCLIP học sẵn 15M cặp) |

**Ở mức công thức (nếu bị vặn sâu):**
- Baseline: `logits = W·GAP(x) + b` → softmax. Mỗi lớp = **một cột của ma trận W học cứng**; số lớp **niêm phong** trong kích thước W. Thêm lớp = đổi W + train lại.
- Hybrid: `logits = s · (f_img · t_kᵀ)` với `f_img` (ảnh, L2) và `t_k` = embedding **prompt văn bản** (BiomedCLIP frozen, cache 1 lần). Mỗi lớp = **một vector prompt**; thêm lớp = mã hoá thêm 1 prompt, **không đụng trọng số**.
- Một câu: head của họ là **bộ phân loại đóng** (lớp khắc trong W); head của em là **đối sánh ảnh–văn bản mở** (lớp = prompt) → đó là lý do em zero-shot được còn họ thì không.

> **Trả lời:** *"Khác cốt lõi ở đầu ra: SpineNetV2 dùng đầu tuyến tính softmax trên tập lớp cố định — trọng số lớp học cứng trong ma trận W. Hybrid thay bằng đầu cosine đối sánh ảnh với prompt văn bản, nên 'prototype' của mỗi lớp chính là embedding của câu mô tả, đổi prompt là đổi/thêm lớp mà không cần học lại. Ngoài ra Hybrid có thêm một nguồn đặc trưng nền từ BiomedCLIP mà baseline không có."*

## 13. "Hybrid phải có prompt → vector từ text encoder; lúc nào cũng phụ thuộc text encoder? Source of truth từ đó?" ⭐⭐

- **Đúng:** mỗi nhãn cần 1 prompt → text encoder BiomedCLIP mã hóa thành 1 **vector prototype**; phân loại = cosine giữa embedding ảnh và các prototype này.
- **Nhưng với nhãn đã học (RSNA): prompt được mã hóa MỘT LẦN, cache sẵn** — không chạy text encoder mỗi lần suy luận (code `grading_hybrid.py`: *"text embeddings encoded once via frozen BiomedCLIP and stored"*, *"text encoder called once, reused every batch"*). Lúc suy luận chỉ dùng vector đã lưu → **nhanh như một bộ phân loại cố định**.
- "Source of truth" của *ranh giới lớp* đúng là đến từ text encoder — đây là **chủ đích thiết kế**: biến tên lớp thành vector trong không gian ảnh–văn bản, nhờ vậy thêm nhãn mới = mã hóa thêm 1 prompt (1 lần). Với nhãn đã học, mô hình được train để **căn embedding ảnh khớp với các prototype đó**.
- **Điểm cần biết (thành thật):** có **prompt sensitivity** — chất lượng tùy cách viết prompt và việc BiomedCLIP hiểu thuật ngữ y khoa đó. Nhãn RSNA đã được calibrate qua huấn luyện nên ổn; nhãn zero-shot xa ngữ nghĩa (Modic, Pfirrmann) thì yếu hơn.

> **Trả lời:** *"Phụ thuộc text encoder chỉ ở khâu tạo prototype lớp, và làm một lần rồi cache lại; lúc suy luận thực tế nó chạy như một bộ phân loại cố định, không gọi lại text encoder. Cái lợi của việc dùng text làm prototype là thêm nhãn mới chỉ cần thêm prompt, không phải huấn luyện lại."*

## 14. "Điểm yếu model mình + điểm mạnh/thiếu sót model khác — sao mình tốt hơn?" ⭐⭐

**Điểm mạnh Hybrid:** (1) Severe recall cao — cứu lớp hiếm; (2) mở nhãn **zero-shot**; (3) **ổn định cross-dataset** (BiomedCLIP frozen đóng vai regularizer, không overfit như CBAM-only).

**Điểm yếu Hybrid (nói thẳng, đừng giấu):**
- Accuracy tổng thấp hơn baseline (đánh đổi lớp đa số — **có chủ đích**).
- AUPRC trên SPIDER hơi thấp hơn SpineNetV2 (trade-off calibration của xác suất).
- Phụ thuộc chất lượng prompt / text encoder cho nhãn mới (prompt sensitivity).
- Zero-shot còn **khiêm tốn** (Mean F1 0.362).
- Huấn luyện **2 giai đoạn** (cần checkpoint CBAM trước); mang theo 86M tham số frozen → tốn bộ nhớ hơn.

**Thiếu sót của các model khác:**
- **SpineNetV2:** head softmax cố định → không mở nhãn; sụp lớp Severe (recall 11.5%); không có tri thức nền.
- **CBAM-only:** cải thiện Severe trên RSNA nhưng **suy giảm khi sang SPIDER** (overfit phân bố RSNA).
- **BMC-only:** thiếu chú ý 3D cục bộ → định vị tổn thương nhỏ kém.
- **MedGemma:** VLM zero-shot lệch về "bình thường", recall rất thấp, không định vị được.
- **Ning Shen:** **pipeline đa tầng, chuyên biệt từng bệnh** — Faster R-CNN (ResNet-50-FPN) định vị từng tầng đĩa đệm + Swin Transformer phân loại độ nặng, **mỗi bệnh một mô hình riêng (5 bộ)**, lại cần **3 chuỗi MRI** (Sagittal T1 + T2/STIR + Axial T2). → mạnh trên RSNA nhưng **phức tạp để huấn luyện/triển khai/bảo trì**, **bám chặt giao thức dữ liệu RSNA** (khó tổng quát sang bộ khác như SPIDER), và **head đóng 5 bệnh → không mở nhãn mới**; thêm bệnh = thêm detector + classifier + train lại. Chưa qua bình duyệt.

> **Trả lời:** *"Cái của em tốt hơn không phải vì một module mạnh, mà vì kết hợp được hai điểm mạnh bổ trợ — chú ý 3D cục bộ và tri thức nền tổng quát — cộng với đầu cosine–prompt cho thêm khả năng zero-shot mà không model đơn lẻ nào có. Đổi lại em chấp nhận một số điểm yếu như accuracy tổng giảm và zero-shot còn khiêm tốn, nhưng đó là đánh đổi hợp lý cho mục tiêu lâm sàng là không bỏ sót ca nặng và mở rộng được nhãn."*

## 15. "Khối Fusion MLP để làm gì — sao không cộng trung bình (logit-average) hai nhánh là xong?" ⭐⭐

**Trả lời ngắn:** Khối học được **duy nhất** trong Hybrid chính là **Fusion MLP** (concat $1024 \to 768 \to 512$, chuẩn hóa L2) + đầu cosine + slice-pool + logit-scale ($\sim$0.5M); hai backbone CBAM (21.3M) và BiomedCLIP (86.2M) đều **frozen**. (Đây cũng là "khối MLP nhỏ" nói ở câu hỏi #2 về tốc độ hội tụ.) Nó **cần** vì 2 lý do:
1. **Căn (align) hai không gian đặc trưng khác nhau:** đặc trưng CBAM (512-d, không gian ảnh 3D) và BiomedCLIP (512-d, không gian ảnh--văn bản) nằm ở **hai không gian khác nhau**; MLP học cách chiếu chúng vào **một embedding 512-d chung**, khớp với embedding của prompt văn bản để tính cosine.
2. **Fusion học được > cộng trung bình:** thử trực tiếp — lấy trung bình softmax của SpineNetV2 + BMC-only chỉ đạt Mean F1 **0.45** / Severe F1 **0.18**, kém xa Hybrid học-fusion **0.53 / 0.36**. → trộn KHÔNG quy về phép cộng tầm thường; MLP học được phần tương tác giữa hai nhánh.

> **Câu chốt:** *"Fusion MLP là phần duy nhất em huấn luyện. Nó không chỉ nối hai vector lại mà học cách căn hai không gian đặc trưng khác nhau vào chung một embedding để đối sánh với prompt văn bản. Bằng chứng nó cần thiết: cộng trung bình logit hai nhánh chỉ đạt 0.45/0.18, còn fusion học được đạt 0.53/0.36."*

## 16. "Contrastive learning (học tương phản) là gì?" ⭐

**Trả lời ngắn:** Là cách huấn luyện để **kéo cặp khớp lại gần, đẩy cặp không khớp ra xa** trong một không gian embedding. BiomedCLIP học trên **15M cặp ảnh y khoa--chú thích**: với mỗi ảnh, embedding của ảnh được kéo gần embedding của đúng câu mô tả của nó, và đẩy xa các câu của ảnh khác (và ngược lại). Kết quả là **ảnh và văn bản cùng nằm trong MỘT không gian**, nên độ giống giữa chúng đo được bằng **cosine**.

**Vì sao quan trọng với đề tài:** chính nhờ ảnh--văn bản chung không gian mà em **phân loại bằng cách so ảnh với prompt** (cosine), và **thêm nhãn mới chỉ bằng đổi prompt** (zero-shot) — không cần đầu phân loại cố định.

> **Câu chốt:** *"Contrastive learning dạy mô hình đặt ảnh và mô tả văn bản của nó cạnh nhau trong cùng một không gian; nhờ vậy em chỉ cần so cosine giữa embedding ảnh và embedding của câu mô tả nhãn để dự đoán, và thêm bệnh mới chỉ là thêm một câu mô tả."*

## 17. "Từ cosine ra nhãn thế nào — softmax/argmax là gì, và khác kiến trúc gốc (SpineNetV2) ở đâu?" ⭐⭐

**Chuỗi ra nhãn của Hybrid (em):**
1. Với mỗi nhãn `k` (mỗi prompt): `logit_k = scale · cosine(f_img, t_k)` → ví dụ 3 nhãn ra **điểm số** `[0.31, 0.28, 0.45]`.
2. **Softmax** biến dãy điểm thô thành **xác suất** (mỗi giá trị $\in[0,1]$, tổng $=1$): $\mathrm{softmax}(z)_k = e^{z_k}/\sum_j e^{z_j}$ → `[0.30, 0.28, 0.42]`. Điểm cao → xác suất cao.
3. **Argmax** = chọn **vị trí có xác suất lớn nhất** → **nhãn dự đoán** (ở đây = Severe, độ tin cậy 0.42).

**Khác kiến trúc gốc SpineNetV2 ở chỗ nào:** phần **softmax → argmax giống hệt nhau**; chỉ khác **cách tính `logit_k`**:
- **SpineNetV2 (gốc):** `logit_k = w_k \cdot \text{feature}` — mỗi lớp có **một hàng trọng số `w_k` học cứng** trong ma trận Linear head. Muốn thêm lớp = thêm hàng `w_k` mới + **train lại**.
- **Hybrid (em):** `logit_k = scale \cdot \cos(f_{img}, t_k)` — "hàng trọng số của lớp `k`" được **thay bằng embedding văn bản `t_k`** của nhãn đó. Muốn thêm lớp = thêm **một prompt** → có ngay `t_k` mới, **không train lại** (zero-shot).

> **Câu chốt:** *"Về bản chất, head softmax thông thường tính điểm mỗi lớp bằng một vector trọng số học cứng. Em chỉ thay vector trọng số đó bằng embedding văn bản của tên lớp — phần softmax/argmax phía sau y hệt. Nhờ vậy 'trọng số lớp' giờ đến từ câu mô tả, thêm bệnh mới chỉ là thêm một câu, không phải huấn luyện lại."*

## 18. "Sao chỉ train 25--30 epoch — ít vậy có đủ hội tụ không?" ⭐

**Trả lời ngắn — hợp lý, vì 4 lý do:**
1. **Không train from scratch:** tất cả khởi tạo từ **pretrained** (3D ResNet-34 backbone pretrained + BiomedCLIP pretrained) → đây là **fine-tuning**, cần ít epoch hơn nhiều so với train từ đầu (vốn cần hàng trăm).
2. **Hybrid chỉ train ~0.5M** (fusion head, hai backbone frozen) → hội tụ rất nhanh, **val_loss đáy ep9**.
3. **Báo cáo best-checkpoint (early stopping)**, không lấy epoch cuối. Best epoch đều **nằm trong ngân sách**: Hybrid ~ep9--10, CBAM ep14--23, Baseline ep16--18 → 25--30 epoch phủ dư.
4. val_loss đã **chạm đáy/đi ngang** trong ngân sách (Hybrid ep9, CBAM ep20) → train thêm không cải thiện, dễ overfit.

**Follow-up dễ bị hỏi: "Baseline val_loss còn giảm tới ep25 — có train thiếu baseline (so sánh thiếu công bằng) không?"**
→ Dùng **CÙNG ngân sách cho cả 4 cấu hình** để công bằng; best severe-F1 của baseline đã đạt ở **ep16--18** (trong ngân sách). Kể cả train lâu hơn, baseline chỉ cải thiện **lớp đa số** chứ không cứu được lớp **Severe hiếm** — vì đó là bài toán **mất cân bằng**, giải bằng Focal/attention/BiomedCLIP, **không phải bằng thêm epoch** → kết luận không đổi.

> **Câu chốt:** *"Em fine-tune từ trọng số pretrained chứ không train từ đầu, lại báo cáo theo best-checkpoint trên validation chứ không phải epoch cuối; val_loss đã chạm đáy trong ngân sách (Hybrid ep9) nên 25--30 epoch là đủ, train thêm chỉ overfit. Em dùng cùng ngân sách cho cả bốn cấu hình để so sánh công bằng."*

---

## Bảng số liệu nhanh (để khỏi lật nhiều)

| Chỉ số | Baseline | Hybrid (đề xuất) |
|---|---|---|
| Mean F1 (RSNA) | 0.420 | **0.528** |
| Severe Recall | 11.5% | **46.4%** (~4×) |
| Severe F1 | 0.149 | **0.343** (~2.3×) |
| Accuracy tổng | 0.814 | 0.721 (giảm có chủ đích) |
| Thời gian train | 16.4 phút | 24.8 phút |
| Suy luận | ~62 ms/mẫu | ~62 ms/mẫu |
| Tham số học / frozen | 21.3M / 0 | ~0.5M / ~107.5M |
| Hội tụ (val_loss đáy) | ep25 | **ep9** |
| Zero-shot SPIDER (Mean F1) | — | 0.362 (không train) |

*Nguồn: log huấn luyện RSNA (`experiments/`), single-seed, 1×RTX 4090. Khớp với slide và báo cáo single-seed.*
