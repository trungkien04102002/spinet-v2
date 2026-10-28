# Đọc file này trước: lộ trình ôn bảo vệ trong một tối

Bảo vệ **Thứ Hai 02/11/2026**. Trình bày tối đa 20 phút, hỏi đáp 10–15 phút.

---

## 1. Thứ tự đọc

Các file được đánh số theo thứ tự đọc. Đọc **in đậm** là bắt buộc, phần còn lại chỉ tra khi cần.

| Thứ tự | File | Đọc gì | Thời gian |
|---|---|---|---|
| 0 | `0_DOC_TRUOC.md` (file này) | **Toàn bộ**: ý nghĩa chỉ số, con số phải thuộc, chỗ dễ bị bắt lỗi | 20 phút |
| 1 | `1_SCRIPT_BAO_VE_LVTN.md` | **Lời nói từng slide** (khung `>`). Bỏ qua các khối 📘 lần đầu | 30 phút |
| 2 | `2_HOI_DAP_BAO_VE.md` | **phần trả lời** các câu hỏi, theo thứ tự slide. Lướt phần "Vì sao" | 40 phút |
| 3 | `3_CONG_THUC_LUAN_VAN.md` | **Mục 3** (công thức mô hình, chương 4) và **mục 5** (cách tính các bảng). Mục 1–2 chỉ lướt | 40 phút |
| 4 | `4_GIAI_THICH_METRIC_VA_CONG_THUC.md` | Tra cứu khi muốn hiểu sâu một chỉ số (ma trận nhầm lẫn, đường ROC/PR) | khi cần |
| 5 | `5_CHUAN_BI_BAO_VE_LVTN.md` | Ghi chú cũ, đã gộp vào file 2. Chỉ xem mục softmax/argmax | khi cần |

Tổng phần bắt buộc: khoảng **2 tiếng 10 phút**.

**Mẹo đọc:** đọc xong mỗi file, gập lại và tự nói to 3 ý chính. Nói được là thuộc.

---

## 2. Ý nghĩa của từng chỉ số: hỏi gì cũng nói được

Ví dụ xuyên suốt: **lớp Severe** (ca nặng), chiếm khoảng **4%** số đĩa đệm.

### 2.1 Bốn chỉ số cơ bản (dựa trên đếm đúng/sai)

| Chỉ số | Câu hỏi nó trả lời | Đọc số của Hybrid thế nào | Cao hay thấp là tốt |
|---|---|---|---|
| **Accuracy** | Đoán đúng bao nhiêu phần trên tất cả? | Mean Accuracy 0.724: đúng khoảng 72 trên 100 đĩa đệm | Cao, **nhưng dễ bị lừa**: đoán "bình thường" hết đã được 81% |
| **Recall** (độ nhạy) | Trong các ca **thật sự bệnh**, bắt được bao nhiêu? | Severe Recall 0.486: trong 100 ca nặng thật, bắt được khoảng **49** (mô hình nền chỉ 12) | Cao. **Chỉ số quan trọng nhất** vì bỏ sót ca nặng nguy hiểm nhất |
| **Precision** | Trong các ca **mô hình báo bệnh**, đúng bao nhiêu? | Severe Precision khoảng 0.27: trong 100 lần báo "nặng", khoảng 27 lần đúng | Cao. Thấp nghĩa là báo nhầm nhiều, bác sĩ phải xem lại nhiều |
| **F1** | Cân bằng Recall và Precision | Severe F1 0.356 | Cao. **Chỉ cao khi cả hai cùng cao**: Recall 1.0 mà Precision 0.01 thì F1 gần 0 |

**Recall và Precision kéo nhau:** báo "nặng" nhiều hơn thì bắt được nhiều hơn (Recall tăng) nhưng
báo nhầm cũng nhiều hơn (Precision giảm). Luận văn **ưu tiên Recall** vì báo nhầm thì bác sĩ loại
được, còn bỏ sót thì không ai biết.

### 2.2 Gộp nhiều lớp, nhiều nhãn: "Mean", "macro", "Severe"

| Tên trong bảng | Nghĩa | Vì sao dùng |
|---|---|---|
| **F1 của một lớp** | F1 tính riêng cho một mức (ví dụ chỉ Severe), coi lớp đó là "dương", hai lớp kia là "âm" | Để thấy rõ mô hình làm tốt ở mức nào |
| **Macro F1** | Trung bình **đều** F1 của 3 mức (Normal/Mild, Moderate, Severe) | Lớp Severe 4% vẫn chiếm **1/3 điểm**, không bị lớp đông lấn át |
| **Mean F1 (macro)** | Macro F1 của từng nhãn bệnh, rồi trung bình **3 nhãn** (ống sống, lỗ liên hợp trái, phải) | Con số tổng cho cả bài toán. Hybrid 0.527 so với 0.420 |
| **Mean Recall / Precision** | Giống Mean F1, thay F1 bằng Recall / Precision | |
| **Severe Recall / F1 / AUC / AUPRC** | Chỉ lấy **mức Severe**, trung bình 3 nhãn bệnh | Đo đúng lớp quan trọng nhất về lâm sàng |
| **± (độ lệch chuẩn)** | Kết quả dao động bao nhiêu giữa **3 lần chạy** với 3 seed | Nhỏ là ổn định. Hybrid Severe Recall 0.486 ± 0.019: cả 3 lần đều quanh 0.46–0.50 |

**Ví dụ đi từng bậc, số thật của Hybrid (seed 42).** Hình dung một bảng 3 hàng (3 nhãn bệnh) × 3
cột (3 mức). Mỗi ô là một F1 riêng, tổng cộng 9 ô. Các chỉ số trên chỉ là các cách gộp 9 ô đó.

| Bậc | Làm gì | Tính thế nào | Ra |
|---|---|---|---|
| 1. Một ô | Ống sống, mức Severe | Có 81 ca nặng thật. Mô hình bắt được 57, sót 24, báo nhầm thêm 90. Recall = 57/81 = 0.704; Precision = 57/(57+90) = 0.388; F1 = 0.500 | **F1 ô này 0.500** |
| 2. Gộp theo hàng (macro) | Ống sống, 3 mức | F1 Normal/Mild 0.952, Moderate 0.390, Severe 0.500 → (0.952 + 0.390 + 0.500)/3 | **Macro F1 ống sống 0.614** |
| 3a. Gộp cả bảng | Lấy macro của 3 hàng | Ống sống 0.614, lỗ liên hợp trái 0.471, phải 0.498 → trung bình | **Mean F1 0.528** |
| 3b. Chỉ lấy cột Severe | Lấy ô Severe của 3 hàng | 0.500, 0.278, 0.252 → trung bình | **Severe F1 0.343** |
| 4. Ba lần chạy | Làm lại bậc 1–3 với seed 42, 123, 456 | Severe F1 0.343 / 0.348 / 0.375 → trung bình ± độ lệch chuẩn | **0.356 ± 0.017** (số trên slide) |

Mean F1 trên slide là **0.527**, là trung bình 3 lần chạy; riêng seed 42 ra 0.528. Mean Recall, Severe
Recall… tính y hệt, chỉ thay F1 trong mỗi ô bằng Recall. Muốn xem cả đường từ ảnh tới con số: file 3,
mục 3.0, phần C.

**Vì sao macro mà không lấy trung bình theo số mẫu (weighted)?** Ở bậc 2, nếu lấy theo số mẫu (giả sử tỉ lệ
81% / 15% / 4%) thì được khoảng 0.81 × 0.952 + 0.15 × 0.390 + 0.04 × 0.500 ≈ **0.85**: ô Severe gần như
không ảnh hưởng. Lấy theo số mẫu thì lớp Normal/Mild
81% quyết định gần hết điểm, mô hình bỏ sót hết ca nặng vẫn được điểm cao.

### 2.3 AUC và AUPRC: không cần ngưỡng, đo khả năng xếp hạng

Hai chỉ số này không nhìn vào quyết định cuối cùng (argmax), mà nhìn vào **điểm xác suất** mô hình
cho từng ca: có xếp ca bệnh lên trên ca không bệnh không.

| Chỉ số | Ý nghĩa dễ nhớ | Đoán bừa được bao nhiêu | Số của luận văn |
|---|---|---|---|
| **AUC** (diện tích dưới đường ROC) | **Bốc ngẫu nhiên 1 ca nặng và 1 ca không nặng: xác suất mô hình chấm ca nặng cao hơn** | **0.5** | Severe AUC 0.899: khoảng 90% số cặp xếp đúng |
| **AUPRC** (diện tích dưới đường Precision–Recall) | Trung bình Precision mỗi khi bắt thêm được một ca bệnh, khi đi từ ca điểm cao xuống thấp | **Bằng tỉ lệ ca dương**: Severe khoảng **0.04** | Severe AUPRC 0.333: khoảng **8 lần** đoán bừa |

**Vì sao cần cả hai?**
- AUC tính cả các ca không bệnh đoán đúng, mà lớp hiếm thì ca không bệnh rất nhiều, nên AUC dễ
  trông cao.
- AUPRC chỉ quan tâm ca bệnh, nên khắt khe hơn với lớp hiếm. Đó là lý do Severe AUC 0.899 nhưng
  Severe AUPRC chỉ 0.333.

**Vì sao AUC cao mà F1 thấp?** AUC đo xếp hạng, F1 đo quyết định sau khi chọn mức cao nhất. Xếp hạng
tốt nhưng ở ngưỡng mặc định vẫn còn nhiều ca không bệnh điểm cao, kéo Precision và F1 xuống.

**Ý nghĩa khi cả hai cùng tăng:** Severe AUC 0.872 → 0.899 và AUPRC 0.284 → 0.333. Vì hai chỉ số
này không phụ thuộc ngưỡng, nên mô hình **thật sự xếp hạng ca nặng tốt hơn**, không phải chỉ hạ
ngưỡng để bắt nhiều hơn.

### 2.4 Chỉ số phân vùng (slide trang 13)

| Chỉ số | Ý nghĩa | Tốt khi | TotalSpineSeg |
|---|---|---|---|
| **Dice** | Vùng khoanh và vùng chuẩn **chồng nhau** bao nhiêu (0 → 1). Chính là F1 tính trên từng voxel | Cao | 70.81% |
| **IoU** | Phần giao chia phần hợp. Giống Dice nhưng chặt hơn (chỉ trong luận văn) | Cao | 55.74% |
| **Precision (voxel)** | Trong vùng mô hình khoanh, bao nhiêu phần **đúng là đốt sống** | Cao | **97.87%**: gần như không khoanh lấn |
| **Recall (voxel)** | Trong đốt sống thật, mô hình **khoanh được** bao nhiêu | Cao | 56.49%: thấp vì không tính gai sau, mỏm ngang |
| **ASSD** | Đường viền khoanh **lệch** đường viền chuẩn trung bình bao nhiêu **mm** | **Thấp** | **6.85 mm**, thấp nhất |

### 2.5 Các con số khác trên slide

| Con số | Ý nghĩa |
|---|---|
| **Mean F1 zero-shot SPIDER 0.362** | Trung bình F1 macro của 8 nhãn SPIDER khi **không huấn luyện thêm gì**, chỉ đổi câu mô tả |
| **Mean AUC zero-shot 0.733** | Trung bình AUC 8 nhãn; trên 0.5 nghĩa là có xếp hạng đúng hơn đoán bừa |
| **Recall 0.500 ở thoát vị (SPIDER)** | Recall trong bảng này là trung bình của lớp "có" và "không". 0.500 = (1.0 + 0.0) / 2: đoán "không" cho **mọi** ca, **không bắt được ca nào** |
| **Inference ms/mẫu** | Thời gian chấm một đĩa đệm. Khoảng 62–67 ms cho mọi cấu hình, nên nhánh BiomedCLIP gần như không làm chậm |
| **Số tham số 1.18M** | Số trọng số được học ở bước 2. Hai nhánh giữ cố định (CBAM-ResNet 3D 63.5M; slide trang 41 ghi nhầm 21.3M là số bản 2D) |

---

## 3. Mười hai con số phải thuộc

| # | Con số | Nói thế nào |
|---|---|---|
| 1 | Severe chiếm **~4%** | "Lớp nặng chỉ khoảng 4% dữ liệu" |
| 2 | Severe Recall **0.123 → 0.486** | "Mô hình nền bắt 1/8 ca nặng, Hybrid bắt gần một nửa, gấp khoảng 4 lần" |
| 3 | Severe F1 **0.152 → 0.356** | "Hơn gấp đôi" |
| 4 | Mean F1 **0.420 → 0.527** | "Cao nhất trong 4 cấu hình" |
| 5 | Mean Accuracy **0.810 → 0.724** | "Giảm vì mô hình nền đoán lớp đa số; em ưu tiên không bỏ sót" |
| 6 | Severe AUC **0.872 → 0.899**, AUPRC **0.284 → 0.333** | "Xếp hạng ca nặng tốt hơn thật, không chỉ hạ ngưỡng" |
| 7 | CBAM 0.262 / BMC-only 0.268 → Hybrid **0.356** (Severe F1) | "Hai nhánh bổ sung cho nhau" |
| 8 | Zero-shot F1 **0.362**, AUC **0.733** | "Chấm được 8 nhãn mới, không huấn luyện lại" |
| 9 | SPIDER: Hybrid **0.653**, SpineNetV2 0.646, CBAM **0.619** | "Hybrid giữ được mức mô hình nền, CBAM một mình thì tụt" |
| 10 | Thoát vị: Recall **0.500 → 0.636** | "Mô hình nền bỏ sót mọi ca thoát vị, Hybrid bắt được một phần" |
| 11 | TotalSpineSeg Precision **97.87%**, ASSD **6.85 mm** | "Khoanh đúng chỗ, dùng để định vị đĩa đệm" |
| 12 | SpineNetV2 F1 **77.5%**, Ning Shen 80.7%, MedGemma Recall **11.8%** | "Chọn SpineNetV2: kém 3 điểm nhưng mã nguồn mở, dễ tinh chỉnh" |

Thêm: học **1.18 triệu** tham số; 3 seed (42, 123, 456); val **395 bệnh nhân**; MIWAI 2026, LNAI
Springer.

---

## 4. Ghi chú phòng khi bị hỏi vặn

Đã tách ra `9_PHONG_KHI_BI_HOI_VAN.md`. **Không cần đọc**, chỉ mở khi bị hỏi trúng.

**Năm câu không được nói:** "Grad-CAM chứng minh…", "Hybrid thắng trên SPIDER", "giải nhất RSNA",
"phần mềm chạy Hybrid", "có ý nghĩa thống kê".
