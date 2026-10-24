# Giải thích các chỉ số đánh giá và cách tính trong luận văn

Tài liệu ôn bảo vệ luận văn (02/11/2026). Mọi số liệu khớp luận văn đã nộp (commit `ef23e38`).
Mỗi chỉ số gồm bốn phần: công thức, cách tính trong code, ý nghĩa, và một ví dụ bằng số thật
của luận văn.

Đọc kèm `1_SCRIPT_BAO_VE_LVTN.md` (kịch bản nói, có giải thích công thức mô hình).
File `DEFENSE_METRICS_VI.md` là bản cũ từ đợt bảo vệ thực tập, dùng số một lần chạy (0.528,
46.4%…). **Đừng lấy số từ file đó.**

Mục lục
1. Bối cảnh: đang đánh giá cái gì
2. Ma trận nhầm lẫn và cách "một-lớp-so-với-phần-còn-lại"
3. Accuracy
4. Recall
5. Precision
6. F1
7. Từ từng lớp đến con số trong bảng: macro, Severe, trung bình ± độ lệch chuẩn
8. AUC (diện tích dưới đường ROC)
9. AUPRC (diện tích dưới đường Precision–Recall)
10. Chọn checkpoint khi huấn luyện
11. Thí nghiệm chọn mô hình phân vùng: Dice, IoU, Precision/Recall voxel, ASSD
12. Thí nghiệm chọn mô hình chấm nhãn: quy về bài toán có/không bệnh
13. Cách tính trên SPIDER (zero-shot và có tinh chỉnh)
14. Bảng tra nhanh và câu trả lời một dòng
15. Slice attention pooling: nhánh BiomedCLIP gộp các lát thế nào

---

## 1. Bối cảnh: đang đánh giá cái gì

Tập validation RSNA 2024 có **1.942 khối ảnh đĩa đệm** (mỗi khối 9 × 112 × 224). Mỗi khối có
**3 nhãn bệnh** $k$, mỗi nhãn có **3 mức** $c$:

| Nhãn bệnh | Normal/Mild | Moderate | Severe |
|---|---|---|---|
| Hẹp ống sống | 1.722 | 139 | **81** |
| Hẹp lỗ liên hợp trái | 1.497 | 360 | **80** |
| Hẹp lỗ liên hợp phải | 1.482 | 372 | **83** |

Tổng cộng 244 ca Severe trên 5.826 lượt chấm, khoảng **4.2%**. Nhãn bị thiếu được ghi `-1` và
bị bỏ ra trước khi tính mọi chỉ số.

Với mỗi khối ảnh và mỗi nhãn bệnh, mô hình trả về:
- **xác suất** của 3 mức (softmax, cộng lại bằng 1), dùng cho AUC và AUPRC;
- **dự đoán** = mức có xác suất cao nhất (argmax), dùng cho Accuracy, Recall, Precision, F1.

Hai nhóm chỉ số khác nhau ở chỗ này. **Recall, Precision, F1 và Accuracy đo quyết định cuối cùng**:
mô hình đã chọn một mức rồi thì chấm đúng hay sai. **AUC và AUPRC đo cách mô hình xếp hạng xác
suất**, không phụ thuộc vào ngưỡng chọn.

---

## 2. Ma trận nhầm lẫn và cách "một-lớp-so-với-phần-còn-lại"

Recall, Precision và F1 vốn định nghĩa cho bài toán hai lớp (dương/âm). Với 3 mức, ta tính
**cho từng lớp một**: lớp đang xét là "dương", hai lớp còn lại gộp thành "âm". Cách này gọi là
**one-vs-rest** (một lớp so với phần còn lại).

Ví dụ với lớp **Severe**, nhãn hẹp ống sống:

| | Mô hình đoán Severe | Mô hình đoán không Severe |
|---|---|---|
| **Thật sự Severe** | **TP** (đúng dương) | **FN** (bỏ sót) |
| **Thật sự không Severe** | **FP** (báo nhầm) | **TN** (đúng âm) |

**Số thật** (Hybrid, seed 42, hẹp ống sống; làm tròn): có 81 ca Severe; Recall 0.704 và
Precision 0.388.

| | Đoán Severe | Đoán không Severe | Tổng |
|---|---|---|---|
| Thật Severe | TP ≈ **57** | FN ≈ **24** | 81 |
| Thật không Severe | FP ≈ **90** | TN ≈ **1.771** | 1.861 |
| Tổng | 147 | 1.795 | 1.942 |

Ví dụ này được dùng xuyên suốt các mục 3–6.

---

## 3. Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Accuracy là tỉ lệ dự đoán đúng trên tổng số mẫu. Với 3 mức, accuracy của một nhãn bệnh là số
khối được đoán đúng mức chia cho tổng số khối. "Mean Accuracy" là trung bình qua 3 nhãn bệnh.

**Vì sao không dùng làm chỉ số chính?** Trên hẹp ống sống, 1.722/1.942 = **88.7%** số khối là
Normal/Mild. Một mô hình đoán Normal/Mild cho **mọi** ca đạt accuracy 88.7% mà không bắt được
ca nặng nào. Đó là lý do SpineNetV2 có Mean Accuracy **0.810**, cao hơn Hybrid (**0.724**): nó
thiên về lớp đa số. Accuracy phản ánh lớp đa số, không phản ánh lớp hiếm.

**Câu trả lời khi bị hỏi "accuracy giảm 8.6 điểm thì sao?":** Mô hình nền đạt accuracy cao nhờ
đoán lớp đa số, đổi lại bỏ sót gần chín trên mười ca nặng. Trong sàng lọc, bỏ sót ca nặng nguy
hiểm hơn báo nhầm ca nhẹ. Báo nhầm thì bác sĩ xem lại ảnh là loại được; bỏ sót thì có thể không
ai phát hiện.

---

## 4. Recall (độ nhạy, sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Recall trả lời câu hỏi: **trong số ca thật sự Severe, mô hình bắt được bao nhiêu phần?**

Ví dụ: $57 / 81 = 0.704$. Mô hình bắt được 57 ca nặng, bỏ sót 24 ca.

**Ý nghĩa lâm sàng:** Recall thấp nghĩa là bỏ sót bệnh nhân nặng. Đây là chỉ số quan trọng nhất
của luận văn.

**Số trong luận văn:** Severe Recall **0.123 → 0.486**. Mô hình nền bắt được khoảng 1 trên 8 ca
nặng, Hybrid bắt được gần 1 trên 2.

---

## 5. Precision (độ chính xác dương)

$$\text{Precision} = \frac{TP}{TP + FP}$$

Precision trả lời câu hỏi: **trong số ca mô hình báo Severe, bao nhiêu ca đúng là Severe?**

Ví dụ: $57 / 147 = 0.388$. Mô hình báo 147 ca nặng, trong đó 57 ca đúng, 90 ca là báo nhầm.

**Ý nghĩa lâm sàng:** Precision thấp nghĩa là báo động giả nhiều, bác sĩ mất thời gian xem lại.

**Đánh đổi giữa Recall và Precision:** Muốn bắt nhiều ca nặng hơn (tăng Recall) thì mô hình
phải "dễ dãi" hơn khi báo Severe, nên báo nhầm cũng nhiều hơn (giảm Precision). Luận văn chấp
nhận đánh đổi này vì báo nhầm ít nguy hiểm hơn bỏ sót.

---

## 6. F1

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 là **trung bình điều hòa** của Precision và Recall.

Ví dụ: $\dfrac{2 \times 0.388 \times 0.704}{0.388 + 0.704} = \dfrac{0.546}{1.092} = 0.500$
(khớp số Severe F1 ống sống seed 42 trong code).

**Vì sao trung bình điều hòa mà không trung bình cộng?** Trung bình điều hòa bị kéo về số nhỏ
hơn. Ví dụ một mô hình báo Severe cho mọi ca thì Recall = 1.0 nhưng Precision = 0.042, trung
bình cộng = 0.52 (trông khá), còn F1 = **0.08** (đúng bản chất là tệ). F1 chỉ cao khi **cả
hai** cùng cao.

---

## 7. Từ từng lớp đến con số trong bảng

Code: `scripts/aggregate_rsna_seeds.py` và hàm `evaluate` trong `train_rsna_hybrid.py`
(dùng `sklearn.metrics.precision_recall_fscore_support(..., average=None)`).

### 7.1 Macro (trung bình đều theo lớp)

$$F1_{macro} = \frac{1}{3}\big(F1_{Normal/Mild} + F1_{Moderate} + F1_{Severe}\big)$$

Mỗi lớp có trọng số bằng nhau, **bất kể lớp đó nhiều hay ít mẫu**.

| Cách trung bình | Công thức | Hệ quả trên RSNA |
|---|---|---|
| **Macro** (luận văn dùng) | trung bình đều 3 lớp | Severe chiếm 4% mẫu nhưng chiếm **1/3** điểm |
| Weighted | trung bình theo số mẫu mỗi lớp | Normal/Mild chiếm ~81% điểm, Severe gần như vô hình |
| Micro | gộp TP/FP/FN mọi lớp rồi tính | với bài toán đơn nhãn thì **bằng accuracy** |

Ví dụ thật (seed 42, Hybrid, ống sống): F1 ba lớp cho macro F1 = 0.614.

### 7.2 Hai tầng trung bình trong bảng RSNA

Mọi chỉ số "Mean" trong bảng 5.1 đều tính theo **hai tầng**:

1. Với mỗi nhãn bệnh $k$: tính chỉ số cho từng lớp, rồi lấy macro qua 3 lớp.
2. Lấy trung bình 3 nhãn bệnh.

$$\text{Mean F1} = \frac{1}{3}\sum_{k \in \{\text{ống sống, LLH trái, LLH phải}\}} F1_{macro}^{(k)}$$

Các chỉ số "Severe" **không** lấy macro, mà lấy chỉ số của riêng lớp Severe ở từng nhãn bệnh,
rồi trung bình 3 nhãn:

$$\text{Severe F1} = \frac{1}{3}\big(F1^{\text{ống sống}}_{Severe} + F1^{\text{LLH trái}}_{Severe} + F1^{\text{LLH phải}}_{Severe}\big)$$

Ví dụ thật (seed 42, Hybrid): Severe F1 của ba nhãn là 0.500, 0.278, 0.252, trung bình
**0.343**. Severe Recall là 0.704, 0.375, 0.313, trung bình **0.464**. Ống sống tốt hơn hẳn lỗ
liên hợp, đúng như phần Hạn chế đã nói.

### 7.3 Trung bình ± độ lệch chuẩn qua 3 lần chạy

Mỗi cấu hình được huấn luyện 3 lần với seed 42, 123, 456 (seed quyết định khởi tạo trọng số và
thứ tự xáo trộn dữ liệu). Mỗi lần cho một số; bảng báo cáo:

$$\bar{x} = \frac{x_1 + x_2 + x_3}{3}, \qquad s = \sqrt{\frac{\sum_i (x_i - \bar{x})^2}{n - 1}}$$

Ví dụ Severe F1 của Hybrid: 0.343, 0.348, 0.375 → trung bình **0.356**, độ lệch chuẩn
**0.017**.

**Ý nghĩa của độ lệch chuẩn:** nó cho biết kết quả có ổn định giữa các lần chạy không. CBAM
có Severe F1 0.262 ± **0.082** (dao động mạnh), Hybrid 0.356 ± **0.017** (ổn định). Nghĩa là
ghép hai nhánh còn giúp mô hình ổn định hơn.

> ⚠️ **Lệch nhỏ cần biết:** riêng dòng **Severe Recall** trong bảng luận văn và slide dùng độ
> lệch chuẩn chia cho $n$ (tổng thể) thay vì $n - 1$. Tính lại từ file kết quả:
> Hybrid 0.464 / 0.497 / 0.497 cho 0.486 ± **0.019** (chia $n-1$) thay vì ± 0.016;
> SpineNetV2 0.115 / 0.127 / 0.125 cho 0.123 ± **0.006** thay vì ± 0.005. Các dòng khác đều
> chia $n - 1$. Trung bình không đổi, kết luận không đổi. Nếu bị hỏi: "Dòng đó em dùng độ lệch
> chuẩn tổng thể; tính theo độ lệch chuẩn mẫu thì là 0.019, kết luận vẫn giữ nguyên."

**Ba lần chạy có đủ không?** Không đủ để làm kiểm định thống kê chặt chẽ, luận văn đã nêu trong
Hạn chế. Nhưng khoảng cách Severe Recall 0.123 → 0.486 lớn hơn rất nhiều so với độ lệch chuẩn
(0.006 và 0.019), nên khó là do may mắn.

---

## 8. AUC (diện tích dưới đường ROC)

### 8.1 Đường ROC

Mô hình cho ra **xác suất** Severe, ví dụ 0.73. Muốn ra quyết định thì phải chọn một **ngưỡng**:
xác suất trên ngưỡng thì báo Severe. Mỗi ngưỡng cho một cặp:

$$\text{TPR} = \text{Recall} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}$$

- TPR: tỉ lệ ca Severe được bắt.
- FPR: tỉ lệ ca **không** Severe bị báo nhầm.

Quét ngưỡng từ 1 xuống 0 thì được một đường cong đi từ (0, 0) đến (1, 1). Đó là **đường ROC**.
AUC là diện tích dưới đường này:

$$\text{AUC} = \int_0^1 \text{TPR}\; d(\text{FPR})$$

### 8.2 Cách hiểu dễ nhớ nhất

> **AUC = xác suất để một ca Severe chọn ngẫu nhiên được mô hình cho điểm Severe cao hơn một
> ca không-Severe chọn ngẫu nhiên.**

- AUC = 0.5: đoán ngẫu nhiên.
- AUC = 1.0: mọi ca Severe đều được xếp trên mọi ca không-Severe.
- AUC = 0.899 (Severe, Hybrid): khoảng 90% số cặp (ca nặng, ca không nặng) được xếp đúng thứ tự.

### 8.3 Ví dụ tính tay

20 khối ảnh, trong đó **2 ca Severe**, xếp theo xác suất Severe giảm dần:

| Hạng | 1 | 2 | **3** | 4 | 5 | **6** | 7 … 20 |
|---|---|---|---|---|---|---|---|
| Nhãn thật | không | không | **Severe** | không | không | **Severe** | không |

Đếm các cặp (Severe, không-Severe): có $2 \times 18 = 36$ cặp.
- Ca Severe hạng 3: đứng trên 16 ca không-Severe (hạng 4, 5, 7…20).
- Ca Severe hạng 6: đứng trên 14 ca không-Severe (hạng 7…20).

$$\text{AUC} = \frac{16 + 14}{36} = 0.833$$

### 8.4 Cách tính trong code

`spinenet/auc_metrics.py`, hàm `compute_auc_auprc_per_condition`:
- Với mỗi nhãn bệnh và mỗi lớp $c$: nhãn nhị phân = (mức thật == $c$), điểm = xác suất lớp $c$,
  gọi `sklearn.metrics.roc_auc_score`. Đây là AUC **one-vs-rest**.
- **Mean AUC** = macro 3 lớp, trung bình 3 nhãn bệnh (hai tầng như mục 7.2).
- **Severe AUC** = AUC lớp Severe, trung bình 3 nhãn bệnh.

### 8.5 Ưu và nhược điểm

- **Ưu:** không phụ thuộc ngưỡng; đo chất lượng **xếp hạng** của xác suất.
- **Nhược trên dữ liệu mất cân bằng:** FPR có mẫu số là $FP + TN$, mà TN rất lớn (hàng nghìn
  ca Normal/Mild). Báo nhầm 90 ca cũng chỉ làm FPR tăng 90/1.861 ≈ 0.05. Vì vậy AUC có thể
  **trông cao dù mô hình báo nhầm rất nhiều**. Đó là lý do cần thêm AUPRC.

**Vì sao Severe AUC (0.899) lại cao hơn Mean AUC (0.838)?** Mean AUC gồm cả lớp Moderate, mà
Moderate là lớp "ở giữa", khó tách khỏi hai lớp hai bên (AUC Moderate chỉ khoảng 0.75). Ca
Severe thì khác biệt rõ hơn so với phần còn lại nên dễ xếp hạng hơn.

---

## 9. AUPRC (diện tích dưới đường Precision–Recall)

### 9.1 Đường PR

Cũng quét ngưỡng như ROC, nhưng mỗi ngưỡng lấy cặp (Recall, Precision). Diện tích dưới đường
này là AUPRC.

**Khác biệt then chốt:** Precision có mẫu số là $TP + FP$, **không có TN**. Vì vậy hàng nghìn ca
Normal/Mild đúng âm không còn "che" được số ca báo nhầm.

### 9.2 Cách tính trong code: Average Precision

Code dùng `sklearn.metrics.average_precision_score`:

$$\text{AP} = \sum_n (R_n - R_{n-1})\, P_n$$

Đi dần xuống danh sách đã xếp hạng. Mỗi lần gặp một ca dương thì Recall tăng một bậc; lấy
Precision tại điểm đó nhân với bước tăng Recall, rồi cộng lại.

Tính tiếp ví dụ 20 khối ảnh ở mục 8.3:
- Tại hạng 3 (gặp ca Severe thứ 1): Recall = 1/2, Precision = 1/3.
- Tại hạng 6 (gặp ca Severe thứ 2): Recall = 2/2, Precision = 2/6 = 1/3.

$$\text{AP} = 0.5 \times \tfrac{1}{3} + 0.5 \times \tfrac{1}{3} = 0.333$$

Cùng một danh sách xếp hạng, **AUC = 0.833 nhưng AUPRC chỉ 0.333**. Hai ca không-Severe ở hạng 1
và 2 hầu như không làm giảm AUC, nhưng kéo Precision xuống mạnh. Ví dụ này giống hệt tình huống
thật: Hybrid có Severe AUC 0.899 nhưng Severe AUPRC 0.333.

### 9.3 Mốc so sánh: AUPRC của đoán ngẫu nhiên

- AUC ngẫu nhiên luôn là **0.5**.
- AUPRC ngẫu nhiên bằng **tỉ lệ ca dương**. Với Severe trên RSNA, tỉ lệ này khoảng **0.042**.

Vậy Severe AUPRC 0.333 **gấp khoảng 8 lần** đoán ngẫu nhiên, dù con số trông thấp. Ví dụ thật
tốt nhất: hẹp ống sống seed 42, Severe AUPRC **0.594** so với mức ngẫu nhiên 81/1.942 = 0.042,
tức gấp 14 lần.

### 9.4 So sánh AUC và AUPRC

| | AUC | AUPRC |
|---|---|---|
| Trục | TPR theo FPR | Precision theo Recall |
| Có dùng TN? | Có | **Không** |
| Mức ngẫu nhiên | 0.5 | tỉ lệ ca dương (≈ 0.04 với Severe) |
| Nhạy với báo nhầm ở lớp hiếm | Ít | **Nhiều** |
| Dùng để | xếp hạng tổng quát | đánh giá riêng lớp hiếm |

**Câu trả lời một dòng:** "AUC đo mô hình xếp hạng tốt không. AUPRC đo xếp hạng tốt không *khi
chỉ quan tâm tới lớp hiếm*. Với lớp chỉ chiếm 4%, AUPRC phản ánh đúng hơn."

---

## 10. Chọn checkpoint khi huấn luyện

Đây là "cách tính để chọn mô hình" **bên trong** một lần huấn luyện. Phần chọn mô hình **giữa
các phương pháp** ở mục 11 và 12.

Code: `train_rsna_hybrid.py`, tham số `--select-by` (mặc định `severe_f1`).

1. Sau mỗi epoch, đánh giá trên tập validation.
2. Tính **Severe F1 trung bình 3 nhãn bệnh** (mục 7.2).
3. Nếu cao hơn mức tốt nhất trước đó thì lưu checkpoint `best_model_*.pth` và ghi nhận epoch.
4. **Dừng sớm** nếu 5 epoch liên tiếp không cải thiện; tối đa 20 epoch.
5. Số liệu báo cáo là của checkpoint tốt nhất, **không phải** epoch cuối cùng.

**Vì sao chọn theo Severe F1?** Vì đó là mục tiêu chính của luận văn. Chọn theo validation loss
hoặc accuracy thì sẽ ưu tiên checkpoint thiên về lớp đa số.

**Câu hỏi có thể gặp:** "Chọn checkpoint trên validation rồi báo cáo trên validation, có lạc
quan quá không?" Trả lời thật: có một phần. RSNA chỉ công bố nhãn của tập train, nên luận văn chia
80/20 theo bệnh nhân; chưa có tập test riêng. Tuy vậy cả bốn cấu hình đều được chọn theo cùng một
quy tắc, nên so sánh giữa chúng vẫn công bằng. Thêm một tập test độc lập là việc nên làm tiếp.

> Ghi chú: code có ghi lại một phát hiện sau luận văn. Severe F1 dao động giữa các epoch nhiều
> hơn Severe AUPRC khoảng 10–15 lần, nên chọn theo F1 dễ "bắt đỉnh" của một lần dao động. Số liệu
> luận văn vẫn dùng `severe_f1` như các lần chạy đã công bố. Chỉ nói chuyện này nếu bị hỏi sâu.

---

## 11. Thí nghiệm chọn mô hình phân vùng

Luận văn mục 5.2, slide trang 13. Bốn mô hình (TotalSpineSeg, SPINEPS, MedCLIP-SAMv2,
SpineNetV2) chạy trên **23 ca** của bộ Lumbar Spine Segmentation (Chu và cộng sự). Mỗi chỉ số
tính **riêng cho từng ca**, rồi lấy **trung bình 23 ca**.

Ký hiệu: $P$ = tập voxel mô hình đánh dấu là đốt sống; $G$ = tập voxel đốt sống theo nhãn chuẩn.
Voxel là "điểm ảnh 3 chiều". Nhãn chuẩn của bộ này là **một mặt nạ nhị phân cho toàn bộ đốt
sống**, gồm cả thân đốt, gai sau và mỏm ngang.

### 11.1 Dice

$$\text{Dice} = \frac{2\,|P \cap G|}{|P| + |G|}$$

Đo độ chồng lắp, từ 0 (không chạm nhau) đến 1 (trùng khít).

Ví dụ: $|P| = 100$ voxel, $|G| = 120$ voxel, phần giao 80 voxel:
$\text{Dice} = \dfrac{2 \times 80}{100 + 120} = 0.727$.

**Mẹo nhớ:** Dice **chính là F1 tính ở mức voxel**. Phần giao là TP, $|P| = TP + FP$,
$|G| = TP + FN$, nên $\dfrac{2TP}{2TP + FP + FN}$ đúng bằng công thức F1.

### 11.2 IoU (Jaccard, có trong luận văn, không có trên slide)

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

Ví dụ trên: phần hợp $= 100 + 120 - 80 = 140$, $\text{IoU} = 80/140 = 0.571$.
Quan hệ: $\text{Dice} = \dfrac{2\,\text{IoU}}{1 + \text{IoU}}$, nên hai chỉ số luôn xếp hạng các
mô hình giống nhau. Trên bảng, quan hệ này chỉ đúng gần đúng vì mỗi chỉ số được lấy trung bình
riêng qua 23 ca.

### 11.3 Precision và Recall ở mức voxel

$$\text{Precision} = \frac{|P \cap G|}{|P|}, \qquad \text{Recall} = \frac{|P \cap G|}{|G|}$$

Ví dụ trên: Precision = 80/100 = 0.80 (trong vùng khoanh, 80% đúng là đốt sống); Recall =
80/120 = 0.67 (khoanh được 67% đốt sống thật).

### 11.4 ASSD (khoảng cách bề mặt đối xứng trung bình)

$$\text{ASSD} = \frac{1}{|\partial P| + |\partial G|}\left(\sum_{p \in \partial P} d(p, \partial G) + \sum_{g \in \partial G} d(g, \partial P)\right)$$

- $\partial P$, $\partial G$: các điểm nằm trên **bề mặt** (đường biên 3D) của vùng dự đoán và
  vùng chuẩn.
- $d(x, S)$: khoảng cách Euclid ngắn nhất từ điểm $x$ đến bề mặt $S$, tính bằng **mm** (đã nhân
  với kích thước voxel thật).
- **Đối xứng**: đo cả hai chiều, từ biên dự đoán sang biên chuẩn **và** ngược lại, rồi lấy
  trung bình chung. Nếu chỉ đo một chiều thì một mô hình khoanh thiếu hẳn một đốt sống vẫn có
  thể có khoảng cách nhỏ.

Ví dụ: biên dự đoán có 4 điểm, khoảng cách tới biên chuẩn là 1, 1, 2, 4 mm; biên chuẩn có 4
điểm, khoảng cách tới biên dự đoán là 1, 2, 2, 3 mm. Vậy
$\text{ASSD} = (8 + 8)/8 = 2$ mm.

**Dice và ASSD đo hai thứ khác nhau.** Dice đo "chồng lên nhau bao nhiêu", ASSD đo "đường viền
lệch bao xa". Hai vùng có thể Dice khá mà biên vẫn lệch xa ở vài chỗ, và ASSD sẽ bắt được điều
đó.

### 11.5 Cách đọc bảng và lý do chọn TotalSpineSeg

| | TotalSpineSeg | SPINEPS | MedCLIP-SAMv2 | SpineNetV2 |
|---|---|---|---|---|
| Dice | **70.81%** | 58.68% | 40.55% | 67.82% |
| IoU | **55.74%** | 41.56% | 26.47% | 51.41% |
| Precision | **97.87%** | 52.17% | 29.16% | 51.76% |
| Recall | 56.49% | 68.34% | **91.98%** | 91.79% |
| ASSD | **6.85 mm** | 7.48 mm | 19.50 mm | 8.04 mm |

- **Recall của TotalSpineSeg thấp (56%) không phải vì nó kém.** Nó chỉ khoanh **thân đốt sống**,
  còn nhãn chuẩn tính cả gai sau và mỏm ngang. Phần nó bỏ là phần nằm ngoài định nghĩa của nó.
- **Precision 97.87%**: gần như mọi voxel nó khoanh đều đúng là đốt sống.
- **Tiêu chí chọn theo nhu cầu của hệ thống**: mặt nạ chỉ dùng để định vị đĩa đệm (nằm giữa hai
  thân đốt) rồi cắt khối ảnh. Cần **khoanh đúng chỗ** (Precision cao, ASSD thấp), không cần
  khoanh đủ gai sau.
- SpineNetV2 và MedCLIP-SAMv2 có Recall cao vì khoanh rộng, lấn sang mô xung quanh (Precision
  chỉ 30–52%).
- Độ ổn định: Dice của TotalSpineSeg có độ lệch chuẩn ±5.23% giữa 23 ca.

---

## 12. Thí nghiệm chọn mô hình chấm nhãn

Luận văn mục 5.3, slide trang 14. Ba mô hình (Ning Shen, SpineNetV2, MedGemma) có đầu ra khác
nhau: MedGemma trả lời bằng văn bản, Ning Shen và SpineNetV2 cho mức độ theo cách riêng. Để so
được, cả ba được **quy về cùng một bài toán nhị phân: có bệnh hay không**.

### 12.1 Quy tắc quy về nhị phân

1. **Từ mức độ sang có/không:** Normal/Mild → **0** (không bệnh); Moderate hoặc Severe → **1**
   (có bệnh).
2. **Từ tầng đĩa đệm sang bệnh nhân (luật ANY):** mỗi bệnh nhân có 5 tầng (L1/L2 … L5/S1). Bệnh
   nhân được tính "có bệnh" ở một nhóm nếu **ít nhất một tầng** có bệnh. Áp dụng cho cả nhãn
   thật lẫn dự đoán.
3. **Cùng tập bệnh nhân:** chỉ so trên những bệnh nhân mà cả ba mô hình đều có dự đoán (phép
   giao theo `study_id`). Kết quả là **150 bệnh nhân** của tập validation RSNA.
4. Tính Accuracy, Precision, Recall, F1 **của lớp "có bệnh"** cho **từng nhóm bệnh** (ống sống,
   LLH trái, LLH phải).
5. Cột trên slide = **trung bình cộng 3 nhóm bệnh**.

Ví dụ: một bệnh nhân có ống sống Mild ở L1–L4 và Moderate ở L4/L5, thì nhãn nhóm ống sống của
bệnh nhân đó là **1**.

### 12.2 Kiểm tra lại số trên slide

| Mô hình | F1 ống sống | F1 LLH trái | F1 LLH phải | **Trung bình** |
|---|---|---|---|---|
| Ning Shen | 81.20 | 79.43 | 81.48 | **80.70** |
| SpineNetV2 | 78.79 | 77.30 | 76.40 | **77.50** |
| MedGemma | 31.17 | 13.86 | 10.75 | **18.59** |

Recall của MedGemma: $(20.87 + 8.43 + 6.02)/3 = 11.77\%$, tức "khoảng 12%" trên slide.

> **Lưu ý tinh tế:** cột F1 trung bình là **trung bình của ba F1**, không phải F1 tính từ
> Precision và Recall trung bình. Với Ning Shen: $2 \times 77.7 \times 84.2 / (77.7 + 84.2) =
> 80.8$, khác 80.7 một chút. Nếu ai đó tính lại từ hai cột trên slide mà ra số hơi khác thì đây
> là lý do.

### 12.3 Vì sao F1 ở thí nghiệm này cao (77–81%) mà F1 ở bảng RSNA chỉ 0.42–0.53?

Hai thí nghiệm đo **hai bài toán khác nhau**:

| | Thí nghiệm chọn mô hình (slide 14) | Thí nghiệm chính (slide 21) |
|---|---|---|
| Đơn vị | bệnh nhân (luật ANY trên 5 tầng) | từng đĩa đệm |
| Số lớp | 2 (có/không bệnh) | 3 (Normal/Mild, Moderate, Severe) |
| Gộp Moderate + Severe | có | không, Severe tách riêng |
| F1 | F1 của lớp "có bệnh" | **macro** F1 qua 3 lớp |
| Số mẫu | 150 bệnh nhân | 1.942 đĩa đệm |

Phát hiện có bệnh ở mức bệnh nhân dễ hơn nhiều: chỉ cần đúng một tầng, và không phải phân biệt
Moderate với Severe. Vì vậy cuối mục 5.3 luận văn nói: F1 77.5% chỉ đo có/không bệnh; khi chấm
đúng 3 mức, SpineNetV2 chỉ bắt được 12.3% ca Severe. **Đây là cầu nối từ thí nghiệm chọn mô hình
sang mô hình đề xuất.**

### 12.4 Tiêu chí chọn

Luận văn đặt hai điều kiện: (1) đủ chính xác cho sàng lọc; (2) mã nguồn mở, kiến trúc đủ đơn
giản để tinh chỉnh.
- Ning Shen: thỏa (1) tốt nhất, nhưng là tổ hợp nhiều mô hình lớn (Faster R-CNN + Swin
  Transformer), khó thỏa (2).
- SpineNetV2: kém Ning Shen khoảng 3 điểm F1, thỏa cả (1) lẫn (2). **Được chọn.**
- MedGemma: không thỏa (1), Recall khoảng 12%.

---

## 13. Cách tính trên SPIDER

### 13.1 Zero-shot (slide 23)

Code: `eval_zeroshot_spider.py`. Kết quả lưu ở `experiments/paper_results/spider_zeroshot/`.

1. Mỗi mức của mỗi nhãn SPIDER được viết thành một câu mô tả theo mẫu
   *"a magnetic resonance image of [label]"*. Câu mẫu được viết **trước** khi chạy, không dò
   theo kết quả.
2. Mô hình Hybrid đã học trên RSNA, **không huấn luyện thêm**, tính cô-sin giữa vector ảnh và
   vector từng câu.
3. Với mỗi nhãn: dự đoán = câu có cô-sin cao nhất. F1 = **macro F1** qua các mức của nhãn đó;
   AUC = AUC one-vs-rest macro, tính từ softmax của các cô-sin.
4. Dòng "Trung bình (8)" = trung bình đều 8 nhãn: F1 **0.362**, AUC **0.733**.

**Đọc kết quả:** nhãn hai lớp như phồng, thoát vị, hẹp khe đĩa đạt F1 0.56–0.61. Pfirrmann (5
mức) và Modic (4 mức) rất thấp, vì phân biệt nhiều mức chỉ bằng câu mô tả là việc khó. Trượt đốt
sống có AUC 0.807 nhưng F1 0.028: xếp hạng khá, nhưng ngưỡng mặc định (argmax) gần như không bao
giờ chọn lớp hiếm. Đây là ví dụ thật cho việc AUC và F1 đo hai thứ khác nhau (mục 1).

### 13.2 Có tinh chỉnh (slide 24)

Code: `train_spider_hybrid.py` và `train_spider.py`. Cả bốn cấu hình được huấn luyện thêm trên
SPIDER. Cách tính giống RSNA: với mỗi nhãn, tính macro qua các mức, rồi **trung bình qua các
nhãn**. Mean Accuracy là trung bình accuracy các nhãn.

Bảng chi tiết (trang 52) cho thấy một điểm hay: với thoát vị, SpineNetV2 có **Recall 0.500**.
Đây là **macro Recall của bài toán hai lớp**. Khi mô hình đoán "không thoát vị" cho mọi ca thì
Recall lớp "không" = 1.0 và Recall lớp "có" = 0, nên macro = 0.5. Vì vậy slide nói SpineNetV2
"đoán âm tính cho mọi ca". Hybrid đạt 0.636, tức bắt được một phần ca thoát vị thật.

---

## 14. Bảng tra nhanh và câu trả lời một dòng

| Chỉ số | Công thức | Hỏi gì | Dùng ở đâu trong luận văn |
|---|---|---|---|
| Accuracy | (TP+TN)/tổng | Đoán đúng bao nhiêu phần? | Mean Accuracy (văn xuôi), chọn mô hình chấm nhãn |
| Recall | TP/(TP+FN) | Bắt được bao nhiêu ca bệnh? | **Severe Recall: chỉ số chính** |
| Precision | TP/(TP+FP) | Báo bệnh thì đúng bao nhiêu? | Bảng RSNA, phân vùng |
| F1 | 2PR/(P+R) | Cân bằng P và R | Mean F1 macro, Severe F1, chọn mô hình |
| Macro | trung bình đều các lớp | Lớp hiếm có được tính công bằng không? | Mọi chỉ số "Mean" |
| AUC | diện tích dưới ROC | Xếp hạng có tốt không? | Mean AUC, Severe AUC |
| AUPRC | diện tích dưới PR | Xếp hạng lớp hiếm có tốt không? | Mean AUPRC, Severe AUPRC |
| Dice | 2\|P∩G\|/(\|P\|+\|G\|) | Chồng lắp bao nhiêu? | Chọn mô hình phân vùng |
| IoU | \|P∩G\|/\|P∪G\| | Như Dice, chặt hơn | Chọn mô hình phân vùng |
| ASSD | trung bình khoảng cách biên (mm) | Đường viền lệch bao xa? | Chọn mô hình phân vùng |

**Câu trả lời một dòng khi bị hỏi:**
- *Recall?* "Trong các ca thật sự nặng, mô hình bắt được bao nhiêu phần."
- *Precision?* "Trong các ca mô hình báo nặng, bao nhiêu ca đúng là nặng."
- *F1?* "Trung bình điều hòa của hai cái trên; chỉ cao khi cả hai cùng cao."
- *Macro?* "Mỗi lớp tính ngang nhau, nên lớp Severe 4% vẫn chiếm một phần ba điểm."
- *AUC?* "Xác suất một ca nặng được mô hình xếp trên một ca không nặng; 0.5 là đoán bừa."
- *AUPRC?* "Như AUC nhưng không bị lớp đa số làm đẹp số; đoán bừa chỉ được khoảng 0.04."
- *Sao AUC 0.9 mà F1 chỉ 0.36?* "AUC đo xếp hạng, không cần ngưỡng; F1 đo quyết định sau khi chọn
  mức cao nhất. Mô hình xếp hạng tốt nhưng ở ngưỡng mặc định vẫn còn báo nhầm và bỏ sót."
- *Dice?* "Độ chồng lắp giữa vùng khoanh và vùng chuẩn; thực chất là F1 ở mức voxel."
- *ASSD?* "Khoảng cách trung bình giữa hai đường viền, tính bằng mm, càng nhỏ càng tốt."

---

## 15. Slice attention pooling: nhánh BiomedCLIP gộp các lát thế nào

Phần này không phải metric, nhưng hội đồng hay hỏi "f_bmc lấy từ đâu". Công thức nằm ở dòng đầu
slide phụ lục trang 40. Slide công thức không trình bày, chỉ mở khi được hỏi.

### 15.1 Vì sao cần gộp lát

- Khối ảnh của một đĩa đệm có **9 lát** mặt cắt dọc, mỗi lát 112 × 224.
- BiomedCLIP là mô hình **2D**: mỗi lần chỉ đọc một ảnh phẳng và trả về một vector 512 chiều.
- Đưa 3 lát vào thì được 3 vector, nhưng khối hợp nhất cần **một** vector f_bmc. Bước gộp 3
  vector thành 1 chính là slice attention pooling.

### 15.2 Từng bước, theo code

Code: `spinenet/models/grading_hybrid.py`, lớp `SliceAttentionPool` và hàm `forward` của mô hình.

| Bước | Làm gì | Kích thước |
|---|---|---|
| 1. Chọn lát | Lấy 3 lát giữa khối: lát 4, 5, 6 trên 9 (code `volume[:, 0, 3:6]`) | 3 × 112 × 224 |
| 2. Chuẩn bị ảnh | Mỗi lát: đổi kích thước về 224 × 224, nhân thành 3 kênh màu, chuẩn hóa theo thống kê của CLIP | 3 × (3 × 224 × 224) |
| 3. Mã hóa | Bộ mã hóa ảnh ViT-B/16 của BiomedCLIP, **giữ cố định**, ra vector $z_i$ cho từng lát | 3 × 512 |
| 4. Chấm điểm | Mỗi lát một điểm: $e_i = w_a^\top z_i + b$ (một lớp tuyến tính 512 → 1) | 3 số |
| 5. Softmax | Biến 3 điểm thành 3 trọng số $\alpha_i$, cộng lại bằng 1 | 3 số |
| 6. Cộng có trọng số | $f_{bmc} = \alpha_1 z_1 + \alpha_2 z_2 + \alpha_3 z_3$ | 512 |

Công thức:

$$\alpha_i = \frac{\exp(w_a^\top z_i)}{\sum_{j=1}^{3}\exp(w_a^\top z_j)}, \qquad f_{bmc} = \sum_{i=1}^{3} \alpha_i z_i$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $z_i$ | vector 512 chiều của lát thứ $i$, do BiomedCLIP tạo ra |
| $w_a$ | vector 512 chiều **học được**: mô hình học "lát thế nào thì đáng chú ý" |
| $w_a^\top z_i$ | điểm chú ý của lát $i$ (tích vô hướng); càng lớn thì lát càng được coi trọng |
| $\alpha_i$ | trọng số của lát $i$, nằm trong (0, 1), ba trọng số cộng lại bằng 1 |
| $f_{bmc}$ | vector đại diện cho cả khối ảnh, đưa vào khối hợp nhất cùng với $f_{cbam}$ |

Trong code, lớp chấm điểm có thêm độ lệch $b$, nhưng $b$ **không ảnh hưởng**: cộng cùng một số vào
cả 3 điểm thì softmax không đổi. Vì vậy công thức trong luận văn bỏ $b$.

### 15.3 Ví dụ bằng số

Giả sử điểm của 3 lát là $w_a^\top z_i$ = 2, 1, 0.

| | Lát 4 | Lát 5 | Lát 6 | Tổng |
|---|---|---|---|---|
| Điểm $e_i$ | 2 | 1 | 0 | |
| $\exp(e_i)$ | 7.39 | 2.72 | 1.00 | 11.11 |
| $\alpha_i$ = $\exp(e_i)$ / tổng | **0.67** | 0.24 | 0.09 | 1.00 |

Kết quả: $f_{bmc} = 0.67 z_4 + 0.24 z_5 + 0.09 z_6$. Lát có điểm cao nhất đóng góp khoảng 2/3,
thay vì 1/3 như khi lấy trung bình đều.

### 15.4 Những điểm dễ bị hỏi

- **Trọng số α có cố định cho từng vị trí lát không?** Không. $w_a$ cố định sau khi học, nhưng
  $z_i$ khác nhau ở mỗi đĩa đệm, nên mỗi đĩa đệm có bộ α riêng. Đĩa này có thể dồn trọng số vào
  lát 5, đĩa khác vào lát 4.
- **Không có nhãn "lát nào quan trọng", sao học được $w_a$?** $w_a$ được học cùng khối hợp nhất,
  qua cùng hàm loss chấm mức độ. Nếu dồn trọng số vào một lát giúp chấm đúng hơn, đạo hàm sẽ đẩy
  $w_a$ theo hướng đó. Không cần nhãn riêng cho từng lát.
- **Sao không lấy trung bình đều?** Trung bình đều là trường hợp đặc biệt α = 1/3, 1/3, 1/3. Lớp
  chú ý để mô hình tự chọn. Nếu trung bình đều là tốt nhất thì mô hình vẫn học ra được (khi $w_a$
  cho 3 điểm bằng nhau).
- **Sao không lấy max?** Max lấy giá trị lớn nhất theo từng chiều, nên vector kết quả ghép từ nhiều
  lát khác nhau, không còn là vector của một lát có thật. Cộng có trọng số cho ra vector nằm "giữa"
  các lát thật, hợp với không gian mà BiomedCLIP đã học.
- **Sao chỉ 3 lát, không đủ 9?** BiomedCLIP chạy riêng từng lát, mỗi lát một lượt ViT, nên 9 lát
  tốn gấp 3. Ba lát giữa đi qua tâm khối ảnh. Thông tin 3 chiều đầy đủ đã có nhánh CBAM đọc cả 9
  lát.
- **Phần này có bao nhiêu tham số học?** 512 số của $w_a$ và 1 số $b$, tổng 513. Rất nhỏ so với
  khoảng 1.18 triệu của khối hợp nhất.
- **BiomedCLIP có được học lại không?** Không. Bộ mã hóa giữ cố định (code chạy dưới
  `torch.no_grad()`), chỉ $w_a$ và các phần phía sau được học.
- **Đổi 112 × 224 thành 224 × 224 có làm méo ảnh không?** Có, ảnh bị giãn gấp đôi theo chiều dọc.
  Luận văn chọn cách đơn giản này vì BiomedCLIP giữ cố định và nhánh CBAM vẫn đọc ảnh gốc. Nếu bị
  hỏi, thừa nhận và nói có thể cải thiện bằng cách đệm thêm viền (padding) thay cho kéo giãn.
- **Trọng số α có dùng để giải thích mô hình được không?** Luận văn **không** phân tích α, nên
  đừng nói "mô hình chọn đúng lát có tổn thương". Chỉ nói: lớp chú ý cho phép mô hình tự cân các
  lát.

**Trả lời một dòng:** "BiomedCLIP đọc từng lát 2D, nên em mã hóa 3 lát giữa thành 3 vector, chấm
điểm từng lát bằng một vector học được, đưa qua softmax ra trọng số, rồi cộng có trọng số thành
một vector f_bmc."
