# Giải thích các chỉ số đánh giá (cho defense)

> Tài liệu tham khảo cho defense: định nghĩa, ý nghĩa, chiều tốt/xấu, cách tính và ví dụ cho TẤT CẢ metric dùng trong báo cáo.
>
> - **PHẦN A — Metric phân loại / grading**: Recall, Precision, F1, AUC, AUPRC.
> - **PHẦN B — Metric phân vùng / segmentation**: Dice, Precision/Recall (voxel), ASSD.

---

## PHẦN A — Metric phân loại (grading)

## Setup chung — Bài toán Severe class

Giả sử có **100 ca bệnh** đang test, trong đó:
- **10 ca thật sự bị Severe** (lớp hiếm — đây là cái mình quan tâm)
- **90 ca thật sự Not-Severe** (Normal/Mild/Moderate)

Model chạy xong cho ra kết quả:

|  | Model bảo "Severe" | Model bảo "Not-Severe" |
|---|---|---|
| **Thực tế Severe** | **6 ca** ✅ (đúng) | **4 ca** ❌ (bỏ sót) |
| **Thực tế Not-Severe** | **2 ca** ❌ (báo nhầm) | **88 ca** ✅ (đúng) |

Đây gọi là **Confusion Matrix**. 4 ô có tên kỹ thuật:

| Ký hiệu | Tên (EN) | Ý nghĩa lâm sàng |
|---|---|---|
| **TP = 6** | True Positive | Bệnh nhân Severe → model phát hiện đúng |
| **FN = 4** | False Negative | Bệnh nhân Severe → model bỏ sót (NGUY HIỂM) |
| **FP = 2** | False Positive | Người khỏe → model báo Severe nhầm (gây hoang mang) |
| **TN = 88** | True Negative | Người khỏe → model nói khỏe (đúng) |

---

## 1. Recall (Độ nhạy / Sensitivity)

### Công thức
```
Recall = TP / (TP + FN)
       = số ca Severe MODEL BẮT ĐƯỢC / tổng số ca Severe THỰC TẾ
```

### Ví dụ
```
Recall = 6 / (6 + 4) = 6/10 = 60%
```

### Ý nghĩa lâm sàng

> *"Trong 10 bệnh nhân thật sự bị Severe, model chỉ phát hiện được 6 người. **4 bệnh nhân Severe bị bỏ sót** — họ về nhà mà không được điều trị."*

**Recall thấp = nguy hiểm cho bệnh nhân**. Trong y khoa, Recall (Sensitivity) thường được ưu tiên hơn Precision vì **bỏ sót bệnh** nguy hiểm hơn **báo động giả**.

---

## 2. Precision (Độ chính xác)

### Công thức
```
Precision = TP / (TP + FP)
          = số ca model GỌI ĐÚNG là Severe / tổng số ca model GỌI là Severe
```

### Ví dụ
```
Precision = 6 / (6 + 2) = 6/8 = 75%
```

### Ý nghĩa lâm sàng

> *"Trong 8 lần model 'báo động' Severe, có 6 lần đúng và 2 lần báo nhầm. Tức là **75% lần model nói Severe là đáng tin cậy**, 25% là báo động giả."*

**Precision thấp** = model "kêu loạn" — bệnh nhân bị triệu tập đi MRI tiếp / sinh thiết tiếp dù không cần thiết.

---

## 3. F1 Score (Trung bình điều hòa)

### Công thức
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Ví dụ
```
F1 = 2 × (0.75 × 0.60) / (0.75 + 0.60)
   = 2 × 0.45 / 1.35
   = 0.667
```

### Ý nghĩa

F1 là **điểm trung bình giữa Recall và Precision** — nhưng không phải trung bình cộng, mà là **harmonic mean** (trung bình điều hòa). Đặc tính: nếu **một trong hai chỉ số rất thấp**, F1 sẽ kéo xuống thấp theo.

> *"F1 = 0.667 nghĩa là model có 'cân bằng' giữa bắt đúng bệnh và không báo nhầm. Nếu Recall rất cao nhưng Precision rất thấp (model kêu Severe cho mọi người), F1 vẫn thấp → công bằng hơn Accuracy."*

**F1 = 1.0** = perfect (hiếm), **F1 = 0** = tệ nhất, **F1 > 0.5** thường được coi là tốt.

---

## 4. AUC (Area Under ROC Curve)

### 4.1. AUC viết tắt là gì?

**AUC = Area Under the Curve** = **Diện tích dưới đường cong**.

Cụ thể là diện tích dưới đường cong **ROC** (Receiver Operating Characteristic).

> *Chú thích: tên "ROC" có nguồn gốc từ Thế chiến 2 — radar operator curve, dùng để đánh giá khả năng phân biệt máy bay địch (positive) và nhiễu (negative). Sau này được áp dụng vào medical diagnosis.*

### 4.2. Khác với 3 chỉ số trên ở chỗ nào?

3 chỉ số trên (Recall, Precision, F1) đều cần model phải **quyết định cứng**: ca này là Severe hay không?

Nhưng thực tế model output **xác suất** (vd: 0.85, 0.42, 0.93...). Phải có **ngưỡng (threshold)** để biến xác suất thành quyết định:
- Threshold = 0.5: prob ≥ 0.5 → gọi Severe
- Threshold = 0.7: prob ≥ 0.7 → gọi Severe (chặt hơn)
- Threshold = 0.3: prob ≥ 0.3 → gọi Severe (lỏng hơn)

**Mỗi threshold cho một (Recall, Precision) khác nhau**. Đây là vấn đề: chọn threshold nào là đúng?

### 4.3. ROC Curve là gì?

ROC Curve = đồ thị thể hiện **đánh đổi (trade-off)** giữa 2 đại lượng khi ta thay đổi threshold:

- **Trục Y — TPR (True Positive Rate)** = Recall = tỉ lệ bắt được bệnh thật
- **Trục X — FPR (False Positive Rate)** = FP/(FP+TN) = tỉ lệ báo nhầm người khỏe

Mỗi threshold → 1 cặp (FPR, TPR) → 1 điểm trên đồ thị. Quét threshold từ 0 đến 1 → vẽ thành đường cong.

**Đồ thị ROC trông như sau** (text-based):

```
TPR (Recall)
  1.0 ┤ ┌─────────────●●●●●●●  ← model tốt: cong sát góc trên-trái
      │ │       ●●●●●          
  0.8 ┤ │   ●●●               
      │ │ ●●                  
  0.6 ┤ │●                    ← model trung bình: nằm gần đường chéo
      │●        ╱
  0.4 ┤●     ╱
      │   ╱    ← Random model (đường chéo): AUC = 0.5
  0.2 ┤ ╱
      │╱
  0.0 ┼───────────────────────→
      0.0  0.2  0.4  0.6  0.8  1.0   FPR
```

**Diện tích dưới đường cong này = AUC**.

- Đường cong càng "phình lên góc trên-trái" → AUC càng cao → model càng tốt
- Đường thẳng chéo (45°) → AUC = 0.5 → random
- Đường cong "lõm xuống dưới chéo" → AUC < 0.5 → tệ hơn random (đảo ngược)

### 4.4. Cách hiểu trực quan AUC

**Định nghĩa thay thế (rất hữu ích để defense)**:

> AUC = **xác suất một ca Severe ngẫu nhiên có điểm xác suất CAO hơn một ca Not-Severe ngẫu nhiên**.

→ AUC đo **chất lượng RANKING** — model có xếp ca Severe lên trên ca Not-Severe không, **bất kể threshold cụ thể nào**.

### 4.5. Ví dụ trực quan với 10 ca

Giả sử model output xác suất Severe cho 10 ca:

| Ca | Thực tế | Xác suất Severe |
|---|---|---|
| #1 | Severe | 0.95 |
| #2 | Severe | 0.88 |
| #3 | Not | 0.72 |
| #4 | Severe | 0.65 |
| #5 | Not | 0.45 |
| #6 | Not | 0.30 |
| #7 | Severe | 0.25 |
| #8 | Not | 0.20 |
| #9 | Not | 0.15 |
| #10 | Not | 0.05 |

**Cách tính AUC bằng định nghĩa "ranking":**

Có 4 ca Severe (#1, #2, #4, #7) và 6 ca Not-Severe (#3, #5, #6, #8, #9, #10).
→ Tổng số cặp (Severe, Not-Severe) = **4 × 6 = 24 cặp**.

Đếm số cặp model đúng (Severe có điểm > Not-Severe):

| Cặp Severe / Not | Severe điểm > Not? |
|---|---|
| #1 (0.95) vs #3 (0.72) | ✅ |
| #1 (0.95) vs #5 (0.45) | ✅ |
| #1 (0.95) vs #6 (0.30) | ✅ |
| ... (tất cả #1 đều thắng) | ✅✅✅✅✅✅ (6/6) |
| #2 (0.88) vs các Not | ✅✅✅✅✅✅ (6/6) |
| #4 (0.65) vs #3 (0.72) | ❌ (Severe thua!) |
| #4 (0.65) vs #5,#6,#8,#9,#10 | ✅✅✅✅✅ (5/6) |
| #7 (0.25) vs #3 (0.72) | ❌ |
| #7 (0.25) vs #5 (0.45) | ❌ |
| #7 (0.25) vs #6 (0.30) | ❌ |
| #7 (0.25) vs #8,#9,#10 | ✅✅✅ (3/6) |

Tổng số cặp đúng = 6 + 6 + 5 + 3 = **20**
**AUC = 20/24 = 0.833**

→ Model xếp đúng 20/24 cặp = 83.3% → khá tốt.

### 4.6. Ý nghĩa số

| AUC | Ý nghĩa |
|---|---|
| 1.0 | Perfect — tất cả ca Severe đều xếp trên ca Not-Severe |
| 0.9 | Rất tốt |
| 0.8 | Khá (model của bạn = 0.896 ⭐) |
| 0.7 | Tạm ổn |
| 0.5 | Random — như tung đồng xu |
| < 0.5 | Tệ hơn random (đảo ngược) |

### 4.7. Vì sao AUC quan trọng

1. **Không phụ thuộc threshold**: Trong y khoa, threshold tối ưu thay đổi tùy bệnh viện, tùy chính sách. AUC đo "chất lượng tổng thể" của model.

2. **Phản ánh khả năng phân biệt**: Model tốt = ca bệnh có điểm CAO, ca không bệnh có điểm THẤP — phân biệt rõ.

3. **So sánh giữa model**: Có thể so 2 model A vs B mà không cần fix cùng threshold.

### 4.8. Defense câu hỏi "AUC nói gì?"

> *"AUC = 0.896 nghĩa là khi em chọn ngẫu nhiên 1 bệnh nhân Severe và 1 bệnh nhân không-Severe, có **89.6% khả năng model gán xác suất Severe cao hơn cho người thật sự Severe**. Đây là chỉ số đo chất lượng RANKING, không phụ thuộc vào việc em chọn threshold cụ thể nào để cắt — nên có giá trị tổng quát hơn F1."*

---

## 5. AUPRC (Area Under Precision-Recall Curve)

### 5.1. AUPRC viết tắt là gì?

**AUPRC = Area Under the Precision-Recall Curve** = **Diện tích dưới đường cong PR**.

Cũng có tên khác: **AP (Average Precision)** — về mặt số liệu thường tương đương AUPRC trong sklearn.

### 5.2. Vì sao cần thêm AUPRC khi đã có AUC?

**Vấn đề của AUC**: trên dataset **mất cân bằng nghiêm trọng** (rare class), AUC có thể **đánh lừa người đọc**.

**Ví dụ minh họa cụ thể**:
- 1000 ca, chỉ **50 ca Severe (5%)**, 950 ca Not-Severe
- Model "lười": predict Not-Severe cho mọi ca với confidence cao
- Nhưng vẫn có ranking nhỏ — Severe được điểm slightly cao hơn Not-Severe
- → Tính ra **AUC vẫn có thể = 0.85** (nghe rất tốt!)
- Nhưng thực tế Recall@threshold=0.5 = 0% (không bắt được ca Severe nào)

→ AUC bị **lừa bởi 950 ca Not-Severe đông đảo** mà model rank đúng.

### 5.3. AUPRC fix điều này

AUPRC chỉ quan tâm **Severe class** (positive class):
- Vẽ đường cong **Precision (trục Y) vs Recall (trục X)** ở mọi threshold
- Tính diện tích dưới đường cong đó

**Quan trọng**: trong công thức Precision và Recall **KHÔNG CÓ TN** (True Negative).
- Recall = TP / (TP + FN) — chỉ liên quan ca Severe thật
- Precision = TP / (TP + FP) — chỉ liên quan dự đoán Severe

→ Số lượng ca Not-Severe (TN) không ảnh hưởng → AUPRC **không bị đè** bởi negative class đông đảo.

### 5.4. PR Curve trông như thế nào?

```
Precision
  1.0 ┤●●●●●●●●●●●●○            ← Model tốt: cong sát góc trên-phải
      │           ●○             (Precision cao kể cả khi Recall cao)
  0.8 ┤            ●○
      │             ●○
  0.6 ┤              ●○         
      │               ●○        
  0.4 ┤                ●─○○○○   ← AUPRC = diện tích dưới đường cong
      │                   ●○○
  0.2 ┤- - - - - - - - - - - -  ← Baseline = % Severe trong data (vd 5%)
      │                       ●●●  
  0.0 ┼───────────────────────────→
      0.0  0.2  0.4  0.6  0.8  1.0   Recall
```

**Đặc điểm khác ROC**:
- Đường ngang ở dưới = **baseline = tỷ lệ class positive** trong data, KHÔNG phải 0.5 như ROC
- Nếu Severe chiếm 5% → baseline AUPRC random ≈ 0.05
- Nếu Severe chiếm 50% → baseline AUPRC random ≈ 0.50

### 5.5. Cách tính AUPRC từng bước

Lấy lại bảng 10 ca trên (4 Severe, 6 Not-Severe). Sắp xếp theo xác suất giảm dần. Đi từ trên xuống, dần dần nới threshold (lấy thêm ca):

| Lấy top-K | Cụ thể | Số TP | Số FP | Số FN | Recall | Precision |
|---|---|---|---|---|---|---|
| Top 1 | #1 (Severe) | 1 | 0 | 3 | 1/4 = 0.25 | 1/1 = 1.00 |
| Top 2 | + #2 (Severe) | 2 | 0 | 2 | 2/4 = 0.50 | 2/2 = 1.00 |
| Top 3 | + #3 (Not) | 2 | 1 | 2 | 2/4 = 0.50 | 2/3 = 0.67 |
| Top 4 | + #4 (Severe) | 3 | 1 | 1 | 3/4 = 0.75 | 3/4 = 0.75 |
| Top 5 | + #5 (Not) | 3 | 2 | 1 | 3/4 = 0.75 | 3/5 = 0.60 |
| Top 6 | + #6 (Not) | 3 | 3 | 1 | 3/4 = 0.75 | 3/6 = 0.50 |
| Top 7 | + #7 (Severe) | 4 | 3 | 0 | 4/4 = 1.00 | 4/7 = 0.57 |

Vẽ các điểm `(Recall, Precision)` lên đồ thị → đường cong → tính diện tích → **AUPRC**.

(Tính diện tích bằng phép trapezoidal hoặc rectangle approximation — sklearn tự lo.)

### 5.6. Ý nghĩa số

| AUPRC | Ý nghĩa |
|---|---|
| 1.0 | Perfect |
| > 0.5 | Tốt cho rare class (nếu rare class < 50%) |
| ~ baseline = tỷ lệ positive | Random level |
| < baseline | Tệ hơn random — model **gán điểm sai chiều** |

**LƯU Ý CỰC KỲ QUAN TRỌNG**: AUPRC **baseline phụ thuộc class prevalence**.

Cùng AUPRC = 0.30 có thể mang ý nghĩa rất khác:
- Nếu Severe chiếm 50% → AUPRC = 0.30 → **TỆ** (dưới baseline 0.50)
- Nếu Severe chiếm 5% → AUPRC = 0.30 → **TỐT** (gấp 6 lần baseline 0.05)

→ Khi report AUPRC, **phải kèm class prevalence** để đối chiếu.

### 5.7. So sánh AUC vs AUPRC

| | AUC (ROC) | AUPRC (PR) |
|---|---|---|
| Quan tâm cả 2 class? | ✅ Có (TN matter) | ❌ Không (chỉ positive) |
| Có dùng TN không? | ✅ Có | ❌ Không |
| Bị lừa bởi imbalance? | ⚠️ Có | ✅ Robust |
| Baseline random | 0.5 (cố định) | = % positive class |
| Khi nào dùng? | Class cân bằng | Rare class quan trọng |
| Phù hợp y khoa rare disease? | ⚠️ Lừa | ✅ Tốt |

### 5.8. Defense câu hỏi "AUPRC nói gì khác AUC?"

> *"AUPRC = 0.321 nghĩa là model có **chất lượng ranking tốt cho riêng class Severe** — class hiếm và quan trọng nhất lâm sàng. AUC có thể bị lừa bởi class đa số (Normal/Mild chiếm 85% data) — model dự đoán đúng class đa số nhiều thì AUC tự cao, nhưng AUPRC thì không bị ảnh hưởng vì công thức không có TN. Trong y khoa khi rare class quan trọng, AUPRC thường được prefer hơn AUC."*

> *"Cụ thể: Severe trong RSNA chỉ chiếm ~5% → baseline random AUPRC ≈ 0.05. Em đạt 0.321 → tức là **gấp ~6.5 lần baseline random** → đây là gain thực chất, không phải nhờ class imbalance."*

### 5.9. Vì sao paper y khoa thường report cả AUC và AUPRC

1. **AUC** = "Model phân biệt được Severe vs Not-Severe ở mức tổng thể không?"
2. **AUPRC** = "Khi tập trung riêng vào class Severe, ranking có còn tốt không?"

→ Hai chỉ số bổ sung nhau. Đặc biệt với task **screening rare disease**, AUPRC là chỉ số **chính xác hơn** để đánh giá model có hữu ích lâm sàng hay không.

---

## Bảng tóm tắt — Khi nào dùng cái nào

| Chỉ số | Đo gì | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **Recall** | Bắt được bao nhiêu % bệnh thật | Quan trọng nhất lâm sàng (đừng bỏ sót bệnh) | Có thể tăng giả tạo bằng cách "kêu loạn" |
| **Precision** | Bao nhiêu % lời cảnh báo là đúng | Tránh báo động giả | Có thể tăng giả tạo bằng cách quá thận trọng |
| **F1** | Cân bằng Recall + Precision | 1 con số tóm tắt | Cần threshold cố định |
| **AUC** | Chất lượng ranking tổng | Không phụ thuộc threshold | Bị lừa bởi class imbalance |
| **AUPRC** | Chất lượng ranking cho rare class | Robust với imbalance | Khó so sánh giữa dataset khác nhau |

---

## Câu trả lời ngắn cho thầy

> *"Em dùng 5 chỉ số bổ sung nhau:*
>
> *— **Recall** đo % ca Severe model bắt được — quan trọng nhất lâm sàng, đừng bỏ sót bệnh nhân.*
>
> *— **Precision** đo % cảnh báo Severe của model là đúng — tránh báo động giả.*
>
> *— **F1** là trung bình điều hòa của Recall và Precision — 1 con số cân bằng cả hai.*
>
> *— **AUC** đo chất lượng RANKING của model bất kể threshold — model có xếp đúng ca Severe lên trên ca không Severe không.*
>
> *— **AUPRC** đo riêng chất lượng ranking cho class Severe — robust hơn AUC khi data mất cân bằng (Severe chỉ chiếm <5% trong RSNA).*
>
> *Em dùng cả 5 vì mỗi chỉ số có blind spot riêng: Recall có thể giả tạo cao bằng kêu loạn, Precision có thể giả tạo cao bằng quá thận trọng, AUC có thể bị lừa bởi class đa số. Tổ hợp 5 chỉ số đảm bảo defense đa chiều."*

---

## Số liệu thực tế của Hybrid model trên RSNA

| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| Mean F1 macro | **0.528** | Cân bằng tốt giữa P và R cho 3 class |
| Mean AUC macro | **0.837** | Ranking tốt — model phân biệt được class |
| Mean AUPRC macro | **0.526** | Tốt cho rare class |
| **Severe F1** | **0.343** | Bắt được class quan trọng nhất |
| **Severe AUC** | **0.896** ⭐ | Ranking Severe rất tốt |
| **Severe AUPRC** | **0.321** | Gấp ~6.5x baseline (5% prevalence) |

**Điểm nhấn để defense**:
- **Severe AUC = 0.896** là số đẹp nhất để emphasize.
- **AUPRC = 0.321** nghe nhỏ nhưng so với baseline random ~0.05 → **gain 6.5 lần**.
- **F1 = 0.528** macro across 3 class → balanced, không bias về Normal.

---

## FAQ — Các câu hỏi thầy có thể hỏi sâu

### Q: "Tại sao macro F1 mà không phải micro hay weighted?"

**Macro F1** = trung bình F1 của từng class, **mỗi class trọng số như nhau**.
**Weighted F1** = trung bình có trọng số theo số lượng mẫu mỗi class.

> *"Em chọn macro F1 vì task của em **mất cân bằng nghiêm trọng** (Severe < 5%). Nếu dùng weighted F1, class Normal/Mild (85%) sẽ chiếm trọng số lớn — model cứ predict Normal cho mọi ca cũng có weighted F1 cao. Macro F1 ép model phải cân bằng giữa cả 3 class, đặc biệt class hiếm Severe."*

### Q: "ROC curve và PR curve khác nhau thế nào?"

| | ROC Curve | PR Curve |
|---|---|---|
| Trục Y | True Positive Rate (= Recall) | Precision |
| Trục X | False Positive Rate = FP/(FP+TN) | Recall |
| Nhạy với imbalance? | ❌ Không | ✅ Có |
| Diện tích = | AUC | AUPRC |

### Q: "Threshold mặc định là 0.5 đúng không?"

Cho binary classification → có. Nhưng RSNA có **3 class** (Normal/Mild, Moderate, Severe). Model output 3 xác suất cho mỗi ca, lấy class có xác suất cao nhất (`argmax`) làm prediction.

### Q: "Per-class metric có nghĩa là gì?"

Tách bài toán 3-class thành 3 bài toán binary one-vs-rest:
- **Severe vs (Normal+Moderate)**: tính Recall/Precision/F1/AUC/AUPRC riêng
- **Moderate vs (Normal+Severe)**: tính riêng
- **Normal vs (Moderate+Severe)**: tính riêng

Sau đó lấy macro average → tổng kết toàn task.

### Q: "Sao không dùng accuracy?"

Accuracy = (TP + TN) / Total. Trên data mất cân bằng:
- 100 ca, 5 Severe, 95 Not-Severe
- Model "lười" predict Not-Severe cho mọi ca → accuracy = 95% (rất cao)
- Nhưng Severe Recall = 0% (không bắt được ca nào)
- Accuracy lừa người đọc → không phù hợp với task class-imbalanced.

---

## PHẦN B — Metric phân vùng (segmentation)

> Dùng cho bảng "Lựa chọn mô hình phân vùng" (TotalSpineSeg, SPINEPS, MedCLIP-SAMv2, SpineNetV2).
> Phân vùng = gán nhãn cho **từng voxel** (điểm ảnh 3D): thuộc vùng cột sống hay nền. Nên metric tính trên **mặt nạ dự đoán** so với **mặt nạ chuẩn (ground truth)**.

Đặt: với mỗi voxel có 4 khả năng giống classification — TP (voxel cột sống đoán đúng), FP (nền bị đoán thành cột sống), FN (cột sống bị bỏ sót), TN (nền đoán đúng).

### B1. Dice Score (Dice Similarity Coefficient, DSC)

**Công thức:**
```
Dice = 2·|A ∩ B| / (|A| + |B|) = 2·TP / (2·TP + FP + FN)
```
với A = vùng dự đoán, B = vùng ground truth.

**Ý nghĩa:** đo **độ chồng lấp (overlap)** giữa mặt nạ dự đoán và mặt nạ chuẩn. Giá trị 0 (không trùng gì) đến 1 (trùng khít hoàn toàn).

**Ví dụ:** ground truth có 1000 voxel cột sống, model đoán 1000 voxel, trùng nhau 850 voxel.
```
Dice = 2·850 / (1000 + 1000) = 1700/2000 = 0.85  (85%)
```

**Lưu ý:** Dice = F1-score tính ở mức voxel (harmonic mean của Precision và Recall voxel). Đây là metric **chính** của phân vùng y khoa.

### B2. Precision & Recall (mức voxel)

Giống công thức ở Phần A nhưng đếm trên voxel:
```
Precision = TP / (TP + FP)   → trong các voxel model gọi là "cột sống", bao nhiêu % đúng (không phình ra nền)
Recall    = TP / (TP + FN)   → trong các voxel cột sống thật, model phủ được bao nhiêu %
```

**Đối chiếu lựa chọn:** TotalSpineSeg **Precision 97.87%** rất cao = gần như không phân vùng lan ra nền (đường biên sạch) → định vị khối liên đốt sống (IVV) chính xác. Recall 56.49% thấp hơn = phủ thiếu một phần, nhưng với mục tiêu **cắt ROI chính xác** thì Precision quan trọng hơn. SpineNetV2/MedCLIP-SAMv2 Recall cao (~92%) nhưng Precision thấp (~50%) = phân vùng **thừa**, dễ trùm sang đốt sống lân cận.

### B3. ASSD (Average Symmetric Surface Distance)

**Là gì:** khoảng cách trung bình giữa **bề mặt (đường biên)** của vùng dự đoán và bề mặt của vùng ground truth, đo **hai chiều** (đối xứng) — đơn vị mm.

**Cách tính (ý tưởng):**
1. Lấy tập điểm trên bề mặt dự đoán S_pred và bề mặt chuẩn S_gt.
2. Với mỗi điểm trên S_pred, tìm khoảng cách tới điểm gần nhất trên S_gt; lấy trung bình → d(pred→gt).
3. Làm ngược lại d(gt→pred).
4. `ASSD = trung bình của cả hai chiều` (nên gọi "symmetric").

```
ASSD = ( Σ d(p, S_gt) + Σ d(q, S_pred) ) / (|S_pred| + |S_gt|)
```

**Ý nghĩa:** Dice đo độ chồng lấp **thể tích**; ASSD đo **sai lệch đường biên về mặt hình học**. Hai mặt nạ có Dice gần nhau vẫn có thể khác nhau ở độ "lệch biên".

**Chiều tốt/xấu:** ASSD **càng NHỎ càng tốt** (0 mm = biên trùng khít). Ngược chiều với Dice/Precision/Recall (càng cao càng tốt).

**Ví dụ:** ASSD = 6.85 mm nghĩa là trung bình mỗi điểm biên dự đoán lệch khỏi biên chuẩn ~6.85 mm. TotalSpineSeg đạt ASSD thấp nhất trong 4 mô hình → biên bám sát giải phẫu nhất.

### B4. Bảng tóm tắt metric phân vùng

| Chỉ số | Đo gì | Chiều tốt | Đơn vị |
|---|---|---|---|
| **Dice** | Độ chồng lấp thể tích (overlap) | càng cao | 0–1 |
| **Precision** (voxel) | % voxel đoán "cột sống" là đúng | càng cao | 0–1 |
| **Recall** (voxel) | % voxel cột sống thật được phủ | càng cao | 0–1 |
| **ASSD** | Sai lệch đường biên trung bình | **càng thấp** | mm |

### B5. Số liệu thực tế (bảng chọn mô hình phân vùng)

| Mô hình | Dice | Precision | Recall | ASSD |
|---|---|---|---|---|
| **TotalSpineSeg** (chọn) | **70.81%** | **97.87%** | 56.49% | **6.85 mm** |
| SPINEPS | 58.68% | 52.17% | 68.34% | 7.48 mm |
| MedCLIP-SAMv2 | 40.55% | 29.16% | 91.98% | 19.50 mm |
| SpineNetV2 | 67.82% | 51.76% | 91.79% | 8.04 mm |

> **Câu trả lời ngắn nếu thầy hỏi "vì sao chọn TotalSpineSeg":** *"Em ưu tiên Precision và ASSD vì mục tiêu là cắt khối liên đốt sống chính xác để đưa vào grading. TotalSpineSeg có Precision 97.87% (biên sạch, không lan sang đốt lân cận) và ASSD thấp nhất 6.85 mm (biên bám giải phẫu nhất). Các mô hình Recall cao hơn nhưng Precision chỉ ~50% sẽ phân vùng thừa, làm sai vùng ROI."*

---

## PHẦN C — Khái niệm mô hình (đọc kèm Q&A #9, #15--#17)

> Mục này giải thích các thuật ngữ trong kiến trúc Hybrid để khi đọc lại Q&A defense dễ hiểu hơn.

### C1. Embedding vector (vector nhúng / vector đặc trưng)
- **Là gì:** kết quả encoder nén một đầu vào (ảnh hoặc câu chữ) thành **một dãy số có độ dài cố định** — ở đây **512 số**, đã chuẩn hóa L2 (độ dài vector = 1).
- **Ý nghĩa:** mỗi đầu vào trở thành **một điểm trong không gian 512 chiều**; điểm gần nhau = nội dung giống nhau.
- **Trong đề tài:** ảnh → `f_img` (512-d); mỗi prompt nhãn → `t_k` (512-d). Cùng số chiều nên so sánh được.

### C2. Cosine similarity (độ giống cosine)
- **Là gì:** đo **góc giữa 2 vector**: $\cos(a,b) = \frac{a\cdot b}{\|a\|\,\|b\|}$. Với vector đã chuẩn hóa L2 thì chỉ còn tích vô hướng $a\cdot b$.
- **Đọc số:** gần **1** = cùng hướng (rất giống); **0** = vuông góc (không liên quan); âm = ngược hướng.
- **Trong đề tài:** đo `f_img` giống prompt nào nhất → nhãn đó.

### C3. Logit
- **Là gì:** **điểm số thô** của mỗi lớp trước khi đổi sang xác suất (có thể âm/dương bất kỳ).
- **Trong đề tài:** `logit_k = scale · cosine(f_img, t_k)`. Baseline SpineNetV2: `logit_k = w_k · feature` (w_k = trọng số học cứng).

### C4. Softmax
- **Là gì:** biến dãy logit thô thành **xác suất**: $\mathrm{softmax}(z)_k = e^{z_k}/\sum_j e^{z_j}$, mỗi giá trị $\in[0,1]$, **tổng = 1**.
- **Ví dụ:** logit `[0.31, 0.28, 0.45]` → xác suất `[0.30, 0.28, 0.42]`. Logit cao → xác suất cao.

### C5. Argmax
- **Là gì:** chọn **vị trí (nhãn) có giá trị lớn nhất**.
- **Ví dụ:** `[0.30, 0.28, 0.42]` → argmax = vị trí 3 = **Severe** (độ tin cậy 0.42). Đây là **nhãn dự đoán cuối cùng**.

### C6. Contrastive learning (học tương phản)
- **Là gì:** cách huấn luyện **kéo cặp khớp lại gần, đẩy cặp không khớp ra xa** trong không gian embedding.
- **BiomedCLIP:** học trên 15M cặp ảnh y khoa--chú thích; ảnh được kéo gần đúng câu mô tả của nó, đẩy xa câu của ảnh khác → **ảnh và văn bản chung một không gian** → so được bằng cosine → cho phép zero-shot.

### C7. val_loss (mất mát trên tập kiểm định) + early stopping
- **loss là gì:** **hàm mất mát** = số đo mô hình dự đoán **sai bao nhiêu** (càng nhỏ = càng khớp nhãn đúng).
- **train_loss vs val_loss:** train_loss đo trên **dữ liệu mô hình đang học**; val_loss đo trên **tập kiểm định (validation) — dữ liệu mô hình KHÔNG học trực tiếp**.
- **Tại sao dùng val_loss để check (không dùng train_loss):** train_loss gần như **luôn giảm** (mô hình "học thuộc" tập train), không cho biết tổng quát hóa. val_loss đo trên dữ liệu chưa thấy nên **phản ánh khả năng tổng quát thật** → dùng nó để biết khi nào nên dừng.
- **Đọc đường val_loss:** giảm dần → **chạm đáy** → đi ngang hoặc **tăng lại**. val_loss **THẤP hơn = mô hình tốt hơn**. Khi val_loss **ngừng giảm (đi ngang/tăng) vài epoch liên tiếp** → mô hình **hết cải thiện** (nếu tăng = bắt đầu overfit, học thuộc nhiễu).
- **Early stopping:** lấy **checkpoint tại đáy val_loss** (epoch tốt nhất), KHÔNG lấy epoch cuối → tránh overfit.
- **Trong đề tài:** Hybrid val_loss đáy **ep9** (thấp nhất, 0.139) → **hội tụ nhanh + tốt nhất**; Baseline tới ep25 vẫn còn giảm chậm (hội tụ chậm). → đây là dẫn chứng "Hybrid hội tụ nhanh hơn" (Q&A #2).

### C8. Flow tổng thể (vào gì → ra gì)
`Khối MRI 9×112×224` → **CBAM-3D** ra `f_cbam` (512-d, đặc trưng tổn thương cục bộ) **+** **BiomedCLIP image encoder** ra `f_bmc` (512-d, tri thức y khoa nền) → **concat = 1024** → **Fusion MLP** (1024→768→512, L2) ra `f_img` → **cosine** với `t_k` (text encoder mã hóa mỗi prompt nhãn) → **logit mỗi nhãn → softmax → xác suất → argmax → nhãn + độ tin cậy**.

### C9. Chuẩn hóa L2 (chữ "L2" trong "Fusion MLP 1024→768→512, L2")
- **Là gì:** chia vector cho độ dài Euclid của nó: $\hat v = v/\|v\|_2$ với $\|v\|_2=\sqrt{x_1^2+\dots+x_{512}^2}$ → vector mới có **độ dài = 1**, **giữ nguyên hướng**. Mọi `f_img` nằm trên **mặt cầu đơn vị** → chỉ còn quan trọng **hướng (góc)**, không quan trọng độ lớn.
- **Áp ở đâu (chính xác theo code):** lên **`f_img` — đầu ra cuối của Fusion MLP** (`F.normalize(image_projection(concat))`), KHÔNG normalize riêng từng nhánh trước khi ghép. Text encoder cũng xuất `t_k` đã L2.
- **Vì sao chỉ L2 ở fusion output, KHÔNG ở 2 nhánh (đừng nhầm "chiều" với "độ dài"):**
  - **Cùng 512 chiều** = điều kiện để **CONCAT** (512+512=1024) — đây là chuyện *kích thước*, KHÔNG liên quan L2.
  - **L2** = chuyện *độ dài = 1*, chỉ cần cho **vector đem so cosine**. Mà chỉ `f_img` mới so cosine với `t_k` (cả hai phía cosine phải dài 1) → nên chỉ chuẩn hóa `f_img`.
  - `f_cbam`, `f_bmc` chỉ là đặc trưng **trung gian**, không trực tiếp so với văn bản → không cần L2; để MLP (có trọng số/BatchNorm học được) tự co giãn từ giá trị thô, rồi L2 cuối ép output về độ dài 1.
  - → Vị trí L2 quyết định bởi **"vector nào đi vào cosine"**, KHÔNG phải bởi 2 nhánh cùng chiều hay không.
- **Vì sao cần (lý do CHÍNH, chắc chắn):** để **cosine = tích vô hướng**. Khi $\|f_{img}\|=\|t_k\|=1$ thì $\cos(f_{img},t_k)=f_{img}\cdot t_k$ → tính nhanh, và **khớp đúng cách BiomedCLIP/CLIP được huấn luyện** (CLIP học trên embedding đã chuẩn hóa + cosine). Không chuẩn hóa thì độ lớn vector sẽ bóp méo điểm cosine.
- **Lý do phụ (đúng mức):** đưa `f_img` về thang cố định (độ dài 1) nên `logit = scale·cosine` $\in[-scale, scale]$; cùng **temperature học được (logit_scale)** giúp **softmax/huấn luyện ổn định** (không bùng nổ logit). Đây là setup chuẩn của CLIP.
- **⚠️ KHÔNG nên nói (dễ bị phản biện bắt):** (a) "L2 cân bằng biên độ giữa 2 nhánh" — SAI vì L2 áp **sau** khi đã ghép, không normalize riêng `f_cbam`/`f_bmc`; (b) "L2 chống overfitting lớp Severe" — overclaim, mất cân bằng do **Focal loss + oversampling** xử lý, không phải L2.

> **Câu chốt:** *"L2 ép embedding ảnh về độ dài 1, nên cosine với prompt rút gọn thành tích vô hướng — đúng cách CLIP/BiomedCLIP hoạt động, tính nhanh và ổn định. Em chuẩn hóa ở đầu ra fusion để khớp không gian với vector văn bản."*

### C10. Hàm mất mát huấn luyện của đề tài: Focal Loss + Uncertainty Loss
**Dùng loss gì:** kết hợp 2 thành phần (sơ đồ ghi "Loss: Focal + Uncertainty"):

**① Focal Loss~(Lin 2017) — cho từng tác vụ phân loại**
- Công thức: $FL(p_t) = -\alpha_t (1-p_t)^{\gamma}\log(p_t)$, với $\gamma=2$, $\alpha_t$ = trọng số lớp, `ignore_index=-1` (bỏ qua nhãn thiếu).
- **Cơ chế:** thừa số $(1-p_t)^{\gamma}$ làm mẫu **dễ** ($p_t\to1$) → hệ số $\to 0$ (gần như bỏ qua); mẫu **khó/hiếm** ($p_t$ nhỏ) → hệ số $\to 1$ (giữ nguyên trọng lượng).
- **Vì sao dùng:** lớp Severe ~4% → cross-entropy thường bị lớp đa số (Normal/Mild ~85%) **lấn át**. Focal **ép mô hình tập trung vào ca khó/hiếm** + $\alpha_t$ tăng trọng số lớp hiếm → **nâng Recall lớp Severe** (mục tiêu lâm sàng).

**② Uncertainty Loss~(Kendall 2018) — gộp 3 tác vụ (canal / foraminal trái / phải)**
- Công thức: $L = \sum_i \big[\frac{1}{2\sigma_i^2} L_i + \log\sigma_i\big]$, $\sigma_i$ = **độ bất định học được** cho mỗi tác vụ.
- **Cơ chế:** tác vụ **khó** ($\sigma$ lớn) → tự **giảm trọng số**; tác vụ **dễ** ($\sigma$ nhỏ) → **tăng trọng số**; số hạng $\log\sigma_i$ chặn $\sigma\to\infty$.
- **Vì sao dùng:** 3 bệnh độ khó khác nhau → **tự cân bằng trọng số 3 tác vụ**, không phải dò tay (manual weight).

**Tóm lại:** mỗi tác vụ tính Focal (xử lý mất cân bằng *trong* lớp) → Uncertainty gộp 3 tác vụ (cân bằng *giữa* các bệnh). **`val_loss` em theo dõi (C7) chính là tổng hàm mất mát này đo trên tập kiểm định.**

> **Câu chốt:** *"Em dùng Focal Loss để mô hình tập trung vào ca Severe hiếm/khó thay vì bị lớp đa số lấn át, cộng Uncertainty Loss của Kendall để tự cân bằng trọng số giữa ba bệnh thay vì chỉnh tay. val_loss theo dõi chính là tổng loss này trên tập validation."*

---

## Tham khảo

- **Sklearn metric docs**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Dice & ASSD (segmentation metrics)**: Taha & Hanbury 2015, *BMC Medical Imaging* — "Metrics for evaluating 3D medical image segmentation"
- **AUC vs AUPRC trên imbalanced data**: Saito & Rehmsmeier 2015, *PLoS ONE*
- **F1 macro vs weighted**: Sklearn `classification_report` documentation
- **Implementation trong project**:
  - `eval_rsna_auc.py` — standalone AUC/AUPRC eval
  - `train_rsna_hybrid.py` — built-in eval mỗi epoch
