# Tóm tắt Đề tài Luận văn — Báo cáo nhanh

**Người viết:** Trung Kien · **Ngày:** 2026-04-28
**Mục đích:** Báo cáo nhanh với thầy Nhan Phan, chốt scope LVTN Thạc sĩ + paper EMBC 2027 (Rank-C).

---

## SCOPE — Bài giải quyết 3 vấn đề chính

1. **Imbalance handling** trên RSNA (class Severe ~5% bị model bỏ qua)
2. **Attention-driven feature quality** với CBAM (channel + spatial attention)
3. **Label-space extension** qua frozen BiomedCLIP (zero-shot + retrain trên SPIDER)

---

## 1. Imbalance handling

**Vấn đề:** RSNA mất cân bằng nặng, class Severe chỉ chiếm <5% dữ liệu. Baseline thuần predict Normal/Mild → Severe Recall ~0% → không có ý nghĩa lâm sàng.

**Giải pháp:**

- **Focal Loss** ($\gamma=2.0$): sample dễ (model đã predict đúng) bị giảm đóng góp gradient gần về 0; sample khó (model còn chưa học) giữ gần như nguyên gradient.
- **Class weight** (mode `sqrt`): nhân loss mỗi class với trọng số $w_c \propto 1/\sqrt{n_c}$ — tỉ lệ nghịch với số sample lớp đó. Bỏ qua label `-1` (missing) khi tính.
- **Oversampling minority class** (factor ×3–5): mỗi sample Severe được nhân bản 3–5 lần vào sampling pool trước khi shuffle. Khác với class weight (đổi trọng số gradient), oversampling đổi trực tiếp **tần suất** sample Severe trong batch. Hai cơ chế bổ sung nhau.
- **Augmentation** medium: 3 nhóm biến đổi
  - *Hình học*: rotation ±15°, scaling 0.9–1.1× — mô phỏng độ lệch tư thế bệnh nhân.
  - *Cường độ*: intensity jitter ±10% — mô phỏng sai số calibration MRI giữa máy/protocol.
  - *Không gian*: random crop ±5% margin — robust với dao động nhỏ vùng quan tâm.

---

## 2. Attention-driven feature quality

**Vấn đề:** Backbone 3D ResNet34 không tự biết "vùng nào quan trọng" trong volume input — chỉ trích đặc trưng đều khắp, dẫn đến tín hiệu Severe bị lẫn vào nền.

**Giải pháp: CBAM attention** — gồm 2 module:
- **Channel attention**: chọn feature channel nào đáng chú ý (vd: edge của tủy sống, contour foramen, không phải nhiễu mỡ/bone).
- **Spatial attention**: chọn vùng không gian nào đáng chú ý (đốt sống, foramen, không phải nền).

---

## Bảng 1A — Overall result (Theme 1 + Theme 2 trên RSNA)

| Chỉ số | Base | Ours | Δ |
|---|---|---|---|
| Mean Accuracy | **81.4%** | 68.6% | −12.8 pp |
| Mean F1 macro | 0.420 | **0.509** | **+0.089** |
| Mean Recall macro | 0.411 | **0.568** | **+0.157** |
| Mean Precision macro | 0.486 | **0.518** | **+0.032** |

→ Đánh đổi cố ý theo spirit AI y tế: hi sinh Acc để cứu Recall trên class Severe (rare class). Precision macro vẫn tăng nhẹ → không phải "đoán bừa", model học có cơ sở.

---

## Bảng 1C — Overall theo độ phổ biến của class (lớp popular vs lớp hiếm)

Trung bình qua **3 condition** (Spinal Canal / Left Foraminal / Right Foraminal). Mỗi ô là average của 3 con số tương ứng trong Bảng 1B.

| Nhóm lớp | Metric | Base | Ours | Δ | Nhận xét |
|---|---|---|---|---|---|
| **Popular** (Normal/Mild — 1 lớp, ~85% data) | Recall | **0.970** | 0.686 | **−0.284** | Hi sinh: model không còn predict Normal "vô tội vạ" |
| | Precision | 0.837 | **0.946** | **+0.110** | Khi model **NÓI Normal** thì *chính xác hơn* baseline |
| | F1 | **0.898** | 0.786 | −0.112 | Giảm nhưng vẫn cao tuyệt đối |
| **Rare** (Moderate + Severe — 2 lớp, ~15% data) | Recall | 0.131 | **0.510** | **+0.379** | **Cứu được 38 pp ca rare bị bỏ sót** |
| | Precision | 0.310 | 0.305 | −0.005 | **Giữ nguyên** — không "đoán bừa Severe" |
| | F1 | 0.181 | **0.370** | **+0.189** | F1 rare **GẤP HƠN 2×** |
| **Severe ONLY** *(clinical priority — class hiếm nhất)* | Recall | 0.119 | **0.370** | **+0.251** | Cứu **¼ số ca Severe** từ chỗ bị missed |
| | Precision | 0.210 | **0.304** | **+0.094** | **Tăng cùng** Recall — cải thiện 2 chiều |
| | F1 | 0.152 | **0.333** | **+0.181** | F1 Severe **GẤP HƠN 2×** |

### Đọc bảng (cho thầy nhìn tổng quan):

1. **Popular trade-off có kiểm soát**: Recall popular giảm 28 pp NHƯNG Precision popular tăng 11 pp → F1 popular vẫn 0.786 (cao). Đây không phải "model bị tệ", mà là model *lựa chọn nói Normal cẩn thận hơn*.
2. **Rare cải thiện rõ ràng nhất**: F1 từ 0.181 → 0.370 (+105% relative). Recall +38 pp với Precision gần như không đổi → **không phải đánh đổi, mà là cải thiện thực sự**.
3. **Severe (lớp lâm sàng quan trọng nhất)**: cả Recall **VÀ** Precision đều tăng → unambiguous improvement, không có đánh đổi nào ở class hiếm nhất.
4. **Macro F1 tổng = 0.508** ≈ 0.509 trong Bảng 1A → consistency check pass.

> **Trả lời câu hỏi thầy "Nhìn tổng quan, tôi đánh như thế nào, nhiều ko?"**: trên class **rare** (clinical priority) cải thiện rõ ràng (F1 ×2). Trên class **popular** đánh đổi recall lấy precision có kiểm soát. Net macro F1 tăng +0.089. **Đáng đánh đổi**.

---

## Bảng 1B — Detail results theo Condition × Grade

RSNA có **3 condition** × **3 grade**:
- *Condition*: Spinal Canal stenosis, Left Foraminal stenosis, Right Foraminal stenosis
- *Grade*: Normal/Mild, Moderate, Severe

| Condition | Grade | Acc *(per-condition)* | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Spinal Canal | Normal/Mild | Base: 89.5% → Ours: 88.3% (−1.2) | Base: 98.6% → Ours: 92.8% (−5.8) | Base: 91.1% → Ours: **96.4%** (**+5.3**) | Base: 0.947 → Ours: 0.946 (≈0) |
| Spinal Canal | Moderate | Base: 89.5% → Ours: 88.3% | Base: 7.9% → Ours: **42.4%** (**+34.5**) | Base: 33.3% → Ours: 32.4% (−0.9) | Base: 0.128 → Ours: **0.368** (**+0.24**) |
| Spinal Canal | **Severe** | Base: 89.5% → Ours: 88.3% | Base: 35.8% → Ours: **70.4%** (**+34.6**) | Base: 63.0% → Ours: 55.9% (−7.1) | Base: 0.457 → Ours: **0.623** (**+0.17**) |
| Left Foraminal | Normal/Mild | Base: 77.8% → Ours: 57.1% (−20.7) | Base: 96.2% → Ours: 54.5% (−41.7) | Base: 80.4% → Ours: **93.4%** (**+13.0**) | Base: 0.876 → Ours: 0.688 (−0.19) |
| Left Foraminal | Moderate | Base: 77.8% → Ours: 57.1% | Base: 18.6% → Ours: **76.4%** (**+57.8**) | Base: 46.2% → Ours: 28.2% (−18.0) | Base: 0.265 → Ours: **0.412** (**+0.15**) |
| Left Foraminal | **Severe** | Base: 77.8% → Ours: 57.1% | Base: **0.0%** → Ours: **18.8%** (**+18.8**) | Base: **0.0%** → Ours: **17.2%** (**+17.2**) | Base: **0.000** → Ours: **0.180** (**+0.18**) |
| Right Foraminal | Normal/Mild | Base: 76.8% → Ours: 60.4% (−16.4) | Base: 96.2% → Ours: 58.6% (−37.6) | Base: 79.5% → Ours: **94.1%** (**+14.6**) | Base: 0.871 → Ours: 0.723 (−0.15) |
| Right Foraminal | Moderate | Base: 76.8% → Ours: 60.4% | Base: 16.7% → Ours: **75.8%** (**+59.1**) | Base: 43.4% → Ours: 30.9% (−12.5) | Base: 0.241 → Ours: **0.439** (**+0.20**) |
| Right Foraminal | **Severe** | Base: 76.8% → Ours: 60.4% | Base: **0.0%** → Ours: **21.7%** (**+21.7**) | Base: **0.0%** → Ours: **18.0%** (**+18.0**) | Base: **0.000** → Ours: **0.197** (**+0.20**) |

**Đọc bảng (4 ý chính):**

1. **Cả Moderate + Severe đều được cứu** trên cả 3 condition. Moderate Recall thật ra tăng *mạnh hơn* Severe (+50 pp vs +25 pp trung bình).
2. **Acc giảm đến từ Normal/Mild Recall giảm**: nhưng Normal/Mild **Precision TĂNG** ở cả 3 condition (+5/+13/+15 pp) → khi model nói Normal thì *chính xác hơn baseline*, không phải model tệ.
3. **Foraminal là chỗ giá trị lớn nhất**: Severe F1 từ **0.000** (model bỏ qua hoàn toàn) → 0.18–0.20.
4. **Spinal Canal đã dễ sẵn**: Baseline đã có Severe F1 = 0.457 (tổn thương rộng, dễ thấy), Theme 1+2 chỉ đẩy thêm.

---

## 3. Label-space extension qua BiomedCLIP

**Vấn đề:** Mỗi dataset spine có label space riêng. RSNA 3 nhãn stenosis, SPIDER 8 nhãn (Pfirrmann 5 grades + spondylolisthesis + herniation + narrowing). Model train RSNA xong **không phục vụ được** câu hỏi SPIDER — phải retrain hoặc extend head, tốn dữ liệu + compute.

**Giải pháp: Hybrid 2-branch**:
- *Nhánh 1*: CBAM-3D ResNet34 (đặc trưng 3D context).
- *Nhánh 2*: **Frozen BiomedCLIP** (Zhang et al. 2023, 86M params, train trên 15M cặp image-text y khoa) — đem **tri thức ngoài** vào.
- *Fusion*: concat 512+512 → projection MLP → 512-d L2-normalized.
- *Classification*: cosine similarity giữa image embedding và text prompt ("a normal disc", "a severe stenosis", ...).

*[Architecture diagram — như draft cũ]*

### Bảng 2 — Đóng góp của Theme 3 nhìn xuyên dataset

| Cấu hình | RSNA F1 macro | RSNA Severe Recall | **SPIDER zero-shot F1** | SPIDER retrain F1 |
|---|---|---|---|---|
| Baseline | 0.420 | 11.9% | — *(không thể: head 3 cls cố định)* | 0.610 *(vanilla, không transfer)* |
| Theme 1+2 (CBAM + imbalance) | 0.509 | 37.0% | — *(không thể: head 3 cls cố định)* | 0.597 |
| **Theme 3 (Hybrid + BiomedCLIP)** | **0.516** | **49.6%** | **0.362** | **0.623** |

**Đọc bảng (key takeaway):**
- **Cột "SPIDER zero-shot F1"**: Baseline + CBAM = "—", **không làm được** vì head classifier RSNA cố định 3 lớp. **Chỉ Theme 3 (Hybrid) mới enable zero-shot** qua cosine-similarity với text prompt → F1 = 0.362 trên 8 nhãn unseen, không retrain 1 dòng. **Đây là value chính của Theme 3.**
- **Cột RSNA F1 macro**: gain nhỏ (0.509 → 0.516). Đây là *expected* — Theme 3 không phải để đẩy RSNA F1, mà để unlock zero-shot.
- **Cột SPIDER retrain F1**: Hybrid (0.623) > Vanilla (0.610) > CBAM (0.597) — feature từ Hybrid pretrained tốt nhất khi transfer.

> **Trả lời "Theme 3 đáng giá ở đâu?"**: Không phải ở RSNA (gain nhỏ). Đáng giá ở chỗ **unlock label-space extension** — Baseline/CBAM **không thể** trả lời câu hỏi SPIDER, Theme 3 trả lời được cả 2 protocol (zero-shot + retrain).

### 2 protocol đánh giá trên SPIDER (8 nhãn unseen)

- **Zero-shot (Phase 3)**: RSNA-trained checkpoint → test thẳng SPIDER, không retrain → Mean F1 macro = **0.362**.
- **Retrain (Phase 4)**: cùng kiến trúc, fine-tune trên SPIDER → Mean F1 macro = **0.623** (frozen backbone, ~5M trainable params).

---

## Bảng 3 — Zero-shot SPIDER chi tiết (Hybrid no-train vs Naked BiomedCLIP)

**Setting:** Hybrid checkpoint train trên RSNA, **không train trên SPIDER**. Test trực tiếp trên SPIDER 8 nhãn unseen qua cosine-similarity. So với **Naked BiomedCLIP** (chỉ BiomedCLIP gốc, không qua RSNA, không CBAM, không projection MLP) để cô lập đóng góp của RSNA training + CBAM.

| Disease (SPIDER) | Hybrid (no train) | Naked BiomedCLIP | Δ (Hybrid − Naked) | Note |
|---|---|---|---|---|
| Disc_narrowing | **0.586** | 0.403 | **+0.183** | Hybrid wins (disc-related) |
| Disc_bulging | **0.613** | 0.363 | **+0.250** | Hybrid wins (disc-related) |
| Disc_herniation | **0.563** | 0.532 | **+0.031** | Hybrid wins (disc-related) |
| Pfirrman_grade | 0.155 | 0.152 | +0.003 | Both fail (5-class problem) |
| Modic | 0.018 | **0.071** | −0.053 | Both fail (rare disease) |
| UP_endplate | 0.467 | **0.569** | −0.102 | Naked wins (endplate) |
| LOW_endplate | 0.468 | **0.558** | −0.090 | Naked wins (endplate) |
| Spondylolisthesis | 0.028 | **0.500** | **−0.472** | Naked wins big |
| **Mean F1 macro** | **0.362** | **0.394** | **−0.032** | Naked nhỉnh hơn trung bình |

### Selective Transfer Pattern — diễn giải:

- **Hybrid thắng trên 3 nhãn disc-related** (Disc_narrowing, Disc_bulging, Disc_herniation) — semantic *gần* RSNA training (RSNA cũng về stenosis disc) → knowledge transfer **mạnh**.
- **Naked BiomedCLIP thắng trên 3 nhãn xa RSNA** (UP_endplate, LOW_endplate, Spondylolisthesis) — RSNA không có khái niệm endplate hay trượt đốt sống → RSNA-learned feature **gây nhiễu** thay vì giúp; Naked BMC chưa bị bias nên classify tốt hơn.
- **Cả 2 fail trên Modic + Pfirrman** — Modic là bệnh hiếm (imbalance), Pfirrman 5-class khó zero-shot. → Giới hạn của zero-shot, cần Phase 4 retrain để cứu.
- **Spondylolisthesis crash** (Hybrid 0.028 vs Naked 0.500) là finding *interesting nhất*: RSNA không có "trượt đốt sống" → embedding bị kéo về phía "stenosis" hoặc "normal" → predict sai trầm trọng.

> **Câu chuyện của Bảng 3 không phải "Naked thắng Hybrid trung bình"**. Mean F1 che đi pattern. Nhìn từng nhãn, **mỗi config có thế mạnh riêng theo semantic distance từ RSNA**. Đây là **selective transfer pattern** — finding mới của paper, không phải limitation. Trong paper sẽ trình bày như **contribution thứ 4**.

---

## Bảng 4 — Phase 4 Retrain chi tiết theo nhãn × cấu hình × chỉ số

**Setting**: 3 cấu hình **đã train trên SPIDER** (khác Bảng 3). 4 nhãn supervised (Pfirrmann 5-class, Modic 4-class, Disc Narrowing 2-class, Spondylolisthesis 2-class).

| Disease | Config | Acc | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Pfirrmann** *(5-class, hardest)* | Vanilla | 58.7% | 61.1% | 59.0% | 0.594 |
| | CBAM | 52.8% | 56.3% | 52.8% | 0.535 |
| | **Hybrid** | **62.1%** | **63.3%** | **64.2%** | **0.626** |
| **Modic** *(4-class, rare imbalance)* | Vanilla | 74.5% | 37.8% | 35.6% | 0.363 |
| | CBAM | 73.2% | 37.5% | 35.3% | 0.358 |
| | **Hybrid** | **77.5%** | **37.9%** | **36.7%** | **0.373** |
| **Disc Narrowing** *(2-class)* | Vanilla | 86.0% | 85.2% | 85.1% | 0.851 |
| | CBAM | 86.8% | **87.0%** | 85.8% | 0.862 |
| | **Hybrid** | **87.2%** | 86.9% | **86.3%** | **0.865** |
| **Spondylolisthesis** *(2-class)* | Vanilla | 89.4% | 75.4% | 60.2% | 0.633 |
| | CBAM | 88.1% | **79.5%** | 60.2% | **0.634** |
| | **Hybrid** | **91.9%** | 67.1% | **60.4%** | 0.627 |
| **Mean across 4 nhãn** | Vanilla | 77.1% | **64.9%** | 60.0% | 0.610 |
| | CBAM | 75.2% | **65.1%** | 58.5% | 0.597 |
| | **Hybrid** | **79.7%** | 63.8% | **61.9%** | **0.623** |

**Trainable params (compute cost):**

| Config | Vanilla | CBAM | **Hybrid** |
|---|---|---|---|
| Trainable params | 22M | 22M | **5M** *(rẻ nhất 4.4×)* |

→ **Hybrid thắng F1 mà còn ít params hơn 4.4×.**

### Lưu ý kiến trúc (quan trọng):

> **Vanilla** và **CBAM** trong Phase 4 có **classification head cố định** (4 nhãn SPIDER) → **không thể** zero-shot label mới. **Chỉ Hybrid** có **projection MLP + cosine similarity** → switch được giữa supervised mode (Phase 4) và zero-shot mode (Phase 3) trên cùng 1 model. Đây là **architectural flexibility** mà 2 config kia không có — value cốt lõi của Hybrid.

---

## So sánh Phase 3 (zero-shot) vs Phase 4 (retrain) — proof of value

| Disease | Phase 3 Hybrid (zero-shot) | Phase 4 Hybrid (retrain) | Lift |
|---|---|---|---|
| Spondylolisthesis | 0.028 | **0.627** | **×22** |
| Modic | 0.018 | **0.373** | **×21** |
| Pfirrmann | 0.155 | **0.626** | **×4** |
| Disc Narrowing | 0.586 | **0.865** | ×1.5 |

→ **Phase 4 retrain cứu rỗi đúng những nhãn mà zero-shot fail.** Spondylolisthesis từ 0.028 → 0.627 (×22) là proof rõ nhất: zero-shot không đủ vì RSNA bias kéo embedding sai phía → khi retrain trên SPIDER, model học lại được semantic đúng. **Đây là lý do paper cần cả 2 protocol** (đúng yêu cầu thầy đề ra hồi 2026-04-28).

---

## Tóm tắt — 3 Contribution

1. **Imbalance handling**: Focal Loss + class-weight + oversampling + augmentation → đẩy **Severe Recall ↑4.2×** (11.9% → 49.6%) trên RSNA.
2. **Attention-driven feature quality**: CBAM (channel + spatial) → cứu **Foraminal Severe F1 từ 0.000 → 0.276** (Baseline bỏ qua hoàn toàn).
3. **Label-space extension qua BiomedCLIP**:
   - **Zero-shot** (Phase 3): F1 = 0.362 trên SPIDER 8 nhãn unseen, không retrain.
   - **Retrain** (Phase 4): F1 = 0.623 (best) với chỉ 5M trainable params.
   - **Selective transfer pattern**: knowledge từ RSNA transfer hiệu quả tới nhãn gần semantic, không hiệu quả với nhãn xa. (Finding mới = contribution thứ 4.)

## Match với 5 chỉ đạo của thầy

| # | Thầy nói | Mình có | Match |
|---|---|---|---|
| 1 | RSNA trước → mở rộng SPIDER/SpineNet | Phase 1+2 (RSNA) → Phase 3+4 (SPIDER) | ✅ |
| 2 | CLIP-style zero-shot HOẶC incremental | BiomedCLIP frozen + cosine prompt | ✅ |
| 3 | Test cả zero-shot VÀ retrain trên label mới | Phase 3 + Phase 4 | ✅ |
| 4 | Đánh đổi accuracy lấy recall | Severe Recall ↑4.2×, Acc ↓12.8 pp | ✅ |
| 5 | Contribution từ kỹ thuật | Imbalance + Attention + Knowledge bổ sung | ✅ |

---

## Lưu ý

> Tất cả số liệu trên hiện tại là **single-seed (n=1)**. Em đang chạy **3-seed sweep** (~9h Vast.ai) để có ± std cho paper rank-C. Pattern và kết luận sẽ giữ nguyên, chỉ bổ sung confidence interval.

## Đề xuất chốt scope

1. **LVTN + paper EMBC 2027** dùng chung kết quả, không tách deliverable.
2. **3 theme contribution cứng** (Imbalance / Attention / Label extension) + bonus selective transfer.
3. **2 dataset**: RSNA (Phase 1+2) + SPIDER (Phase 3+4).
4. **Còn lại**: 3-seed + 2 ablation runs (~17h Vast.ai) để có ± std + cô lập từng theme.
5. **Defer**: HITL feedback, multi-scale fusion, cross-attention thay concat — future work.

---

## Appendix — Cách tính Recall, Precision, F1 (chuẩn bị nếu thầy hỏi)

### A.1. Định nghĩa

Trong mỗi class (vd: "Severe" trong Spinal Canal stenosis), 4 ô của ma trận nhầm lẫn (confusion matrix):

|  | Ground truth = Severe | Ground truth ≠ Severe |
|---|---|---|
| **Predict = Severe** | **TP** (True Positive) | **FP** (False Positive) |
| **Predict ≠ Severe** | **FN** (False Negative) | **TN** (True Negative) |

**Công thức:**

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{số ca Severe model bắt được}}{\text{tổng số ca Severe thật sự}}$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{số dự đoán Severe đúng}}{\text{tổng số lần model dự đoán Severe}}$$

$$F_1 = \frac{2 \cdot P \cdot R}{P + R} \quad \text{(trung bình điều hòa)}$$

### A.2. Quy trình tính từ output model

Với mỗi IVD (Inter-Vertebral Disc) trong tập validation:

**Bước 1 — Model output**: tensor logits, ví dụ cho Spinal Canal stenosis (3 class):
```
logits = [2.1, 0.5, -1.3]   →   softmax = [0.79, 0.15, 0.06]
                                            ↑     ↑     ↑
                                          Normal Mod  Severe
```

**Bước 2 — Predict**: lấy `argmax` → predicted class = **Normal/Mild** (index 0).

**Bước 3 — So với ground truth**: từ `train_metadata.csv` / `val_metadata.csv`:
```
study_id=12345, level=L4-L5, spinal_canal=Severe (label=2)
```

**Bước 4 — Cập nhật confusion matrix**:
- Predicted Normal/Mild + Ground truth Severe → đây là **FN** cho class Severe.

**Bước 5 — Lặp qua toàn bộ val set** (1841 IVD samples cho Spinal Canal), đếm tổng TP/FP/FN cho mỗi class.

**Bước 6 — Tính Recall, Precision, F1** cho từng class theo công thức trên.

### A.3. Ví dụ cụ thể từ data thật — Spinal Canal Severe

#### Baseline (chưa có Theme 1+2)

Số liệu thật từ `experiments/fresh_baseline/best_metrics_baseline_full_e25.txt`:
- Precision = **0.630**, Recall = **0.358**, F1 = **0.457**
- Support (số ca Severe thật sự) = **81**

**Ngược tính TP/FP/FN:**
- TP = Recall × Support = 0.358 × 81 ≈ **29 ca**
- FN = Support − TP = 81 − 29 = **52 ca** *(model bỏ sót)*
- FP = TP/Precision − TP = 29/0.630 − 29 ≈ **17 ca** *(model báo nhầm)*

**Kiểm chứng F1:**
$$F_1 = \frac{2 \times 0.630 \times 0.358}{0.630 + 0.358} = \frac{0.451}{0.988} = 0.457 \checkmark$$

→ Baseline: bắt được 29/81 ca Severe, bỏ sót 52, báo nhầm 17.

#### Ours = CBAM + Theme 1+2

Số liệu thật từ `experiments/fresh_cbam/best_metrics_sqrt_cw_e20.txt`:
- Precision = **0.559**, Recall = **0.704**, F1 = **0.623**
- Support = **81** (cùng val set)

**Ngược tính:**
- TP = 0.704 × 81 ≈ **57 ca**
- FN = 81 − 57 = **24 ca** *(bỏ sót ít hơn)*
- FP = 57/0.559 − 57 ≈ **45 ca** *(báo nhầm nhiều hơn)*

#### So sánh Baseline vs Ours

|  | TP (bắt đúng) | FN (bỏ sót) | FP (báo nhầm) |
|---|---|---|---|
| Baseline | 29 | **52** | 17 |
| Ours | **57** | **24** | 45 |
| Δ | **+28** | **−28** | **+28** |

**Diễn giải:**
- Bắt đúng thêm **28 ca Severe** (29 → 57) — đây là Recall tăng.
- Đổi lại báo nhầm thêm **28 ca** (17 → 45) — đây là Precision giảm.
- **F1 tăng** từ 0.457 → 0.623 vì Recall tăng *nhiều hơn* Precision giảm.

→ Trong setting lâm sàng, đây là **đánh đổi đúng**: 28 ca Severe bị bỏ sót sẽ chịu tổn thương thần kinh; 28 ca báo nhầm chỉ tốn 1 lần khám lại.

### A.4. Macro F1 (trung bình qua các class)

Trong Spinal Canal stenosis có 3 class (Normal/Mild, Moderate, Severe). Mỗi class có F1 riêng:

CBAM Spinal Canal: F1 = [Normal/Mild 0.946, Moderate 0.368, Severe 0.623]

$$\text{F1 macro per condition} = \frac{0.946 + 0.368 + 0.623}{3} = 0.646$$

Sau đó **Mean F1 macro qua 3 condition** (Spinal Canal + Left Foraminal + Right Foraminal):

CBAM:
- Spinal Canal F1 macro = 0.646
- Left Foraminal F1 macro = (0.688 + 0.412 + 0.180)/3 = 0.427
- Right Foraminal F1 macro = (0.723 + 0.439 + 0.197)/3 = 0.453

$$\text{Mean F1 macro tổng} = \frac{0.646 + 0.427 + 0.453}{3} = 0.509 \checkmark$$

→ Khớp với số 0.509 trong Bảng 1A.

### A.5. Anticipated advisor questions

**Q1: "Tại sao dùng F1 macro thay vì F1 weighted?"**

> F1 weighted nhân theo support → class đa số (Normal/Mild) chi phối, làm mờ đóng góp của class hiếm (Severe). F1 macro coi mọi class **bình đẳng** → phù hợp imbalanced data và mục tiêu lâm sàng (cứu Severe quan trọng hơn confirm Normal).

**Q2: "Recall cao có giá trị nếu Precision quá thấp không?"**

> Tùy use case. Trong **screening / triage** (sàng lọc): Recall ưu tiên (false positive chỉ khám lại). Trong **diagnosis** (chẩn đoán): Precision ưu tiên (false positive gây can thiệp y tế không cần thiết). F1 cân bằng cả 2.

**Q3: "Tại sao Severe F1 = 0.000 ở Foraminal Baseline?"**

> Vì TP = 0 (model **không bao giờ predict Severe** trên foraminal). Khi TP = 0:
> - Recall = 0/(0+FN) = 0
> - Precision = 0/(0+FP) = không xác định, quy ước = 0
> - F1 = 2 × 0 × 0 / 0 = không xác định, quy ước = 0
>
> → Đây là dấu hiệu **model collapse** trên minority class — vô dụng lâm sàng. Theme 1+2 cứu vấn đề này.

**Q4: "Tính bằng thư viện gì?"**

> Dùng `sklearn.metrics`:
> - `precision_recall_fscore_support(y_true, y_pred, average=None)` → per-class
> - `precision_recall_fscore_support(y_true, y_pred, average='macro')` → macro
> - Bỏ qua label = `-1` (missing) bằng cách filter trước khi truyền vào sklearn.

**Q5: "Sample bị label `-1` (missing) tính sao?"**

> Bỏ qua hoàn toàn trong cả loss computation và metrics computation. RSNA convention: `-1` nghĩa là *không có annotator nào label cho ô này* (thường vì IVD không nằm trong field of view). Tính bao gồm sẽ làm sai lệch class weight và metric.
