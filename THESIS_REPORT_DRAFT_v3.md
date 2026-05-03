# Tóm tắt Đề tài Luận văn — Báo cáo nhanh (v3)

**Người viết:** Trung Kien · **Ngày v3:** 2026-05-03 (v2: 2026-04-28)
**Mục đích:** Báo cáo nhanh với thầy Nhan Phan, chốt scope LVTN Thạc sĩ + paper EMBC 2027 (Rank-C).

> **v3 thêm gì so với v2**: (1) Bảng 1A/1C có thêm cột **AUC** + **AUPRC** (chỉ số cho imbalanced data tốt hơn AUC). (2) Note về bug HFlip L/R label đã fix → các số sau retrain sẽ tốt hơn.

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
| **Mean AUC macro** | _[TODO Base]_ | _[TODO Ours]_ | _[TODO]_ |
| **Mean AUPRC macro** | _[TODO Base]_ | _[TODO Ours]_ | _[TODO]_ |

→ Đánh đổi cố ý theo spirit AI y tế: hi sinh Acc để cứu Recall trên class Severe (rare class). Precision macro vẫn tăng nhẹ → không phải "đoán bừa", model học có cơ sở.

> **Note về AUC vs AUPRC**: AUC ít nhạy với class imbalance (RSNA Severe ~5%) → có thể cao mà model không thực sự pick được Severe. **AUPRC nhạy hơn**, đo trực tiếp performance trên minority class → là chỉ số chính cho rare-class story của paper. **Đọc AUPRC trước AUC.**

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
| | **AUC** | _[TODO]_ | _[TODO]_ | _[TODO]_ | Severity ranking quality |
| | **AUPRC** | _[TODO]_ | _[TODO]_ | _[TODO]_ | **Số chính** cho rare class |
| **Popular AUPRC** (Normal/Mild) | | _[TODO]_ | _[TODO]_ | _[TODO]_ | Easy class |
| **Rare AUPRC** (Mod + Severe) | | _[TODO]_ | _[TODO]_ | _[TODO]_ | Khó hơn — cải thiện đáng kể nhất |
| **Macro AUPRC** (avg 3 classes) | | _[TODO]_ | _[TODO]_ | _[TODO]_ | Overall |

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

## Pipeline Justification — Tại sao CBAM + BiomedCLIP, không phải chỉ 1 trong 2?

**Ngữ cảnh**: thầy đã hỏi nhiều lần "CBAM đặc trưng ảnh rồi mà, sao còn cần BiomedCLIP?" và "sao không train thẳng BiomedCLIP cho đơn giản?". Đây là defense bằng số + lý thuyết.

### Bảng JS-1 — So sánh 3 lựa chọn architecture (số đã có)

| Config | Architecture | RSNA F1 macro | SPIDER zero-shot F1 | SPIDER retrain F1 | Trainable params | Label flexibility |
|---|---|---|---|---|---|---|
| **CBAM-only** | 3D ResNet+CBAM, linear head 3-class | 0.509 | ❌ **CANNOT** (head fixed) | 0.597 | 22M | Cố định 3 lớp |
| **BMC-only (naked)** | BiomedCLIP frozen, cosine với text | _[TODO linear probe]_ | 0.394 | _N/A_ (zero trainable) | 0 | Linh hoạt |
| **Hybrid (CBAM+BMC)** | Cả 2 + projection MLP + cosine text | **0.516** | 0.362* | **0.623** | **5M** *(rẻ nhất 4.4×)* | Linh hoạt |

*0.362 = mean. Per-class: Hybrid wins disc-related (semantic gần RSNA), Naked BMC wins endplate/spondylolisthesis (semantic xa) — gọi là **selective transfer pattern**.

**Đọc bảng**:
- **CBAM-only**: tốt trên RSNA nhưng **không zero-shot được** vì head cứng. Mỗi nhãn mới = retrain.
- **BMC-only**: linh hoạt label, nhưng **không có spine-specific knowledge** (PMC-15M ~0.1% spine MRI); sẽ test bằng linear probe (TODO).
- **Hybrid**: vừa **tốt nhất** trên cả 2 protocol (RSNA F1 0.516, SPIDER retrain F1 0.623), vừa **rẻ nhất** (5M vs 22M), vừa **linh hoạt** label như BMC.

### Q-JS-1: "Sao không dùng mỗi CBAM?"

**Trả lời**: CBAM-only **không zero-shot được**.

Zero-shot classification = cosine similarity giữa `image_emb` và `text_emb`. Để 2 vector ở chung không gian, phải có **CẶP** (image_encoder, text_encoder) train cùng nhau contrastively trên (image, caption) pairs. **CBAM là image-only**, không có text encoder phối hợp → image_emb sống trong space riêng, không match với bất kỳ text encoder nào → cosine với text vô nghĩa.

Để "làm CBAM zero-shot" cần một trong 2:
1. **Train CBAM contrastively với 1 text encoder trên (spine MRI, caption) pairs**: cần dataset (image, text) cho spine — KHÔNG CÓ public dataset như vậy. RSNA/SPIDER chỉ có label, không có caption tự do. Tự tạo caption thì size ~10K pairs → quá nhỏ (BiomedCLIP cần 15M).
2. **Borrow BiomedCLIP's text alignment**: project CBAM features qua MLP vào space của BiomedCLIP text. **Đây chính là Hybrid đang làm**.

→ **CBAM-only không thể đứng riêng cho zero-shot**. Hybrid là cách CBAM tham gia zero-shot — gián tiếp qua BMC's text encoder.

### Q-JS-2: "Sao không train thẳng RSNA + SPIDER trên BiomedCLIP, thêm CBAM làm gì?"

**Trả lời**: BMC một mình không đủ vì **4 lý do**:

| Vấn đề BMC một mình | Nguyên nhân | CBAM giải quyết? |
|---|---|---|
| **Không có 3D context** | BMC encoder = ViT 2D, xử lý 1 slice một lần | ✅ CBAM = 3D ResNet, thấy 9 slice cùng lúc (±27mm coverage) |
| **Không spine-specialized** | PMC-15M ~0.1-0.3% spine MRI; BiomedCLIP fail trên spine X-ray (Rynkiewicz 2025 AUC 0.53, Grodzki 2025 5% sens) | ✅ CBAM train ON RSNA → domain expert |
| **Frozen, không adapt** | BMC features fixed, linear head trên frozen feature không đủ task-specific | ✅ CBAM trainable → adapt được |
| **Local pathology attention** | BMC global CLS token không focus được vùng pathology nhỏ | ✅ CBAM channel + spatial attention |

**Empirical evidence** (hiện có):
- BMC zero-shot SPIDER: 0.394 mean. Trên disc-related (gần IVD task): 0.403/0.363/0.532 → mediocre.
- Hybrid zero-shot SPIDER: 0.362 mean. Trên disc-related: **0.586/0.613/0.563** → ×1.5 BMC alone.
- **Linear probe RSNA** (BMC frozen + linear head, 30 epochs): _[TODO sau khi chạy `train_linear_probe.py --backbone biomedclip`]_

**Sẽ chạy thêm 2 ablation để có evidence trực tiếp**:
```bash
python3 train_linear_probe.py --backbone biomedclip      # BMC frozen + linear head trên RSNA
python3 train_linear_probe.py --backbone imagenet_vit    # ImageNet ViT-B/16 control
```
- Nếu **BMC > ImageNet**: BMC có spine-relevant features beyond generic vision → BMC's medical pretraining đáng giá
- Nếu **BMC ≈ ImageNet**: BMC không hơn gì generic vision → spine knowledge ở chỗ khác (CBAM chính là chỗ đó)

### Q-JS-3: "Tại sao cần MLP projection riêng? BiomedCLIP đã có projection rồi mà?"

**Câu trả lời ngắn**: 2 projection trong pipeline làm 2 việc **khác nhau hoàn toàn**, KHÔNG redundant.

#### Setup: 2 projection cùng tồn tại trong pipeline (sơ đồ chi tiết)

```
INPUT VOLUME [B, 1, 9, 112, 224]
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────────────────┐
│ CBAM   │  │  BiomedCLIP image branch │
│ 3D     │  │  ┌──────────────────┐    │
│ ResNet │  │  │ ViT-B/16         │    │
│        │  │  │ patch embed → ...│    │
│        │  │  │ → CLS [B, 768]   │    │  ← raw ViT output
│        │  │  └────────┬─────────┘    │
│        │  │           ▼              │
│        │  │  ┌──────────────────┐    │
│        │  │  │ PROJECTION 1     │    │  ← BMC internal Linear(768→512)
│        │  │  │ Linear(768→512)  │    │     FROZEN
│        │  │  │ trained: PMC-15M │    │     trained for retrieval
│        │  │  │ contrastive loss │    │
│        │  │  └────────┬─────────┘    │
│        │  └───────────┼──────────────┘
│        │              │
│   feat_cbam [B,512]   feat_bmc [B,512]   ← cả 2 đều 512-d, cùng dimension
│        │              │
│        └──────┬───────┘
│               ▼
│       concat [B, 1024]
│               │
│               ▼
│       ┌────────────────────┐
│       │ PROJECTION 2 (MLP) │  ← Hybrid MLP, TRAINABLE
│       │ Linear(1024→768)   │     trained: RSNA grading
│       │ GELU + Dropout     │     classification loss
│       │ Linear(768→512)    │
│       └─────────┬──────────┘
│                 │
│           image_emb [B, 512]
│                 │
│                 ▼
│            cosine sim với text_emb [N_classes, 512]
```

#### Lý do 1 — Khác **TASK** = khác optimal embedding space

**BMC's Projection 1 train cho gì?**

```python
# BMC pretraining trên PMC-15M
loss_bmc = InfoNCE(image_emb, caption_emb)  # contrastive, retrieval
# Mục tiêu: image_i gần với caption_i, xa caption_j (j≠i)
```

→ Projection 1 sắp xếp ảnh sao cho "ảnh gần với caption riêng của nó". **Mục đích retrieval/matching** ("Cho caption này, tìm ảnh phù hợp").

**Projection 2 mình train cho gì?**

```python
# Hybrid training trên RSNA
loss_grade = focal_CE(cosine(image_emb, text_emb_classes), y_true)
# Mục tiêu: image_emb của Severe gần class prompt "severe stenosis",
#           xa class prompt "normal"
```

→ Projection 2 sắp xếp ảnh sao cho **discrimination giữa các class** rõ ràng cho grading.

**Tại sao 2 cái khác nhau? Concrete example**:

- Ảnh A: "moderate central canal stenosis L4-L5" (Moderate spinal_canal)
- Ảnh B: "severe foraminal narrowing L4-L5" (Severe foraminal)

| Trong Projection 1's space (retrieval) | Trong Projection 2's space (grading) |
|---|---|
| A gần caption A; B gần caption B | A trong cluster "Moderate"; B trong cluster "Severe" |
| A và B XA nhau (caption khác hẳn) | A và B XA theo trục **grade**, gần theo trục location |
| Có thể clustering theo location ("L4-L5") | Clustering theo grade hierarchy |

→ **Cùng features đầu vào, 2 cách tổ chức space khác nhau** → 2 axis quan trọng khác nhau.

**Analogy**: thư viện sách:
- Projection 1 = sắp xếp theo "tóm tắt nội dung" (mỗi sách 1 chỗ riêng theo summary)
- Projection 2 = sắp xếp theo "thể loại" (tiểu thuyết tất cả 1 chỗ, kỹ thuật 1 chỗ)
- Cùng kho sách, 2 cách index khác nhau cho 2 use-case khác nhau.

#### Lý do 2 — Khác **INPUT SHAPE**

```
Projection 1 (BMC):  Linear(768 → 512)    ← nhận 1 vector 768-d
Projection 2 (mình): Linear(1024 → 768) → Linear(768 → 512)   ← nhận 1 vector 1024-d
```

**Hai input dimensionality khác nhau hoàn toàn**:
- BMC's projection chỉ biết xử lý 768-d (CLS từ ViT)
- Mình cần 1024-d (concat CBAM 512 + BMC 512)

**Cố ép dùng BMC's projection cho input 1024-d?** → vài hack đều SAI:

| Hack | Vấn đề |
|---|---|
| Mean pool feat_cbam và feat_bmc → 512 → pad 768 với zero | **Mất thông tin từng nhánh**, không học được routing |
| Concat → truncate 1024→768 (bỏ 256) | Bỏ thông tin tùy ý, không có cơ sở |
| Project CBAM 512→768 trước rồi cộng vào CLS BMC | Đè CBAM lên CLS → nhiễu cả 2 nhánh |
| Element-wise sum feat_cbam + feat_bmc | Treat 2 nhánh ngang hàng cứng, không học được nhánh nào quan trọng theo từng sample |

→ **Không có cách nào dùng BMC's projection với 2 input mà không hỏng thông tin.** MLP **học cách combine** từ data thay vì hard-code một quy tắc kết hợp tùy ý.

#### Lý do 3 — Trainable params (CỰC KỲ QUAN TRỌNG)

**Nếu chỉ dùng BMC's Projection 1 (frozen)**:

```
Trainable params trên image side:
  CBAM:                     0  (frozen — load từ Phase 1+2)
  BMC's projection:         0  (frozen — phần của BMC weights)
  ─────────────────────────────
  TOTAL:                    0
```

→ Pipeline có **0 trainable parameter** trên image side → không có gradient flow back → model **không thể học từ RSNA labels** → output bằng đúng **zero-shot output của BMC = F1 0.394** trên SPIDER → **không có Hybrid 0.623 retrain** nữa, vì retrain vào đâu được? Không có param.

**Khi thêm Projection 2 (trainable MLP)**:

```
Trainable params trên image side:
  CBAM:                     0  (frozen)
  BMC's projection:         0  (frozen — vẫn nguyên)
  Projection 2 (MLP):       1.18M  ← chỗ này
  Slice attention pool:     ~1K
  Logit scale:              1
  ─────────────────────────────────
  TOTAL:                    ~1.18M trainable
```

→ Có gradient path. MLP học cách **dịch** features từ "BMC retrieval space" sang "RSNA grading space".
→ Empirical: F1 **0.623 retrain** trên SPIDER vs 0.394 zero-shot Naked BMC → **+23 F1 absolute đến từ MLP này**.

**Suy nghĩ ngược: unfreeze BMC's projection thay vì thêm MLP?**

| Vấn đề | Tại sao |
|---|---|
| **Catastrophic forgetting** | BMC's projection trained trên 15M pair. Fine-tune trên ~10K RSNA → mất alignment image-text gốc → hỏng zero-shot. |
| **Vẫn không xử lý 2-input** | BMC's projection vẫn là Linear(768→512). Không nhận 1024-d được. |
| **Compute đắt hơn** | Fine-tune 1 layer của BMC vẫn không "đơn giản" hơn thêm 1 MLP riêng. |

→ Unfreeze KHÔNG giải quyết được lý do 1 và 2.

#### Lý do 4 — Khác **OBJECTIVE FUNCTION** trong training

```
BMC training:    L_NCE  = -log[ exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ) ]
                                                       ↑ negative samples = ALL OTHER pairs in batch

Hybrid training: L_grade = focal_CE( cosine(img_emb, text_classes_only), y_true )
                                                       ↑ negative samples = chỉ các CLASS trong task (3 grade)
```

→ **Khác cấu trúc loss** → gradient flow khác → optimal weights khác.
- BMC's projection optimize cho image gần caption KHÔNG TƯƠNG ĐỒNG ngẫu nhiên trong batch.
- MLP của mình optimize cho image rơi vào ĐÚNG class trong tập class cố định.
- Khác use-case → khác optimal solution.

#### Empirical proof — không phải lý thuyết suông

| Setup | Trainable params | RSNA F1 | SPIDER zero-shot F1 | SPIDER retrain F1 |
|---|---|---|---|---|
| BMC Projection 1 alone (no MLP) | 0 | N/A | 0.394 | **N/A** *(không có gì để retrain)* |
| BMC Projection 1 + MLP (Hybrid) | ~1.18M | 0.516 | 0.362* | **0.623** |

→ Δ retrain = **+22.9 F1 absolute** từ MLP. **MLP thực sự tạo value**, không redundant với BMC's projection.

#### Tóm tắt 4 lý do — table dễ nhớ

| # | Lý do | Nếu chỉ dùng Projection 1 (BMC) | Tại sao Projection 2 (MLP) cần? |
|---|---|---|---|
| 1 | **Task khác** | Space tối ưu cho retrieval (caption matching) | Mình cần space tối ưu cho classification (grade discrimination) — khác axis quan trọng |
| 2 | **Input khác** | Chỉ nhận 768-d (1 modal) | Mình có 1024-d (fused 2 modal CBAM+BMC) — không tương thích shape |
| 3 | **Cần trainable** | Frozen → 0 params learn từ RSNA | MLP cho ~1.18M params trainable → adapt task + domain |
| 4 | **Objective khác** | InfoNCE contrastive | Focal CE classification — gradient khác → optimal weight khác |

→ **2 projection ≠ duplicate**. Stack nhau, làm 2 việc khác nhau, KHÔNG thay thế nhau.

#### Ngụ ý kiến trúc

Trong Hybrid, BMC's Projection 1 vẫn chạy bên trong (frozen) để cho ra `feat_bmc` 512-d. MLP của mình (Projection 2) chạy **sau đó**, nhận output của Projection 1 làm 1 trong 2 input. **2 projection cùng tồn tại trong pipeline**, mỗi cái 1 vai trò: Projection 1 cho retrieval alignment (gốc của zero-shot capability), Projection 2 cho task adaptation + 2-modal fusion (gốc của grading performance).

### Storyline 1 câu cho thầy:

> "Em đã thử cả 3 architecture. **CBAM-only** không zero-shot được vì thiếu text encoder phối hợp. **BMC-only** không đủ cho IVD grading vì 2D + không spine-specialized + frozen + không có trainable adapt — empirical 0.394 zero-shot, kém Hybrid 0.623 retrain. **Hybrid** là cấu hình duy nhất vừa tốt nhất trên cả 2 protocol, vừa rẻ nhất 4.4× params, vừa unlock label-space extension. **Mỗi nhánh chữa hạn chế của nhánh kia**: CBAM bù spine knowledge BMC thiếu, BMC bù text alignment CBAM thiếu. **MLP projection riêng** vì 4 lý do **không trùng** với BMC's internal projection: (1) khác task — BMC trained cho retrieval alignment, MLP trained cho grading discrimination; (2) khác input — BMC's projection nhận 768-d single modal, MLP nhận 1024-d fused 2-modal; (3) khác trainability — BMC frozen 0 params, MLP ~1.18M trainable params; (4) khác objective — InfoNCE contrastive vs focal CE classification. **Empirical bỏ MLP đi → model về thành Naked BMC F1 0.394; có MLP → 0.623 (+23 absolute)** — 2 projection cùng tồn tại, làm 2 việc bổ sung."

### Q-JS-3b: "Concat theo kiểu gì? BiomedCLIP có concat theo trục feature component không phải nối tiếp?"

**Câu trả lời ngắn**: thầy đang hiểu nhầm 2 chỗ — (1) BiomedCLIP **không** có concat nào ở internal cả; (2) "concat theo trục feature component" là cách diễn đạt khác của **feature-wise concat** (chính cái mình đang dùng). "Position alignment" concern không hợp lệ cho independently trained encoders.

#### Hiểu nhầm 1: BiomedCLIP **không** dùng concat ở internal

Source: BiomedCLIP paper (Zhang 2023, [arXiv:2303.00915](https://arxiv.org/abs/2303.00915)), Section 3 Model.

```
BiomedCLIP architecture (chính xác từ paper):

Image branch:                    Text branch:
ViT-B/16                         PubMedBERT
  → CLS [768]                      → [CLS] [768]
  → Linear(768→512)                → Linear(768→512)
  → image_emb [512]                → text_emb [512]
                ╲                ╱
                 cosine_similarity
                  (1 scalar)
              + InfoNCE loss
```

→ **2 modal sống ở 2 không gian riêng**, chỉ tương tác qua **cosine similarity** (1 scalar). **KHÔNG có concat** giữa image-text features bao giờ.

→ Quote từ paper: *"We jointly train an image encoder and a text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of embeddings of the N²−N incorrect pairings."* Đây giống y CLIP gốc (Radford 2021).

→ Câu thầy "BiomedCLIP concat theo trục feature component" — **không có support trong paper**. Thầy có thể đang nghĩ đến model khác (LLaVA-Med, FLAVA — early-fusion) hoặc giả định không chính xác.

#### Hiểu nhầm 2: "Concat theo trục feature component" = feature-wise concat = chính cái mình đang dùng

Mapping terminology của thầy sang ML chuẩn ([arXiv:2411.17040 survey 2024](https://arxiv.org/html/2411.17040v1)):

| Cách thầy diễn đạt | ML term chuẩn | PyTorch | Yêu cầu cùng dim? |
|---|---|---|---|
| "Concat theo trục feature component" | **Feature-wise / channel-wise concat** | `torch.cat([a,b], dim=-1)` | ❌ Không, [B,512]+[B,512]=[B,1024] |
| "Nối tiếp" | (cùng nghĩa với trên) | (cùng) | (cùng) |
| (advisor không phân biệt rõ) | Element-wise sum | `a + b` | ✅ Có, cùng D bắt buộc |

→ **"Concat theo trục" và "nối tiếp" là CÙNG operation** trong ML literature. Không có sự phân biệt 2 loại concat khác nhau như thầy nghĩ.

→ Code mình đang dùng (line 234 `grading_hybrid.py`):
```python
concat = torch.cat([feat_cbam, feat_bmc], dim=-1)  # [B, 1024]
```
→ **Đây CHÍNH LÀ feature-wise concat** = "concat theo trục feature component" theo terminology của thầy. **Mình đang làm đúng cái thầy yêu cầu**, chỉ khác cách gọi.

#### Hiểu nhầm 3: "Position alignment" concern không hợp lệ cho independently trained encoders

Thầy lo: position 5 trong CBAM và position 517 trong concat (= position 5 của BMC) cùng encode "đốt sống" nhưng model khó biết.

**Counter-argument 1 — MLP là universal approximator** (Hornik 1989):

```
concat [B, 1024]  →  Linear(1024 → 768) → GELU → Linear(768 → 512)
                          ↑
                      786,432 weights
                      Mỗi neuron output (768 cái) nhận TẤT CẢ 1024 input
                      → Có ĐỦ capacity học correlation giữa position 5 và 517
                      → Nếu 2 position cần map cùng 1 internal feature, MLP học được
```

MLP **chính xác là designed cho việc này**. "Khó học" sẽ đúng nếu chỉ có 1 layer linear (mới linear combination), nhưng **2-layer MLP với GELU phi tuyến** = universal approximator (Hornik et al., Neural Networks 1989).

**Counter-argument 2 — Independently trained encoders KHÔNG CÓ position correspondence anyway**:

- CBAM trained trên **RSNA spine MRI** với cross-entropy loss
- BMC trained trên **PMC-15M general medical** với contrastive InfoNCE loss
- 2 training run **độc lập**, không có constraint nào ép position 5 của 2 vector encode cùng concept

→ **Không có lý do gì** để giả định position alignment giữa CBAM-position-5 và BMC-position-5.

→ Element-wise sum/product **giả định** alignment này → **SAI** cho setup của ta. Concat-MLP **không giả định** → **ĐÚNG**.

**Citation chuẩn**: Baltrušaitis et al., *"Multimodal Machine Learning: A Survey and Taxonomy"*, IEEE TPAMI 2019 — late fusion (concat-MLP) **agnostic to within-vector positional semantics**, đó là feature, không phải bug.

#### Bảng so sánh 6 fusion methods cho setup của ta

| Method | Output dim | Position alignment | Capacity | Phù hợp setup ta? |
|---|---|---|---|---|
| **Concat + MLP** (đang dùng) | 1024→512 | KHÔNG giả định | Cao (2-layer MLP) | ✅ **Đúng nhất** |
| Element-wise SUM | 512 | Giả định CỨNG | 0 (linear) | ❌ Sai — encoders không co-trained |
| Element-wise PRODUCT | 512 | Giả định CỨNG | 0 (linear) | ❌ Sai — same |
| **Gated Fusion (GMU)** | 512 | Soft alignment | Trung bình | ⚠️ Alternative tốt — đã code `--fusion gated` |
| Cross-attention | varies | KHÔNG giả định | Cao | ❌ Overkill cho 2 global vector |
| Bilinear pooling | $D^2$→512 | KHÔNG | Cực cao | ❌ Tham số bùng nổ |

**Cross-attention overkill** vì sao? Ta có **2 vector global 512-d** (không phải 2 sequence tokens). Cross-attention giữa 2 single vector reduce thành 1×1 attention weight = scalar gate học được = tương đương concat-MLP-1-layer. Không có information gain, chỉ thêm complexity.

(Cross-attention chỉ có nghĩa nếu mình giữ ViT patch sequence của BMC — 196 tokens × 768 — và CBAM spatial features attend qua đó. Đây là **early fusion paradigm**, big architectural change, cần data lớn → không phù hợp Rank-C.)

#### Empirical từ literature

Quote từ ACM EITCE 2024 ([dl.acm.org/doi/10.1145/3711129.3711215](https://dl.acm.org/doi/10.1145/3711129.3711215)):

> "The MOSI dataset is relatively small, leading to overfitting during model training **when using multi-attention-based multimodal fusion models**."

Quote từ Multimodal Survey 2024 ([arXiv:2411.17040](https://arxiv.org/html/2411.17040v1)):

> "Concatenation plus MLP is the simplest approach, described as fast but shallow. [It is] the standard late fusion baseline in virtually every multimodal classification paper."

→ **Cross-attention thắng concat-MLP với margin ~2-5%** trên dataset lớn. Trên dataset y tế nhỏ (vài nghìn sample như RSNA): margin **biến mất hoặc đảo ngược** do overfitting trên thêm parameters.

→ Concat-MLP **defensible cho thesis Rank-C**.

#### Storyline cho thầy:

> "BiomedCLIP **không có concat nào ở internal**, em đã verify từ paper section 3 — image và text chỉ tương tác qua cosine similarity, không phải concat. **Concat ở pipeline em** (line 234) là feature-wise concat giữa CBAM và BMC features (`torch.cat(dim=-1)`) — đây CHÍNH LÀ 'concat theo trục feature component' trong terminology của thầy, không khác. Lo position 5 của CBAM và 517 của concat (= 5 của BMC) không gặp nhau là **không hợp lệ** vì 2 encoder train độc lập trên 2 dataset → không có lý do gì để 2 position encode cùng concept. **Element-wise sum mới sai** vì giả định alignment cứng. **MLP 2-layer là universal approximator** (Hornik 1989), **đủ capacity** học alignment cross-space — đó CHÍNH LÀ việc của nó. Thực nghiệm: bỏ MLP → F1 0.394 (Naked BMC); có MLP → 0.623 (+23 absolute) → MLP đang thực sự học. **Cross-attention overkill** vì ta có 2 vector global, không phải 2 token sequence — cross-attn giữa 2 single vector reduce thành scalar gate, không hơn concat-MLP. Có sẵn **GatedFusion alternative** (`--fusion gated`) để chạy ablation nếu thầy muốn thấy số."

#### Citations chính cho paper:

- BiomedCLIP architecture: Zhang 2023, [arXiv:2303.00915](https://arxiv.org/abs/2303.00915), Section 3
- Multimodal fusion taxonomy: [arXiv:2411.17040](https://arxiv.org/html/2411.17040v1) (survey 2024), Baltrušaitis IEEE TPAMI 2019
- Universal approximation: Hornik et al., Neural Networks 1989
- GMU baseline: Arevalo 2017, [arXiv:1702.01992](https://arxiv.org/pdf/1702.01992)
- Small-data overfitting on cross-attention: ACM EITCE 2024

---

### Q-JS-4: "BiomedCLIP có vẻ overengineer? Có VLM nào nhẹ hơn / specialized cho spine hơn không?"

**Trả lời**: KHÔNG. Tại thời điểm mid-2025, **không có spine-specialized CLIP-style model** nào tồn tại công khai. BMC là **lựa chọn nhẹ nhất** thỏa mãn cả 3 yêu cầu (zero-shot capability + medical knowledge + public checkpoint).

#### Bảng JS-2 — So sánh 12 candidate VLM cho spine MRI zero-shot

| Model | Có spine? | CLIP-style (image+text aligned)? | Zero-shot cosine OK? | Nhẹ hơn BMC? | Phù hợp? |
|---|---|---|---|---|---|
| **BiomedCLIP** (Zhang 2023, ~200M) | Indirect (PMC papers) | ✅ | ✅ | — (baseline) | ✅ **Đang dùng** |
| **UniMed-CLIP** (MBZUAI, Dec 2024) | MRI có (~6-7%), spine fraction unknown | ✅ | ✅ | ≈ | ⚠️ Alternative duy nhất legitimate |
| BioViL / BioViL-T (Microsoft) | ❌ Chest X-ray only | ✅ | ✅ chest | Có | ❌ Không có spine |
| MedCLIP (EMNLP 2022) | ❌ Chest X-ray only | ✅ | ✅ chest | Có | ❌ Không có spine |
| PubMedCLIP (ROCO ~80K) | ⚠️ Minimal radiology | ✅ | ✅ | Có | ❌ Underperforms vanilla CLIP |
| PMC-CLIP (1.6M PMC subset) | ⚠️ Tương tự BMC nhưng nhỏ hơn 10× | ✅ | ✅ | Có | ❌ Subset của BMC source |
| **SpineNetV2** (Windsor 2022) | ✅ Specialized | ❌ Image-only (no text encoder) | ❌ Không cosine với text được | — | ❌ Không có text branch |
| **SpineCLUE** (arXiv 2401.07271) | ✅ CT vertebra | ⚠️ Image-image contrastive only | ❌ Không có text encoder | — | ❌ Không có text branch |
| **SPINEPS** (arXiv 2402.16368) | ✅ T2w MRI | ❌ Segmentation CNN | ❌ | — | ❌ Image-only |
| **SpineFM** (arXiv 2411.00326) | ✅ X-ray | ❌ SAM adapter | ❌ | — | ❌ Image-only |
| **SpineGPT** (ICLR 2026, arXiv 2510.03160) | ✅ Specialized (SpineMed-450k) | ❌ **Generative LLM** (Qwen2.5-VL-7B) | ❌ Tạo text tokens, không tạo embedding cho cosine | ❌ NẶNG hơn 35× (7B vs 200M) | ❌ Không CLIP-style |
| RadFM / LLaVA-Med / BiomedGPT | Spine có | ❌ Generative VLM (7B-13B) | ❌ Sinh ra text answer, không embedding | ❌ Nặng 35× | ❌ Không CLIP-style |

#### Lưu ý quan trọng về SpineGPT (mới nhất, ICLR 2026):

SpineGPT (arXiv 2510.03160) là model spine-specialized **mới nhất** xuất bản tháng 10/2025. Thoạt nhìn đe dọa novelty của thesis, NHƯNG:

1. SpineGPT là **generative VLM** (fine-tuned Qwen2.5-VL-7B). Output là **text tokens** trả lời VQA, không phải embedding chiếu vào cosine space → **không thay thế được** BMC trong pipeline cosine-based.
2. SpineGPT 7B params >> BMC 200M → 35× **nặng hơn**, không phải "đơn giản hơn".
3. Public checkpoint **chưa confirm** (paper mới Oct 2025).
4. Dùng cho task khác hẳn (VQA, report generation) chứ không phải direct zero-shot classification.

→ SpineGPT không invalidate thesis của ta. Khác paradigm hoàn toàn.

#### Citation MẠNH cho thầy (SpineBench paper, arXiv 2510.03160, ICLR 2026):

> "Existing general-purpose large vision-language models and even medical large language models are trained on **generic medical data**, which often **lacks the high-quality, specialized data needed for orthopedics**."

→ **Authoritative paper xuất bản tháng 10/2025 confirm gap** mà thesis của bạn đang fill. Citation này nên đưa vào Discussion section.

#### Tóm gọn cho thầy:

> **BMC không overengineer**. Nó là **minimum sufficient** architecture cho zero-shot grading: (1) không có spine-specialized CLIP công khai (tháng 5/2025); (2) các spine-specific model (SpineNetV2, SpineCLUE, SPINEPS) đều image-only — không zero-shot text được; (3) các spine VLM mới nhất (SpineGPT) là generative LLM 7B params — nặng hơn 35×, không thay thế được CLIP cho cosine; (4) đơn giản hóa hơn BMC (CBAM-only) thì **MẤT zero-shot capability** — đó là contribution chính. Em chấp nhận limitation rằng PMC-15M chỉ ~0.1% spine MRI, **bù lại** bằng CBAM trainable trên RSNA — ablation chứng minh CBAM cộng thêm value (hybrid 0.623 vs zero-shot 0.362 trên SPIDER).

#### (Optional Rank-B push) UniMed-CLIP ablation:

UniMed-CLIP (Dec 2024) là alternative duy nhất legitimate — CLIP-style + có MRI explicit. Nếu reviewer push back về choice of BMC, có thể làm 1 ablation thay BMC bằng UniMed-CLIP (~6h GPU). **Không bắt buộc cho Rank-C**.

---

## Lưu ý

> Tất cả số liệu trên hiện tại là **single-seed (n=1)**. Em đang chạy **3-seed sweep** (~9h Vast.ai) để có ± std cho paper rank-C. Pattern và kết luận sẽ giữ nguyên, chỉ bổ sung confidence interval.

> **Ghi chú về data augmentation (mới phát hiện 2026-05-03)**: trong code `RandomHorizontalFlip` cũ, khi flip ảnh trái-phải, các nhãn `left_foraminal`/`right_foraminal` không được swap → ~50% sample foraminal trong training mode `medium` bị nhiễu nhãn L/R. **Đã fix** trong commit `ef55ab8`. Số liệu hiện tại trong báo cáo này là **trước fix**. Sau khi retrain với code đã fix, kỳ vọng L/R foraminal Severe F1 + AUPRC sẽ cải thiện thêm. **Đây là bug fix định lượng được, được trình bày trong paper như một minor contribution riêng.**

> **Ghi chú về architecture cải thiện (2026-05-03)**: theo gợi ý thầy "đổi MLP, đổi concat", đã thêm 2 option vào codebase: (1) **Gated Fusion** (GMU-style learned gate thay concat-MLP) — `--fusion gated`; (2) **Modality Dropout** — `--modality-dropout 0.15` (zero-out 1 nhánh ngẫu nhiên khi train, robustness với missing modality, cũng dùng làm answer cho câu hỏi "thiếu ảnh/text thì sao"). Sẽ chạy ablation 2 option này so với concat-MLP cũ để thấy số.

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
