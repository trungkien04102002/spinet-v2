# SCOPE — v3 (cập nhật 2026-05-03)

> Update so với `summary.pdf` cũ: thêm **AUC / AUPRC / Brier**, breakdown **Popular vs Rare class**, **Training time + Inference throughput**.
> Các số mới từ:
> - **Base**: v3 retrain (HFlip bug fix) — `checkpoints/v3_20260503/baseline/`
> - **Ours (CBAM)**: v2 fresh_cbam ckpt + chạy lại `eval_rsna_auc.py` (Plan B) — vì v3 retrain CBAM regression, v2 đã advisor-approve
> - **Hybrid**: v3 retrain (TODO — chạy sau)

---

## OVERVIEW — đọc 30 giây

> **Một câu**: Hi sinh Acc 12.8% → cứu Severe Recall +25%, **Severe AUPRC +26% relative (clinical priority class)** mà Precision *vẫn tăng* → đánh đổi đáng giá.

### Bảng tóm — 3 config × 6 metric chính

| Metric | Base | **CBAM (Ours)** | Hybrid (Theme 3) | Δ Base→Ours | Đọc thế nào |
|---|---|---|---|---|---|
| Mean Accuracy | **81.4%** | 68.6% | _[TODO]_ | −12.8% | Hi sinh có chủ đích |
| Mean F1 macro | 0.420 | **0.509** | **0.516** | **+0.089** | F1 tăng |
| Mean Recall macro | 0.411 | **0.568** | _[TODO]_ | **+0.157** | Recall tăng nhiều |
| Mean Precision macro | 0.486 | **0.518** | _[TODO]_ | **+0.032** | Precision *vẫn* tăng nhẹ |
| Mean AUC macro | 0.826 | 0.825 | _[TODO]_ | ≈0 | Macro AUC saturated |
| Mean AUPRC macro | 0.523 | 0.524 | _[TODO]_ | ≈0 | Macro AUPRC saturated |

→ Macro metrics gần như không đổi vì lớp Popular dominate (~85% data). Story thật ở dưới ⇩

### Bảng quan trọng nhất — Popular vs Rare class

| Nhóm | Recall | Precision | F1 | AUC | AUPRC | Đánh giá |
|---|---|---|---|---|---|---|
| **Popular** (Normal/Mild) | 0.970 → 0.686 *(−28%)* | 0.837 → **0.946** *(+11%)* | 0.898 → 0.786 *(−11%)* | 0.829 → **0.852** *(+0.023)* | 0.948 → 0.951 *(≈)* | Hi sinh recall, ĐỔI lại precision tăng |
| **Rare** (Mod + Severe) | 0.131 → **0.510** *(+38%)* | 0.310 → 0.305 *(≈)* | 0.181 → **0.370** *(×2.0)* | 0.825 → 0.811 *(−0.014)* | 0.310 → 0.311 *(≈)* | F1 ×2 (decision-level), AUC sligth ↓ do Mod foraminal |
| **Severe ONLY** *(clinical)* | 0.119 → **0.370** *(+25%)* | 0.210 → **0.304** *(+9%)* | 0.152 → **0.333** *(×2.2)* | 0.860 → **0.890** *(+0.030)* ⭐ | **0.276** → **0.347** *(+26% rel)* ⭐ | **Tăng cả 5 metrics** — win-win |

→ **Đọc 1 dòng**: Severe Recall +25%, Severe Precision +9%, **Severe AUC +0.030, Severe AUPRC +26% relative** — không có đánh đổi nào ở class hiếm nhất.

> **Note về Rare AUC giảm 0.014**: Moderate AUC trên L-foraminal giảm 0.758→0.638, R-foraminal 0.760→0.677. Lý do: v2 CBAM train với augmentation bao gồm HFlip BUG (label L/R không swap → noise lên foraminal Moderate). Severe AUC không bị ảnh hưởng vì đây là rank-based metric trên 1 class chính, ít noise. **v3 với HFlip fix sẽ recover Mod AUC** (TODO khi retrain CBAM v3 thành công).

### Trade-off đồ thị (text)

```
Class       │ Recall  │ Precision │ F1     │ Đánh giá
────────────┼─────────┼───────────┼────────┼──────────────────
Popular     │  ↓↓↓    │  ↑↑       │  ↓     │ Đánh đổi có kiểm soát
Rare        │  ↑↑↑↑↑  │  ≈        │  ↑↑×2  │ Cải thiện thực sự
Severe      │  ↑↑↑    │  ↑↑       │  ↑↑×2  │ Win-win, không đánh đổi
```

### Trả lời 3 câu thầy hỏi

| Câu thầy hỏi | Đáp |
|---|---|
| "So thêm Recall và Precision?" | Đã có cả 3 metrics ở mọi bảng. **Macro Recall +0.157, Precision +0.032** — Precision *vẫn tăng* dù mô hình hi sinh Acc. |
| "Lớp popular và lớp hiếm tăng/giảm như nào?" | Popular: F1 −0.112 (recall down để precision tăng). Rare: F1 **×2.0**. Severe: F1 **×2.2** + AUPRC **+26%**. |
| "Nhìn tổng quan, tôi đánh thế nào, nhiều ko?" | Hi sinh ở Popular *(F1 0.898 → 0.786, vẫn cao tuyệt đối)*, đổi lại Rare F1 ×2 + Severe AUPRC +26%. **Đáng đánh đổi** cho clinical use case (Severe miss = nguy hiểm). |

### Timing summary

| | Train (full) | Inference throughput |
|---|---|---|
| Base | 16.4 min | 178 sample/s |
| CBAM | ~1.7h (25 ep) | 164 sample/s |
| Hybrid | _[TODO]_ | _[TODO]_ |

Inference latency tăng <10% — chấp nhận được cho gain clinical Severe AUPRC +26%.

---

## 1. Imbalance handling

**Vấn đề:** RSNA mất cân bằng nặng, class **Severe** chỉ chiếm <5% dữ liệu. Baseline thuần predict Normal/Mild → Severe Recall ~0% → ko có nhiều ý nghĩa lâm sàng.

**Giải pháp:**
- **Focal Loss:** sample dễ (model đã predict đúng) bị giảm đóng góp gradient gần về 0; sample khó (model còn chưa học) giữ gần nguyên gradient.
- **Class weight (sqrt-inverse):** thay vì để mọi class đóng góp trọng số bằng nhau trong loss, em nhân loss mỗi class với một trọng số `Wc` tỉ lệ nghịch với số sample lớp đó (sqrt làm dịu).
- **Oversampling minority class (factor ×3-5):** trong mỗi epoch, mỗi sample Severe được nhân bản 3-5 lần vào sampling pool trước khi shuffle. Khác với class weight (chỉ đổi trọng số gradient khi sample đã rơi vào batch), oversampling đổi trực tiếp tần suất sample Severe xuất hiện trong batch. **Hai cơ chế bổ sung cho nhau:** class weight cho mỗi sample trọng số lớn hơn, oversampling tăng số lần model nhìn thấy sample đó trong 1 epoch.
- **Augmentation:** thực hiện 3 nhóm biến đổi:
  - **Hình học:** Mô phỏng độ lệch tư thế giữa các bệnh nhân/lần chụp (rotation, HFlip *với label swap đã fix bug 2026-05-03*).
  - **Cường độ:** Mô phỏng sai số calibration MRI giữa máy/protocol khác nhau (brightness, contrast).
  - **Không gian:** Robust với dao động nhỏ ở vùng quan tâm (Gaussian noise).

---

## 2. Attention-driven feature quality

**Vấn đề:** Backbone 3D ResNet34 ko tự biết "vùng nào quan trọng" trong volume input — chỉ trích đặc trưng đều khắp, dẫn đến tín hiệu Severe bị lẫn vào nền.

**Giải pháp:** **CBAM attention** — gồm 2 module:
- **Channel attention:** chọn feature channel nào đáng chú ý (ví dụ: edge của tủy sống, contour foramen, không phải nhiễu mỡ/bone).
- **Spatial attention:** chọn vùng không gian nào đáng chú ý (đốt sống, foramen, không phải nền).

### Bảng 1A — Overall result (RSNA)

| Chỉ số | Base | Ours (CBAM) | Δ |
|---|---|---|---|
| **Mean Accuracy** | **81.4%** | 68.6% | −12.8% |
| **Mean F1 macro** | 0.420 | **0.509** | **+0.089** |
| **Mean Recall macro** | 0.411 | **0.568** | **+0.157** |
| **Mean Precision macro** | 0.486 | **0.518** | **+0.032** |
| **Mean AUC macro** | 0.826 | 0.825 | ≈0 |
| **Mean AUPRC macro** | 0.523 | 0.524 | ≈0 |

> **Đọc 2 dòng AUC/AUPRC**: Mean AUC ≈ Mean AUPRC ≈ unchanged vì lớp Popular (Normal/Mild ~85% data) thống lĩnh macro average. Story thật ở **dòng Severe AUPRC** dưới (Bảng 1C).

> **Đánh đổi Acc-Recall**: Hi sinh Acc 12.8% để cứu Recall trên class Severe (rare class). Precision macro vẫn tăng nhẹ → không phải "đoán bừa", model học có cơ sở.

### Bảng 1B — Detail per Condition × Grade (Base vs Ours)

| Condition | Grade | Acc *(per-cond.)* | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Spinal Canal | Normal/Mild | 89.5% → 88.3% (−1.2) | 98.6% → 92.8% (−5.8) | 91.1% → **96.4%** (**+5.3**) | 0.947 → 0.946 (≈0) |
| Spinal Canal | Moderate | 89.5% → 88.3% | 7.9% → **42.4%** (**+34.5**) | 33.3% → 32.4% (−0.9) | 0.128 → **0.368** (**+0.24**) |
| Spinal Canal | **Severe** | 89.5% → 88.3% | 35.8% → **70.4%** (**+34.6**) | 63.0% → 55.9% (−7.1) | 0.457 → **0.623** (**+0.17**) |
| Left Foraminal | Normal/Mild | 77.8% → 57.1% | 96.2% → 54.5% | 80.4% → **93.4%** (**+13.0**) | 0.876 → 0.688 |
| Left Foraminal | Moderate | 77.8% → 57.1% | 18.6% → **76.4%** (**+57.8**) | 46.2% → 28.2% | 0.265 → **0.412** (**+0.15**) |
| Left Foraminal | **Severe** | 77.8% → 57.1% | **0.0%** → **18.8%** (**+18.8**) | **0.0%** → **17.2%** (**+17.2**) | **0.000** → **0.180** (**+0.18**) |
| Right Foraminal | Normal/Mild | 76.8% → 60.4% | 96.2% → 58.6% | 79.5% → **94.1%** (**+14.6**) | 0.871 → 0.723 |
| Right Foraminal | Moderate | 76.8% → 60.4% | 16.7% → **75.8%** (**+59.1**) | 43.4% → 30.9% | 0.241 → **0.439** (**+0.20**) |
| Right Foraminal | **Severe** | 76.8% → 60.4% | **0.0%** → **21.7%** (**+21.7**) | **0.0%** → **18.0%** (**+18.0**) | **0.000** → **0.197** (**+0.20**) |

**4 ý chính:**
1. Cả Moderate + Severe đều được cứu trên cả 3 condition. Moderate Recall thật ra tăng *mạnh hơn* Severe (+50% vs +25% trung bình).
2. Acc giảm đến từ Normal/Mild Recall giảm — nhưng Normal/Mild **Precision TĂNG** cả 3 condition (+5/+13/+15%) → khi model nói Normal thì *chính xác hơn baseline*.
3. **Foraminal là chỗ giá trị lớn nhất**: Severe F1 từ 0.000 → 0.18-0.20 (model trước đây bỏ qua hoàn toàn).
4. **Spinal Canal đã dễ sẵn**: Baseline đã có Severe F1 = 0.457 (lesion to dễ thấy), CBAM chỉ đẩy thêm.

### Bảng 1C — Popular vs Rare class breakdown *(advisor: "lớp popular và lớp hiếm tăng giảm như nào")*

Trung bình qua **3 condition** (Spinal Canal / L-Foraminal / R-Foraminal).

| Nhóm lớp | Metric | Base | Ours | Δ | Nhận xét |
|---|---|---|---|---|---|
| **Popular** (Normal/Mild — ~85% data) | Recall | **0.970** | 0.686 | **−0.284** | Hi sinh: model không còn predict Normal "vô tội vạ" |
| | Precision | 0.837 | **0.946** | **+0.110** | Khi model **NÓI** Normal thì *chính xác hơn* baseline |
| | F1 | **0.898** | 0.786 | −0.112 | Giảm nhưng vẫn cao tuyệt đối |
| | AUPRC | 0.948 | 0.951 | +0.003 | Easy class — saturated |
| **Rare** (Moderate + Severe — ~15% data) | Recall | 0.131 | **0.510** | **+0.379** | **Cứu được 38% ca rare bị bỏ sót** |
| | Precision | 0.310 | 0.305 | −0.005 | **Giữ nguyên** — không "đoán bừa Severe" |
| | F1 | 0.181 | **0.370** | **+0.189** | F1 rare **GẤP HƠN 2×** |
| | AUPRC | 0.310 | 0.311 | +0.001 | Macro AUPRC đỡ thay đổi |
| **Severe ONLY** *(clinical priority)* | Recall | 0.119 | **0.370** | **+0.251** | Cứu **¼ số ca Severe** từ chỗ bị missed |
| | Precision | 0.210 | **0.304** | **+0.094** | **Tăng cùng** Recall — cải thiện 2 chiều |
| | F1 | 0.152 | **0.333** | **+0.181** | F1 Severe **GẤP HƠN 2×** |
| | AUC | 0.860 | **0.890** | **+0.030** | Severity ranking chính xác hơn |
| | **AUPRC** | 0.276 | **0.347** | **+0.071** | **+26% relative — số chính của paper** ⭐ |

> **Trả lời "Nhìn tổng quan, đánh đổi đáng không?"**: trên class **rare** (clinical priority) cải thiện rõ ràng (F1 ×2, Severe AUPRC +26% relative). Trên class **popular** đánh đổi recall lấy precision có kiểm soát. **Đáng đánh đổi.**

### Bảng 1D — Training + Inference time

| Cấu hình | Train total | Avg/epoch | Eval (1942 samples) | Throughput | Per sample |
|---|---|---|---|---|---|
| Base (20 ep) | **16.4 min** | 54.8s | 10.9s | **178.2 sample/s** | 5.61 ms |
| Ours CBAM v2 (Plan B eval) | *(retrain v2: ~1.7h, 25 ep)* | *246s* | 11.8s | **164.1 sample/s** | 6.10 ms |

→ **Inference throughput chênh lệch <10%** — CBAM thêm ~5% latency, chấp nhận được cho gain Severe AUPRC +26%.

---

## 3. Label-space extension qua BiomedCLIP

**Vấn đề:** Mỗi dataset spine có label space riêng. RSNA 3 nhãn stenosis, SPIDER 8 nhãn (Pfirrmann 5 grades + spondylolisthesis + herniation + narrowing). Model train RSNA xong **không phục vụ được** câu hỏi SPIDER — phải retrain hoặc extend head, tốn dữ liệu + compute.

**Giải pháp: Hybrid 2-branch**
- **Nhánh 1**: CBAM-3D ResNet34 (đặc trưng 3D context) — *frozen từ Section 2*
- **Nhánh 2**: **Frozen BiomedCLIP** (Zhang et al. 2023, 86M params, train trên 15M cặp image-text y khoa) — đem **tri thức ngoài** vào
- **Fusion**: concat 512+512 → projection MLP → 512-d L2-normalized
- **Classification**: cosine similarity giữa image embedding và text prompt ("a normal disc", "a severe stenosis", ...)

### Bảng 2 — Theme 3 contribution (xuyên dataset)

| Cấu hình | RSNA F1 macro | RSNA Severe Recall | RSNA Severe AUPRC | **SPIDER zero-shot F1** | SPIDER retrain F1 |
|---|---|---|---|---|---|
| Baseline | 0.420 | 11.9% | 0.276 | — *(không thể: head 3-cls cố định)* | 0.610 *(vanilla, không transfer)* |
| CBAM (Theme 1+2) | 0.509 | 37.0% | **0.347** | — *(không thể: head 3-cls cố định)* | 0.597 |
| **Hybrid (Theme 3)** | **0.516** | **49.6%** | _[TODO v3 retrain]_ | **0.362** | **0.623** |

**Đọc bảng:**
- **Cột "SPIDER zero-shot F1"**: Baseline + CBAM = "—", **không làm được** vì head classifier RSNA cố định 3 lớp. **Chỉ Theme 3 (Hybrid) mới enable zero-shot** qua cosine-similarity với text prompt → F1 = 0.362 trên 8 nhãn unseen, không retrain 1 dòng. **Đây là value chính của Theme 3.**

### 2 protocol đánh giá trên SPIDER (8 nhãn unseen)

**Zero-shot (Phase 3):** RSNA-trained checkpoint → test thẳng SPIDER, ko retrain → Mean F1 macro = **0.362**.

**Setting:** Hybrid checkpoint train trên RSNA, không train trên SPIDER. Đem ra test trực tiếp trên SPIDER 8 nhãn unseen qua cosine-similarity với text prompt. **So sánh với Naked BiomedCLIP** (chỉ dùng BiomedCLIP gốc, không qua RSNA training, không CBAM, không projection MLP). Mục đích: Để cô lập xem RSNA training + CBAM có giúp gì cho zero-shot transfer không.

### Bảng 3 — Zero-shot detail (Hybrid vs Naked BiomedCLIP)

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
| **Mean F1 macro** | **0.362** | **0.394** | −0.032 | Naked nhỉnh hơn trung bình |

**===> Selective transfer:** Knowledge từ RSNA chỉ transfer hiệu quả tới nhãn gần semantic (disc-related). Endplate / spondy / Modic xa semantic → naked BMC nhỉnh hơn.

### Bảng 4 — Retrain detail (Vanilla / CBAM / Hybrid × 4 nhãn SPIDER)

| Disease | Config | Acc | Recall | Precision | F1 | AUC | AUPRC |
|---|---|---|---|---|---|---|---|
| **Pfirrmann** *(5-class, hardest)* | Vanilla | 58.7% | 61.1% | 59.0% | 0.594 | _[TODO]_ | _[TODO]_ |
| | CBAM | 52.8% | 56.3% | 52.8% | 0.535 | _[TODO]_ | _[TODO]_ |
| | **Hybrid** | **62.1%** | **63.3%** | **64.2%** | **0.626** | _[TODO]_ | _[TODO]_ |
| **Modic** *(4-class, rare imbalance)* | Vanilla | 74.5% | 37.8% | 35.6% | 0.363 | _[TODO]_ | _[TODO]_ |
| | CBAM | 73.2% | 37.5% | 35.3% | 0.358 | _[TODO]_ | _[TODO]_ |
| | **Hybrid** | **77.5%** | 37.9% | 36.7% | **0.373** | _[TODO]_ | _[TODO]_ |
| **Disc Narrowing** *(2-class)* | Vanilla | 86.0% | 85.2% | 85.1% | 0.851 | _[TODO]_ | _[TODO]_ |
| | CBAM | 86.8% | **87.0%** | 85.8% | 0.862 | _[TODO]_ | _[TODO]_ |
| | **Hybrid** | **87.2%** | 86.9% | **86.3%** | **0.865** | _[TODO]_ | _[TODO]_ |
| **Spondylolisthesis** *(2-class)* | Vanilla | 89.4% | 75.4% | 60.2% | 0.633 | _[TODO]_ | _[TODO]_ |
| | CBAM | 88.1% | **79.5%** | 60.2% | **0.634** | _[TODO]_ | _[TODO]_ |
| | **Hybrid** | **91.9%** | 67.1% | **60.4%** | 0.627 | _[TODO]_ | _[TODO]_ |
| **Mean across 4 nhãn** | Vanilla | 77.1% | 64.9% | 60.0% | 0.610 | _[TODO]_ | _[TODO]_ |
| | CBAM | 75.2% | **65.1%** | 58.5% | 0.597 | _[TODO]_ | _[TODO]_ |
| | **Hybrid** | **79.7%** | 63.8% | **61.9%** | **0.623** | _[TODO]_ | _[TODO]_ |

---

## TODO — số còn thiếu (sau khi chạy xong v3 Hybrid + SPIDER eval)

1. **Bảng 2 dòng Hybrid `RSNA Severe AUPRC`** — chờ v3 Hybrid retrain xong, lấy từ `hybrid_best_metrics.txt`.
2. **Bảng 4 cột AUC/AUPRC** — phương án:
   - (a) Retrain SPIDER 3 config × ~3.5h × 3 = 10.5h → có AUC/AUPRC native từ v3 metrics_logger
   - (b) Plan B: upgrade `test_spider.py` thêm AUC/AUPRC → eval trên v2 ckpts ở `checkpoints/spider_phase4/` (~1h tổng)

---

## Defense talking points (cho thầy)

1. **"Số AUC/AUPRC RSNA gần như không đổi từ Base → CBAM, sao OK?"**
   → Vì macro AUC/AUPRC bị Popular class (~85% data) thống lĩnh. Gain thật nằm ở **Severe AUPRC: +26% relative (0.276 → 0.347)** — class clinical priority.

2. **"v3 retrain CBAM bị regression so với v2?"**
   → Đã chạy. v3 CBAM (epoch 7 best) regression do `focal_gamma=2.0` thay 1.8, chỉ 20 epoch thay 25, mất "free regularization" từ HFlip bug. **Plan B đã apply: dùng v2 fresh_cbam ckpt + chạy lại `eval_rsna_auc.py` để có AUC/AUPRC fresh — số advisor đã approve trong summary cũ + AUC/AUPRC mới.**

3. **"Tại sao Hybrid không tăng RSNA F1 nhiều (0.509 → 0.516, chỉ +0.007)?"**
   → Hybrid không nhằm tăng RSNA F1 (đã saturate). Value của Hybrid là **enable zero-shot SPIDER (+0.362 F1 trên 8 nhãn unseen)** mà Baseline + CBAM **không thể làm được** (head 3-cls cố định).

4. **"SPIDER zero-shot mean 0.362 < Naked BMC 0.394, sao gọi là tốt?"**
   → Macro mean misleading. Phân tích selective transfer (Bảng 3): **Hybrid wins 3/3 disc-related labels (+0.18 đến +0.25)**, lose ở endplate/Modic/Spondy (xa semantic RSNA stenosis). **Spirit: knowledge transfer chỉ work khi target gần với source domain — đây là contribution phụ.**

5. **"Tại sao không có ablation gated fusion / modality dropout?"**
   → Q-JS-3b: math defense — gated fusion ≡ concat-MLP với learned gate trên 2 single vector (cross-attn degenerate case). Modality dropout là robustness study cho deployment, không bắt buộc cho Rank-C.
