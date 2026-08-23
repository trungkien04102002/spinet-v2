# Meeting prep — Concerns cần hỏi thầy Nhan Phan

> Date: 2026-04-24  
> Mục đích: list tất cả concern về pipeline BiomedCLIP + paper scope để hỏi 1 lần trong meeting.

---

## A. Architecture concerns

### A.1. Final pipeline: chọn option nào

**3 option em đã phân tích**:

| Option | Mô tả | Zero-shot bệnh khác | Complexity |
|---|---|---|---|
| **Hybrid** | CBAM 3D + BiomedCLIP Image + BiomedCLIP Text | ✅ Mạnh nhất | Cao (3 tuần) |
| **C-Lite** | CBAM 3D + BiomedCLIP Text (bỏ BMC Image) | ⚠️ Weak cho bệnh khác hẳn | Thấp (2 tuần) |
| **BiomedCLIP only** | Bỏ CBAM, chỉ BiomedCLIP Image + Text | ✅ OK | Thấp (2 tuần) |

**Hiểu biết hiện tại của em**:
- Hybrid mạnh nhất cho zero-shot nhờ BiomedCLIP Image đã thấy bệnh khác trong 15M cặp PubMed
- C-Lite yếu zero-shot bệnh khác vì CBAM chưa thấy
- BiomedCLIP only = đơn giản nhưng mất CBAM (spine-specific, 3D)

**Hỏi thầy**:
- Thầy recommend option nào cho paper Rank C? Em nghiêng về **Hybrid** để cover cả 2 yêu cầu thầy (severity + label space), nhưng 3 tuần tight.

---

### A.2. Vấn đề 3D → 2D cho BiomedCLIP Image

**Concern**: BiomedCLIP Image là **2D ViT-B/16**, không ăn 3D volume trực tiếp.

**Em có 2 option để handle**:

**Option A — Static slice selection (đơn giản)**:
- Pick 3 slice trung tâm (slice 3, 4, 5 trong volume 9 slice)
- Encode từng slice qua BiomedCLIP 2D
- Attention pooling (trainable) để weight 3 slice features thành 1 vector

**Option B — Dynamic slice selection (borrow từ Lian et al. 2026 Scientific Reports)**:
- Encode TẤT CẢ 9 slice qua BiomedCLIP 2D
- Tính cosine similarity giữa slice feature và neighbor
- Keep top N% slice có similarity thấp nhất (= unique nhất, ít redundant)
- Paper dùng ratio 28% → với 9 slice ~ giữ 2-3 slice
- Attention pool trên slice được keep

**Diagram chung**:
```
Volume (9, 112, 224) ─┬─ CBAM 3D (full 3D) ──────────────────┐
                      │                                        │
                      └─ Slice Selection (A hoặc B)             │
                         → BiomedCLIP 2D encode                 │
                         → Attention pool                       │
                                                                │
                      Concat CBAM + BMC pooled ─────────────────┘
                      → Projection → image emb
```

**Trade-off 2 option**:

| | Option A (static) | Option B (dynamic) |
|---|---|---|
| Code complexity | Đơn giản | +20 dòng |
| Robust khi lesion lệch trung tâm | Yếu | Mạnh hơn |
| Với volume 9 slice (nhỏ) | OK | Benefit marginal |
| Cite được Lian et al. | ❌ | ✅ (methodology inspiration) |
| Hyperparam mới | Không | Cần tune ratio |

**Precedent check**:
- Lian et al. 2026 paper focus **language generation** (report + VQA), **không làm zero-shot classification**
- Paper này chỉ cite được cho **methodology 2D encoder cho 3D data** + **dynamic slice selection**
- **Không cite được cho zero-shot hoặc CBAM hybrid** (paper không có)

**Trade-off chung**: mất 1 phần 3D info ở BiomedCLIP path (cả 2 option). Mitigate: CBAM path giữ full 3D compensate. Zero-shot capability KHÔNG bị ảnh hưởng bởi slice selection (text encoder frozen), chỉ accuracy bị ảnh hưởng.

**Hỏi thầy**:
- Thầy recommend Option A hay Option B?
- Có nên ablation **cả 2** trong paper (1 row so sánh static vs dynamic) để cite Lian et al., hay chọn 1?
- Có cần test "single slice" baseline nữa không (=3 variant: 1 slice / 3 static / dynamic)?

---

### A.3. CBAM có thực sự cần không

**Thầy từng nói**: "BiomedCLIP có image encoder rồi, em double check lợi thế CBAM/ResNet, nếu cần hybrid".

**Em check lại**:

| Tiêu chí | BiomedCLIP Image (pretrain) | CBAM 3D (em train RSNA) |
|---|---|---|
| Input | 2D slice | 3D volume native |
| Domain | Medical generic (15M cặp) | Spine-specific |
| Severe recall | Chưa tuned | Đã tuned với focal loss |
| 3D spatial context | ❌ Mất | ✅ Full |

**Kết luận em**: CBAM **bổ sung** chứ không thay thế được BiomedCLIP. Hybrid dùng cả 2 là hợp lý.

**Hỏi thầy**: confirm hướng này ok.

---

### A.4. Text encoder projection (đã update diagram theo feedback thầy)

**Feedback trước của thầy**: "Bên text encoder người ta vẫn dùng projection head, sao hình em chỉ có projection cho image".

**Em check**:
- BiomedCLIP text encoder **đã built-in text projection** (là phần cuối của `model.encode_text()`, pretrain trên 15M cặp)
- Em giữ **toàn bộ text encoder + text projection frozen** — đã update trong diagram

**Em KHÔNG add trainable projection mới ở text side** vì:
- Nếu train thêm → projection overfit 9 RSNA labels → **mất zero-shot**
- Text projection internal đã đủ để đưa text vào không gian shared

**Hỏi thầy**: confirm design này ok, hay thầy muốn em add LoRA ở text side (risk mất zero-shot)?

---

## B. Training concerns

### B.1. Trainable vs Frozen — em train cái gì

| Component | Params | Status |
|---|---|---|
| CBAM 3D backbone | ~22M | **FROZEN** (load RSNA checkpoint cũ) |
| BiomedCLIP Image | ~86M | **FROZEN** (load HuggingFace) |
| BiomedCLIP Text + text projection | ~110M | **FROZEN** (load HuggingFace) |
| **Image Projection (MLP 1024→512)** | **~500K** | **TRAIN** |
| **Attention pool weights** | **~1K** | **TRAIN** |
| **Logit_scale (temperature)** | **1** | **TRAIN** |

→ Chỉ train **~500K params (0.3%)**. Tất cả encoder frozen.

**Hỏi thầy**: confirm approach parameter-efficient này ok, hay cần LoRA fine-tune encoder?

---

### B.2. Training paradigm — supervised vs contrastive

**BiomedCLIP gốc**: contrastive InfoNCE trên 15M cặp, batch 32K.

**Pipeline em**: supervised CrossEntropy + Focal Loss trên 20K RSNA, batch 16-32.

**Em không làm full contrastive được** vì:
- Không có 15M cặp spine
- GPU 4090/5090 không đủ batch 32K

**Nhưng em có thể làm supervised contrastive** (Khosla 2020) auxiliary:
```python
loss = FocalLoss(logits, labels) + 0.3 * SupConLoss(image_embs, labels)
```

**Hỏi thầy**:
- OK dùng FocalLoss thay CrossEntropy (giữ công severity recovery)?
- Có nên add SupCon auxiliary term để alignment chặt hơn?

---

### B.3. Focal Loss + Augmentation + Oversampling

Công severity recovery hiện tại của em gồm:
- Focal Loss (thay CE)
- Heavy augmentation
- Oversampling Severe class
- Uncertainty Loss multi-task weighting

**Em giữ toàn bộ** trong pipeline mới. Không bỏ đi.

**Hỏi thầy**: ok giữ?

---

## C. Paper scope concerns

### C.1. 1 paper hay 2 paper

**Paper hiện tại có 2 contribution**:
1. CBAM + Focal → severity recovery
2. BiomedCLIP integration → zero-shot label extension

**Concern**: 2 contribution là 2 problem khác nhau → có nên gộp 1 paper?

**Em phân tích**:
- **1 paper (gộp)**: Rank C phù hợp, 2 contribution đủ, story "deployment-ready spine MRI"
- **2 paper (tách)**: 2x effort, paper 1 (CBAM) novelty thấp (đã có nhiều)

**Hỏi thầy**: gộp 1 paper có risk perceive "scattered" không?

---

### C.2. Framing: zero-shot main hay severity main

**Option A — Zero-shot main, CBAM là component**:
- Title: "Zero-Shot Spine MRI Grading via Attention-Augmented Hybrid Vision-Language Model"
- Main claim: zero-shot capability
- Severity improvement: ablation result, không phải main

**Option B — Severity main, BiomedCLIP là tool**:
- Title: "Attention-Augmented Language-Guided Grading for Imbalanced Spine MRI"
- Main claim: severity recovery
- Zero-shot: bonus section

**Em nghiêng Option A** vì:
- Novelty cao hơn (zero-shot 3D spine ít paper)
- Answer thầy label-space direct
- Severity đã có nhiều công trình

**Hỏi thầy**: chọn framing nào? Title gợi ý có ổn không?

---

### C.3. Phase 1 per-severity metrics gap

**Vấn đề**: Phase 1 baseline table hiện tại report **aggregate metric** (overall Acc/F1), **không có per-severity breakdown**.

**Nhưng improvement (CBAM+Focal)** lại focus vào **Severe class recall** → narrative không coherent.

**Fix**: Re-compute per-severity metrics từ raw predictions đã có (`medgemma.json`, `ningshen.csv`, `spinenetv2_upstream.csv`) → ~3 giờ code.

**Hỏi thầy**: fix này critical, em sẽ làm trước khi start BiomedCLIP implementation. Confirm ok?

---

## D. Experiment concerns

### D.1. SPIDER zero-shot test plan

**SPIDER có 8 disease labels** (`radiological_gradings.csv`):
- Modic (4-class), UP/LOW endplate (binary)
- Spondylolisthesis, Disc herniation, Disc narrowing, Disc bulging (binary)
- Pfirrman grade (5-class)

**3 tier zero-shot difficulty**:
- **Tier 1 (dễ)**: Disc narrowing (gần RSNA narrowing)
- **Tier 2 (medium)**: Spondylolisthesis, Disc herniation, Disc bulging
- **Tier 3 (khó)**: Pfirrman grade (5-class), Modic (4-class)

**Metric**: F1 macro, Balanced accuracy, Positive-class recall (KHÔNG dùng raw accuracy vì imbalanced)

**Hỏi thầy**:
- Test 8 disease hay 4 disease chính?
- Metric chọn thế nào?

---

### D.2. Few-shot fallback

**Nếu zero-shot accuracy thấp** (ví dụ < 50% cho Modic):

Fine-tune chỉ image projection (~500K params) với K=10 sample/class SPIDER → expected boost 15-25%.

**Hỏi thầy**:
- Few-shot là backup acceptable không?
- K=10 hay K=20 per class?

---

## E. Timeline + feasibility

### E.1. Scope timeline

| Phase | Task | Tuần |
|---|---|---|
| 0 | Fix Phase 1 per-severity metrics | Tuần 1 (3 giờ) |
| 1 | Implement Hybrid model | Tuần 1-2 |
| 2 | Train RSNA + ablation | Tuần 3 |
| 3 | Cross-dataset SPIDER zero-shot | Tuần 4 |
| 4 | Paper writing (có AI support) | Tuần 5-10 |
| Total | | **~10 tuần** |

**Target venue**: EMBC 2027 (deadline ~Jan 2027) — còn ~9 tháng → **thoải mái buffer**.

**Hỏi thầy**:
- Timeline này realistic không?
- Có nên stretch Rank B (MICCAI Workshop / conference chính)?

---

### E.2. Hardware

Em dùng RTX 4090 / 5090 (rent Vast.ai).
- Training RSNA Hybrid: ~10-15 giờ
- Inference SPIDER: ~30 phút
- Few-shot fine-tune: ~1-2 giờ

→ Đủ.

---

## F. Novelty concern

### F.1. Paper competing: "A Novel Adaptation of a MedCLIP-Based Vision-Language Model for Imbalanced Multi-Label Datasets"

**Em tìm được paper này** (MedCLIP + imbalanced multi-label).

**Differentiation**:
- Paper kia: 2D, multi-label binary (CheXpert-style), không zero-shot cross-dataset
- **Em**: 3D, multi-label **multi-class ordinal** (severity scale), **cross-dataset zero-shot**

→ **Setup khác hẳn**, novelty không bị trùng core.

**Hỏi thầy**: có cần đọc kỹ paper này làm baseline comparison không?

---

## G. Status architecture files

Em đã update 3 file diagram theo feedback thầy:

| File | Update |
|---|---|
| `new_architecture.md` | Đã có hybrid + text projection internal |
| `HIGH_LEVEL_ARCHITECTURE.md` | Text encoder node + bảng giải thích rõ text projection |
| `OPTION_B_vs_C_ARCHITECTURE.md` | Text encoder node + frozen section rõ text projection |

→ Sẵn sàng show thầy.

---

## Summary questions để hỏi thầy (priority order)

### Must-ask (quyết định scope)

1. **Option A (zero-shot main) vs Option B (severity main)** cho paper framing?
2. **1 paper gộp** OK, hay thầy muốn split 2 paper?
3. **Pipeline final**: Hybrid hay C-Lite?
4. Phase 1 per-severity re-compute trước khi start BiomedCLIP — confirm?

### Technical (xác nhận design)

5. Slice selection 3 center + attention pool cho BiomedCLIP 2D — ổn?
6. Text encoder frozen (không train projection mới) — confirm?
7. Supervised contrastive auxiliary loss — có nên add?
8. Test SPIDER 8 disease hay 4 chính?

### Nice-to-ask (optional)

9. Timeline 10 tuần + target EMBC 2027 — realistic?
10. Paper MedCLIP competitor — cần làm baseline comparison không?
11. Stretch Rank B có realistic không?

---

## After meeting

- [ ] Document thầy's answers
- [ ] Update plan file nếu thầy change scope
- [ ] Start fix Phase 1 per-severity metrics
- [ ] Start Hybrid implementation theo direction thầy chọn
