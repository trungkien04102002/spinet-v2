# SOTA Comparison Research Plan for Rank B Submission

**Created:** 2026-04-28
**Owner:** Trung Kien
**Purpose:** Liệt kê các SOTA candidate để so sánh với SpineNetV2 + CBAM + BiomedCLIP, chuẩn bị cho submission Rank B (MICCAI workshops, MIDL, IPMI).

**Cách dùng file này (trong session sau):**
1. Mở session Claude mới
2. Invoke `/research <method_name>` cho từng method ưu tiên cao
3. Hoặc dùng `WebSearch` / `WebFetch` để tra paper + code repo
4. Mục tiêu cuối: chốt 3–5 SOTA baselines feasible để chạy + viết comparison table cho paper

---

## Context

Bài hiện tại đã có:
- 3 contributions (Imbalance toolkit / CBAM Attention / BiomedCLIP fusion)
- Cross-dataset eval (RSNA + SPIDER)
- Selective transfer finding
- Multi-seed + ablation (planned, ~17h compute)

**Gap cho Rank B**: thiếu so sánh với *published methods* trên cùng dataset/benchmark. Phần lớn paper Rank B yêu cầu:
- ≥ 2 baselines từ literature (không phải tự-baseline)
- SOTA comparison trên ít nhất 1 task chính
- Justify vì sao method mình tốt hơn/khác biệt

---

## Candidate methods (ưu tiên giảm dần)

### 🔴 Tier 1 — MUST HAVE cho Rank B (high priority)

#### 1. RSNA 2024 Kaggle top solutions

**Tại sao**: RSNA 2024 là benchmark *trực tiếp* cho task của bài. Top Kaggle solutions = de-facto SOTA.

**Methods cần research:**
- **Ningshen rsna-2024** (https://github.com/nshen7/rsna-2024) — đã đề cập trong motivation doc; pipeline 2D Faster R-CNN + Swin Transformer
- **1st place RSNA 2024** — search Kaggle leaderboard discussion
- **2nd-5th place** — thường có solution writeup

**Research questions:**
1. Top 5 RSNA 2024 Kaggle solutions là gì? Architecture nào? Có public code không?
2. Score Mean F1 macro / weighted log loss của top solutions?
3. Các solution dùng IVD-level vs study-level evaluation? (Bài mình IVD-level)
4. Reproducible trên consumer GPU không? Cần inference time bao nhiêu?

**Feasibility**:
- ⚠️ Code available cho 1-3 solution top (Kaggle write-ups)
- ⚠️ Reproducing có thể cần dữ liệu raw DICOM (đã có)
- ✅ Inference từ pretrained weights nhanh (<1h)

**Effort estimate**: ~1-2 ngày để reproduce 2-3 top solutions, run trên cùng val split của bài.

#### 2. SpineNet upstream (Windsor et al. 2024)

**Tại sao**: Bài này extend từ SpineNet — *phải* so với upstream để chứng minh CBAM + BiomedCLIP add value.

**Methods cần research:**
- **SpineNet original** (https://github.com/rwindsor1/SpineNet) — đã có code trong repo này
- Inference đã có sẵn (`test_spinenet.py`, `test_spinenet_on_rsna.py`)

**Research questions:**
1. SpineNet upstream apply trực tiếp lên RSNA 2024 cho score bao nhiêu?
2. Có thể adapt label space SpineNet (11 conditions) sang RSNA (3 conditions overlap) không?
3. Compare 11-class vs 3-class — fair comparison làm sao?

**Feasibility**: ✅ Đã có code + weights trong repo.

**Effort estimate**: ~半天 chạy inference + table compare.

---

### 🟡 Tier 2 — STRONG có thì tốt (medium priority)

#### 3. MedGemma (Google 2025)

**Tại sao**: 4B/27B medical foundation model — direct competitor cho BiomedCLIP fusion approach.

**Methods cần research:**
- **MedGemma 4B / 27B** (paper: arXiv:2507.05201, đã có PDF `med_gemma.pdf` trong repo)
- Zero-shot capability cho medical imaging

**Research questions:**
1. MedGemma có image encoder zero-shot không? Hay chỉ text reasoning?
2. Apply MedGemma cho RSNA 2024 hoặc SPIDER zero-shot có khả thi không?
3. Cần GPU bao nhiêu để chạy inference? (4B model: 1× A100 80GB; 27B: 4× A100)
4. So với Hybrid zero-shot F1=0.362 trên SPIDER, MedGemma đạt được bao nhiêu?

**Feasibility**:
- ⚠️ MedGemma 27B yêu cầu compute lớn (Vast.ai 4× A100 đắt)
- ⚠️ MedGemma 4B feasible nhưng có thể không có image encoder
- ⚠️ License: Google có thể restrict commercial — cần check

**Effort estimate**: ~2-3 ngày research + ~6h compute nếu khả thi.

**Alternative if not feasible**: Comment trong paper "MedGemma is concurrent work; comparison left for future" — vẫn ổn cho Rank B.

#### 4. RadFM (Radiology Foundation Model)

**Tại sao**: Radiology-specific foundation model, có thể là baseline cho zero-shot transfer.

**Methods cần research:**
- **RadFM** (Wu et al. 2023, https://github.com/chaoyi-wu/RadFM)
- Pretrained trên CT, MRI, X-ray — có thể có spine MRI

**Research questions:**
1. RadFM có support spinal MRI không?
2. Zero-shot eval format (cosine similarity vs prompt)?
3. Compare với BiomedCLIP zero-shot — RadFM tốt hơn hay tệ hơn?

**Feasibility**:
- ⚠️ Code public nhưng setup phức tạp
- ⚠️ Compute cần ~24GB VRAM

**Effort estimate**: ~2-3 ngày setup + run.

---

### 🟢 Tier 3 — NICE to have (low priority, easy wins)

#### 5. Imbalance handling baselines

**Tại sao**: Để chứng minh *toolkit của bài* (Focal + CW + Oversample) tốt hơn published baselines.

**Methods cần research:**
- **Class-Balanced Loss (CB-Loss)** — Cui et al. 2019
- **LDAM Loss** — Cao et al. 2019
- **Logit Adjustment** — Menon et al. 2021 (ICLR)
- **Decoupled head training** — Kang et al. 2020

**Research questions:**
1. Apply lên cùng backbone (3D ResNet34) trên RSNA, score bao nhiêu so với toolkit của bài?
2. Có cần thêm hyperparameter tuning không?

**Feasibility**: ✅ Easy — chỉ cần swap loss function, không cần đổi data pipeline.

**Effort estimate**: ~半天 cho mỗi baseline. Tổng 2-3 baseline = ~1.5-2 ngày.

#### 6. Attention mechanism baselines

**Tại sao**: Để chứng minh *CBAM* tốt hơn các attention modules khác.

**Methods cần research:**
- **SE-Net** (Squeeze-and-Excitation, Hu et al. 2018)
- **Non-local Network** (Wang et al. 2018)
- **ECA-Net** (Wang et al. 2020)
- **Swin Transformer** (Liu et al. 2021) — different architecture entirely

**Research questions:**
1. Plug SE-Net / ECA vào 3D ResNet34, so với CBAM trên RSNA?
2. Có paper nào đã so sánh CBAM với các attention khác trên 3D medical imaging chưa?

**Feasibility**: ✅ Easy — swap attention block.

**Effort estimate**: ~半天 mỗi baseline. Tổng 2 baseline = ~1 ngày.

#### 7. Vision-Language baselines (cho zero-shot)

**Tại sao**: Compare BiomedCLIP-based zero-shot với general CLIP.

**Methods cần research:**
- **OpenAI CLIP** (Radford et al. 2021) — original CLIP
- **OpenCLIP** (Cherti et al. 2023) — open re-implementation
- **SigLIP** (Zhai et al. 2023) — sigmoid loss variant

**Research questions:**
1. CLIP gốc zero-shot trên SPIDER 8 nhãn cho F1 bao nhiêu?
2. Naked CLIP vs Naked BiomedCLIP — show medical pretraining đáng giá bao nhiêu?

**Feasibility**: ✅ Easy — just swap encoder.

**Effort estimate**: ~半天 cho mỗi variant.

---

### ⚪ Tier 4 — Extra (chỉ làm nếu thừa thời gian)

#### 8. SPIDER dataset paper baselines

**Tại sao**: SPIDER paper (van der Graaf et al. 2024) có thể có baseline models.

**Research questions:**
1. SPIDER paper có baseline cho 8 nhãn không? Score là gì?
2. Bài mình so với baseline gốc của SPIDER ra sao?

#### 9. LLaVA-Med, MedSAM, BioGPT

Other medical foundation models — chỉ research nếu Tier 1-3 chưa đủ.

---

## Recommended SOTA set cho Rank B paper

**Minimum set (đủ pass Rank B reviewer):**

1. ✅ **SpineNet upstream** (Tier 1, Windsor et al. 2024) — đã có code
2. ✅ **1-2 RSNA 2024 Kaggle top solutions** (Tier 1, Ningshen + 1 nữa)
3. ✅ **CB-Loss hoặc LDAM** (Tier 3) — easy imbalance baseline
4. ✅ **Naked CLIP zero-shot** (Tier 3) — show BiomedCLIP-specific training value
5. ⚠️ **MedGemma** nếu compute budget cho phép (Tier 2)

→ **5 baselines** trên RSNA + 4 SPIDER nhãn → bảng comparison đầy đủ.

**Stretch goal (boost lên Rank B-cao):**
- 6. RadFM (Tier 2)
- 7. SE-Net + ECA attention baselines (Tier 3)

---

## Compute + time estimate

| Phase | Tasks | Time |
|---|---|---|
| **Research** (this file deep-dive) | Tra paper + code cho 5-7 methods | 2-3 ngày |
| **Implementation** | Reproduce + adapt cho RSNA val split | 3-5 ngày |
| **Run baselines** | Inference / training 5-7 baselines | ~30h compute (chia nhiều instance) |
| **Comparison table + writing** | Build table, write Section 4 (Results) | 2-3 ngày |
| **TỔNG** | | **~2 tuần (1 dev full-time)** |

→ **Cost**: ~$30-50 Vast.ai compute + 2 tuần effort.

→ **Benefit**: Rank B acceptance probability từ ~30% → ~50–60%. *Có thể* push lên MICCAI workshop / MIDL.

---

## Research questions cho session sau

Khi mở session mới với Claude và `/research`, hãy hỏi từng câu này:

### Q1: RSNA 2024 Kaggle top solutions
- "Research the top 5 RSNA 2024 Lumbar Spine Degenerative Classification Kaggle competition solutions. List their architectures, scores (Mean F1 macro or weighted log loss), and code availability. Focus on solutions that can be reproduced without massive compute."

### Q2: SpineNet upstream applicability
- "How does SpineNet (Windsor et al. 2024) handle RSNA 2024-style 3-class stenosis grading? What's the overlap between SpineNet's 11 conditions and RSNA's 3 conditions? Can the upstream pipeline be applied directly to RSNA 2024 data?"

### Q3: MedGemma capability
- "Does MedGemma (Google 2025, arXiv:2507.05201) support image-text matching or zero-shot classification? What's the smallest variant feasible to run on a single A100? Are there reports of MedGemma applied to spinal MRI?"

### Q4: Imbalance handling SOTA
- "What are the current SOTA imbalance handling techniques for 3D medical image classification (especially spinal MRI)? Compare CB-Loss, LDAM, Logit Adjustment, and Decoupled training. Which has been applied to RSNA-like datasets?"

### Q5: Vision-Language zero-shot for spinal MRI
- "What general-purpose CLIP variants have been applied for medical image zero-shot classification? How does BiomedCLIP compare to OpenCLIP, SigLIP, and CLIP gốc on radiology tasks?"

### Q6: SPIDER baselines
- "What baseline models are reported in the SPIDER dataset paper (van der Graaf et al. 2024)? What scores do they report for the 8 disease labels?"

---

## Output format expected

Sau khi research xong, có thể output thành 1 bảng tổng hợp như:

| Method | Citation | Code | Compute | RSNA Mean F1 | SPIDER Zero-shot F1 | Feasibility | Priority |
|---|---|---|---|---|---|---|---|
| SpineNet upstream | Windsor 2024 | ✅ | <1h | ? | N/A | ✅ Easy | 🔴 |
| Ningshen RSNA | Ning 2024 Kaggle | ✅ | ~6h | ? | N/A | ⚠️ Reproduce | 🔴 |
| MedGemma 4B | Google 2025 | ✅ | ~12h | ? | ? | ⚠️ Compute | 🟡 |
| ... | ... | ... | ... | ... | ... | ... | ... |

Khi có bảng này → quyết định 5-7 methods feasible nhất → implement.

---

## Note quan trọng

**Đừng over-engineer**. Rank B yêu cầu *strong empirical evaluation*, không phải *exhaustive comparison*. 5-7 baselines well-chosen thắng 15 baselines half-implemented.

**Đừng quên ablation đã có**. Multi-seed + 2 ablation (CBAM-only, BiomedCLIP-only) **đã đủ ablation depth** cho Rank B. SOTA comparison là *thêm vào* để chứng minh framework competitive với existing methods.

**Timeline thực tế**: Nếu submit MICCAI 2027 (deadline thường tháng 3) thì có ~6-8 tháng từ giờ. Đủ để làm thêm SOTA comparison nếu commit 2 tuần dedicated.

Nếu submit EMBC 2027 (deadline ~tháng 1-2) thì nên **giữ scope hiện tại + Rank C trước** rồi extend journal Q3 sau.
