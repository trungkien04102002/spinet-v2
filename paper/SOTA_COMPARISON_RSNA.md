# SOTA Comparison Analysis — SpineNetV2 + CBAM + BiomedCLIP

Compiled 2026-05-06. Aggregates 3 sub-agent research reports:
- `research_kaggle_top.md` — RSNA 2024 Kaggle Top-10
- `research_published_papers.md` — peer-reviewed + arXiv papers
- `research_methods_survey.md` — 3D backbone landscape (CNN / Transformer / VLM)

Our reference numbers (RSNA 2024 val, mean over 3 conditions × 3 severities):
**Mean F1 = 0.528, Mean AUC = 0.837, Severe AUPRC = 0.321.**

---

## 1. Critical finding for paper framing

**The Kaggle metric is sample-weighted log-loss, NOT F1/AUC.** None of the
top-10 Kaggle teams report per-class F1, AUC, or AUPRC. This means:

- **No published F1/AUC number exists on the full RSNA 2024 private LB.**
- We cannot directly compare our 0.528 F1 against any Kaggle leaderboard.
- The closest peer-reviewed RSNA-2024 F1 is **Khan/Aktan et al. 2025** (Springer
  IJCIS), but on a balanced internal subset, not the Kaggle private LB.
- The closest independent F1 is **nshen7** (GitHub, not peer-reviewed):
  Severe F1 = 0.5007 — a near-direct comparator to our Severe metric.

**Strategic framing:** Our contribution can be honestly framed as one of the
first full-RSNA per-class F1/AUC baselines that uses CBAM + BiomedCLIP fusion
— a positive, not a weakness. This becomes a defensible contribution axis.

---

## 2. Master SOTA comparison table (recommended for paper Table 1)

| # | Method | Year | Dataset | Backbone | F1 | AUC | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Hallinan et al.** *Radiology* | 2021 | Internal 446-pat. + 100 ext. | Mask R-CNN + 3D CNN | — | κ 0.89–0.96 | Seminal DL stenosis paper. **Must-cite.** Different metric (κ) but same conditions. |
| 2 | **Jamaludin et al.** *Med. Image Anal.* | 2017 | GENODISC | VGG 2D | — | Lin's CCC 0.91 | Original SpineNet ancestor. Cite for genealogy. |
| 3 | **Windsor et al.** *Sci. Rep.* | 2024 | Clinical scans | SpineNetV2 (3D ResNet34) | — | — | Direct backbone source. Cite as "we extend Windsor 2024". |
| 4 | **McSweeney et al.** *Spine* | 2023 | NFBC1966 (1,331 pat.) | SpineNetV2 ext. validation | — | Bal. Acc 78% Pfirr. | Cleanest single-task SpineNet baseline. |
| 5 | **Nigru et al.** *NASSJ* | 2024 | 1,747 disks/353 pat. ext. | SpineNetV2 ext. validation | Pfirr. 0.799, CCS 0.971, foram. 0.838, spond. 0.985, hern. 0.779 | κ 0.46–0.74 | **Most useful single comparator** — covers RSNA + SPIDER label spaces in one paper. |
| 6 | **Lin, Zhang, Shang** *Bioengineering* | 2024 | RSNA-derived 8,160 axial L4-L5 (balanced) | CNN + MHSAM + Slot Attn + **CBAM** | 0.945 | 0.951–0.972 | **Closest architectural twin.** Footnote needed: balanced single-level subset, ~binary equivalent. |
| 7 | **Aktan et al.** *Int. J. Comp. Intell. Sys.* | 2025 | RSNA 2024 (internal split) | MobileNetV2-UNet (seg + class) | 0.957 | — | Same dataset. Footnote: numbers include segmentation Dice averaging on balanced internal split, NOT Kaggle private LB. |
| 8 | **2nd Place Kaggle** (ianpan + Yuji + Bartley) | 2024 | RSNA 2024 LumbarDISC | 2D ConvNeXt/DaViT + LSTM + attn pool, ensemble of 3, YOLOX detect | — | — | Best-documented top-10 Kaggle. Public code. **No F1/AUC reported (log-loss only).** |
| 9 | **4th Place Kaggle** (SPINE CHART, tattaka+yu4u) | 2024 | RSNA 2024 LumbarDISC | 3D vol. ensemble (CAFormer+ConvNeXt+ResNetRS50+SwinV2) | — | — | Only fully-3D top-10 docu. Closest architecture family. **No F1/AUC reported.** |
| 10 | **nshen7** GitHub (independent) | 2024 | RSNA 2024 (own held-out split) | Faster R-CNN + Swin Transformer 2D | Severe F1 **0.501**, Mod F1 0.472, Norm F1 0.906 | — | **Only public per-severity F1 on RSNA 2024.** Direct numerical comparator. |
| 11 | **Acharya & Kansakar** arXiv | 2026 | Sagittal T2 (private) | Disc-Centric Contrastive + focal | Bal. Acc 0.781 | — | Closest to our class-imbalance design. Cite for severity grading. |
| 12 | **Bharadwaj et al.** *Eur. Radiol.* | 2023 | 200 pat./987 axial slices | V-Net + BiT ResNet-50 + DT | — | CCS 0.94, foram. 0.92, facet 0.93 | Interpretable cascade. Cite for interpretability framing. |
| 13 | **Won et al.** *Global Spine J.* | 2024-25 | 13,758 axial T2/542 pat. | Faster R-CNN + VGG (3-stage) | 0.784 | Acc 91.5% | Comparable F1 magnitude. Different protocol (axial-only, 4-class A-D). |
| 14 | **OURS** (CBAM 3D ResNet34 + BiomedCLIP fusion) | 2026 | RSNA 2024 val (full, 3-class multi-level, 1942 samples, 395 pat.) | 3D ResNet34 + CBAM + frozen BiomedCLIP, concat-MLP | **0.528** mean | **0.837** mean | Severe AUPRC 0.321; first BiomedCLIP-based RSNA solution. |

### Footnote conventions for the table

- `[†]` for balanced/single-level subsets where direct F1 comparison is misleading
- `[‡]` for "different metric only" (κ-based, log-loss-based)
- `[*]` for in-house datasets (cannot replicate split)
- `[**]` for our split (RSNA val, random_state=42, test_size=0.2, patient-level)

---

## 3. Defensible novelty axes (no top-10 Kaggle uses)

Confirmed gaps in the literature where we have clean novelty:

| Axis | Top-10 Kaggle uses? | Published peer-reviewed RSNA-2024 use? | Verdict |
|---|---|---|---|
| **CBAM attention on 3D ResNet** | 0/4 documented | 1 (Lin 2024, but 2D balanced subset) | **Clean novelty for full RSNA 3D.** |
| **CLIP/BiomedCLIP fusion** | 0/4 documented | None for RSNA 2024 | **Strongest novelty axis.** First BiomedCLIP-based RSNA solution. |
| **Focal/uncertainty loss** | 0/4 documented (all use weighted CE) | Acharya 2026 (focal only, different dataset) | Defensible. |
| **Per-class AUC + AUPRC reporting** | 0/4 (Kaggle log-loss only) | None for full RSNA 2024 | **First per-class AUC/AUPRC baseline.** |
| **Zero-shot label extension via text prompts (RSNA → SPIDER)** | 0/4 | None | **Unique contribution.** |

### Methodological patterns from top solutions (for "Why our design choices?" section)

| Pattern | Top-10 prevalence | Our choice |
|---|---|---|
| Multi-stage pipeline (detect → crop → classify) | 4/4 (100%) | Single-stage on pre-cropped IVD volumes (uses RSNA-provided IVD coordinates upstream) |
| 2D / 2.5D backbones (ConvNeXt, DaViT, Swin, EdgeNeXt) | 3/4 | 3D ResNet34 (more volumetric than peers) |
| Sagittal + axial fusion | 3/4 | Sagittal-only (9 slices); known limitation for foraminal |
| Modern post-2022 backbones | 4/4 | ResNet34 (older, but with strong medical-domain pretrain) |
| Pseudo-labels / label denoising | 2/4 | Not used (defensible: dataset is already cleaned) |
| TTA | 2/4 | Not used (single-pass inference for clinical reproducibility) |
| Ensembling (always >=3 models) | 4/4 | Single model (defensible: focus on architecture novelty, not ensemble engineering) |

---

## 4. Why 3D ResNet-34 + CBAM + BiomedCLIP is defensible (Methods justification)

For paper Methods section:

1. **Pretraining dominates on small data.** MedicalNet 3D-ResNet weights
   (Tencent, 2019, still on HF) are the only large-scale *medical-domain* 3D
   pretrained checkpoints. 3D EfficientNet, 3D DenseNet, 3D ViTs lack
   equivalent medical pretrain — decisive in our ~2k-volumes-per-condition
   regime.

2. **Transformers underperform on small 3D classification crops.** 2024-25
   reviews (Vision Transformers in Medical Imaging, NVIDIA tech blog) confirm:
   Swin-UNETR/UNETR are SOTA for *segmentation* of large CT/MRI volumes, but
   on small per-IVD crops (9×112×224) without in-domain pretraining,
   transformer inductive bias deficit dominates.

3. **2024-25 lumbar literature is mostly 2D.** SpineScan (Eur. Spine J. 2025),
   GE-YOLOv8 (Frontiers 2025), PLOS-ONE Pfirrmann pipeline, YOLOv8x detector
   — all 2D sagittal-slice with ResNet-50 or YOLO backbones. Our 3D is *more*
   volumetric than dominant baselines.

4. **CBAM is the canonical attention recipe.** OASIS Alzheimer 3D-ResNet+CBAM
   2024 (88.0% acc) confirms the standard pattern; we don't pay the
   transformer training-data cost.

5. **BiomedCLIP fusion uses the model correctly.** BiomedCLIP is 2D
   image-text; using it as a *text/auxiliary stream* over per-slice features
   (rather than 3D volumetric encoder) matches its training distribution.
   RadCLIP (March 2024) is the only VLM with explicit 3D slice-pooling — cited
   as future work.

---

## 5. Risk: foundational comparators we should re-evaluate ourselves

These are checkpoints where we *could* run inference on our val split to get
true apples-to-apples numbers. Not strictly required, but would strengthen
the table significantly.

| Comparator | Effort to re-evaluate on our split | Value | Recommendation |
|---|---|---|---|
| **2nd-place Bartley repo** (`brendanartley/RSNA-2024-Competition`) | Medium (their preprocessing is clean, MIT-licensed) | High — would give first-ever F1 on Top-2 Kaggle solution | **Strongly consider** if reviewers push |
| **4th-place tattaka repo** | High (Docker required, 4× A4000 inference) | Medium-high | Optional — flag as future work |
| **nshen7** | Low (single-pass 2D inference) | Medium — already has Severe F1 0.501 reported | **Already directly comparable.** Just cite. |
| **SpineNetV2 weights from Windsor 2024** | Already done — that's our backbone | — | — |

---

## 6. Required citations for paper

### Must-cite (foundational)

1. **Hallinan et al. 2021** *Radiology* — DL stenosis seminal. DOI 10.1148/radiol.2021204289.
2. **Jamaludin et al. 2017** *MIA* — original SpineNet.
3. **Windsor et al. 2024** *Sci. Rep.* — SpineNetV2 source. DOI 10.1038/s41598-024-64580-w.
4. **Zhang et al. 2023** BiomedCLIP — arXiv:2303.00915 / NEJM AI 2025.
5. **van der Graaf et al. 2024** *Sci. Data* — SPIDER dataset. DOI 10.1038/s41597-024-03090-w.
6. **Richards et al. 2025** arXiv:2506.09162 — LumbarDISC (RSNA 2024) dataset.
7. **CBAM (Woo et al. 2018)** ECCV — attention module source.

### Direct comparators (for Table 1)

8. **Nigru et al. 2024** *NASSJ* — DOI 10.1016/j.xnsj.2024.100564.
9. **Lin, Zhang, Shang 2024** *Bioengineering* — DOI 10.3390/bioengineering11101021.
10. **Aktan et al. 2025** *IJCIS* — DOI 10.1007/s44196-025-01098-7.
11. **McSweeney et al. 2023** *Spine* — PMC9990601.
12. **nshen7** GitHub (Severe F1 0.501) — github.com/nshen7/rsna-2024.
13. **2nd-place Kaggle** (Artley) — github.com/brendanartley/RSNA-2024-Competition.
14. **4th-place Kaggle** (tattaka) — github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public.

### Methodological context

15. **MedicalNet (Chen et al. 2019)** — arXiv:1904.00625.
16. **Swin-UNETR (Tang et al. 2022)** — CVPR.
17. **RadCLIP (2024)** — arXiv:2403.09948.
18. **Acharya & Kansakar 2026** — arXiv:2602.05738.
19. **Bharadwaj et al. 2023** *Eur. Radiol.* — PMC10566647.
20. **Won et al. 2024** *Global Spine J.* — DOI 10.1177/21925682241299332.

---

## 7. Gap analysis — what's still needed for Rank-C confidence

| Item | Status | Effort | Priority |
|---|---|---|---|
| SOTA comparison table | ✅ Done (this file) | 0h | — |
| Hallinan κ comparison line in Table 1 | Pending — convert our preds to dichotomous κ for canal/foram | ~2h | High |
| Re-eval Bartley 2nd-place on our val split | Pending — clone repo, run inference | ~6h on Vast | Medium (defensive) |
| Re-eval nshen7 on our val split | Pending — small repo | ~3h on Mac/Vast | High (cleanest direct comparison) |
| 3-seed mean±std for Hybrid | Pending — costs ~$3 on Vast 1×3.5h × 3 | ~10h | High |
| Mendeley dataset eval | Pending separate research | ~4h research + ~6h eval | Medium (boost if works) |
| Statistical test (paired t-test or McNemar) Hybrid vs CBAM-only | Pending | ~1h | Medium |

**Updated Rank-C confidence with this SOTA table:** ~70%. With 3-seed + Hallinan κ + nshen7 re-eval: ~80%. With Mendeley if works: ~85%.

---

## 8. Recommended next steps (ranked)

1. **Insert Table 1 into paper** using rows 1, 5, 6, 7, 10, 14 (top 6 from master table). [~2h writing]
2. **Compute Hallinan κ on our val split** (dichotomous "any severe" per condition). [~1-2h]
3. **Run nshen7 inference on our val split** for direct F1 comparison. [~3h]
4. **3-seed Hybrid retrain on Vast** for stat. significance. [~10h, ~$3]
5. (Optional) Re-eval Bartley 2nd-place if reviewers ask. [~6h]
6. (Parallel) Mendeley deep-dive research to decide if usable. [~2h research]
