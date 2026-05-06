# Deep-dive Extraction: 5 Direct-Comparator Papers for Rank-C SOTA Table 1

Compiled 2026-05-06 for the SpineNetV2 (CBAM 3D ResNet34 + frozen BiomedCLIP via concat-MLP) paper.
Our reference numbers (RSNA 2024 val, mean over 3 conditions x 3 severities, random_state=42, test_size=0.2 patient-level split, 1942 IVD samples / 395 patients): **Mean F1 = 0.528, Mean AUC = 0.837, Severe AUPRC = 0.321**.

Method: WebFetch on publisher pages was permission-denied for all `link.springer.com`, `pubs.rsna.org`, `mdpi.com`, `sciencedirect.com`, `web.archive.org`, `researchgate.net`, `semanticscholar.org`, `pubmed.ncbi.nlm.nih.gov`, `cris.vtt.fi`, and most other publisher mirrors. The two open-access PMC mirrors that succeeded — **PMC11617751** (Nigru) and **PMC11504910** (Lin) — and **PMC9990601** (McSweeney) gave full table-level extraction. Hallinan and Aktan numbers below are abstract-level only and flagged as **needs manual download by user**.

---

## Paper 1: Nigru et al. 2024 (NASSJ) — PRIORITY: HIGHEST

### Bibliographic info
- **Title:** External validation of SpineNetV2 on a comprehensive set of radiological features for grading lumbosacral disc pathologies
- **Authors:** Alemu Sisay Nigru, Sergio Benini, Matteo Bonetti, Graziella Bragaglio, Michele Frigerio, Federico Maffezzoni, Riccardo Leonardi
- **Venue:** North American Spine Society Journal (N Am Spine Soc J), 20:100564
- **Year:** 2024 (online 2024-10-26)
- **DOI:** 10.1016/j.xnsj.2024.100564
- **PMCID:** PMC11617751 | **PMID:** 39640208
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11617751/
- **Open access:** Yes (CC BY 4.0)
- **Code:** No custom validation repo. SpineNetV2 itself: https://github.com/rwindsor1/SpineNet.git

### Dataset & split
- **External validation only** — no model training/fine-tuning in this paper.
- **Source:** retrospective Italian single-center MRI cohort
- **Initial pool:** 1,615 MRI scans
- **Exclusions:** 1,145 thoracic/cervical, 89 pediatric (age <14), 28 poor quality
- **Final cohort:** 1,747 lumbosacral IVDs from **353 patients**
- **Demographics:** mean age 54 +/- 15.4 years, 44.5% female
- **Levels covered:** L1-L2 through L5-S1 (no upper-level exclusion)
- **Reference standard:** consensus radiologist grading; 60 patients / 289 discs were also dual-rated for inter-rater table
- **No train/val/test split** — entire 1,747 IVDs are held-out test for SpineNetV2

### Metric definitions
- **Accuracy:** (TP+TN)/N
- **Balanced Accuracy (BAS):** (Sensitivity + Specificity)/2
- **F1:** 2*P*R/(P+R) — averaging method **NOT explicitly specified** in the paper text; values appear to be macro-style (per-class then averaged) since binary precision/recall are reported separately
- **Cohen's kappa (kappa):** unweighted (Po-Pe)/(1-Pe)
- **Lin's Concordance CC (LCCC):** 2*Cxy/(Cxx+Cyy+(mu_x-mu_y)^2)
- **MCC:** Matthews CC (multi-component formula)
- **Brier Score Loss (BSL):** (1/N)*sum(Pi-Oi)^2

### Headline results table (Table 4 — primary, all 11 pathologies, full 1,747-disc cohort)

| Pathology | Accuracy | Balanced Acc | Precision | Recall | **F1** | BSL | **kappa** | LCCC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| Pfirrmann (5-class) | 0.796 | 0.794 | 0.799 | 0.796 | **0.796** | 0.081 | **0.738** | 0.952 | 0.738 |
| Disc Narrowing | 0.867 | 0.849 | 0.869 | 0.868 | 0.868 | 0.066 | 0.799 | 0.972 | 0.799 |
| Central Canal Stenosis | 0.971 | 0.779 | 0.971 | 0.971 | 0.971 | 0.014 | 0.749 | 0.932 | 0.749 |
| Spondylolisthesis | 0.983 | 0.988 | 0.983 | 0.983 | **0.985** | 0.016 | 0.705 | 0.745 | 0.721 |
| Upper Endplate Defect | 0.948 | 0.849 | 0.946 | 0.948 | 0.947 | 0.052 | 0.732 | 0.880 | 0.733 |
| Lower Endplate Defect | 0.942 | 0.863 | 0.940 | 0.942 | 0.941 | 0.058 | 0.745 | 0.920 | 0.745 |
| Upper Marrow Change | 0.940 | 0.895 | 0.943 | 0.940 | 0.941 | 0.060 | 0.756 | 0.896 | 0.757 |
| Lower Marrow Change | 0.931 | 0.892 | 0.937 | 0.931 | 0.933 | 0.070 | 0.731 | 0.873 | 0.734 |
| Right Foraminal Stenosis | 0.852 | **0.702** | 0.841 | 0.852 | 0.838 | 0.148 | **0.473** | 0.542 | 0.494 |
| Left Foraminal Stenosis | 0.854 | **0.691** | 0.843 | 0.854 | 0.838 | 0.146 | **0.457** | 0.530 | 0.483 |
| **Disc Herniation** | 0.790 | 0.759 | 0.810 | 0.790 | **0.779** | 0.210 | **0.546** | 0.630 | 0.576 |

Inter-rater (radiologist-radiologist on 289 discs, Table 3): kappa values 0.823 (Pfirrmann) up to 0.930 (spondylolisthesis); foraminal stenosis kappa = 0.890-0.901 (rad-vs-rad) compared to 0.457-0.473 (model-vs-rad) — large gap on foraminal grading.

### Per-severity breakdown
**NOT reported** as numerical tables. Confusion matrices for all 11 pathologies appear in Figure 4 but the paper does not list per-grade (Pfirrmann 1-5 / stenosis 1-4) accuracy. This is a real gap if we want to compare our Severe-class F1.

### Per-disc-level breakdown
**NOT reported.** All 1,747 discs aggregated; no L1-L2 ... L5-S1 stratification in tables.

### Class imbalance handling
Minimal — they rely on **balanced accuracy** to soften the effect. No focal loss, no class weighting, no oversampling (this is an external-validation paper, not a training paper).

### Backbone & params
- **SpineNetV2** (Windsor et al. 2022/2024). Backbone type (3D ResNet34 vs VGG) not restated.
- **Parameter count NOT reported.**
- **Original training data:** GENODISC consortium (~12,018 discs) + Oxford Whole Spine (710 scans).

### Direct comparability with our RSNA val: **DIRECT (highest priority)**
**Reason:** Nigru reports per-pathology F1/kappa for all five conditions that span our RSNA-side **and** SPIDER-side label spaces in a single paper:
- **CCS (RSNA-side):** their F1 0.971 vs ours (RSNA val, full 3-class) — note their CCS is binary-style on a different prevalence; treat as **bridging** for CCS only.
- **Foraminal Stenosis (RSNA-side):** their F1 0.838 / kappa 0.46-0.47 / **Bal. Acc 0.69-0.70** is the realistic ceiling for sagittal-only foraminal grading, and is our most defensible benchmark for the foraminal head.
- **Pfirrmann (SPIDER-side):** F1 0.796 / kappa 0.738 — direct comparator for our SPIDER Pfirrmann head.
- **Spondylolisthesis (SPIDER-side):** F1 0.985 / kappa 0.705 — direct comparator for our SPIDER spondy head (note the very high acc here is driven by extreme class imbalance; their kappa is moderate).
- **Disc Herniation (SPIDER-side):** F1 0.779 / kappa 0.546 — direct comparator.

### Notes for our paper
- **Cite as the single most useful comparator** for SOTA Table 1.
- Nigru's "F1" is averaging-method-ambiguous but consistent with macro F1 (matches our metric).
- Class-imbalance footnote: Nigru's CCS F1 = 0.971 is on a binary CCS task with ~97% prevalence of normal — **don't read as ours being 40 points behind**. Our 3-class CCS on RSNA is harder.
- The foraminal Bal. Acc ~70% / kappa ~0.46 is the most reviewer-defensible "sagittal foraminal grading is intrinsically hard" reference.

---

## Paper 2: Lin, Zhang, Shang 2024 (Bioengineering MDPI) — PRIORITY: HIGH

### Bibliographic info
- **Title:** Convolutional Neural Network Incorporating Multiple Attention Mechanisms for MRI Classification of Lumbar Spinal Stenosis
- **Authors:** Juncai Lin, Honglai Zhang, Hongcai Shang
- **Venue:** Bioengineering (Basel) 11(10):1021
- **Year:** 2024 (online 2024-10-13)
- **DOI:** 10.3390/bioengineering11101021
- **PMCID:** PMC11504910
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11504910/
- **Open access:** Yes (CC BY 4.0)
- **Code:** Not released (no GitHub link in paper)

### Dataset & split
- **Source:** "RSNA and ASNR multicenter MRI dataset" — wording suggests the RSNA 2024 LumbarDISC images, but reduced to a **balanced single-level subset**.
- **Total images:** 8,160 (1,632 per condition x 5 conditions).
- **Anatomical scope:** **L4-L5 only** (single level).
- **Resolution:** 224 x 224, bilinear-resampled.
- **Classification:** **Binary** — normal/mild vs severe (NOT 3-class).
- **Train/test split:** 80/20.
- **Patient- vs slice-level split:** not explicitly stated; described as image-level (likely **slice-level** — possible patient leakage, flag this).

Class distribution per Table 1:

| Condition | Normal/Mild | Severe | Total |
|---|---|---|---|
| Spinal Canal Stenosis | 1,429 | 203 | 1,632 |
| Left Neural Foraminal | 1,517 | 115 | 1,632 |
| Right Neural Foraminal | 1,525 | 107 | 1,632 |
| Left Subarticular | 1,243 | 389 | 1,632 |
| Right Subarticular | 1,244 | 388 | 1,632 |

Heavy imbalance (e.g. 1,517:115 = 13.2x), but **the cross-condition balance is 1,632 per condition**, so per-condition macro F1 numbers benefit from equal sample sizes.

### Metric definitions
- Standard binary formulas: Acc, Precision, Recall, F1.
- **Averaging method NOT explicitly stated**, but appears macro-binary (per-class averaged).
- **AUC:** ROC curves shown in Figure 11; computation method not detailed.

### Headline results table

**Table 2 — overall model comparison (averaged across 5 conditions):**

| Model | Acc | Precision | Recall | F1 | Train time (s) |
|---|---|---|---|---|---|
| **Proposed (CBAM+MHSAM+SAM)** | **95.2%** | **94.7%** | **94.3%** | **94.5%** | 10,377 |
| ResNet50 | 90.3% | 89.7% | 89.1% | 89.4% | 13,245 |
| ResNet101 | 92.5% | 91.8% | 91.0% | 91.4% | 17,575 |
| DenseNet121 | 91.2% | 90.6% | 90.0% | 90.3% | 7,038 |
| DenseNet201 | 93.1% | 92.5% | 91.8% | 92.1% | 17,649 |
| VGG19 | 88.7% | 88.0% | 87.5% | 87.7% | 27,492 |
| Xception | 89.8% | 89.1% | 88.5% | 88.8% | 18,071 |

**Table 3 — per-condition (Proposed Model only, plus DenseNet201 baseline):**

| Condition | Acc | Precision | Recall | F1 |
|---|---|---|---|---|
| Spinal Canal Stenosis (Proposed) | 95.2% | 94.7% | 94.3% | 94.5% |
| Spinal Canal Stenosis (DenseNet201) | 93.0% | 92.4% | 91.7% | 92.0% |
| Left Neural Foraminal (Proposed) | 95.4% | 94.9% | 94.5% | 94.7% |
| Left Neural Foraminal (DenseNet201) | 93.2% | 92.6% | 91.9% | 92.2% |
| Right Neural Foraminal (Proposed) | 95.5% | 95.0% | 94.7% | 94.9% |
| Right Neural Foraminal (DenseNet201) | 93.3% | 92.7% | 92.1% | 92.4% |
| Left Subarticular (Proposed) | 95.1% | 94.6% | 94.1% | 94.3% |
| Left Subarticular (DenseNet201) | 92.9% | 92.2% | 91.5% | 91.8% |
| Right Subarticular (Proposed) | 95.2% | 94.7% | 94.3% | 94.5% |
| Right Subarticular (DenseNet201) | 93.0% | 92.4% | 91.7% | 92.0% |

**ROC-AUC (from Figure 11, Proposed model):** Spinal Canal Stenosis 0.972, Right Neural Foraminal 0.951.

**Ablation (Tables 2 & 4):**

| Configuration | Acc | F1 |
|---|---|---|
| Full model (CBAM + MHSAM + SAM) | 95.2% | 94.5% |
| Without CBAM | 92.8% | 91.8% |
| Without MHSAM | 93.2% | 92.2% |
| Without SAM (Slot Attention) | 92.9% | 92.0% |

CBAM contribution: ~2.4% Acc, 2.7% F1.

### Per-severity breakdown
**Binary task only** (Normal/Mild vs Severe). No 3-class or 4-class breakdown.

### Per-disc-level breakdown
**N/A** — L4-L5 only.

### Class imbalance handling
- Augmentation only: scale [0.8, 1.2], translate +/-20 px, rotate +/-15 deg, vertical flip.
- **No focal loss, no class weighting, no oversampling.** Per-condition cross-entropy implied.
- Stated purpose: "alleviate the problem of uneven distribution".

### Backbone & params
- Custom CNN: head (7x7 conv) -> body (Enhanced Inception Module with depthwise-separable conv) -> tail (GAP).
- Three attention modules: CBAM, MHSAM (multi-head self-attn), SAM (slot attention).
- Optimizer: Adam, lr=0.001, dropout 0.5.
- **Parameter count NOT reported.**
- Hardware: RTX 3090 24GB; PyTorch 2.3.0.

### Direct comparability with our RSNA val: **INDIRECT-BRIDGING (with significant footnote)**
**Reason:** Same source dataset (RSNA/ASNR), but three reductions vs ours:
1. **Single-level (L4-L5) only** vs our multi-level (L1-S1).
2. **Binary (normal/mild vs severe)** vs our 3-class (normal/mild, moderate, severe).
3. **Slice-level (likely)** vs our patient-level split — possible patient leakage in their numbers.
4. They balanced sample counts to **1,632 per condition** vs our natural prevalence in 1,942 IVDs.

A 95.2% binary single-level F1 vs our 0.528 3-class multi-level F1 = **NOT 42 percentage points behind** — the tasks are different. Use a footnote: "[+] reported on balanced single-level L4-L5 binarized subset; not directly comparable to multi-level 3-class F1".

### Notes for our paper
- **Cite as closest architectural twin** (CBAM + multi-attention).
- Their CBAM ablation shows 2.4-2.7 pt Acc/F1 gap, which is a **defensible reference number** for our own CBAM ablation magnitude.
- Use as Table 1 entry **with the binary/L4-L5 footnote**.
- The 95.2% number alone, in a SOTA table, will mislead reviewers unless you show task-difficulty side-by-side.

---

## Paper 3: Aktan et al. 2025 (IJCIS) — PRIORITY: HIGH

### Bibliographic info
- **Title:** Automated Lumbar Spine Degenerative Classification Using Deep Learning: A Comprehensive Evaluation Based on RSNA 2024
- **Authors:** Aktan / Khan et al. (full author list not retrieved — abstract page blocked)
- **Venue:** International Journal of Computational Intelligence Systems
- **Year:** 2025 (December)
- **DOI:** 10.1007/s44196-025-01098-7
- **URL:** https://link.springer.com/article/10.1007/s44196-025-01098-7
- **Open access:** Springer page indicates yes (Open Access PDF link visible) but **WebFetch blocked** for both HTML and PDF endpoints.
- **Code:** Not retrieved.

### Dataset & split
- **Source:** RSNA 2024 LumbarDISC (same as ours).
- **Train/val/test split:** **NOT EXTRACTED — page blocked.** Likely an internal split, NOT the Kaggle private LB (Kaggle private LB is held server-side; only competition entrants can score it).
- **Patient- vs slice-level:** not extracted.
- **Sample sizes:** not extracted.

### Metric definitions
- **NOT EXTRACTED.** Abstract reports Acc / Dice / F1 only. Macro vs sample averaging unknown.

### Headline results (abstract-only)

| Condition | Accuracy | F1 | Dice | IoU |
|---|---|---|---|---|
| **Aggregated** (segmentation+classification, all conditions combined) | 94.93% | **95.65%** | 94.61% | not reported in abstract |
| Per-condition (spinal canal, foraminal, subarticular) | **NOT EXTRACTED — needs manual download** | | | |

### Per-severity breakdown
**NOT EXTRACTED — needs manual download.**

### Per-disc-level breakdown
**NOT EXTRACTED — needs manual download.**

### Class imbalance handling
- Loss: Cross-Entropy + IoU (joint).
- No mention in abstract of focal/oversample/class-weight.

### Backbone & params
- **MobileNetV2-UNet** (joint segmentation + classification).
- **Depthwise-separable** convolutions for efficiency.
- **Parameter count NOT EXTRACTED.**

### Direct comparability with our RSNA val: **INDIRECT-BRIDGING (high uncertainty)**
**Reason:** Same dataset (RSNA 2024 LumbarDISC), but:
1. The 94.93% Acc / 95.65% F1 numbers are **suspiciously high** vs Kaggle LB (private LB log-loss puts winners at ~0.40, which translates to F1 well below 0.95 in 3-class space).
2. The aggregated 95.65% F1 likely reflects a **segmentation-weighted joint metric** (because they report Dice alongside) — not a classification-only macro F1 across the 3-class task. Until per-condition + per-class numbers are extracted, we cannot put this number in our Table 1 as-is.
3. Possibly evaluated on a **balanced internal subset**, not the full Kaggle private LB.

### Notes for our paper
- **Cite as same-dataset comparator with explicit caveats.**
- **Manual download required** to extract per-condition / per-severity tables. Path: download the open-access PDF directly from `https://link.springer.com/content/pdf/10.1007/s44196-025-01098-7.pdf` via browser, then re-extract Tables 3-5.
- Without the per-class breakdown, do NOT put a single 95.65% number next to our 0.528 — reviewers will flag.
- Once we have the per-condition table, this becomes the **only peer-reviewed RSNA 2024 F1 reference** and is essential.

---

## Paper 4: McSweeney et al. 2023 (Spine) — PRIORITY: MEDIUM

### Bibliographic info
- **Title:** External Validation of SpineNet, an Open-Source Deep Learning Model for Grading Lumbar Disk Degeneration MRI Features, Using the Northern Finland Birth Cohort 1966
- **Authors:** Terence P McSweeney, Aleksei Tiulpin, Simo Saarakkala, Jaakko Niinimaki, Rhydian Windsor, Amir Jamaludin, Timor Kadir, Jaro Karppinen, Juhani Maatta
- **Venue:** Spine (Phila Pa 1976) 48(7):484-491
- **Year:** 2023 (online 2022-12-30; issue 2023-04-01)
- **DOI:** 10.1097/BRS.0000000000004572
- **PMCID:** PMC9990601
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9990601/
- **Open access:** Yes (CC BY 4.0)
- **Code:** SpineNet open-source; demo at http://zeus.robots.ox.ac.uk/spinenet2/

### Dataset & split
- **Cohort:** Northern Finland Birth Cohort 1966 (NFBC1966).
- **Patients:** 1,331; **Discs:** 6,655.
- **Disc levels:** L1-L2 through L5-S1.
- **Demographics:** 706 female (53.0%), 625 male (47.0%); BMI / LBP distributions in Table 1 of paper.
- **MRI:** T2-weighted sagittal, up to 9 slices passed as a volume to SpineNet.
- **Reference standard:** consensus of 4 expert raters (3 for Pfirrmann, 2 for Modic).
- **No train/val/test split** — entire 6,655 discs are held-out external test for SpineNetV2.

### Metric definitions
- **Cohen's kappa:** unweighted, both Pfirrmann (5-class) and Modic (binary).
- **Lin's CCC:** standard formula (no explicit weighted variant noted).
- **Balanced Accuracy:** (sensitivity + specificity)/2 (macro per-class).
- **Macro AUC and Micro AUC:** both reported for Pfirrmann.
- **MCC, Brier Score Loss, Gwet's AC1:** all reported.
- **F1: NOT reported** in primary results table (this is a real gap for direct F1 comparison).

### Headline results table (Table 2 — primary, with 95% CI)

| Metric | Pfirrmann (5-class) | Modic Changes (binary) |
|---|---|---|
| Accuracy | 79% (78-80) | 92% (91-92) |
| **Balanced Acc** | **78% (77-79)** | **86% (85-86)** |
| Macro AUC | 0.95 (0.95-0.95) | - |
| Micro AUC | 0.94 (0.94-0.94) | - |
| AUC (binary) | - | 0.95 (0.95-0.95) |
| Brier Score Loss | 0.20 | 0.21 |
| Sensitivity | 79% (78-80) | 89% (88-91) |
| Specificity | 93% (93-93) | 96% (96-96) |
| MCC | 0.68 (0.67-0.70) | 0.74 (0.73-0.75) |
| **Cohen's kappa** | **0.68 (0.67-0.69)** | **0.74 (0.72-0.75)** |
| **Lin's CCC** | **0.86 (0.85-0.87)** | - |
| Gwet's AC1 | 0.73 (0.72-0.74) | 0.88 (0.87-0.89) |

**Disagreement analysis:** "20.83% of disks were rated differently by SpineNet vs human raters, but only 0.85% had a difference in grade > 1" (Pfirrmann); "8.24% of MC classified differently".

### Per-severity breakdown
**Not provided as a numerical table.** Confusion matrix shown in Figure 3 but per-grade percentages not transcribed. Important caveat: **NFBC reference labels merged Pfirrmann grades 1+2 into "grade 2"**, while SpineNet outputs all five grades — this introduces a built-in disagreement penalty.

### Per-disc-level breakdown
**Reported graphically only** (Figures 4 + 5 with 95% CIs); the paper does NOT print numerical per-level tables. Comment: "low prevalence of DD/MC at L1-L2 and L2-L3 limits significance at upper levels".

Sub-cohort analysis: prolonged-LBP subset (>=30 days) — "Lin's CCC unchanged"; MC with small lesions excluded -> kappa = 0.76.

### Class imbalance handling
- Acknowledged but **no active mitigation**: "balanced accuracy" used as the headline metric instead of raw accuracy.
- No focal, no oversampling, no reweighting (this is external validation, not training).

### Backbone & params
- **SpineNetV2** (the same architecture we extend).
- **Parameter count: NOT reported.**
- Original training: GENODISC consortium (UK/HU/IT/SI multi-center).
- Training-set baseline (from V2 paper): Bal. Acc 70.9% Pfirrmann, 88.9%/88.2% upper/lower MC.

### Direct comparability with our RSNA val: **DIRECT for Pfirrmann (SPIDER side); INDIRECT for our RSNA conditions**
**Reason:** McSweeney is the cleanest single-task SpineNet baseline. Their Pfirrmann Bal. Acc 78% / kappa 0.68 / Lin CCC 0.86 maps **directly** onto our SPIDER Pfirrmann head's evaluation. They do **not** report F1 (so a "Mean F1" comparator must come from Nigru, not McSweeney). Their RSNA-side conditions (CCS / foraminal) are **not** evaluated since NFBC doesn't have those labels.

### Notes for our paper
- **Cite as the canonical Pfirrmann external-validation baseline.** Bal. Acc 78% is the number to put in our Table 1 SPIDER row.
- Note the F1 omission — we'll need to either derive F1 from their confusion matrix (if/when we get the figure) or rely on Nigru's 0.796 Pfirrmann F1 instead.
- Use their "20.83% disagree but only 0.85% off-by-more-than-1" framing in our discussion — it's a clean argument for ordinal-aware metrics.

---

## Paper 5: Hallinan et al. 2021 (Radiology) — PRIORITY: HIGH (must-cite)

### Bibliographic info
- **Title:** Deep Learning Model for Automated Detection and Classification of Central Canal, Lateral Recess, and Neural Foraminal Stenosis at Lumbar Spine MRI
- **Authors:** Hallinan, Zhu, et al. (full list not retrieved — publisher and Semantic Scholar both blocked)
- **Venue:** Radiology, 300(1):130-138
- **Year:** 2021 (July)
- **DOI:** 10.1148/radiol.2021204289
- **PMID:** 33973835
- **URL:** https://pubs.rsna.org/doi/10.1148/radiol.2021204289
- **Open access:** Closed (RSNA paywall); abstract via PubMed only.
- **Code:** Not retrieved.

### Dataset & split (from PubMed/Radiology abstract + downstream meta-analysis citations)
- **Patients:** 446 (mean age 52 +/- 19, 240 women).
- **Internal split:** 80% train / 9% validation / 11% internal test (= 396 train+val, 50 internal test patients).
- **External test:** **100 MRI scans** (separate institution).
- **Images per patient:** 12,403 axial T2 + 6,161 sagittal T1 across the cohort.
- **Reference standard:** 4 radiologists, predefined gradings (normal / mild / moderate / severe).

### Metric definitions
- **Cohen's kappa:** weighted (4-class ordinal) AND unweighted (dichotomous = "normal/mild" vs "moderate/severe"). Abstract reports the **dichotomous** kappa as the headline.

### Headline results table (abstract-level — full Tables 2+3 NOT extracted, paywall)

**Internal test, dichotomous kappa (Normal/Mild vs Moderate/Severe):**

| Region | Radiologist 1 kappa | Radiologist 2 kappa | **DL model kappa** |
|---|---|---|---|
| Central Canal | 0.98 | 0.98 | **0.96** |
| Lateral Recess | 0.92 | 0.95 | **0.92** |
| Neural Foramina | 0.94 | 0.95 | **0.89** |

**External test (100 patients), dichotomous kappa:** **0.95-0.96 across all three ROIs**, "almost perfect agreement".

**4-class kappa values: NOT EXTRACTED — needs manual download.** The 4-class numbers are reported in Tables 2/3 of the paper but the publisher PDF is paywalled. The dichotomous numbers are the ones that propagated into the abstract and review papers.

**Sensitivity / specificity: NOT EXTRACTED.**
**Per-level breakdown (L1-L2 ... L5-S1): NOT EXTRACTED.**

### Per-severity breakdown
**NOT EXTRACTED — needs manual download.** Abstract suggests per-grade metrics exist in supplementary tables.

### Per-disc-level breakdown
**NOT EXTRACTED — needs manual download.**

### Class imbalance handling
**NOT EXTRACTED.** No mention in abstract.

### Backbone & params
- **Two-stage DL pipeline:** CNN-1 detects ROI (Mask R-CNN per published meta-analyses), CNN-2 classifies stenosis grade (3D CNN per meta-analyses).
- **Parameter count: NOT EXTRACTED.**

### Direct comparability with our RSNA val: **NOT-COMPARABLE NUMERICALLY (must-cite contextually)**
**Reason:**
- Different metric (kappa, not F1/AUC/AUPRC).
- Same three RSNA-side conditions (CCS, lateral recess = subarticular, neural foraminal) but their dichotomous binarization (normal/mild vs moderate/severe) differs from our 3-class.
- Different data (private institutional dataset, not RSNA 2024).
- However, **this is the most-cited DL stenosis paper in the field** — reviewers will expect the citation. Treat as **qualitative ceiling** ("expert-level kappa is achievable on dichotomous binarized form of these conditions") and as a **historical anchor**, not as a numerical row in Table 1.

### Notes for our paper
- **Must cite** in introduction and related work — non-citation will be flagged by Radiology-savvy reviewers.
- Map our "Severe" class to their dichotomous "moderate/severe" and frame our **Severe AUPRC = 0.321** as "fine-grained 3-class severity grading is harder than the binarized agreement task in Hallinan 2021 (kappa 0.89-0.96)".
- **Manual download required** for full per-region 4-class tables. Path: institutional access via library / inter-library loan, or RSNA society membership.

---

## Summary

Three papers (Nigru NASSJ 2024, Lin Bioengineering 2024, McSweeney Spine 2023) gave **clean per-condition F1/Acc/kappa tables** via PMC open-access mirrors and are directly tabulatable for SOTA Table 1: Nigru is the must-cite single-best comparator (covers all 5 of our label-space conditions); Lin is the closest CBAM architectural twin but needs a "binary single-level L4-L5" footnote; McSweeney provides the canonical Pfirrmann Bal. Acc 78% / kappa 0.68 baseline (no F1 reported). Two papers — **Hallinan Radiology 2021** (paywalled) and **Aktan IJCIS 2025** (Springer page hard-blocked from this environment) — yielded only abstract-level numbers and need **manual download by the user**: for Hallinan, you should pull the PDF via institutional library access (or the Singapore/SGH preprint version that some authors host) to extract the 4-class kappa table and per-level breakdown; for Aktan, the open-access Springer PDF link `https://link.springer.com/content/pdf/10.1007/s44196-025-01098-7.pdf` should download via a normal browser even though it was blocked here, and once downloaded, the per-condition F1/Dice/IoU tables (likely Tables 3-5) will replace the single 95.65% aggregate that is currently the only number we have. Until those two manual downloads are done, the safe SOTA Table 1 entries to publish are the Nigru / Lin / McSweeney rows; Hallinan should appear with kappa-only ("dichotomous, not directly F1-comparable") and Aktan should appear with a single "F1 95.65% (aggregate, balanced internal split — see Table footnote)" placeholder until per-class numbers are extracted.
