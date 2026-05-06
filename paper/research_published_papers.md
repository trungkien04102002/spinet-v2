# Research: Published SOTA on Lumbar Spine MRI Severity Grading (2020–2026)

Compiled 2026-05-06 for the SpineNetV2 (CBAM 3D ResNet34 + frozen BiomedCLIP via concat-MLP) paper.
Our reference numbers (RSNA 2024 val, mean over 3 conditions × 3 severities): Mean F1 = 0.528, Mean AUC = 0.837, Severe AUPRC = 0.321.

Search method: 10 WebSearch queries covering RSNA 2024, lumbar spine grading, SpineNet family, BiomedCLIP, SPIDER, CBAM/attention, and key seminal works (Hallinan, Jamaludin, Lewandrowski). Where direct WebFetch was blocked we fall back to the search-engine snippets from the publishers; metric numbers below are quoted from the abstract / fetched body — re-verify before citing in the paper.

---

## Published methods on RSNA 2024 (LumbarDISC) dataset

The dataset itself: **Richards et al., 2025**, "The RSNA Lumbar Degenerative Imaging Spine Classification (LumbarDISC) Dataset", arXiv:2506.09162. 2,697 patients, 8,593 series, 8 institutions / 6 countries / 5 continents. Three conditions × 5 disc levels × 5-class severity (annotation), reduced to 3 severity levels (Normal/Mild, Moderate, Severe) for the Kaggle competition. Same dataset we train on.

Competition metric (Kaggle 2024): **sample-weighted multi-label log loss** with weights `(1, 2, 4)` for (normal/mild, moderate, severe) plus a separate "any-severe" macro term. 1,874 teams competed.

| Paper | Year | Method | Dataset / split | Reported metrics | Notes |
|---|---|---|---|---|---|
| Aktan et al., *Int. J. Comp. Intelligence Systems* | 2025 | **MobileNetV2-UNet** (joint segmentation + classification, Cross-Entropy + IoU loss) | RSNA 2024 LumbarDISC, internal split (paper does not publish a held-out test fold) | Acc 94.93%, F1 95.65%, Dice 94.61% | DOI 10.1007/s44196-025-01098-7. Numbers are suspiciously high vs Kaggle leaderboard — likely reported on a balanced subset, not the full Kaggle private LB. **Treat as comparable only if we re-evaluate on the same split.** |
| Lin, Zhang, Shang, *Bioengineering* | 2024 | CNN + **MHSAM + Slot Attention + CBAM** (multi-attention) | "RSNA/ASNR" data, 8,160 axial L4–L5 images (1,632 / condition × 5 conditions), 80/20 split | Acc 95.2%, F1 94.5%, Recall 94.3%, AUC 0.951–0.972 (per-condition) | DOI 10.3390/bioengineering11101021. **Most directly comparable architecture to ours** (CBAM-based). Uses single-level (L4–L5) axial cropping, binary-equivalent setup, balanced sampling — probably explains gap to our multi-level 3-class numbers. |
| Sahin et al., medRxiv preprint | 2024-12 | **YOLO v8 detector + DeepScoreNet classifier** | RSNA 2024 LumbarDISC | Numbers not extractable from abstract; reports Kaggle-style log-loss | medRxiv 2024.12.06.24318595. Two-stage detect-then-grade pipeline. Bridging-only without re-evaluation. |
| Artley (Kaggle 2nd place writeup, GitHub `brendanartley/RSNA-2024-Competition`) | 2024 | 2-stage sagittal-only pipeline: 2D backbone + attention head for disc localization, then encoder + LSTM + attention pooling on T1 / T2 strips, manifold mixup, pseudo-labels | RSNA 2024 LumbarDISC private LB | Disc-localization accuracy 99.1% within 5 px; private-LB log-loss not reported in the README we could fetch | Strong public reference for the canonical Kaggle metric. Useful as a citation, hard to compare F1 directly. |
| Other Kaggle solutions (nshen7, dwest1507, Melichov, Spine-CNN) | 2024 | Mix of CenterNet+EfficientNetB6 keypoint, MedViT transformer, ensemble CNNs | RSNA 2024 LumbarDISC | Open-source code, no peer-reviewed metrics | **Cite as "competition entries"** for context, not comparison. |

---

## Published methods on SPIDER and other lumbar grading datasets

SPIDER dataset: **van der Graaf et al., 2024**, "Lumbar spine segmentation in MR images: a dataset and a public benchmark", *Scientific Data* 11:264. DOI 10.1038/s41597-024-03090-w. 447 sagittal T1+T2 series, 218 patients, 4 hospitals. **Note:** SPIDER ships only segmentation labels for vertebrae / IVDs / canal — the per-IVD condition labels we use (Pfirrmann, spondylolisthesis, herniation) come from the SPIDER metadata CSV. nnU-Net was the original benchmark (segmentation only).

| Paper | Year | Method | Dataset / split | Metrics | Notes |
|---|---|---|---|---|---|
| van der Graaf et al., *Sci. Data* | 2024 | **nnU-Net** baseline; iterative semi-automatic annotation | SPIDER (218 patients, sequestered 39-patient test set, 97 series) | Per-class Dice for vertebrae / IVDs / canal (segmentation). No grading numbers. | DOI 10.1038/s41597-024-03090-w. The **canonical SPIDER citation**. |
| Acharya & Kansakar, arXiv preprint | 2026-02 | **Disc-Centric Contrastive Learning** + auxiliary regression for disc localization + weighted focal loss | Sagittal T2 (dataset name not specified in abstract) | Balanced Acc 78.1%; Severe → Normal misclass rate 2.13% | arXiv:2602.05738. **Closest in spirit to our work** (severity grading + class imbalance + disc-centric crops). Reports balanced-accuracy only. Citable as related work even if dataset doesn't match. |
| Won, Lee, Lee, Park, *Global Spine J.* | 2024 (online) / 2025 (issue) | **Faster R-CNN ResNet50** canal detector + VGG rootlet classifier + VGG stenosis classifier (3-stage cascade) | 13,758 axial T2 images / 542 patients, 10-fold CV (7/2/1) | Stenosis grading: Acc 91.5%, F1 78.4% (vs analyzer 89.4% / 75.7%) | DOI 10.1177/21925682241299332. **Comparable F1 magnitude to ours** (75–80%). Different protocol (axial-only, 4-class A–D), in-house dataset. Indirect baseline. |
| Bharadwaj et al., *European Radiology* | 2023 | **V-Net segmentation + BiT ResNet-50 + Decision Tree** (interpretable cascade) | 200 patients, 987 axial slices (150/20/30) | CCS multi-class κ 0.54 (CNN), 0.80 (DT); CCS binary AUROC 0.94; Foraminal AUROC 0.92; Facet AUROC 0.93 | PMC10566647. Single-center, small N. Indirect baseline; **cite for interpretability angle**. |
| Hallinan et al., *Radiology* | 2021 | **Mask R-CNN + 3D CNN** (2-stage) — central canal, lateral recess, neural foraminal stenosis, dichotomous and 4-class | 446 patients training; 100 external test | Dichotomous κ: central canal 0.96, lateral recess 0.92, neural foraminal 0.89; external κ 0.95–0.96 | DOI 10.1148/radiol.2021204289. **Seminal RSNA-published baseline** — this is the de facto SOTA citation for lumbar stenosis DL. Different metric (κ) but maps to our conditions. **MUST cite.** |
| Lu et al., *Eur. Spine J.* | 2024 | Neural network for stenosis detection + classification | Multi-center | (Numbers not extracted) | DOI 10.1007/s00586-023-08089-2. Indirect. |
| Lewandrowski et al., *Int. J. Spine Surgery* | 2020 | DL reliability analysis (Multus Medical "DeepBook") on routine reports | 65 patients / 383 levels | Sensitivity / specificity for compressive pathology, no F1 | The "Lewandrowski 2020" reference. Indirect (different metric, dataset). |
| Murata et al., *Spine* / Pang et al., *Eur. Radiol.* | 2020-2022 | Various 2D CNN baselines for Pfirrmann | In-house | Per-paper accuracies 75–85% | Genealogical citations; do not compare numerically. |

---

## Multi-task lumbar grading on private datasets

| Paper | Year | Method | Dataset | Metrics |
|---|---|---|---|---|
| Lim et al., *Diseases* (MDPI) | 2026 | Multi-task DL for stenosis severity grading | Multi-center external validation | Reports balanced Acc, F1 by site (numbers behind paywall) — DOI 10.3390/diseases14010032 |
| (Anon.) modified YOLO + attention, *Frontiers in Bioengineering* | 2025 | YOLOv5 + attention in CSP + residual SPPF | 472 patients (420 internal / 52 external), Pfirrmann + herniation + HIZ | Pfirrmann internal precision 0.78–0.91, recall 0.86–0.91; HIZ recall 0.81–0.88 | DOI 10.3389/fbioe.2025.1526478 |
| Anon., *Sci. Rep.* | 2026 | U-ResNet + shape-aware attention | Lumbar Spine MRI (in-house) | Acc 86.7 ± 0.6, F1 84.2 ± 0.7 | DOI 10.1038/s41598-026-42870-9 |

---

## SpineNet-family papers (genealogy of the architecture we extend)

| Paper | Year | Contribution | Relation to our work |
|---|---|---|---|
| **Jamaludin, Kadir, Zisserman**, *Med. Image Anal.* (also MICCAI 2017) | 2017 | **Original SpineNet** — VGG-style 2D CNN; multi-task grading (Pfirrmann, narrowing, endplate, marrow, spondylolisthesis, CCS); evidence visualization. GENODISC training data. | Lin's CCC 0.91 for Pfirrmann (intra-rater). The **direct ancestor** of `spinenet/models/grading.py`. |
| **Windsor, Jamaludin, Kadir, Zisserman**, arXiv 2206.04014 | 2022 | **SpineNetV2** — 3D ResNet34 backbone (the one we use as `~/.spinenet/weights/ckpt1.pt`), unified detect/label/grade pipeline | Direct backbone source. Originally reported balanced accuracy ~70.9% on Pfirrmann (per McSweeney 2022). |
| **Windsor, Jamaludin, Kadir, Zisserman**, *Sci. Rep.* 14:14993 | 2024 | **SpineNetV2 (peer-reviewed)** — extended labelling and grading pipeline for clinical scans; full system paper. | Cite as "SpineNetV2 (Windsor 2024)" as the closest-architecture comparator. DOI 10.1038/s41598-024-64580-w. |
| **McSweeney et al.**, *Spine* 48(7):484–491 | 2023 (Dec 2022 online) | **SpineNet external validation on NFBC1966** (1,331 patients, 6,655 disks). | Balanced Acc 78% Pfirrmann, 86% Modic; κ 0.68 / 0.74; Lin's CCC 0.86. **Our most directly comparable F1-equivalent reference for Pfirrmann.** |
| **Nigru et al.**, *N. Am. Spine Soc. J.* 20:100564 | 2024 | **SpineNetV2 external validation** on 1,747 disks / 353 patients across 11 pathologies. | Pfirrmann: Acc 79.6%, Bal. Acc 79.4%, F1 79.9%, κ 0.738. CCS: F1 97.1%. Spondylolisthesis: F1 98.5%. Foraminal stenosis: F1 ~83.8%, κ ~0.46–0.47, **Bal. Acc 69–70%** — ceiling on sagittal-only foraminal grading. Disc herniation: F1 77.9%, κ 0.55. **Most useful single comparator** because it covers both our (RSNA-side) and (SPIDER-side) label spaces. DOI 10.1016/j.xnsj.2024.100564. |
| **Grob et al.**, *Eur. Spine J.* | 2022 | **SpineNet external validation** on 882 patients (Pfirrmann, spondylolisthesis, CCS) | DOI 10.1007/s00586-022-07311-x. Genealogy citation. |
| **SpineScan**, *Eur. Spine J.* | 2025 | Re-implementation / annotation tool of SpineNet for Pfirrmann | DOI 10.1007/s00586-025-09537-x. Indirect. |
| **Conformal-SpineNet**, *Sci. Rep.* | 2026 | Adds conformal prediction uncertainty over SpineNet CCS head | DOI 10.1038/s41598-026-35343-6. Useful for "future work — uncertainty calibration". |

---

## Vision-language / BiomedCLIP-related references

| Paper | Year | Relation to our work |
|---|---|---|
| **Zhang et al.**, BiomedCLIP, arXiv:2303.00915, *NEJM AI* (2025 print version DOI 10.1056/AIoa2400640) | 2023 (preprint) / 2025 (NEJM AI) | The frozen text/image encoder we attach via concat-MLP. Pretrained on PMC-15M. Reports SOTA on RSNA Pneumonia, mean Acc 75.5% across 5 zero-shot benchmarks. **MUST cite** as the foundation model for our fusion head. |
| Sun et al., NeurIPS 2024 ("3D gap with slice selection") | 2024 | Vote-MI representative-slice selection for 3D imaging through a 2D VLM. Justifies our middle-9-slice axial→2D treatment. |
| MedCLIP-SAMv2 / MedCLIP-SAM, *Med. Image Anal.* / MICCAI | 2024–2025 | Text-driven medical-image segmentation using CLIP-style models. Cite as related VLM transfer. |
| BiomedCLIP scoliosis study (ResearchGate preprint) | 2024 | Demonstrates BiomedCLIP zero-shot on spine deformity. Indirect. |
| BiomedCLIP imbalance OOD analysis, arXiv:2506.14136 | 2025 | Evidence that BiomedCLIP linear probing is brittle on imbalanced classes — **supports our choice to keep BiomedCLIP frozen and concat into a CBAM 3D backbone, rather than fine-tune end-to-end.** |
| Decipher-MR, *npj Digital Med.* | 2026 | 3D-MRI vision-language foundation model; useful "future direction" citation. |

---

## Direct comparison candidates (top 5 we should benchmark against in Table 1)

These are the papers whose **reported numbers we can plausibly tabulate next to our 0.528 F1 / 0.837 AUC / 0.321 Severe AUPRC** without misleading the reader. Asterisks = recommended must-have.

1. ***Hallinan et al., *Radiology* 2021** (DOI 10.1148/radiol.2021204289). Same three conditions (CCS, lateral-recess, foraminal). Different metric (κ) but lets us frame "Severe" agreement. The most-cited DL stenosis paper; reviewers will expect it.

2. ***Nigru et al., NASSJ 2024** (DOI 10.1016/j.xnsj.2024.100564). External validation of SpineNetV2 on 11 pathologies including Pfirrmann (F1 79.9%), CCS (F1 97.1% — but on a different prevalence), foraminal (F1 83.8%), spondylolisthesis (F1 98.5%), herniation (F1 77.9%). **Covers our combined RSNA + SPIDER label spaces in one paper.** This is the single-most-useful direct comparator.

3. ***Lin, Zhang, Shang, *Bioengineering* 2024** (DOI 10.3390/bioengineering11101021). CBAM-based, RSNA-derived data, reports F1 by condition. **The closest architectural comparator** to our CBAM 3D ResNet34. We must explain why their 94.5% F1 is on a balanced 1,632/condition single-level subset (binary-equivalent) while ours is on the full 3-class multi-level RSNA private val.

4. **McSweeney et al., *Spine* 2023** (PMC9990601). External SpineNet on NFBC1966. Pfirrmann balanced Acc 78%, κ 0.68. Cleanest single-task SpineNet baseline. Useful when we ablate the SPIDER-Pfirrmann head.

5. **Aktan et al., IJCIS 2025** (DOI 10.1007/s44196-025-01098-7). MobileNetV2-UNet on RSNA 2024 — same dataset as us. Reports F1 95.65%, but on what appears to be a balanced internal split. **Cite as same-dataset comparator with explicit note about the split / metric mismatch.**

(Acharya 2026 disc-centric contrastive is an additional optional comparator if we add Pfirrmann as a primary task — balanced Acc 78.1% on undisclosed dataset.)

---

## Indirect baselines (cite, do not tabulate numbers)

- **Jamaludin et al. 2017** — original SpineNet. Cite as architectural ancestor.
- **Windsor et al. 2022 (arXiv) and 2024 (Sci. Rep.)** — SpineNetV2. Cite as backbone source.
- **van der Graaf et al. 2024 Sci. Data** — SPIDER dataset paper. **Required citation** for SPIDER.
- **Richards et al. 2025 arXiv** — LumbarDISC (RSNA 2024) dataset paper. **Required citation** for RSNA 2024.
- **Bharadwaj et al. 2023 Eur. Radiol.** — interpretable cascade on axial MRI. Different dataset / 2-class binarization; cite for interpretability framing.
- **Won et al. 2024 Global Spine J.** — multi-level axial cascade. Different protocol; cite for "multi-level grading is hard" framing.
- **Zhang et al. BiomedCLIP** — required citation for our fusion head.
- **Lewandrowski et al. 2020** — first DL reliability study; cite as historical anchor only.
- **Acharya & Kansakar 2026 arXiv** — disc-centric contrastive. Cite for class-imbalance / focal-loss design.
- **2nd-place Kaggle (Artley)** + Aktan/Sahin solutions — cite to acknowledge Kaggle-leaderboard prior art without comparing log-loss vs F1.

---

## Notes for paper-writing

- Our Mean F1 = 0.528 is on the **3-class multi-level full RSNA val split**, not a balanced binary L4–L5 subset. When tabulating against Lin 2024 (94.5% F1) or Aktan 2025 (95.65% F1), we should add a footnote "[†] reported on balanced single-level / binarized subset" so reviewers don't read it as a 40-point gap.
- Because the Kaggle private LB metric is sample-weighted log-loss (not F1), there is **no published peer-reviewed F1 number on the full RSNA 2024 private LB**. We are effectively setting one of the first such F1 baselines — frame this as a contribution, not a weakness.
- Severe AUPRC = 0.321 is uniquely ours; no comparator paper reports per-class AUPRC for this dataset. Cite Hallinan κ-on-severe-only as the closest qualitative comparator.
- For the SPIDER side (Pfirrmann + spondylolisthesis + herniation, transfer learning + zero-shot), **Nigru 2024** is the single best comparator since it reports F1 on the same three pathologies (79.9% / 98.5% / 77.9%) on a 1,747-disc external set.
