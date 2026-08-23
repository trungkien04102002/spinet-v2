# 2025-2026 Lumbar Spine Grading Papers — Extended Search

Goal: surface 2025-2026 papers that were missed in the first-pass SOTA review (`research_published_papers.md`). Anchor task: lumbar MRI severity grading (RSNA train, SPIDER test). Our reference numbers: Mean F1 = 0.528, Mean AUC = 0.837, Severe AUPRC = 0.321.

Already covered (do **not** re-add): Nigru 2024 NASSJ, Lin 2024 Bioengineering, Aktan 2025 IJCIS (= Springer s44196-025-01098-7, MobileNetV2-UNet on RSNA 2024), McSweeney 2023 Spine, Hallinan 2021 Radiology, Acharya 2026 arXiv (= disc-centric contrastive 2602.05738), Bharadwaj 2023 Eur Radiol, Won 2024 Global Spine J, SpineNet family (Jamaludin, Windsor 2022/2024 = CAST, Grob 2022), van der Graaf 2024 SPIDER, Richards 2025 LumbarDISC (= arXiv 2506.09162), Lewandrowski 2020.

---

## Pfirrmann disc degeneration grading (1-5)

| Paper | Year | Venue | Method | Dataset | Key metric | URL |
|---|---|---|---|---|---|---|
| Magnani et al. — *AI-based intervertebral discs grading: from unsupervised detection to explainable Pfirrmann classification* | 2026 (in press) | Biomedical Signal Processing & Control (Elsevier) | Unsupervised disc detection + CNN classifier; saliency maps; confidence-rejection mechanism; **8-level extended Pfirrmann scale** | Single-institution; CNN trained for 8-level Pfirrmann | Acc 93.2 % (within ±1 grade); 96.8 % with rejection | https://www.sciencedirect.com/science/article/abs/pii/S1746809426007317 |
| Patarroyo et al. (Russian study, RuDDS) — *SpineScan: a deep-learning model for lumbar spine MRI annotation and Pfirrmann grading* | 2025 | Eur Spine J (s00586-025-09537-x) | YOLOv8x detect-and-classify; open-source Streamlit web app | RuDDS + open-access; 484 lumbar MRIs | Per-level Acc 0.78–0.82; precision 0.75 / recall 0.808; mAP50 = 0.872 (Grade IV), 0.525 (Grade V) | https://link.springer.com/article/10.1007/s00586-025-09537-x |
| Joliat et al. — *Comparison of lumbar disc degeneration grading between SpineNet and radiologist: 14-year follow-up* | 2025 | Eur Spine J (s00586-025-08900-2) | External validation of SpineNet | 14-year longitudinal cohort | κ between SpineNet and radiologist; complements Murto 2025 progression study | https://link.springer.com/article/10.1007/s00586-025-08900-2 |
| Zheng et al. — *Modified YOLO framework for automated diagnosis & grading of lumbar IVDD* | 2025 | Front. Bioeng. Biotechnol. | YOLOv5 + SE-attention + residual block; multi-task (Pfirrmann + herniation + HIZ) | Internal lumbar MRI | Pfirrmann precision 0.78–0.91, recall 0.86–0.91; herniation 0.90–0.92 P / 0.90–0.93 R; HIZ 0.82 P / 0.81–0.88 R | https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1526478/full |
| Liu et al. — *YOLOv9-AID for simultaneous Pfirrmann + disc-herniation + HIZ + Schmorl's node detection* | 2025 | Front. Bioeng. Biotechnol. | YOLOv9 + SCSA attention + SlideLoss + ExtraDW redesign | 1,100 MRI images | mAP50 = 82.8 %; precision 80.3 %; +5 % recall vs YOLOv9 (+15 % on Schmorl's nodes) | https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1626299/full |
| Esposito et al. — *MRI-Based Grading Systems for Assessing Lumbar Disc Degeneration: A Scoping Review* | 2025 | JOR Spine | Scoping review of 569 studies, 93 grading systems | — | Pfirrmann used in >50 % of all reports; no consensus on metric reporting | https://onlinelibrary.wiley.com/doi/10.1002/jsp2.70113 |
| Beheshtian et al. — *CNN Autoencoder for Learning Latent Disc Geometry from Segmented Lumbar MRI* | 2025 | Annals of Biomedical Engineering | nnUNet seg → CNN autoencoder; latent disc shape features for narrowing prediction | 195 sagittal T1, multi-institutional public | Seg IoU 0.82; AE IoU 0.9984; latent + geometric beats either alone (no F1 reported) | https://link.springer.com/article/10.1007/s10439-025-03840-w |

## Foraminal stenosis (priority — our weakness)

| Paper | Year | Venue | Method | Dataset | Key metric | URL |
|---|---|---|---|---|---|---|
| Kim et al. — *Deep-learning lightweight model for automated lumbar foraminal stenosis classification: sagittal **CT** vs subspecialists* | 2025 | Eur Spine J (s00586-025-09281-2) | Faster R-CNN ROI + EfficientNet-B0 / MobileNetV3-Large-100 | Sagittal CT | Comparable to subspecialist radiologists (κ values reported) | https://link.springer.com/article/10.1007/s00586-025-09281-2 |
| Kim et al. — *Deep-learning model for automated detection & classification of central canal & neural foraminal stenosis on cervical MRI* | 2024 (Nov) | BMC Med Imaging (s12880-024-01489-w) | Cascaded part-specific CNNs | Cervical (related task) | Foramen ROI recall 89.3 %; F1 84.8 % vs radiologist 83.8 % | https://link.springer.com/article/10.1186/s12880-024-01489-w |
| *Medical Spine Sagittal MRI Dataset for Segmentation and Foraminal Stenosis detection* | 2026 | Sci. Data (s41597-026-07138-x) | Public dataset release: 500 patients, bbox + clinical severity per level (left/right) | 500 patients sagittal lumbar MRI | Dataset paper, no model F1 — but valuable benchmark | https://www.nature.com/articles/s41597-026-07138-x |

## Disc herniation grading

| Paper | Year | Venue | Method | Dataset | Key metric | URL |
|---|---|---|---|---|---|---|
| Liu et al. — *AFFM-YOLOv8 for MRI grading of lumbar disc herniation (MSU classification)* | 2025 | Sci. Rep. (s41598-025-18417-9) | YOLOv8 + Adaptive Feature Fusion Module; 11-subtype MSU classification | Multicenter; 8,428 patients, ~100,000 axial MRIs | F1 91.01 % (+3.05 % over baseline); +3 % recall; physician-level concordance κ > 0.9 | https://www.nature.com/articles/s41598-025-18417-9 |
| Sun et al. — *Deep-learning automatic detection & grading of disk herniation in lumbar MRI (GE-YOLOv8)* | 2025 | Sci. Rep. (s41598-025-10401-7) | YOLOv8 + Gradient-Search module + ECA attention | Pingtan Branch of Union Hospital + external test | mAP50 +4.4 % over YOLOv8 baseline; −2.1 % params; p < 0.001 | https://www.nature.com/articles/s41598-025-10401-7 |
| Li et al. — *Quantification and classification of LDH on axial MRI using deep-learning models* | 2025 | La Radiologia Medica (s11547-025-01996-y) | YOLOv8 detect + YOLOv8-seg + YOLOv8-pose; **18-class / 8-class / 4-class** taxonomies | 2,500 patients (2,120 train / 80 internal / 300 external) | Outperforms baselines on accuracy; severity 4-class evaluated separately | https://link.springer.com/article/10.1007/s11547-025-01996-y |

## Spondylolisthesis grading

| Paper | Year | Venue | Method | Dataset | Key metric | URL |
|---|---|---|---|---|---|---|
| Kim, Hong et al. — *Deep-learning diagnosis of lumbar spondylolisthesis using **X-ray*** | 2025 | Diagnostics (MDPI 15/16/2015) | CNN on X-ray, not MRI | X-ray cohort | Accuracy/F1 reported; X-ray not MRI — bridging only | https://www.mdpi.com/2075-4418/15/16/2015 |
| Liu et al. — *MRI to digital medicine diagnosis: integrating deep learning into clinical decision-making for lumbar degenerative diseases (PP-YOLOv2)* | 2025 (Jan) | Front. Surg. (s_fsurg.2024.1424716) | PP-YOLOv2 detects normal disc / herniation / spondylolisthesis on lumbar MRI | Lumbar MRI (single-center) | mAP 90.08 % across the 3 classes; binary spondylolisthesis (no Meyerding grading) | https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2024.1424716/full |
| Loyola et al. — *Deep-learning algorithms to assist in imaging diagnosis in disc herniation or spondylolisthesis: scoping review* | 2025 (Apr) | (PubMed 40252304) | Scoping review only | — | Spondylolisthesis subtask is genuinely sparse in 2025-2026 DL literature | https://pubmed.ncbi.nlm.nih.gov/40252304/ |

## Multi-task lumbar grading

| Paper | Year | Venue | Method | Dataset | Key metric | URL |
|---|---|---|---|---|---|---|
| Phaphuangwittayakul et al. (or similar) — *Multi-Task Deep Learning Model for Automated Detection and Severity Grading of Lumbar Spinal Stenosis on MRI: Multi-Center External Validation* | 2026 | Diseases (MDPI 14/1/32) | VGG19 / ConvNeXt-Tiny / DINOv2 feature extraction + classical classifier; **trained RSNA-2024-style + external validation on 100 MRI from U. of Phayao** | RSNA 2024 + external (Phayao) | F1 0.9558 (Logistic Regression on external); AUC 0.994–1.000 across grades | https://www.mdpi.com/2079-9721/14/1/32 |
| Zhang et al. — *Deep-learning-based computer-aided diagnostic system for lumbar degenerative diseases classification using MRI* | 2025 | Biomed. Signal Process. Control (S1746809425005130) | nnUNetv2 disc localization → binary sagittal classifier → axial multi-label classifier (**7 lesion types**); reports Fleiss κ | Internal + external test set | Internal mean Acc 0.890, F1 0.867, Fleiss κ 0.830–0.860; External Acc 0.879, F1 0.760, κ 0.739–0.782 | https://www.sciencedirect.com/science/article/pii/S1746809425005130 |
| Hong et al. — *M-SCAN: Multistage Framework for Lumbar Spinal Canal Stenosis Grading using Multi-View Cross Attention* | 2025 (Mar) | arXiv 2503.01634 | Sagittal-T1 + Sagittal-T2/STIR + Axial-T2 multi-view + cross-attention; sequence backbone | 1,975 unique studies (RSNA 2024) | AUROC 0.971 for canal stenosis grading | https://arxiv.org/abs/2503.01634 |
| Wang et al. — *Deep-learning models for lumbar spinal stenosis on MRI: model comparison and clinical benchmarking* | 2025 | Eur Spine J (s00586-025-09443-2) | Head-to-head CNN vs Transformer benchmarking | 564 MRIs, 464/100 train/test | κ for canal 0.97–0.99, lateral recess 0.81–0.94, foramina 0.91–0.95; CNN > Transformer; both ≥ clinicians | https://link.springer.com/article/10.1007/s00586-025-09443-2 |
| Liu et al. — *External validation of SpineNetv2: multi-pathology diagnostic agreement* | 2025 | Eur Spine J (s00586-025-09543-z) | External validation on Pfirrmann + CCS + spondylolisthesis + herniation + bilateral foraminal stenosis | 491 patients, 2,455 disc levels | Per-pathology agreement vs expert orthopedic surgeon; junior orthopedic surgeon as comparator | https://link.springer.com/article/10.1007/s00586-025-09543-z |
| Tabarestani et al. — *A novel interpretable classification of lumbar spinal stenosis using a cascade DL approach and T2-weighted MRI* | 2026 (Mar) | (PubMed 41894804) | 3-stage DL pipeline: identify → classify → grade | 17,440 MRI slices, 640 patients | Interpretable; head-to-head vs radiologists | https://pubmed.ncbi.nlm.nih.gov/41894804/ |
| Won, Lee, Lee, Park — *Lumbar Spinal Stenosis Grading in Multi-Level MRI Using Deep CNNs* (a different paper from our existing Won 2024 GSJ paper — note the exact citation) | 2025 | Global Spine Journal (10.1177/21925682241299332) | Multi-level automatic detection + stenosis grading | 13,758 MRI / 542 patients | Inter-analyzer agreement 89.2 %, F1 76.5 %; Analyzer-classifier 91.5 / 89.4 %, F1 78.4 / 75.7 % | https://journals.sagepub.com/doi/10.1177/21925682241299332 |

## Foundation-model / fusion approaches (most relevant to our BiomedCLIP angle)

| Paper | Year | Venue | Method | Dataset | Key metric | URL |
|---|---|---|---|---|---|---|
| Karam et al. — *Clinical utility of foundation models in musculoskeletal MRI for biomarker fidelity and predictive outcomes* | 2025 | arXiv 2501.13376 | Survey/empirical evaluation of foundation models on MSK MRI (incl. spine) | MSK MRI | Argues for foundation-model fine-tuning; relevant context for our BiomedCLIP fusion | https://arxiv.org/html/2501.13376 |
| *DP-Net: Advancing Fine-Grained Spine Segmentation Through Visual-Language Model with Omni- and Pixel-Level Semantic Enhancements* | 2025 | (Springer chapter, MICCAI workshop) | Vision-language model with dual prompts, omni- and pixel-level | Spine MRI segmentation | Bridging-only — segmentation not classification | https://link.springer.com/chapter/10.1007/978-981-95-5631-1_3 |
| Mostafiz et al. — *Exploring BiomedCLIP's capabilities in medical image analysis: scoliosis detection & severity assessment* | 2025 (preprint) | ResearchGate / preprint | Direct BiomedCLIP fine-tuning on spine images | Scoliosis cohort | Most directly comparable methodological paper to our BiomedCLIP fusion idea (different task) | https://www.researchgate.net/publication/387720641_Exploring_BiomedCLIP's_Capabilities_in_Medical_Image_Analysis_A_Focus_on_Scoliosis_Detection_and_Severity_Assessment |

## Reviews / meta-analyses (not new methods, but useful for citing the field's state)

| Paper | Year | Venue | URL |
|---|---|---|---|
| Gete et al. — *Diagnostic Accuracy of DL for Automated Detection of Spinal Degenerative Disease on MRI: Systematic Review & Meta-Analysis* | 2026 | J. Imaging Inform. Med. (s10278-026-01897-0) | https://link.springer.com/article/10.1007/s10278-026-01897-0 |
| Zhang et al. — *Advances and challenges in AI-assisted MRI for lumbar disc degeneration detection and classification* | 2025 | Eur Spine J (s00586-025-09179-z) | https://link.springer.com/article/10.1007/s00586-025-09179-z |
| Sun et al. — *AI for segmentation and classification in lumbar spinal stenosis: an overview* | 2025 | Eur Spine J (s00586-025-08672-9) | https://link.springer.com/article/10.1007/s00586-025-08672-9 |
| *Evaluating AI-powered predictive solutions for MRI in lumbar spinal stenosis: a systematic review* | 2025 | Artif. Intell. Rev. (s10462-025-11185-y) | https://link.springer.com/article/10.1007/s10462-025-11185-y |
| Lim et al. — *Conformal prediction for SpineNet central-canal stenosis uncertainty quantification* | 2026 | Sci. Rep. (s41598-026-35343-6) | https://www.nature.com/articles/s41598-026-35343-6 |
| *Uncertainty Quantification of CCS Deep-Learning Classifier from Lumbar Sagittal T2 MRI* | 2025 (Oct) | medRxiv 2025.10.24.25338153 | SGN balanced acc 79.4 %, macro F1 68.8 % | https://www.medrxiv.org/content/10.1101/2025.10.24.25338153v1.full |

---

## Most relevant additions to SOTA table — ranked

These are the **NEW** papers most worth including in the comparison table for our Rank-C paper (sorted by comparability with our setup: RSNA-trained, multi-pathology, severity grading, 2025-2026):

1. **Multi-Task DL on RSNA 2024 + external validation (Diseases 2026, MDPI 14/1/32)** — directly comparable: trains on RSNA 2024, externally validated on Phayao Hospital cohort; reports F1 0.9558 on external (note: dichotomous severity rather than 3-class severity, and uses VGG19/ConvNeXt-T/DINOv2 features + classical classifier — a strong head-to-head). **Highest priority.**
2. **M-SCAN (arXiv 2503.01634)** — RSNA 2024 dataset, multi-view cross-attention, AUROC 0.971 on canal-stenosis severity grading. Most direct architectural competitor to our CBAM-3D-ResNet+BiomedCLIP fusion (also fuses multi-view information).
3. **Lumbar CAD multi-label system (Biomed. Signal Proc. Control 2025, S1746809425005130)** — multi-label 7-lesion classifier with internal+external Fleiss κ. Useful "multi-task" comparator outside the RSNA challenge.
4. **AFFM-YOLOv8 (Sci. Rep. 2025)** — disc-herniation MSU 11-subtype classification with F1 91.01 %, multicenter 100k+ axial MRIs. Best new benchmark for disc-herniation subtask (our SPIDER zero-shot weakness).
5. **SpineNetv2 external validation (Eur Spine J 2025, s00586-025-09543-z)** — same task taxonomy as ours (Pfirrmann + CCS + spondylolisthesis + herniation + bilateral foraminal stenosis). Citing this is essential because SpineNet is our backbone family.
6. **Magnani et al., explainable Pfirrmann classification (BSPC 2026, S1746809426007317)** — 8-level Pfirrmann + saliency + confidence rejection. Best new Pfirrmann-only paper; matches our SPIDER Pfirrmann head.
7. **YOLOv9-AID (Frontiers Bioeng 2025)** — multi-task Pfirrmann + herniation + HIZ + Schmorl's; mAP50 82.8 %. Strong "multi-condition" comparator and mirrors our multi-task framing.

---

## Sources (URLs visited)

- https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1526478/full
- https://www.sciencedirect.com/science/article/abs/pii/S1746809426007317
- https://link.springer.com/article/10.1007/s00586-025-08900-2
- https://link.springer.com/article/10.1007/s00586-025-09537-x
- https://link.springer.com/article/10.1007/s10278-024-01251-2
- https://link.springer.com/article/10.1007/s00586-025-09179-z
- https://www.sciencedirect.com/science/article/pii/S1746809425005130
- https://link.springer.com/article/10.1007/s10439-025-03840-w
- https://www.medrxiv.org/content/10.1101/2025.02.28.25323111v1.full
- https://link.springer.com/article/10.1186/s12880-024-01489-w
- https://link.springer.com/article/10.1007/s00586-025-09281-2
- https://www.nature.com/articles/s41597-026-07138-x
- https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2025.1503625/full
- https://pubmed.ncbi.nlm.nih.gov/41894804/
- https://link.springer.com/article/10.1007/s11547-025-01996-y
- https://www.nature.com/articles/s41598-025-10401-7
- https://www.nature.com/articles/s41598-025-18417-9
- https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1626299/full
- https://www.mdpi.com/2075-4418/15/16/2015
- https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2024.1424716/full
- https://pubmed.ncbi.nlm.nih.gov/40252304/
- https://www.mdpi.com/2079-9721/14/1/32
- https://link.springer.com/article/10.1007/s44196-025-01098-7
- https://arxiv.org/abs/2506.09162
- https://arxiv.org/abs/2503.01634
- https://arxiv.org/abs/2602.05738
- https://link.springer.com/article/10.1007/s00586-025-09443-2
- https://link.springer.com/article/10.1007/s00586-025-09543-z
- https://journals.sagepub.com/doi/10.1177/21925682241299332
- https://onlinelibrary.wiley.com/doi/10.1002/jsp2.70113
- https://link.springer.com/article/10.1140/epjp/s13360-025-07148-5
- https://link.springer.com/article/10.1007/s10462-025-11185-y
- https://link.springer.com/article/10.1007/s10278-026-01897-0
- https://www.nature.com/articles/s41598-026-35343-6
- https://www.medrxiv.org/content/10.1101/2025.10.24.25338153v1.full
- https://www.medrxiv.org/content/10.1101/2024.12.06.24318595v1
- https://proceedings.aijr.org/index.php/ap/article/view/39/27
- https://arxiv.org/html/2501.13376
- https://link.springer.com/chapter/10.1007/978-981-95-5631-1_3
- https://www.researchgate.net/publication/387720641_Exploring_BiomedCLIP's_Capabilities_in_Medical_Image_Analysis_A_Focus_on_Scoliosis_Detection_and_Severity_Assessment

---

### Final summary

The **disc herniation** subtask got the strongest new finds: AFFM-YOLOv8 (multicenter 100 k axials, F1 91.01 % with MSU 11-subtype taxonomy) and the radiologia medica YOLOv8 quantification paper (4/8/18-class taxonomies on 2,500 patients) are both 2025 publications that explicitly grade severity, making them direct competitors for our SPIDER disc-herniation head. The **multi-task / RSNA-trained** axis is also rich: M-SCAN (arXiv 2503.01634) and the Diseases 2026 multi-task model with Phayao external validation are the two most-comparable papers to our pipeline. The **spondylolisthesis** subtask is genuinely sparse in 2025-2026 — what little exists is X-ray-based (Kim 2025 Diagnostics) or buried inside multi-task models (PP-YOLOv2 in Liu 2025 Front. Surg.); the Loyola scoping review (PubMed 40252304) explicitly confirms the gap. The **foraminal stenosis** subtask — where our model is weakest — has no flagship single-task 2025 lumbar-MRI paper; the strongest new external benchmark is bundled inside SpineNetv2 external validation (s00586-025-09543-z). For priority arXiv reads I recommend, in order: **2503.01634 (M-SCAN)**, **2602.05738 (Acharya disc-centric — already on your list, but the v2 may be worth re-checking)**, **2506.09162 (LumbarDISC)**, and **2501.13376 (foundation models in MSK MRI)** which directly motivates the BiomedCLIP fusion choice in our paper.
