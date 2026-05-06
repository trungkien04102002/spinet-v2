# Paper Citations Lookup

Flat list of all papers referenced for the SpineNetV2 + CBAM + BiomedCLIP paper.
Sorted by category. Each entry: `[Authors] [Year] — [Short title] — [Venue] — [DOI/URL] — [Key numbers]`.

Compiled 2026-05-06.

---

## A. Direct comparators (Table 1 candidates)

| # | Citation | Venue | Link | Key numbers (relevant to ours) |
|---|---|---|---|---|
| A1 | **Nigru, Benini, Bonetti et al. 2024** — *External validation of SpineNetV2 on lumbosacral disc pathologies* | N. Am. Spine Soc. J. 20:100564 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11617751/ · DOI 10.1016/j.xnsj.2024.100564 | Pfirrmann F1 0.796, foram. F1 0.838 (κ 0.46-0.47), spond. F1 0.985, hern. F1 0.779, CCS F1 0.971; 1,747 disks/353 pat. |
| A2 | **Lin, Zhang, Shang 2024** — *Multi-attention CNN for lumbar stenosis MRI* | Bioengineering 11(10):1021 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11504910/ · DOI 10.3390/bioengineering11101021 | Acc 95.2%, F1 94.5%, AUC 0.951-0.972; balanced binary L4-L5 only |
| A3 | **Aktan/Khan et al. 2025** — *Comprehensive Eval Based on RSNA 2024 (MobileNetV2-UNet)* | Int. J. Comp. Intell. Sys. | https://link.springer.com/article/10.1007/s44196-025-01098-7 · DOI 10.1007/s44196-025-01098-7 | F1 95.65%, Acc 94.93%, Dice 94.61% (aggregate, balanced internal split — needs PDF download) |
| A4 | **McSweeney, Tiulpin, Saarakkala et al. 2023** — *External validation of SpineNet on NFBC1966* | Spine 48(7):484-491 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9990601/ · DOI 10.1097/BRS.0000000000004572 | Pfirrmann Bal. Acc 78%, κ 0.68, Lin's CCC 0.86; 1,331 pat./6,655 discs |
| A5 | **Hallinan, Zhu et al. 2021** — *DL stenosis detection on lumbar MRI (Radiology)* | Radiology 300(1):130-138 | https://pubs.rsna.org/doi/10.1148/radiol.2021204289 · DOI 10.1148/radiol.2021204289 | Dichotomous κ: CCS 0.96, lat. recess 0.92, foramina 0.89 (internal); 0.95-0.96 (external 100 pat.) |
| A6 | **Hong et al. 2025 — M-SCAN** — *Multi-view cross-attention for canal stenosis grading* | arXiv 2503.01634 | https://arxiv.org/abs/2503.01634 | AUROC 0.971 canal stenosis on RSNA 2024 (1,975 studies) |
| A7 | **Phaphuangwittayakul et al. 2026** — *Multi-Task DL with multi-center external validation* | Diseases 14(1):32 (MDPI) | https://www.mdpi.com/2079-9721/14/1/32 | F1 0.9558 ext. (Phayao Hospital); AUC 0.994-1.000 |
| A8 | **Wang et al. 2025** — *Lumbar stenosis MRI: CNN vs Transformer benchmarking* | Eur. Spine J. | https://link.springer.com/article/10.1007/s00586-025-09443-2 | κ canal 0.97-0.99, lat. recess 0.81-0.94, foramina 0.91-0.95; 564 MRIs |
| A9 | **Liu et al. 2025** — *External validation of SpineNetv2 multi-pathology* | Eur. Spine J. | https://link.springer.com/article/10.1007/s00586-025-09543-z | Pfirrmann + CCS + spond. + hern. + bilat. foram.; 491 pat./2,455 discs |
| A10 | **Liu et al. 2025 — AFFM-YOLOv8** — *MRI grading of lumbar disc herniation (MSU 11-subtype)* | Sci. Rep. | https://www.nature.com/articles/s41598-025-18417-9 | F1 91.01%, +3% recall, κ > 0.9; 8,428 pat./100k axials |
| A11 | **Liu et al. 2025 — YOLOv9-AID** — *Pfirrmann + hern. + HIZ + Schmorl multi-task* | Front. Bioeng. Biotechnol. | https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1626299/full | mAP50 82.8%, +5% recall vs YOLOv9; 1,100 MRIs |
| A12 | **Zhang et al. 2025** — *Multi-label 7-lesion classifier (nnUNet + binary + multi-label)* | Biomed. Sig. Proc. Ctrl. (S1746809425005130) | https://www.sciencedirect.com/science/article/pii/S1746809425005130 | Internal Acc 0.890 / F1 0.867 / Fleiss κ 0.83-0.86; ext. Acc 0.879 / F1 0.760 |
| A13 | **Magnani et al. 2026** — *Explainable Pfirrmann classification (8-level + saliency + rejection)* | Biomed. Sig. Proc. Ctrl. (S1746809426007317) | https://www.sciencedirect.com/science/article/abs/pii/S1746809426007317 | Acc 93.2% (±1 grade), 96.8% with rejection |
| A14 | **Acharya & Kansakar 2026** — *Disc-Centric Contrastive Learning* | arXiv 2602.05738 | https://arxiv.org/abs/2602.05738 | Bal. Acc 78.1%; Severe→Normal misclass 2.13% |
| A15 | **Bharadwaj et al. 2023** — *Interpretable cascade (V-Net + BiT ResNet-50 + Decision Tree)* | Eur. Radiol. (PMC10566647) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10566647/ | CCS κ 0.54-0.80, AUROC 0.94 (CCS), 0.92 (foram.), 0.93 (facet); 200 pat. |
| A16 | **Won, Lee, Lee, Park 2024** — *3-stage cascade Faster R-CNN + VGG* | Global Spine J. | https://journals.sagepub.com/doi/10.1177/21925682241299332 | Acc 91.5%, F1 78.4%; 13,758 axial T2/542 pat. |

---

## B. Kaggle RSNA 2024 leaderboard (top solutions)

| # | Citation | Rank | Link | Key info |
|---|---|---|---|---|
| B1 | **RSNA AI Challenge 2024 — Winners** | — | https://www.rsna.org/news/2024/november/2024-ai-challenge-winners | Official ranks: 1=Avengers, 2=ianpan-Kevin-Yuji-Bartley, 3=SonySpine s & tkmn & Moyashii, 4=SPINE CHART, 5=Two People, etc. |
| B2 | **ianpan + Kevin + Yuji + Bartley** — *2nd place solution* | 2 | https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution | 2D ConvNeXt+DaViT+LSTM+attn ensemble; pseudo-labels; 9-rotation TTA; weighted ensemble (2*Yuji+2*Ian+Bartley)/5 |
| B3 | **Bartley GitHub** | 2 | https://github.com/brendanartley/RSNA-2024-Competition | Public code |
| B4 | **Yuji GitHub** | 2 | https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification | YOLOX+axial T2 pipeline |
| B5 | **SPINE CHART (tattaka + yu4u)** — *4th place solution* | 4 | https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public | Only fully-3D top-10; CAFormer+ConvNeXt+ResNetRS50+SwinV2; MIT license |
| B6 | **SonySpine s & tkmn & Moyashii** — *3rd place* | 3 | https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution | Educational Merit Award |
| B7 | **Two People** — *5th place* | 5 | https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution | — |
| B8 | **nshen7** — *Independent solution (not prize-winning, but with per-class F1)* | n/a | https://github.com/nshen7/rsna-2024 | Faster R-CNN + Swin 2D; **Severe F1 0.5007**, Mod F1 0.4715, Norm F1 0.9063 |
| B9 | **Max Melichov** — *EdgeNeXt baseline writeup* | n/a | https://medium.com/@maxme006/rsna-2024-a-story-of-solving-low-back-pain-with-ai-687a66b8e209 | Private LB 0.55 single-backbone |
| B10 | **medRxiv 2024 — YOLOv8 + DeepScoreNet** | n/a | https://www.medrxiv.org/content/10.1101/2024.12.06.24318595v1 | RSNA 2024 preprint |

---

## C. Datasets

| # | Citation | Venue | Link | Description |
|---|---|---|---|---|
| C1 | **Richards/Tushar et al. 2025** — *RSNA LumbarDISC dataset paper* | arXiv 2506.09162 | https://arxiv.org/abs/2506.09162 | 2,697 patients, 8,593 series, 8 institutions / 6 countries / 5 continents; 3 conditions × 5 disc levels × 5-class (reduced to 3) |
| C2 | **van der Graaf et al. 2024** — *SPIDER lumbar segmentation dataset* | Sci. Data 11:264 | https://www.nature.com/articles/s41597-024-03090-w · DOI 10.1038/s41597-024-03090-w | 447 sagittal T1+T2 series, 218 patients, 4 hospitals; segmentation labels for vertebrae/IVDs/canal |
| C3 | **Sci. Data 2026** — *Medical Spine Sagittal MRI Dataset for Segmentation and Foraminal Stenosis* | Sci. Data | https://www.nature.com/articles/s41597-026-07138-x | 500 patients sagittal lumbar; bbox + clinical severity per level (left/right) |
| C4 | **Mendeley dataset (k57fr854j2/2)** | Mendeley Data | https://data.mendeley.com/datasets/k57fr854j2/2 | Lumbar MRI free-text radiologist notes (text-based, not structured labels) |

---

## D. SpineNet family (genealogy)

| # | Citation | Venue | Link | Description |
|---|---|---|---|---|
| D1 | **Jamaludin, Kadir, Zisserman 2017** — *SpineNet original (VGG 2D)* | Med. Image Anal. | DOI 10.1016/j.media.2017.07.002 | Original VGG 2D multi-task SpineNet |
| D2 | **Windsor, Jamaludin, Kadir, Zisserman 2022** — *SpineNetV2 (3D ResNet34)* | arXiv 2206.04014 | https://arxiv.org/abs/2206.04014 | 3D ResNet34 backbone, unified detect/label/grade; what we use as `~/.spinenet/weights/ckpt1.pt` |
| D3 | **Windsor et al. 2024** — *SpineNetV2 peer-reviewed paper* | Sci. Rep. 14:14993 | https://www.nature.com/articles/s41598-024-64580-w · DOI 10.1038/s41598-024-64580-w | Direct backbone source for our work |
| D4 | **McSweeney et al. 2023** — *SpineNet ext. val. NFBC1966* | Spine | (see A4) | Pfirrmann Bal. Acc 78%, κ 0.68 |
| D5 | **Nigru et al. 2024** — *SpineNetV2 ext. val.* | NASSJ | (see A1) | All 11 pathologies |
| D6 | **Liu et al. 2025** — *SpineNetv2 ext. val.* | Eur. Spine J. | (see A9) | 491 pat./2,455 discs |
| D7 | **Joliat et al. 2025** — *SpineNet 14-yr follow-up* | Eur. Spine J. | https://link.springer.com/article/10.1007/s00586-025-08900-2 | Longitudinal cohort |
| D8 | **Grob et al. 2022** — *SpineNet ext. val. 882 pat.* | Eur. Spine J. | DOI 10.1007/s00586-022-07311-x | Pfirr.+spond.+CCS |
| D9 | **Patarroyo et al. 2025 — SpineScan** | Eur. Spine J. (s00586-025-09537-x) | https://link.springer.com/article/10.1007/s00586-025-09537-x | YOLOv8x re-implementation; per-level Acc 0.78-0.82 |

---

## E. Foundation models (BiomedCLIP + alternatives)

| # | Citation | Venue | Link | Description |
|---|---|---|---|---|
| E1 | **Zhang et al. 2023 — BiomedCLIP** | arXiv 2303.00915 / NEJM AI 2024 | https://arxiv.org/abs/2303.00915 · DOI 10.1056/AIoa2400640 | The frozen text/image encoder we use; PMC-15M pretrain; ViT-B + PubMedBERT |
| E2 | **Lin et al. 2023 — PMC-CLIP** | MICCAI / arXiv 2303.07240 | https://arxiv.org/abs/2303.07240 | PMC-OA (1.6M pairs); +8.1% R@10 retrieval |
| E3 | **Lu et al. 2024 — RadCLIP** | arXiv 2403.09948 | https://arxiv.org/abs/2403.09948 | Slice-pooling adapter for volumetric; closest VLM with built-in 3D |
| E4 | **Wang et al. 2022 — MedCLIP** | EMNLP 2022 | https://www.researchgate.net/publication/372919441 | Chest X-ray focus |
| E5 | **Yang et al. 2026 — Decipher-MR** | npj Digital Med. / arXiv 2509.21249 | https://www.nature.com/articles/s41746-026-02596-4 · https://arxiv.org/abs/2509.21249 | Native 3D-MRI VLM; 200K series |
| E6 | **Blankemeier et al. 2026 — Merlin** | Nature / arXiv 2406.06512 | https://www.nature.com/articles/s41586-026-10181-8 · https://arxiv.org/abs/2406.06512 | 3D abdominal CT VLM; +34.4% over BiomedCLIP on CT zero-shot |
| E7 | **Med3DVLM 2025** | arXiv 2503.20047 | https://arxiv.org/abs/2503.20047 | DCFormer + SigLIP 3D efficient |
| E8 | **MRI-CORE 2025** | arXiv 2506.12186 | https://arxiv.org/html/2506.12186v1 | 6M MRI slices vision FM |
| E9 | **MedGemma 2025** | arXiv 2507.05201 | https://arxiv.org/abs/2507.05201 | SigLIP-400M + Gemma-3 4B/27B; no MRI in pretrain |
| E10 | **Koleilat et al. 2025 — BiomedCoOp** | CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/papers/Koleilat_BiomedCoOp_Learning_to_Prompt_for_Biomedical_Vision-Language_Models_CVPR_2025_paper.pdf | Prompt-tuning BiomedCLIP +5-10% |
| E11 | **arXiv 2506.14136 — BiomedCLIP imbalance** | arXiv | https://arxiv.org/abs/2506.14136 | BiomedCLIP brittleness under high imbalance — anti-citation |

---

## F. Defense citations (frozen VLM + CNN fusion)

| # | Citation | Venue | Link | Why cite |
|---|---|---|---|---|
| F1 | **Liu et al. 2024 — Frozen VLM Breast Cancer** | IEEE-J-BHI | https://pmc.ncbi.nlm.nih.gov/articles/PMC12145120/ · https://ieeexplore.ieee.org/document/10769012/ | Direct architectural twin: frozen CLIP + lightweight clf > full FT; AUC 0.867→0.902 |
| F2 | **arXiv 2506.18434 (2025) — PEFT Prognosis Benchmark** | arXiv | https://arxiv.org/abs/2506.18434 | "Foundation models showed limited generalization, linear probing most stable" |
| F3 | **arXiv 2510.16973 — FM Medical Systematic Review** | arXiv | https://arxiv.org/pdf/2510.16973 | Recommends PEFT for low-data clinical tasks |
| F4 | **arXiv 2510.23807 — Why FM in Pathology Are Failing** | arXiv | https://arxiv.org/html/2510.23807v2 | Full FT degrades acc relative to linear probing |
| F5 | **arXiv 2505.16338 — Frozen FM + ViT Fusion (Derm)** | arXiv | https://arxiv.org/html/2505.16338 | Frozen FM + non-linear MLP probe > end-to-end FT |
| F6 | **CLIP in Medical Imaging Survey 2025** | Med. Image Anal. / arXiv 2312.07353 | https://arxiv.org/abs/2312.07353 | Supports CLIP-style backbones as transferable priors |

---

## G. Anti-citations (VLMs alone insufficient)

| # | Citation | Venue | Link | Why cite |
|---|---|---|---|---|
| G1 | **Hu et al. 2025 — VLM Neurorad Accuracy 35%** | npj Digital Med. | https://www.nature.com/articles/s41746-025-02047-6 | Best VLM 35% vs neuroradiologists 86%; hallucinations |
| G2 | **arXiv 2505.15425 — MVLM Robustness** | arXiv | https://arxiv.org/html/2505.15425v1 | MVLMs degrade severely under image corruption |
| G3 | **npj DM 2025 — VLM Artefact Robustness** | npj DM | https://www.nature.com/articles/s41746-025-02108-w | VLM acc drops 3-10% with weak artifacts; CNNs more robust |
| G4 | **MDPI Appl. Sci. 2025 — BiomedCLIP Scoliosis** | Appl. Sci. 15(1):398 | https://www.mdpi.com/2076-3417/15/1/398 | Closest spine-VLM; AUC 0.53 single-curve type — VLM alone insufficient |

---

## H. Other related work (cite for context)

| # | Citation | Venue | Link | Why cite |
|---|---|---|---|---|
| H1 | **Lewandrowski et al. 2020** | Int. J. Spine Surg. | (search "DeepBook DL reliability") | Historical anchor — first DL reliability study |
| H2 | **Esposito et al. 2025 — JOR Spine scoping review** | JOR Spine | https://onlinelibrary.wiley.com/doi/10.1002/jsp2.70113 | 569 studies, 93 grading systems — Pfirrmann used >50% |
| H3 | **Loyola et al. 2025 — DL for hern./spond. scoping review** | (PubMed 40252304) | https://pubmed.ncbi.nlm.nih.gov/40252304/ | Confirms spondylolisthesis sparse in 2025-26 DL |
| H4 | **Kim et al. 2025 — DL foraminal stenosis CT vs subspecialists** | Eur. Spine J. (s00586-025-09281-2) | https://link.springer.com/article/10.1007/s00586-025-09281-2 | Sagittal CT (different modality) |
| H5 | **Kim et al. 2024 — Cervical foraminal stenosis MRI** | BMC Med Imaging | https://link.springer.com/article/10.1186/s12880-024-01489-w | Cervical (related task), F1 84.8% |
| H6 | **Tabarestani et al. 2026 — Cascade DL T2 stenosis** | (PubMed 41894804) | https://pubmed.ncbi.nlm.nih.gov/41894804/ | 17,440 slices/640 pat. cascade |
| H7 | **Sun et al. NeurIPS 2024 — Vote-MI representative slice selection** | NeurIPS 2024 | (search title) | Justifies 9-slice 2D treatment of 3D volumes |
| H8 | **Beheshtian et al. 2025 — CNN Autoencoder latent disc geometry** | Ann. Biomed. Eng. (s10439-025-03840-w) | https://link.springer.com/article/10.1007/s10439-025-03840-w | Latent shape features for narrowing |
| H9 | **Li et al. 2025 — YOLOv8 LDH quantification** | La Radiologia Medica (s11547-025-01996-y) | https://link.springer.com/article/10.1007/s11547-025-01996-y | 18/8/4-class LDH taxonomies; 2,500 pat. |
| H10 | **Sun et al. 2025 — GE-YOLOv8 disc herniation** | Sci. Rep. (s41598-025-10401-7) | https://www.nature.com/articles/s41598-025-10401-7 | mAP50 +4.4% over baseline |

---

## I. Method components (architecture citations)

| # | Citation | Venue | Link | Why cite |
|---|---|---|---|---|
| I1 | **Woo, Park, Lee, Kweon 2018 — CBAM** | ECCV 2018 | https://arxiv.org/abs/1807.06521 | Channel + spatial attention module source |
| I2 | **Chen, Ma, Zheng 2019 — MedicalNet** | arXiv 1904.00625 | https://arxiv.org/abs/1904.00625 | Pretrained 3D ResNet weights for medical imaging |
| I3 | **Tang et al. 2022 — Self-Supervised Swin for 3D Medical** | CVPR 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf | Swin-UNETR pretraining (3D Transformer baseline) |
| I4 | **He et al. 2016 — ResNet** | CVPR 2016 | https://arxiv.org/abs/1512.03385 | Backbone family |
| I5 | **Lin et al. 2017 — Focal Loss** | ICCV 2017 | https://arxiv.org/abs/1708.02002 | Class imbalance loss we use |
| I6 | **Kendall, Gal 2017 — Uncertainty weighting** | NeurIPS 2017 | https://arxiv.org/abs/1705.07115 | Multi-task uncertainty loss source |
| I7 | **Selvaraju et al. 2017 — Grad-CAM** | ICCV 2017 | https://arxiv.org/abs/1610.02391 | Visualization method |

---

## J. Reviews / surveys / meta-analyses

| # | Citation | Venue | Link |
|---|---|---|---|
| J1 | **Gete et al. 2026 — DL spinal degen meta-analysis** | J. Imaging Inform. Med. | https://link.springer.com/article/10.1007/s10278-026-01897-0 |
| J2 | **Zhang et al. 2025 — AI-MRI lumbar disc degen review** | Eur. Spine J. | https://link.springer.com/article/10.1007/s00586-025-09179-z |
| J3 | **Sun et al. 2025 — AI for lumbar stenosis seg+class overview** | Eur. Spine J. | https://link.springer.com/article/10.1007/s00586-025-08672-9 |
| J4 | **AI Rev. 2025 — Predictive AI for LSS systematic review** | Artif. Intell. Rev. | https://link.springer.com/article/10.1007/s10462-025-11185-y |
| J5 | **Lim et al. 2026 — Conformal prediction for SpineNet** | Sci. Rep. | https://www.nature.com/articles/s41598-026-35343-6 |
| J6 | **Karam et al. 2025 — FM in MSK MRI clinical utility** | arXiv 2501.13376 | https://arxiv.org/html/2501.13376 |
| J7 | **Esposito 2025 — JOR Spine grading review** | JOR Spine | https://onlinelibrary.wiley.com/doi/10.1002/jsp2.70113 |

---

## Recommended priority reading order (top 10 for paper writing)

1. **Nigru et al. 2024 NASSJ** (A1) — single best comparator, must read full
2. **Lin et al. 2024 Bioengineering** (A2) — closest CBAM architectural twin
3. **Aktan 2025 IJCIS** (A3) — only peer-reviewed RSNA 2024 method paper
4. **M-SCAN arXiv 2503.01634** (A6) — direct architectural competitor on RSNA 2024
5. **McSweeney 2023 Spine** (A4) — canonical Pfirrmann SpineNet baseline
6. **Hallinan 2021 Radiology** (A5) — must-cite seminal, even if different metric
7. **Diseases 2026 multi-task** (A7) — RSNA 2024 + external validation
8. **BiomedCLIP** (E1) — required citation for our backbone
9. **Frozen VLM Breast Cancer** (F1) — strongest defense citation for our design
10. **RadCLIP** (E3) — strongest "should have tried this" future-work citation
