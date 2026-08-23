# Foundation Models for Spine MRI Grading — Literature Survey

**Date:** 2026-05-06
**Author:** literature-search agent (for SpineNetV2 paper, biomedclip-integration branch)
**Purpose:** Defend the BiomedCLIP fusion choice in our Rank-C paper and identify any direct VLM-based comparator we should benchmark against.

**Our model (for reference):** CBAM 3D ResNet34 + frozen BiomedCLIP (concat-MLP fusion) for lumbar spine grading. RSNA 2024 val: Mean F1 = 0.528, Mean AUC = 0.837, Severe AUPRC = 0.321.

---

## TL;DR

After ~15 targeted web searches across 11 candidate foundation models, **no published peer-reviewed paper applies BiomedCLIP (or any of: RadCLIP, MedCLIP, MedGemma, PMC-CLIP, Decipher-MR, Merlin, Med3DVLM, MedSigLIP) to the RSNA 2024 lumbar spine task or to MRI-based Pfirrmann/disc-degeneration grading.** The only direct spine application of BiomedCLIP we found is a 2025 *Applied Sciences* paper on **scoliosis severity from posturographic X-rays** — different modality, different task, different anatomy plane — so it is *not* a head-to-head comparator. This makes our work, to the best of public knowledge, the **first reported BiomedCLIP-based pipeline for RSNA 2024 lumbar MRI degenerative classification**, which is itself a defensible Rank-C-paper-worthy contribution.

---

## 1. Direct VLM-on-spine applications (closest comparators)

| Paper | Year | VLM | Task | Modality / Dataset | Reported metrics | URL |
|---|---|---|---|---|---|---|
| Exploring BiomedCLIP's Capabilities ... Scoliosis Detection and Severity Assessment | 2025 | BiomedCLIP (fine-tuned) | Scoliosis severity (mild/mod/severe) + curve type | **Posturographic X-ray**, 262 pediatric scans | AUC 0.87 (severe), 0.75 (mild), 0.74 (moderate); type-classification weak (sens 0.35, AUC 0.53 single-curve) | [MDPI Appl. Sci. 15(1):398](https://www.mdpi.com/2076-3417/15/1/398) |
| Comparative Evaluation of Large Language and Multimodal Models in Detecting Spinal Stabilization Systems on X-Ray Images | 2025 | BiomedCLIP vs ChatGPT-4o | Implant-system identification (MCGR vs PSF) | Posturographic X-ray, 270 images | BiomedCLIP best on system-type; GPT-4o best on presence | [MDPI J. Clin. Med. 14(10):3282](https://www.mdpi.com/2077-0383/14/10/3282) |
| Evaluating Scoliosis Severity Based on Posturographic X-ray Images Using a CLIP Model | 2023 | CLIP (general) | Scoliosis severity | Posturographic X-ray | Earlier work that the 2025 BiomedCLIP study extends | [MDPI Diagnostics 13(13):2142](https://www.mdpi.com/2075-4418/13/13/2142) |
| 3D Vision-Language Models with Segmentation-Guided Multimodal Data for Spinal MRI Report Generation | 2026 | 3D-VLM (Vicuna-1.5 / 3D ViT, project SPINE) | **Report generation** (BLEU/ROUGE/METEOR/BERTScore) — *not classification* | Sagittal+axial spinal MRI (515+190 pts) | T1+T2+seg variant best | [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-07502-4_14), [code](https://github.com/serag-ai/SPINE), [T&F 2026](https://www.tandfonline.com/doi/full/10.1080/08839514.2026.2626117) |
| Advancing Fine-Grained Spine Segmentation Through VLM with Omni- and Pixel-Level Semantic Enhancements (DP-Net) | 2025 | CLIP-based dual-prompt (TIEO + TIEP) | **Segmentation** of vertebrae/IVD/canal | Spine MRI | — | [Springer](https://link.springer.com/chapter/10.1007/978-981-95-5631-1_3) |
| SpinalSAM-R1: VLM Multimodal Interactive System for Spine CT Segmentation | 2025 | SAM + LLM | **Segmentation** (CT, not MRI) | Spine CT | — | [arXiv 2511.00095](https://arxiv.org/html/2511.00095v1) |

**Observation.** Of the six spine + VLM papers found, **zero target Pfirrmann grading or RSNA-style stenosis severity from MRI**. The two BiomedCLIP-spine papers are X-ray scoliosis. The two VLM-MRI papers are report generation / segmentation. **There is no direct comparator** for BiomedCLIP-fusion lumbar MRI grading.

---

## 2. Indirect: foundation models on related medical tasks (cite for context)

### 2.1 BiomedCLIP itself
- **Zhang et al., 2023 / 2024 (NEJM AI 2024).** *BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs.* arXiv:2303.00915; NEJM AI 2024 (DOI 10.1056/AIoa2400640). Pretraining on PMC-15M, ViT-B + PubMedBERT. Linear-probe AUC on RSNA pneumonia + few-shot on PCam. Beats prior radiology-specific BioViL on RSNA pneumonia. [arXiv](https://arxiv.org/abs/2303.00915), [NEJM AI](https://ai.nejm.org/doi/abs/10.1056/AIoa2400640), [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

### 2.2 Other CLIP-style biomedical foundation models
- **Lin et al., MICCAI 2023.** *PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents.* PMC-OA (1.6M pairs); +8.1% R@10 retrieval, +3.9% accuracy on classification vs prior. [arXiv 2303.07240](https://arxiv.org/abs/2303.07240)
- **Lu et al., 2024.** *RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training.* arXiv:2403.09948. Adds slice-pooling adapter (attention) for volumetric. **Frozen text encoder, fine-tuned 2D image encoder + trained slice pooler.** Demonstrates volumetric radiology classification superiority over CLIP/BiomedCLIP on radiologic tasks (no spine specifically). [arXiv](https://arxiv.org/abs/2403.09948)
- **Wang et al., EMNLP 2022.** *MedCLIP: Contrastive Learning from Unpaired Medical Images and Text.* Chest X-ray focus; no spine application reported. [ResearchGate](https://www.researchgate.net/publication/372919441_MedCLIP_Contrastive_Learning_from_Unpaired_Medical_Images_and_Text)
- **Koleilat et al., CVPR 2025.** *BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models.* Prompt-tuning BiomedCLIP; +5–10% on CTKidney/Kvasir over baseline BiomedCLIP. [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Koleilat_BiomedCoOp_Learning_to_Prompt_for_Biomedical_Vision-Language_Models_CVPR_2025_paper.pdf)
- **arXiv 2506.14136 (2025).** *Interpreting Biomedical VLMs on High-Imbalance Out-of-Distributions: An Insight into BiomedCLIP on Radiology.* Linear-probe stability analysis of BiomedCLIP on imbalanced radiology. **Useful anti-citation: documents BiomedCLIP brittleness under heavy class imbalance** — exactly our RSNA Severe-class scenario. [arXiv](https://arxiv.org/abs/2506.14136)

### 2.3 3D / volumetric medical foundation models (potential future-work comparators)
- **Yang et al., npj Digital Medicine 2026 (arXiv 2509.21249).** *Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations.* GE HealthCare; 200K MRI series across body parts including spine. **Frozen encoder + task-specific decoders.** No published spine-grading benchmark reported. [arXiv](https://arxiv.org/abs/2509.21249), [npj DM](https://www.nature.com/articles/s41746-026-02596-4)
- **Blankemeier et al., Nature 2026 (arXiv 2406.06512).** *Merlin: A CT Vision-Language Foundation Model and Dataset.* 3D abdominal CT VLM. **Outperforms BiomedCLIP by +34.4% on certain CT tasks (zero-shot F1 0.741 vs lower BiomedCLIP).** Demonstrates the gap between 2D-VLM (BiomedCLIP) and native-3D VLM. CT only, no spine. [arXiv](https://arxiv.org/abs/2406.06512), [Nature](https://www.nature.com/articles/s41586-026-10181-8)
- **Med3DVLM (arXiv 2503.20047, 2025).** Efficient 3D medical VLM (DCFormer + SigLIP + MLP-Mixer). M3D dataset (120K 3D images). Retrieval R@1 61.0% vs SOTA M3D-LaMed 19.1%. **No Pfirrmann or RSNA spine benchmark.** [arXiv](https://arxiv.org/abs/2503.20047), [code](https://github.com/mirthAI/Med3DVLM)
- **MRI-CORE (arXiv 2506.12186, 2025).** Vision foundation model on 6M MRI slices, 110K volumes, 18 body locations. Reports +6.97% Dice on segmentation few-shot. **Segmentation-only, no grading.** [arXiv](https://arxiv.org/html/2506.12186v1)
- **MedGemma (arXiv 2507.05201, Google 2025).** SigLIP-400M medical encoder + Gemma-3 LLM (4B/27B). +15.5–18.1% on chest X-ray classification, but **trained on chest/derm/ophth/histology — no MRI/spine in pretraining.** [arXiv](https://arxiv.org/abs/2507.05201), [model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- **Vision-language foundation model for 3D medical imaging (npj AI 2025).** General 3D medical VLM survey/proposal. [npj AI](https://www.nature.com/articles/s44387-025-00015-9)

### 2.4 RSNA 2024 lumbar spine classification — prior work (CNN baselines)
- **Lumbar Spine Degenerative Classification Using YOLO v8 and DeepScoreNet** — medRxiv 2024.12.06.24318595 (2024). YOLO+CNN, no VLM. [medRxiv](https://www.medrxiv.org/content/10.1101/2024.12.06.24318595v1)
- **Automated Lumbar Spine Degenerative Classification Using Deep Learning: A Comprehensive Evaluation Based on RSNA 2024.** Int. J. Comput. Intell. Syst. (Springer 2025), DOI 10.1007/s44196-025-01098-7. MobileNetV2-UNet, **no VLM.** Reports 94.93% acc / 95.65% F1 (note: head-level only, segmentation framing). [Springer](https://link.springer.com/article/10.1007/s44196-025-01098-7)
- **RSNA LumbarDISC dataset paper (arXiv 2506.09162, 2025).** Dataset description; not method paper. [arXiv](https://arxiv.org/abs/2506.09162)
- **RSNA AI Challenge 2024 winners** — top Kaggle solutions (SonySpine, tkmn, Moyashii Avengers): all CNN/transformer ensembles (SwinUNETR, SegResNet, DynUNet, ViT). **None used BiomedCLIP or any VLM.** [RSNA results](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners), [Kaggle leaderboard](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/leaderboard)

### 2.5 Pfirrmann / disc-degeneration deep learning — recent CNN SOTA
- **Modified YOLOv5 framework, Front. Bioeng. Biotechnol. 2025.** [link](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1526478/full)
- **Graph Neural Network for Pfirrmann (Sagittal MRI), J. Imaging Inform. Med. 2025.** [link](https://link.springer.com/article/10.1007/s10278-024-01251-2)
- **Explainable Pfirrmann (8-level) framework, Biomed. Signal Process. Control 2026.** Tolerance accuracy 93.2%. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809426007317)
- **Nature Comms 2022 high-accuracy quantitation.** [link](https://www.nature.com/articles/s41467-022-28387-5)

---

## 3. Defense citations: "frozen VLM + CNN fusion is the right call"

These directly justify our architecture choice (frozen BiomedCLIP, learned CBAM-3D-ResNet34, late concat-MLP fusion):

1. **Liu et al., IEEE-J-BHI 2024/2025 — Frozen Large-Scale Pretrained Vision-Language Models are the Effective Foundational Backbone for Multimodal Breast Cancer Prediction.** Direct architectural twin: frozen CLIP + lightweight trainable classifier. AUC up from 0.867→0.902 (CBIS-DDSM val), 0.803→0.830 (test); 0.780→0.805 (EMBED). **"Frozen pretrained vision-language models alongside a lightweight trainable classifier consistently outperform conventional full finetuning methods."** This is the strongest single defensive citation. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12145120/), [IEEE Xplore](https://ieeexplore.ieee.org/document/10769012/), [TechRxiv](https://www.techrxiv.org/users/715048/articles/699670)
2. **arXiv 2506.18434 (2025) — Benchmarking Foundation Models and Parameter-Efficient Fine-Tuning for Prognosis Prediction in Medical Imaging.** Key finding: in few-shot regimes on small medical datasets, **"foundation models showed limited generalization, with linear probing yielding the most stable results"** vs full fine-tune. CNNs with full fine-tune robust on small imbalanced datasets. Justifies our hybrid: keep VLM frozen, fine-tune the CNN. [arXiv](https://arxiv.org/abs/2506.18434)
3. **arXiv 2510.16973 (2025) — Foundation Models in Medical Image Analysis: Systematic Review.** Recommends PEFT (frozen backbone + lightweight adapter/probe) for low-data clinical tasks. [arXiv](https://arxiv.org/pdf/2510.16973)
4. **arXiv 2511.01284 (2025) — Adaptation of Foundation Models for Medical Image Analysis.** Categorizes ZST / PEFT / FPFT and notes FPFT is overkill and overfits on small medical sets. [arXiv](https://arxiv.org/html/2511.01284v1)
5. **arXiv 2510.23807 (2025) — Why Foundation Models in Pathology Are Failing.** Shows full fine-tuning frequently degrades accuracy relative to linear probing due to overfitting / catastrophic forgetting. Same logic transfers to small medical-imaging targets. [arXiv](https://arxiv.org/html/2510.23807v2)
6. **arXiv 2506.14136 (2025) — Interpreting BiomedCLIP on Radiology (high-imbalance OOD).** Documents BiomedCLIP's brittleness under heavy class imbalance — explains why we add a CBAM-3D-ResNet34 with FocalLoss/oversampling on top instead of using BiomedCLIP alone. [arXiv](https://arxiv.org/abs/2506.14136)
7. **arXiv 2505.16338 (2025) — Fusion of Foundation and Vision Transformer Model Features for Dermatoscopic Image Classification.** Demonstrates "non-linear probing of frozen foundation features with MLP" beats end-to-end fine-tuning. Architectural analogue to our concat-MLP. [arXiv](https://arxiv.org/html/2505.16338)
8. **CLIP in Medical Imaging: A Comprehensive Survey (arXiv 2312.07353, Med Image Anal 2025).** General survey supporting CLIP-style backbones as transferable feature priors in radiology. [arXiv](https://arxiv.org/abs/2312.07353), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1361841525000982)
9. **Vision-language foundation models for medical imaging review (Biomed. Eng. Lett. 2025).** Reviews fusion strategies; supports concat/MLP late fusion as a standard choice. [Springer](https://link.springer.com/article/10.1007/s13534-025-00484-6)

---

## 4. Challenge citations: "newer VLMs may outperform — must acknowledge as future work"

These should appear in our Discussion / Limitations / Future Work, not in defense:

1. **Decipher-MR (npj DM 2026, arXiv 2509.21249).** Native 3D-MRI VLM, frozen encoder + task decoders. Released Sept 2025 (after our experimental cut-off). Future-work candidate. [npj DM](https://www.nature.com/articles/s41746-026-02596-4)
2. **Merlin (Nature 2026, arXiv 2406.06512).** Beats BiomedCLIP by +34.4% on CT zero-shot; argues 3D-pretraining > 2D-VLM-lifted. CT-only, but the architectural lesson transfers. We should explicitly note: *"a native 3D-MRI BiomedCLIP analogue would likely improve performance; Merlin demonstrates this for CT."* [arXiv](https://arxiv.org/abs/2406.06512)
3. **MedGemma (arXiv 2507.05201, July 2025).** Recent multimodal medical foundation; +15.5–18.1% on chest X-ray. **No spine pretraining**, so unclear advantage on lumbar MRI; but reviewers may ask. Acknowledge as untested alternative. [arXiv](https://arxiv.org/abs/2507.05201)
4. **RadCLIP (arXiv 2403.09948).** Slice-pooling attention adapter is **architecturally what we should have ideally adopted** for our 9-slice volumes. Strongest "should have tried this" citation. Future-work obligation. [arXiv](https://arxiv.org/abs/2403.09948)
5. **Med3DVLM (arXiv 2503.20047, 2025).** Native 3D efficient VLM (DCFormer+SigLIP). [arXiv](https://arxiv.org/abs/2503.20047)
6. **MRI-CORE (arXiv 2506.12186, 2025).** MRI-specific vision foundation. [arXiv](https://arxiv.org/html/2506.12186v1)
7. **BiomedCoOp (CVPR 2025).** Prompt-tuning BiomedCLIP gains +5–10% — a cheap drop-in upgrade we did not pursue. [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Koleilat_BiomedCoOp_Learning_to_Prompt_for_Biomedical_Vision-Language_Models_CVPR_2025_paper.pdf)

---

## 5. Anti-citations: "VLMs alone do not beat CNNs on small/imbalanced medical sets"

These reinforce the **hybrid CNN-foundation design**:

1. **Hu et al., npj Digital Medicine 2025 — Evaluating diagnostic accuracy of vision-language models for neuroradiological image interpretation.** Best VLM (Gemini-2.0) accuracy 35% vs neuroradiologists 86.2% on brain+spine cases. Hallucinations in 45% of cases. Strong evidence that **off-the-shelf VLMs are NOT clinically reliable on spine alone.** [npj DM](https://www.nature.com/articles/s41746-025-02047-6)
2. **arXiv 2505.15425 — On the Robustness of Medical Vision-Language Models.** MVLMs degrade severely under image corruption / domain-modality shift. [arXiv](https://arxiv.org/html/2505.15425v1)
3. **npj DM 2025 — Understanding the robustness of VLMs to medical image artefacts.** VLM accuracy drops 3–10% with weak artifacts; CNNs more robust. [npj DM](https://www.nature.com/articles/s41746-025-02108-w)
4. **Benchmarking Foundation Models for Prognosis (arXiv 2506.18434).** Already cited above; explicitly: CNN+full-FT > FM+linear-probe in some imbalanced regimes. [arXiv](https://arxiv.org/abs/2506.18434)
5. **Why Foundation Models in Pathology Are Failing (arXiv 2510.23807).** Already cited above. [arXiv](https://arxiv.org/abs/2510.23807)
6. **BiomedCLIP scoliosis paper (Appl. Sci. 2025).** AUC 0.53 on curve-type classification — direct evidence BiomedCLIP alone is insufficient for spine. [MDPI](https://www.mdpi.com/2076-3417/15/1/398)

---

## 6. Recommendation for paper's Methods/Discussion section

### Drop-in defense paragraph (Methods, Section 3.X "Fusion design rationale")

> *We integrate BiomedCLIP \cite{zhang2023biomedclip} as a frozen visual feature extractor rather than fine-tuning it end-to-end. This design follows the emerging consensus that, on small and class-imbalanced clinical datasets, freezing large-scale pretrained vision-language backbones and training only a lightweight task-specific head outperforms full fine-tuning, which tends to overfit and induce catastrophic forgetting \cite{frozen_vlm_breast2024, benchmark_fm_prognosis2025, why_fm_pathology2025}. BiomedCLIP, pretrained on 15M PubMed Central image-text pairs, is the most widely adopted general-purpose biomedical CLIP variant and has been shown to outperform radiology-specific BioViL on RSNA Pneumonia \cite{zhang2023biomedclip}. We pair the frozen BiomedCLIP visual stream with a learnable 3D-CBAM-ResNet34 \cite{cbam2018} that captures volumetric, anatomy-specific features that 2D pretrained encoders necessarily miss; the two streams are combined through late concat-MLP fusion, an architecture analogous to that of \cite{frozen_vlm_breast2024} for multimodal breast cancer prediction. Recent analyses of BiomedCLIP under high class imbalance \cite{biomedclip_imbalance2025} confirm that BiomedCLIP features alone are brittle on the long tail (relevant to our Severe class), motivating the addition of a CNN branch with focal loss and oversampling.*

### Drop-in future-work paragraph (Discussion / Limitations)

> *We acknowledge several alternative foundation models that could plausibly improve performance and that we did not benchmark. RadCLIP \cite{radclip2024} introduces a slice-pooling attention adapter natively suited to volumetric inputs and is the closest 2D-VLM analogue with a built-in 3D aggregation. Native 3D-MRI vision-language models — most notably Decipher-MR \cite{decipher_mr2026} and MRI-CORE \cite{mricore2025} — and 3D-CT analogues such as Merlin \cite{merlin2026} (which outperforms BiomedCLIP by 34% on CT zero-shot) suggest that domain-aligned 3D pretraining is a stronger prior than our 2D-lifted approach. Recent prompt-tuning of BiomedCLIP (BiomedCoOp \cite{biomedcoop2025}) reports 5–10% gains on biomedical classification with minimal compute. Finally, Google's MedGemma \cite{medgemma2025} offers a multimodal alternative, although its pretraining corpus does not include MRI. To the best of our knowledge, no public study has applied any of these models to the RSNA 2024 lumbar spine task; we leave a head-to-head comparison to future work.*

### Direct comparators for Table 1 (literature comparison)

There is **no published direct comparator** (BiomedCLIP-on-RSNA-2024 or any-VLM-on-Pfirrmann-MRI). Our Table 1 should therefore compare against:
- **Top RSNA 2024 Kaggle leaderboard solutions** (CNN/transformer ensembles — already collected in `paper/research_kaggle_top.md`).
- **Springer 2025 MobileNetV2-UNet RSNA paper** (DOI 10.1007/s44196-025-01098-7) — only published RSNA 2024 method paper.
- **medRxiv YOLOv8+DeepScoreNet (2024)**.
- For spine-MRI grading more broadly: SpineNet \cite{spinenet2024}, Pfirrmann-GNN, modified YOLOv5.

We can position ourselves as: *"the first reported BiomedCLIP-fusion pipeline for RSNA 2024 lumbar MRI degenerative classification."*

---

## 7. Sources (all URLs visited)

### BiomedCLIP & spine
- https://www.mdpi.com/2076-3417/15/1/398  — *BiomedCLIP scoliosis (closest spine-VLM)*
- https://www.mdpi.com/2077-0383/14/10/3282  — *BiomedCLIP vs GPT-4o spine implants*
- https://www.mdpi.com/2075-4418/13/13/2142  — *Pre-BiomedCLIP CLIP scoliosis*
- https://arxiv.org/abs/2303.00915  — *BiomedCLIP*
- https://ai.nejm.org/doi/abs/10.1056/AIoa2400640  — *BiomedCLIP NEJM AI 2024*
- https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

### Other CLIP-style biomedical
- https://arxiv.org/abs/2403.09948  — *RadCLIP*
- https://arxiv.org/abs/2303.07240  — *PMC-CLIP*
- https://arxiv.org/abs/2506.14136  — *BiomedCLIP imbalance*
- https://openaccess.thecvf.com/content/CVPR2025/papers/Koleilat_BiomedCoOp_Learning_to_Prompt_for_Biomedical_Vision-Language_Models_CVPR_2025_paper.pdf  — *BiomedCoOp*

### 3D / MRI / CT VLMs
- https://arxiv.org/abs/2509.21249  — *Decipher-MR*
- https://www.nature.com/articles/s41746-026-02596-4  — *Decipher-MR npj DM*
- https://arxiv.org/abs/2406.06512  — *Merlin*
- https://www.nature.com/articles/s41586-026-10181-8  — *Merlin Nature 2026*
- https://arxiv.org/abs/2503.20047  — *Med3DVLM*
- https://arxiv.org/html/2506.12186v1  — *MRI-CORE*
- https://arxiv.org/abs/2507.05201  — *MedGemma*
- https://www.nature.com/articles/s44387-025-00015-9  — *3D medical VLM npj AI*

### RSNA 2024 spine classification (CNN baselines)
- https://www.medrxiv.org/content/10.1101/2024.12.06.24318595v1
- https://link.springer.com/article/10.1007/s44196-025-01098-7  — *MobileNetV2-UNet RSNA 2024*
- https://arxiv.org/abs/2506.09162  — *RSNA LumbarDISC dataset paper*
- https://www.rsna.org/news/2024/november/2024-ai-challenge-winners
- https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification

### Spine-MRI VLM (segmentation / report generation)
- https://link.springer.com/chapter/10.1007/978-3-032-07502-4_14  — *SPINE 3D-VLM report generation*
- https://www.tandfonline.com/doi/full/10.1080/08839514.2026.2626117
- https://github.com/serag-ai/SPINE
- https://link.springer.com/chapter/10.1007/978-981-95-5631-1_3  — *DP-Net spine segmentation*
- https://arxiv.org/html/2511.00095v1  — *SpinalSAM-R1*
- https://www.nature.com/articles/s41598-024-64580-w  — *SpineNet 2024*

### Defense / anti-citations (frozen vs fine-tuned, VLM weaknesses)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12145120/  — *Frozen VLM breast cancer*
- https://ieeexplore.ieee.org/document/10769012/
- https://arxiv.org/abs/2506.18434  — *PEFT prognosis benchmark*
- https://arxiv.org/pdf/2510.16973  — *FM medical systematic review*
- https://arxiv.org/html/2511.01284v1  — *FM adaptation strategies*
- https://arxiv.org/html/2510.23807v2  — *Why FM in pathology fail*
- https://arxiv.org/html/2505.16338  — *Frozen FM + ViT fusion derm*
- https://arxiv.org/abs/2312.07353  — *CLIP medical survey*
- https://link.springer.com/article/10.1007/s13534-025-00484-6  — *VLM medical imaging review*
- https://www.nature.com/articles/s41746-025-02047-6  — *VLM neurorad accuracy 35%*
- https://arxiv.org/html/2505.15425v1  — *MVLM robustness*
- https://www.nature.com/articles/s41746-025-02108-w  — *VLM artefact robustness*

### Pfirrmann / disc-degeneration CNN SOTA
- https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1526478/full
- https://link.springer.com/article/10.1007/s10278-024-01251-2
- https://www.sciencedirect.com/science/article/abs/pii/S1746809426007317
- https://www.nature.com/articles/s41467-022-28387-5

---

## Final summary

**Did we find a direct VLM-spine-MRI-grading comparator?** **No.** The only published BiomedCLIP-on-spine work is on scoliosis from posturographic X-rays (Appl. Sci. 2025) — wrong modality and task. The two MRI-spine VLM papers (SPINE/serag-ai 2026, DP-Net 2025) target report generation and segmentation, not severity grading. None of the RSNA 2024 leaderboard solutions or published RSNA 2024 method papers used a vision-language foundation model. To our knowledge, our work is the first BiomedCLIP-fusion pipeline reported on RSNA 2024 lumbar MRI grading, which is a clear contribution claim for the paper.

**Strongest defensive citations for our BiomedCLIP-frozen + CBAM-3D-ResNet34 hybrid:** (1) "Frozen Large-Scale Pretrained VLMs are the Effective Foundational Backbone for Multimodal Breast Cancer Prediction" (IEEE-J-BHI 2024) — direct architectural twin showing frozen-VLM + lightweight head beats full fine-tune; (2) "Benchmarking Foundation Models and PEFT for Prognosis Prediction" (arXiv 2506.18434) — establishes that on small imbalanced medical datasets, frozen-FM-with-linear-probe is the most stable regime; (3) "Interpreting BiomedCLIP on Radiology under High-Imbalance OOD" (arXiv 2506.14136) — documents BiomedCLIP brittleness under class imbalance, motivating the additional CNN branch. **Strongest future-work citations to acknowledge:** RadCLIP (slice-pooling, the architecture we should ideally have used), Decipher-MR / Merlin / MRI-CORE (native-3D priors), and BiomedCoOp (prompt-tuning upgrade). Position the paper as the first BiomedCLIP-on-RSNA-2024 effort, with explicit acknowledgement that newer 3D-native MRI foundation models are obvious next steps.
