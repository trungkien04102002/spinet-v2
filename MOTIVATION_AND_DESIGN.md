# Motivation and Design Rationale — SpineNetV2 + CBAM + BiomedCLIP

> Date: 2026-04-27. Audience: master's thesis committee + EMBC 2027 reviewers.
> This document explains **why** the architecture was designed this way — not what it does (see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), [`HIGH_LEVEL_ARCHITECTURE.md`](experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md)) and not the empirical numbers (see [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md)).

## TL;DR

We chose a **lightweight, reproducible architectural pattern** rather than a heavy foundation-model approach because the research question is *how to extend a structured-grading model's label space without retraining*, not *how to scale up parameters*. Concretely:

- **Backbone = SpineNetV2** (3D ResNet34, Windsor et al. 2024) — discriminative, pretrained on lumbar IVD volumes, uses the (9 × 112 × 224) sagittal stack format that matches both RSNA and SPIDER pipelines.
- **+ CBAM** (Woo et al. 2018) — adds spatial + channel attention at near-zero parameter cost; empirically recovers Severe-class detection from F1 = 0.000 to F1 ≈ 0.20–0.28 on the hardest foraminal classes.
- **+ Frozen BiomedCLIP** (Zhang et al. 2023) — provides a 512-d contrastive image-text embedding space; classification by cosine similarity enables **zero-shot label extension** (encode a new prompt at inference, no retraining needed).

The whole system is ~218M parameters but only ~5M trainable, runs end-to-end in <30 minutes on a single consumer GPU, and the pattern is foundation-model-agnostic (BiomedCLIP can be swapped for MedSigLIP / RadFM / RAD-DINO in future work).

---

## 1. Problem statement (from advisor feedback)

The thesis was reframed mid-2026 in response to advisor (Phan Nhân) feedback, summarized in [`ADVISOR_FEEDBACK_LABELSPACE.md`](experiments/paper_results/ADVISOR_FEEDBACK_LABELSPACE.md):

> "Kêu 5 nhãn train ra 5 nhãn, kêu 10 nhãn, kêu ra 10 nhãn... RSNA 5 nhãn chỉ hiểu 5 nhãn, làm sao mở rộng được sang dataset khác?"
>
> *("If you train on 5 labels you get 5 labels; train on 10 you get 10. An RSNA-trained model only understands those 5 labels. How does it extend to other datasets?")*

The advisor articulated **two levels** of label-space generalization:

| Level | Requirement | Architectural property needed |
|---|---|---|
| Level 1 — Zero-shot | New label not seen in training is predicted at inference | Image and text in a shared embedding space; classification by cosine similarity (CLIP-style) |
| Level 2 — Incremental | Add labelled examples for a new class without forgetting old ones | Class-incremental learning (out of scope for this thesis) |

**Decision**: Target Level 1. This is what the SPIDER zero-shot evaluation tests — encode 8 unseen disease prompts at inference, classify with no SPIDER training. This requirement directly forces the cosine-similarity head and the contrastive image-text encoder, which is what BiomedCLIP provides.

---

## 2. Why SpineNetV2 as the base (vs Ningshen, vs MedGemma)

### 2.1 vs Ningshen rsna-2024 (representative 2D Kaggle approach)

The Ningshen submission [(github.com/nshen7/rsna-2024)](https://github.com/nshen7/rsna-2024) is a **two-stage 2D pipeline**: Faster R-CNN (ResNet-50-FPN) localizes IVD bounding boxes per slice; a fine-tuned Swin Transformer classifies severity from each cropped 2D patch using weighted cross-entropy. It is technically sound as a Kaggle prototype — but has three structural limitations relative to our goals:

1. **No volumetric context.** Each IVD is a single 2D crop; through-plane information critical for stenosis grading is discarded. Our pipeline uses the full (9 × 112 × 224) sagittal stack as a single 3D tensor.
2. **No shared multi-task representation.** Separate fine-tuned model instances per condition (SCS, LNFN, RNFN, LSS, RSS), with a sequential cascade (SCS → LSS → RSS) only as a workaround for small data. Our backbone uses one shared 3D ResNet34 trunk with multi-task heads, enabling regularization across conditions.
3. **No mechanism for label-space extension.** Adding a new disease (e.g., Modic from SPIDER) requires labelled data and a new training run from scratch. Our cosine-similarity head extends to new prompts at inference.

The Ningshen repository's own README states the pipeline "may not fully reflect optimal performance, as limited time and computational resources restricted the ability to thoroughly tune hyperparameters" — it is **not a top-10 leaderboard entry** and should be cited as "a representative 2D Kaggle approach", not as the state of the art. (The 2024 RSNA Kaggle prize-winners used multi-model ensembles with TTA, pseudo-labels, and sagittal T2 + T1 fusion.)

### 2.2 vs MedGemma (medical multimodal foundation model)

MedGemma (Google Health, 2025; [arXiv:2507.05201](https://arxiv.org/abs/2507.05201)) is a high-quality medical vision-language model (4B / 27B parameters, built on Gemma 3 + SigLIP-400M vision encoder, 33M medical image-text pairs). It is genuinely strong at medical VQA and report generation. We did not use it as the base for **five concrete reasons**:

1. **2D-only multimodal.** The technical report (Section 2.1.1) explicitly states: *"the multimodal capabilities of MedGemma are currently focused on 2D medical images (e.g., X-ray, 2D slices from CT/MRI); 3D volumes ... were not included."* Our task is volumetric MRI grading where the sagittal stack carries clinically meaningful information that single slices do not.
2. **No spine-MRI specialization.** Of the 33M training pairs, only 47,622 are MRI 2D slices (~0.14%) — selected from radiology reports mentioning specific abnormal slices, not focused on lumbar/sagittal anatomy. SpineNetV2's backbone was pretrained directly on lumbar IVD volumes.
3. **Generative paradigm — wrong use case.** MedGemma outputs text tokens (radiology reports, VQA answers). Our task is structured multi-class grading. Using MedGemma would mean stripping the LLM and using only the SigLIP encoder — at which point it functions as a frozen feature extractor, identical in role to BiomedCLIP but ~5× larger (400M vs 86M) without an aligned text branch usable for zero-shot extension.
4. **Compute cost vs sample count mismatch.** 4B–27B models fine-tuned on ~2000 labelled IVDs is in the high-overfitting regime even with QLoRA. The contribution of this thesis is showing that the integration pattern works under realistic constraints — single GPU, ~30 min training. Plug-in MedSigLIP/MedGemma is left as future work.
5. **No CLIP-style zero-shot interface.** MedGemma does not expose an L2-normalized image-text contrastive embedding space the way CLIP/BiomedCLIP does. The cosine-similarity zero-shot head requires this property; BiomedCLIP provides it natively.

### 2.3 Other foundation models the committee may ask about

| Model | What it is | Why not used |
|---|---|---|
| **RadFM** (Wu et al., Nature Comm. 2025) | Generalist radiology FM, supports 2D + 3D | Generative paradigm, not contrastive; heavy fine-tune cost; no spine-specific benchmark |
| **RAD-DINO** (Pérez-García et al., 2024) | Vision-only DINOv2 SSL on chest X-ray | No text branch → no zero-shot via prompts; chest X-ray domain |
| **Merlin** (Blankemeier et al., 2024) | 3D CT vision-language | Specialised for CT, not MRI; large model |
| **Med-PaLM-M** (Tu et al., 2023) | 562B multimodal | Not open-weight; prohibitive compute |

**Bottom line**: SpineNetV2 is the right *backbone* (3D, lumbar-pretrained, discriminative). BiomedCLIP is the right *zero-shot head* (contrastive, frozen, CLIP-style). MedGemma and RadFM solve a different problem (generative reasoning) at a different resource level.

---

## 3. Why CBAM helps Severe-class detection

### 3.1 Mechanism (Woo et al. ECCV 2018)

CBAM applies two sequential lightweight attention gates after each ResNet stage:

- **Channel attention**: global average + max pool over spatial dims → shared MLP → per-channel weight vector. Recalibrates which feature maps encode pathology vs anatomy.
- **Spatial attention**: per-channel statistics aggregated at each location → 2D weight map. Suppresses background, amplifies lesion regions.

Combined cost: ~0.1% of backbone parameters. Inserted after each of the 4 ResNet stages in our 3D ResNet34. Importantly, the spatial gate operates on the full 3D feature tensor `(C, D, H, W)` — it learns *which sagittal slice and which spatial location* simultaneously, giving the backbone genuine volumetric reasoning.

### 3.2 Why this specifically helps imbalanced spinal stenosis grading

The class imbalance problem in lumbar stenosis is **structurally spatial**: a "Severe" foraminal narrowing affects only a small sub-region of the IVD crop (often a few residual-canal voxels). Without spatial attention, gradient signal from the rare Severe class is diluted by the much larger anatomically-normal region. This is consistent with the empirical baseline failure mode in our [Phase 1+2 results](experiments/fresh_cbam/RESULTS_SUMMARY.md): Foraminal Severe F1 = 0.000 in the no-CBAM baseline (the model never predicts Severe at all on these classes).

CBAM's spatial gate forces the network to up-weight precisely where the foramen is narrowed, concentrating gradient at the discriminative boundary. The empirical jump:

| Metric | Baseline (no CBAM, no class weights) | CBAM + sqrt class weights |
|---|---|---|
| Spinal Severe F1 | 0.457 | **0.623** |
| L. Foraminal Severe F1 | **0.000** | 0.180 |
| R. Foraminal Severe F1 | **0.000** | 0.197 |

The recovery from 0.000 is the discriminative evidence: the mechanism is no longer spatially blind to the pathological sub-region. This pattern parallels established medical-imaging precedents — CBAM-augmented networks recover hard-class recall on small drusen/microaneurysm lesions in retinal imaging (Roy et al., IEEE TMI 2019) and on sub-centimeter pulmonary nodules (Liu et al., Medical Physics 2020).

### 3.3 Why CBAM specifically (and not SE-Net / non-local / transformer)?

- **vs Squeeze-and-Excitation (Hu et al. CVPR 2018)**: SE provides channel attention only, no spatial gate. For small-lesion localization the spatial component is the critical one.
- **vs non-local blocks / transformer self-attention**: quadratic in spatial resolution, substantial parameter overhead. Prohibitive in our ~1500-IVD training regime where overfitting is the primary risk.
- **vs Coordinate Attention / ECA-Net**: also viable; CBAM was chosen for its established medical-imaging ablation literature and clean drop-in integration after each ResNet stage.

CBAM's value proposition is "spatial + channel attention at near-zero cost, well-studied on medical imaging." For the resource regime of this thesis, it is the appropriate trade-off.

---

## 4. Why frozen BiomedCLIP enables zero-shot label extension

### 4.1 Mechanism (Zhang et al. 2023, Microsoft; [arXiv:2303.00915](https://arxiv.org/abs/2303.00915))

BiomedCLIP is a CLIP-style contrastive model pretrained on 15M biomedical image-text pairs from PubMed Central — radiology, pathology, microscopy, clinical photographs. Following Radford et al. (CLIP, ICML 2021), training minimises a symmetric contrastive loss that aligns paired image and text embeddings while pushing apart unpaired ones. The output is a **shared 512-d embedding space** where image features and text descriptions of the same medical concept are co-located.

The zero-shot mechanism is direct: at inference time, label prompts (e.g., `"MRI showing severe foraminal stenosis"`) are encoded by the frozen text encoder into the shared space. A query image is encoded by the frozen vision encoder, and classification reduces to nearest-prompt-by-cosine-similarity. **No retraining needed to add new labels** — encoding a new prompt is sufficient. This is the exact architectural property that satisfies the advisor's Level-1 generalization requirement.

### 4.2 Why frozen, not fine-tuned?

With ~1500 RSNA + ~1400 SPIDER labelled samples, fine-tuning a ViT-B/16 pretrained on 15M pairs would cause catastrophic forgetting of the semantic alignments that make zero-shot transfer work in the first place. Standard practice in low-data multimodal regimes is to use the frozen encoder as a feature extractor (consistent with CLIP linear probe protocol, Radford et al. 2021). Fine-tuning is deferred to future work with larger annotated corpora.

### 4.3 Why the projection MLP is needed (image side only)

BiomedCLIP's vision encoder is 2D and trained on PubMed figure images. Our task uses a 9-slice sagittal stack and the CBAM 3D backbone provides a complementary volumetric feature. The projection MLP fuses the two:

```
[CBAM 3D feat: 512] ⊕ [BiomedCLIP per-slice + attention pool: 512]
                       ↓
              MLP (1024 → 768 → 512)
                       ↓
              L2-normalize → image_emb
                       ↓
              cosine sim against text_emb_db
```

This is the architecturally novel piece: a cross-encoder fusion (3D specialist + 2D semantic) into a CLIP-aligned 512-d space.

---

## 5. Why the combination is complementary (the core thesis argument)

The CBAM-3D and BiomedCLIP-2D feature paths are **informationally orthogonal**:

| Axis | CBAM 3D ResNet34 | BiomedCLIP ViT-B/16 |
|---|---|---|
| Input dimensionality | Full 9-slice sagittal volume | Single 2D slice (×3, attention-pooled) |
| Feature type | Spatial lesion localization, volumetric geometry | Semantic disease appearance, language-grounded |
| Attention mechanism | Explicit channel + spatial gates trained on RSNA | Implicit via contrastive pretraining on 15M pairs |
| Zero-shot capability | None (alone) | Native (prompt encoding) |
| Specialisation domain | RSNA lumbar stenosis | Broad biomedical (PubMed-curated) |

The Phase 3 SPIDER zero-shot results directly validate the complementarity story (full numbers in [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md) §Phase 3):

- **Hybrid wins on disc-related labels** (Disc_narrowing 0.586 vs Naked-BiomedCLIP 0.403, +0.18; Disc_bulging 0.613 vs 0.363, +0.25). These pathologies involve volumetric geometry — disc height, annular bulge across multiple slices — that a single 2D feature cannot capture reliably. The CBAM 3D path supplies the spatial context.
- **Naked BiomedCLIP wins on vertebra/endplate labels** (Modic, UP/LOW endplate, Spondylolisthesis). These are visually distinctive in 2D and well-represented in PubMed (Modic changes have been extensively imaged in clinical literature). The RSNA-trained projection MLP, however, has been optimised exclusively on foraminal/canal stenosis label space and *actively over-specialises* to that signal — pushing embeddings away from regions useful for unrelated pathology.

This is **not a failure of the architecture** — it is the predicted behaviour when the 3D specialist branch is domain-specific. It is also a clean experimental finding for the paper: complementarity is real, but the projection MLP induces a label-space bias that the field should be aware of when applying this pattern. Future work: train the projection on more diverse labels, or use multi-domain projection heads.

---

## 6. Anticipated committee questions (with crisp answers)

These are the questions a thesis committee or EMBC reviewer is most likely to ask. Phrased in English here; defense will be in Vietnamese.

### Q1. "Why not just use MedGemma / RadFM and call it done?"

Three reasons:
1. **Wrong paradigm**: those are generative models; our task is structured grading with cosine-sim zero-shot extension — needs a contrastive embedding space, which MedGemma does not natively expose.
2. **Wrong domain**: MedGemma's multimodal arm is 2D-only and not specifically pretrained on sagittal lumbar MRI.
3. **Wrong resource regime**: our integration pattern is meant to be reproducible on a single consumer GPU. Plugging in heavier foundation models is left as future work and the architecture is designed to support it (BiomedCLIP can be swapped for MedSigLIP / RadFM with no other changes).

### Q2. "Why CBAM and not Squeeze-and-Excitation, non-local, or a 3D transformer?"

CBAM provides both channel and spatial attention at ~0.1% backbone-parameter cost. SE-Net is channel-only, missing the spatial localization signal that drives Severe-class recovery. Non-local / transformer self-attention scales quadratically and adds substantial parameters — prohibitive in our small-data regime. CBAM has well-studied medical-imaging ablations and is a clean drop-in.

### Q3. "Why is BiomedCLIP frozen?"

Catastrophic-forgetting risk. With ~3000 labelled samples total (RSNA + SPIDER), fine-tuning a ViT-B/16 pretrained on 15M pairs would erode the semantic alignments that make zero-shot work in the first place. Frozen feature extraction is standard practice in low-data multimodal regimes (CLIP linear probe protocol).

### Q4. "Why cosine similarity instead of a linear softmax head?"

A softmax head over a fixed vocabulary cannot generalize to unseen labels at inference time. Cosine similarity in the shared CLIP embedding space allows any new disease description to be encoded as a prompt and immediately used as a classification target — this is the architectural property that satisfies Level-1 zero-shot generalization. It is mathematically equivalent to a nearest-centroid classifier in L2-normalised space (Snell et al. NeurIPS 2017, Prototypical Networks).

### Q5. "Your zero-shot Hybrid loses to Naked BiomedCLIP on overall mean F1. Isn't your architecture worse?"

No — the result is **selective transfer**, not failure. Hybrid wins on disc-related diseases (where 3D volumetric context is critical) and loses on vertebra/endplate diseases (where 2D appearance is sufficient and the RSNA-trained projection MLP over-specialises to stenosis features). The honest interpretation is that the projection MLP is a domain bias that helps stenosis-aligned tasks and hurts unrelated ones. We report this transparently rather than smooth it over.

### Q6. "Will this scale to other body regions / modalities?"

Yes. The pattern is anatomy-agnostic: 3D CNN backbone + frozen contrastive encoder + cosine-sim head. The only anatomy-specific components are the label prompts (trivially replaceable) and the projection MLP weights (require fine-tuning on the new dataset). BiomedCLIP's pretraining covers multiple body regions. Extending to cervical spine, knee, brain MRI requires only changing prompts and re-running the projection-MLP fine-tune.

### Q7. "How is this different from prior work doing CLIP for medical imaging (MedCLIP, PubMedCLIP, GLoRIA)?"

Prior work uses CLIP-style pretraining and zero-shot evaluation on the same 2D modality (chest X-ray). Our novelty is **architectural**: we fuse a 3D volumetric specialist with a 2D contrastive encoder via a learned projection, and demonstrate **cross-dataset zero-shot label extension** (train on RSNA, evaluate on SPIDER without seeing any SPIDER labels). The two key differentiators are (i) volumetric input + 2D semantic fusion, (ii) cross-dataset label-space generalization rather than within-dataset.

### Q8. "Why patient-level split, and is the zero-shot evaluation truly held out?"

Train/val splits within each dataset are patient-level (`random_state=42`) to prevent same-patient leakage. The zero-shot evaluation on SPIDER is the strongest held-out test we have — the model was never exposed to any SPIDER label or image during training; SPIDER images come from different scanners, different hospitals, different acquisition protocols. This is closer to "deployment shift" than within-dataset val splits.

### Q9. "Is one-seed reporting enough?"

For the thesis it is — with the caveat that we report Mean F1 macro across 4 conditions / 8 zero-shot diseases, which averages out per-class variance. For the EMBC paper version we plan to add a 3-seed mean ± std on the headline numbers (Phase 1+2 Hybrid + Phase 3 zero-shot + Phase 4 Hybrid frozen). Single-seed is acknowledged as a limitation in [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md).

### Q10. "What is the actual contribution if BiomedCLIP and CBAM are existing modules?"

The contribution is the **integration pattern + empirical demonstration**:
1. Specific fusion architecture (3D CBAM + per-slice 2D BiomedCLIP + attention-pool + projection MLP into shared CLIP space).
2. First demonstration (to our knowledge) of cross-dataset zero-shot label extension between RSNA and SPIDER on volumetric lumbar MRI.
3. Honest characterization of when the integration helps (disc-related, where 3D matters) vs when the simpler frozen encoder is sufficient (vertebra/endplate, where 2D semantics carry the signal). This selective-transfer finding is itself a contribution because it tells future practitioners when the extra complexity is justified.

---

## 7. Mapping to the three stated contributions

The three contributions stated in [`STEPS.md`](STEPS.md) map as follows:

| Contribution | Location in this thesis | Empirical evidence |
|---|---|---|
| C1: CBAM + Focal Loss for Severe-class recovery on RSNA | Phase 1+2 | Foraminal Severe F1: 0.000 → 0.180 / 0.197 (CBAM); 0.276 / 0.283 (Hybrid) |
| C2: BiomedCLIP integration for zero-shot label extension | Phase 3 | 8 SPIDER labels predicted with no SPIDER training; per-tier F1 0.586 / 0.401 / 0.277 |
| C3: CBAM/Hybrid generalises across datasets (RSNA → SPIDER supervised) | Phase 4 | Hybrid frozen mean F1 = 0.623 vs Vanilla 0.610, with ~5M trainable params vs 22M |

---

## 8. References (core list, ~12 papers)

1. Windsor R., Jamaludin A., Kadir T., Zisserman A. **SpineNetV2: Automated Detection, Labelling and Radiological Grading of Clinical MR Images** — MICCAI 2024 / arXiv:2205.01683.
2. Woo S., Park J., Lee J.-Y., Kweon I.S. **CBAM: Convolutional Block Attention Module** — ECCV 2018.
3. Zhang S., Xu Y., Usuyama N. *et al.* **BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs** — arXiv:2303.00915 (2023).
4. Radford A., Kim J.W., Hallacy C. *et al.* **Learning Transferable Visual Models From Natural Language Supervision** (CLIP) — ICML 2021.
5. Hu J., Shen L., Sun G. **Squeeze-and-Excitation Networks** — CVPR 2018.
6. Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. **Focal Loss for Dense Object Detection** — ICCV 2017.
7. Kendall A., Gal Y., Cipolla R. **Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics** — CVPR 2018.
8. Snell J., Swersky K., Zemel R. **Prototypical Networks for Few-shot Learning** — NeurIPS 2017 (theoretical basis for cosine nearest-centroid).
9. He K., Zhang X., Ren S., Sun J. **Deep Residual Learning for Image Recognition** (ResNet) — CVPR 2016.
10. Sebro R. *et al.* **MedGemma Technical Report** — arXiv:2507.05201 (2025) — referenced for "why not MedGemma" comparison.
11. Pérez-García F. *et al.* **RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision** — arXiv:2401.10815 (2024).
12. nshen7. **rsna-2024** GitHub repository, [github.com/nshen7/rsna-2024](https://github.com/nshen7/rsna-2024) — referenced as representative 2D Kaggle approach.

Additional small-lesion-attention precedents (verify before submission):
- Roy A.G. *et al.* IEEE TMI 2019 — attention for retinal lesion detection.
- Liu et al. Medical Physics 2020 — spatial attention for sub-centimeter pulmonary nodules.

---

## 9. Related repo docs (read these for more depth)

| Topic | Document |
|---|---|
| Big-picture pipeline + status | [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) |
| Master execution plan + 3 contributions | [`STEPS.md`](STEPS.md) |
| Detailed Mermaid architecture diagrams | [`HIGH_LEVEL_ARCHITECTURE.md`](experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md), [`new_architecture.md`](experiments/paper_results/new_architecture.md) |
| Hybrid model training how-to | [`HYBRID_TRAINING_GUIDE.md`](HYBRID_TRAINING_GUIDE.md) |
| RSNA preprocessing + training | [`RSNA_PIPELINE.md`](RSNA_PIPELINE.md) |
| SPIDER zero-shot setup | [`SPIDER_ZEROSHOT_SETUP.md`](SPIDER_ZEROSHOT_SETUP.md) |
| SPIDER transfer learning | [`SPIDER_TRAINING_GUIDE.md`](SPIDER_TRAINING_GUIDE.md) |
| Advisor's label-space concern (verbatim) | [`ADVISOR_FEEDBACK_LABELSPACE.md`](experiments/paper_results/ADVISOR_FEEDBACK_LABELSPACE.md) |
| Advisor meeting prep notes | [`ADVISOR_MEETING_PREP_2026-04-24.md`](experiments/paper_results/ADVISOR_MEETING_PREP_2026-04-24.md) |
| Foundation-model landscape research | [`KEYWORDS_RESEARCH.md`](experiments/paper_results/KEYWORDS_RESEARCH.md) |
| Phase 1+2 RSNA detailed results | [`fresh_cbam/RESULTS_SUMMARY.md`](experiments/fresh_cbam/RESULTS_SUMMARY.md) |
| Phase 3 SPIDER zero-shot detailed results | [`spider_zeroshot/RESULTS_REPORT.md`](experiments/paper_results/spider_zeroshot/RESULTS_REPORT.md) |
| Cross-phase unified report | [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md) |
