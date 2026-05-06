# Methods Survey: 3D Backbones for Medical Image Classification (2024-2026)

Scope: defending the choice of a 3D ResNet-34 (with CBAM + BiomedCLIP fusion) against more
modern alternatives, in the context of lumbar spine MRI per-IVD grading on RSNA-2024 / SPIDER.
All "comparable to ours" judgments assume small-data regime (~2k IVD volumes per RSNA condition,
~447 series in SPIDER) and a per-IVD multi-task classification head.

---

## 3D CNN backbones

| Model | 1-line description | Spine MRI use? | Pretrained checkpoint | Reported perf. (comparable task) | Assessment vs. our 3D ResNet-34 + CBAM |
|---|---|---|---|---|---|
| **3D ResNet-18 / 34 (Med3D)** | Tencent MedicalNet 3D ResNets pretrained on 23 medical datasets (Chen et al. 2019). | Indirect — used widely for 3D MRI/CT classification; not the dominant choice in 2024-25 spine literature. | `TencentMedicalNet/MedicalNet-Resnet10` ... `Resnet50` on HuggingFace; original `MedicalNet_pytorch_files2.zip` (2019). | LiTS lung-nodule transfer ~+3 Dice over scratch; OASIS-2 Alzheimer 3D-ResNet+CBAM 88.0% acc. | **Comparable / our baseline.** Same family as ours; CBAM + RSNA-pretrain matches typical 3D-ResNet+attention recipe used in 2024-25. |
| **3D DenseNet** | Dense connectivity adapted to volumes; popular for brain/Alzheimer MRI. | No spine-specific work found; brain MRI dominant. | MONAI `DenseNet121` (3D); no large medical pretrain. | OASIS-2 Alzheimer 97.3% acc. with 3D DenseNet + self-attention. | **Comparable.** Higher capacity but no medical-domain pretrain at scale; would lose to MedicalNet-init ResNet on small data. |
| **3D EfficientNet** | Compound-scaled 3D ConvNet; rare in volumetric medical because pretraining doesn't transfer cleanly from 2D ImageNet. | One custom lumbar repo (Lumber_spine_efficientNet); no peer-reviewed spine paper found. | No 3D medical pretrain on HF; only 2D `google/efficientnet-b0..b7`. | N/A on spine MRI. | **Weaker for our setting.** Without 3D medical pretrain, sample efficiency drops below MedicalNet-init ResNet-34. |
| **MedicalNet 3D ResNets (Tencent)** | Same as row 1 — listed separately because it is the de-facto checkpoint source. | The pretrain backbone we already use. | HF org `TencentMedicalNet`; 2019 release, still current. | Med3D paper: ~5-10 Dice/acc points over scratch across 8 downstream tasks. | **Equivalent (this is what we use).** |
| **I3D / R(2+1)D** | Inflated-2D and (2+1)D conv backbones from video understanding (Kinetics). | No lumbar spine MRI work. | Kinetics-pretrained on torchvision/timm; no medical pretrain. | Brain-tumour MRI papers report parity with 3D ResNet. | **Weaker.** Kinetics features transfer poorly to MRI; no spine adoption. |

**CNN summary:** the 2024-25 lumbar literature (SpineScan, GE-YOLOv8, the PLOS-ONE Pfirrmann
pipeline, Frontiers Bioeng. paper) is dominated by *2D* detectors (YOLOv5/v8, DeepLabV3 +
ResNet-50) operating slice-by-slice. Our 3D ResNet-34 is actually more volumetric than what most
spine baselines do.

---

## 3D Transformer backbones

| Model | 1-line description | Spine MRI use? | Pretrained checkpoint | Reported perf. (comparable task) | Assessment vs. ours |
|---|---|---|---|---|---|
| **Swin UNETR / Swin-3D** | Hierarchical 3D Swin Transformer with shifted windows; SOTA on MSD/BTCV segmentation. | Used as detector backbone for lumbar severity in one Kaggle solution; primary use is segmentation. | MONAI / `Project-MONAI` checkpoints pretrained on 5,050 CT volumes (Tang et al. CVPR 2022). | BTCV Dice 0.918; brain-infarct *classification* (Swin-UNETR encoder) reported high acc on 110 classes. | **Comparable but heavier.** Designed for segmentation; classification adaptation is custom. ~4x params and FLOPs of ResNet-34. |
| **3D Swin Transformer (brain MRI pretrain)** | Domain-aware multi-task pretrained 3D Swin on 13,687 brain MRIs (ACCV 2024). | Brain only — no spine pretrain available. | Released with arXiv:2410.00410. | Outperforms supervised + SSL baselines on AD / PD / age regression. | **Weaker for spine.** Domain mismatch (brain pretrain ≠ spine MRI); we'd lose pretraining benefit. |
| **UNETR** | Pure ViT encoder + U-Net decoder for 3D medical (Hatamizadeh et al. 2021). | Segmentation-only in spine literature; no published lumbar classification head. | MONAI; CT/MRI segmentation weights. | BTCV Dice 0.891. | **Weaker.** No classification pretrain; ViT is data-hungry and our IVD-crop dataset is small. |
| **TransUNet** | 2D ViT + CNN hybrid for medical segmentation (TMI 2024 revision). | Segmentation, 2D. Not 3D classification. | Public repo. | Synapse multi-organ Dice ~77.5. | **Not applicable.** 2D segmentation backbone, wrong task. |
| **MedViT-3D / ViT3D variants** | 3D Vision Transformer adaptations; mostly research prototypes. | None found for spine. | No mainstream HF release for 3D MRI classification. | Sparse evidence. | **Weaker.** Insufficient pretraining + data-hungry on small IVD crops. |
| **VideoMAE / Video Swin** | Masked autoencoder / Swin for video; sometimes repurposed for 3D volumes. | None for spine MRI. | Kinetics pretrain on HF; no medical pretrain. | Kinetics-400 86.6% acc (VideoMAE). | **Weaker.** Kinetics → MRI transfer poor; transformers under-perform on small medical datasets without inductive bias. |

**Transformer summary:** for *segmentation* of 3D medical volumes, Swin-UNETR and UNETR++ are
SOTA. For *classification* of small per-IVD crops (9×112×224), there is no published
transformer that clearly beats a 3D-ResNet+CBAM. Reviews note transformers' "limited inductive
bias is a disadvantage on small datasets," which matches our regime.

---

## Medical Vision-Language Models (VLMs)

| Model | 1-line description | Spine MRI use? | Pretrained checkpoint | Reported perf. | Assessment vs. ours |
|---|---|---|---|---|---|
| **BiomedCLIP** | CLIP variant pretrained on 15M PubMed image-caption pairs (Microsoft, arXiv:2303.00915). | Indirect — biomedical figures include some MRI; **2D image encoder only**. Used by us for text-side fusion. | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (HF). | RSNA pneumonia zero-shot 79.7% acc (vs. 73.2% BioViL, 70.7% PubMedCLIP). | **Complementary (already integrated).** 2D only; we use it as a fusion stream, not as the volumetric encoder — exactly the documented usage pattern. |
| **MedCLIP** | Decoupled image-text contrastive on chest X-ray + reports (EMNLP 2022). | No 3D / no spine. | `RyanWW/MedCLIP` on HF. | Outperformed by BiomedCLIP on most zero-shot tasks. | **Weaker than BiomedCLIP** for our fusion role. |
| **CXR-CLIP / CXR-CML** | Chest-X-ray-specific CLIPs, long-tail multi-label improvements (MICCAI 2025). | Chest X-ray only — not spine, not 3D. | Github releases. | Improved long-tail F1 on CheXpert. | **Not applicable.** Wrong anatomy + 2D. |
| **RadCLIP** | Radiology VLM with **slice-pooling for volumetric** images (arXiv:2403.09948, 2024). | Trained on diverse radiology incl. MRI; no published spine numbers. | Public repo; HF release in progress. | Beats BiomedCLIP on radiology image classification + image-text matching. | **Comparable — interesting alternative.** Closest match to "VLM for 3D MRI classification". Worth citing as a future-work direction; switching backbones now would invalidate v3 results. |
| **MedSAM / MedSAM2** | Foundation segmentation models (MedSAM2 = April 2025, 3D + video). | MedSAM2 trained on 77k MRI 3D image-mask pairs incl. some spine. | `bowang-lab/MedSAM`, MedSAM2 HF release April 2025. | Universal segmentation; not a classifier. | **Not a backbone for classification.** Could provide IVD masks upstream; out of scope. |
| **SAM-Med3D** | Promptable 3D segmentation foundation model. | General 3D medical; spine evaluated. | `uni-medical/SAM-Med3D`. | Strong segmentation generalisation. | **Not applicable** (segmentation only). |

**VLM summary:** of all surveyed VLMs, only **RadCLIP** is explicitly designed for volumetric
classification, and it appeared after our pipeline was finalised (March 2024). BiomedCLIP — what
we already use — remains the most-cited general biomedical VLM with a stable HF release and
strong zero-shot baselines.

---

## Why 3D ResNet-34 + CBAM + BiomedCLIP fusion is a defensible choice

- **Pretraining dominates on small data.** Med3D / MedicalNet 3D-ResNet weights are the only
  *medical-domain* 3D pretrained checkpoints with a large public release (23 datasets, available
  on HuggingFace since 2019 and still current). 3D EfficientNet, 3D DenseNet, and 3D ViT
  variants lack equivalent large medical pretraining; in our ~2k-volumes-per-condition regime
  this is decisive.

- **Transformers underperform on small 3D classification crops.** Multiple 2024 reviews
  (Vision-Transformers-in-Medical-Imaging, NVIDIA technical blog) note that Swin-UNETR and
  UNETR are SOTA for *segmentation* of large CT/MRI volumes, but on small per-IVD volumes
  (9×112×224) without large in-domain pretraining, transformers' missing inductive bias hurts
  more than their global attention helps.

- **2024-25 lumbar spine literature is mostly 2D.** SpineScan (Eur. Spine J. 2025), GE-YOLOv8
  (Frontiers 2025), the PLOS-ONE Pfirrmann pipeline, and the YOLOv8x detector all operate on
  2D sagittal slices with ResNet-50 or YOLO backbones. Our 3D ResNet-34 is *more* volumetric
  than the dominant published baselines, not less.

- **Attention is added the standard way.** Adding CBAM after each ResNet stage is the
  canonical recipe (e.g., the 2024 OASIS Alzheimer 3D-ResNet+CBAM paper, 88.0% acc). This is
  the well-trodden path for incremental attention without paying transformer training-data cost.

- **VLM fusion uses BiomedCLIP correctly.** BiomedCLIP is a 2D image-text model; using it as a
  *text/auxiliary* stream over per-slice features (rather than as the 3D volumetric encoder) is
  consistent with its training distribution and HuggingFace usage. RadCLIP (March 2024) is the
  only VLM with explicit 3D slice-pooling and is a natural follow-up cited as future work.

---

## Sources

- [Tencent/MedicalNet (GitHub)](https://github.com/Tencent/MedicalNet)
- [TencentMedicalNet on HuggingFace](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10)
- [Med3D: Transfer Learning for 3D Medical Image Analysis (arXiv:1904.00625)](https://arxiv.org/abs/1904.00625)
- [Construction and Validation of a General Medical Image Dataset for Pretraining (Springer 2024)](https://link.springer.com/article/10.1007/s10278-024-01226-3)
- [SpineScan (Eur. Spine J. 2025)](https://link.springer.com/article/10.1007/s00586-025-09537-x)
- [Advances and challenges in AI-assisted MRI for lumbar disc degeneration (Eur. Spine J. 2025)](https://link.springer.com/article/10.1007/s00586-025-09179-z)
- [GE-YOLOv8 lumbar IVD grading (Frontiers Bioeng. 2025)](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1526478/full)
- [PLOS One Pfirrmann pipeline 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0302067)
- [Self-Supervised Pre-Training of Swin Transformers for 3D Medical (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf)
- [Domain-Aware Multi-Task Pretraining of 3D Swin for Brain MRI (arXiv:2410.00410)](https://arxiv.org/abs/2410.00410)
- [UNETR: Transformers for 3D Medical Image Segmentation (arXiv:2103.10504)](https://arxiv.org/abs/2103.10504)
- [UNETR++ (IEEE TMI 2024)](https://github.com/Amshaker/unetr_plus_plus)
- [Swin-UNETR for brain-infarct classification (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11372279/)
- [BiomedCLIP (arXiv:2303.00915)](https://arxiv.org/abs/2303.00915)
- [BiomedCLIP HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [RadCLIP (arXiv:2403.09948)](https://arxiv.org/html/2403.09948)
- [MedSAM (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-44824-z)
- [MedSAM2 (arXiv:2504.03600)](https://arxiv.org/html/2504.03600v1)
- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)
- [Vision Transformers in Medical Imaging Review (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12701147/)
- [Foundation Models for Volumetric Medical Imaging (MDPI Electronics 2026)](https://www.mdpi.com/2079-9292/15/6/1245)
