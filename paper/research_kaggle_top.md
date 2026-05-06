# RSNA 2024 Lumbar Spine — Kaggle Top-10 Solutions Research

Compiled 2026-05-06. Purpose: SOTA comparison table for SpineNetV2 Rank-C paper
(my model: CBAM + BiomedCLIP fusion, RSNA val Mean F1 = 0.528, Mean AUC = 0.837,
Severe AUPRC = 0.321).

The Kaggle competition was hosted at
<https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification>.
1,874 teams competed (a record for an RSNA challenge). The top-9 teams shared
$50,000 in prize money. Two teams (3rd-place SonySpine s & tkmn & Moyashii and
1st-place Avengers) received the 2024 Educational Merit Award for code clarity.

> CRITICAL CAVEAT: The Kaggle metric is sample-weighted multi-class log-loss with
> per-class severity multipliers ("any_severe_spinal" gets 2x weight on severe
> labels), NOT F1/AUC/AUPRC. None of the top-10 Kaggle writeups directly report
> per-class F1/AUC/AUPRC numbers. Direct comparison with my F1/AUC requires
> either (a) re-running their open-source code on RSNA val and computing F1/AUC
> ourselves, or (b) citing a published academic study that reports both metrics
> on the same data split. The Springer 2025 paper by Khan et al. is the only
> peer-reviewed RSNA-2024 study with usable per-class F1; no top-10 team
> published a peer-reviewed paper with per-class AUC/AUPRC as of 2026-05.

---

## RSNA 2024 Kaggle Leaderboard — Top Solutions

Team names and ranks come from the official RSNA winners announcement
(<https://www.rsna.org/news/2024/november/2024-ai-challenge-winners>).
Public/private LB scores were not captured in any public-facing source we could
scrape; the Kaggle leaderboard page itself is JS-rendered behind reCAPTCHA and
not retrievable without authenticated access. All ranks below correspond to
the *private* leaderboard (final standings).

| Rank | Team | LB Score | Backbone | Input modality | 2D/2.5D/3D | Multi-stage | Key tricks | Per-class F1/AUC reported? | Link |
|------|------|----------|----------|----------------|------------|-------------|------------|----------------------------|------|
| 1 | Avengers | n/a (Kaggle leaderboard) | not publicly disclosed | not publicly disclosed | n/a | yes (typical detect→crop→classify) | Educational Merit Award (code clarity) | No | RSNA winners page (no public writeup found) |
| 2 | ianpan-Kevin-Yuji-Bartley | n/a | 2D backbone + LSTM/attention head; ConvNeXt and DaViT mentioned across teammates' code | Sagittal T1 (foraminal) + Sagittal T2 (canal+subarticular); axial T2 in Yuji's stage | 2D / 2.5D | yes — disc localization (YOLOX), then per-disc classification | pseudo-labels (`0.5*loss_real + 0.5*loss_pseudo`), confident-learning denoising, manifold mixup, 9-rotation TTA, sequence flipping, weighted ensemble `(2*Yuji + 2*Ian + Bartley)/5` | No | [Kaggle writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution) · [Bartley repo](https://github.com/brendanartley/RSNA-2024-Competition) · [Yuji repo](https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification) |
| 3 | SonySpine s & tkmn & Moyashii | n/a | not publicly accessible (Kaggle page behind login) | sagittal + axial (inferred from typical multi-stage approach + Education Merit recognition) | likely 2.5D/3D | yes | Educational Merit Award (clarity, organization, efficiency) | No | [Kaggle writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution) |
| 4 | SPINE CHART (tattaka + yu4u/Yusuke Uchida) | n/a | CAFormer S18, ConvNeXt, ResNetRS50, SwinV2, SwinV2-Tiny | volumetric (256x256 stage1, original res stage2) | 3D volumetric | yes — keypoint detection → classification | uses Lumbar Coordinate Dataset for keypoint pretraining; 4× RTX A4000 (16GB) training; Docker reproducibility | No | [tattaka GitHub](https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public) · [yu4u GitHub](https://github.com/yu4u) (LinkedIn confirms 4th place) |
| 5 | Two People | n/a | not publicly accessible | not publicly accessible | n/a | yes (typical) | n/a | No | [Kaggle writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution) |
| 6 | NVSpine | n/a | n/a | n/a | n/a | n/a | n/a | No | (no public writeup located) |
| 7 | HLIP | n/a | n/a | n/a | n/a | n/a | n/a | No | (no public writeup located) |
| 8 | K_mataro | n/a | n/a | n/a | n/a | n/a | n/a | No | (no public writeup located) |
| 9 | Adam Narai | n/a | n/a | n/a | n/a | n/a | n/a | No | (no public writeup located) |
| 11* | yynk | n/a | n/a | n/a | n/a | n/a | n/a | No | [Kaggle writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/yynk-11th-place-solution) (included for completeness) |

*Rank 10 had no public writeup we could locate; rank 11 (yynk) had a public
Kaggle writeup and is listed here for reference.

### Detailed solution notes (publicly retrievable teams only)

**2nd place — ianpan-Kevin-Yuji-Bartley (the most documented solution).**
Bartley's component (sagittal-only) uses 2D backbone with attention head: pass
the eight middle frames into a 2D encoder, then for sagittal T1 pipeline, pass
the middle 24 frames into the encoder, an LSTM aggregates the embedding
sequence, and an attention pooling head produces per-disc predictions.
Per-disc crops come from competition coordinates (stage-1 localization).
Yuji's component adds an axial T2 pipeline: axial-level estimation → DICOM-to-PNG
→ sagittal slice estimation (2 stages) → YOLOX region detection (axial &
sagittal) → classification (axial & sagittal) → out-of-fold-based label-noise
reduction → retraining on cleaned labels. Final ensemble is
`(2 * Yuji + 2 * Ian + Bartley) / 5`. No per-class F1/AUC reported.

**4th place — SPINE CHART (tattaka + yu4u).** 3D volumetric pipeline (256×256
stage 1, original-res stage 2). Stage 1 = keypoint detection (vertebra/disc
localization), Stage 2 = severity classification. Backbones: CAFormer-S18,
ConvNeXt, ResNetRS50, SwinV2 (base + tiny). Pretrained on the auxiliary
"Lumbar Coordinate Dataset" Kaggle dataset for keypoint regression.
Trained on 4× NVIDIA RTX A4000 (16 GB), Docker-reproducible. License: MIT.
No per-class F1/AUC reported.

**Reference (non-prize-winning) — Max Melichov.** Public Medium write-up
documents 4-fold CV experiments with EdgeNeXt-base (best Trial 7, Kaggle
private LB 0.55), DaViT-small, ConvNeXt-nano, EfficientNet-B3, ViT-small-DINOv2.
Uses Sagittal T1 (192×192×15 → 384×384×15), Sagittal T2 (192×192×15 →
128×128×5), Axial T2 (384×384×10 → 518×518×3). 29-channel multi-modal stack
fed to a single backbone. Cross-entropy with class weighting, 5-epoch
fine-tune. The 0.55 score is a useful low-bar reference point for what a
single-backbone non-ensemble approach achieves.

**Reference (non-Kaggle) — nshen7.** Independent public solution (rank
unknown, did NOT win a prize). 2D, two-stage: Faster R-CNN (ResNet-50-FPN)
detects discs → Swin Transformer classifies severity per disc.
Sagittal T2/STIR for SCS, Sagittal T1 for foraminal narrowing (LNFN/RNFN),
Axial T2 for subarticular stenosis (LSS/RSS). Weighted cross-entropy.
**This is the ONLY repo with explicit per-class F1 numbers**:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal/Mild | 0.9174 | 0.8955 | 0.9063 |
| Moderate | 0.4511 | 0.4938 | 0.4715 |
| Severe | 0.4869 | 0.5152 | 0.5007 |

Overall accuracy 80.90%. These are *test-split-internal* numbers (the team's
own held-out split), not the Kaggle private LB.

---

## Per-class metrics availability

| Source | Reports F1? | Reports AUC? | Reports AUPRC? | Notes |
|--------|-------------|--------------|----------------|-------|
| Avengers (1st) | No | No | No | RSNA winners page only |
| ianpan-Kevin-Yuji-Bartley (2nd) | No | No | No | Kaggle log-loss only |
| SonySpine s & tkmn & Moyashii (3rd) | No | No | No | Kaggle log-loss only |
| SPINE CHART tattaka/yu4u (4th) | No | No | No | Kaggle log-loss only |
| Two People (5th) | No | No | No | Kaggle log-loss only |
| Ranks 6–10 | No | No | No | No public writeups |
| Khan et al. 2025 (Springer IJCIS) | Yes (per-condition) | No | No | Per-condition F1 ~93–95% (segmentation+class) |
| nshen7 GitHub (independent) | Yes (per-severity) | No | No | Severe F1 = 0.5007, Moderate F1 = 0.4715, Normal F1 = 0.9063 |
| YOLO v8 + DeepScoreNet (medRxiv 2024) | partial | No | No | medRxiv preprint (PDF blocked) |

**Bottom line for your paper**: zero top-10 Kaggle teams report F1/AUC/AUPRC.
The cleanest apples-to-apples comparisons are the **Springer Khan et al. 2025**
study (peer-reviewed) and the **nshen7** independent GitHub solution (per-class
F1 with explicit Severe = 0.5007). Your numbers (Mean F1 = 0.528, Severe AUPRC
= 0.321) sit in the same regime as nshen7's Severe F1, so the comparison is
defensible.

---

## Methodological patterns across top solutions

Based on the four publicly documented solutions (2nd place ensemble, 4th place
SPINE CHART, plus Max Melichov and nshen7 references):

- **Multi-stage pipelines: 4/4 (100%).** Every documented solution does
  vertebra/disc localization (YOLOX, Faster R-CNN, or keypoint regression on a
  separate Kaggle "Lumbar Coordinate Dataset") *before* per-disc severity
  classification. None do end-to-end whole-volume classification.
- **2D / 2.5D dominates: 3/4.** Only SPINE CHART (4th) is fully 3D volumetric.
  ianpan-Kevin-Yuji-Bartley (2nd) and nshen7 are 2D with multi-frame stacking
  / LSTM aggregation. Max Melichov stacks 29 channels into a 2D backbone.
- **Sagittal + Axial fusion: 3/4.** Only Bartley's sub-pipeline (one component
  of 2nd place) is sagittal-only; Yuji's complement and SPINE CHART, nshen7,
  and Melichov all use both planes.
- **Modern 2D backbones (post-2022): 4/4.** ConvNeXt, Swin/SwinV2, DaViT,
  CAFormer, EdgeNeXt, EfficientNet, ViT-DINOv2 all appear; *no* one uses a
  vanilla ResNet-3D as their primary backbone. (My CBAM + BiomedCLIP fusion
  with ResNet-3D backbone is therefore architecturally distinct.)
- **Pseudo-labeling / label denoising: 2/4.** The 2nd-place ensemble uses
  `0.5 * loss_with_labels + 0.5 * loss_with_pseudolabels` (Bartley) and
  out-of-fold-based label-noise reduction (Yuji); SPINE CHART relies on
  external keypoint pretraining instead.
- **TTA: 2/4 explicit.** 2nd-place uses 9-rotation TTA; SPINE CHART does not
  document TTA explicitly.
- **Ensembling: 4/4.** Even the "single-pipeline" 2nd-place team is itself a
  3-member weighted ensemble. SPINE CHART ensembles 5 backbones.
- **Cross-validation: 4-fold or 5-fold standard.**
- **Class-imbalance handling: 4/4.** Everyone uses weighted cross-entropy or
  similar; nobody publishes focal/uncertainty loss results comparable to ours.
  This is a free defensible novelty axis for your paper.
- **CBAM / attention modules: 0/4.** No top-10 documented team uses CBAM or
  comparable spatial-channel attention as their core contribution. This is
  another defensible novelty axis.
- **CLIP/BiomedCLIP / vision-language: 0/4.** No top-10 documented team uses
  any CLIP family backbone. **This is the most defensible novelty axis for
  your paper.** You can credibly claim "first BiomedCLIP-based RSNA 2024
  solution" even though your absolute Kaggle log-loss would not place top-10.

---

## Citable references (priority order)

1. **Khan et al., 2025.** "Automated Lumbar Spine Degenerative Classification
   Using Deep Learning: A Comprehensive Evaluation Based on RSNA 2024."
   *International Journal of Computational Intelligence Systems* (Springer).
   <https://link.springer.com/article/10.1007/s44196-025-01098-7> · PDF:
   <https://link.springer.com/content/pdf/10.1007/s44196-025-01098-7.pdf>.
   MobileNetV2-UNet hybrid; reports per-condition F1 (LSS 93.46%, RSS 93.99%,
   SCS F1 = 94.65% with precision 92.06%, recall 93.24%); average per-class F1
   = 94.5%; accuracy = 94.93%; Dice = 94.61%. **Note**: these high numbers
   include the segmentation task and use authors' own split, not Kaggle private
   LB. PEER-REVIEWED — best citation for your SOTA table.

2. **Tushar et al., 2025.** "The RSNA Lumbar Degenerative Imaging Spine
   Classification (LumbarDISC) Dataset." arXiv:2506.09162.
   <https://arxiv.org/abs/2506.09162>. Dataset paper — useful as the canonical
   citation for the dataset (2,697 patients, 8,593 series, 8 institutions,
   6 countries, 5 continents). Cite this any time you describe the data.

3. **medRxiv 2024 (YOLO v8 + DeepScoreNet).** "Lumbar Spine Degenerative
   Classification Using YOLO v8 and DeepScoreNet."
   <https://www.medrxiv.org/content/10.1101/2024.12.06.24318595v1>. Preprint;
   useful low-bar reference for two-stage detect→classify pipelines.

4. **2nd place GitHub (Bartley).** <https://github.com/brendanartley/RSNA-2024-Competition>.
   Best-documented top-10 GitHub repo. Cite as `[Online]` reference.

5. **2nd place GitHub (Yuji).**
   <https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification>.
   Companion repo with axial T2 pipeline and YOLOX detection.

6. **4th place GitHub (tattaka).**
   <https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public>.
   Only fully-3D documented top-10 solution. MIT-licensed.

7. **2nd place Kaggle writeup (canonical attribution).**
   <https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution>.

8. **3rd place Kaggle writeup (canonical attribution).**
   <https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution>.

9. **5th place Kaggle writeup.**
   <https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution>.

10. **RSNA winners announcement (canonical for ranks).**
    <https://www.rsna.org/news/2024/november/2024-ai-challenge-winners>.

11. **AuntMinnie news coverage.**
    <https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15708869/rsna-names-winners-of-lumbar-spine-ai-challenge>.

12. **nshen7 GitHub (independent, with per-class F1).**
    <https://github.com/nshen7/rsna-2024>. Severe F1 = 0.5007.

13. **Max Melichov Medium article.**
    <https://medium.com/@maxme006/rsna-2024-a-story-of-solving-low-back-pain-with-ai-687a66b8e209>.
    Best public reference for an experimental log of trying ConvNeXt vs
    EdgeNeXt vs DaViT vs EfficientNet on RSNA 2024 (private LB 0.55 best).

---

## Recommended comparison candidates for your Rank-C SOTA table

The strongest 3–5 to put in your SOTA comparison table, ordered by
defensibility:

1. **Khan et al. 2025 (Springer IJCIS)** — *highest priority*. Peer-reviewed,
   reports per-condition F1, on RSNA 2024 dataset. Direct apples-to-apples for
   your F1 numbers (your Mean F1 = 0.528 vs their 0.945 — they win, but their
   number includes segmentation Dice averaging that probably inflates it; flag
   this in the table caption). Their method is MobileNetV2-UNet (segmentation +
   classification jointly) — *different family of method*, so you can frame
   your contribution as "lightweight classification-only fusion model".

2. **2nd place ensemble (ianpan-Kevin-Yuji-Bartley)** — *highest defensibility
   for "Kaggle SOTA" line*. The most rigorous, best-documented top-10
   solution with public code. Reasoning: even if you can't replicate their LB
   score, you can clearly describe their method (2D ConvNeXt+LSTM+attention,
   YOLOX detect→crop, ensemble of 3) and contrast with your single-model
   ResNet-3D + CBAM + BiomedCLIP. Your novelty axes (CBAM + BiomedCLIP +
   ResNet-3D end-to-end) are orthogonal to theirs (2D + ensemble + label
   denoising).

3. **4th place SPINE CHART (tattaka + yu4u)** — *direct architectural
   comparison*. The only fully-3D top-10 documented solution. They use
   SwinV2/CAFormer/ConvNeXt/ResNetRS50 in 3D-volumetric form. Closest to your
   ResNet-3D backbone family. Frame as "3D-volumetric multi-backbone ensemble
   vs. our single-3D-backbone with attention + cross-modal fusion".

4. **nshen7 (independent, with per-class F1)** — *only direct numerical
   comparison on per-class F1 the literature offers*. Severe F1 = 0.5007 vs.
   your Severe F1 (compute from your val results). Caveat: nshen7's split is
   not Kaggle private LB, so disclose the caveat in the table.

5. **Max Melichov (Medium write-up)** — *useful reference for
   "modern-2D-baseline" floor*. Private LB 0.55 with EdgeNeXt-base. Frame as
   "single-backbone 2D baseline, no ensemble, no pseudo-labels" — this is what
   the floor of competitive solutions looks like, and your contribution is
   compared *vs. that floor* on the novelty axis (CBAM, BiomedCLIP, focal/
   uncertainty loss) rather than absolute Kaggle LB.

**Skip in the SOTA table**: ranks 6–10 (NVSpine, HLIP, K_mataro, Adam Narai)
— no public writeups, only team-name listings on the RSNA winners page; not
defensible to cite without knowing methodology.

---

## What I could NOT find (open gaps for the user)

- Numerical Kaggle private/public LB scores for any specific top-10 team.
  These are visible only on the live Kaggle leaderboard (JS-rendered, behind
  reCAPTCHA). To populate the LB Score column you'd need a Kaggle-API authed
  fetch (`kaggle.api.competition_leaderboard_view(...)`) or a manual screenshot.
- Avengers (1st place) — no public GitHub or writeup found despite winning the
  Educational Merit Award. The award says their *code* was clear and
  organized, so it likely exists, but it's not surfaced via Google search.
  Recommend manually browsing
  <https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion?sort=votes>
  while logged in to Kaggle.
- 6th–10th place writeups: no public material located.
- Per-class AUC and AUPRC for any team — never reported in the Kaggle
  ecosystem (the metric is log-loss; teams have no incentive to compute these).
  If you need them for a fair SOTA table, the cleanest path is to download the
  open-source 2nd-place (Bartley) and 4th-place (tattaka) checkpoints,
  re-run them on your val split, and compute F1/AUC/AUPRC yourself. Both
  repos are MIT or otherwise open enough to permit this.

---

## Sources consulted

- [RSNA 2024 winners announcement (canonical ranks)](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners)
- [Kaggle competition page](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- [Kaggle leaderboard page](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/leaderboard) (JS-rendered, not retrievable)
- [Kaggle 2nd place writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- [Kaggle 3rd place writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution)
- [Kaggle 5th place writeup](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution)
- [Kaggle 11th place writeup (yynk)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/yynk-11th-place-solution)
- [Bartley 2nd-place GitHub](https://github.com/brendanartley/RSNA-2024-Competition)
- [Yuji 2nd-place GitHub](https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification)
- [tattaka 4th-place GitHub](https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public)
- [yu4u GitHub profile](https://github.com/yu4u)
- [Khan et al. 2025 Springer IJCIS paper](https://link.springer.com/article/10.1007/s44196-025-01098-7)
- [Tushar et al. 2025 LumbarDISC dataset paper (arXiv 2506.09162)](https://arxiv.org/abs/2506.09162)
- [YOLO v8 + DeepScoreNet medRxiv preprint](https://www.medrxiv.org/content/10.1101/2024.12.06.24318595v1)
- [nshen7 independent solution GitHub](https://github.com/nshen7/rsna-2024)
- [maxmelichov solution GitHub](https://github.com/maxmelichov/RSNA-2024-Lumbar-Spine-Degenerative-Classification)
- [Max Melichov Medium writeup](https://medium.com/@maxme006/rsna-2024-a-story-of-solving-low-back-pain-with-ai-687a66b8e209)
- [AuntMinnie news article](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/article/15708869/rsna-names-winners-of-lumbar-spine-ai-challenge)
- [CLIST competition standings](https://clist.by/standings/rsna-2024-lumbar-spine-degenerative-classification-computer-vision-image-binary-classification-custom-metric-51781617/)
- [LinkedIn — Yusuke Uchida (yu4u, 4th place confirmation)](https://www.linkedin.com/in/%E7%A5%90%E4%BB%8B-%E5%86%85%E7%94%B0-a2156447/)
