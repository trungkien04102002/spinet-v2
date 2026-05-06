# SOTA Deep Research — Master Aggregation

Compiled 2026-05-06. Aggregates 6 sub-agent research outputs:
- Layer 1: `research_kaggle_top.md`, `research_published_papers.md`, `research_methods_survey.md` (broad survey)
- Layer 2: `research_deep_5_papers.md`, `research_papers_2025_2026.md`, `research_foundation_models_spine.md` (deep extraction)

Our reference numbers (RSNA 2024 val, mean over 3 conditions × 3 severities):
**Mean F1 = 0.528, Mean AUC = 0.837, Severe AUPRC = 0.321.**

---

## 1. Top-line findings (executive summary)

| # | Finding | Implication for paper |
|---|---|---|
| 1 | **No published F1/AUC on full RSNA 2024 private LB** (Kaggle metric is log-loss only) | Frame our F1 as one of the first such baselines |
| 2 | **No VLM applied to RSNA 2024 or Pfirrmann MRI grading** in published literature | Claim "first BiomedCLIP-based RSNA solution" |
| 3 | **Nigru NASSJ 2024** covers all 5 of our label-space conditions in one paper | Single most useful comparator for Table 1 |
| 4 | **M-SCAN (arXiv 2503.01634)** uses RSNA 2024, multi-view + cross-attention, AUROC 0.971 canal | Strong direct competitor for our architecture |
| 5 | **Diseases 2026 multi-task** trains on RSNA 2024 + external Phayao validation, F1 0.9558 | Direct apple-to-apple if dichotomous binarization used; flag in footnote |
| 6 | **Hallinan 2021 Radiology** dichotomous κ 0.89-0.96 | Must-cite seminal; not direct numerical comparator |
| 7 | **Spondylolisthesis subtask sparse** in 2025-2026 lumbar MRI DL | Frame as gap; SPIDER zero-shot is contribution |
| 8 | **Foraminal stenosis ~70% Bal. Acc** is the realistic literature ceiling (Nigru) | Defends our weakness — "even SOTA stays at this level on sagittal-only" |

---

## 2. Final SOTA Table 1 — recommended for paper

Sorted by direct comparability with our work. **`[footnote_X]`** marks indicate caveats below the table.

| # | Method | Year | Dataset | Backbone | Mean F1 | AUC / κ | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **Hallinan et al.** *Radiology* | 2021 | 446 internal + 100 ext (private) | Mask R-CNN + 3D CNN | — | κ 0.89-0.96 [‡] | Seminal dichotomous κ. Different metric. |
| 2 | **Jamaludin et al.** *MIA* | 2017 | GENODISC | VGG 2D | — | Lin's CCC 0.91 | Original SpineNet ancestor. |
| 3 | **Windsor et al.** *Sci. Rep.* | 2024 | Multi-cohort | SpineNetV2 (3D ResNet34) | — | — | Direct backbone source; we extend. |
| 4 | **McSweeney et al.** *Spine* | 2023 | NFBC1966 (1,331 pat.) | SpineNetV2 ext. validation | — | Pfirr. Bal. Acc 78%, κ 0.68, Lin's CCC 0.86 | F1 not reported; Pfirrmann SPIDER baseline |
| 5 | **Nigru et al.** *NASSJ* ⭐ | 2024 | 1,747 disks/353 pat. (private) | SpineNetV2 ext. validation | Pfirr. **0.796**, foram. **0.838**, spond. **0.985**, hern. **0.779** | κ 0.46-0.74 | **Best single comparator** — covers all our label spaces |
| 6 | **Lin, Zhang, Shang** *Bioengineering* | 2024 | RSNA-derived 8,160 axial L4-L5 (balanced binary) | CNN + MHSAM + Slot Attn + **CBAM** | **0.945** [†] | 0.951-0.972 | **Closest CBAM twin.** Footnote: balanced single-level binary subset |
| 7 | **Aktan/Khan et al.** *IJCIS* | 2025 | RSNA 2024 (internal split, balanced) | MobileNetV2-UNet (seg+class) | **0.957** [†] | — | Same dataset; aggregate metric, joint seg+class — needs PDF download for per-class |
| 8 | **M-SCAN (Hong et al.)** *arXiv* ⭐ | 2025 | RSNA 2024 (1,975 studies) | Multi-view cross-attention (Sag-T1, Sag-T2/STIR, Ax-T2) + sequence backbone | — | **AUROC 0.971** canal stenosis | Direct architectural competitor — also fuses multi-view |
| 9 | **Phaphuangwittayakul (Diseases 2026)** ⭐ | 2026 | RSNA 2024 + ext. Phayao | VGG19/ConvNeXt-T/DINOv2 features + classical clf | **0.956** ext. | AUC 0.994-1.000 | Multi-task DL with external validation |
| 10 | **Wang et al.** *Eur Spine J* | 2025 | 564 MRIs, 464/100 split | CNN vs Transformer benchmark | — | κ 0.91-0.99 (canal/foram) | CNN > Transformer; both ≥ clinicians |
| 11 | **Liu et al.** *Eur Spine J* | 2025 | 491 pat./2,455 discs (private) | SpineNetV2 ext. validation | — | per-pathology agreement | Same task taxonomy as ours |
| 12 | **AFFM-YOLOv8 (Liu)** *Sci. Rep.* | 2025 | Multicenter 8,428 pat./100k axials | YOLOv8 + AFFM | **0.910** (MSU 11-subtype) | κ > 0.9 vs physician | Best disc-herniation comparator |
| 13 | **YOLOv9-AID (Liu)** *Front. Bioeng.* | 2025 | 1,100 lumbar MRI | YOLOv9 + SCSA + SlideLoss | mAP50 0.828 | — | Multi-task Pfirr+hern+HIZ+Schmorl |
| 14 | **Zhang et al.** *Biomed. Sig. Proc. Ctrl.* | 2025 | Internal + external | nnUNet + sagittal binary + axial multi-label | int. 0.867 / ext. 0.760 | Fleiss κ 0.74-0.86 | 7-lesion multi-label |
| 15 | **Magnani et al.** *BSPC* | 2026 | Single-institution | Unsupervised detect + 8-level Pfirrmann CNN + saliency | Acc 0.932 (±1 grade) | — | Best Pfirrmann-only with explainability |
| 16 | **nshen7** GitHub | 2024 | RSNA 2024 (own held-out) | Faster R-CNN + Swin 2D | Severe **0.501**, Mod 0.472, Norm 0.906 | — | **Only public per-severity F1 on RSNA 2024** |
| 17 | **2nd Place Kaggle** (Bartley et al.) | 2024 | RSNA 2024 LumbarDISC | 2D ConvNeXt+LSTM+attn ensemble | — (log-loss only) | — | Best-documented top-10 Kaggle |
| 18 | **4th Place Kaggle** (tattaka+yu4u) | 2024 | RSNA 2024 LumbarDISC | 3D vol. ensemble (CAFormer+ConvNeXt+ResNetRS50+SwinV2) | — | — | Only fully-3D top-10 |
| 19 | **Acharya & Kansakar** *arXiv* | 2026 | Sagittal T2 (private) | Disc-Centric Contrastive + focal | Bal. Acc 0.781 | — | Closest class-imbalance design |
| 20 | **Bharadwaj et al.** *Eur Radiol* | 2023 | 200 pat./987 axial slices | V-Net + BiT ResNet-50 + DT | — | CCS 0.94, foram. 0.92, facet 0.93 | Interpretable cascade |
| 21 | **Won et al.** *Global Spine J* | 2024-25 | 13,758 axial T2/542 pat. | Faster R-CNN + VGG (3-stage) | 0.784 | Acc 91.5% | 4-class A-D, axial-only |
| 22 | **OURS** (CBAM 3D ResNet34 + BiomedCLIP) ⭐ | 2026 | RSNA 2024 val (full, 3-class multi-level, 1942 samples, 395 pat., random_state=42) | 3D ResNet34 + CBAM + frozen BiomedCLIP, concat-MLP | **0.528** mean | **0.837** mean | Severe AUPRC 0.321; **first BiomedCLIP-based RSNA solution** |

### Footnotes

- `[†]` Number reported on balanced/single-level/binarized subset; not directly comparable to full multi-level 3-class F1.
- `[‡]` Different metric only (κ); cite as qualitative anchor.
- `[*]` In-house dataset; cannot replicate split.
- `[**]` Our split (RSNA val, random_state=42, test_size=0.2, patient-level).
- ⭐ = Strongest comparator candidates for paper Table 1.

### Recommended Table 1 short version (6 rows for paper)

For a tight paper Table 1, pick rows: **5 (Nigru), 6 (Lin), 7 (Aktan), 8 (M-SCAN), 9 (Diseases 2026), 16 (nshen7), 22 (Ours)** — provides RSNA-direct + SPIDER-direct + architectural-twin + per-severity-F1 + foundation-model-context.

---

## 3. Defensible novelty axes (final, with citations)

| Novelty | Defended by (anti-citation) | Strength |
|---|---|---|
| **First BiomedCLIP-based RSNA 2024 solution** | Layer-2 search across 11 VLMs found zero spine-MRI-grading uses | **Strongest** |
| **First per-class AUC + AUPRC baseline on full RSNA 2024** | Layer-1 Kaggle research: 0/10 top teams report F1/AUC | Strong |
| **Frozen VLM + CNN concat-MLP fusion design** | IEEE-J-BHI 2024 (frozen VLM + lightweight clf > full FT for breast cancer); arXiv 2506.18434 (PEFT benchmark); arXiv 2506.14136 (BiomedCLIP imbalance brittleness) | Strong defense |
| **CBAM 3D + focal loss + oversample** | Lin 2024 has CBAM 2D balanced subset; we have CBAM 3D full multi-level | Medium |
| **Zero-shot label extension RSNA → SPIDER via text prompts** | Layer-2 found no comparable VLM-zero-shot work on spine | Strong |
| **Single-model architecture** (vs all top Kaggle ensembles) | 4/4 documented top Kaggle ensemble | Defensive contrast |

---

## 4. Foundation model defense + future work (for Methods/Discussion)

### Drop-in defense paragraph (Methods § "Fusion design rationale")

> We integrate BiomedCLIP \cite{zhang2023biomedclip} as a frozen visual feature
> extractor rather than fine-tuning it end-to-end. This design follows the
> emerging consensus that, on small and class-imbalanced clinical datasets,
> freezing large-scale pretrained vision-language backbones and training only
> a lightweight task-specific head outperforms full fine-tuning, which tends
> to overfit and induce catastrophic forgetting \cite{frozen_vlm_breast2024,
> benchmark_fm_prognosis2025, why_fm_pathology2025}. BiomedCLIP, pretrained
> on 15M PubMed Central image-text pairs, is the most widely adopted
> general-purpose biomedical CLIP variant. We pair the frozen BiomedCLIP
> visual stream with a learnable 3D-CBAM-ResNet34 \cite{cbam2018} that
> captures volumetric, anatomy-specific features that 2D pretrained encoders
> necessarily miss; the two streams are combined through late concat-MLP
> fusion, an architecture analogous to that of \cite{frozen_vlm_breast2024}
> for multimodal breast cancer prediction. Recent analyses of BiomedCLIP
> under high class imbalance \cite{biomedclip_imbalance2025} confirm that
> BiomedCLIP features alone are brittle on the long tail (relevant to our
> Severe class), motivating the additional CNN branch with focal loss and
> oversampling.

### Drop-in future-work paragraph (Discussion § Limitations)

> We acknowledge several alternative foundation models that could plausibly
> improve performance and that we did not benchmark. RadCLIP
> \cite{radclip2024} introduces a slice-pooling attention adapter natively
> suited to volumetric inputs and is the closest 2D-VLM analogue with a
> built-in 3D aggregation. Native 3D-MRI vision-language models — most
> notably Decipher-MR \cite{decipher_mr2026} and MRI-CORE \cite{mricore2025}
> — and 3D-CT analogues such as Merlin \cite{merlin2026} (which outperforms
> BiomedCLIP by 34% on CT zero-shot) suggest that domain-aligned 3D
> pretraining is a stronger prior than our 2D-lifted approach. Recent
> prompt-tuning of BiomedCLIP (BiomedCoOp \cite{biomedcoop2025}) reports
> 5-10% gains with minimal compute. To the best of our knowledge, no public
> study has applied any of these models to the RSNA 2024 lumbar spine task;
> we leave a head-to-head comparison to future work.

---

## 5. Manual download backlog (user action items)

These papers had abstract-level numbers only due to publisher access blocks:

| Paper | Why blocked | URL to manually download | Priority |
|---|---|---|---|
| Hallinan 2021 Radiology | RSNA paywall | https://pubs.rsna.org/doi/10.1148/radiol.2021204289 (institutional access) | Medium — must-cite |
| Aktan 2025 IJCIS | Springer hard-block | https://link.springer.com/content/pdf/10.1007/s44196-025-01098-7.pdf (open access in browser) | **High** — only peer-reviewed RSNA 2024 method paper |
| Joliat 2025 Eur Spine J | Springer block | https://link.springer.com/article/10.1007/s00586-025-08900-2 | Low |
| Liu 2025 SpineNetv2 ext. val. | Springer block | https://link.springer.com/article/10.1007/s00586-025-09543-z | Medium |
| Wang 2025 CNN vs Transformer | Springer block | https://link.springer.com/article/10.1007/s00586-025-09443-2 | Medium |

---

## 6. Recommended action items (ranked by Rank-C impact)

| Rank | Action | Effort | Rank-C boost |
|---|---|---|---|
| 1 | Insert SOTA Table 1 into paper (rows 5,6,7,8,9,16,22) with footnotes | ~3h | **+15%** |
| 2 | Manual download Aktan 2025 IJCIS PDF, extract per-class F1 | ~1h | +5% |
| 3 | Run nshen7 inference on our val split for direct F1 comparison | ~3h | +5% |
| 4 | Compute Hallinan-style dichotomous κ on our predictions | ~2h | +5% |
| 5 | 3-seed Hybrid retrain on Vast | ~10h, ~$3 | +10% |
| 6 | Mendeley deep-dive (separate research) | ~2h research | conditional |

---

## 7. Reading priority for user

1. `paper/PAPER_CITATIONS.md` — flat lookup of all papers with name + link + key numbers
2. `paper/SOTA_DEEP_RESEARCH.md` (this file) — analysis + recommended Table 1
3. `paper/research_deep_5_papers.md` — full extracted tables for top 5 comparators
4. `paper/research_papers_2025_2026.md` — 2025-2026 newer papers
5. `paper/research_foundation_models_spine.md` — defense for BiomedCLIP choice
6. `paper/research_kaggle_top.md` — Kaggle top solutions context
7. `paper/research_published_papers.md` — original broad survey (Layer 1)
8. `paper/research_methods_survey.md` — 3D backbone landscape
