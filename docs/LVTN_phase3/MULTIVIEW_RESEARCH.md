# Multi-Sequence Fusion (T1/T2/axial leader-supporter) — Research + Architecture Plan

> Compiled 2026-07-19 for the advisor's ask (add axial, treat T1/T2 differently
> with per-condition "leader/supporter" weighting). ⚠️ = verify before citing.

## 🔴 Root finding
`rsna_dataloader.py:127-144` (`is_sagittal_t2`) → our pipeline grades **every**
head (incl. foraminal) on **Sagittal T2**. Per RSNA protocol, **foraminal must be
graded on Sagittal T1**. We are reading the wrong sequence for exactly the
condition with our worst precision → cheapest, highest-leverage fix first.

## (a) Per-condition sequence assignment (confirmed)
| Condition | Leader sequence | Source |
|---|---|---|
| Spinal canal stenosis | Sagittal T2 | RSNA LumbarDISC (Richards et al., Radiology:AI, arXiv:2506.09162) |
| Neural foraminal narrowing | **Sagittal T1** (perineural fat) | same + Lee grading AJR 2010 10.2214/AJR.09.2772 |
| Subarticular / lateral recess | **Axial T2** | same + AJR stenosis reviews |

Nuance: not either/or — read together in practice. Correct framing = **primary
(leader) sequence per condition + other sequences as supporter**, learned soft
weighting (not a hard gate). T1 also aids canal (fat next to dura).

## (1) How RSNA-2024 winners route sequences
- **2nd place** (brendanartley, vendored in `experiments/sota_comparison/external/`): canal+subarticular ← Sag-T2; foraminal ← Sag-T1; pipelines independent, combined only at ensemble. **Axial branch was built but dropped — didn't improve team CV.** ⚠️ key lesson: naive axial doesn't help → need selective/gated fusion.
- **3rd place** (Kaggle disc #539453): `SCS = cat(T2, axial_center, stack, level_emb)` — condition-specific concat; weighted CE (Normal 1 / Moderate 2 / Severe 4); axial→level via CenterNet keypoints + line-plane intersection.
- Near-universal convention across top solutions: SCS←SagT2, NFN←SagT1, SS←AxialT2. Late fusion (ranks 1/9/10) vs early/transformer fusion (ranks 3/4).

## (2) Fusion architectures, ranked for our constraints
1. **Late fusion** (per-sequence encoder → concat/attention-pool → per-condition head) — proven by most winners; trivial, reuses our encoder. Week-1 baseline.
2. **Condition-conditioned gated fusion (leader/supporter)** — per-condition learned gate over sequence embeddings. Small MLP, cheap. **This implements the advisor's ask.** Evidence: M-SCAN attention > concat (+~8 AUROC pts); 2nd-place negative axial result argues for selective gating.
3. **Cross-attention (M-SCAN, arXiv:2503.01634, code github.com/Deep-learning-exp/M-SCAN)** — full multi-view cross-attention; their ablation (canal, N=1975): Sag-only concat 89.8%/0.851 → multi-view GRU 92.6%/0.891 → cross-attn 93.8%/**0.971 AUROC**. Medium-high effort. Stretch goal.
4. **Early/channel-stack fusion** — cheapest but CANNOT express per-condition leader/supporter (fusion before heads). Not recommended.

## (3) Leader/supporter, concretely (GMU — Arevalo 2017, arXiv:1702.01992)
Per-condition query-attention over sequence embeddings:
```
a_{c,s} = MLP([e_c ; emb_s])       # e_c = learnable per-condition embedding
w_{c,s} = softmax_s(a_{c,s})       # per-condition weights over {T1, T2sag, T2ax}
fused_c = Σ_s w_{c,s} · emb_s
logits_c = head_c(fused_c)
```
Init gate bias toward clinical prior (T1↑ for foraminal, T2/axial↑ for canal/subarticular) but keep learnable. Ablation to report: flat-concat vs gated vs hard-leader-only. Analog: task-specific weighted fusion arXiv:2307.00885 (has a contribution-visualization figure reusable for the thesis).

## (4) Axial T2 into per-disc pipeline
- **Training crops:** RSNA `train_label_coordinates.csv` gives axial (x,y,instance) for subarticular → extract per-disc axial crops like sagittal, no projection. ⚠️ verify column layout against local file.
- **Inference alignment:** geometric projection (DICOM ImagePositionPatient/Orientation/PixelSpacing → line-plane intersection with axial slices) or a trained axial-point predictor. Effort medium-high (3-5 days, DICOM geometry debugging).
- Pitfalls: axial series↔level messy (1 series all levels vs per-pair series; inconsistent SeriesDescription; T12/L1 & L5/S1 coverage gaps). Defer robust alignment to future work; use given coords for training first.

## (5) Does axial + specialization help Severe/foraminal? — hypothesis, not proven ⚠️
- Mechanistic support strong (foraminal defined by T1 finding we currently lack). But **no clean isolated ablation found** for "axial → Severe-F1 gain."
- Counter-evidence: 2nd-place axial didn't help team CV; PMC11913898 finds sagittal-only AI ≈ radiologist-with-axial for canal. Axial's clearest value is **subarticular**.
- → Strongest lever is **T1-for-foraminal**, not axial. Prioritize T1 fix.

## Recommended staged build (4090, ~2-3 weeks) — see 00_TRACKER §3
- **#0** threshold/calibration (free, no train).
- **#1** T1-foraminal diagnostic (add `is_sagittal_t1`, grade foraminal on T1) — 1-2 days, likely biggest win, isolates "wrong sequence" vs "needs fusion".
- **#2** two-branch late fusion (T2 + T1) — ~1 wk.
- **#3** gated leader-supporter fusion — ~1 wk (the advisor's core ask).
- **#4** axial branch (stretch) — 3-5 days.
Report as ablation rows: T2-only → +T1/flat → +gated → +axial.

## Code refs
- M-SCAN: github.com/Deep-learning-exp/M-SCAN (arXiv:2503.01634)
- brendanartley 2nd place: already vendored in experiments/sota_comparison/external/
- GMU: arXiv:1702.01992 · task-weighted fusion: arXiv:2307.00885 · Disc-centric SSL: arXiv:2602.05738

## ⚠️ Unverified
1st-place exact per-condition fusion (secondhand); `train_label_coordinates.csv` axial columns (verify local); no direct axial→Severe-F1 ablation; Zenn compilation is a secondary machine-translated aggregator.
