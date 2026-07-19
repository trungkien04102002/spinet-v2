# Doctor-Feedback Loop — Literature Research (for Future Work + advisor discussion)

> Compiled 2026-07-18 via web research for the LVTN feedback-loop (radiologist
> corrects AI grade/mask → correction_log → periodic retrain). This is FUTURE
> WORK scope (not built this semester). **Verification note:** several citations
> below were gathered from search snippets/abstracts, not full-text fetches —
> items marked ⚠️ MUST be re-verified (exact author/year/venue) before quoting in
> the thesis.

## 1. Where it sits (taxonomy + surveys)
The loop intersects: **human-in-the-loop (HITL)** deep learning, **active learning**,
**learning-from-label-corrections / label-noise**, and **continual learning**.

- **Budd, Robinson, Kainz (2021)** — *A survey on active learning and HITL deep learning for medical image analysis*, Medical Image Analysis 71:102062 (arXiv:1910.02923). Foundational taxonomy; argues the clinician stays in the loop post-deployment. ⚠️ ScienceDirect abstract fetch was blocked — verify wording.
- **Deep active learning in medical image analysis survey (2024)** — MedIA / arXiv:2310.14230 (curated code list: github.com/LightersWang/Awesome-Active-Learning-for-Medical-Image-Analysis). Updated 2021–2024 view.
- **Continual-learning-in-medical-imaging surveys (2024)** — arXiv:2405.13482, arXiv:2312.17004. Catastrophic forgetting = still the main open problem.

## 2. Mechanisms + trade-offs
**Batch/offline retrain vs online:** batch is the only credible clinical choice
(monitor → accumulate → retrain job → validate → swap). Online/incremental risks
catastrophic forgetting + regulatory re-validation. Precedent: Pianykh, Langs,
Dewey et al., *Continuous Learning AI in Radiology*, Radiology 2020;297(1):6–14 ⚠️.

**Active learning sample selection** (matters for triage — which cases to review
first / which corrections matter most):
- **Uncertainty sampling** (aleatoric vs epistemic; MC-dropout/ensembles) — Gal, Islam, Ghahramani, *Deep Bayesian Active Learning*, ICML 2017 (arXiv:1703.02910). Cheapest, best-supported for medical classification. **MONAI Label** (Diaz-Pinto et al., arXiv:2203.12362, MedIA 2024) is the closest open-source prior-art tool (same category as our app; computes aleatoric+epistemic uncertainty to triage).
- Query-by-committee / coreset (Sener & Savarese, ICLR 2018) — academically nice but too costly for single-4090, small-N. Skip.
- **Cold-start risk:** small/biased initial pool can make AL *worse than random* (COLosSAL, MICCAI 2023, arXiv:2307.12004). State as a limitation.

**Label noise (corrections are NOT gold truth):** Pfirrmann inter-rater kappa ≈
0.5–0.8 across studies ⚠️ (aggregated, pin a specific source before citing).
→ **Confident Learning** (Northcutt, Jiang, Chuang, JAIR 2021 70:1–55; code:
github.com/cleanlab/cleanlab) to filter noisy corrections before retraining.

**Catastrophic forgetting mitigations (small grading head):**
- **EWC** — Kirkpatrick et al., PNAS 2017 (arXiv:1612.00796). Fisher-weighted penalty keeping weights near pre-correction values. Cheap to add.
- **LwF** — Li & Hoiem, TPAMI 2018 (ECCV 2016). Distill old-model outputs; no old data needed.
- **Rehearsal/replay** — mix a buffer of original training samples into each fine-tune batch. Simple, effective; privacy caveat doesn't apply (single-site, we keep the training set).
- **Frozen BN running stats + EWC** — arXiv:2011.08096 ⚠️ reports the combination "completely mitigates" forgetting. Very low-effort with our BatchNorm3d backbone.
- **Simplest fallback:** freeze backbone, head-only fine-tune, small LR, early stop — *already our `--freeze-backbone` pattern in train_spider.py*. Most defensible.

**Retrain trigger:** N-correction-count threshold (e.g. 50–100 new corrections)
is more defensible than performance-drop (we have no live ground-truth signal).
MedMLOps framework, European Radiology 2025 (PMC12559124) ⚠️ for the monitoring/
re-validation framing. Closest real deployment analogue: VIOLA-AI / NeoMedSys,
arXiv:2505.09380 (2025) — radiologist review → iterative refinement, 2 hospitals.

## 3. Evaluation (how to prove it helped)
1. **Before/after on a FROZEN held-out test set** (never only on corrected cases — circular).
2. **Agreement-with-expert = weighted Cohen's kappa** (comparable to published inter-rater ~0.5–0.8), not just accuracy/F1.
3. **Learning curve** (perf vs #corrections used).
4. **Per-class recall/F1** (our convention; Severe class is the one most corrected + most at risk of regression).
5. **Forgetting check** — original held-out perf before vs after each cycle (most theses skip this; surveys treat it as mandatory).

## 4. Simplest credible loop to propose (feasibility: small head, 1×4090)
1. Gather N corrected cases from `correction_log` (old→new grade).
2. Light noise filter (Cleanlab / second-reviewer agreement).
3. Build fine-tune set (label = corrected grade).
4. Batch fine-tune offline: freeze backbone + frozen BN stats, head-only, + EWC penalty vs original RSNA/SPIDER distribution, small LR, early stop.
5. Eval before/after on frozen held-out (per-class F1/recall + weighted kappa) AND forgetting check on original test.
6. Swap only if both pass a threshold; else accumulate more.

**Pitfalls to state (all literature-backed):** tiny/biased correction set
(cold-start), forgetting (must check, not assume), correction label noise
(~0.5–0.8 kappa), mask-edits harder to use for grading head → defer to a separate
segmentation-retrain loop, no live perf signal → count-based trigger only.

## 5. Public code
- MONAI Label — github.com/Project-MONAI/MONAILabel (closest prior-art tool)
- Cleanlab (confident learning) — github.com/cleanlab/cleanlab
- Core-set AL — github.com/ozansener/active_learning_coreset
- nnU-Net — github.com/MIC-DKFZ/nnUNet

## 6. One-paragraph Future Work draft (ready to adapt)
> The versioned correction log enables a future closed-loop retraining pipeline:
> batch (not online) fine-tuning of only the grading head on accumulated
> corrections, using frozen backbone batch-norm statistics plus an EWC term to
> prevent catastrophic forgetting of the base RSNA/SPIDER distribution
> (Kirkpatrick et al., 2017), triggered once a fixed number of corrections
> accumulate. Evaluation reports per-class F1/recall and weighted kappa on a
> frozen held-out test set before/after each cycle, explicitly checking for
> regression to rule out forgetting. Given Pfirrmann inter-rater variability
> (kappa ≈ 0.5–0.8) and the small, non-representative size of an initial
> correction pool (a cold-start failure mode, arXiv:2307.12004), corrections are
> treated as noisy signal (confident-learning filtering; Northcutt et al., 2021)
> rather than gold truth. Mask-edit corrections are deferred to a separate future
> segmentation-retraining loop.
