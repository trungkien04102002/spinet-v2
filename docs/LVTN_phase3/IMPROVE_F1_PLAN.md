# Improve F1 vs. Feedback-Loop — Research Synthesis + Action Plan

> Compiled 2026-07-18 from 3 parallel research/diagnosis agents. Question the
> user asked: (1) is building the doctor-feedback retrain loop feasible in the
> 15-week timeline? (2) if not, how else can we raise the still-low Severe F1?
> **Answer: skip the retrain loop (future work), spend the effort on F1 —
> starting with a near-free threshold/calibration fix.**

## Diagnosis — where F1 is actually lost (from real 3-seed metrics)
Hybrid model, per-condition × per-class:
- **Worst = foraminal Severe** (F1 0.27–0.29), bottleneck is **PRECISION** (0.22–0.23), not recall — the model **over-calls Severe**. Recall is decent (0.34–0.70).
- **AUC–F1 gap is huge** (Severe AUC 0.899 vs F1 0.356). AUC = ranking quality (good); F1 at argmax = decision rule (bad for a rare class). → **primarily a threshold / operating-point problem**, not missing representation.
- spinal_canal is easiest; foraminal branches have genuinely weaker separability too (AUPRC 0.21–0.23 vs 0.56 for canal) → some representation headroom there.
- **spinal_canal Moderate low F1 = seed-instability artifact** (seed 123 recall collapses to 0.083 vs ~0.4 on other seeds), not a structural weakness. Worth investigating stabilization (more epochs / better early-stop / seed-robust selection).

## The domain says argmax is wrong
RSNA 2024's own official metric weights Normal-Mild : Moderate : Severe errors **1 : 2 : 4** — the standard cost function itself says the default equal-cost argmax decision rule is wrong for this task. This makes threshold/prior-correction the principled first move, not an ad-hoc knob.

## Ranked plan to raise Severe F1 (gain ÷ effort)

| # | Move | Why | Effort | Retrain? |
|---|---|---|---|---|
| **1 (do first)** | **Per-class threshold optimization + calibration** on existing 3-seed checkpoints: coordinate-descent thresholds to maximize macro-F1 / match the 1:2:4 cost; also try post-hoc **logit adjustment / Balanced Softmax** (Menon 2020, Ren 2020) and **τ-normalization** of classifier weights (Kang 2020). | Exactly matches the high-AUC/low-F1 diagnosis; likely recovers a big chunk for free. Also a diagnostic: if it doesn't close the gap, the rest is representation. | Hours | **No** |
| 2 | **cRT** — freeze backbone, retrain only classifier head with class-balanced sampling (Kang 2020, facebookresearch/classifier-balancing) | Fixes the mechanistic cause (classifier weight-norms track class frequency); backbone already trained. | ~1 day | head only |
| 3 | **Self-supervised contrastive pretraining** on RSNA disc ROIs then fine-tune (SimCLR-style) | Only lever with dataset-matched published evidence: Acharya & Kansakar 2026 (arXiv:2602.05738) report Severe recall 73.4% + catastrophic Severe→Normal miss 5.32%→2.13% on the SAME RSNA cohort. | Days (fits 4090) | full |
| 4 | **LDAM-DRW or Balanced-Softmax loss** replacing/augmenting focal (Cao 2019 kaidic/LDAM-DRW; Ren 2020) | Margin/prior-based imbalance handling not yet tried; complements focal. | 1–3 days ×3 seeds | yes |
| 5 | **Multi-view fusion (add axial T2)** | Highest ceiling — only lever adding NEW information; M-SCAN (arXiv:2503.01634) hits AUROC 0.971 on matched cohort with axial+sagittal. But real sub-project (axial→disc alignment). | 1–2 weeks | yes |

Secondary/stackable: Balanced-MixUp (Galdran MICCAI 2021, medical ordinal precedent), ordinal heads CORAL/CORN (double-edged — can pull rare-class predictions to the middle; only with rebalancing), seed-ensemble logits + per-class thresholds. Avoid: naive TTA (can hurt medical), pseudo-labeling (amplifies majority bias). Already tried (don't re-pitch as novel): focal, uncertainty weighting, oversampling, standard aug, CBAM+BiomedCLIP.

**Single cheapest highest-leverage move:** post-hoc per-class threshold sweep / logit-adjustment / τ-norm on the already-trained 3-seed checkpoints — NO retraining. Run this first; the result is itself diagnostic.

## Feedback-loop — verdict: FUTURE WORK (don't build the retrain half)
- **Capture already done + tested** (`correction_log`, versioned annotations) → satisfies the advisor's "do capture first, not an auto-evaluator." Priority-1 half is complete.
- Retrain half (EWC + frozen-BN + forgetting-eval + model-swap + simulated corrections) = **~10–15 person-days of risky ML eng** competing for the same 4090 as SOTA. No real doctors → only a caveated *mechanism* demo, weak evidence.
- **Opportunity cost:** F1 gains land on the thesis's central claim and are uncontestable; a simulated feedback loop is a hedged secondary result. Spend the slack on F1 + de-risking SOTA + writing buffer.
- **Cheap partial-credit (~1 day) if visible P4 progress wanted:** build only the "corrections → fine-tune dataset builder" (reuses crops.py + correction_log; invert severity→class-idx) to prove captured corrections are usable as training data; write EWC/forgetting design as future-work pseudocode (already drafted in FEEDBACK_LOOP_RESEARCH.md §6).

## Recommended sequence
1. **Now (free):** build the threshold/logit-adjustment/τ-norm sweep script, run on existing 3-seed checkpoints, report new Severe/macro F1. (Also report RSNA 1:2:4 cost metric.)
2. If more headroom wanted: cRT (~1 day) → SSL pretraining (days) → LDAM-DRW. Multi-view only if time.
3. Feedback loop: keep as future work; optionally the 1-day dataset-builder for narrative.
4. All new numbers get their own before/after table; keep the existing 3-seed protocol + frozen split.
