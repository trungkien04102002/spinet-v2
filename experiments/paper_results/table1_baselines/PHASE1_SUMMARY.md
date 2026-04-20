# Phase 1 Summary — Per-Level Baseline Evaluation

**Completed on:** 2026-04-20
**Branch:** `phase1-baseline-analysis`

## What changed vs the original LaTeX draft

The original LaTeX reported **patient-level** metrics aggregated with the ANY-rule ("if any vertebral level is predicted positive, the patient is positive"). At the advisor's request, this has been redone at **per-level** granularity: each evaluation unit is a `(study_id, condition, level)` triple with a binary label (Moderate|Severe → 1, Normal/Mild → 0).

## Files

| File | Purpose |
|---|---|
| `evaluate_per_level.py` | Evaluator (canonical source) — reads 3 prediction files + GT, emits per-level metrics + evidence. |
| `metrics_per_level/per_level_metrics.csv` | Machine-readable metrics (9 rows = 3 models × 3 conditions). |
| `metrics_per_level/per_level_metrics_report.md` | Human-readable metrics tables + evidence. |
| `metrics_per_level/evidence_cases.csv` | 3 representative patients per model for advisor demo. |
| `metrics_per_level/{ningshen,spinenetv2_upstream,medgemma}_long.csv` | Parsed long tables (debugging / verification). |
| `latex_updated.tex` | Ready-to-paste LaTeX for the thesis/paper (replaces the old LaTeX). |

## Headline results — per-level

### Spinal Canal Stenosis
| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Ning Shen | 935 | 94.76% | 75.00% | 80.00% | **77.42%** |
| SpineNetV2 upstream | 760 | 66.18% | 27.46% | 86.79% | 41.72% |
| MedGemma | 750 | 85.73% | 12.50% | 3.37% | 5.31% |

### Left Foraminal Narrowing
| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Ning Shen | 935 | 85.78% | 64.07% | 74.75% | **69.00%** |
| SpineNetV2 upstream | 760 | 57.63% | 31.82% | 86.42% | 46.51% |
| MedGemma | 750 | 74.00% | 0.00% | 0.00% | 0.00% |

### Right Foraminal Narrowing
| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Ning Shen | 925 | 89.62% | 74.76% | 78.50% | **76.59%** |
| SpineNetV2 upstream | 760 | 53.16% | 28.67% | 78.66% | 42.02% |
| MedGemma | 750 | 77.73% | 10.00% | 0.63% | 1.18% |

## Key insights — differences from the patient-level numbers

1. **SpineNetV2 upstream F1 drops drastically** (was 76–79% patient-level; now 42–47% per-level). Cause: the upstream model does not output per-level predictions — we replicate the patient-ANY prediction across all 5 levels, which inflates False Positives. This is an honest finding, not a bug. Paper narrative now notes this (footnote in `latex_updated.tex`).
2. **SpineNetV2 Recall remains high** (78–87%) because ANY-rule catches the correct patient. The problem is localization: the model says "something is wrong somewhere" but cannot pinpoint which vertebra.
3. **Ning Shen results hold up** at per-level: F1 69–77% vs patient-level 79–81%. Small drop, still best baseline.
4. **MedGemma collapses further** at per-level: F1 ~0–5% vs patient-level 11–31%. Expected — a model that barely predicts positive at patient-level will predict almost nothing at level-level.

## Evidence cases for advisor

Three shared study IDs have predictions in all 3 models — use these for side-by-side demo:

| Study ID | Ning Shen (TP/FP/FN/TN) | SpineNetV2 | MedGemma |
|---|---|---|---|
| **105895264** | 10/2/1/2 | 11/4/0/0 | 2/0/9/4 |
| **100206310** | 6/1/4/4 | 10/5/0/0 | 1/0/9/5 |
| **75336136** | — | — | 0/0/13/2 |

**Recommended for demo:** `105895264` — has abnormal findings in most levels. Shows:
- Ning Shen catches 10/11 true positives with only 2 false positives → precise.
- SpineNetV2 catches all 11 but flags 4 extras → recall high, precision low (ANY-rule artifact).
- MedGemma catches only 2/11 → massively undersensitive.

Per-level breakdown for `105895264` can be reconstructed from `ningshen_long.csv`, `spinenetv2_upstream_long.csv`, `medgemma_long.csv`.

## Reproduction

```bash
cd /Users/kienha/spinet-v2/experiments/paper_results
python3 evaluate_per_level.py
```

All metrics and reports regenerate in `table1_baselines/metrics_per_level/`.

## What's next (Phase 2 Stage A)

- Apply the same per-level evaluation protocol to the `best_model_attention.pth` (CBAM+Focal) checkpoint on its own validation set.
- Compare against the ResNet34 retrained baseline (`best_model.pth`) at per-level.
- This forms **Table 2** of the paper (ablation: baseline vs proposed improvements).

## Known limitations documented for paper

- SpineNetV2 upstream per-level approximation via replicated patient-ANY (noted in LaTeX footnote).
- N differs per model (187/152/150) because each model covers a different subset of RSNA studies; this is reported rather than hidden.
- No McNemar significance testing yet (optional for Rank-C).
