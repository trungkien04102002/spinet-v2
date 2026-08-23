# Paper Results — Centralized Artifacts

This directory consolidates all inputs and outputs needed to build the two paper tables (Table 1: baseline comparison, Table 2: ablation). Files are **copies** of originals — changes here will NOT propagate back. If you regenerate anything in the source repositories, re-copy here.

## Source repositories

| What | Where (canonical source) |
|---|---|
| Training code (baseline, CBAM+Focal) | `/Users/kienha/spinet-v2/` |
| Third-party model predictions + original ground truth | `/Users/kienha/thesis-experiments/detect-abnormal/` |
| Trained checkpoints | `/Users/kienha/spinet-v2/checkpoints/` (gitignored, NOT copied here) |

## Layout

```
paper_results/
├── README.md                                  (this file)
├── table1_baselines/                          Table 1: 3 third-party models vs GT
│   ├── ground_truth.csv                       RSNA train.csv (569 KB)
│   ├── predictions/                           Raw prediction files (3 models)
│   │   ├── spinenetv2_upstream.csv            SpineNetV2 upstream inference
│   │   ├── medgemma.json                      MedGemma zero-shot prompted output
│   │   └── ningshen.csv                       Ning Shen Kaggle 2024 submission
│   └── metrics/                               Computed metrics (CSV)
│       ├── per_model_independent.csv          N differs per model (SpineNetV2 152, MedGemma 150, NingShen 187) — matches LaTeX draft
│       ├── inner_join_N47.csv                 Inner-join across 3 models, N=47 (fair comparison)
│       ├── inner_join_N47_alt.csv             Alternative inner-join with different numbers — needs investigation
│       └── merged_binary_predictions.csv      Per-patient binary predictions for all 3 models
│
├── table2_ablation/                           Table 2: our baseline vs +CBAM+Focal
│   └── ivd_metrics/                           Per-IVD (9700 samples) per-class metrics
│       └── 01_baseline.txt                    SpineNetV2 ResNet34 no CBAM no Focal — DONE
│       └── 02_cbam_focal.txt                  CBAM+Focal (to be generated in Stage A of Phase 2)
│       └── 03_cbam_only.txt                   (optional ablation)
│       └── 04_focal_only.txt                  (optional ablation)
│
└── logs/                                      Raw training logs (optional)
```

## Table 1 — numbers at a glance

From `table1_baselines/metrics/per_model_independent.csv` (matches current LaTeX draft):

| Model | Condition | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| SpineNetV2 (upstream) | Spinal Canal | 152 | 81.58 | 77.61 | 80.00 | 78.79 |
| SpineNetV2 (upstream) | Left Foraminal | 152 | 75.66 | 71.59 | 84.00 | 77.30 |
| SpineNetV2 (upstream) | Right Foraminal | 152 | 72.37 | 75.56 | 77.27 | 76.40 |
| MedGemma | Spinal Canal | 150 | 66.00 | 57.14 | 22.22 | 32.00 |
| MedGemma | Left Foraminal | 150 | 47.33 | 62.50 | 12.05 | 20.20 |
| MedGemma | Right Foraminal | 150 | 46.67 | 71.43 | 6.02 | 11.11 |
| Ning Shen | Spinal Canal | 187 | 86.63 | 81.82 | 80.60 | 81.20 |
| Ning Shen | Left Foraminal | 187 | 77.01 | 72.81 | 87.37 | 79.43 |
| Ning Shen | Right Foraminal | 187 | 78.61 | 78.57 | 84.62 | 81.48 |

**⚠️ Known issues:**
1. N differs per model (152/150/187) — **not a fair comparison**. The "150" stated in the LaTeX narrative is MedGemma's N only.
2. Inner-join version (N=47) exists but has different numbers — see `inner_join_N47.csv`.
3. Two different inner-join files (`inner_join_N47.csv` vs `inner_join_N47_alt.csv`) give different results for same N=47 — one of the two has a bug or uses a different threshold; needs investigation before citing.

## Table 2 — current status

`01_baseline.txt` (our retrained ResNet34 baseline, no CBAM, no Focal):

| Condition | Overall Acc | Severe Recall | Severe F1 |
|---|---|---|---|
| Spinal Canal | 88.78% | 31.2% | 40.8% |
| Left Foraminal | 78.18% | 0.0% | 0.0% |
| Right Foraminal | 78.55% | 0.0% | 0.0% |

`02_cbam_focal.txt` — **NOT YET GENERATED** (see Phase 2 Stage A).

## Re-copy command

Run this from `/Users/kienha/spinet-v2/` to refresh copies from sources:

```bash
cp /Users/kienha/thesis-experiments/detect-abnormal/prediction/all_results_spinnet_v2.csv experiments/paper_results/table1_baselines/predictions/spinenetv2_upstream.csv
cp /Users/kienha/thesis-experiments/detect-abnormal/prediction/med_gemma_result.json experiments/paper_results/table1_baselines/predictions/medgemma.json
cp /Users/kienha/thesis-experiments/detect-abnormal/prediction/ningshen_submission.csv experiments/paper_results/table1_baselines/predictions/ningshen.csv
cp /Users/kienha/thesis-experiments/detect-abnormal/ground_truth/train.csv experiments/paper_results/table1_baselines/ground_truth.csv
cp /Users/kienha/thesis-experiments/detect-abnormal/outputs/all_models_independent_metrics.csv experiments/paper_results/table1_baselines/metrics/per_model_independent.csv
cp /Users/kienha/thesis-experiments/detect-abnormal/outputs/all_conditions_metrics.csv experiments/paper_results/table1_baselines/metrics/inner_join_N47.csv
cp /Users/kienha/thesis-experiments/detect-abnormal/outputs/detailed_metrics.csv experiments/paper_results/table1_baselines/metrics/inner_join_N47_alt.csv
cp /Users/kienha/thesis-experiments/detect-abnormal/outputs/merged_binary_predictions.csv experiments/paper_results/table1_baselines/metrics/merged_binary_predictions.csv
cp experiments/rsna/base_line.txt experiments/paper_results/table2_ablation/ivd_metrics/01_baseline.txt
```

## Status tracker

- [x] Table 1 baseline predictions copied (3 models)
- [x] Table 1 metrics copied (per-model independent + inner-join variants)
- [x] Table 2 baseline IVD metrics copied (`01_baseline.txt`)
- [ ] Table 2 CBAM+Focal IVD metrics **(Phase 2 Stage A)**
- [ ] Table 1 N-discrepancy resolved
- [ ] Table 2 ablation variants (CBAM-only, Focal-only) — optional
- [ ] Final LaTeX tables written
