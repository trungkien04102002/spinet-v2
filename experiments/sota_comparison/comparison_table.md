# External SOTA Comparison (RSNA-2024, seeds [42, 123, 456])

Mean +/- std over available seeds. `*` marks a cell aggregated from fewer than all requested seeds. `pending` = no runs found yet (GPU training not run).

## Quantitative comparison

| Metric | SpineNetV2 | brendanartley | transformer | Hybrid (Ours) |
|---|---|---|---|---|
| Mean Accuracy | 81.0 +/- 0.6 | pending | pending | 72.4 +/- 0.3 |
| Mean F1 macro | 0.420 +/- 0.010 | pending | pending | 0.527 +/- 0.027 |
| Severe F1 | 0.152 +/- 0.008 | pending | pending | 0.356 +/- 0.017 |
| Severe Recall | 12.3 +/- 0.6 | pending | pending | 48.6 +/- 1.9 |

## Capability comparison (why external SOTA baselines can't do what we do)

| Capability | SpineNetV2 | brendanartley | transformer | Hybrid (Ours) |
|---|---|---|---|---|
| Takes a new label set as input | x (fixed head) | x (fixed head) | x (fixed head) | yes (prompt-based) |
| Cross-dataset transfer (RSNA to SPIDER) | x (requires retrain) | x (requires retrain) | x (requires retrain) | yes |
| Zero-shot to unseen labels | x | x | x | yes |
| RSNA Mean F1 | 0.420 +/- 0.010 | pending | pending | 0.527 +/- 0.027 |
