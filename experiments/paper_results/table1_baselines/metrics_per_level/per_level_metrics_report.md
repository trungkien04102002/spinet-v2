# Per-Level Evaluation Results

Granularity: per (study_id, condition, level). Binary: Moderate|Severe -> 1.


## Spinal Canal Stenosis

| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| NingShen | 935 | 94.76% | 75.0% | 80.0% | 77.42% |
| SpineNetV2 upstream | 760 | 66.18% | 27.46% | 86.79% | 41.72% |
| MedGemma | 750 | 85.73% | 12.5% | 3.37% | 5.31% |

## Left Foraminal Narrowing

| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| NingShen | 935 | 85.78% | 64.07% | 74.75% | 69.0% |
| SpineNetV2 upstream | 760 | 57.63% | 31.82% | 86.42% | 46.51% |
| MedGemma | 750 | 74.0% | 0.0% | 0.0% | 0.0% |

## Right Foraminal Narrowing

| Model | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| NingShen | 925 | 89.62% | 74.76% | 78.5% | 76.59% |
| SpineNetV2 upstream | 760 | 53.16% | 28.67% | 78.66% | 42.02% |
| MedGemma | 750 | 77.73% | 10.0% | 0.63% | 1.18% |

## Evidence cases (top 3 per model by abnormal positives)

| Model | Study ID | TP | FP | FN | TN | Total |
|---|---|---|---|---|---|---|
| NingShen | 105895264 | 10 | 2 | 1 | 2 | 15 |
| NingShen | 245660566 | 9 | 0 | 1 | 5 | 15 |
| NingShen | 100206310 | 6 | 1 | 4 | 4 | 15 |
| SpineNetV2 upstream | 105895264 | 11 | 4 | 0 | 0 | 15 |
| SpineNetV2 upstream | 1038453736 | 11 | 4 | 0 | 0 | 15 |
| SpineNetV2 upstream | 100206310 | 10 | 5 | 0 | 0 | 15 |
| MedGemma | 75336136 | 0 | 0 | 13 | 2 | 15 |
| MedGemma | 105895264 | 2 | 0 | 9 | 4 | 15 |
| MedGemma | 100206310 | 1 | 0 | 9 | 5 | 15 |
