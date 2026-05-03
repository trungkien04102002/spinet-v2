# V3 Retrain Results Log — 2026-05-03

Backup raw outputs từ Vast.ai. Append từng run khi xong. Survive compaction + session restart vì check vào git.

---

## Run 1: Baseline RSNA — DONE 2026-05-03

**CMD:**
```
python3 train_rsna_baseline.py --epochs 25 --batch-size 64 --lr 1e-3 \
    --save-dir checkpoints/v3_20260503/baseline
```
*(chạy thực tế: 20 epoch, dừng tự nhiên hoặc early-stop, best @ epoch 18)*

**Raw `cat baseline_best_metrics.txt`:**
```
=== BEST MODEL METRICS (baseline) ===
Saved at:  2026-05-03T14:44:03
Epoch:     18
Train Loss: 0.4435
Val Loss:   0.4596
Avg Severe F1: 0.1493

Validation Accuracies:
  spinal_canal       0.8950
  left_foraminal     0.7790
  right_foraminal    0.7656

Per-Class Metrics:

  spinal_canal:
    Class             Prec Recall     F1  Support
    Normal/Mild      0.909  0.987  0.946     1722
    Moderate         0.379  0.079  0.131      139
    Severe           0.636  0.346  0.448       81

  left_foraminal:
    Class             Prec Recall     F1  Support
    Normal/Mild      0.812  0.950  0.875     1497
    Moderate         0.470  0.242  0.319      360
    Severe           0.000  0.000  0.000       80

  right_foraminal:
    Class             Prec Recall     F1  Support
    Normal/Mild      0.800  0.951  0.869     1482
    Moderate         0.423  0.199  0.271      372
    Severe           0.000  0.000  0.000       83

AUC / AUPRC / Brier (per class, one-vs-rest):

  spinal_canal:
    Class              AUC  AUPRC  Brier  Support
    Normal/Mild      0.892  0.984  0.069     1722
    Moderate         0.851  0.272  0.058      139
    Severe           0.922  0.513  0.028       81
    macro            0.888  0.590

  left_foraminal:
    Class              AUC  AUPRC  Brier  Support
    Normal/Mild      0.794  0.932  0.143     1497
    Moderate         0.758  0.387  0.134      360
    Severe           0.815  0.147  0.037       80
    macro            0.789  0.489

  right_foraminal:
    Class              AUC  AUPRC  Brier  Support
    Normal/Mild      0.802  0.929  0.143     1482
    Moderate         0.760  0.375  0.136      372
    Severe           0.842  0.167  0.038       83
    macro            0.801  0.490

AUC / AUPRC overall (averaged across 3 conditions):
  macro AUC      : 0.826
  macro AUPRC    : 0.523
  popular AUPRC  : 0.948  (Normal/Mild)
  rare    AUPRC  : 0.310  (Moderate + Severe)
  Severe  AUPRC  : 0.276  (clinical priority)

Extra:
  val_weighted_logloss: 0.4599203933041056
  best_path: checkpoints/v3_20260503/baseline/best_model.pth
  total_train_seconds: 986.3957138061523
  avg_epoch_seconds: 54.799761878119575
  eval_seconds: 10.897956609725952
  eval_samples: 1942
  eval_throughput_samples_per_sec: 178.1985439606953
  eval_ms_per_sample: 5.61171813065188
```

**Aggregated numbers (filled into report Bảng 1A/1C Base column):**
- Mean AUC macro: **0.826**
- Mean AUPRC macro: **0.523**
- Severe AUC: **0.860** (avg of 0.922, 0.815, 0.842)
- Severe AUPRC: **0.276**
- Popular AUPRC: **0.948**
- Rare AUPRC: **0.310**
- Total train time: 16.4 min (986s, 20 epochs, ~55s/epoch)
- Eval throughput: 178 samples/s, 5.6 ms/sample

**Status:** ✓ Filled into THESIS_REPORT_DRAFT_v3.md (Bảng 1A + Bảng 1C Base column, commit pending).

---

## Run 2: CBAM RSNA — IN PROGRESS

**CMD:**
```
python3 train_rsna_attention.py --epochs 20 --batch-size 64 --lr 1e-3 \
    --use-focal --use-uncertainty true --class-weight-mode sqrt \
    --augmentation medium --oversample-factor 5 \
    --save-dir checkpoints/v3_20260503/cbam
```

**Progress notes:**
- Epoch 2: avg_severe_f1=0.094 (still warming up, foraminal Severe F1=0.000)
- Epoch 3: avg_severe_f1=**0.184** (over-correction phase, Severe Recall=0.79 spinal but Precision=0.135)
  - L-Foram Severe F1: 0.000 → 0.203 (breakthrough vs baseline)
  - R-Foram Severe F1: 0.000 → 0.118 (breakthrough vs baseline)
- Epoch time ~245s = ~4 min/epoch → 20 ep ≈ 80 min total

**Raw cat (paste khi xong):**
```
[TODO]
```

---

## Run 3-9: Hybrid + linear probe — PENDING

| # | Run | Status |
|---|---|---|
| 3 | Hybrid full | pending |
| 4 | Hybrid `--ablate-branch cbam_only` | pending |
| 5 | Hybrid `--ablate-branch biomedclip_only` | pending |
| 6 | Hybrid `--fusion gated` | pending |
| 7 | Hybrid `--modality-dropout 0.15` | pending |
| 8 | Linear probe BiomedCLIP | pending |
| 9 | Linear probe ImageNet ViT | pending |
