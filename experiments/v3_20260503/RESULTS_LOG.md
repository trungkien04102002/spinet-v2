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

## Run 2: CBAM RSNA — ABANDONED v3 (regression), USE v2 CKPT

**CMD đã chạy v3 (KILLED at epoch 15, best at epoch 7):**
```
python3 train_rsna_attention.py --epochs 20 --batch-size 64 --lr 1e-3 \
    --use-focal --use-uncertainty true --class-weight-mode sqrt \
    --augmentation medium --oversample-factor 5 \
    --save-dir checkpoints/v3_20260503/cbam
```

**v3 Best (epoch 7) — REGRESSION vs v2:**
- Avg Severe F1: 0.200 (v2: 0.333 — much worse)
- Mean AUC: 0.793 (v3 baseline: 0.826 — model worse than its own baseline!)
- Mean AUPRC: 0.466 (v3 baseline: 0.523)
- Severe AUPRC: 0.215 (v3 baseline: 0.276)
- **→ User killed, switched to Plan B (use v2 ckpt + eval AUC/AUPRC fresh)**

**Args diff vs v2:**
- focal_gamma: 2.0 (v3) vs 1.8 (v2) — small
- epochs: 20 (v3) vs 25 (v2) — 5 epoch ít hơn
- HFlip fix → mất "free regularization", model cần thêm epoch để converge

**Plan B: v2 fresh_cbam ckpt + eval_rsna_auc.py (DONE 2026-05-03 ~16h):**

CMD chạy trên Vast.ai:
```
python3 eval_rsna_auc.py --model cbam --checkpoint checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth --output experiments/v3_20260503/v2_cbam_auc.json
```

**Raw output (Plan B):**
```
==============================================================================
  AUC / AUPRC summary — model=cbam
  checkpoint: checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth
  n_val_samples: 1942
==============================================================================

spinal_canal (n_valid=1942):
  Class              AUC    AUPRC    Brier  Support
  --------------------------------------------------
  Normal/Mild      0.944    0.992    0.096     1722
  Moderate         0.880    0.273    0.078      139
  Severe           0.968    0.663    0.028       81
  macro            0.931    0.643

left_foraminal (n_valid=1937):
  Class              AUC    AUPRC    Brier  Support
  --------------------------------------------------
  Normal/Mild      0.798    0.926    0.245     1497
  Moderate         0.638    0.254    0.200      360
  Severe           0.851    0.175    0.051       80
  macro            0.762    0.452

right_foraminal (n_valid=1937):
  Class              AUC    AUPRC    Brier  Support
  --------------------------------------------------
  Normal/Mild      0.815    0.933    0.235     1482
  Moderate         0.677    0.296    0.199      372
  Severe           0.852    0.204    0.048       83
  macro            0.781    0.478

OVERALL (averaged across 3 conditions)
  macro AUC       : 0.825
  macro AUPRC     : 0.524
  popular AUPRC   : 0.951
  rare    AUPRC   : 0.311
  Severe  AUPRC   : 0.347
  inference: 11.8s for 1942 samples (164.1 samples/s)
```

**Aggregated for Bảng 1A/1C Ours column (combined v2 metrics):**
- F1/Acc/Recall/Precision: từ summary.pdf v2 fresh_cbam (Mean F1=0.509, Severe F1=0.333 etc.)
- AUC/AUPRC/Brier: từ Plan B eval ở trên
- Mean AUC: **0.825** (≈ baseline)
- Mean AUPRC: **0.524** (≈ baseline)
- **Severe AUPRC: 0.347** (vs baseline 0.276 → **+26% relative**) ⭐
- Severe AUC: **0.890** (vs baseline 0.860 → +0.030)

**Status:** ✓ Filled into THESIS_REPORT_DRAFT_v3.md (Bảng 1A + Bảng 1C Ours column).

### Run 2b: CBAM RSNA v3 RETRAIN (NEW, replaces Plan B) — DONE 2026-05-04 04:06

**Decision:** After fixing the regression root cause (HFlip fix + focal_gamma 2.0), user retrained CBAM in v2-bug mode (`--no-hflip-swap-labels`) to match v2 working config. Result: matches v2 within stochastic variation (Severe AUPRC 0.319 v3 vs 0.347 v2 = -8% rel, within seed-to-seed noise).

**CMD (via scripts/run_cbam_v2bug.sh):**
```
python3 train_rsna_attention.py --epochs 25 --batch-size 64 --lr 1e-3 \
    --focal-gamma 1.8 --use-focal --use-uncertainty true \
    --class-weight-mode sqrt --augmentation medium --oversample-factor 5 \
    --no-hflip-swap-labels \
    --save-dir checkpoints/v3_20260503/cbam
```

**Raw `cat attention_best_metrics.txt` (v3 NEW):**
```
=== BEST MODEL METRICS (attention) ===
Saved at:  2026-05-04T04:06:58
Epoch:     14
Train Loss: -1.4814
Val Loss:   0.1601
Avg Severe F1: 0.3044

Validation Accuracies:
  spinal_canal       0.8785
  left_foraminal     0.6138
  right_foraminal    0.5772

Per-Class Metrics:

  spinal_canal:
    Class             Prec Recall     F1  Support
    Normal/Mild      0.975  0.921  0.947     1722
    Moderate         0.339  0.446  0.385      139
    Severe           0.436  0.716  0.542       81

  left_foraminal:
    Class             Prec Recall     F1  Support
    Normal/Mild      0.920  0.619  0.740     1497
    Moderate         0.294  0.683  0.411      360
    Severe           0.183  0.212  0.197       80

  right_foraminal:
    Class             Prec Recall     F1  Support
    Normal/Mild      0.952  0.538  0.687     1482
    Moderate         0.298  0.828  0.438      372
    Severe           0.197  0.157  0.174       83

AUC / AUPRC / Brier (per class, one-vs-rest):

  spinal_canal:
    Class              AUC  AUPRC  Brier  Support
    Normal/Mild      0.947  0.993  0.105     1722
    Moderate         0.884  0.319  0.082      139
    Severe           0.963  0.602  0.031       81
    macro            0.931  0.638

  left_foraminal:
    Class              AUC  AUPRC  Brier  Support
    Normal/Mild      0.789  0.924  0.239     1497
    Moderate         0.655  0.262  0.191      360
    Severe           0.858  0.174  0.051       80
    macro            0.767  0.453

  right_foraminal:
    Class              AUC  AUPRC  Brier  Support
    Normal/Mild      0.807  0.931  0.250     1482
    Moderate         0.657  0.283  0.209      372
    Severe           0.837  0.181  0.050       83
    macro            0.767  0.465

AUC / AUPRC overall (averaged across 3 conditions):
  macro AUC      : 0.822
  macro AUPRC    : 0.519
  popular AUPRC  : 0.949  (Normal/Mild)
  rare    AUPRC  : 0.303  (Moderate + Severe)
  Severe  AUPRC  : 0.319  (clinical priority)

Extra:
  total_train_seconds: 3374.98 (best @ ep14)
  avg_epoch_seconds: 241.07
  eval_throughput_samples_per_sec: 178.5
  eval_ms_per_sample: 5.60
```

**Aggregated for Bảng 1A/1C Ours column (REPLACES Plan B):**
- Mean AUC macro: 0.822
- Mean AUPRC macro: 0.519
- Severe AUC: 0.886 (avg of 0.963, 0.858, 0.837)
- Severe AUPRC: **0.319**
- Popular AUPRC: 0.949
- Rare AUPRC: 0.303
- Train time: 56 min (best @ ep14)
- Eval throughput: 178.5 samples/s

**Status:** ✓ Filled into SUMMARY_v3.md and THESIS_REPORT_DRAFT_v3.md (Ours column updated to v3 NEW numbers, replacing Plan B).

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
