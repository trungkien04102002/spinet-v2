# RSNA Training Results Summary

> Date: 2026-04-26. All results on val split (1942 samples, patient-level held-out).

## TL;DR

**Hybrid (logit_scale fix + sqrt class weights), Epoch 7** is the best overall result:
- Avg Severe F1 = **0.353** (highest)
- Foraminal Severe F1 = **0.276 / 0.283** (highest)
- Mean Acc = **70.0%** (better than CBAM-only)

---

## Full comparison table

All numbers on validation split (1942 samples, 80/20 patient-level split, `random_state=42`).

| Method | Spinal Severe F1 | L-Fora Severe F1 | R-Fora Severe F1 | **Avg Severe F1** | Mean Acc | Notes |
|---|---|---|---|---|---|---|
| Baseline (no CBAM, no CW) | 0.457 | **0.000** | **0.000** | 0.152 | 81.4% | Foraminal collapses |
| CBAM + sqrt CW (E20) | **0.623** | 0.180 | 0.197 | 0.333 | 68.6% | Strong on spinal, weak on foraminal |
| Hybrid buggy (no CW), E1 | 0.588 | 0.267 | 0.222 | 0.359 | 80.7% | Inflated by clamped logit_scale |
| **Hybrid fixed + sqrt CW, E7** ⭐ | 0.500 | **0.276** | **0.283** | **0.353** | **70.0%** | Best balanced result |

## Per-class metrics (val split)

### Baseline (no CBAM, no class weights), E25

| Condition | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Spinal Canal | Normal | 0.911 | 0.986 | 0.947 | 1722 |
| Spinal Canal | Moderate | 0.333 | 0.079 | 0.128 | 139 |
| Spinal Canal | Severe | 0.630 | 0.358 | 0.457 | 81 |
| Left Foraminal | Severe | 0.000 | 0.000 | 0.000 | 80 |
| Right Foraminal | Severe | 0.000 | 0.000 | 0.000 | 83 |

### CBAM + sqrt CW, E20

| Condition | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Spinal Canal | Severe | 0.559 | 0.704 | 0.623 | 81 |
| Left Foraminal | Severe | 0.172 | 0.188 | 0.180 | 80 |
| Right Foraminal | Severe | 0.180 | 0.217 | 0.197 | 83 |

### Hybrid fixed + sqrt CW, E7 ⭐

| Condition | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Spinal Canal | Severe | 0.374 | **0.753** | 0.500 | 81 |
| Left Foraminal | Severe | 0.219 | **0.375** | 0.276 | 80 |
| Right Foraminal | Severe | 0.233 | **0.361** | 0.283 | 83 |

→ Hybrid Severe Recall: 0.75 / 0.38 / 0.36 (catches more Severe across all conditions)

---

## Key findings

### 1. Class weights (sqrt mode) is the biggest single contributor

Adding `--class-weight-mode sqrt` to FocalLoss alpha (Severe weight ~4.4x Normal) was the single change that recovered Foraminal Severe detection from F1=0.000 to F1≈0.20. Without it, both CBAM-only and Hybrid collapsed on Foraminal Severe.

### 2. logit_scale init was a real bug

The original `nn.Parameter(torch.ones([]) * (1/0.07))` initialized the parameter at 14.28; combined with `exp().clamp(max=100)` in the forward, this gave `exp(14.28) ≈ 1.6M` clamped to 100 from step 1 → dead gradient on logit_scale → projection MLP couldn't train properly.

After fix to `nn.Parameter(torch.tensor(math.log(1/0.07)))` (init ≈ 2.659, exp ≈ 14.3 — CLIP convention), the logit_scale grew naturally during training (E1: 14.74 → E7: 17.59) and the model trained healthily across 7+ epochs instead of peaking at E1.

### 3. Hybrid genuinely improves Foraminal Severe over CBAM-only

After both fixes, Hybrid achieves F1 = 0.276 / 0.283 on L/R foraminal Severe vs CBAM's 0.180 / 0.197. This is +50% improvement on the hardest classes. Plausible mechanism: BiomedCLIP image features add complementary semantic signal that the CBAM-only path lacks.

### 4. Spinal Severe Recall trade-off

Hybrid achieves 0.753 Spinal Severe Recall (vs CBAM's 0.704), but at lower precision (0.374 vs 0.559). The model predicts Severe more aggressively → catches more cases (good for clinical) at the cost of more false positives.

---

## Caveats / honest limitations

1. **Single-seed result**: Best E7 might not replicate exactly across random seeds. Ideal would be 3-5 seeds + mean ± std.
2. **Val set has data the model "saw" 25 times**: not a true held-out test. Patient-level split mitigates but doesn't eliminate the issue.
3. **Comparison is apples-to-apples for class weights** (sqrt) but Hybrid uses a slightly different loss recipe (gamma 2.0 vs CBAM 1.8, oversample 3 vs 5, has SupCon). Some confounders remain.
4. **Foraminal Severe is anatomically harder** than Spinal Canal Severe. CBAM averaged across conditions doesn't focus on lateral edge → both CBAM and Hybrid struggle, just Hybrid struggles less.

---

## What's already done

- ✅ Baseline (no CBAM, no CW): val 81.4% Mean Acc, Avg Severe F1 0.152
- ✅ CBAM + sqrt CW, E20: val 68.6% Mean Acc, Avg Severe F1 0.333
- ✅ Hybrid buggy logit (E1 lucky): val 80.7%, Avg Severe F1 0.359 — DO NOT use, bug
- ✅ Hybrid fixed + sqrt CW, E7: val 70.0%, Avg Severe F1 **0.353** — best, use this

## What to do next

### Phase 3 — Zero-shot SPIDER eval (highest priority)

This is the actual contribution of the BiomedCLIP integration. Hybrid RSNA results just demonstrate the model is functional; the real test is whether it transfers to unseen labels.

```bash
python3 prepare_spider_zeroshot.py     # ~30 min, prep SPIDER test set
python3 eval_zeroshot_spider.py \
    --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \
    --cbam-checkpoint checkpoints/best_model_attention.pth
```

Expected (from STEPS.md):
- Easy disease (Disc_narrowing): F1 macro ≥ 0.50
- Medium (Modic, Spondylolisthesis): F1 macro ≥ 0.35
- Hard (Pfirrman 5-class): F1 macro ≥ 0.25

### Phase 4 — SPIDER transfer learning (Table 3 in paper)

Show CBAM generalizes vs Vanilla on a new dataset. Uses hardcoded SPIDER heads, not Hybrid.

```bash
python3 train_spider.py --model baseline --rsna-checkpoint checkpoints/best_model.pth --freeze-backbone
python3 train_spider.py --model cbam --rsna-checkpoint checkpoints/best_model_attention.pth --freeze-backbone
```

---

## Configuration references

### Best Hybrid run (this is what gave the E7 result)

```bash
python3 train_rsna_hybrid.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 20 --batch-size 32 --lr 1e-4 \
    --slice-strategy static --supcon-weight 0.1 \
    --num-workers 16 --oversample-factor 3 \
    --class-weight-mode sqrt
```

### Best CBAM run (for `--rsna-checkpoint` in SPIDER transfer)

```bash
python3 train_rsna_attention.py \
    --epochs 25 --batch-size 64 --focal-gamma 1.8 \
    --lr 1e-3 --num-workers 8 \
    --class-weight-mode sqrt
```

### Baseline run (for `--rsna-checkpoint` baseline in SPIDER transfer)

```bash
python3 train_rsna_baseline.py --epochs 25 --batch-size 64 --lr 1e-3 --num-workers 8
```

---

## Files in this directory

| File | Source |
|---|---|
| `best_metrics_baseline_full_e25.txt` | Baseline best epoch |
| `best_metrics_attention_sqrt_cw_e20.txt` (or similar) | CBAM best epoch |
| `best_metrics_hybrid_*.txt` | Hybrid best epoch |
| `training_log_*.csv` | Per-epoch trajectory for each method |
| `RESULTS_SUMMARY.md` | This file |

Checkpoints live in `checkpoints/fresh/` and `checkpoints/fresh_baseline/`.
