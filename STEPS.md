# STEPS — Hybrid CBAM + BiomedCLIP

> Master execution plan. Date: 2026-04-25. Advisor approved Option 1 (Hybrid).

## 3 contributions

1. **CBAM + Focal Loss** — improve Severe class recall (RSNA in-distribution)
2. **BiomedCLIP integration** — enable zero-shot label extension (no retrain for new diseases)
3. **CBAM transfers across datasets** — show generality on SPIDER

## Files in this implementation

**New** (anh code):
- `spinenet/models/biomedclip_wrapper.py` — frozen BiomedCLIP wrapper
- `spinenet/models/grading_hybrid.py` — Hybrid model
- `train_rsna_hybrid.py` — train Hybrid script
- `prepare_spider_zeroshot.py` — SPIDER → test set
- `eval_zeroshot_spider.py` — zero-shot eval
- `compute_per_severity_metrics.py` — Table 1 metrics

**Edited** (1 file):
- `spinenet/models/grading_attention.py` — added `encode()` method

**Reuse existing** (no change):
- `train_rsna_baseline.py`, `train_rsna_attention.py`, `train_spider.py`
- `rsna_preprocessed_dataloader.py`, `spider_dataloader.py`

## Execution sequence

```
Step 1 — Sanity test (10 min)
  python3 -m spinenet.models.biomedclip_wrapper
  python3 -m spinenet.models.grading_hybrid checkpoints/rsna/best_model_attention.pth

Step 2 — Train Hybrid on RSNA (1-2 days GPU)
  python3 train_rsna_hybrid.py \
      --cbam-checkpoint checkpoints/rsna/best_model_attention.pth \
      --epochs 30 --batch-size 16 --lr 1e-4
  → Output: checkpoints/hybrid/best_model_hybrid.pth

Step 3 — Prepare SPIDER zero-shot test (30 min)
  python3 prepare_spider_zeroshot.py
  → Output: rsna_preprocessed_spider/spider_zeroshot_test.csv

Step 4 — Zero-shot eval on SPIDER (30 min)
  python3 eval_zeroshot_spider.py \
      --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \
      --cbam-checkpoint checkpoints/rsna/best_model_attention.pth
  → Output: experiments/paper_results/spider_zeroshot/results.csv

Step 5 — SPIDER transfer learning (2-3 days GPU)
  python3 train_spider.py --model baseline \
      --rsna-checkpoint checkpoints/rsna/best_model.pth --freeze-backbone
  python3 train_spider.py --model cbam \
      --rsna-checkpoint checkpoints/rsna/best_model_attention.pth --freeze-backbone

Step 6 — Phase 0 metrics (when ready, 3 hours, no rush)
  python3 compute_per_severity_metrics.py --include-rsna-models

Step 7 — Paper writing (4-6 weeks, AI-assisted)
```

## Verification gates

- After Step 1: forward pass returns `[B, 512]`, no NaN, ~500K trainable params
- After Step 2: avg Severe F1 ≥ 0.35 on RSNA val
- After Step 4: tier 1 (Disc_narrowing) F1 macro ≥ 0.50
- After Step 5: CBAM ≥ Vanilla on ≥ 2/3 SPIDER tasks

## 4 paper tables

| Table | Source | What it shows |
|---|---|---|
| 1. RSNA per-severity (off-the-shelf + RSNA-trained) | `compute_per_severity_metrics.py` | Severity recovery |
| 2. SPIDER zero-shot per-disease | `eval_zeroshot_spider.py` | Label extension |
| 3. SPIDER transfer learning | `train_spider.py` + `test_spider.py` | CBAM generality |
| 4. Hybrid ablation | Multiple runs of `train_rsna_hybrid.py` | Component contributions |

## Pending decisions

- Slice strategy: default `static` (3 center). Run `dynamic` as ablation if time.
- Few-shot extension methods (Tip-Adapter, CoOp): noted for later, not in current scope.

## Reference docs

- Architecture: `experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md`
- Detailed architecture: `experiments/paper_results/new_architecture.md`
- Advisor meeting prep: `experiments/paper_results/ADVISOR_MEETING_PREP_2026-04-24.md`

## Hardware target

- RTX 4090 / 5090 via Vast.ai
- Total GPU time: ~5-7 days across all phases
- Total wall-clock: ~10 weeks to submission

## Target venue

EMBC 2027 (deadline ~Jan 2027). Buffer ~3 months.
