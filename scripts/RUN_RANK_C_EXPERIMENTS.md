# Rank-C readiness experiments — run guide

Commands for the multi-seed + ablation runs identified in
`MOTIVATION_AND_DESIGN.md` and the paper-readiness assessment.

All commands assume:
- working dir = repo root
- virtualenv activated (`source spinenet-venv/bin/activate`)
- `PYTHONPATH=$PYTHONPATH:$(pwd)` is set
- a CBAM RSNA checkpoint exists at `checkpoints/best_model_attention.pth`

---

## P0.2 — Multi-seed (3 seeds × 3 phases)

### Phase 1+2: RSNA Hybrid × 3 seeds

Each run ~3 h on a single A100. Total ~9 h.

```bash
for SEED in 42 0 7; do
  python train_rsna_hybrid.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 20 --batch-size 32 --lr 1e-4 \
    --slice-strategy static --supcon-weight 0.1 \
    --num-workers 16 --oversample-factor 3 \
    --class-weight-mode sqrt \
    --seed $SEED \
    --save-dir checkpoints/hybrid
done
```

Outputs (filenames auto-tagged when seed != 42):
- `checkpoints/hybrid/best_model_hybrid.pth`           (seed 42)
- `checkpoints/hybrid/best_model_hybrid_seed0.pth`     (seed 0)
- `checkpoints/hybrid/best_model_hybrid_seed7.pth`     (seed 7)
- `checkpoints/hybrid/best_metrics_hybrid*.json`       (one per seed)

### Phase 3: SPIDER zero-shot × 3 (deterministic given checkpoint)

```bash
for CKPT in best_model_hybrid.pth best_model_hybrid_seed0.pth best_model_hybrid_seed7.pth; do
  python eval_zeroshot_spider.py \
    --hybrid-checkpoint checkpoints/hybrid/$CKPT \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --output-dir experiments/paper_results/spider_zeroshot_${CKPT%.pth}
done
```

### Phase 4: SPIDER Hybrid frozen × 3 seeds

Each run ~30 min. Total ~1.5 h.

```bash
for SEED in 42 0 7; do
  python train_spider_hybrid.py \
    --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 15 --batch-size 16 --lr 1e-4 \
    --num-workers 4 \
    --seed $SEED
done
```

---

## P0.3 — Component ablations

Each ~3 h. Total ~6 h.

```bash
# Ablation A: CBAM-only (drop BiomedCLIP image branch, keep text head)
python train_rsna_hybrid.py \
  --cbam-checkpoint checkpoints/best_model_attention.pth \
  --epochs 20 --batch-size 32 --lr 1e-4 \
  --slice-strategy static --supcon-weight 0.1 \
  --num-workers 16 --oversample-factor 3 \
  --class-weight-mode sqrt \
  --ablate-branch cbam_only \
  --save-dir checkpoints/hybrid

# Ablation B: BiomedCLIP-image-only (drop CBAM 3D branch)
python train_rsna_hybrid.py \
  --cbam-checkpoint checkpoints/best_model_attention.pth \
  --epochs 20 --batch-size 32 --lr 1e-4 \
  --slice-strategy static --supcon-weight 0.1 \
  --num-workers 16 --oversample-factor 3 \
  --class-weight-mode sqrt \
  --ablate-branch biomedclip_only \
  --save-dir checkpoints/hybrid
```

Outputs:
- `best_model_hybrid_cbam_only.pth` and `best_metrics_hybrid_cbam_only.json`
- `best_model_hybrid_biomedclip_only.pth` and `best_metrics_hybrid_biomedclip_only.json`

---

## After all runs finish — aggregate to paper-ready tables

```bash
# RSNA Phase 1+2 — 3 seeds
python scripts/aggregate_seeds.py rsna \
  experiments/hybrid/best_metrics_hybrid.json \
  experiments/hybrid/best_metrics_hybrid_seed0.json \
  experiments/hybrid/best_metrics_hybrid_seed7.json

# SPIDER Phase 4 — 3 seeds
python scripts/aggregate_seeds.py spider \
  experiments/spider_phase4/best_metrics_hybrid_spider_frozen.json \
  experiments/spider_phase4/best_metrics_hybrid_spider_frozen_seed0.json \
  experiments/spider_phase4/best_metrics_hybrid_spider_frozen_seed7.json

# Ablation comparison — 3 RSNA configs side-by-side
python scripts/aggregate_seeds.py rsna \
  experiments/hybrid/best_metrics_hybrid.json \
  experiments/hybrid/best_metrics_hybrid_cbam_only.json \
  experiments/hybrid/best_metrics_hybrid_biomedclip_only.json
```

Each invocation prints a markdown table you can paste straight into the
paper Tables I/II/IV (replacing the current single-seed numbers with
mean $\pm$ std) and a new Table V for the ablation.

---

## Total compute budget

| Block | Wall-clock on A100 |
|---|---|
| 3-seed RSNA Hybrid | ~9 h |
| 3-seed SPIDER Phase 4 frozen | ~1.5 h |
| 2 ablation runs (RSNA) | ~6 h |
| Phase 3 zero-shot eval × 3 | <10 min |
| **Total** | **~17 h** (one Vast.ai overnight + half-day) |

You can parallelise across multiple Vast.ai instances if budget allows
(e.g. 3 instances × 1 seed each cuts wall-clock to ~3 h for the seed
sweep).
