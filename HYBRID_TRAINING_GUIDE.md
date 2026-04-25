# Hybrid Training Guide — `train_rsna_hybrid.py`

Reference for all hyperparameters of the Hybrid CBAM + BiomedCLIP training script.

## Architecture recap

| Component | Role | Trainable? |
|---|---|---|
| CBAM 3D ResNet34 (loaded from `best_model_attention.pth`) | Volumetric encoder → 512-d feature | Frozen |
| BiomedCLIP ViT-B/16 image encoder | Per-slice 2D encoder → 512-d × N slices | Frozen |
| BiomedCLIP PubMedBERT text encoder | Encode label prompts → 512-d × 9 | Frozen |
| Slice attention pool | Aggregate N slice embeddings → 1 embedding | **Trainable** (~1K) |
| Image projection MLP (1024 → 768 → 512) | Fuse CBAM + BiomedCLIP features | **Trainable** (~500K) |
| `logit_scale` | Cosine-similarity temperature | **Trainable** (1) |

Total trainable: ~500K params (~0.3% of 218M total). Inference = cosine similarity between image embedding and 9 frozen text embeddings.

## Hyperparameter reference

### Data

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `rsna_preprocessed` | Path to RSNA preprocessed `.npy` files. Must contain `volumes/` and `train_metadata.csv`. |
| `--cbam-checkpoint` | **required** | Path to trained CBAM checkpoint. Loaded as frozen backbone. Use `checkpoints/best_model_attention.pth`. |
| `--save-dir` | `checkpoints/hybrid` | Output directory for Hybrid checkpoints. Only trainable weights are saved (~2 MB per checkpoint). |
| `--val-split` | `0.2` | Patient-level validation split ratio. **Same `random_state=42`** as `train_rsna_attention.py` so val set is identical. |

### Training

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `30` | Total epochs. Hybrid converges in 15–20 epochs because only 500K params are trained. |
| `--batch-size` | `16` | Smaller than CBAM-only because each batch runs 3 BiomedCLIP slice forwards (3× memory + compute). On RTX 4090 (24 GB), 32 fits if `--slice-strategy static`. |
| `--lr` | `1e-4` | Learning rate for projection MLP + attention pool. Lower than CBAM (`1e-3`) because we're fine-tuning a small head, not training from scratch. |
| `--weight-decay` | `1e-4` | AdamW weight decay. Standard for transformer-style heads. |

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--slice-strategy` | `static` | How to pick which slices to feed BiomedCLIP. `static` = always 3 center slices (indices 3, 4, 5 of 9). `dynamic` = filter slices by cosine similarity to a query embedding (slower, ablation only). |

### Loss

| Flag | Default | Meaning |
|---|---|---|
| `--use-focal` | `True` | Use FocalLoss instead of CrossEntropy. Down-weights easy examples to focus on hard Severe cases. |
| `--focal-gamma` | `2.0` | Focusing parameter. Higher = more weight on hard examples. Common range: 1.5–3.0. CBAM training used 1.8 in best run. |
| `--use-supcon` | `True` | Add Supervised Contrastive auxiliary loss (Khosla 2020). Pulls same-label embeddings together, pushes different-label apart. Improves embedding quality for cosine-similarity classifier. |
| `--no-supcon` | — | Disable SupCon (ablation). |
| `--supcon-weight` | `0.1` | Weight on SupCon loss. **Kept low (0.1) due to class imbalance** — high SupCon weight collapses minority embeddings. |
| `--use-uncertainty` | `True` | Use Kendall et al. uncertainty weighting across 3 conditions. Each condition gets a learned `log_var` so dominant tasks don't drown minority ones. Pass `False` to disable. |

### Augmentation

| Flag | Default | Meaning |
|---|---|---|
| `--augmentation` | `medium` | Strength of training augmentation. `none / light / medium / heavy`. Defined in `spinenet/augmentation.py`. `medium` = mild rotation + brightness, safe for medical images. |
| `--oversample-factor` | `5` | Replicate samples whose label is Moderate (1) or Severe (2) by this factor. Set `1` to disable. **Trade-off**: higher factor improves Severe recall but extends epoch time linearly and lengthens dataset-build time at startup. |

### System

| Flag | Default | Meaning |
|---|---|---|
| `--num-workers` | `4` | DataLoader worker processes. Use `8–16` on Vast.ai (more workers ≠ always faster — IO bound). |
| `--save-freq` | `5` | Snapshot every N epochs (in addition to best model). |
| `--early-stop-patience` | `15` | Stop if avg Severe F1 doesn't improve for N epochs. Best model is saved by **avg Severe F1** (not val_loss). |
| `--resume` | `None` | Path to a Hybrid checkpoint to resume from. Loads only trainable weights + optimizer state. |

## Recommended commands

### Quick first run (verify pipeline)

```bash
python3 train_rsna_hybrid.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 5 --batch-size 16 --lr 1e-4 \
    --oversample-factor 1 --num-workers 8
```
Expected: avg Severe F1 ≥ 0.20 by epoch 5. Confirms forward/backward pass works.

### Full run (paper-quality)

```bash
python3 train_rsna_hybrid.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 20 --batch-size 32 --lr 1e-4 \
    --slice-strategy static \
    --supcon-weight 0.1 \
    --num-workers 16 \
    --oversample-factor 5
```
Expected: avg Severe F1 ≥ 0.35 by epoch 15–20.

### Ablation: no SupCon

```bash
python3 train_rsna_hybrid.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 20 --batch-size 32 --lr 1e-4 \
    --no-supcon \
    --oversample-factor 5
```

### Ablation: dynamic slice selection

```bash
python3 train_rsna_hybrid.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 20 --batch-size 16 --lr 1e-4 \
    --slice-strategy dynamic \
    --oversample-factor 5
```
Slower (~1.5×) than static; use only as ablation.

## Why these defaults?

- **`lr=1e-4`** (not 1e-3): only the small projection head is trained; high LR destabilises the cosine-similarity geometry.
- **`batch-size=16`** default: BiomedCLIP image encoder runs 3 forward passes per batch (one per selected slice), so peak memory is ~3× a CBAM-only run.
- **`supcon-weight=0.1`**: empirically, larger weights collapsed minority-class clusters when paired with oversampling. Low weight keeps the auxiliary signal without overpowering FocalLoss.
- **`oversample-factor=5`**: matches the value used in CBAM training so the comparison is fair. Using `1` makes Hybrid train ~1.5× faster per epoch but typically lowers Severe F1 by 0.05–0.10.
- **Best-model criterion = avg Severe F1, not val_loss**: val_loss is dominated by majority class. The whole point of this work is recovering Severe recall — saving by Severe F1 prevents the early-stopping false-positive seen in CBAM training (best `val_loss` epoch was epoch 11, but Severe F1 peaked at epoch 5).

## Startup timing (what happens before epoch 1)

| Step | Time | Notes |
|---|---|---|
| Load BiomedCLIP from HuggingFace | 2–5 min first time, <10 s cached | Cache lives in `~/.cache/huggingface/hub/` |
| Load CBAM checkpoint + init Hybrid | ~10 s | Reads ~243 MB from disk |
| Build text database (9 prompts) | ~5 s | 9 forwards through frozen text encoder |
| Compute class weights | ~5 s | Iterates metadata once |
| Build oversample indices (if factor > 1) | 2–5 min | Reads every `.npy` once to inspect labels |
| Spawn DataLoader workers | 10–30 s | Scales with `--num-workers` |

Total cold start: 5–10 min (factor=5) or 3–5 min (factor=1). Subsequent runs skip BiomedCLIP download.

## Checkpoint format

Hybrid checkpoints store **only trainable weights** to keep file size small:

```python
{
    'epoch': int,
    'attention_pool_state_dict': ...,
    'image_projection_state_dict': ...,
    'logit_scale': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'best_severe_f1': float,
    'args': vars(args),
}
```

To run inference / zero-shot eval, load `cbam-checkpoint` (frozen backbone) + Hybrid checkpoint (trainable head).

## Per-class metrics (printed every epoch)

After each epoch the script prints:

```
========== Epoch N Per-Class Metrics ==========
spinal_canal:
  Normal/Mild   F1=0.xxx  Recall=0.xxx  Support=xxxx
  Moderate      F1=0.xxx  Recall=0.xxx  Support=xxxx
  Severe        F1=0.xxx  Recall=0.xxx  Support=xxx
left_foraminal:  ...
right_foraminal: ...
Avg Severe F1: 0.xxx
================================================
```

**Watch the Avg Severe F1.** Target ≥ 0.35 for paper. If stuck at 0.0 for 3+ epochs, the projection MLP is collapsing — try:
- Lower `--supcon-weight` to 0.05
- Higher `--focal-gamma` to 2.5
- Higher `--oversample-factor` to 7

## Related docs

- `STEPS.md` — master execution plan across all phases.
- `experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md` — architecture diagram + advisor-feedback alignment.
- `RSNA_PIPELINE.md` — RSNA preprocessing and CBAM training pipeline.
