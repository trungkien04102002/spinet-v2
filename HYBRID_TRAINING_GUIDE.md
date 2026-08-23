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

## Forward pass detail

The two encoders run **in parallel** on the same input volume — not sequentially.
Their outputs are concatenated, projected, L2-normalized, and matched against a
**precomputed** text-embedding database via cosine similarity.

```
Input volume [B, 1, 9, 112, 224]   (B = batch, 9 sagittal slices)
        │
        ├──── Branch 1 (parallel): CBAM 3D ResNet34 (frozen) ──────────────┐
        │     Treats the 9 slices as a 3-D stack.                          │
        │     Output: [B, 512]                                              │
        │                                                                   │
        └──── Branch 2 (parallel): BiomedCLIP image encoder (frozen) ──────┤
              Pick 3 center slices (indices 3, 4, 5) — the "static"          │
              strategy. Each slice is processed as a 2-D image:              │
                grayscale [112, 224] → resize 224×224 → repeat to 3 ch       │
                  → CLIP normalization → ViT-B/16 → [B, 512] per slice       │
              Stacked: [B, 3, 512]                                           │
                          │                                                  │
                          ▼                                                  │
              SliceAttentionPool (TRAINABLE, ~1K params):                    │
                a learned scorer assigns weights to the 3 slices, then       │
                returns a weighted sum. The pooler does NOT modify slice     │
                embeddings — it just decides which slice matters more.       │
              Output: [B, 512]                                               │
                                                                              │
                ┌─────────────────────────────────────────────────────────────┘
                ▼
        Concatenate features:        [B, 1024]   (CBAM 512 + BiomedCLIP 512)
                │
                ▼
        Image projection MLP (TRAINABLE, ~500K):
                Linear(1024 → 768) → GELU → Dropout(0.1) → Linear(768 → 512)
                │
                ▼
        L2 normalize           image_emb / ||image_emb||₂   →  [B, 512]
                │
                │  (the text side runs once at training start, then cached)
                │
                │     ┌───────────────────────────────────────────┐
                │     │ Build text database (one-time, frozen):    │
                │     │   for each (condition, severity) prompt:   │
                │     │     tokens = tokenizer(prompt)             │
                │     │     emb = biomedclip.encode_text(tokens)   │
                │     │     L2 normalize                           │
                │     │   stack → text_embs [9, 512]               │
                │     │   stored on device, reused every batch     │
                │     └───────────────────────────────────────────┘
                │                              │
                ▼                              ▼
        cosine = image_emb @ text_embs.T   →  [B, 9]
                │
                ▼
        logits = clamp(exp(logit_scale), max=100) × cosine     [B, 9]
                  └─ logit_scale: TRAINABLE temperature (1 scalar). Larger
                     temperature sharpens the softmax; clamp prevents the
                     parameter from exploding during early training.
                │
                ▼
        argmax → predicted class (or softmax → calibrated probs)
```

### Why every step matters

| Step | Why it's there |
|---|---|
| **Parallel branches** | CBAM captures 3-D spatial structure (vertebrae, canal); BiomedCLIP captures 2-D radiology semantics from PubMed pretraining. They see the volume differently — concatenation gives the projection MLP both views. |
| **3 center slices, not 9** | BiomedCLIP is 2-D. We could feed all 9, but they're highly correlated (adjacent sagittal slices), and 9 forwards per batch tripled training time. Center slices contain the most diagnostic IVD information. |
| **SliceAttentionPool** | Even within 3 center slices, the most diagnostic one varies per case (some pathologies show on slice 3, others on slice 5). A learned pooler is more robust than a hard-coded mean. |
| **L2 normalization before cosine** | Cosine similarity = dot product **only when vectors are unit-norm**. Without L2 norm, the dot product reflects vector magnitude, not direction — collapses CLIP-style classifiers. |
| **Text DB precomputed** | Encoding 9 prompts costs ~5 s. Encoding them every batch would cost ~5 s × N batches = hours wasted. The text encoder is frozen, so the embeddings never change after init — cache once. |
| **`logit_scale` (temperature)** | Without scaling, cosine values live in [-1, 1] — softmax is too soft and gradients vanish. CLIP introduced learnable `exp(logit_scale)` (init 1/0.07 ≈ 14.3) so the model picks its own sharpness. The clamp at 100 prevents runaway growth in the first few epochs. |

### Adding a new label (zero-shot extension)

```python
# 1. Define new prompts (text)
new_prompts = ["normal disc", "modic change type 1", "modic change type 2"]

# 2. Encode once via frozen BiomedCLIP text encoder
new_text_embs = biomedclip.encode_text(new_prompts)   # [3, 512]
# No retraining. No weight updates.

# 3. Reuse the SAME image encoder
image_emb = hybrid.encode_image(volume)               # [B, 512]
logits = logit_scale * (image_emb @ new_text_embs.T)  # [B, 3]
predictions = logits.argmax(dim=-1)
```

This is exactly what `eval_zeroshot_spider.py` does for SPIDER's 8 unseen disease labels.

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
