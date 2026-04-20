# SPIDER Dataset Transfer Learning Guide

Complete guide for training and testing baseline + CBAM models on SPIDER dataset.

## Dataset Structure

**SPIDER Dataset:**
- **3 conditions** (same structure as RSNA):
  - `pfirrmann`: 5 classes (grades 1-5)
  - `spondylolisthesis`: 2 classes (No/Yes)
  - `disc_herniation`: 2 classes (No/Yes)

**IVD Extraction:**
- SPIDER provides segmentation masks with IVD labels (201-207)
- Each IVD is automatically extracted from full spine MRI
- Output format: (9, 112, 224) - same as RSNA

---

## Quick Start

### 1. Test Dataset Loading

```bash
# Test SPIDER dataloader
python spider_dataloader.py

# Expected output:
# ✓ Loaded SPIDER training dataset
#   - Samples: 2512 IVDs
#   - Format: 3 conditions (pfirrmann, spondylolisthesis, disc_herniation)
#   - Output shape: (9, 112, 224)
```

### 2. Train Baseline Model

**Step 1: Linear Probing (freeze backbone)**
```bash
python train_spider.py \
  --model baseline \
  --rsna-checkpoint checkpoints/best_model_baseline.pth \
  --freeze-backbone \
  --epochs 15 \
  --batch-size 16 \
  --lr 1e-3
```

**Step 2: Fine-tuning (unfreeze backbone)** *(optional)*
```bash
python train_spider.py \
  --model baseline \
  --rsna-checkpoint checkpoints/best_model_baseline.pth \
  --no-freeze \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4
```

### 3. Train CBAM Model

**Step 1: Linear Probing (freeze backbone + CBAM)**
```bash
python train_spider.py \
  --model cbam \
  --rsna-checkpoint checkpoints/best_model_attention.pth \
  --freeze-backbone \
  --epochs 15 \
  --batch-size 16 \
  --lr 1e-3
```

**Step 2: Fine-tuning (unfreeze everything)** *(optional)*
```bash
python train_spider.py \
  --model cbam \
  --rsna-checkpoint checkpoints/best_model_attention.pth \
  --no-freeze \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4
```

### 4. Test Models

**Test baseline:**
```bash
python test_spider.py \
  --model baseline \
  --checkpoint checkpoints_spider/best_model_baseline.pth
```

**Test CBAM:**
```bash
python test_spider.py \
  --model cbam \
  --checkpoint checkpoints_spider/best_model_cbam.pth
```

---

## Transfer Learning Strategy

### Option 1: Linear Probing (Recommended First)

**What it does:**
- Loads pretrained RSNA backbone (+ CBAM if using CBAM model)
- **Freezes** backbone weights
- Only trains new classification heads (5, 2, 2)

**When to use:**
- Limited data (SPIDER has ~2500 samples)
- Prevent overfitting
- Fast training (~5-10 min per epoch)

**Command:**
```bash
--freeze-backbone --epochs 15 --lr 1e-3
```

### Option 2: Fine-tuning (Advanced)

**What it does:**
- Loads pretrained RSNA backbone (+ CBAM)
- **Unfreezes** all weights
- Trains entire model end-to-end

**When to use:**
- After linear probing shows good results
- Want to adapt features to SPIDER dataset
- Slower but potentially better performance

**Command:**
```bash
--no-freeze --epochs 30 --lr 1e-4  # Note: smaller lr!
```

---

## Training Arguments

### Essential Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `baseline` | Model type: `baseline` or `cbam` |
| `--rsna-checkpoint` | None | Path to RSNA pretrained weights **(required!)** |
| `--freeze-backbone` | False | Freeze backbone (linear probing) |
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-3 | Learning rate |

### Advanced Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `spider` | Path to SPIDER dataset |
| `--modality` | `t2` | MRI modality (`t1` or `t2`) |
| `--loss` | `ce` | Loss type: `ce` or `weighted` |
| `--val-split` | 0.2 | Validation split ratio |
| `--early-stop-patience` | 15 | Early stopping patience |
| `--resume` | None | Resume from checkpoint |

---

## Expected Results

### Baseline vs CBAM Comparison

**RSNA Dataset (your current results):**
```
Model      Mean Accuracy  Severe Detection
Baseline   81.84%         Poor
CBAM       82.25%         Good (60% recall)
```

**SPIDER Dataset (expected):**
```
Model      Pfirrmann Acc  Spondy Acc  Herniation Acc  Mean Acc
Baseline   ??%            ??%         ??%             ??%
CBAM       ??%            ??%         ??%             ??%
```

**Goal:** Demonstrate that CBAM also outperforms baseline on SPIDER!

---

## Training Tips

### 1. Start with Linear Probing
```bash
# Always start here - much faster and prevents overfitting
--freeze-backbone --epochs 15
```

### 2. Monitor Validation Loss
- Training will save best model based on validation loss
- Early stopping triggers if no improvement for 15 epochs

### 3. Check Per-Class Metrics
- Printed every 5 epochs
- Look for imbalanced performance (like RSNA severe detection issue)

### 4. Batch Size
- SPIDER: use 16 (RSNA used 32)
- Adjust based on GPU memory

### 5. Learning Rate
- Linear probing: `1e-3` (larger)
- Fine-tuning: `1e-4` (smaller to preserve pretrained features)

---

## File Structure

After training, you should have:

```
spinet-v2/
├── spider/                        # SPIDER dataset
│   ├── images/                    # MRI volumes
│   ├── masks/                     # Segmentation masks
│   ├── overview.csv
│   └── radiological_gradings.csv
│
├── spider_dataloader.py           # Dataset loader
├── spinenet/models/grading_spider.py  # Models
├── train_spider.py                # Training script
├── test_spider.py                 # Testing script
│
├── checkpoints/                   # RSNA pretrained weights
│   ├── best_model_baseline.pth    # For baseline transfer
│   └── best_model_attention.pth   # For CBAM transfer
│
└── checkpoints_spider/            # SPIDER trained models
    ├── best_model_baseline.pth    # Baseline on SPIDER
    └── best_model_cbam.pth        # CBAM on SPIDER
```

---

## Common Issues

### 1. FileNotFoundError: RSNA checkpoint
```
Error: No such file 'checkpoints/best_model_baseline.pth'
```
**Solution:** You need to train RSNA models first! See `RSNA_TRAINING_GUIDE.md`

### 2. CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size: `--batch-size 8` or `--batch-size 4`

### 3. No improvement during training
```
Validation loss not decreasing
```
**Possible causes:**
- Learning rate too high (try `--lr 1e-4`)
- Backbone frozen but should be fine-tuned (use `--no-freeze`)
- Need more epochs

---

## Complete Workflow

### Step-by-Step Guide for Fair Comparison

**1. Train RSNA models (if not done yet):**
```bash
# Baseline
python train_rsna_baseline.py --epochs 15 --batch-size 32 --lr 1e-3

# CBAM
python train_rsna_attention.py --epochs 15 --batch-size 32 --lr 1e-3 \
  --focal-gamma 1.5 --oversample-factor 5 --use-uncertainty false
```

**2. Test RSNA models:**
```bash
python test_rsna_preprocessed.py --model checkpoints/best_model_baseline.pth
python test_rsna_preprocessed.py --model checkpoints/best_model_attention.pth
```

**3. Transfer to SPIDER (baseline):**
```bash
python train_spider.py --model baseline \
  --rsna-checkpoint checkpoints/best_model_baseline.pth \
  --freeze-backbone --epochs 15
```

**4. Transfer to SPIDER (CBAM):**
```bash
python train_spider.py --model cbam \
  --rsna-checkpoint checkpoints/best_model_attention.pth \
  --freeze-backbone --epochs 15
```

**5. Test SPIDER models:**
```bash
python test_spider.py --model baseline \
  --checkpoint checkpoints_spider/best_model_baseline.pth

python test_spider.py --model cbam \
  --checkpoint checkpoints_spider/best_model_cbam.pth
```

**6. Compare results:**
```
RSNA Dataset:
  Baseline: 81.84%
  CBAM:     82.25% (+0.41%)

SPIDER Dataset:
  Baseline: ??%
  CBAM:     ??% (+??%)
```

---

## Questions?

If you encounter issues:
1. Check the error message carefully
2. Verify dataset is loaded correctly: `python spider_dataloader.py`
3. Verify RSNA checkpoints exist
4. Try reducing batch size if CUDA OOM

Good luck with transfer learning! 🚀
