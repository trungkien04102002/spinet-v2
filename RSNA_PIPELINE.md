# RSNA 2024 Lumbar Spine - Complete Pipeline

Complete guide for preprocessing and training SpineNet on RSNA 2024 dataset.

---

## 📋 Quick Reference

```bash
# 1. Setup (on GPU instance)
./setup_rsna_preprocessing.sh

# 2. Prepare data
python3 prepare_rsna_data.py

# 3. Train model
python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3
```

---

## 🚀 Step 1: Setup Environment

### On Local Machine

```bash
# Clone repository
git clone <your-repo>
cd spinet-v2

# Install dependencies
pip install torch torchvision numpy pandas pydicom scikit-learn pillow tqdm gdown
```

### On GPU Instance (Recommended for Training)

```bash
# Clone repository
git clone <your-repo>
cd spinet-v2

# Run automated setup
chmod +x setup_rsna_preprocessing.sh
./setup_rsna_preprocessing.sh
```

**What it does:**
- Installs all dependencies
- Downloads RSNA dataset from Google Drive (~8-10GB)
- Extracts and preprocesses to `.npy` files
- Creates patient-level folder structure
- Generates metadata CSV

**Time:** 30-60 minutes (mostly downloading)

---

## 📦 Step 2: Prepare Data

### Option A: Automatic (Recommended)

Already done if you ran `setup_rsna_preprocessing.sh` above!

### Option B: Manual

```bash
# Download, extract, and preprocess all-in-one
python3 prepare_rsna_data.py
```

**OR** if you already have the dataset:

```bash
# Skip download, just preprocess
python3 prepare_rsna_data.py --skip-download --data-dir rsna-2024-lumbar-spine-degenerative-classification
```

### Output Structure

```
rsna_preprocessed/
├── volumes/
│   ├── 4003253/              # Patient ID (study_id)
│   │   ├── 702807833_l1_l2.npy
│   │   ├── 702807833_l2_l3.npy
│   │   ├── 702807833_l3_l4.npy
│   │   ├── 702807833_l4_l5.npy
│   │   └── 702807833_l5_s1.npy
│   ├── 4646740/
│   └── ...                   # ~1974 patients
└── train_metadata.csv        # Labels + paths
```

**Each `.npy` file:**
- Shape: `(9, 112, 224)` - 9 sagittal slices
- Type: float32, normalized to [0, 1]
- Size: ~8GB total for all files

### Verify Preprocessing

```bash
# Check statistics
python3 visualize_npy.py --show-stats --num-samples 3

# View 2D slices
python3 visualize_npy.py --num-samples 2

# View in 3D
python3 visualize_npy.py --3d --num-samples 2

# Verify crop extraction
python3 visualize_npy.py --check-crop --num-samples 2
```

---

## 🎯 Step 3: Train Model

### Phase 1: Train Classification Heads (Frozen Backbone)

**Recommended for first run:**

```bash
python3 train_rsna_baseline.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --save-dir checkpoints
```

**What happens:**
- ✅ Loads pretrained backbone (from original SpineNet)
- ✅ Freezes backbone weights
- ✅ Trains only 3 new classification heads
- ✅ Fast convergence (20-30 epochs)
- ✅ Saves best model to `checkpoints/best_model.pth`

### Phase 2: Fine-tune Entire Network (Optional)

**Only if Phase 1 results aren't good enough:**

```bash
python3 train_rsna_baseline.py \
    --unfreeze-backbone \
    --lr 1e-5 \
    --epochs 20 \
    --batch-size 16 \
    --resume checkpoints/best_model.pth \
    --save-dir checkpoints_finetuned
```

**Important:** Use much lower learning rate (1e-5 vs 1e-3)!

### All Training Options

```bash
python3 train_rsna_baseline.py \
    --data-dir rsna_preprocessed       # Data directory
    --epochs 30                        # Number of epochs
    --batch-size 32                    # Batch size (adjust for GPU)
    --lr 1e-3                          # Learning rate
    --weight-decay 1e-4                # L2 regularization
    --val-split 0.2                    # Validation split
    --save-dir checkpoints             # Checkpoint directory
    --save-freq 5                      # Save every N epochs
    --early-stop-patience 10           # Early stopping
    --num-workers 4                    # Data loader workers
    --use-pretrained                   # Use pretrained backbone (default)
    --no-pretrained                    # Train from scratch (not recommended)
    --unfreeze-backbone                # Fine-tune entire network
    --resume checkpoints/best_model.pth  # Resume from checkpoint
```

### Monitor Training

```bash
# Watch GPU usage
nvidia-smi -l 1

# View training output
# (metrics printed every epoch)
```

---

## 📊 Step 4: Evaluate Model

### Quick Test

```bash
# Test with best model on a few patients
python3 test_rsna_preprocessed.py \
    --num-patients 5 \
    --use-pretrained
```

### What to Look For

**Good signs:**
- ✅ Validation loss decreasing
- ✅ Accuracy > 50% (better than random)
- ✅ Confusion matrix shows predictions across all 3 classes
- ✅ Weighted log loss < 1.0

**Bad signs:**
- ❌ Val loss increases (overfitting)
- ❌ Model always predicts same class
- ❌ Accuracy ~33% (random guessing)

**Training output shows:**
- Train/Val loss per epoch
- Accuracy per condition (spinal canal, left/right foraminal)
- Weighted log loss (RSNA competition metric)
- Confusion matrices (every 5 epochs)

---

## 💾 Disk Space Requirements

| Item | Size |
|------|------|
| Dataset zip | ~8-10GB |
| Extracted dataset | ~8-10GB |
| Preprocessed .npy | ~8GB |
| Model checkpoints | ~500MB |
| **Total needed** | **~25-30GB** |

💡 **Tip:** Delete the zip file after extraction to save space.

---

## 🔧 Troubleshooting

### "No samples found"
- Check if dataset extracted to `rsna-2024-lumbar-spine-degenerative-classification/`
- Verify `train.csv` and `train_label_coordinates.csv` exist
- Make sure condition/level names are normalized

### "Out of memory" during training
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Reduce num workers: `--num-workers 2`

### "gdown download failed"
- Google Drive may have rate limits
- Download manually: https://drive.google.com/file/d/19M2C--zpbretFJtYQ6S16FgGt9in3_Qq/view
- Then run: `python3 prepare_rsna_data.py --skip-download`

### Preprocessing is slow
- Normal! ~1974 patients × 5 levels = 9,870 samples
- Takes 20-40 minutes depending on CPU
- Only need to do once

### Training is slow
- Make sure you're using GPU: Check for "Device: cuda" in output
- Adjust batch size based on GPU memory
- Phase 1 should take 2-4 hours on modern GPU (30 epochs)

---

## 📁 File Structure

### Core Pipeline Files

```
spinet-v2/
├── prepare_rsna_data.py              # Preprocessing script
├── rsna_dataloader.py                # DICOM dataloader
├── rsna_preprocessed_dataloader.py   # Fast .npy dataloader
├── train_rsna_baseline.py            # Training script
├── test_rsna_preprocessed.py         # Testing script
├── visualize_npy.py                  # Verification tool
├── setup_rsna_preprocessing.sh       # Automated setup
└── RSNA_PIPELINE.md                  # This guide
```

### Data Directories (Not in Git)

```
rsna-2024-lumbar-spine-degenerative-classification/  # Original dataset
rsna_preprocessed/                                    # Preprocessed .npy files
checkpoints/                                          # Saved models
```

---

## 🎓 Dataset Details

**Tasks:** 3 conditions
- Spinal Canal Stenosis
- Left Neural Foraminal Narrowing
- Right Neural Foraminal Narrowing

**Classes:** 3 severity levels (per condition)
- 0: Normal/Mild
- 1: Moderate
- 2: Severe

**IVD Levels:** 5 lumbar levels per patient
- L1/L2, L2/L3, L3/L4, L4/L5, L5/S1

**Input Format:**
- Volume shape: `(9, 112, 224)`
- 9 sagittal T2-weighted MRI slices
- Aspect ratio: 2:1 (width:height)
- Normalized to [0, 1] range

**Model Architecture:**
- Backbone: ResNet34 (pretrained on original SpineNet)
- 3 classification heads (one per condition)
- Each head: 512 → 3 classes

---

## 🚀 Quick Start Example

Complete workflow from scratch:

```bash
# 1. Setup (one-time)
git clone <your-repo> && cd spinet-v2
./setup_rsna_preprocessing.sh

# 2. Verify data
python3 visualize_npy.py --show-stats --num-samples 3

# 3. Train
python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3

# 4. Evaluate
python3 test_rsna_preprocessed.py --num-patients 10 --use-pretrained

# Done! Best model saved at: checkpoints/best_model.pth
```

---

## 📊 Expected Results

**Phase 1 (Frozen Backbone):**
- Training time: 2-4 hours (30 epochs, GPU)
- Expected val accuracy: 50-70%
- Weighted log loss: 0.6-1.0

**Phase 2 (Fine-tuned):**
- Additional training: 1-2 hours (20 epochs)
- Expected val accuracy: 60-75%
- Weighted log loss: 0.5-0.8

*Note: Results depend on data quality, hyperparameters, and random seed.*

---

## 💡 Tips

1. **Always preprocess first** - Training on .npy files is 50x faster than loading DICOMs
2. **Start with Phase 1** - Frozen backbone trains fast and often works well
3. **Monitor confusion matrices** - Make sure model predicts all classes, not just one
4. **Use early stopping** - Training stops automatically if no improvement
5. **Save disk space** - Delete zip file after extraction
6. **Patient-level folders** - Makes it easy to debug specific patients
7. **Adjust batch size** - Based on your GPU memory (32 for 16GB, 16 for 8GB)

---

## 🔗 Related Files

- [setup_rsna_preprocessing.sh](setup_rsna_preprocessing.sh) - Automated setup script
- [.gitignore](.gitignore) - Excludes large data files from git

---

## 📝 Notes

- Preprocessing creates patient-level folders for better organization
- Split is done by patient (no data leakage between train/val)
- Pretrained backbone uses weights from original SpineNet (11-task model)
- RSNA uses different task format: 3 tasks × 3 classes (vs original 11 tasks)
- Google Drive link: https://drive.google.com/file/d/19M2C--zpbretFJtYQ6S16FgGt9in3_Qq/view
