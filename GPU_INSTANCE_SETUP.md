# RSNA Preprocessing on GPU Instance

Quick guide to preprocess RSNA dataset on a rented GPU instance.

## Option 1: Automated Setup (Easiest)

```bash
# 1. Upload your code to GPU instance
# (use git clone, scp, or your provider's method)

# 2. Run the automated setup
chmod +x setup_rsna_preprocessing.sh
./setup_rsna_preprocessing.sh
```

This will:
- Install all dependencies
- Download dataset from Google Drive (~8-10GB)
- Extract and preprocess to .npy files
- Organize by patient folders
- Create metadata CSV

**Time estimate:** 30-60 minutes depending on internet speed

---

## Option 2: Manual Step-by-Step

### Step 1: Install Dependencies
```bash
pip install gdown tqdm pandas pydicom numpy torch pillow scikit-learn
```

### Step 2: Download & Preprocess
```bash
# Download, extract, and preprocess in one command
python3 prepare_rsna_data.py
```

**OR** if you already have the dataset downloaded:
```bash
# Skip download, just preprocess
python3 prepare_rsna_data.py --skip-download
```

### Step 3: Verify Output
```bash
# Check if preprocessing worked
ls -lh rsna_preprocessed/
cat rsna_preprocessed/train_metadata.csv | wc -l

# Quick visualization test (optional)
python3 visualize_npy.py --show-stats --num-samples 3
```

---

## Output Structure

After preprocessing completes, you'll have:

```
rsna_preprocessed/
├── volumes/
│   ├── 4003253/              # Patient folder
│   │   ├── 702807833_l1_l2.npy
│   │   ├── 702807833_l2_l3.npy
│   │   ├── 702807833_l3_l4.npy
│   │   ├── 702807833_l4_l5.npy
│   │   └── 702807833_l5_s1.npy
│   ├── 4646740/              # Another patient
│   │   └── ...
│   └── ...                   # ~1974 patient folders
├── train_metadata.csv        # Labels and file paths
└── train_failed.json         # Failed samples (if any)
```

Each `.npy` file contains:
- Shape: (9, 112, 224) - 9 sagittal slices, 224×112 resolution
- Data type: float32
- Normalized to [0, 1] range

---

## Troubleshooting

### "No samples found"
- Check if dataset extracted to `rsna-2024-lumbar-spine-degenerative-classification/`
- Make sure `train.csv` and `train_label_coordinates.csv` exist

### "gdown download failed"
- Google Drive may have rate limits
- Try downloading manually: https://drive.google.com/file/d/19M2C--zpbretFJtYQ6S16FgGt9in3_Qq/view
- Then run: `python3 prepare_rsna_data.py --skip-download`

### "Out of disk space"
- Preprocessing creates ~8GB of .npy files
- Make sure you have at least 20GB free space (10GB zip + 8GB extracted + 8GB .npy)

---

## Next Steps: Training

Once preprocessing is complete:

```bash
# Train the model (Phase 1: freeze backbone)
python3 train_rsna_baseline.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --save-dir checkpoints

# Monitor GPU usage
nvidia-smi -l 1

# Check training progress
tail -f checkpoints/training.log  # If logging to file
```

**Recommended GPU instance specs:**
- GPU: NVIDIA T4 or better (16GB+ VRAM)
- RAM: 16GB+
- Disk: 30GB+ free space
- Batch size: 32 (adjust based on GPU memory)

---

## Disk Space Summary

| Item | Size |
|------|------|
| Dataset zip | ~8-10GB |
| Extracted dataset | ~8-10GB |
| Preprocessed .npy | ~8GB |
| **Total needed** | **~25GB** |

You can delete the zip file after extraction to save space.
