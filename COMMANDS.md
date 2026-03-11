# SpineNetV2 RSNA Training Commands

Complete command reference for training SpineNetV2 on RSNA 2024 Lumbar Spine dataset.

## 📋 Table of Contents
1. [Initial Setup (GPU Instance)](#initial-setup-gpu-instance)
2. [Baseline Model Training](#baseline-model-training)
3. [Attention Model Training (Improved)](#attention-model-training-improved)
4. [Testing and Evaluation](#testing-and-evaluation)
5. [Download Trained Weights](#download-trained-weights)

---

## Initial Setup (GPU Instance)

Run these commands **once** when starting a new GPU instance:

```bash
# Enable mouse in tmux
tmux set -g mouse on

# Clone repository and checkout branch
./vast_setup.sh --branch train-attention

# Install Python dependencies
pip3 install -r requirements.txt

# Make scripts executable
chmod +x 2_download_preprocessed.sh
chmod +x 3_download_weights.sh

# Download preprocessed data from Google Drive (8GB)
./2_download_preprocessed.sh 1DCndO_ppTDMGT1H19b8XIqfureMqABi_

# Download pretrained backbone weights from Google Drive (242MB)
./3_download_weights.sh 1GCmJ0OuNdw9c1E4giLwyA9EK4uwWL6HT
```

**Check disk space** (recommended):
```bash
df -h
du -sh rsna_preprocessed/
du -sh ~/.spinenet/weights/
```

---

## Baseline Model Training

Train the **baseline model** (no attention, simple CrossEntropyLoss):

### Quick Test (1 epoch)
```bash
python3 train_rsna_baseline.py --epochs 1 --batch-size 32 --lr 1e-3
```

### Short Training (5 epochs)
```bash
python3 train_rsna_baseline.py --epochs 5 --batch-size 32 --lr 1e-3
```

### Full Training (30 epochs)
```bash
python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3
```

### Advanced Options
```bash
# Increase batch size for faster training (requires more GPU memory)
python3 train_rsna_baseline.py --epochs 30 --batch-size 64 --lr 1e-3

# Increase data loading workers
python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3 --num-workers 8

# Both (recommended for high-end GPUs)
python3 train_rsna_baseline.py --epochs 30 --batch-size 64 --lr 1e-3 --num-workers 8
```

**Expected Results** (30 epochs):
- Training time: ~30 minutes (depends on GPU)
- Spinal Canal accuracy: ~89%
- Foraminal accuracy: ~77%
- **Problem**: Severe class F1 = 0.000 (model doesn't learn minority class)

---

## Attention Model Training (Improved)

Train the **improved model** with CBAM attention, FocalLoss, and augmentation:

### Quick Test (1 epoch)
```bash
python3 train_rsna_attention.py --epochs 1 --batch-size 32 --lr 1e-3
```

### Recommended Training (30 epochs)
```bash
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3
```

### Advanced Options

**Increase batch size** (for high-end GPUs):
```bash
python3 train_rsna_attention.py --epochs 30 --batch-size 64 --lr 1e-3 --num-workers 8
```

**Disable specific improvements** (for ablation study):
```bash
# Disable CBAM attention
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 --no-cbam

# Use CrossEntropyLoss instead of FocalLoss
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 --use-focal=False

# Disable UncertaintyLoss
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 --use-uncertainty=False

# Light augmentation
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 --augmentation light

# No augmentation
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 --augmentation none
```

**Train full model** (unfreeze backbone):
```bash
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-4 --no-freeze
```
⚠️ Note: Use lower learning rate (1e-4) when training backbone

**Resume from checkpoint**:
```bash
python3 train_rsna_attention.py --epochs 50 --batch-size 32 --lr 1e-3 \
    --resume checkpoints/checkpoint_attention_epoch_30.pth
```

**Expected Results** (30 epochs):
- Training time: ~40 minutes (slightly slower than baseline)
- Spinal Canal accuracy: ~90% (+1%)
- Foraminal accuracy: ~80% (+3%)
- **Improvement**: Severe class F1 = 0.4-0.5 (learns minority class!)

---

## Testing and Evaluation

Test trained models on validation set:

### Baseline Model
```bash
python3 test_rsna_preprocessed.py \
    --model checkpoints/best_model.pth \
    --num-patients 5
```

### Attention Model
```bash
python3 test_rsna_preprocessed.py \
    --model checkpoints/best_model_attention.pth \
    --num-patients 5
```

### Full Validation Set
```bash
python3 test_rsna_preprocessed.py \
    --model checkpoints/best_model_attention.pth \
    --num-patients -1
```

---

## Download Trained Weights

Copy trained model from GPU instance to local machine:

```bash
# From your local machine (in spinet-v2 directory)

# Baseline model
scp -P <PORT> root@<IP>:/root/spinet-v2/checkpoints/best_model.pth checkpoints/

# Attention model
scp -P <PORT> root@<IP>:/root/spinet-v2/checkpoints/best_model_attention.pth checkpoints/

# All checkpoints
scp -P <PORT> root@<IP>:/root/spinet-v2/checkpoints/*.pth checkpoints/
```

Example (your previous instance):
```bash
scp -P 59873 root@175.28.230.22:/root/spinet-v2/checkpoints/best_model_attention.pth checkpoints/
```

---

## Monitoring Training

### Monitor GPU Usage
```bash
# In a separate terminal/tmux pane
watch -n 1 nvidia-smi
```

### Check Training Logs
```bash
# If training in background
tail -f training.log
```

### Tmux Commands
```bash
# Create new window
Ctrl+b c

# Split pane horizontally
Ctrl+b "

# Split pane vertically
Ctrl+b %

# Switch between panes
Ctrl+b arrow keys

# Detach from session
Ctrl+b d

# Reattach to session
tmux attach
```

---

## Ablation Study Workflow

To compare improvements systematically:

```bash
# 1. Baseline (no improvements)
python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3

# 2. + FocalLoss only
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 \
    --no-cbam --use-uncertainty=False --augmentation none

# 3. + FocalLoss + Augmentation
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 \
    --no-cbam --use-uncertainty=False --augmentation medium

# 4. + FocalLoss + Augmentation + CBAM
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3 \
    --augmentation medium

# 5. Full model (all improvements)
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3
```

Compare results in your thesis:
- Table: Accuracy, F1-score per class
- Figure: Attention maps from CBAM
- Analysis: Which improvement had the biggest impact?

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python3 train_rsna_attention.py --epochs 30 --batch-size 16 --lr 1e-3
```

### Training Too Slow
```bash
# Increase workers and batch size
python3 train_rsna_attention.py --epochs 30 --batch-size 64 --lr 1e-3 --num-workers 8
```

### GPU Not Being Used
```bash
# Check CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

### Pretrained Weights Not Found
```bash
# Re-download weights
./3_download_weights.sh 1GCmJ0OuNdw9c1E4giLwyA9EK4uwWL6HT

# Verify weights exist
ls -lh ~/.spinenet/weights/
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup GPU instance | `./vast_setup.sh --branch train-attention && pip3 install -r requirements.txt` |
| Download data | `./2_download_preprocessed.sh 1DCndO_ppTDMGT1H19b8XIqfureMqABi_` |
| Download weights | `./3_download_weights.sh 1GCmJ0OuNdw9c1E4giLwyA9EK4uwWL6HT` |
| Train baseline | `python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3` |
| Train attention model | `python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3` |
| Test model | `python3 test_rsna_preprocessed.py --model checkpoints/best_model_attention.pth --num-patients 5` |
| Monitor GPU | `watch -n 1 nvidia-smi` |
| Download weights | `scp -P <PORT> root@<IP>:~/spinet-v2/checkpoints/best_model_attention.pth checkpoints/` |

---

## File Locations

| Description | Path |
|-------------|------|
| Preprocessed data | `rsna_preprocessed/` (8GB) |
| Pretrained backbone | `~/.spinenet/weights/ckpt1.pt` (242MB) |
| Training checkpoints | `checkpoints/` |
| Best baseline model | `checkpoints/best_model.pth` |
| Best attention model | `checkpoints/best_model_attention.pth` |
| Periodic checkpoints | `checkpoints/checkpoint_*_epoch_*.pth` |

---

## Expected Training Time

| Configuration | Time per Epoch | Total (30 epochs) |
|---------------|----------------|-------------------|
| Baseline, batch=32 | ~1 min | ~30 min |
| Attention, batch=32 | ~1.5 min | ~45 min |
| Attention, batch=64 | ~1 min | ~30 min |

*Times are approximate and depend on GPU model

---

## For Your Thesis

### Recommended Experiments

1. **Baseline**: `train_rsna_baseline.py` (30 epochs)
2. **Attention Full**: `train_rsna_attention.py` (30 epochs)
3. **Ablation Studies**: Test each component individually (see Ablation Study Workflow)

### Results to Report

- **Table 1**: Per-class metrics (Precision, Recall, F1) for each model
- **Table 2**: Ablation study showing incremental improvements
- **Figure 1**: Training curves (loss over epochs)
- **Figure 2**: CBAM attention maps (visualize what model focuses on)
- **Figure 3**: Confusion matrices (Baseline vs Attention)

### Key Findings to Highlight

1. **Class Imbalance**: Baseline fails on Severe class (F1=0.0)
2. **FocalLoss**: Improves Severe class detection significantly
3. **CBAM**: Adds interpretability + 2-5% accuracy gain
4. **Augmentation**: Critical for minority class generalization

---

**Need help?** Check the error message and refer to Troubleshooting section above.
