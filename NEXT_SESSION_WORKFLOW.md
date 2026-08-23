# Next Session Workflow — Vast.ai resume

> Self-contained playbook for resuming work tonight. Read top-to-bottom.

## Quick state check (where we are)

- ✅ All 3 RSNA models trained, checkpoints local
  - Baseline: `checkpoints/fresh_baseline/best_model_baseline_full_e25.pth`
  - CBAM + sqrt CW: `checkpoints/fresh/best_model_attention_sqrt_cw_e20.pth`
  - Hybrid (logit_scale fix + sqrt CW): `checkpoints/fresh/hybrid_pth_<SUFFIX>.pth`
- ✅ Best Hybrid result: E7, Avg Severe F1 = 0.353 (see `experiments/fresh_cbam/RESULTS_SUMMARY.md`)
- ⏳ Phase 3 — Zero-shot SPIDER eval (NEXT)
- ⏳ Phase 4 — SPIDER transfer learning

## Before destroying the current instance

Run these on **local Mac** while the Vast.ai instance is still up:

```bash
cd /Users/kienha/spinet-v2

# Backup all checkpoints + metrics with a clear suffix
./backup_from_vastai.sh 39036 root@220.130.209.122 hybrid_fixed_e7
```

Verify before destroying:

```bash
ls -lh checkpoints/fresh/baseline_pth_hybrid_fixed_e7.pth
ls -lh checkpoints/fresh/attention_pth_hybrid_fixed_e7.pth
ls -lh checkpoints/fresh/hybrid_pth_hybrid_fixed_e7.pth
ls -lh experiments/fresh_cbam/hybrid_metrics_txt_hybrid_fixed_e7.txt
```

All four must exist. Once confirmed, destroy the Vast.ai instance.

## Tonight: launching a new Vast.ai instance

### 1. Pick instance

- Any RTX 4090 (24 GB VRAM) works.
- Disk: 50 GB minimum (RSNA NOT needed; SPIDER dataset is smaller).

### 2. SSH in, set up code

```bash
# On the new Vast.ai instance
cd /root
git clone https://github.com/trungkien04102002/spinet-v2.git
cd spinet-v2
git checkout biomedclip-integration
git pull

# Install deps
pip install -r requirements.txt
pip install open_clip_torch transformers SimpleITK
```

### 3. Restore checkpoints from local Mac

On **local Mac**:

```bash
cd /Users/kienha/spinet-v2
./restore_to_vastai.sh <NEW_PORT> root@<NEW_IP> hybrid_fixed_e7
```

This uploads the 3 .pth files to the standard locations the scripts expect:
- `/root/spinet-v2/checkpoints/best_model.pth` (baseline)
- `/root/spinet-v2/checkpoints/best_model_attention.pth` (CBAM)
- `/root/spinet-v2/checkpoints/hybrid/best_model_hybrid.pth` (Hybrid)

### 4. Download SPIDER dataset

If you have it on Google Drive (most likely path):

```bash
# On the new instance — replace <gdrive_id> with your SPIDER gdrive file id
cd /root/spinet-v2
mkdir -p spider
# Use your existing download helper if applicable, or gdown:
pip install gdown
gdown <gdrive_id> -O spider/spider.zip
unzip spider/spider.zip -d spider/
```

Quick sanity check:

```bash
ls /root/spinet-v2/spider/   # expect: images/, masks/, radiological_gradings.csv, etc.
```

If you don't yet have a SPIDER mirror on Drive, the canonical source is the Zenodo SPIDER dataset. You'll need to download once and re-upload to your own storage so subsequent instances are fast.

## Phase 3 — Zero-shot SPIDER eval

```bash
cd /root/spinet-v2
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 1: Build zero-shot test set from SPIDER (~30 min, runs on CPU)
python3 prepare_spider_zeroshot.py

# Step 2: Run zero-shot eval (~10 min on GPU)
python3 eval_zeroshot_spider.py \
    --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \
    --cbam-checkpoint checkpoints/best_model_attention.pth
```

Outputs:
- `experiments/paper_results/spider_zeroshot/results.csv`
- `experiments/paper_results/spider_zeroshot/best_metrics.txt`
- `experiments/paper_results/spider_zeroshot/best_metrics.json`

Targets per `STEPS.md`:
- Easy (Disc_narrowing): F1 macro ≥ 0.50
- Medium (Modic, Spondylolisthesis): F1 macro ≥ 0.35
- Hard (Pfirrman 5-class): F1 macro ≥ 0.25

## Phase 4 — SPIDER transfer learning (after zero-shot)

Two runs, ~2-3 hours each:

```bash
# Vanilla SpineNetV2 transferred to SPIDER (hardcoded SPIDER heads)
python3 train_spider.py --model baseline \
    --rsna-checkpoint checkpoints/best_model.pth \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3

# CBAM SpineNetV2 transferred to SPIDER
python3 train_spider.py --model cbam \
    --rsna-checkpoint checkpoints/best_model_attention.pth \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3
```

Goal: show CBAM ≥ Vanilla on ≥ 2/3 SPIDER tasks → CBAM generalizes.

## Backup before stopping the new instance

```bash
# On local Mac
./backup_from_vastai.sh <NEW_PORT> root@<NEW_IP> spider_phase
```

This grabs whatever new files exist, including SPIDER zero-shot results and SPIDER transfer checkpoints.

## Reference: scripts and key files

| File | Purpose |
|---|---|
| `backup_from_vastai.sh` | local ← Vast.ai (download all important files with suffix) |
| `restore_to_vastai.sh` | local → Vast.ai (upload .pth checkpoints to standard remote names) |
| `experiments/fresh_cbam/RESULTS_SUMMARY.md` | All RSNA results + analysis |
| `STEPS.md` | Master phase plan |
| `HYBRID_TRAINING_GUIDE.md` | Hybrid hyperparameters explained |

## Cheatsheet — what's the BEST result for each phase

| Phase | Method | Best metric | Checkpoint |
|---|---|---|---|
| Internal baseline (lower bound) | `train_rsna_baseline.py` | Avg Severe F1 = 0.152, Mean Acc 81.4% | `best_model.pth` |
| RSNA improvement | CBAM + sqrt CW | Avg Severe F1 = 0.333, Mean Acc 68.6% | `best_model_attention.pth` |
| RSNA Hybrid (best) | Hybrid + sqrt CW (logit fix), E7 | **Avg Severe F1 = 0.353, Mean Acc 70.0%** | `best_model_hybrid.pth` |

## If something goes wrong

- **`restore_to_vastai.sh` says "no local file for prefix X"**: check `ls checkpoints/fresh/` to confirm the file with that suffix actually exists locally. If you used a different suffix during backup, pass it as the third arg.
- **`eval_zeroshot_spider.py` complains about missing test_csv**: you forgot to run `prepare_spider_zeroshot.py` first.
- **BiomedCLIP download is slow**: the first `train_rsna_hybrid.py` / `eval_zeroshot_spider.py` run on a fresh instance downloads ~600 MB from HuggingFace Hub. Subsequent runs read from `~/.cache/huggingface/hub/`. Don't interrupt the first download.
- **Logit scale is huge (>100) in eval output**: you're running an old checkpoint trained before the logit_scale fix (commit `21b4922`). Use the post-fix Hybrid checkpoint (`hybrid_pth_hybrid_fixed_e7.pth`).
