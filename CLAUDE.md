# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SpineNetV2 is a research codebase for analyzing spinal MRI scans (vertebra detection/labelling and per-IVD radiological grading). This fork extends the upstream library with training/evaluation pipelines for two external datasets — **RSNA 2024 Lumbar Spine Degenerative Classification** and **SPIDER** — and adds CBAM attention, focal/uncertainty losses, and augmentation on top of the original ResNet34-3D grading backbone for ablation studies.

Upstream inference (`spinenet.SpineNet`) and the RSNA/SPIDER training pipelines live side-by-side in this repo and mostly share only the backbone architecture, weights format, and utilities.

## Environment setup

```bash
python -m venv spinenet-venv
source spinenet-venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)   # required — this repo is NOT installed as a package
```

`local_setup.sh` automates local macOS setup; `vast_setup.sh` provisions a Vast.ai GPU instance (supports `--branch <name>`). Pre-commit runs `black` (line length 79) and `flake8`.

There are **two separate weight stores** — don't confuse them:
- `./spinenet/weights/` — original VGG SpineNet weights (detect-vfr, appearance, context, grading). Downloaded via `spinenet.download_weights()` in Python, consumed by `SpineNet(...)` inference.
- `~/.spinenet/weights/ckpt1.pt` — the pretrained 3D ResNet34 backbone reused by `train_rsna_*` / `train_spider.py`. Downloaded via `./3_download_weights.sh <gdrive_file_id>`.

## Common commands

RSNA preprocessing (DICOM → `(9, 112, 224)` float32 `.npy` per IVD, ~8GB):
```bash
./1_prepare_locally.sh                         # local: DICOM → .npy → zip
./2_download_preprocessed.sh <gdrive_file_id>  # GPU box: download + extract zip
python3 prepare_rsna_data.py --skip-download   # direct run if DICOMs already present
python3 visualize_npy.py --show-stats --num-samples 3   # verify preprocessing
```

Training (RSNA):
```bash
python3 train_rsna_baseline.py  --epochs 30 --batch-size 32 --lr 1e-3
python3 train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3
# Ablation flags: --no-cbam  --use-focal=False  --use-uncertainty=False
#                 --augmentation {none,light,medium,heavy}  --oversample-factor N
# Fine-tune full backbone: --no-freeze --lr 1e-4   (always drop LR when unfreezing)
# Resume: --resume checkpoints/checkpoint_attention_epoch_30.pth
```

Training (SPIDER transfer learning — requires an existing RSNA checkpoint):
```bash
python3 train_spider.py --model baseline \
    --rsna-checkpoint checkpoints/best_model_baseline.pth \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3
python3 train_spider.py --model cbam \
    --rsna-checkpoint checkpoints/best_model_attention.pth \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3
```

Evaluation:
```bash
python3 test_rsna_preprocessed.py --model checkpoints/best_model_attention.pth --num-patients 5
python3 test_spider.py --model cbam --checkpoint checkpoints_spider/best_model_cbam.pth
python3 test_spinenet.py          # upstream end-to-end pipeline on example scans (CPU-friendly)
python3 test_spinenet_on_rsna.py  # upstream pipeline applied to RSNA DICOMs
```

Running a single unit-style smoke test (there is no pytest suite):
```bash
python3 spider_dataloader.py      # dataloader self-tests run in __main__ blocks
python3 rsna_preprocessed_dataloader.py
```

## Architecture

### Upstream inference pipeline (`spinenet/`)
`SpineNet` (`spinenet/main.py`) composes four models loaded from `./spinenet/weights/`:
1. `VFRResNetDetector` (`models/vfr.py`) — corner+centroid detection of vertebral bodies.
2. `AppearanceModel` + `ContextModel` (`models/appearance.py`, `models/context.py`) — sequence labelling of detected vertebrae.
3. `GradingModel` (`models/grading.py`) — 11-task radiological grading from IVD volumes.

Calling convention: `detect_vb(volume, pixel_spacing) → vert_dicts`, then `get_ivds_from_vert_dicts(vert_dicts, volume) → ivd_dicts`, then `grade_ivds(ivd_dicts) → DataFrame`. The detection threshold knobs (`corner_threshold`, `centroid_threshold`, `group_across_slices_threshold`) are attributes on the `SpineNet` instance. `scan_type='lumbar'` disables black-space trimming; `'whole'` enables it.

### Training grading backbones (the part this fork extends)
All three training variants share the same 3D ResNet34 architecture (`BasicBlock` with stride only on spatial dims, BN3d, 9 sagittal slices × 112 × 224) and the multi-head `format='rsna'` output:

- `models/grading_baseline.py` — `GradingModelBaseline`, single linear head per task.
- `models/grading_attention.py` — `GradingModelWithCBAM`, same backbone + `CBAM` (`models/attention.py`) after each ResNet stage. Reuses `BasicBlock`, `conv3x3`, `conv1x1` from the baseline module — do not duplicate them.
- `models/grading_spider.py` — SPIDER variant: same backbone, but heads are `pfirrmann (5)`, `spondylolisthesis (2)`, `disc_herniation (2)` instead of `spinal_canal / left_foraminal / right_foraminal (3 each)`.

`spinenet/losses.py` provides `FocalLoss` (with `ignore_index=-1` for missing labels), `UncertaintyLoss` (Kendall et al. learned log-variance task weighting), and `compute_class_weights`. `spinenet/augmentation.py` exposes `get_training_augmentation(strength)` and `OversamplingDataset` for minority-class oversampling.

### Data pipeline
Input volumes are normalized to `(9, 112, 224)` float32 in `[0, 1]` regardless of source dataset:
- `rsna_dataloader.py` — slow path, reads DICOM from `rsna-2024-lumbar-spine-degenerative-classification/`.
- `rsna_preprocessed_dataloader.py` — fast path, reads `.npy` under `rsna_preprocessed/volumes/<study_id>/<series_id>_<level>.npy` + `{train,test}_metadata.csv`. **Use this for training.**
- `spider_dataloader.py` — reads SPIDER `.mha` via SimpleITK, extracts per-IVD crops using segmentation mask labels 201–207, resamples to the shared `(9, 112, 224)` shape.

Label convention in the RSNA pipeline: `-1` means "missing label" and MUST be preserved through augmentation and class-weight computation; `compute_class_weights` and `FocalLoss` both honor it via `ignore_index`.

### Checkpoint conventions
- RSNA models → `checkpoints/best_model.pth` (baseline) and `checkpoints/best_model_attention.pth` (CBAM); periodic snapshots `checkpoints/checkpoint_{baseline,attention}_epoch_N.pth`.
- SPIDER models → `checkpoints_spider/best_model_{baseline,cbam}.pth`.
- Under `checkpoints/` there are further subdirs (`rsna/`, `spider/`, `no_class_weight/`) used for experiment comparisons — keep experiment artifacts scoped to one of these rather than at the root.

## Things to know when editing

- **PYTHONPATH**: Imports like `from spinenet.models.grading_attention import ...` rely on the repo root being on `sys.path`. Running scripts from subdirs without setting `PYTHONPATH` will break imports.
- **No pip-installable package**: `pyproject.toml` only configures Black. Don't `pip install -e .` — there's no setup metadata.
- **Don't re-download big data**: `rsna_preprocessed/`, `rsna-2024-lumbar-spine-degenerative-classification/`, `checkpoints/`, `*.npy`, `*.pth`, `spider/`, and `spinenet-venv/` are gitignored. `rsna_preprocessed.zip` (~8GB) lives at repo root on some machines — don't delete without asking.
- **Class imbalance is the whole point**: the attention/focal/uncertainty/oversample stack exists because Severe class F1 is ~0.0 on the baseline. When changing training code, watch per-class F1 (printed every 5 epochs), not just mean accuracy.
- **RSNA vs SPIDER head shapes differ** — swapping checkpoints between `GradingModelBaseline`/`WithCBAM` and `grading_spider.py` requires skipping head weights during load (`train_spider.py` already does this via the `--rsna-checkpoint` flow).

## Reference docs in the repo

- `README.md` — upstream SpineNet description, citations, license (non-commercial).
- `SETUP_GUIDE.md` — local macOS + Vast.ai onboarding.
- `RSNA_PIPELINE.md` — end-to-end RSNA preprocessing + training walkthrough.
- `COMMANDS.md` — exhaustive command reference with expected timings and results.
- `SPIDER_TRAINING_GUIDE.md` — SPIDER transfer-learning workflow and expected metrics.
- `tutorials/01-quickstart.ipynb` — upstream inference example.
