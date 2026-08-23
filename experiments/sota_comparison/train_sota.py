#!/usr/bin/env python3
"""
Train external SOTA-baseline grading models on RSNA-2024, using the SAME
harness (split, loss, metrics, JSON schema) as ../../train_rsna_baseline.py
so the resulting numbers are directly comparable to our Table 1.

This is a near-verbatim copy of train_rsna_baseline.py: the patient-level
train/val split code (train_test_split(unique_patients, test_size=..,
random_state=seed)), the -1-ignoring CrossEntropyLoss, the per-class metric
computation, and the best_metrics.json schema (via spinenet.metrics_logger.
MetricsLogger) are preserved byte-identical. The only difference is which
model class is instantiated (--model flag) and where checkpoints/metrics are
written (experiments/sota_comparison/checkpoints/<model>/).

Available --model choices (see MODEL_REGISTRY below):
    brendanartley  2D-CNN -> BiLSTM -> attention-pool (recurrent paradigm)
    transformer    2D-CNN tokens -> [CLS] + positional embed -> TransformerEncoder

Run from the repo root with PYTHONPATH set to the repo root, e.g.:

    source spinenet-venv/bin/activate
    export PYTHONPATH=/Users/kienha/spinet-v2
    python3 experiments/sota_comparison/train_sota.py \
        --model brendanartley --seed 42 --epochs 30 --batch-size 32 --lr 1e-3

Smoke test (tiny subset, 1 epoch, CPU-friendly):

    python3 experiments/sota_comparison/train_sota.py \
        --model brendanartley --fast-dev --epochs 1 --batch-size 2 \
        --num-workers 0 --save-dir experiments/sota_comparison/checkpoints/_smoketest
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from tqdm import tqdm

# --- make repo-root imports work regardless of CWD -------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # experiments/sota_comparison -> repo root
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.metrics_logger import MetricsLogger
from spinenet.auc_metrics import (
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)


def _load_module_from_path(module_name: str, file_path: Path):
    """Load a module by explicit file path.

    NOTE: we deliberately avoid ``from models.grading_brendanartley import
    ...`` here. ``spinenet/main.py`` does ``sys.path.append(str(Path(
    __file__).parent))`` (adds the ``spinenet/`` dir to sys.path), and
    ``spinenet/models/`` is a REGULAR package (has __init__.py). Since a
    regular package match wins over — and discards — any namespace-package
    portions found earlier on sys.path, a bare top-level ``import models``
    can silently resolve to ``spinenet/models`` instead of our
    ``experiments/sota_comparison/models`` dir, even though the latter comes
    first in sys.path. Loading by explicit path sidesteps that collision
    entirely.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_grading_brendanartley = _load_module_from_path(
    "sota_grading_brendanartley", _THIS_DIR / "models" / "grading_brendanartley.py"
)
GradingModelBrendanartley = _grading_brendanartley.GradingModelBrendanartley

_grading_transformer = _load_module_from_path(
    "sota_grading_transformer", _THIS_DIR / "models" / "grading_transformer.py"
)
GradingModelTransformer = _grading_transformer.GradingModelTransformer


MODEL_REGISTRY = {
    "brendanartley": GradingModelBrendanartley,
    "transformer": GradingModelTransformer,
}


def build_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown --model {name!r}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](format="rsna")


def compute_weighted_log_loss(outputs_dict, labels_dict):
    """Weighted log loss (RSNA competition metric). Identical to
    train_rsna_baseline.py::compute_weighted_log_loss."""
    weights = {'spinal_canal': 1.0, 'left_foraminal': 1.0, 'right_foraminal': 1.0}

    total_loss = 0.0
    total_weight = 0.0

    for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
        probs = torch.softmax(outputs_dict[condition], dim=1).cpu().numpy()
        labels = labels_dict[condition].cpu().numpy()

        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            condition_loss = log_loss(labels[valid_mask], probs[valid_mask], labels=[0, 1, 2])
            total_loss += condition_loss * weights[condition]
            total_weight += weights[condition]

    return total_loss / total_weight if total_weight > 0 else float('nan')


def evaluate(model, dataloader, device):
    """Evaluate model on validation set. Identical logic to
    train_rsna_baseline.py::evaluate."""
    model.eval()

    all_preds = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}
    all_labels = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}
    all_outputs = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}

    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validating", leave=False)
        for volumes, labels in val_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            labels_device = {
                'spinal_canal': labels['spinal_canal'].to(device),
                'left_foraminal': labels['left_foraminal'].to(device),
                'right_foraminal': labels['right_foraminal'].to(device)
            }

            outputs = model(volumes)

            loss = 0.0
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                loss += criterion(outputs[condition], labels_device[condition])
            loss /= 3.0

            total_loss += loss.item()
            num_batches += 1

            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                preds = torch.argmax(outputs[condition], dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())
                all_outputs[condition].append(outputs[condition].cpu())

    avg_loss = total_loss / num_batches

    accuracies = {}
    per_class_metrics = {}

    for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
        labels_arr = np.array(all_labels[condition])
        preds_arr = np.array(all_preds[condition])
        valid_mask = labels_arr != -1

        if valid_mask.sum() > 0:
            labels_filtered = labels_arr[valid_mask]
            preds_filtered = preds_arr[valid_mask]

            accuracies[condition] = accuracy_score(labels_filtered, preds_filtered)

            precision, recall, f1, support = precision_recall_fscore_support(
                labels_filtered, preds_filtered,
                labels=[0, 1, 2],
                average=None,
                zero_division=0
            )

            per_class_metrics[condition] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
        else:
            accuracies[condition] = 0.0
            per_class_metrics[condition] = None

    all_outputs_concat = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
    all_labels_tensor = {k: torch.tensor(v) for k, v in all_labels.items()}
    weighted_logloss = compute_weighted_log_loss(all_outputs_concat, all_labels_tensor)

    probs_dict = {k: torch.softmax(v, dim=1).numpy() for k, v in all_outputs_concat.items()}
    labels_dict = {k: np.array(v) for k, v in all_labels.items()}
    auc_auprc_metrics = compute_auc_auprc_per_condition(probs_dict, labels_dict)
    auc_auprc_overall = aggregate_overall_auprc(auc_auprc_metrics)

    return (avg_loss, accuracies, weighted_logloss, all_labels, all_preds,
            per_class_metrics, auc_auprc_metrics, auc_auprc_overall)


def main():
    parser = argparse.ArgumentParser(description='Train external SOTA-baseline RSNA grading models')

    # Model selection (new vs train_rsna_baseline.py)
    parser.add_argument('--model', type=str, default='brendanartley',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Which external SOTA model to train (default: brendanartley)')

    # Data
    parser.add_argument('--data-dir', type=str, default=str(_REPO_ROOT / 'rsna_preprocessed'),
                        help='Path to preprocessed data (default: <repo_root>/rsna_preprocessed)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze the per-frame CNN encoder (heads/LSTM/attention stay trainable)')

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save checkpoints/metrics '
                             '(default: experiments/sota_comparison/checkpoints/<model>)')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split + torch + numpy + cudnn')

    # Smoke-test path — tiny subset, quick to run on CPU/MPS.
    parser.add_argument('--fast-dev', action='store_true',
                        help='Smoke-test mode: subsample the dataset to --fast-dev-samples '
                             'and skip early stopping bookkeeping noise. NOT for real training.')
    parser.add_argument('--fast-dev-samples', type=int, default=20,
                        help='Number of samples to keep in fast-dev mode (default: 20)')

    args = parser.parse_args()

    # Reproducibility — must come BEFORE any DataLoader / model construction
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_tag = "" if args.seed == 42 else f"_seed{args.seed}"

    save_dir = Path(args.save_dir) if args.save_dir else (
        _THIS_DIR / "checkpoints" / args.model
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"External SOTA Baseline Training — model={args.model}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Validation split: {args.val_split:.1%}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Save directory: {save_dir}")
    print(f"  Fast-dev (smoke test): {args.fast_dev}")

    # Load dataset
    print(f"\n[1/6] Loading dataset...")
    dataset = RSNAPreprocessedDataset(data_dir=args.data_dir, split='train')
    print(f"Loaded {len(dataset)} samples")

    # Split by patient (no data leakage!) — byte-identical to
    # train_rsna_baseline.py's split code.
    print(f"\n[2/6] Splitting dataset by patient...")
    unique_patients = dataset.metadata['study_id'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=args.val_split,
        random_state=args.seed
    )

    train_indices = dataset.metadata[dataset.metadata['study_id'].isin(train_patients)].index.tolist()
    val_indices = dataset.metadata[dataset.metadata['study_id'].isin(val_patients)].index.tolist()

    if args.fast_dev:
        # Smoke-test only: shrink both splits so a forward+backward pass
        # runs in seconds. Never used for real training numbers.
        train_indices = train_indices[:args.fast_dev_samples]
        val_indices = val_indices[:max(2, args.fast_dev_samples // 4)]
        print(f"  [fast-dev] Subsampled to {len(train_indices)} train / {len(val_indices)} val indices")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train: {len(train_dataset)} samples ({len(train_patients)} patients)")
    print(f"Val:   {len(val_dataset)} samples ({len(val_patients)} patients)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load model
    print(f"\n[3/6] Loading model ({args.model})...")
    device = torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = build_model(args.model)

    if args.freeze_backbone:
        print("  Freezing per-frame CNN encoder (heads/LSTM/attention trainable)")
        model.freeze_backbone(freeze=True)
    else:
        print("  Training full model (no pretrained backbone available for this architecture)")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    model = model.to(device)

    # Setup training
    print(f"\n[4/6] Setting up training...")
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # ignore missing labels (-1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Loss: CrossEntropyLoss (ignore_index=-1)")
    print(f"Scheduler: ReduceLROnPlateau (patience=5)")

    # Training loop
    print(f"\n[5/6] Training for {args.epochs} epochs...")
    print("=" * 70)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    metrics_logger = MetricsLogger(save_dir=save_dir, prefix=f"{args.model}{run_tag}")

    total_train_start = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)

        for batch_idx, (volumes, labels) in enumerate(train_pbar):
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            labels_device = {
                'spinal_canal': labels['spinal_canal'].to(device),
                'left_foraminal': labels['left_foraminal'].to(device),
                'right_foraminal': labels['right_foraminal'].to(device)
            }

            optimizer.zero_grad()
            outputs = model(volumes)

            loss = 0.0
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                loss += criterion(outputs[condition], labels_device[condition])
            loss /= 3.0

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / num_batches

        val_start = time.time()
        (val_loss, val_accuracies, val_weighted_logloss, all_labels, all_preds,
         val_per_class_metrics, val_auc_auprc_metrics, val_auc_auprc_overall) = evaluate(
            model, val_loader, device
        )
        val_elapsed = time.time() - val_start
        n_val_samples = len(val_loader.dataset)
        val_throughput = float(n_val_samples / val_elapsed) if val_elapsed > 0 else 0.0
        val_ms_per_sample = float(1000 * val_elapsed / n_val_samples) if n_val_samples > 0 else 0.0

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs} Summary ({epoch_time:.1f}s)")
        print("=" * 70)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Weighted Log Loss: {val_weighted_logloss:.4f}")
        print(f"\nValidation Accuracies:")
        print(f"  Spinal Canal:     {val_accuracies['spinal_canal']:.2%}")
        print(f"  Left Foraminal:   {val_accuracies['left_foraminal']:.2%}")
        print(f"  Right Foraminal:  {val_accuracies['right_foraminal']:.2%}")

        avg_severe_f1 = float('nan')
        if val_per_class_metrics:
            severe_f1s = [
                val_per_class_metrics[c]['f1'][2]
                for c in val_per_class_metrics
                if val_per_class_metrics[c] is not None and len(val_per_class_metrics[c]['f1']) >= 3
            ]
            if severe_f1s:
                avg_severe_f1 = float(np.mean(severe_f1s))

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_path = save_dir / f'best_model_{args.model}{run_tag}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_weighted_logloss': val_weighted_logloss,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
            }, best_path)
            print(f"\nSaved best model to {best_path}")

            elapsed_total = time.time() - total_train_start
            metrics_logger.save_best(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_accuracies=val_accuracies,
                per_class_metrics=val_per_class_metrics,
                avg_severe_f1=avg_severe_f1,
                auc_auprc_metrics=val_auc_auprc_metrics,
                auc_auprc_overall=val_auc_auprc_overall,
                extra={
                    'val_weighted_logloss': float(val_weighted_logloss),
                    'best_path': str(best_path),
                    'total_train_seconds': float(elapsed_total),
                    'avg_epoch_seconds': float(elapsed_total / (epoch + 1)),
                    'eval_seconds': float(val_elapsed),
                    'eval_samples': int(n_val_samples),
                    'eval_throughput_samples_per_sec': val_throughput,
                    'eval_ms_per_sample': val_ms_per_sample,
                    'args': vars(args),
                },
            )
        else:
            epochs_without_improvement += 1

        metrics_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            val_accuracies=val_accuracies,
            per_class_metrics=val_per_class_metrics,
            avg_severe_f1=avg_severe_f1,
            is_best=is_best,
            auc_auprc_metrics=val_auc_auprc_metrics,
            auc_auprc_overall=val_auc_auprc_overall,
            extra={
                'val_weighted_logloss': float(val_weighted_logloss),
                'epoch_seconds': float(epoch_time),
                'eval_seconds': float(val_elapsed),
                'eval_throughput_samples_per_sec': val_throughput,
            },
        )

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_{args.model}_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_weighted_logloss': val_weighted_logloss,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        print(f"\nBest Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"Epochs without improvement: {epochs_without_improvement}")

        if not args.fast_dev and epochs_without_improvement >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {args.early_stop_patience} epochs without improvement")
            break

        print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("[6/6] Training Complete!")
    print("=" * 70)
    print(f"\nBest model:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Metrics JSON: {save_dir / f'{args.model}{run_tag}_best_metrics.json'}")


if __name__ == '__main__':
    main()
