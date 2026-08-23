"""
Train SpineNetV2 with CBAM Attention for RSNA 2024 Dataset.

Improvements over baseline:
1. CBAM attention modules for better feature discrimination
2. FocalLoss to handle class imbalance (Severe is only 4% of data)
3. UncertaintyLoss for automatic task weighting
4. Data augmentation (flip, rotation, brightness, noise)
5. Oversampling for minority classes (Moderate, Severe)

Usage:
    python train_rsna_attention.py --epochs 30 --batch-size 32 --lr 1e-3

Author: SpineNetV2 Improved Implementation
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

# SpineNetV2 imports
from spinenet.models.grading_attention import GradingModelWithCBAM
from spinenet.losses import FocalLoss, UncertaintyLoss, compute_class_weights
from spinenet.augmentation import get_training_augmentation, OversamplingDataset
from spinenet.metrics_logger import MetricsLogger
from spinenet.auc_metrics import (
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)
from rsna_preprocessed_dataloader import RSNAPreprocessedDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RSNA Attention Model')

    # Data
    parser.add_argument('--data-dir', type=str, default='rsna_preprocessed',
                        help='Path to preprocessed data directory')
    parser.add_argument('--weights-dir', type=str, default=os.path.expanduser('~/.spinenet/weights'),
                        help='Path to pretrained backbone weights')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # Training
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')

    # Model
    parser.add_argument('--use-cbam', action='store_true', default=True,
                        help='Use CBAM attention')
    parser.add_argument('--no-cbam', action='store_false', dest='use_cbam',
                        help='Disable CBAM attention')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                        help='Freeze pretrained backbone')
    parser.add_argument('--no-freeze', action='store_false', dest='freeze_backbone',
                        help='Train backbone (no freezing)')

    # Loss
    parser.add_argument('--use-focal', action='store_true', default=True,
                        help='Use FocalLoss')
    parser.add_argument('--focal-gamma', type=float, default=1.8,
                        help='Focal loss gamma parameter (v2 fresh_cbam used 1.8)')
    parser.add_argument('--use-uncertainty', type=lambda x: str(x).lower() == 'true',
                        default=True, help='Use UncertaintyLoss for task weighting (default: True)')
    parser.add_argument('--class-weight-mode', type=str, default='none',
                        choices=['none', 'sqrt', 'inverse', 'effective'],
                        help='Class weight mode for FocalLoss alpha. '
                             'none=no weights (preserves accuracy). '
                             'sqrt=mild boost for minority (~3-4x Severe weight). '
                             'inverse=strong boost (~10x Severe, may drop overall acc). '
                             'effective=class-balanced loss (Cui et al. 2019).')

    # Augmentation
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['none', 'light', 'medium', 'heavy'],
                        help='Augmentation strength')
    # See the matching note in train_rsna_hybrid.py: the flip is
    # anterior-posterior, not left-right, so swapping L/R labels on it corrupts
    # them. Default flipped to False, which reproduces every published run.
    parser.add_argument('--hflip-swap-labels', dest='hflip_swap_labels',
                        action='store_true', default=False,
                        help='DEPRECATED and label-corrupting: swap left_*/right_* '
                             'on the anterior-posterior flip. Reproduction only.')
    parser.add_argument('--no-hflip-swap-labels', dest='hflip_swap_labels',
                        action='store_false',
                        help='Do not swap labels on the AP flip (default, correct).')
    parser.add_argument('--no-ap-flip', dest='ap_flip', action='store_false',
                        default=True,
                        help='Drop the anterior-posterior flip entirely (it mirrors '
                             'the spine front-to-back). On by default for parity '
                             'with published runs.')
    parser.add_argument('--slice-reverse', dest='slice_reverse',
                        action='store_true', default=False,
                        help='Reverse sagittal slice order AND swap left_*/right_* '
                             'labels -- the correct laterality augmentation. Off by '
                             'default; never used in a published run.')
    parser.add_argument('--oversample-factor', type=int, default=5,
                        help='Oversampling factor for minority classes')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints + metrics')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='Early stopping patience')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split + torch + numpy + cudnn')

    return parser.parse_args()


def set_seed(seed: int):
    """Make training as deterministic as possible across torch/numpy/python."""
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, criterion, uncertainty_loss, device):
    """Evaluate model on validation set."""
    model.eval()

    all_losses = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}
    all_preds = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}
    all_labels = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}
    all_probs = {'spinal_canal': [], 'left_foraminal': [], 'right_foraminal': []}
    per_class_metrics = {}

    val_pbar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for volumes, labels in val_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            # Move labels to device
            labels_device = {
                'spinal_canal': labels['spinal_canal'].to(device),
                'left_foraminal': labels['left_foraminal'].to(device),
                'right_foraminal': labels['right_foraminal'].to(device)
            }

            # Forward pass
            outputs = model(volumes)

            # Compute per-task losses
            task_losses = []
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                loss = criterion(outputs[condition], labels_device[condition])
                task_losses.append(loss)
                all_losses[condition].append(loss.item())

                logits = outputs[condition]
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, 3]
                preds = torch.argmax(logits, dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())
                all_probs[condition].append(probs)

    # Compute metrics per condition
    val_accuracies = {}
    val_weighted_logloss = 0.0

    probs_dict = {c: np.concatenate(all_probs[c], axis=0) for c in all_probs}
    labels_dict = {c: np.array(all_labels[c]) for c in all_labels}

    for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
        labels_np = labels_dict[condition]
        preds_np = np.array(all_preds[condition])

        # Filter out -1 labels
        valid_mask = labels_np != -1
        labels_filtered = labels_np[valid_mask]
        preds_filtered = preds_np[valid_mask]

        # Accuracy
        if len(labels_filtered) > 0:
            acc = accuracy_score(labels_filtered, preds_filtered)
            val_accuracies[condition] = acc
        else:
            val_accuracies[condition] = 0.0

        # Per-class metrics: precision, recall, F1
        if len(labels_filtered) > 0:
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

    # Per-class AUC / AUPRC / Brier (one-vs-rest)
    auc_auprc_metrics = compute_auc_auprc_per_condition(probs_dict, labels_dict)
    auc_auprc_overall = aggregate_overall_auprc(auc_auprc_metrics)

    # Compute total loss
    avg_loss = np.mean([np.mean(all_losses[c]) for c in ['spinal_canal', 'left_foraminal', 'right_foraminal']])

    return (avg_loss, val_weighted_logloss, val_accuracies, per_class_metrics,
            auc_auprc_metrics, auc_auprc_overall)


def print_per_class_metrics(per_class_metrics):
    """Print precision, recall, F1-score per class for each condition."""
    conditions = ['spinal_canal', 'left_foraminal', 'right_foraminal']
    condition_names = ['Spinal Canal', 'Left Foraminal', 'Right Foraminal']
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    print("\n" + "="*70)
    print("Per-Class Metrics (Precision, Recall, F1-Score)")
    print("="*70)

    for condition, name in zip(conditions, condition_names):
        if condition not in per_class_metrics:
            continue

        metrics = per_class_metrics[condition]
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        support = metrics['support']

        print(f"\n{name}:")
        print(f"{'Class':<16} {'Support':<9} {'Precision':<10} {'Recall':<9} {'F1-Score'}")
        print("-" * 70)

        for i, class_name in enumerate(class_names):
            print(f"{class_name:<16} {support[i]:<9.0f} {precision[i]:<10.3f} "
                  f"{recall[i]:<9.3f} {f1[i]:<9.3f}")

        print("-" * 70)
        # Macro average (unweighted mean - treats all classes equally)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        print(f"{'Macro Avg':<16} {'':<9} {macro_precision:<10.3f} "
              f"{macro_recall:<9.3f} {macro_f1:<9.3f}")


def main():
    args = parse_args()

    # Reproducibility — must come BEFORE any DataLoader / model construction
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SpineNetV2 Attention Model Training for RSNA 2024")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"CBAM: {args.use_cbam}")
    print(f"FocalLoss: {args.use_focal}")
    print(f"UncertaintyLoss: {args.use_uncertainty}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Oversampling: {args.oversample_factor}x for Moderate/Severe")
    print(f"Class weight mode: {args.class_weight_mode}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print("="*70)

    # [1/6] Load datasets
    print(f"\n[1/6] Loading datasets...")

    # Load full dataset
    full_dataset = RSNAPreprocessedDataset(
        data_dir=args.data_dir,
        split='train',
        transform=None
    )
    print(f"  ✓ Loaded {len(full_dataset)} samples")

    # Split by patient (no data leakage!)
    print(f"\n[2/6] Splitting dataset by patient...")
    unique_patients = full_dataset.metadata['study_id'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=args.val_split,
        random_state=args.seed
    )

    train_indices = full_dataset.metadata[full_dataset.metadata['study_id'].isin(train_patients)].index.tolist()
    val_indices = full_dataset.metadata[full_dataset.metadata['study_id'].isin(val_patients)].index.tolist()

    # Create base train/val datasets
    base_train_dataset = Subset(full_dataset, train_indices)
    base_val_dataset = Subset(full_dataset, val_indices)

    print(f"  ✓ Train: {len(base_train_dataset)} samples ({len(train_patients)} patients)")
    print(f"  ✓ Val:   {len(base_val_dataset)} samples ({len(val_patients)} patients)")

    # Apply augmentation to training set
    if args.augmentation != 'none':
        train_transform = get_training_augmentation(
            mode=args.augmentation,
            hflip_swap_labels=args.hflip_swap_labels,
            ap_flip=args.ap_flip,
            slice_reverse=args.slice_reverse,
        )
        geo = ["AP-flip on" if args.ap_flip else "AP-flip OFF"]
        if args.slice_reverse:
            geo.append("slice-reverse + L/R swap ON")
        if args.hflip_swap_labels:
            geo.append("!! AP-flip swaps L/R labels (corrupting)")
        print(f"  ✓ Augmentation: {args.augmentation} ({', '.join(geo)})")
        # Create augmented dataset by wrapping the base dataset
        augmented_full_dataset = RSNAPreprocessedDataset(
            data_dir=args.data_dir,
            split='train',
            transform=train_transform
        )
        train_dataset = Subset(augmented_full_dataset, train_indices)
    else:
        train_dataset = base_train_dataset
        print(f"  ✓ No augmentation")

    # Apply oversampling for minority classes
    if args.oversample_factor > 1:
        train_dataset = OversamplingDataset(
            train_dataset,
            oversample_factor=args.oversample_factor,
            target_classes=[1, 2]  # Moderate, Severe
        )
        print(f"  ✓ Oversampling: {len(train_dataset)} samples (from {len(base_train_dataset)} base samples)")

    # Validation dataset (no augmentation or oversampling)
    val_dataset = base_val_dataset

    # Create dataloaders
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

    print(f"\n✓ Datasets ready")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")

    # [3/7] Create model
    print(f"\n[3/7] Creating model...")
    model = GradingModelWithCBAM(format='rsna', use_cbam=args.use_cbam)
    model = model.to(device)

    # Load pretrained backbone
    if os.path.exists(args.weights_dir):
        print(f"  Loading pretrained backbone from {args.weights_dir}...")
        try:
            model.load_pretrained_backbone(args.weights_dir, verbose=True)
        except Exception as e:
            print(f"  ⚠ Warning: Could not load pretrained weights: {e}")
            print(f"  Continuing with random initialization...")
    else:
        print(f"  ⚠ Warning: Weights directory not found: {args.weights_dir}")
        print(f"  Training from scratch...")

    # Freeze backbone if requested
    if args.freeze_backbone:
        model.freeze_backbone(freeze=True)
        print(f"  ✓ Backbone frozen (training heads + CBAM only)")
    else:
        print(f"  ✓ Full model training (backbone + heads + CBAM)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created: {trainable_params:,} / {total_params:,} trainable params")

    # [4/7] Setup loss functions
    print(f"\n[4/7] Setting up loss functions...")

    # Compute class weights from base dataset (pre-oversampling) when requested
    class_alpha = None
    if args.class_weight_mode != 'none':
        print(f"  Computing class weights (mode={args.class_weight_mode}) from base train set...")
        class_alpha = compute_class_weights(
            base_train_dataset, num_classes=3, mode=args.class_weight_mode,
        )
        print(f"  ✓ Class weights: Normal={class_alpha[0]:.3f} "
              f"Moderate={class_alpha[1]:.3f} Severe={class_alpha[2]:.3f}")

    if args.use_focal:
        if class_alpha is not None:
            print(f"  ✓ FocalLoss (gamma={args.focal_gamma}, class_weight_mode={args.class_weight_mode})")
        else:
            print(f"  ✓ FocalLoss (gamma={args.focal_gamma}, no class weights)")
        criterion = FocalLoss(alpha=class_alpha, gamma=args.focal_gamma, ignore_index=-1)
    else:
        if class_alpha is not None:
            print(f"  ✓ CrossEntropyLoss (class_weight_mode={args.class_weight_mode})")
            criterion = nn.CrossEntropyLoss(weight=class_alpha, ignore_index=-1)
        else:
            print(f"  ✓ CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss(ignore_index=-1)

    if args.use_uncertainty:
        uncertainty_loss = UncertaintyLoss(num_tasks=3).to(device)
        print(f"  ✓ UncertaintyLoss for multi-task weighting")
    else:
        uncertainty_loss = None

    # [5/7] Setup optimizer and scheduler
    print(f"\n[5/7] Setting up optimizer and scheduler...")

    # Separate parameter groups
    if args.use_uncertainty:
        optimizer = AdamW([
            {'params': model.parameters(), 'lr': args.lr},
            {'params': uncertainty_loss.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    print(f"✓ Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"✓ Loss: {'FocalLoss' if args.use_focal else 'CrossEntropyLoss'}")
    print(f"✓ Scheduler: ReduceLROnPlateau (patience=5)")

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  ✓ Resumed from epoch {start_epoch}")

    # [6/7] Training loop
    print(f"\n[6/7] Training for {args.epochs} epochs...")
    print("="*70)

    best_epoch = 0
    epochs_without_improvement = 0
    # Tag output filenames by seed so multi-seed runs don't overwrite.
    # Default seed=42 → no tag (backward compat with v3 results).
    run_tag = "" if args.seed == 42 else f"_seed{args.seed}"
    metrics_logger = MetricsLogger(save_dir=save_dir, prefix=f"attention{run_tag}")

    # Total wall-clock from first training step (for paper Table: train time)
    total_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

        for batch_idx, (volumes, labels) in enumerate(train_pbar):
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            # Move labels to device
            labels_device = {
                'spinal_canal': labels['spinal_canal'].to(device),
                'left_foraminal': labels['left_foraminal'].to(device),
                'right_foraminal': labels['right_foraminal'].to(device)
            }

            # Forward pass
            optimizer.zero_grad()
            outputs = model(volumes)

            # Compute per-task losses
            task_losses = []
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                loss_cond = criterion(outputs[condition], labels_device[condition])
                task_losses.append(loss_cond)

            # Aggregate loss
            if args.use_uncertainty:
                loss, task_weights = uncertainty_loss(task_losses)
            else:
                loss = sum(task_losses) / 3.0

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / num_batches

        # Validation (timed end-to-end for inference throughput stat)
        val_start = time.time()
        (val_loss, val_weighted_logloss, val_accuracies, val_per_class_metrics,
         val_auc_auprc_metrics, val_auc_auprc_overall) = evaluate(
            model, val_loader, criterion, uncertainty_loss, device
        )
        val_elapsed = time.time() - val_start
        n_val_samples = len(val_loader.dataset)
        val_throughput = float(n_val_samples / val_elapsed) if val_elapsed > 0 else 0.0
        val_ms_per_sample = float(1000 * val_elapsed / n_val_samples) if n_val_samples > 0 else 0.0

        # Scheduler step
        scheduler.step(val_loss)

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary ({epoch_time:.1f}s)")
        print("="*70)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Weighted Log Loss: {val_weighted_logloss:.4f}")

        print(f"\nValidation Accuracies:")
        print(f"  Spinal Canal:     {val_accuracies['spinal_canal']:.2%}")
        print(f"  Left Foraminal:   {val_accuracies['left_foraminal']:.2%}")
        print(f"  Right Foraminal:  {val_accuracies['right_foraminal']:.2%}")

        # Print uncertainty weights if using UncertaintyLoss
        if args.use_uncertainty:
            task_weights = uncertainty_loss.get_task_weights().detach().cpu().numpy()
            print(f"\nTask Weights (from UncertaintyLoss):")
            print(f"  Spinal Canal:   {task_weights[0]:.4f}")
            print(f"  Left Foraminal: {task_weights[1]:.4f}")
            print(f"  Right Foraminal: {task_weights[2]:.4f}")

        # Print detailed per-class metrics every epoch (Severe F1 is critical)
        print_per_class_metrics(val_per_class_metrics)

        # Compute avg Severe F1 for logging
        avg_severe_f1 = float('nan')
        if val_per_class_metrics:
            severe_f1s = [
                val_per_class_metrics[c]['f1'][2]
                for c in val_per_class_metrics
                if 'f1' in val_per_class_metrics[c] and len(val_per_class_metrics[c]['f1']) >= 3
            ]
            if severe_f1s:
                avg_severe_f1 = float(np.mean(severe_f1s))

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            best_path = save_dir / f'best_model_attention{run_tag}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, best_path)
            print(f"\n✓ Saved best model to {best_path}")

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

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_attention{run_tag}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")

        print(f"\nBest Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"Epochs without improvement: {epochs_without_improvement}")

        # Early stopping
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {args.early_stop_patience} epochs without improvement")
            break

        print("="*70 + "\n")

    # Final summary
    print("\n" + "="*70)
    print("[7/7] Training Complete!")
    print("="*70)
    print(f"Best Model: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
    print(f"Saved to: {save_dir / f'best_model_attention{run_tag}.pth'}")
    print("="*70)


if __name__ == '__main__':
    main()
