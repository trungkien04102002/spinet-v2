#!/usr/bin/env python3
"""
Train RSNA Grading Model with Transfer Learning

Strategy:
- Phase 1: Freeze backbone, train classification heads only
- Phase 2 (optional): Fine-tune entire network

Usage:
    # Phase 1: Train heads only (recommended start)
    python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3

    # Phase 2: Fine-tune all (optional, if Phase 1 isn't good enough)
    python3 train_rsna_baseline.py --epochs 20 --batch-size 16 --lr 1e-5 --unfreeze-backbone --resume best_model.pth

    # Custom configuration
    python3 train_rsna_baseline.py --epochs 50 --batch-size 64 --lr 5e-4 --weight-decay 1e-4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support, classification_report
from tqdm import tqdm

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.models.grading_baseline import GradingModelBaseline


def compute_weighted_log_loss(outputs_dict, labels_dict):
    """
    Compute weighted log loss (RSNA competition metric)

    Weights per condition:
    - Spinal Canal Stenosis: 1.0
    - Left Neural Foraminal: 1.0
    - Right Neural Foraminal: 1.0
    (All equal weights for simplicity)
    """
    weights = {'spinal_canal': 1.0, 'left_foraminal': 1.0, 'right_foraminal': 1.0}

    total_loss = 0.0
    total_weight = 0.0

    for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
        # Get predictions (probabilities) and labels
        probs = torch.softmax(outputs_dict[condition], dim=1).cpu().numpy()
        labels = labels_dict[condition].cpu().numpy()

        # Filter out -1 labels (missing data)
        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            # Compute log loss for this condition
            condition_loss = log_loss(labels[valid_mask], probs[valid_mask], labels=[0, 1, 2])

            # Weight it
            total_loss += condition_loss * weights[condition]
            total_weight += weights[condition]

    return total_loss / total_weight


def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()

    all_preds = {
        'spinal_canal': [],
        'left_foraminal': [],
        'right_foraminal': []
    }
    all_labels = {
        'spinal_canal': [],
        'left_foraminal': [],
        'right_foraminal': []
    }
    all_outputs = {
        'spinal_canal': [],
        'left_foraminal': [],
        'right_foraminal': []
    }

    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore missing labels (-1)

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validating", leave=False)
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

            # Compute loss
            loss = 0.0
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                loss += criterion(outputs[condition], labels_device[condition])
            loss /= 3.0

            total_loss += loss.item()
            num_batches += 1

            # Store predictions and labels
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                preds = torch.argmax(outputs[condition], dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())
                all_outputs[condition].append(outputs[condition].cpu())

    # Compute metrics
    avg_loss = total_loss / num_batches

    accuracies = {}
    per_class_metrics = {}  # Store precision, recall, F1 per class

    for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
        # Filter out -1 labels (missing data)
        labels_arr = np.array(all_labels[condition])
        preds_arr = np.array(all_preds[condition])
        valid_mask = labels_arr != -1

        if valid_mask.sum() > 0:
            labels_filtered = labels_arr[valid_mask]
            preds_filtered = preds_arr[valid_mask]

            # Overall accuracy
            accuracies[condition] = accuracy_score(labels_filtered, preds_filtered)

            # Per-class metrics: precision, recall, F1
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

    # Compute weighted log loss
    all_outputs_concat = {
        k: torch.cat(v, dim=0) for k, v in all_outputs.items()
    }
    all_labels_tensor = {
        k: torch.tensor(v) for k, v in all_labels.items()
    }
    weighted_logloss = compute_weighted_log_loss(all_outputs_concat, all_labels_tensor)

    return avg_loss, accuracies, weighted_logloss, all_labels, all_preds, per_class_metrics


def print_per_class_metrics(per_class_metrics):
    """Print precision, recall, F1-score per class for each condition"""
    conditions = ['spinal_canal', 'left_foraminal', 'right_foraminal']
    condition_names = ['Spinal Canal', 'Left Foraminal', 'Right Foraminal']
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    print("\n" + "="*70)
    print("Per-Class Metrics (Precision, Recall, F1-Score)")
    print("="*70)

    for condition, name in zip(conditions, condition_names):
        metrics = per_class_metrics.get(condition)
        if metrics is not None:
            print(f"\n{name}:")
            print(f"{'Class':<15} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            print("-" * 70)

            for i, class_name in enumerate(class_names):
                print(f"{class_name:<15} "
                      f"{int(metrics['support'][i]):>8} "
                      f"{metrics['precision'][i]:>10.3f} "
                      f"{metrics['recall'][i]:>10.3f} "
                      f"{metrics['f1'][i]:>10.3f}")

            # Macro average (unweighted mean - treats all classes equally)
            macro_p = np.mean(metrics['precision'])
            macro_r = np.mean(metrics['recall'])
            macro_f1 = np.mean(metrics['f1'])
            print("-" * 70)
            print(f"{'Macro Avg':<15} {'':<8} {macro_p:>10.3f} {macro_r:>10.3f} {macro_f1:>10.3f}")
        else:
            print(f"\n{name}: No valid labels")


def print_confusion_matrices(all_labels, all_preds):
    """Print confusion matrices for each condition"""
    conditions = ['spinal_canal', 'left_foraminal', 'right_foraminal']
    condition_names = ['Spinal Canal', 'Left Foraminal', 'Right Foraminal']

    print("\n" + "="*70)
    print("Confusion Matrices")
    print("="*70)

    for condition, name in zip(conditions, condition_names):
        # Filter out -1 labels (missing data)
        labels_arr = np.array(all_labels[condition])
        preds_arr = np.array(all_preds[condition])
        valid_mask = labels_arr != -1

        if valid_mask.sum() > 0:
            cm = confusion_matrix(labels_arr[valid_mask], preds_arr[valid_mask], labels=[0, 1, 2])
            print(f"\n{name} (excluding {(~valid_mask).sum()} missing labels):")
            print("              Pred: Normal  Moderate  Severe")
            print(f"  True: Normal     {cm[0, 0]:6d}    {cm[0, 1]:6d}  {cm[0, 2]:6d}")
            print(f"        Moderate   {cm[1, 0]:6d}    {cm[1, 1]:6d}  {cm[1, 2]:6d}")
            print(f"        Severe     {cm[2, 0]:6d}    {cm[2, 1]:6d}  {cm[2, 2]:6d}")
        else:
            print(f"\n{name}: No valid labels")


def main():
    parser = argparse.ArgumentParser(description='Train RSNA grading model')

    # Data
    parser.add_argument('--data-dir', type=str, default='rsna_preprocessed',
                        help='Path to preprocessed data')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # Model
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                        help='Use pretrained backbone (default: True)')
    parser.add_argument('--no-pretrained', dest='use_pretrained', action='store_false',
                        help='Train from scratch (not recommended)')
    parser.add_argument('--unfreeze-backbone', action='store_true',
                        help='Unfreeze backbone for fine-tuning (Phase 2)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3 for Phase 1, use 1e-5 for Phase 2)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    print("="*70)
    print("RSNA Grading Model Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Validation split: {args.val_split:.1%}")
    print(f"  Use pretrained: {args.use_pretrained}")
    print(f"  Freeze backbone: {not args.unfreeze_backbone}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Save directory: {args.save_dir}")

    # Load dataset
    print(f"\n[1/6] Loading dataset...")
    dataset = RSNAPreprocessedDataset(
        data_dir=args.data_dir,
        split='train'
    )
    print(f"✓ Loaded {len(dataset)} samples")

    # Split by patient (no data leakage!)
    print(f"\n[2/6] Splitting dataset by patient...")
    unique_patients = dataset.metadata['study_id'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=args.val_split,
        random_state=42
    )

    train_indices = dataset.metadata[dataset.metadata['study_id'].isin(train_patients)].index.tolist()
    val_indices = dataset.metadata[dataset.metadata['study_id'].isin(val_patients)].index.tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"✓ Train: {len(train_dataset)} samples ({len(train_patients)} patients)")
    print(f"✓ Val:   {len(val_dataset)} samples ({len(val_patients)} patients)")

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

    # Load model
    print(f"\n[3/6] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = GradingModelBaseline(format='rsna')

    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  ✓ Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')

        if args.use_pretrained:
            print("  Loading pretrained backbone...")
            weights_dir = os.path.expanduser('~/.spinenet/weights')
            try:
                model.load_pretrained_backbone(weights_dir, verbose=False)
                print("  ✓ Pretrained backbone loaded")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load pretrained weights: {e}")
                print("  → Training from scratch")

    # Freeze/unfreeze backbone
    if args.unfreeze_backbone:
        print("  Unfreezing backbone (fine-tuning mode)")
        model.freeze_backbone(freeze=False)
    else:
        print("  Freezing backbone (training heads only)")
        model.freeze_backbone(freeze=True)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    model = model.to(device)

    # Setup training
    print(f"\n[4/6] Setting up training...")
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore missing labels (-1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    print(f"✓ Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"✓ Loss: CrossEntropyLoss")
    print(f"✓ Scheduler: ReduceLROnPlateau (patience=5)")

    # Training loop
    print(f"\n[5/6] Training for {args.epochs} epochs...")
    print("="*70)

    best_epoch = 0
    epochs_without_improvement = 0

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

            # Compute loss (average of 3 conditions)
            loss = 0.0
            for condition in ['spinal_canal', 'left_foraminal', 'right_foraminal']:
                loss += criterion(outputs[condition], labels_device[condition])
            loss /= 3.0

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / num_batches

        # Validation
        val_loss, val_accuracies, val_weighted_logloss, all_labels, all_preds, val_per_class_metrics = evaluate(
            model, val_loader, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print("\n" + "="*70)
        print(f"Epoch {epoch+1}/{args.epochs} Summary ({epoch_time:.1f}s)")
        print("="*70)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Weighted Log Loss: {val_weighted_logloss:.4f}")
        print(f"\nValidation Accuracies:")
        print(f"  Spinal Canal:     {val_accuracies['spinal_canal']:.2%}")
        print(f"  Left Foraminal:   {val_accuracies['left_foraminal']:.2%}")
        print(f"  Right Foraminal:  {val_accuracies['right_foraminal']:.2%}")

        # Print detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print_per_class_metrics(val_per_class_metrics)
            print_confusion_matrices(all_labels, all_preds)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            best_path = save_dir / 'best_model.pth'
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
            print(f"\n✓ Saved best model to {best_path}")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
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
    print("[6/6] Training Complete!")
    print("="*70)
    print(f"\nBest model:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Saved at: {save_dir / 'best_model.pth'}")

    print("\nNext steps:")
    print("  1. Evaluate on test set: python3 test_rsna_preprocessed.py --model checkpoints/best_model.pth")
    print("  2. Fine-tune (Phase 2): python3 train_rsna_baseline.py --unfreeze-backbone --lr 1e-5 --resume checkpoints/best_model.pth")
    print("="*70)


if __name__ == '__main__':
    main()
