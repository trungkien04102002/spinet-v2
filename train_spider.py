"""
Train SpineNetV2 on SPIDER Dataset (Transfer Learning from RSNA).

Supports both Baseline and CBAM models with SPIDER-specific heads:
- Pfirrmann: 5 classes (grades 1-5)
- Spondylolisthesis: 2 classes (No/Yes)
- Disc herniation: 2 classes (No/Yes)

Transfer Learning Strategy:
1. Load pretrained RSNA backbone (+ CBAM if using CBAM model)
2. Initialize new classification heads randomly
3. Option A: Freeze backbone, train only heads (linear probing)
4. Option B: Fine-tune entire model with small learning rate

Usage:
    # Train baseline with frozen backbone
    python train_spider.py --model baseline --freeze-backbone --epochs 15

    # Train CBAM with frozen backbone
    python train_spider.py --model cbam --freeze-backbone --epochs 15 \\
        --rsna-checkpoint checkpoints/best_model_attention.pth

    # Fine-tune entire CBAM model
    python train_spider.py --model cbam --no-freeze --epochs 30 --lr 1e-4 \\
        --rsna-checkpoint checkpoints/best_model_attention.pth

Author: SpineNetV2 Transfer Learning
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

# SpineNetV2 imports
from spinenet.models.grading_spider import GradingModelSPIDERBaseline, GradingModelSPIDERCBAM
from spider_dataloader import SPIDERDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SPIDER Transfer Learning Model')

    # Data
    parser.add_argument('--data-dir', type=str, default='spider',
                        help='Path to SPIDER dataset directory')
    parser.add_argument('--modality', type=str, default='t2', choices=['t1', 't2'],
                        help='MRI modality (default: t2)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # Model
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'cbam'],
                        help='Model type: baseline or cbam (default: baseline)')
    parser.add_argument('--rsna-checkpoint', type=str, default=None,
                        help='Path to RSNA pretrained checkpoint (required for transfer learning)')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze backbone (linear probing)')
    parser.add_argument('--no-freeze', action='store_false', dest='freeze_backbone',
                        help='Train entire model (fine-tuning)')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16, smaller than RSNA due to memory)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')

    # Loss
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'weighted'],
                        help='Loss type: ce (CrossEntropy) or weighted (class weights)')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='Early stopping patience')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def compute_class_weights_spider(dataset):
    """
    Compute class weights for SPIDER dataset.

    Returns:
        dict with 'pfirrmann', 'spondylolisthesis', 'disc_herniation' weights
    """
    pfirrmann_counts = torch.zeros(5)
    spondy_counts = torch.zeros(2)
    herniation_counts = torch.zeros(2)

    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        pfirrmann_counts[labels['pfirrmann']] += 1
        spondy_counts[labels['spondylolisthesis']] += 1
        herniation_counts[labels['disc_herniation']] += 1

    # Compute inverse frequency weights
    pfirrmann_weights = 1.0 / pfirrmann_counts
    spondy_weights = 1.0 / spondy_counts
    herniation_weights = 1.0 / herniation_counts

    # Normalize
    pfirrmann_weights = pfirrmann_weights / pfirrmann_weights.sum() * 5
    spondy_weights = spondy_weights / spondy_weights.sum() * 2
    herniation_weights = herniation_weights / herniation_weights.sum() * 2

    return {
        'pfirrmann': pfirrmann_weights,
        'spondylolisthesis': spondy_weights,
        'disc_herniation': herniation_weights
    }


def evaluate(model, dataloader, criteria, device):
    """Evaluate model on validation set."""
    model.eval()

    all_losses = {'pfirrmann': [], 'spondylolisthesis': [], 'disc_herniation': []}
    all_preds = {'pfirrmann': [], 'spondylolisthesis': [], 'disc_herniation': []}
    all_labels = {'pfirrmann': [], 'spondylolisthesis': [], 'disc_herniation': []}

    val_pbar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for volumes, labels in val_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            # Move labels to device
            labels_device = {
                'pfirrmann': labels['pfirrmann'].to(device),
                'spondylolisthesis': labels['spondylolisthesis'].to(device),
                'disc_herniation': labels['disc_herniation'].to(device)
            }

            # Forward pass
            outputs = model(volumes)

            # Compute per-task losses
            for condition in ['pfirrmann', 'spondylolisthesis', 'disc_herniation']:
                loss = criteria[condition](outputs[condition], labels_device[condition])
                all_losses[condition].append(loss.item())

                # Get predictions
                preds = torch.argmax(outputs[condition], dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())

    # Compute metrics per condition
    val_accuracies = {}
    per_class_metrics = {}

    for condition in ['pfirrmann', 'spondylolisthesis', 'disc_herniation']:
        labels_np = np.array(all_labels[condition])
        preds_np = np.array(all_preds[condition])

        # Accuracy
        acc = accuracy_score(labels_np, preds_np)
        val_accuracies[condition] = acc

        # Per-class metrics
        if condition == 'pfirrmann':
            num_classes = 5
        else:
            num_classes = 2

        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, preds_np,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0
        )

        per_class_metrics[condition] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

    # Compute total loss
    avg_loss = np.mean([np.mean(all_losses[c]) for c in ['pfirrmann', 'spondylolisthesis', 'disc_herniation']])

    return avg_loss, val_accuracies, per_class_metrics


def print_per_class_metrics(per_class_metrics):
    """Print per-class metrics for SPIDER dataset."""
    print("\n" + "="*70)
    print("Per-Class Metrics (Precision, Recall, F1-Score)")
    print("="*70)

    # Pfirrmann (5 classes)
    if 'pfirrmann' in per_class_metrics:
        metrics = per_class_metrics['pfirrmann']
        print(f"\nPfirrmann Grading (5 classes):")
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        class_names = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
        for i, name in enumerate(class_names):
            print(f"  {name:<15} {metrics['precision'][i]:<12.3f} "
                  f"{metrics['recall'][i]:<12.3f} {metrics['f1'][i]:<12.3f} "
                  f"{int(metrics['support'][i]):<10}")

    # Spondylolisthesis (binary)
    if 'spondylolisthesis' in per_class_metrics:
        metrics = per_class_metrics['spondylolisthesis']
        print(f"\nSpondylolisthesis (2 classes):")
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        class_names = ['No', 'Yes']
        for i, name in enumerate(class_names):
            print(f"  {name:<15} {metrics['precision'][i]:<12.3f} "
                  f"{metrics['recall'][i]:<12.3f} {metrics['f1'][i]:<12.3f} "
                  f"{int(metrics['support'][i]):<10}")

    # Disc herniation (binary)
    if 'disc_herniation' in per_class_metrics:
        metrics = per_class_metrics['disc_herniation']
        print(f"\nDisc Herniation (2 classes):")
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        class_names = ['No', 'Yes']
        for i, name in enumerate(class_names):
            print(f"  {name:<15} {metrics['precision'][i]:<12.3f} "
                  f"{metrics['recall'][i]:<12.3f} {metrics['f1'][i]:<12.3f} "
                  f"{int(metrics['support'][i]):<10}")

    print("="*70)


def main():
    args = parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints_spider')
    checkpoint_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("SPIDER Dataset Transfer Learning")
    print("="*70)
    print(f"Model: {args.model.upper()}")
    print(f"Modality: {args.modality}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Transfer from RSNA: {args.rsna_checkpoint if args.rsna_checkpoint else 'No (random init)'}")
    print("="*70)

    # Load dataset
    print("\n[1/6] Loading SPIDER dataset...")
    full_dataset = SPIDERDataset(data_dir=args.data_dir, split='training', modality=args.modality)

    # Split train/val
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=args.val_split,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")

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

    # Create model
    print("\n[2/6] Creating model...")
    if args.model == 'baseline':
        model = GradingModelSPIDERBaseline()
    else:  # cbam
        model = GradingModelSPIDERCBAM()

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")

    # Load pretrained RSNA weights
    if args.rsna_checkpoint:
        print(f"\n[3/6] Loading pretrained RSNA weights...")
        model.load_pretrained_rsna_backbone(args.rsna_checkpoint, verbose=True)
    else:
        print(f"\n[3/6] No pretrained weights (training from scratch)")

    # Freeze backbone if requested
    if args.freeze_backbone:
        print(f"\n[4/6] Freezing backbone (linear probing)...")
        model.freeze_backbone(freeze=True)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Trainable parameters: {trainable_params:,}")
    else:
        print(f"\n[4/6] Training entire model (fine-tuning)...")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Trainable parameters: {trainable_params:,}")

    # Create loss functions
    print(f"\n[5/6] Setting up loss functions...")
    if args.loss == 'weighted':
        print("  Computing class weights...")
        weights = compute_class_weights_spider(train_dataset)
        criteria = {
            'pfirrmann': nn.CrossEntropyLoss(weight=weights['pfirrmann'].to(device)),
            'spondylolisthesis': nn.CrossEntropyLoss(weight=weights['spondylolisthesis'].to(device)),
            'disc_herniation': nn.CrossEntropyLoss(weight=weights['disc_herniation'].to(device))
        }
        print(f"  - Pfirrmann weights: {weights['pfirrmann'].numpy()}")
        print(f"  - Spondylolisthesis weights: {weights['spondylolisthesis'].numpy()}")
        print(f"  - Disc herniation weights: {weights['disc_herniation'].numpy()}")
    else:
        criteria = {
            'pfirrmann': nn.CrossEntropyLoss(),
            'spondylolisthesis': nn.CrossEntropyLoss(),
            'disc_herniation': nn.CrossEntropyLoss()
        }
        print(f"  - Using standard CrossEntropyLoss")

    # Create optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Training loop
    print(f"\n[6/6] Starting training...")
    print("="*70)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  - Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for volumes, labels in train_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            # Move labels to device
            labels_device = {
                'pfirrmann': labels['pfirrmann'].to(device),
                'spondylolisthesis': labels['spondylolisthesis'].to(device),
                'disc_herniation': labels['disc_herniation'].to(device)
            }

            # Forward pass
            outputs = model(volumes)

            # Compute per-task losses
            loss_pfirrmann = criteria['pfirrmann'](outputs['pfirrmann'], labels_device['pfirrmann'])
            loss_spondy = criteria['spondylolisthesis'](outputs['spondylolisthesis'], labels_device['spondylolisthesis'])
            loss_herniation = criteria['disc_herniation'](outputs['disc_herniation'], labels_device['disc_herniation'])

            # Total loss (simple average)
            loss = (loss_pfirrmann + loss_spondy + loss_herniation) / 3.0

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_loss, val_accuracies, per_class_metrics = evaluate(model, val_loader, criteria, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Compute mean accuracy
        mean_acc = np.mean(list(val_accuracies.values()))

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracies:")
        print(f"    - Pfirrmann: {val_accuracies['pfirrmann']*100:.2f}%")
        print(f"    - Spondylolisthesis: {val_accuracies['spondylolisthesis']*100:.2f}%")
        print(f"    - Disc herniation: {val_accuracies['disc_herniation']*100:.2f}%")
        print(f"    - Mean: {mean_acc*100:.2f}%")

        # Print per-class metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print_per_class_metrics(per_class_metrics)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = checkpoint_dir / f'best_model_{args.model}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")
        else:
            epochs_no_improve += 1

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_{args.model}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"\n Early stopping triggered after {epoch+1} epochs (patience={args.early_stop_patience})")
            break

        print("-"*70)

    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints_spider/best_model_{args.model}.pth")
    print("="*70)


if __name__ == "__main__":
    main()
