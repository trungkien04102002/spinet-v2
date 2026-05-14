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

    # Output dirs (override defaults to avoid clobbering v2 ckpts)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_spider',
                        help='Where to save best_model_<model>.pth (default: checkpoints_spider)')
    parser.add_argument('--metrics-dir', type=str, default='experiments/spider_phase4',
                        help='Where to save best_metrics_<model>.{json,txt} (default: experiments/spider_phase4)')

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


SPIDER_CONDITIONS = [
    'pfirrmann', 'modic', 'disc_narrowing', 'spondylolisthesis',
    'up_endplate', 'low_endplate', 'disc_herniation', 'disc_bulging',
]
SPIDER_NUM_CLASSES = {
    'pfirrmann': 5, 'modic': 4, 'disc_narrowing': 2, 'spondylolisthesis': 2,
    'up_endplate': 2, 'low_endplate': 2, 'disc_herniation': 2, 'disc_bulging': 2,
}


def compute_class_weights_spider(dataset):
    """
    Compute class weights for SPIDER dataset (sqrt-of-inverse-frequency variant).

    Uses _resolve_get_labels to skip full .mha volume loads at startup.

    Returns:
        dict mapping condition -> tensor of per-class weights, one entry per
        SPIDER_CONDITIONS condition.
    """
    from spinenet.augmentation import _resolve_get_labels

    counts = {c: torch.zeros(SPIDER_NUM_CLASSES[c]) for c in SPIDER_CONDITIONS}
    get_labels = _resolve_get_labels(dataset)

    for idx in range(len(dataset)):
        if get_labels is not None:
            labels = get_labels(idx)
        else:
            _, labels = dataset[idx]
        for c in SPIDER_CONDITIONS:
            counts[c][labels[c]] += 1

    weights = {}
    for c in SPIDER_CONDITIONS:
        # Inverse frequency, then normalize so weights sum to num_classes (mean=1)
        w = 1.0 / counts[c].clamp(min=1)
        w = w / w.sum() * SPIDER_NUM_CLASSES[c]
        weights[c] = w
    return weights


def evaluate(model, dataloader, criteria, device):
    """Evaluate model on validation set across all SPIDER_CONDITIONS."""
    model.eval()

    all_losses = {c: [] for c in SPIDER_CONDITIONS}
    all_preds  = {c: [] for c in SPIDER_CONDITIONS}
    all_labels = {c: [] for c in SPIDER_CONDITIONS}

    val_pbar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for volumes, labels in val_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]
            labels_device = {c: labels[c].to(device) for c in SPIDER_CONDITIONS}

            outputs = model(volumes)

            for c in SPIDER_CONDITIONS:
                loss = criteria[c](outputs[c], labels_device[c])
                all_losses[c].append(loss.item())
                preds = torch.argmax(outputs[c], dim=1)
                all_preds[c].extend(preds.cpu().numpy())
                all_labels[c].extend(labels[c].numpy())

    val_accuracies = {}
    per_class_metrics = {}
    for c in SPIDER_CONDITIONS:
        labels_np = np.array(all_labels[c])
        preds_np = np.array(all_preds[c])
        val_accuracies[c] = accuracy_score(labels_np, preds_np)

        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, preds_np,
            labels=list(range(SPIDER_NUM_CLASSES[c])),
            average=None,
            zero_division=0,
        )
        per_class_metrics[c] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'support': support
        }

    avg_loss = float(np.mean([np.mean(all_losses[c]) for c in SPIDER_CONDITIONS]))
    return avg_loss, val_accuracies, per_class_metrics


CLASS_NAMES = {
    'pfirrmann':         ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5'],
    'modic':             ['Type 0',  'Type 1',  'Type 2',  'Type 3'],
    'disc_narrowing':    ['No', 'Yes'],
    'spondylolisthesis': ['No', 'Yes'],
    'up_endplate':       ['No', 'Yes'],
    'low_endplate':      ['No', 'Yes'],
    'disc_herniation':   ['No', 'Yes'],
    'disc_bulging':      ['No', 'Yes'],
}
DISPLAY_NAMES = {
    'pfirrmann':         'Pfirrmann Grading',
    'modic':             'Modic',
    'disc_narrowing':    'Disc Narrowing',
    'spondylolisthesis': 'Spondylolisthesis',
    'up_endplate':       'UP Endplate',
    'low_endplate':      'LOW Endplate',
    'disc_herniation':   'Disc Herniation',
    'disc_bulging':      'Disc Bulging',
}


def print_per_class_metrics(per_class_metrics):
    """Print per-class metrics for all SPIDER conditions."""
    print("\n" + "="*70)
    print("Per-Class Metrics (Precision, Recall, F1-Score)")
    print("="*70)

    for c in SPIDER_CONDITIONS:
        if c not in per_class_metrics:
            continue
        m = per_class_metrics[c]
        n = SPIDER_NUM_CLASSES[c]
        print(f"\n{DISPLAY_NAMES[c]} ({n} classes):")
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for i, name in enumerate(CLASS_NAMES[c]):
            print(f"  {name:<15} {m['precision'][i]:<12.3f} "
                  f"{m['recall'][i]:<12.3f} {m['f1'][i]:<12.3f} "
                  f"{int(m['support'][i]):<10}")
    print("="*70)


def main():
    args = parse_args()

    # Reproducibility — must come BEFORE any DataLoader / model construction
    set_seed(args.seed)

    # Tag output filenames by seed so multi-seed runs don't overwrite.
    # Default seed=42 → no tag (backward compat with v3 results).
    run_tag = "" if args.seed == 42 else f"_seed{args.seed}"

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    # Folders already exist in the repo (.gitkeep). mkdir is a no-op safety net
    # for the rare case the user deletes them.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history = []  # per-epoch rows for training_log csv

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
        random_state=args.seed
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
            c: nn.CrossEntropyLoss(weight=weights[c].to(device))
            for c in SPIDER_CONDITIONS
        }
        for c in SPIDER_CONDITIONS:
            print(f"  - {DISPLAY_NAMES[c]} weights: {weights[c].numpy()}")
    else:
        criteria = {c: nn.CrossEntropyLoss() for c in SPIDER_CONDITIONS}
        print(f"  - Using standard CrossEntropyLoss")

    # Create optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # PyTorch 2.x removed `verbose`. Pass only the params it still accepts.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
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
            labels_device = {c: labels[c].to(device) for c in SPIDER_CONDITIONS}

            outputs = model(volumes)

            # Per-task loss + simple average across conditions
            losses = [criteria[c](outputs[c], labels_device[c]) for c in SPIDER_CONDITIONS]
            loss = sum(losses) / len(losses)

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

        # Per-condition F1 macro (mean of per-class F1) — more informative than
        # accuracy on imbalanced classes (e.g. Spondylolisthesis 97% prevalence).
        f1_macro = {c: float(np.mean(per_class_metrics[c]['f1'])) for c in SPIDER_CONDITIONS}
        mean_f1 = float(np.mean(list(f1_macro.values())))

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val per-condition (accuracy | F1 macro):")
        for c in SPIDER_CONDITIONS:
            print(f"    - {DISPLAY_NAMES[c]:<20s}  acc={val_accuracies[c]*100:6.2f}%  F1={f1_macro[c]:.3f}")
        print(f"    - {'Mean':<20s}  acc={mean_acc*100:6.2f}%  F1={mean_f1:.3f}")

        # Print per-class metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print_per_class_metrics(per_class_metrics)

        # Append to per-epoch training log
        row = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss),
            'val_mean_acc': float(mean_acc),
            'val_mean_f1_macro': mean_f1,
        }
        for c in SPIDER_CONDITIONS:
            row[f'val_acc_{c}'] = float(val_accuracies[c])
            row[f'val_f1_macro_{c}'] = f1_macro[c]
            for i, f1 in enumerate(per_class_metrics[c]['f1']):
                row[f'f1_{c}_class{i}'] = float(f1)
            for i, p in enumerate(per_class_metrics[c]['precision']):
                row[f'precision_{c}_class{i}'] = float(p)
            for i, r in enumerate(per_class_metrics[c]['recall']):
                row[f'recall_{c}_class{i}'] = float(r)
        history.append(row)

        # Persist running CSV every epoch (so partial runs are recoverable)
        import csv as _csv
        log_path = metrics_dir / f'training_log_{args.model}{run_tag}.csv'
        with open(log_path, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = checkpoint_dir / f'best_model_{args.model}{run_tag}.pth'
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

            # Persist best metrics txt + json (human + machine readable)
            from datetime import datetime as _dt
            best_txt = metrics_dir / f'best_metrics_{args.model}{run_tag}.txt'
            with open(best_txt, 'w') as f:
                f.write(f"=== BEST {args.model.upper()} SPIDER MODEL ===\n")
                f.write(f"Saved at: {_dt.now().isoformat(timespec='seconds')}\n")
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Train loss: {avg_train_loss:.4f}\n")
                f.write(f"Val loss:   {val_loss:.4f}\n")
                f.write(f"Val mean acc:      {mean_acc*100:.2f}%\n")
                f.write(f"Val mean F1 macro: {mean_f1:.3f}\n\n")
                for c in SPIDER_CONDITIONS:
                    f.write(f"{DISPLAY_NAMES[c]:<20s}  acc={val_accuracies[c]*100:6.2f}%  "
                            f"F1_macro={f1_macro[c]:.3f}  "
                            f"per-class F1={[round(float(x), 3) for x in per_class_metrics[c]['f1']]}\n")
                    f.write(f"{'':>20s}  precision={[round(float(x), 3) for x in per_class_metrics[c]['precision']]}  "
                            f"recall={[round(float(x), 3) for x in per_class_metrics[c]['recall']]}\n")

            best_json = metrics_dir / f'best_metrics_{args.model}{run_tag}.json'
            import json as _json
            with open(best_json, 'w') as f:
                _json.dump({
                    'epoch': epoch + 1,
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(val_loss),
                    'val_mean_acc': float(mean_acc),
                    'val_mean_f1_macro': mean_f1,
                    'val_accuracies': {c: float(val_accuracies[c]) for c in SPIDER_CONDITIONS},
                    'val_f1_macro': f1_macro,
                    'per_class_f1':       {c: [float(x) for x in per_class_metrics[c]['f1']]        for c in SPIDER_CONDITIONS},
                    'per_class_precision':{c: [float(x) for x in per_class_metrics[c]['precision']] for c in SPIDER_CONDITIONS},
                    'per_class_recall':   {c: [float(x) for x in per_class_metrics[c]['recall']]    for c in SPIDER_CONDITIONS},
                    'args': vars(args),
                }, f, indent=2)
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
    print(f"Model saved to: {checkpoint_dir}/best_model_{args.model}{run_tag}.pth")
    print("="*70)


if __name__ == "__main__":
    main()
