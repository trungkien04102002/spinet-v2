"""
Test SpineNetV2 on SPIDER Dataset.

Evaluates trained baseline or CBAM model on SPIDER validation set.
Computes per-class metrics for all 3 conditions.

Usage:
    # Test baseline model
    python test_spider.py --model baseline \\
        --checkpoint checkpoints_spider/best_model_baseline.pth

    # Test CBAM model
    python test_spider.py --model cbam \\
        --checkpoint checkpoints_spider/best_model_cbam.pth

Author: SpineNetV2 Transfer Learning
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# SpineNetV2 imports
from spinenet.models.grading_spider import GradingModelSPIDERBaseline, GradingModelSPIDERCBAM
from spider_dataloader import SPIDERDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test SPIDER Model')

    # Model
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'cbam'],
                        help='Model type: baseline or cbam')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')

    # Data
    parser.add_argument('--data-dir', type=str, default='spider',
                        help='Path to SPIDER dataset directory')
    parser.add_argument('--modality', type=str, default='t2', choices=['t1', 't2'],
                        help='MRI modality (default: t2)')
    parser.add_argument('--split', type=str, default='validation', choices=['training', 'validation'],
                        help='Dataset split to test on (default: validation)')

    # System
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    return parser.parse_args()


def print_confusion_matrix(y_true, y_pred, class_names, title):
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{title}:")
    print(f"  {'':>12}", end='')
    for name in class_names:
        print(f"  {name:>10}", end='')
    print()

    for i, name in enumerate(class_names):
        print(f"  {name:>12}", end='')
        for j in range(len(class_names)):
            print(f"  {cm[i, j]:>10}", end='')
        print()


def main():
    args = parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("SPIDER Dataset Testing")
    print("="*70)
    print(f"Model: {args.model.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print("="*70)

    # Load dataset
    print("\n[1/4] Loading SPIDER dataset...")
    dataset = SPIDERDataset(data_dir=args.data_dir, split=args.split, modality=args.modality)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"  - Samples: {len(dataset)}")

    # Create model
    print("\n[2/4] Creating model...")
    if args.model == 'baseline':
        model = GradingModelSPIDERBaseline()
    else:  # cbam
        model = GradingModelSPIDERCBAM()

    # Load checkpoint
    print(f"\n[3/4] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if 'epoch' in checkpoint:
        print(f"  - Trained for {checkpoint['epoch']+1} epochs")
    if 'val_loss' in checkpoint:
        print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")

    # Test
    print("\n[4/4] Testing model...")
    print("="*70)

    all_preds = {'pfirrmann': [], 'spondylolisthesis': [], 'disc_herniation': []}
    all_labels = {'pfirrmann': [], 'spondylolisthesis': [], 'disc_herniation': []}
    all_losses = {'pfirrmann': [], 'spondylolisthesis': [], 'disc_herniation': []}

    criteria = {
        'pfirrmann': nn.CrossEntropyLoss(),
        'spondylolisthesis': nn.CrossEntropyLoss(),
        'disc_herniation': nn.CrossEntropyLoss()
    }

    test_pbar = tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for volumes, labels in test_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            # Move labels to device
            labels_device = {
                'pfirrmann': labels['pfirrmann'].to(device),
                'spondylolisthesis': labels['spondylolisthesis'].to(device),
                'disc_herniation': labels['disc_herniation'].to(device)
            }

            # Forward pass
            outputs = model(volumes)

            # Compute losses and predictions
            for condition in ['pfirrmann', 'spondylolisthesis', 'disc_herniation']:
                loss = criteria[condition](outputs[condition], labels_device[condition])
                all_losses[condition].append(loss.item())

                preds = torch.argmax(outputs[condition], dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())

    # Compute metrics
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)

    # Overall metrics
    print("\nOverall Accuracy:")
    accuracies = {}
    for condition in ['pfirrmann', 'spondylolisthesis', 'disc_herniation']:
        labels_np = np.array(all_labels[condition])
        preds_np = np.array(all_preds[condition])
        acc = accuracy_score(labels_np, preds_np)
        accuracies[condition] = acc
        print(f"  - {condition.replace('_', ' ').title()}: {acc*100:.2f}%")

    mean_acc = np.mean(list(accuracies.values()))
    print(f"  - Mean Accuracy: {mean_acc*100:.2f}%")

    # Per-class metrics
    print("\n" + "="*70)
    print("Per-Class Metrics (Precision, Recall, F1-Score)")
    print("="*70)

    # Pfirrmann (5 classes)
    print(f"\nPfirrmann Grading (5 classes):")
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    labels_np = np.array(all_labels['pfirrmann'])
    preds_np = np.array(all_preds['pfirrmann'])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, preds_np,
        labels=[0, 1, 2, 3, 4],
        average=None,
        zero_division=0
    )

    class_names = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    for i, name in enumerate(class_names):
        print(f"  {name:<15} {precision[i]:<12.3f} {recall[i]:<12.3f} "
              f"{f1[i]:<12.3f} {int(support[i]):<10}")

    # Confusion matrix for Pfirrmann
    print_confusion_matrix(labels_np, preds_np, ['G1', 'G2', 'G3', 'G4', 'G5'],
                          "Confusion Matrix")

    # Spondylolisthesis (binary)
    print(f"\nSpondylolisthesis (2 classes):")
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    labels_np = np.array(all_labels['spondylolisthesis'])
    preds_np = np.array(all_preds['spondylolisthesis'])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, preds_np,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    class_names = ['No', 'Yes']
    for i, name in enumerate(class_names):
        print(f"  {name:<15} {precision[i]:<12.3f} {recall[i]:<12.3f} "
              f"{f1[i]:<12.3f} {int(support[i]):<10}")

    print_confusion_matrix(labels_np, preds_np, ['No', 'Yes'], "Confusion Matrix")

    # Disc herniation (binary)
    print(f"\nDisc Herniation (2 classes):")
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    labels_np = np.array(all_labels['disc_herniation'])
    preds_np = np.array(all_preds['disc_herniation'])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, preds_np,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    class_names = ['No', 'Yes']
    for i, name in enumerate(class_names):
        print(f"  {name:<15} {precision[i]:<12.3f} {recall[i]:<12.3f} "
              f"{f1[i]:<12.3f} {int(support[i]):<10}")

    print_confusion_matrix(labels_np, preds_np, ['No', 'Yes'], "Confusion Matrix")

    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)


if __name__ == "__main__":
    main()
