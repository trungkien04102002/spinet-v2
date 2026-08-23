"""
Train Hybrid CBAM + BiomedCLIP model on RSNA 2024 dataset.

Architecture:
    - CBAM 3D ResNet34 backbone (frozen, loaded from best_model_attention.pth)
    - BiomedCLIP image encoder 2D (frozen, slice selection + attention pool)
    - BiomedCLIP text encoder (frozen)
    - Image projection MLP (TRAINABLE, ~500K params)
    - Logit scale (TRAINABLE)
    - Slice attention pool (TRAINABLE)

Loss: FocalLoss (per task) + alpha * SupConLoss + Uncertainty multi-task weighting.
Total trainable: ~500K params (~0.3% of 218M total).

Replaces 3 hardcoded heads with cosine similarity vs BiomedCLIP text embeddings,
enabling zero-shot label extension. CBAM contribution preserved (Severe recall),
BiomedCLIP contribution added (label space extension).

Usage:
    python3 train_rsna_hybrid.py \\
        --cbam-checkpoint checkpoints/rsna/best_model_attention.pth \\
        --epochs 30 --batch-size 16 --lr 1e-4

Author: SpineNetV2 Improved Implementation
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# SpineNetV2 imports
from spinenet.models.grading_hybrid import SpineNetHybrid
from spinenet.losses import FocalLoss, UncertaintyLoss, compute_class_weights
from spinenet.augmentation import get_training_augmentation, OversamplingDataset
from spinenet.metrics_logger import MetricsLogger
from spinenet.auc_metrics import (
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)
from rsna_preprocessed_dataloader import RSNAPreprocessedDataset


# Label prompts for RSNA 9 classes (3 conditions × 3 severities).
# These are encoded once via BiomedCLIP text encoder (frozen) and stored as
# the label embedding database. Adding new labels = encode new prompt + append.
RSNA_PROMPTS = {
    'spinal_canal': {
        0: "normal or mild spinal canal stenosis",
        1: "moderate spinal canal stenosis",
        2: "severe spinal canal stenosis",
    },
    'left_foraminal': {
        0: "normal or mild left neural foraminal narrowing",
        1: "moderate left neural foraminal narrowing",
        2: "severe left neural foraminal narrowing",
    },
    'right_foraminal': {
        0: "normal or mild right neural foraminal narrowing",
        1: "moderate right neural foraminal narrowing",
        2: "severe right neural foraminal narrowing",
    },
}

ALL_CONDITIONS = tuple(RSNA_PROMPTS.keys())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RSNA Hybrid CBAM + BiomedCLIP Model')

    # Data
    parser.add_argument('--data-dir', type=str, default='rsna_preprocessed',
                        help='Path to preprocessed data directory')
    parser.add_argument('--split', type=str, default='train',
                        help="Metadata split to load: reads <split>_metadata.csv. "
                             "Use 't1' with --data-dir rsna_preprocessed_t1 for the "
                             "per-side Sagittal T1 foraminal crops.")
    parser.add_argument('--conditions', type=str, nargs='+', default=list(ALL_CONDITIONS),
                        choices=list(ALL_CONDITIONS),
                        help="Which conditions to train and evaluate. Defaults to all "
                             "three. Radiologists grade foraminal narrowing on Sagittal "
                             "T1 and canal stenosis on Sagittal T2 -- the coordinate file "
                             "puts them on the same series once in 6291 cases -- so a T1 "
                             "run should pass only the two foraminal conditions. The "
                             "cosine head and its frozen text anchors are unchanged; "
                             "unselected conditions are simply not scored.")
    parser.add_argument('--cbam-checkpoint', type=str, required=True,
                        help='Path to trained CBAM checkpoint (best_model_attention.pth)')
    parser.add_argument('--save-dir', type=str, default='checkpoints/hybrid',
                        help='Directory to save Hybrid checkpoints')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--select-by', type=str, default='severe_f1',
                        choices=['severe_f1', 'severe_auprc', 'macro_auprc',
                                 'weighted_logloss'],
                        help="Metric that decides which epoch is saved as best. "
                             "Default severe_f1 reproduces published runs, but it is "
                             "threshold-dependent and swings ~15x more between epochs "
                             "than severe_auprc on this validation set, so it tends to "
                             "save the top of an oscillation. severe_auprc is the "
                             "steadier choice; weighted_logloss is the RSNA 2024 "
                             "competition metric (lower is better).")
    parser.add_argument('--fast-dev', action='store_true',
                        help='Smoke test: tiny class-stratified subset, 1 epoch. '
                             'Exercises data loading, the text database, class '
                             'weights and one train+eval pass in about a minute.')
    parser.add_argument('--fast-dev-samples', type=int, default=24,
                        help='Training rows to keep under --fast-dev (default 24).')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (smaller than CBAM-only due to BiomedCLIP overhead)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower than CBAM since only projection is trained)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')

    # Model
    parser.add_argument('--slice-strategy', type=str, default='static',
                        choices=['static', 'dynamic'],
                        help='Slice selection: static (3 center) or dynamic (cosine sim filter)')

    # Loss
    parser.add_argument('--use-focal', action='store_true', default=True,
                        help='Use FocalLoss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--use-supcon', action='store_true', default=True,
                        help='Use SupervisedContrastive auxiliary loss')
    parser.add_argument('--no-supcon', action='store_false', dest='use_supcon',
                        help='Disable SupCon loss')
    parser.add_argument('--supcon-weight', type=float, default=0.1,
                        help='SupCon weight (default 0.1, low due to imbalance)')
    parser.add_argument('--use-uncertainty', type=lambda x: str(x).lower() == 'true',
                        default=True, help='Use UncertaintyLoss for task weighting')
    parser.add_argument('--class-weight-mode', type=str, default='none',
                        choices=['none', 'sqrt', 'inverse', 'effective'],
                        help='Class weight mode for FocalLoss alpha. '
                             'none=no weights (default). sqrt=mild boost (~4.4x Severe). '
                             'effective=class-balanced (Cui 2019, ~7.7x). '
                             'inverse=strong boost (~19x).')

    # Augmentation
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['none', 'light', 'medium', 'heavy'],
                        help='Augmentation strength')
    # NOTE: this used to default to True and its help text had the two modes
    # backwards. RandomHorizontalFlip flips dims=[-1], which on a (9,112,224)
    # sagittal crop is the ANTERIOR-POSTERIOR axis, not left-right (laterality
    # is the slice axis). Swapping left_*/right_* on that flip therefore
    # corrupts the labels rather than correcting them. Default is now False,
    # which is what every published run passed explicitly via
    # --no-hflip-swap-labels, so defaults now reproduce those runs.
    parser.add_argument('--hflip-swap-labels', dest='hflip_swap_labels',
                        action='store_true', default=False,
                        help='DEPRECATED and label-corrupting: swap left_*/right_* '
                             'on the anterior-posterior flip. Only for reproducing '
                             'a run that was configured this way.')
    parser.add_argument('--no-hflip-swap-labels', dest='hflip_swap_labels',
                        action='store_false',
                        help='Do not swap labels on the AP flip (default, correct).')
    parser.add_argument('--no-ap-flip', dest='ap_flip', action='store_false',
                        default=True,
                        help='Drop the anterior-posterior flip entirely. It mirrors '
                             'the spine front-to-back, placing the canal anterior to '
                             'the disc, which no anatomy produces. On by default '
                             'because every published run used it.')
    parser.add_argument('--slice-reverse', dest='slice_reverse',
                        action='store_true', default=False,
                        help='Add the correct laterality augmentation: reverse the '
                             'sagittal slice order AND swap left_*/right_* labels. '
                             'Off by default; never used in a published run.')
    parser.add_argument('--oversample-factor', type=int, default=5,
                        help='Oversampling factor for minority classes')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='Early stopping patience')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to Hybrid checkpoint to resume from')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split + torch + numpy + cudnn')

    # Component ablation (for paper Table V)
    parser.add_argument('--ablate-branch', type=str, default='none',
                        choices=['none', 'cbam_only', 'biomedclip_only'],
                        help='Zero out one branch for component ablation. '
                             '"cbam_only" = drop BiomedCLIP image features; '
                             '"biomedclip_only" = drop CBAM 3D features.')

    # Fusion architecture (advisor: "đổi MLP, đổi concat cho đúng đắn")
    parser.add_argument('--fusion', type=str, default='concat_mlp',
                        choices=['concat_mlp', 'gated'],
                        help='Image fusion head. concat_mlp (default) = '
                             'concat -> 2-layer MLP (original). gated = '
                             'GMU (Arevalo 2017) per-dim learned gate. Use to '
                             'test whether a smarter fusion improves over the '
                             'fixed-mixing MLP.')

    # Modality dropout (Q10 robustness ablation)
    parser.add_argument('--modality-dropout', type=float, default=0.0,
                        help='Probability of zero-ing one branch (50/50 which) '
                             'each training step. Trains the model to handle '
                             'missing-modality at inference. Recommended 0.15 '
                             'when enabled. Default 0.0 = disabled.')

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


def supervised_contrastive_loss(image_embs, labels, temperature=0.07):
    """
    Supervised contrastive loss (Khosla et al. 2020).
    Pulls together same-label samples, pushes apart different-label.

    Args:
        image_embs: [B, D] L2-normalized
        labels: [B] integer labels (-1 means ignore)
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=image_embs.device, requires_grad=True)

    embs = image_embs[valid_mask]
    lbls = labels[valid_mask]

    sim = embs @ embs.T / temperature
    label_eq = lbls.unsqueeze(0) == lbls.unsqueeze(1)
    self_mask = torch.eye(len(lbls), dtype=torch.bool, device=embs.device)
    pos_mask = label_eq & ~self_mask

    if pos_mask.sum(dim=1).min() == 0:
        return torch.tensor(0.0, device=image_embs.device, requires_grad=True)

    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1).float() + 1e-12)
    return -mean_log_prob_pos.mean()


def _stratified_head(metadata, indices, n_total, conditions):
    """Pick a small subset of `indices` that still covers all three severity
    classes (for --fast-dev).

    A plain ``indices[:n]`` slice usually lands on Normal-only rows, because
    Severe is under 5% of the data. compute_class_weights would then divide by a
    zero count -- ``1/0`` -> inf -> NaN weights after normalisation -- and the
    smoke test would "pass" without ever exercising the class-weight path it
    exists to check.

    On the per-side T1 crops each row is labelled on exactly one side and -1 on
    the other, so max() over the active conditions recovers that row's class.
    """
    n_per_class = max(2, n_total // 3)
    sub = metadata.loc[indices]
    row_class = sub[list(conditions)].max(axis=1)
    picked = []
    for cls in (0, 1, 2):
        picked.extend(sub.index[row_class == cls][:n_per_class].tolist())
    return picked or list(indices[:n_total])


def build_text_database(model, device, conditions=None):
    """
    Pre-compute text embeddings for the selected conditions' labels.
    Frozen BiomedCLIP text encoder is called once - reused every batch.

    Args:
        conditions: iterable of condition names, or None for all of them. The
            returned dict's key order defines the task order everywhere else,
            so it is kept in ALL_CONDITIONS order regardless of CLI order.

    Returns:
        {condition: [3, 512] tensor of L2-normalized text embeddings on device}
    """
    selected = set(conditions) if conditions is not None else set(ALL_CONDITIONS)
    db = {}
    for condition in ALL_CONDITIONS:
        if condition not in selected:
            continue
        prompts = RSNA_PROMPTS[condition]
        texts = [prompts[i] for i in sorted(prompts.keys())]
        embs = model.encode_text(texts).to(device)
        db[condition] = embs
    return db


def evaluate(model, dataloader, text_db, criterion, uncertainty_loss, device):
    """Evaluate model on validation set."""
    model.eval()

    # text_db carries exactly the conditions selected by --conditions, so every
    # loop below is driven by it rather than by a hardcoded list of three.
    conditions = list(text_db.keys())
    all_losses = {c: [] for c in conditions}
    all_preds = {c: [] for c in conditions}
    all_labels = {c: [] for c in conditions}
    all_probs = {c: [] for c in conditions}
    per_class_metrics = {}

    val_pbar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for volumes, labels in val_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            # Move labels to device
            labels_device = {c: labels[c].to(device) for c in conditions}

            # Forward pass: image embedding via Hybrid encoder
            image_emb = model.encode_image(volumes)

            # Compute per-task losses + predictions via cosine similarity
            for condition in conditions:
                text_embs = text_db[condition]
                logits = model.logit_scale.exp().clamp(max=100.0) * (image_emb @ text_embs.T)

                # On the per-side T1 crops a whole batch can be -1 for one side,
                # and CrossEntropyLoss(ignore_index=-1) returns NaN when it has
                # nothing to average over. Skipping keeps the epoch mean finite.
                if (labels_device[condition] != -1).any():
                    loss = criterion(logits, labels_device[condition])
                    all_losses[condition].append(loss.item())

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())
                all_probs[condition].append(probs)

    # Compute metrics per condition
    val_accuracies = {}
    val_weighted_logloss = 0.0

    probs_dict = {c: np.concatenate(all_probs[c], axis=0) for c in all_probs}
    labels_dict = {c: np.array(all_labels[c]) for c in all_labels}

    for condition in conditions:
        labels_np = labels_dict[condition]
        preds_np = np.array(all_preds[condition])

        valid_mask = labels_np != -1
        labels_filtered = labels_np[valid_mask]
        preds_filtered = preds_np[valid_mask]

        if len(labels_filtered) > 0:
            acc = accuracy_score(labels_filtered, preds_filtered)
            val_accuracies[condition] = acc
        else:
            val_accuracies[condition] = 0.0

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

    auc_auprc_metrics = compute_auc_auprc_per_condition(probs_dict, labels_dict)
    auc_auprc_overall = aggregate_overall_auprc(auc_auprc_metrics)

    # Only average conditions that actually recorded a loss: on the per-side T1
    # crops a condition can legitimately have zero labelled batches.
    scored = [np.mean(all_losses[c]) for c in conditions if all_losses[c]]
    avg_loss = float(np.mean(scored)) if scored else float('nan')

    return (avg_loss, val_weighted_logloss, val_accuracies, per_class_metrics,
            auc_auprc_metrics, auc_auprc_overall)


def print_per_class_metrics(per_class_metrics):
    """Print precision, recall, F1-score per class for each condition."""
    # Driven by what was actually scored, so a --conditions subset prints only
    # the conditions it trained on. `args` is deliberately not referenced here.
    display = {'spinal_canal': 'Spinal Canal',
               'left_foraminal': 'Left Foraminal',
               'right_foraminal': 'Right Foraminal'}
    conditions = [c for c in ALL_CONDITIONS if c in per_class_metrics]
    condition_names = [display[c] for c in conditions]
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
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        print(f"{'Macro Avg':<16} {'':<9} {macro_precision:<10.3f} "
              f"{macro_recall:<9.3f} {macro_f1:<9.3f}")


def main():
    args = parse_args()

    # Canonical task order for the whole run: ALL_CONDITIONS order, filtered by
    # --conditions. Everything downstream (text database, loss aggregation,
    # UncertaintyLoss task slots, metrics) keys off this, so a T1 run that scores
    # only the two foraminal heads stays internally consistent.
    conditions = [c for c in ALL_CONDITIONS if c in set(args.conditions)]

    # Reproducibility — must be set before any RNG-using import path
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SpineNetV2 Hybrid CBAM + BiomedCLIP Training for RSNA 2024")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"CBAM checkpoint: {args.cbam_checkpoint}")
    print(f"Slice strategy: {args.slice_strategy}")
    print(f"FocalLoss: {args.use_focal} (gamma={args.focal_gamma})")
    print(f"SupCon: {args.use_supcon} (weight={args.supcon_weight})")
    print(f"UncertaintyLoss: {args.use_uncertainty}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Oversampling: {args.oversample_factor}x for Moderate/Severe")
    print(f"Class weight mode: {args.class_weight_mode}")
    print("="*70)

    # [1/7] Load datasets
    print(f"\n[1/7] Loading datasets...")

    full_dataset = RSNAPreprocessedDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=None
    )
    print(f"  ✓ Loaded {len(full_dataset)} samples")

    # Split by patient (no data leakage!)
    print(f"\n[2/7] Splitting dataset by patient...")
    unique_patients = full_dataset.metadata['study_id'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=args.val_split,
        random_state=args.seed,
    )

    train_indices = full_dataset.metadata[full_dataset.metadata['study_id'].isin(train_patients)].index.tolist()
    val_indices = full_dataset.metadata[full_dataset.metadata['study_id'].isin(val_patients)].index.tolist()

    # RandomSliceReverse mirrors the patient left-to-right and swaps the L/R
    # labels together. That pairing is only valid when the crop actually spans
    # the midline, as the Sagittal T2 crop does (it is centred on the canal).
    #
    # The per-side Sagittal T1 crops are centred on ONE foramen, so reversing
    # their slice order still shows that same foramen -- swapping the label
    # would assert the opposite side and inject exactly the kind of label noise
    # that cost 40% relative Severe F1 in the v3 regression (0.333 -> 0.200,
    # experiments/v3_20260503/RESULTS_LOG.md).
    #
    # Detect per-side data rather than trusting the caller: on per-side crops
    # every row has exactly one of left/right labelled and the other at -1.
    if args.slice_reverse:
        lr = full_dataset.metadata[['left_foraminal', 'right_foraminal']]
        one_side_only = ((lr != -1).sum(axis=1) == 1).mean()
        if one_side_only > 0.9:
            raise SystemExit(
                "--slice-reverse is unsafe on this dataset.\n"
                f"  {one_side_only:.1%} of rows are labelled on exactly one side, "
                "so these are PER-SIDE crops.\n"
                "  Reversing their slice order does not move the foramen to the "
                "other side, so swapping\n"
                "  left/right labels would corrupt them. Use --slice-reverse only "
                "with the midline\n"
                "  Sagittal T2 crops (--data-dir rsna_preprocessed --split train)."
            )

    if args.fast_dev:
        train_indices = _stratified_head(full_dataset.metadata, train_indices,
                                         args.fast_dev_samples, conditions)
        val_indices = _stratified_head(full_dataset.metadata, val_indices,
                                       max(6, args.fast_dev_samples // 2), conditions)
        args.epochs = min(args.epochs, 1)
        print(f"  ⚡ --fast-dev: {len(train_indices)} train / {len(val_indices)} val, 1 epoch")

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
        geo = []
        geo.append("AP-flip on" if args.ap_flip else "AP-flip OFF")
        if args.slice_reverse:
            geo.append("slice-reverse + L/R swap ON")
        if args.hflip_swap_labels:
            geo.append("!! AP-flip swaps L/R labels (corrupting)")
        print(f"  ✓ Augmentation: {args.augmentation} ({', '.join(geo)})")
        augmented_full_dataset = RSNAPreprocessedDataset(
            data_dir=args.data_dir,
            split=args.split,
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

    # [3/7] Create Hybrid model
    print(f"\n[3/7] Creating Hybrid model...")
    print(f"  Loading frozen CBAM backbone from {args.cbam_checkpoint}")
    print(f"  Loading frozen BiomedCLIP from HuggingFace...")

    model = SpineNetHybrid(
        cbam_checkpoint_path=args.cbam_checkpoint,
        biomedclip_device=str(device),
        slice_strategy=args.slice_strategy,
        ablate_branch=args.ablate_branch,
        fusion_mode=args.fusion,
        modality_dropout_p=args.modality_dropout,
    ).to(device)
    if args.ablate_branch != 'none':
        print(f"  Ablation mode: {args.ablate_branch} "
              f"(other branch zeroed before fusion)")
    if args.fusion != 'concat_mlp':
        print(f"  Fusion head: {args.fusion}")
    if args.modality_dropout > 0:
        print(f"  Modality dropout: p={args.modality_dropout} (training only)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created: {trainable_params:,} / {total_params:,} trainable params "
          f"({100*trainable_params/total_params:.2f}%)")

    # [4/7] Pre-compute text embeddings (frozen, computed once)
    print(f"\n[4/7] Pre-computing label text embeddings (frozen BiomedCLIP)...")
    text_db = build_text_database(model, device, conditions)
    for cond, embs in text_db.items():
        print(f"  ✓ {cond}: {embs.shape}")

    # [5/7] Setup loss functions
    print(f"\n[5/7] Setting up loss functions...")

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
        uncertainty_loss = UncertaintyLoss(num_tasks=len(conditions)).to(device)
        print(f"  ✓ UncertaintyLoss for multi-task weighting")
    else:
        uncertainty_loss = None

    if args.use_supcon:
        print(f"  ✓ Supervised contrastive auxiliary loss (weight={args.supcon_weight})")

    # [6/7] Setup optimizer and scheduler
    print(f"\n[6/7] Setting up optimizer and scheduler...")

    train_param_list = [p for p in model.parameters() if p.requires_grad]
    if args.use_uncertainty:
        optimizer = AdamW([
            {'params': train_param_list, 'lr': args.lr},
            {'params': uncertainty_loss.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(train_param_list, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"✓ Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"✓ Loss: {'FocalLoss' if args.use_focal else 'CrossEntropyLoss'}"
          f"{' + SupCon' if args.use_supcon else ''}")
    print(f"✓ Scheduler: CosineAnnealingLR (T_max={args.epochs})")

    # Resume from checkpoint if provided
    start_epoch = 0
    best_severe_f1 = -1.0
    best_score = -float('inf')

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # Only load trainable weights (frozen modules already loaded fresh)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_severe_f1 = checkpoint.get('best_severe_f1', -1.0)
        print(f"  ✓ Resumed from epoch {start_epoch}")

    # [7/7] Training loop
    print(f"\n[7/7] Training for {args.epochs} epochs...")
    print("="*70)

    best_epoch = 0
    epochs_without_improvement = 0

    # Total wall-clock from first training step (for paper Table: train time)
    total_train_start = time.time()

    # Tag output filenames by seed + ablation so multi-seed / ablation runs
    # don't overwrite each other. Default seed=42, ablate=none → tag="" (backward compat).
    _tag_parts = []
    if args.seed != 42:
        _tag_parts.append(f"seed{args.seed}")
    if args.ablate_branch != 'none':
        _tag_parts.append(args.ablate_branch)
    if args.fusion != 'concat_mlp':
        _tag_parts.append(args.fusion)
    if args.modality_dropout > 0:
        _tag_parts.append(f"mdrop{args.modality_dropout:g}")
    run_tag = "_" + "_".join(_tag_parts) if _tag_parts else ""
    metrics_logger = MetricsLogger(save_dir=save_dir, prefix=f"hybrid{run_tag}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        # Keep frozen modules in eval mode (BatchNorm stats etc.)
        model.cbam.eval()
        model.biomedclip.eval()

        train_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

        for batch_idx, (volumes, labels) in enumerate(train_pbar):
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            labels_device = {c: labels[c].to(device) for c in conditions}

            # Forward pass
            optimizer.zero_grad()
            image_emb = model.encode_image(volumes)

            # Compute per-task losses (cosine similarity vs frozen text embeddings)
            task_losses = []
            for condition in conditions:
                # Each per-side T1 crop is labelled on exactly one side, so a
                # whole batch can be -1 for the other one. CrossEntropyLoss with
                # ignore_index=-1 returns NaN in that case, which would poison
                # the backward pass, so that task sits this batch out.
                if not (labels_device[condition] != -1).any():
                    continue

                text_embs = text_db[condition]
                logits = model.logit_scale.exp().clamp(max=100.0) * (image_emb @ text_embs.T)
                loss_cls = criterion(logits, labels_device[condition])

                if args.use_supcon:
                    loss_con = supervised_contrastive_loss(image_emb, labels_device[condition])
                    task_loss = loss_cls + args.supcon_weight * loss_con
                else:
                    task_loss = loss_cls

                task_losses.append(task_loss)

            if not task_losses:
                continue  # nothing labelled in this batch at all

            # Aggregate loss
            if args.use_uncertainty and len(task_losses) == len(conditions):
                # UncertaintyLoss holds one log-variance per task and expects them
                # in a fixed order, so it only applies when every task contributed.
                loss, task_weights = uncertainty_loss(task_losses)
            else:
                loss = sum(task_losses) / len(task_losses)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_param_list, 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / num_batches

        # Validation (timed end-to-end for inference throughput stat)
        val_start = time.time()
        (val_loss, val_weighted_logloss, val_accuracies, val_per_class_metrics,
         val_auc_auprc_metrics, val_auc_auprc_overall) = evaluate(
            model, val_loader, text_db, criterion, uncertainty_loss, device
        )
        val_elapsed = time.time() - val_start
        n_val_samples = len(val_loader.dataset)
        val_throughput = float(n_val_samples / val_elapsed) if val_elapsed > 0 else 0.0
        val_ms_per_sample = float(1000 * val_elapsed / n_val_samples) if n_val_samples > 0 else 0.0

        # Scheduler step (cosine, per epoch)
        scheduler.step()

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary ({epoch_time:.1f}s)")
        print("="*70)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Logit scale: {model.logit_scale.exp().item():.2f}")

        print(f"\nValidation Accuracies:")
        print(f"  Spinal Canal:     {val_accuracies['spinal_canal']:.2%}")
        print(f"  Left Foraminal:   {val_accuracies['left_foraminal']:.2%}")
        print(f"  Right Foraminal:  {val_accuracies['right_foraminal']:.2%}")

        if args.use_uncertainty:
            task_weights = uncertainty_loss.get_task_weights().detach().cpu().numpy()
            print(f"\nTask Weights (from UncertaintyLoss):")
            print(f"  Spinal Canal:   {task_weights[0]:.4f}")
            print(f"  Left Foraminal: {task_weights[1]:.4f}")
            print(f"  Right Foraminal: {task_weights[2]:.4f}")

        # Compute average Severe F1 (selection metric for "best")
        avg_severe_f1 = float('nan')
        if val_per_class_metrics:
            avg_severe_f1 = float(np.mean([
                val_per_class_metrics[c]['f1'][2]
                for c in val_per_class_metrics
            ]))
        print(f"\nAvg Severe F1: {avg_severe_f1:.4f}")

        # Print detailed per-class metrics every epoch (Severe F1 is critical)
        print_per_class_metrics(val_per_class_metrics)

        # Which epoch gets saved. Severe F1 is the historical default and stays
        # so, but it is a poor selector on this validation set: across the last
        # five epochs of the 2026-08-22 runs its epoch-to-epoch SD was 0.007-0.010
        # while Severe AUPRC's was 0.0006 -- 10-15x steadier, because AUPRC is
        # threshold-free and reads the ranking over all ~1940 rows instead of
        # whatever the argmax happened to do. Selecting on F1 therefore tends to
        # save the top of an oscillation rather than the best model; it inflated
        # run #1's reported figure by 0.018 over its own plateau.
        selector = args.select_by
        if selector == 'severe_f1':
            score = avg_severe_f1
        elif selector == 'severe_auprc':
            score = (val_auc_auprc_overall or {}).get('severe_auprc', float('nan'))
        elif selector == 'macro_auprc':
            score = (val_auc_auprc_overall or {}).get('macro_auprc', float('nan'))
        else:  # weighted_logloss -- lower is better, so negate to keep one rule
            score = -val_weighted_logloss

        if score != score:  # NaN, e.g. a condition had no Severe cases this epoch
            score = -float('inf')

        is_best = score > best_score
        if is_best:
            best_score = score
            best_severe_f1 = avg_severe_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_path = save_dir / f'best_model_hybrid{run_tag}.pth'
            # Save only trainable weights (keeps file small ~2MB)
            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if 'cbam.' not in k and 'biomedclip.' not in k
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'best_severe_f1': best_severe_f1,
                'args': vars(args),
            }, best_path)
            print(f"\n✓ Saved best model to {best_path} "
                  f"({args.select_by}={score if args.select_by != 'weighted_logloss' else -score:.4f}, "
                  f"Severe F1: {best_severe_f1:.4f})")

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
                    'best_path': str(best_path),
                    'cbam_checkpoint': args.cbam_checkpoint,
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
                'epoch_seconds': float(epoch_time),
                'eval_seconds': float(val_elapsed),
                'eval_throughput_samples_per_sec': val_throughput,
            },
        )

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_hybrid_epoch_{epoch+1}.pth'
            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if 'cbam.' not in k and 'biomedclip.' not in k
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'best_severe_f1': best_severe_f1,
                'args': vars(args),
            }, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")

        print(f"\nBest Severe F1: {best_severe_f1:.4f} (Epoch {best_epoch})")
        print(f"Epochs without improvement: {epochs_without_improvement}")

        # Early stopping
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {args.early_stop_patience} epochs without improvement")
            break

        print("="*70 + "\n")

    # Final summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Model: Epoch {best_epoch} (Severe F1: {best_severe_f1:.4f})")
    print(f"Saved to: {save_dir / f'best_model_hybrid{run_tag}.pth'}")
    print("="*70)


if __name__ == '__main__':
    main()
