"""Standalone AUC + AUPRC evaluator for trained SPIDER checkpoints.

Why this script exists
----------------------
``train_spider.py`` and ``train_spider_hybrid.py`` only persist argmax-based
metrics (Acc / Recall / Precision / F1) to ``best_metrics_*.txt``. They do
NOT keep softmax probabilities, so AUC and AUPRC cannot be computed from
the saved logs alone.

This script reruns *inference only* on the same val split that training used
(``train_test_split(range(len(dataset)), test_size=val_split, random_state=42)``),
collects softmax probabilities, and writes per-class F1 + Acc + Recall +
Precision + AUC + AUPRC + Brier per disease to a JSON next to the checkpoint.

Inference takes ~2-5 minutes on a single GPU per checkpoint — no training.

Usage
-----
    python3 eval_spider_auc.py \\
        --model baseline \\
        --checkpoint checkpoints/v3_spider/best_model_baseline.pth

    python3 eval_spider_auc.py \\
        --model cbam \\
        --checkpoint checkpoints/v3_spider/best_model_cbam.pth

    python3 eval_spider_auc.py \\
        --model hybrid \\
        --checkpoint checkpoints/v3_spider/best_model_hybrid_spider_unfreeze.pth \\
        --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth

Output
------
JSON file at ``<checkpoint_dir>/auc_auprc_<checkpoint_stem>.json`` with
per-class metrics + macro-average + binary AUC/AUPRC for each of 4 SPIDER
conditions (pfirrmann, modic, disc_narrowing, spondylolisthesis).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SPIDER_CONDITIONS = ["pfirrmann", "modic", "disc_narrowing", "spondylolisthesis"]
SPIDER_NUM_CLASSES = {
    "pfirrmann": 5,
    "modic": 4,
    "disc_narrowing": 2,
    "spondylolisthesis": 2,
}
CLASS_NAMES = {
    "pfirrmann": ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"],
    "modic": ["Type 0", "Type 1", "Type 2", "Type 3"],
    "disc_narrowing": ["No", "Yes"],
    "spondylolisthesis": ["No", "Yes"],
}
DISPLAY_NAMES = {
    "pfirrmann": "Pfirrmann Grading",
    "modic": "Modic",
    "disc_narrowing": "Disc Narrowing",
    "spondylolisthesis": "Spondylolisthesis",
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=["baseline", "cbam", "hybrid"])
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to the SPIDER .pth checkpoint")
    p.add_argument("--data-dir", default="spider")
    p.add_argument("--modality", default="t2", choices=["t1", "t2"])
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42,
                   help="random_state for split (matches training default)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cbam-checkpoint", default=None, type=str,
                   help="(hybrid only) frozen CBAM checkpoint path")
    p.add_argument("--slice-strategy", default="static", choices=["static", "dynamic"])
    p.add_argument("--output", default=None, type=str,
                   help="Override output JSON path. Default: alongside checkpoint")
    return p.parse_args()


def build_val_loader(data_dir, modality, val_split, seed, batch_size, num_workers):
    from spider_dataloader import SPIDERDataset
    full = SPIDERDataset(data_dir=data_dir, split="training", modality=modality)
    _, val_idx = train_test_split(
        list(range(len(full))),
        test_size=val_split,
        random_state=seed,
    )
    val_subset = Subset(full, val_idx)
    loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, len(val_subset)


def _collect_init():
    return {c: {"probs": [], "labels": []} for c in SPIDER_CONDITIONS}


def _stack(buf):
    out = {}
    for c in SPIDER_CONDITIONS:
        out[c] = {
            "probs": np.concatenate(buf[c]["probs"], axis=0),
            "labels": np.concatenate(buf[c]["labels"], axis=0),
        }
    return out


def infer_baseline_or_cbam(model, loader, device):
    model.eval()
    buf = _collect_init()
    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Inference", leave=False):
            volumes = volumes.unsqueeze(1).to(device)
            outputs = model(volumes)
            for c in SPIDER_CONDITIONS:
                probs = F.softmax(outputs[c], dim=1).cpu().numpy()
                buf[c]["probs"].append(probs)
                buf[c]["labels"].append(labels[c].numpy())
    return _stack(buf)


def infer_hybrid(model, loader, text_db, device):
    model.eval()
    buf = _collect_init()
    logit_scale = model.logit_scale.exp().clamp(max=100.0).item()
    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Inference", leave=False):
            volumes = volumes.unsqueeze(1).to(device)
            image_emb = model.encode_image(volumes)  # [B, 512] L2-normalized
            for c in SPIDER_CONDITIONS:
                text = text_db[c].to(device)  # [num_classes, 512]
                logits = logit_scale * (image_emb @ text.T)  # [B, num_classes]
                probs = F.softmax(logits, dim=1).cpu().numpy()
                buf[c]["probs"].append(probs)
                buf[c]["labels"].append(labels[c].numpy())
    return _stack(buf)


def compute_metrics(probs, labels, num_classes):
    """Returns dict with per-class + macro metrics for one condition."""
    preds = probs.argmax(axis=1)
    out = {}

    # Per-class precision/recall/f1
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds,
        labels=list(range(num_classes)),
        average=None, zero_division=0,
    )

    out["accuracy"] = float(accuracy_score(labels, preds))

    per_class = []
    for k in range(num_classes):
        cls_metrics = {
            "support": int(support[k]),
            "precision": float(prec[k]),
            "recall": float(rec[k]),
            "f1": float(f1[k]),
        }
        # AUC + AUPRC: one-vs-rest binary on this class
        y_binary = (labels == k).astype(int)
        if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
            try:
                cls_metrics["auc"] = float(roc_auc_score(y_binary, probs[:, k]))
            except Exception:
                cls_metrics["auc"] = float("nan")
            try:
                cls_metrics["auprc"] = float(average_precision_score(y_binary, probs[:, k]))
            except Exception:
                cls_metrics["auprc"] = float("nan")
        else:
            cls_metrics["auc"] = float("nan")
            cls_metrics["auprc"] = float("nan")
        # Brier
        cls_metrics["brier"] = float(np.mean((probs[:, k] - y_binary) ** 2))
        per_class.append(cls_metrics)

    out["per_class"] = per_class
    # macro
    valid_aucs = [c["auc"] for c in per_class if not np.isnan(c["auc"])]
    valid_auprcs = [c["auprc"] for c in per_class if not np.isnan(c["auprc"])]
    out["macro_auc"] = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    out["macro_auprc"] = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")
    out["macro_f1"] = float(np.mean(f1))
    out["macro_recall"] = float(np.mean(rec))
    out["macro_precision"] = float(np.mean(prec))
    return out


def build_model(args, device):
    if args.model == "baseline":
        from spinenet.models.grading_spider import GradingModelSPIDERBaseline
        m = GradingModelSPIDERBaseline()
    elif args.model == "cbam":
        from spinenet.models.grading_spider import GradingModelSPIDERCBAM
        m = GradingModelSPIDERCBAM()
    elif args.model == "hybrid":
        from spinenet.models.grading_hybrid import SpineNetHybrid
        if args.cbam_checkpoint is None:
            raise ValueError("--cbam-checkpoint required for hybrid")
        m = SpineNetHybrid(
            cbam_checkpoint_path=args.cbam_checkpoint,
            slice_strategy=args.slice_strategy,
        )
    else:
        raise ValueError(args.model)

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN: missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  WARN: unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    m.to(device)
    return m


def encode_spider_text_prompts_for_hybrid(model, device):
    """For hybrid mode: encode SPIDER class prompts via frozen BiomedCLIP text."""
    from prepare_spider_zeroshot import SPIDER_DISEASES
    db = {}
    # Map SPIDER_DISEASES (zero-shot prompt set) → SPIDER_CONDITIONS (training labels)
    cond_to_disease = {
        "pfirrmann": "Pfirrman_grade",
        "modic": "Modic",
        "disc_narrowing": "Disc_narrowing",
        "spondylolisthesis": "Spondylolisthesis",
    }
    for c in SPIDER_CONDITIONS:
        d = cond_to_disease[c]
        info = SPIDER_DISEASES[d]
        prompts = info["prompts"]
        keys = sorted(prompts.keys())
        texts = ["a magnetic resonance image of " + prompts[k] for k in keys]
        with torch.no_grad():
            emb = model.encode_text(texts).to(device)  # [num_classes, 512]
        db[c] = emb
    return db


def main():
    args = parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SPIDER AUC eval ===")
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    print(f"\n[1/4] Building val loader (split={args.val_split}, seed={args.seed})...")
    loader, n_val = build_val_loader(
        args.data_dir, args.modality, args.val_split, args.seed,
        args.batch_size, args.num_workers,
    )
    print(f"  Val samples: {n_val}")

    print(f"\n[2/4] Building model + loading checkpoint...")
    model = build_model(args, device)

    print(f"\n[3/4] Running inference...")
    if args.model == "hybrid":
        text_db = encode_spider_text_prompts_for_hybrid(model, device)
        results = infer_hybrid(model, loader, text_db, device)
    else:
        results = infer_baseline_or_cbam(model, loader, device)

    print(f"\n[4/4] Computing metrics per condition...")
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "checkpoint": args.checkpoint,
        "val_split": args.val_split,
        "seed": args.seed,
        "n_val_samples": n_val,
        "per_condition": {},
    }
    for c in SPIDER_CONDITIONS:
        m = compute_metrics(results[c]["probs"], results[c]["labels"], SPIDER_NUM_CLASSES[c])
        summary["per_condition"][c] = m

    # Macro across 4 conditions
    summary["macro_across_conditions"] = {
        "f1": float(np.mean([summary["per_condition"][c]["macro_f1"] for c in SPIDER_CONDITIONS])),
        "accuracy": float(np.mean([summary["per_condition"][c]["accuracy"] for c in SPIDER_CONDITIONS])),
        "auc": float(np.mean([summary["per_condition"][c]["macro_auc"] for c in SPIDER_CONDITIONS])),
        "auprc": float(np.mean([summary["per_condition"][c]["macro_auprc"] for c in SPIDER_CONDITIONS])),
        "recall": float(np.mean([summary["per_condition"][c]["macro_recall"] for c in SPIDER_CONDITIONS])),
        "precision": float(np.mean([summary["per_condition"][c]["macro_precision"] for c in SPIDER_CONDITIONS])),
    }
    summary["inference_seconds"] = time.time() - t0

    # Decide output path
    if args.output:
        out_path = Path(args.output)
    else:
        ck = Path(args.checkpoint)
        out_path = ck.parent / f"auc_auprc_{ck.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Saved JSON: {out_path}")

    # Pretty TXT
    txt_path = out_path.with_suffix(".txt")
    lines = [
        f"=== SPIDER AUC eval ===",
        f"Model:       {args.model}",
        f"Checkpoint:  {args.checkpoint}",
        f"Val samples: {n_val}",
        f"",
        f"Per-condition (macro across classes):",
        f"  {'Condition':<25} {'F1':<8} {'Acc':<8} {'Rec':<8} {'Prec':<8} {'AUC':<8} {'AUPRC':<8}",
    ]
    for c in SPIDER_CONDITIONS:
        m = summary["per_condition"][c]
        lines.append(
            f"  {DISPLAY_NAMES[c]:<25} "
            f"{m['macro_f1']:<8.3f} {m['accuracy']:<8.3f} "
            f"{m['macro_recall']:<8.3f} {m['macro_precision']:<8.3f} "
            f"{m['macro_auc']:<8.3f} {m['macro_auprc']:<8.3f}"
        )
    macro = summary["macro_across_conditions"]
    lines += [
        f"",
        f"Mean across 4 conditions:",
        f"  F1     {macro['f1']:.3f}    Acc   {macro['accuracy']:.3f}",
        f"  Recall {macro['recall']:.3f}    Prec  {macro['precision']:.3f}",
        f"  AUC    {macro['auc']:.3f}    AUPRC {macro['auprc']:.3f}",
        f"",
        f"Inference: {summary['inference_seconds']:.1f}s for {n_val} samples",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"✓ Saved TXT:  {txt_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
