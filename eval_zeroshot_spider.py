"""
Zero-shot evaluation on SPIDER dataset using trained Hybrid model.

Workflow:
    1. Load Hybrid checkpoint (trained on RSNA only)
    2. Encode SPIDER disease text prompts via frozen BiomedCLIP text encoder
    3. For each volume, compute image_emb via Hybrid
    4. Cosine similarity per disease -> argmax -> predict
    5. Per-disease F1 macro, balanced accuracy, AUC

Note: NO training on SPIDER. Pure zero-shot transfer.

Usage:
    python3 eval_zeroshot_spider.py \\
        --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \\
        --cbam-checkpoint checkpoints/rsna/best_model_attention.pth \\
        --spider-test rsna_preprocessed_spider/spider_zeroshot_test.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from spinenet.models.grading_hybrid import SpineNetHybrid
from prepare_spider_zeroshot import SPIDER_DISEASES, DISEASE_TIER


# Optional prompt template variants. Set --prompt-template to switch.
PROMPT_TEMPLATES = {
    "bare": "{label}",
    "med": "a magnetic resonance image of {label}",
    "full": "a sagittal MRI of the lumbar spine showing {label}",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid-checkpoint", type=str, required=True,
                   help="Path to Hybrid checkpoint (best_model_hybrid.pth)")
    p.add_argument("--cbam-checkpoint", type=str, required=True,
                   help="Path to base CBAM checkpoint (used as backbone in Hybrid)")
    p.add_argument("--spider-test", type=str, default="rsna_preprocessed_spider/spider_zeroshot_test.csv")
    p.add_argument("--volumes-dir", type=str, default="rsna_preprocessed_spider")
    p.add_argument("--output", type=str, default="experiments/paper_results/spider_zeroshot/results.csv")
    p.add_argument("--prompt-template", type=str, default="med",
                   choices=list(PROMPT_TEMPLATES.keys()))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--slice-strategy", type=str, default="static",
                   choices=["static", "dynamic"])
    return p.parse_args()


def encode_disease_prompts(model: SpineNetHybrid, template: str) -> Dict[str, torch.Tensor]:
    """
    For each disease, encode all class prompts via BiomedCLIP text.
    Returns: {disease: [num_classes, 512]}
    """
    db = {}
    for disease, info in SPIDER_DISEASES.items():
        prompts = info["prompts"]
        texts = []
        keys = sorted(prompts.keys())
        for k in keys:
            label_text = prompts[k]
            text = PROMPT_TEMPLATES[template].format(label=label_text)
            texts.append(text)
        embs = model.encode_text(texts)  # [num_classes, 512]
        db[disease] = embs
    return db


def evaluate_disease(image_embs: np.ndarray, true_labels: np.ndarray,
                     text_embs: torch.Tensor, label_keys: List[int]) -> Dict:
    """
    Evaluate one disease.

    Args:
        image_embs: [N, 512] numpy
        true_labels: [N] integer labels
        text_embs: [num_classes, 512] tensor
        label_keys: list of class indices in order of text_embs rows

    Returns:
        dict with f1_macro, balanced_acc, per-class metrics, AUC (if binary)
    """
    # Cosine similarity
    img_t = torch.from_numpy(image_embs).float()
    sims = img_t @ text_embs.cpu().T  # [N, num_classes]
    pred_idx = sims.argmax(dim=-1).numpy()
    pred_labels = np.array([label_keys[i] for i in pred_idx])

    # Filter valid samples
    valid = ~np.isnan(true_labels)
    y_true = true_labels[valid].astype(int)
    y_pred = pred_labels[valid]

    metrics = {
        "n_samples": int(valid.sum()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # Per-class P/R/F1
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_keys, average=None, zero_division=0
    )
    for i, k in enumerate(label_keys):
        metrics[f"class_{k}_precision"] = float(p[i])
        metrics[f"class_{k}_recall"] = float(r[i])
        metrics[f"class_{k}_f1"] = float(f[i])
        metrics[f"class_{k}_support"] = int(sup[i])

    # AUC for binary
    if len(label_keys) == 2:
        try:
            scores_binary = sims[:, 1] - sims[:, 0]
            metrics["auc"] = float(roc_auc_score(y_true, scores_binary[valid].numpy()))
        except Exception:
            metrics["auc"] = float("nan")

    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Hybrid checkpoint: {args.hybrid_checkpoint}")
    print(f"CBAM checkpoint:   {args.cbam_checkpoint}")
    print(f"Prompt template:   {args.prompt_template}")
    print(f"Slice strategy:    {args.slice_strategy}")

    # ---- Load Hybrid ----
    print("\nBuilding Hybrid model...")
    model = SpineNetHybrid(
        cbam_checkpoint_path=args.cbam_checkpoint,
        biomedclip_device=str(device),
        slice_strategy=args.slice_strategy,
    ).to(device)

    # Load trainable weights from Hybrid checkpoint
    ckpt = torch.load(args.hybrid_checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded Hybrid weights. Missing keys (frozen modules expected): {len(missing)}; Unexpected: {len(unexpected)}")
    model.eval()

    # ---- Encode disease prompts ----
    print("\nEncoding SPIDER disease prompts...")
    text_db = encode_disease_prompts(model, args.prompt_template)
    for disease, embs in text_db.items():
        print(f"  {disease}: {embs.shape}")

    # ---- Load test set ----
    test_df = pd.read_csv(args.spider_test)
    volumes_root = Path(args.volumes_dir)
    print(f"\nTest samples: {len(test_df)}")

    # ---- Encode volumes ----
    print("\nEncoding volumes...")
    image_embs = []
    valid_mask = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), args.batch_size)):
            batch_df = test_df.iloc[i:i + args.batch_size]
            volumes = []
            for _, row in batch_df.iterrows():
                npy_path = volumes_root / row["npy_path"]
                if not npy_path.exists():
                    volumes.append(None)
                    continue
                v = np.load(npy_path)
                volumes.append(torch.from_numpy(v))

            ok_mask = [v is not None for v in volumes]
            valid_mask.extend(ok_mask)
            valid_volumes = [v for v in volumes if v is not None]
            if not valid_volumes:
                continue
            batch = torch.stack(valid_volumes).unsqueeze(1).to(device)
            embs = model.encode_image(batch).cpu().numpy()
            j = 0
            for ok in ok_mask:
                if ok:
                    image_embs.append(embs[j])
                    j += 1
                else:
                    image_embs.append(np.zeros(512))

    image_embs = np.stack(image_embs)
    print(f"Image embeddings: {image_embs.shape}")

    # ---- Per-disease eval ----
    print("\nEvaluating per disease...")
    results = []
    for disease, info in SPIDER_DISEASES.items():
        true = test_df[disease].values.astype(float)
        # Apply valid mask
        true_with_mask = np.where(valid_mask, true, np.nan)

        keys = sorted(info["prompts"].keys())
        m = evaluate_disease(image_embs, true_with_mask, text_db[disease], keys)
        m["disease"] = disease
        m["tier"] = DISEASE_TIER[disease]
        m["prompt_template"] = args.prompt_template
        m["slice_strategy"] = args.slice_strategy
        results.append(m)

        print(f"  {disease} ({m['tier']}): F1_macro={m['f1_macro']:.3f}, "
              f"balanced_acc={m['balanced_acc']:.3f}, "
              f"AUC={m.get('auc', 'N/A')}, n={m['n_samples']}")

    # ---- Save ----
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    # ---- Summary by tier ----
    print("\n=== Summary by tier ===")
    by_tier = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r["f1_macro"])
    summary_by_tier = {}
    for tier in ["easy", "medium", "hard"]:
        if tier in by_tier:
            avg = float(np.mean(by_tier[tier]))
            summary_by_tier[tier] = {
                "avg_f1_macro": avg,
                "n_diseases": len(by_tier[tier]),
            }
            print(f"  {tier}: avg F1_macro = {avg:.3f} ({len(by_tier[tier])} diseases)")

    # ---- Best metrics file (human-readable + JSON) ----
    summary_json = out.parent / "best_metrics.json"
    summary_txt = out.parent / "best_metrics.txt"

    with open(summary_json, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hybrid_checkpoint": args.hybrid_checkpoint,
            "cbam_checkpoint": args.cbam_checkpoint,
            "test_csv": args.test_csv,
            "summary_by_tier": summary_by_tier,
            "results": results,
        }, f, indent=2)

    with open(summary_txt, "w") as f:
        f.write("=== ZERO-SHOT SPIDER EVAL ===\n")
        f.write(f"Saved at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Hybrid: {args.hybrid_checkpoint}\n")
        f.write(f"CBAM:   {args.cbam_checkpoint}\n\n")
        f.write("Summary by tier:\n")
        for tier, s in summary_by_tier.items():
            f.write(f"  {tier:6s}  avg_F1_macro={s['avg_f1_macro']:.4f}  n={s['n_diseases']}\n")
        f.write("\nPer-disease:\n")
        f.write(f"  {'Disease':24s}  {'Tier':6s}  {'F1m':>6s}  {'BalAcc':>6s}  {'AUC':>6s}  {'n':>5s}\n")
        for r in results:
            auc = r.get("auc", float("nan"))
            try:
                auc_str = f"{float(auc):.3f}"
            except (TypeError, ValueError):
                auc_str = "  N/A"
            f.write(
                f"  {r['disease']:24s}  {r['tier']:6s}  "
                f"{r['f1_macro']:6.3f}  {r['balanced_acc']:6.3f}  "
                f"{auc_str:>6s}  {r['n_samples']:5d}\n"
            )

    print(f"Saved summary -> {summary_json}")
    print(f"Saved summary -> {summary_txt}")


if __name__ == "__main__":
    main()
