"""
Naked BiomedCLIP zero-shot on SPIDER — no CBAM, no Hybrid, no SpineNet training.

Ablation answering: "Does CBAM + RSNA training contribute to zero-shot, or does
BiomedCLIP do all the work alone?"

For each volume:
    9 sagittal slices -> grayscale to 3-channel RGB -> BiomedCLIP image encoder
    -> 9 x 512-d -> mean pool -> 512-d L2-normalized image embedding.

For each disease prompt: BiomedCLIP text encoder -> 512-d L2-normalized.
Predict via cosine similarity argmax. Same metrics format as eval_zeroshot_spider.py.

Usage:
    python3 eval_zeroshot_spider_naked_biomedclip.py
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
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from spinenet.models.biomedclip_wrapper import BiomedCLIPWrapper
from prepare_spider_zeroshot import DISEASE_TIER, SPIDER_DISEASES


PROMPT_TEMPLATES = {
    "bare": "{label}",
    "med": "a magnetic resonance image of {label}",
    "full": "a sagittal MRI of the lumbar spine showing {label}",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spider-test", type=str,
                   default="rsna_preprocessed_spider/spider_zeroshot_test.csv")
    p.add_argument("--volumes-dir", type=str, default="rsna_preprocessed_spider")
    p.add_argument("--output", type=str,
                   default="experiments/paper_results/spider_zeroshot_naked_biomedclip/results.csv")
    p.add_argument("--prompt-template", type=str, default="med",
                   choices=list(PROMPT_TEMPLATES.keys()))
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of volumes per batch; each volume produces 9 slices.")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def encode_volume_batch(wrapper: BiomedCLIPWrapper, volumes: torch.Tensor) -> torch.Tensor:
    """
    volumes: [B, 9, H=112, W=224] in [0, 1]
    returns: [B, 512] L2-normalized image embedding (mean of 9 slice embeddings).
    """
    B, K, H, W = volumes.shape
    flat = volumes.reshape(B * K, H, W)  # tensor [B*K, H, W]
    rgb_list = [wrapper.preprocess_slice(s) for s in flat]
    rgb_batch = torch.stack(rgb_list, dim=0).to(wrapper._device)  # [B*K, 3, 224, 224]
    embs = wrapper.encode_image(rgb_batch).reshape(B, K, -1)  # [B, 9, 512] L2-normalized
    pooled = embs.mean(dim=1)  # mean over 9 slices
    return F.normalize(pooled, dim=-1)


def encode_disease_prompts(wrapper: BiomedCLIPWrapper, template: str) -> Dict[str, tuple]:
    db = {}
    for disease, info in SPIDER_DISEASES.items():
        keys = sorted(info["prompts"].keys())
        texts = [PROMPT_TEMPLATES[template].format(label=info["prompts"][k]) for k in keys]
        embs = wrapper.encode_text(texts)  # [num_classes, 512]
        db[disease] = (keys, embs)
    return db


def evaluate_disease(image_embs: np.ndarray, true_labels: np.ndarray,
                     text_embs: torch.Tensor, label_keys: List[int]) -> Dict:
    img_t = torch.from_numpy(image_embs).float()
    sims = img_t @ text_embs.cpu().T
    pred_idx = sims.argmax(dim=-1).numpy()
    pred_labels = np.array([label_keys[i] for i in pred_idx])

    valid = ~np.isnan(true_labels)
    y_true = true_labels[valid].astype(int)
    y_pred = pred_labels[valid]

    metrics = {
        "n_samples": int(valid.sum()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_keys, average=None, zero_division=0
    )
    for i, k in enumerate(label_keys):
        metrics[f"class_{k}_precision"] = float(p[i])
        metrics[f"class_{k}_recall"] = float(r[i])
        metrics[f"class_{k}_f1"] = float(f[i])
        metrics[f"class_{k}_support"] = int(sup[i])
    if len(label_keys) == 2:
        try:
            scores_binary = sims[:, 1] - sims[:, 0]
            metrics["auc"] = float(roc_auc_score(y_true, scores_binary[valid].numpy()))
        except Exception:
            metrics["auc"] = float("nan")
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Prompt template: {args.prompt_template}")
    print("Configuration: NAKED BiomedCLIP (no CBAM, no SpineNet training)")

    # Load wrapper
    print("\nLoading BiomedCLIP...")
    wrapper = BiomedCLIPWrapper(device=str(device))
    wrapper.eval()

    # Encode prompts
    print("Encoding disease prompts...")
    text_db = encode_disease_prompts(wrapper, args.prompt_template)

    # Load test set
    test_df = pd.read_csv(args.spider_test)
    volumes_root = Path(args.volumes_dir)
    print(f"Test samples: {len(test_df)}")

    # Encode volumes
    print("\nEncoding volumes (9 slices -> mean-pool)...")
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
                volumes.append(torch.from_numpy(np.load(npy_path)))

            ok_mask = [v is not None for v in volumes]
            valid_mask.extend(ok_mask)
            valid_volumes = [v for v in volumes if v is not None]
            if not valid_volumes:
                continue
            batch = torch.stack(valid_volumes)  # [B, 9, 112, 224]
            embs = encode_volume_batch(wrapper, batch).cpu().numpy()
            j = 0
            for ok in ok_mask:
                if ok:
                    image_embs.append(embs[j])
                    j += 1
                else:
                    image_embs.append(np.zeros(512, dtype=np.float32))

    image_embs = np.stack(image_embs)
    print(f"Image embeddings: {image_embs.shape}")

    # Per-disease eval
    print("\nEvaluating per disease...")
    results = []
    for disease, info in SPIDER_DISEASES.items():
        true = test_df[disease].values.astype(float)
        true_with_mask = np.where(valid_mask, true, np.nan)
        keys, text_embs = text_db[disease]
        m = evaluate_disease(image_embs, true_with_mask, text_embs, keys)
        m["disease"] = disease
        m["tier"] = DISEASE_TIER[disease]
        m["prompt_template"] = args.prompt_template
        m["model"] = "naked_biomedclip"
        results.append(m)
        print(f"  {disease} ({m['tier']}): F1_macro={m['f1_macro']:.3f}, "
              f"balanced_acc={m['balanced_acc']:.3f}, AUC={m.get('auc', 'N/A')}, "
              f"n={m['n_samples']}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    # Summary by tier
    print("\n=== Summary by tier (naked BiomedCLIP) ===")
    by_tier = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r["f1_macro"])
    summary_by_tier = {}
    for tier in ["easy", "medium", "hard"]:
        if tier in by_tier:
            avg = float(np.mean(by_tier[tier]))
            summary_by_tier[tier] = {
                "avg_f1_macro": avg, "n_diseases": len(by_tier[tier]),
            }
            print(f"  {tier}: avg F1_macro = {avg:.3f} ({len(by_tier[tier])} diseases)")

    # Best metrics file
    summary_json = out.parent / "best_metrics.json"
    summary_txt = out.parent / "best_metrics.txt"

    with open(summary_json, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "naked_biomedclip",
            "test_csv": args.spider_test,
            "summary_by_tier": summary_by_tier,
            "results": results,
        }, f, indent=2)

    with open(summary_txt, "w") as f:
        f.write("=== ZERO-SHOT SPIDER EVAL — NAKED BIOMEDCLIP ===\n")
        f.write(f"Saved at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Configuration: BiomedCLIP image + text encoders only\n")
        f.write("              (no CBAM, no SpineNet training, no projection MLP)\n\n")
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
