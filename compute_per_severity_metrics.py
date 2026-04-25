"""
Compute per-severity metrics (Normal/Moderate/Severe) for Phase 1 baselines
and Phase 2 RSNA-trained models. Used to populate Table 1.

Reads:
    experiments/paper_results/table1_baselines/predictions/
        spinenetv2_upstream.csv
        ningshen.csv
        medgemma.json
    checkpoints/rsna/best_model.pth        (vanilla SpineNet, optional re-eval)
    checkpoints/rsna/best_model_attention.pth  (CBAM, optional re-eval)

Outputs:
    experiments/paper_results/table1_baselines/metrics_per_severity.csv

Severity convention:
    0 = Normal/Mild
    1 = Moderate
    2 = Severe
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


SEVERITY_NAMES = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}
CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", type=str,
                   default="experiments/paper_results/table1_baselines/predictions")
    p.add_argument("--ground-truth", type=str,
                   default="experiments/paper_results/table1_baselines/ground_truth.csv")
    p.add_argument("--output", type=str,
                   default="experiments/paper_results/table1_baselines/metrics_per_severity.csv")
    p.add_argument("--include-rsna-models", action="store_true",
                   help="Also re-eval best_model.pth and best_model_attention.pth on RSNA val (requires GPU)")
    p.add_argument("--cbam-checkpoint", type=str, default="checkpoints/rsna/best_model_attention.pth")
    p.add_argument("--baseline-checkpoint", type=str, default="checkpoints/rsna/best_model.pth")
    p.add_argument("--data-dir", type=str, default="rsna_preprocessed")
    return p.parse_args()


def normalize_severity(label) -> int:
    """Convert various severity labels to 0/1/2."""
    if pd.isna(label):
        return -1
    if isinstance(label, str):
        s = label.lower().strip()
        if "normal" in s or "mild" in s:
            return 0
        if "moderate" in s:
            return 1
        if "severe" in s:
            return 2
        return -1
    if isinstance(label, (int, float)):
        v = int(label)
        return v if v in (0, 1, 2) else -1
    return -1


def load_predictions_csv(path: Path) -> pd.DataFrame:
    """Generic CSV loader. Expected: study_id, condition, level, prediction columns."""
    return pd.read_csv(path)


def load_predictions_medgemma(path: Path) -> pd.DataFrame:
    """MedGemma predictions in JSON format."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for entry in data:
        rows.append(entry)
    return pd.DataFrame(rows)


def compute_metrics_one_model(preds: List[int], trues: List[int], model_name: str) -> List[Dict]:
    """
    Compute per-severity Precision/Recall/F1 + macro F1.

    Args:
        preds: list of predicted severities (0/1/2)
        trues: list of true severities (0/1/2)
        model_name: model identifier for output rows

    Returns:
        list of dicts (one per severity class + one macro)
    """
    p_arr = np.array(preds)
    t_arr = np.array(trues)
    valid = (t_arr != -1) & (p_arr != -1)
    p_arr, t_arr = p_arr[valid], t_arr[valid]

    if len(t_arr) == 0:
        return []

    precision, recall, f1, support = precision_recall_fscore_support(
        t_arr, p_arr, labels=[0, 1, 2], average=None, zero_division=0
    )

    rows = []
    for i in range(3):
        rows.append({
            "model": model_name,
            "severity": SEVERITY_NAMES[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    # Macro avg
    rows.append({
        "model": model_name,
        "severity": "macro_avg",
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "support": int(np.sum(support)),
    })

    return rows


def eval_phase1_baselines(args) -> List[Dict]:
    """Eval 3 Phase 1 baselines from prediction files + ground truth."""
    pred_dir = Path(args.predictions_dir)
    gt_path = Path(args.ground_truth)

    if not gt_path.exists():
        print(f"WARNING: ground truth not found: {gt_path}, skipping Phase 1 baselines")
        return []
    gt = pd.read_csv(gt_path)
    gt_dict = {}
    for _, row in gt.iterrows():
        key = (str(row["study_id"]), row["condition"], row["level"])
        gt_dict[key] = normalize_severity(row.get("severity"))

    all_rows = []

    # SpineNetV2 upstream
    sn_path = pred_dir / "spinenetv2_upstream.csv"
    if sn_path.exists():
        df = load_predictions_csv(sn_path)
        preds, trues = [], []
        for _, row in df.iterrows():
            key = (str(row.get("study_id", "")), row.get("condition", ""), row.get("level", ""))
            t = gt_dict.get(key, -1)
            p = normalize_severity(row.get("prediction"))
            preds.append(p)
            trues.append(t)
        all_rows.extend(compute_metrics_one_model(preds, trues, "SpineNetV2_upstream"))
        print(f"SpineNetV2 upstream: {len(preds)} rows")

    # NingShen
    ns_path = pred_dir / "ningshen.csv"
    if ns_path.exists():
        df = load_predictions_csv(ns_path)
        preds, trues = [], []
        for _, row in df.iterrows():
            key = (str(row.get("study_id", "")), row.get("condition", ""), row.get("level", ""))
            t = gt_dict.get(key, -1)
            p = normalize_severity(row.get("prediction"))
            preds.append(p)
            trues.append(t)
        all_rows.extend(compute_metrics_one_model(preds, trues, "NingShen"))
        print(f"NingShen: {len(preds)} rows")

    # MedGemma
    mg_path = pred_dir / "medgemma.json"
    if mg_path.exists():
        df = load_predictions_medgemma(mg_path)
        preds, trues = [], []
        for _, row in df.iterrows():
            key = (str(row.get("study_id", "")), row.get("condition", ""), row.get("level", ""))
            t = gt_dict.get(key, -1)
            p = normalize_severity(row.get("prediction"))
            preds.append(p)
            trues.append(t)
        all_rows.extend(compute_metrics_one_model(preds, trues, "MedGemma"))
        print(f"MedGemma: {len(preds)} rows")

    return all_rows


def eval_rsna_models(args) -> List[Dict]:
    """Re-eval vanilla baseline + CBAM on RSNA val. Requires GPU."""
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import train_test_split
        from spinenet.models.grading_baseline import GradingModelBaseline
        from spinenet.models.grading_attention import GradingModelWithCBAM
        from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
    except ImportError as e:
        print(f"WARNING: imports failed for RSNA re-eval: {e}")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: GPU not available for RSNA model re-eval, skipping")
        return []

    # Build val loader (same split as training)
    full = RSNAPreprocessedDataset(data_dir=args.data_dir, split="train", transform=None)
    patients = full.metadata["study_id"].unique()
    _, val_p = train_test_split(patients, test_size=0.2, random_state=42)
    val_idx = full.metadata[full.metadata["study_id"].isin(val_p)].index.tolist()
    val_ds = Subset(full, val_idx)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    all_rows = []

    for model_name, ckpt_path, model_class in [
        ("Vanilla_SpineNetV2_RSNA", args.baseline_checkpoint, GradingModelBaseline),
        ("CBAM_RSNA", args.cbam_checkpoint, GradingModelWithCBAM),
    ]:
        if not Path(ckpt_path).exists():
            print(f"WARNING: {ckpt_path} not found, skipping {model_name}")
            continue

        print(f"\nEvaluating {model_name} from {ckpt_path}...")
        model = model_class(format="rsna")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        all_preds = {c: [] for c in CONDITIONS}
        all_trues = {c: [] for c in CONDITIONS}

        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes = volumes.unsqueeze(1).to(device)
                outputs = model(volumes)
                for cond in CONDITIONS:
                    preds = torch.argmax(outputs[cond], dim=-1).cpu().numpy()
                    all_preds[cond].extend(preds)
                    all_trues[cond].extend(labels[cond].numpy())

        # Concatenate all conditions for overall per-severity metrics
        all_p = np.concatenate([np.array(all_preds[c]) for c in CONDITIONS])
        all_t = np.concatenate([np.array(all_trues[c]) for c in CONDITIONS])
        all_rows.extend(compute_metrics_one_model(all_p.tolist(), all_t.tolist(), model_name))
        print(f"  Done. {len(all_p)} samples eval'd.")

    return all_rows


def main():
    args = parse_args()

    print("=" * 70)
    print("Computing per-severity metrics for paper Table 1")
    print("=" * 70)

    rows = []
    print("\n[1/2] Phase 1 baselines (from prediction files)...")
    rows.extend(eval_phase1_baselines(args))

    if args.include_rsna_models:
        print("\n[2/2] Phase 2 RSNA-trained models (re-eval on val)...")
        rows.extend(eval_rsna_models(args))
    else:
        print("\n[2/2] Skipping RSNA re-eval (use --include-rsna-models to enable)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Print pretty table for Severe class
    print("\n=== Severe class per-model ===")
    df = pd.DataFrame(rows)
    severe = df[df["severity"] == "Severe"]
    print(severe[["model", "precision", "recall", "f1", "support"]].to_string(index=False))


if __name__ == "__main__":
    main()
