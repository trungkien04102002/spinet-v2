"""Per-level evaluation of 3 third-party baselines on RSNA 2024.

Advisor requirement: evaluate at per-(patient, condition, level) granularity
rather than patient-level ANY-rule aggregation.

Ground truth and predictions are each projected into a long table with columns
(study_id, condition, level, binary_label) where:
    Normal/Mild -> 0
    Moderate     -> 1
    Severe       -> 1

Different models have different native granularity:
- NingShen: per-level (explicit)
- MedGemma: per-level (parsed from JSON)
- SpineNetV2 upstream: patient-level only (no level column) -> we replicate
    the patient-ANY prediction across all 5 levels. This is an approximation;
    narrative must note this.

Output: one table per condition matching the existing LaTeX paper structure
        (Accuracy / Precision / Recall / F1). Also saves:
- per_level_metrics.csv (machine-readable)
- per_level_metrics_report.md (human-readable)
- evidence_cases.md (2-3 patients per model per condition for advisor demo)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
GT_PATH = ROOT / "table1_baselines" / "ground_truth.csv"
SPINENET_PATH = ROOT / "table1_baselines" / "predictions" / "spinenetv2_upstream.csv"
MEDGEMMA_PATH = ROOT / "table1_baselines" / "predictions" / "medgemma.json"
NINGSHEN_PATH = ROOT / "table1_baselines" / "predictions" / "ningshen.csv"
OUT_DIR = ROOT / "table1_baselines" / "metrics_per_level"

CONDITIONS = [
    ("spinal_canal_stenosis", "Spinal Canal Stenosis"),
    ("left_neural_foraminal_narrowing", "Left Foraminal Narrowing"),
    ("right_neural_foraminal_narrowing", "Right Foraminal Narrowing"),
]
LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def severity_to_binary(value) -> int | None:
    """Map severity string to binary label; return None for missing/unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in {"normal/mild", "normal", "mild"}:
        return 0
    if s in {"moderate"}:
        return 1
    if s in {"severe"}:
        return 1
    return None


# ---------------------------------------------------------------------------
# Parsers -> long table with (study_id, condition, level, label)
# ---------------------------------------------------------------------------

def parse_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        for cond_short, _ in CONDITIONS:
            for level in LEVELS:
                col = f"{cond_short}_{level}"
                if col not in row:
                    continue
                label = severity_to_binary(row[col])
                if label is None:
                    continue
                rows.append({
                    "study_id": study_id,
                    "condition": cond_short,
                    "level": level,
                    "label": label,
                })
    return pd.DataFrame(rows)


def parse_ningshen(path: Path) -> pd.DataFrame:
    """NingShen row_id = '{study}_{condition_with_underscores}_{level}'.

    Known conditions (from CLAUDE.md):
      - spinal_canal_stenosis
      - left_neural_foraminal_narrowing
      - right_neural_foraminal_narrowing
      - left_subarticular_stenosis
      - right_subarticular_stenosis
    """
    known_conditions = [c for c, _ in CONDITIONS]
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        row_id = row["row_id"]
        # level is the trailing '_l#_l#' or '_l#_s1' suffix
        match = re.match(r"^(\d+)_(.+?)_(l[1-5]_(?:l[1-5]|s1))$", row_id)
        if not match:
            continue
        study_id = int(match.group(1))
        condition = match.group(2)
        level = match.group(3)
        if condition not in known_conditions:
            continue
        # argmax over normal_mild / moderate / severe.
        probs = [row["normal_mild"], row["moderate"], row["severe"]]
        pred_class = int(np.argmax(probs))  # 0=normal, 1=moderate, 2=severe
        label = 0 if pred_class == 0 else 1
        rows.append({
            "study_id": study_id,
            "condition": condition,
            "level": level,
            "label": label,
        })
    return pd.DataFrame(rows)


def parse_medgemma(path: Path) -> pd.DataFrame:
    """MedGemma output is JSON list; each entry has model_output as markdown-wrapped JSON string.

    model_output example:
      "```json\n{\n  \"L1/L2\": {\"Spinal_Canal_Stenosis\": \"Normal/Mild\", ...}, ...}\n```"
    """
    with open(path) as fh:
        entries = json.load(fh)

    condition_map = {
        "Spinal_Canal_Stenosis": "spinal_canal_stenosis",
        "Left_Neural_Foraminal_Narrowing": "left_neural_foraminal_narrowing",
        "Right_Neural_Foraminal_Narrowing": "right_neural_foraminal_narrowing",
    }
    level_map = {
        "L1/L2": "l1_l2", "L2/L3": "l2_l3", "L3/L4": "l3_l4",
        "L4/L5": "l4_l5", "L5/S1": "l5_s1",
    }

    rows = []
    for entry in entries:
        study_id = int(entry["study_id"])
        raw = entry["model_output"]
        # Strip ```json ... ``` fences (case-insensitive).
        cleaned = re.sub(r"^```json\s*\n?|^```\s*\n?|\n?```\s*$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        for level_key, level_norm in level_map.items():
            level_data = parsed.get(level_key, {})
            if not isinstance(level_data, dict):
                continue
            for cond_key, cond_norm in condition_map.items():
                value = level_data.get(cond_key)
                label = severity_to_binary(value)
                if label is None:
                    continue
                rows.append({
                    "study_id": study_id,
                    "condition": cond_norm,
                    "level": level_norm,
                    "label": label,
                })
    return pd.DataFrame(rows)


def parse_spinenetv2(path: Path) -> pd.DataFrame:
    """SpineNetV2 upstream: 7-8 rows per study, no per-level keying.

    Aggregate via patient-ANY rule:
      - CentralCanalStenosis: ordinal {1..4}, threshold >= 3 -> positive
      - ForaminalStenosisLeft/Right: already binary {0,1}

    Replicate patient-level prediction across all 5 vertebral levels.
    """
    df = pd.read_csv(path)

    def _agg_central(g: pd.DataFrame) -> int:
        return int((g["CentralCanalStenosis"] >= 3).any())

    def _agg_foraminal(g: pd.DataFrame, col: str) -> int:
        return int((g[col] > 0).any())

    per_patient = df.groupby("study_id").apply(
        lambda g: pd.Series({
            "spinal_canal_stenosis": _agg_central(g),
            "left_neural_foraminal_narrowing": _agg_foraminal(g, "ForaminalStenosisLeft"),
            "right_neural_foraminal_narrowing": _agg_foraminal(g, "ForaminalStenosisRight"),
        })
    ).reset_index()

    rows = []
    for _, row in per_patient.iterrows():
        study_id = int(row["study_id"])
        for cond_short, _ in CONDITIONS:
            pred = int(row[cond_short])
            for level in LEVELS:
                rows.append({
                    "study_id": study_id,
                    "condition": cond_short,
                    "level": level,
                    "label": pred,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(gt_long: pd.DataFrame, pred_long: pd.DataFrame) -> pd.DataFrame:
    """Compute per-condition metrics by inner-joining GT and predictions on (study, condition, level)."""
    merged = gt_long.merge(
        pred_long, on=["study_id", "condition", "level"],
        how="inner", suffixes=("_gt", "_pred"),
    )
    rows = []
    for cond_short, cond_display in CONDITIONS:
        subset = merged[merged["condition"] == cond_short]
        if subset.empty:
            rows.append({
                "condition": cond_display, "n": 0,
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "TP": 0, "FP": 0, "FN": 0, "TN": 0,
            })
            continue
        y_true = subset["label_gt"].to_numpy()
        y_pred = subset["label_pred"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        # Counts for confusion matrix.
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        rows.append({
            "condition": cond_display,
            "n": int(len(y_true)),
            "n_patients": int(subset["study_id"].nunique()),
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evidence picker
# ---------------------------------------------------------------------------

def pick_evidence_cases(gt_long: pd.DataFrame, pred_long: pd.DataFrame, model_name: str, n: int = 3) -> list[dict]:
    """Pick a few instructive patient cases where prediction agrees/disagrees with GT.

    Returns: list of dicts with study_id, per-condition TP/FP/FN/TN pattern.
    """
    merged = gt_long.merge(
        pred_long, on=["study_id", "condition", "level"],
        how="inner", suffixes=("_gt", "_pred"),
    )
    # Find patients that have at least 1 TP AND 1 error (FP or FN) across conditions.
    interesting_studies = []
    for study_id, group in merged.groupby("study_id"):
        tp = int(((group["label_gt"] == 1) & (group["label_pred"] == 1)).sum())
        fn = int(((group["label_gt"] == 1) & (group["label_pred"] == 0)).sum())
        fp = int(((group["label_gt"] == 0) & (group["label_pred"] == 1)).sum())
        tn = int(((group["label_gt"] == 0) & (group["label_pred"] == 0)).sum())
        interesting_studies.append({
            "model": model_name, "study_id": int(study_id),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "total": len(group),
        })
    interesting_studies.sort(key=lambda d: (d["TP"] + d["FN"], -d["FP"]), reverse=True)
    return interesting_studies[:n]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing ground truth...")
    gt_long = parse_ground_truth(GT_PATH)
    print(f"  GT: {len(gt_long):,} rows, {gt_long['study_id'].nunique():,} patients")

    parsers = {
        "NingShen": (parse_ningshen, NINGSHEN_PATH),
        "SpineNetV2 upstream": (parse_spinenetv2, SPINENET_PATH),
        "MedGemma": (parse_medgemma, MEDGEMMA_PATH),
    }

    all_metrics = []
    all_evidence = []
    for model_name, (fn, path) in parsers.items():
        print(f"\nParsing {model_name}...")
        pred_long = fn(path)
        print(f"  {model_name}: {len(pred_long):,} rows, {pred_long['study_id'].nunique():,} patients")
        # Save parsed long table for inspection.
        pred_long.to_csv(args.output_dir / f"{model_name.lower().replace(' ', '_')}_long.csv", index=False)

        metrics = compute_metrics(gt_long, pred_long)
        metrics["model"] = model_name
        all_metrics.append(metrics)
        print(metrics.to_string(index=False))

        evidence = pick_evidence_cases(gt_long, pred_long, model_name, n=3)
        all_evidence.extend(evidence)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined[["model", "condition", "n", "n_patients", "accuracy", "precision", "recall", "f1", "TP", "FP", "FN", "TN"]]
    combined.to_csv(args.output_dir / "per_level_metrics.csv", index=False)

    evidence_df = pd.DataFrame(all_evidence)
    evidence_df.to_csv(args.output_dir / "evidence_cases.csv", index=False)

    # Human-readable report.
    report_lines = ["# Per-Level Evaluation Results\n"]
    report_lines.append("Granularity: per (study_id, condition, level). Binary: Moderate|Severe -> 1.\n")
    for cond_short, cond_display in CONDITIONS:
        report_lines.append(f"\n## {cond_display}\n")
        report_lines.append("| Model | N | Accuracy | Precision | Recall | F1 |")
        report_lines.append("|---|---|---|---|---|---|")
        for _, row in combined[combined["condition"] == cond_display].iterrows():
            report_lines.append(
                f"| {row['model']} | {row['n']} | {row['accuracy']}% | {row['precision']}% | {row['recall']}% | {row['f1']}% |"
            )
    report_lines.append("\n## Evidence cases (top 3 per model by abnormal positives)\n")
    report_lines.append("| Model | Study ID | TP | FP | FN | TN | Total |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for row in all_evidence:
        report_lines.append(
            f"| {row['model']} | {row['study_id']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | {row['total']} |"
        )
    (args.output_dir / "per_level_metrics_report.md").write_text("\n".join(report_lines) + "\n")

    print(f"\nWrote results to {args.output_dir}/")


if __name__ == "__main__":
    main()
