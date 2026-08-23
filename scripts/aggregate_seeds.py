"""Aggregate `best_metrics_*.json` files across multiple seed runs.

Computes mean ± std for the headline numbers used in the paper tables:

- RSNA Hybrid (file format = MetricsLogger output):
    * Mean F1 macro     (averaged from per_class_metrics across 3 conditions)
    * Mean Accuracy     (averaged from val_accuracies)
    * Avg Severe F1     (already a top-level field)

- SPIDER Hybrid (file format = train_spider_hybrid output):
    * Mean F1 macro     (val_mean_f1_macro, top-level)
    * Mean Accuracy     (val_mean_acc, top-level)
    * Per-condition F1  (val_f1_macro dict)

Usage:
    # Aggregate 3 RSNA Hybrid seeds:
    python scripts/aggregate_seeds.py rsna  \
        experiments/hybrid/best_metrics_hybrid_seed42.json \
        experiments/hybrid/best_metrics_hybrid_seed0.json \
        experiments/hybrid/best_metrics_hybrid_seed7.json

    # Aggregate 3 SPIDER Hybrid frozen seeds:
    python scripts/aggregate_seeds.py spider \
        experiments/spider_phase4/best_metrics_hybrid_spider_frozen.json \
        experiments/spider_phase4/best_metrics_hybrid_spider_frozen_seed0.json \
        experiments/spider_phase4/best_metrics_hybrid_spider_frozen_seed7.json

The output is a markdown table you can paste straight into the paper.
"""
import json
import sys
from pathlib import Path
from statistics import mean, pstdev


def _f1_macro_from_per_class(per_class):
    """For RSNA: per_class_metrics[cond] = {'precision': [...], 'recall': [...], 'f1': [...]}.
    Macro-F1 per condition = mean of f1 list. Returns dict cond -> macro-F1."""
    out = {}
    for cond, m in per_class.items():
        f1_list = m.get("f1") or m.get("F1")
        if isinstance(f1_list, list) and len(f1_list) > 0:
            out[cond] = float(mean(f1_list))
    return out


def _stats(values):
    if len(values) <= 1:
        return mean(values), 0.0
    return mean(values), pstdev(values)


def aggregate_rsna(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        per_cls = d["per_class_metrics"]
        f1_per_cond = _f1_macro_from_per_class(per_cls)
        acc_per_cond = d["val_accuracies"]
        rows.append({
            "path": p,
            "epoch": d.get("epoch"),
            "mean_f1_macro": mean(f1_per_cond.values()) if f1_per_cond else float("nan"),
            "mean_acc": mean(acc_per_cond.values()),
            "avg_severe_f1": d.get("avg_severe_f1", float("nan")),
            "f1_per_cond": f1_per_cond,
            "acc_per_cond": acc_per_cond,
        })
    return rows


def aggregate_spider(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows.append({
            "path": p,
            "epoch": d.get("epoch"),
            "mean_f1_macro": d["val_mean_f1_macro"],
            "mean_acc": d["val_mean_acc"],
            "f1_per_cond": d["val_f1_macro"],
            "acc_per_cond": d["val_accuracies"],
        })
    return rows


def report(kind, rows):
    print(f"# Aggregated {kind.upper()} runs (n={len(rows)})\n")
    for r in rows:
        print(f"- {r['path']} (epoch {r['epoch']}): "
              f"F1m={r['mean_f1_macro']:.4f}  Acc={r['mean_acc']:.4f}")
    print()

    f1m_mean, f1m_std = _stats([r["mean_f1_macro"] for r in rows])
    acc_mean, acc_std = _stats([r["mean_acc"] for r in rows])
    print("## Headline (paste into paper)\n")
    print("| Metric | Mean | Std | Format |")
    print("|---|---|---|---|")
    print(f"| Mean F1 macro | {f1m_mean:.3f} | {f1m_std:.3f} | {f1m_mean:.3f} $\\pm$ {f1m_std:.3f} |")
    print(f"| Mean Accuracy | {acc_mean:.4f} | {acc_std:.4f} | {acc_mean*100:.1f}\\% $\\pm$ {acc_std*100:.1f}\\% |")

    if kind == "rsna":
        sev_mean, sev_std = _stats([r["avg_severe_f1"] for r in rows])
        print(f"| Avg Severe F1 | {sev_mean:.3f} | {sev_std:.3f} | {sev_mean:.3f} $\\pm$ {sev_std:.3f} |")
    print()

    # Per-condition breakdown
    conds = list(rows[0]["f1_per_cond"].keys())
    print("## Per-condition F1 macro (mean ± std)\n")
    print("| Condition | " + " | ".join(f"{c}" for c in conds) + " |")
    print("|" + "---|" * (len(conds) + 1))
    means = []
    stds = []
    for c in conds:
        vals = [r["f1_per_cond"][c] for r in rows]
        m, s = _stats(vals)
        means.append(m)
        stds.append(s)
    print("| Mean | " + " | ".join(f"{m:.3f}" for m in means) + " |")
    print("| Std  | " + " | ".join(f"{s:.3f}" for s in stds) + " |")
    print("| Combined | " + " | ".join(f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(means, stds)) + " |")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    kind = sys.argv[1].lower()
    paths = sys.argv[2:]
    if kind == "rsna":
        rows = aggregate_rsna(paths)
    elif kind == "spider":
        rows = aggregate_spider(paths)
    else:
        print(f"Unknown kind: {kind!r}. Use 'rsna' or 'spider'.")
        sys.exit(2)
    report(kind, rows)


if __name__ == "__main__":
    main()
