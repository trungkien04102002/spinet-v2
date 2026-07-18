#!/usr/bin/env python3
"""
Build the external SOTA comparison table: SpineNetV2 (ours, baseline
architecture) + Hybrid (ours, full method) vs. two external re-implemented
SOTA baselines (brendanartley-style CNN+BiLSTM, and a Transformer-encoder
baseline), each aggregated over 3 seeds (42, 123, 456) as mean +/- std.

Metric-extraction logic (Mean Accuracy, Mean F1 macro) is reused verbatim
from scripts/aggregate_rsna_seeds.py so the numbers match the paper exactly.
Severe Recall is NOT a top-level stored metric in *_best_metrics.json; it is
derived the same way aggregate_rsna_seeds.py derives its other per-class
macro metrics: per_class_metrics[<condition>]['recall'][2] (index 2 ==
Severe class), averaged across the 3 conditions, then mean+/-std across
seeds.

Sanity-check targets (from the paper, seed-aggregated):
    Hybrid:      Mean F1 macro ~= 0.527,  Severe Recall ~= 48.6%
    SpineNetV2:  Severe Recall ~= 12.3%

Outputs:
    - prints a markdown quantitative table + a markdown capability table to
      stdout
    - writes experiments/sota_comparison/comparison_table.md
    - writes experiments/sota_comparison/comparison_table.tex

Usage:
    python3 experiments/sota_comparison/make_table.py
    python3 experiments/sota_comparison/make_table.py --seeds 42 123 456
"""
import argparse
import json
from pathlib import Path

import numpy as np

CONDS = ["spinal_canal", "left_foraminal", "right_foraminal"]

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent

OURS_BASE_DIR = _REPO_ROOT / "checkpoints" / "v3_20260503"
SOTA_BASE_DIR = _THIS_DIR / "checkpoints"


def _tag(seed):
    return "" if seed == 42 else f"_seed{seed}"


# config -> (kind, base_dir/subdir, function(seed) -> file prefix)
# kind == "ours"  -> file lives at OURS_BASE_DIR/<subdir>/<prefix>_best_metrics.json
# kind == "sota"  -> file lives at SOTA_BASE_DIR/<model>/<prefix>_best_metrics.json
CONFIGS = {
    "SpineNetV2": ("ours", "baseline", lambda s: f"baseline{_tag(s)}"),
    "brendanartley": ("sota", "brendanartley", lambda s: f"brendanartley{_tag(s)}"),
    "transformer": ("sota", "transformer", lambda s: f"transformer{_tag(s)}"),
    "Hybrid (Ours)": ("ours", "hybrid", lambda s: f"hybrid{_tag(s)}"),
}

# Display order for the table (kept distinct from CONFIGS insertion order
# above only for readability; currently identical).
DISPLAY_ORDER = ["SpineNetV2", "brendanartley", "transformer", "Hybrid (Ours)"]

METRIC_ORDER = [
    ("Mean Accuracy", "pct"),
    ("Mean F1 macro", "f3"),
    ("Severe F1", "f3"),
    ("Severe Recall", "pct"),
]


def metrics_from_json(d):
    """Reuses the exact metric formulas from scripts/aggregate_rsna_seeds.py
    (Mean Accuracy, Mean F1 macro, Severe F1), plus a derived Severe Recall
    (not stored top-level in the JSON, computed the same way the other
    per-class macro metrics are derived there)."""
    acc = np.mean([d["val_accuracies"][c] for c in CONDS]) * 100.0
    pcm = d["per_class_metrics"]
    f1 = np.mean([np.mean(pcm[c]["f1"]) for c in CONDS])
    severe_recall = np.mean([pcm[c]["recall"][2] for c in CONDS]) * 100.0
    return {
        "Mean Accuracy": acc,
        "Mean F1 macro": f1,
        "Severe F1": d["avg_severe_f1"],
        "Severe Recall": severe_recall,
    }


def _file_for(cfg_name, seed):
    kind, subdir, prefix_fn = CONFIGS[cfg_name]
    prefix = prefix_fn(seed)
    if kind == "ours":
        return OURS_BASE_DIR / subdir / f"{prefix}_best_metrics.json"
    return SOTA_BASE_DIR / subdir / f"{prefix}_best_metrics.json"


def collect(seeds):
    """Returns (collected, missing) where collected[cfg][metric] = list of
    per-seed values (only for seeds whose JSON file exists), and missing is
    a dict cfg -> list of missing file paths (used to decide "pending")."""
    collected = {cfg: {m: [] for m, _ in METRIC_ORDER} for cfg in CONFIGS}
    missing = {cfg: [] for cfg in CONFIGS}

    for cfg in CONFIGS:
        for s in seeds:
            fp = _file_for(cfg, s)
            if not fp.exists():
                missing[cfg].append(str(fp))
                continue
            vals = metrics_from_json(json.load(open(fp)))
            for m in collected[cfg]:
                collected[cfg][m].append(vals[m])

    return collected, missing


def agg(xs):
    a = np.asarray(xs, float)
    if a.size == 0:
        return None
    sd = a.std(ddof=1) if a.size > 1 else 0.0
    return a.mean(), sd, a.size


def fmt_cell(cfg, metric, fmt, collected, missing, n_seeds_requested):
    r = agg(collected[cfg][metric])
    if r is None:
        return "pending" if missing[cfg] else "--"
    mean, sd, n = r
    partial = "*" if n < n_seeds_requested else ""
    if fmt == "pct":
        return f"{mean:.1f} +/- {sd:.1f}{partial}"
    return f"{mean:.3f} +/- {sd:.3f}{partial}"


def fmt_cell_tex(cfg, metric, fmt, collected, missing, n_seeds_requested):
    r = agg(collected[cfg][metric])
    if r is None:
        return "\\textit{pending}" if missing[cfg] else "--"
    mean, sd, n = r
    partial = "$^{*}$" if n < n_seeds_requested else ""
    if fmt == "pct":
        return f"{mean:.1f} $\\pm$ {sd:.1f}{partial}"
    return f"{mean:.3f} $\\pm$ {sd:.3f}{partial}"


# ---------------------------------------------------------------------------
# Static qualitative capability table -- "why they can't do what we do".
# ---------------------------------------------------------------------------
CAPABILITY_ROWS = [
    (
        "Takes a new label set as input",
        {
            "SpineNetV2": "no (fixed head)",
            "brendanartley": "no (fixed head)",
            "transformer": "no (fixed head)",
            "Hybrid (Ours)": "yes (prompt-based)",
        },
    ),
    (
        "Cross-dataset transfer (RSNA to SPIDER)",
        {
            "SpineNetV2": "requires retrain",
            "brendanartley": "requires retrain",
            "transformer": "requires retrain",
            "Hybrid (Ours)": "yes",
        },
    ),
    (
        "Zero-shot to unseen labels",
        {
            "SpineNetV2": "no",
            "brendanartley": "no",
            "transformer": "no",
            "Hybrid (Ours)": "yes",
        },
    ),
    (
        "RSNA Mean F1",
        {
            "SpineNetV2": "__METRIC__",
            "brendanartley": "__METRIC__",
            "transformer": "__METRIC__",
            "Hybrid (Ours)": "__METRIC__",
        },
    ),
]

SYMBOL_MD = {
    "no (fixed head)": "x (fixed head)",
    "no": "x",
    "requires retrain": "x (requires retrain)",
    "yes (prompt-based)": "yes (prompt-based)",
    "yes": "yes",
}

SYMBOL_TEX = {
    "no (fixed head)": "\\ding{55} (fixed head)",
    "no": "\\ding{55}",
    "requires retrain": "\\ding{55} (requires retrain)",
    "yes (prompt-based)": "\\ding{51} (prompt-based)",
    "yes": "\\ding{51}",
}


def build_markdown(collected, missing, seeds):
    lines = []
    lines.append(f"# External SOTA Comparison (RSNA-2024, seeds {seeds})\n")
    lines.append(
        "Mean +/- std over available seeds. `*` marks a cell aggregated "
        "from fewer than all requested seeds. `pending` = no runs found yet "
        "(GPU training not run).\n"
    )
    lines.append("## Quantitative comparison\n")
    header = "| Metric | " + " | ".join(DISPLAY_ORDER) + " |"
    sep = "|---|" + "---|" * len(DISPLAY_ORDER)
    lines.append(header)
    lines.append(sep)
    for metric, fmt in METRIC_ORDER:
        row = [metric]
        for cfg in DISPLAY_ORDER:
            row.append(fmt_cell(cfg, metric, fmt, collected, missing, len(seeds)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Capability comparison (why external SOTA baselines can't do what we do)\n")
    header2 = "| Capability | " + " | ".join(DISPLAY_ORDER) + " |"
    sep2 = "|---|" + "---|" * len(DISPLAY_ORDER)
    lines.append(header2)
    lines.append(sep2)
    for label, per_cfg in CAPABILITY_ROWS:
        row = [label]
        for cfg in DISPLAY_ORDER:
            val = per_cfg[cfg]
            if val == "__METRIC__":
                r = agg(collected[cfg]["Mean F1 macro"])
                if r is None:
                    row.append("pending" if missing[cfg] else "--")
                else:
                    mean, sd, n = r
                    partial = "*" if n < len(seeds) else ""
                    row.append(f"{mean:.3f} +/- {sd:.3f}{partial}")
            else:
                row.append(SYMBOL_MD.get(val, val))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def build_latex(collected, missing, seeds):
    lines = []
    lines.append("% Auto-generated by experiments/sota_comparison/make_table.py")
    lines.append("% Quantitative comparison table")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{External SOTA comparison on RSNA-2024 (mean $\\pm$ std over seeds "
                  + str(seeds) + ").}")
    lines.append("\\label{tab:sota_comparison}")
    lines.append("\\begin{tabular}{l" + "c" * len(DISPLAY_ORDER) + "}")
    lines.append("\\toprule")
    lines.append("Metric & " + " & ".join(DISPLAY_ORDER) + " \\\\")
    lines.append("\\midrule")
    for metric, fmt in METRIC_ORDER:
        cells = [fmt_cell_tex(cfg, metric, fmt, collected, missing, len(seeds)) for cfg in DISPLAY_ORDER]
        lines.append(f"{metric} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("% Capability (qualitative) comparison table")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Capability comparison: why fixed-head external SOTA baselines "
                  "cannot do what our Hybrid model does.}")
    lines.append("\\label{tab:sota_capability}")
    lines.append("\\begin{tabular}{l" + "c" * len(DISPLAY_ORDER) + "}")
    lines.append("\\toprule")
    lines.append("Capability & " + " & ".join(DISPLAY_ORDER) + " \\\\")
    lines.append("\\midrule")
    for label, per_cfg in CAPABILITY_ROWS:
        cells = []
        for cfg in DISPLAY_ORDER:
            val = per_cfg[cfg]
            if val == "__METRIC__":
                r = agg(collected[cfg]["Mean F1 macro"])
                if r is None:
                    cells.append("\\textit{pending}" if missing[cfg] else "--")
                else:
                    mean, sd, n = r
                    partial = "$^{*}$" if n < len(seeds) else ""
                    cells.append(f"{mean:.3f} $\\pm$ {sd:.3f}{partial}")
            else:
                cells.append(SYMBOL_TEX.get(val, val))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = ap.parse_args()

    collected, missing = collect(args.seeds)

    for cfg in CONFIGS:
        if missing[cfg]:
            print(f"NOTE: {cfg} missing metric files (marked pending/partial where applicable):")
            for m in missing[cfg]:
                print("  -", m)
    print()

    md = build_markdown(collected, missing, args.seeds)
    tex = build_latex(collected, missing, args.seeds)

    print(md)

    md_path = _THIS_DIR / "comparison_table.md"
    tex_path = _THIS_DIR / "comparison_table.tex"
    md_path.write_text(md)
    tex_path.write_text(tex)
    print(f"\nWrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
