#!/usr/bin/env python3
"""Adebayo et al. 2018 model-parameter randomisation test: the gate.

If a saliency map does not change when the model's learned weights are
destroyed, the map is a function of the input and the architecture alone -- an
edge detector -- and every number computed from it afterwards is void. So this
runs before the localisation and faithfulness work, not after.

Cascading randomisation: re-initialise the layers from the output backwards, one
stage at a time, and measure how far the map has moved from the original. A
trustworthy method degrades steadily; an untrustworthy one stays put.

Precision when citing this: the methods that FAILED in Adebayo et al. were
Guided Backprop and Guided Grad-CAM. Plain Grad-CAM, Integrated Gradients and
gradient x input passed. "Grad-CAM fails sanity checks" would be a misreading.
Yona & Greenfeld 2021 (arXiv:2110.14297) further argue the original conclusions
are confounded by task choice.

Usage:
    python3 experiments/explainability/sanity_checks.py --n-cases 10
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import sys
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import saliency_methods as S  # noqa: E402
from spinenet.models.grading_attention import GradingModelWithCBAM  # noqa: E402

# Output backwards. cbam4 sits between layer4 and avgpool, so it is randomised
# with the stage it gates.
CASCADE = ["fc", "cbam4", "layer4", "layer3", "layer2", "layer1"]


def reinit(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def randomise_stage(model, stage, conditions):
    if stage == "fc":
        for c in conditions:
            reinit(getattr(model, f"fc_{c}"))
    else:
        reinit(dict(model.named_children())[stage])


def load_model(ckpt):
    m = GradingModelWithCBAM(format="rsna", use_cbam=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(ck.get("model_state_dict", ck),
                                            strict=False)
    if missing or unexpected:
        print(f"  (load: {len(missing)} missing, {len(unexpected)} unexpected)")
    m.eval()
    return m


def similarity(a, b):
    """Spearman rank correlation, plus the fraction of the top decile retained.

    Rank correlation because the absolute scale of a CAM is arbitrary; top-decile
    overlap because that is the part anyone actually looks at.
    """
    rho = spearmanr(a.ravel(), b.ravel()).statistic
    k = max(1, a.size // 10)
    top_a = set(np.argsort(a.ravel())[-k:])
    top_b = set(np.argsort(b.ravel())[-k:])
    return (0.0 if np.isnan(rho) else float(rho),
            len(top_a & top_b) / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbam-checkpoint", type=str,
                    default="checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth")
    ap.add_argument("--data-dir", type=str, default="rsna_preprocessed")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--condition", type=str, default="spinal_canal")
    ap.add_argument("--target-class", type=int, default=2)
    ap.add_argument("--methods", nargs="+",
                    default=["grad_cam", "layer_cam"])
    ap.add_argument("--stage", type=str, default="cbam4")
    ap.add_argument("--n-cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default=str(_HERE / "results" / "sanity_checks.json"))
    args = ap.parse_args()

    md = pd.read_csv(Path(args.data_dir) / f"{args.split}_metadata.csv")
    pos = md[md[args.condition] == args.target_class]
    if pos.empty:
        raise SystemExit(f"no rows with {args.condition}=={args.target_class}")
    rows = pos.sample(n=min(args.n_cases, len(pos)),
                      random_state=args.seed)
    print(f"{len(rows)} cases with {args.condition}=class {args.target_class}, "
          f"stage={args.stage}")

    conditions = ["spinal_canal", "left_foraminal", "right_foraminal"]
    vols = [torch.from_numpy(
        np.load(Path(args.data_dir) / "volumes" / r.filepath)).float()[None, None]
        for r in rows.itertuples()]

    results = {"stage": args.stage, "condition": args.condition,
               "target_class": args.target_class, "n_cases": len(vols),
               "methods": {}}

    for method in args.methods:
        # Baseline maps from the trained model.
        model = load_model(args.cbam_checkpoint)
        base = [S.cam(model, v, args.condition, args.target_class,
                      method, args.stage) for v in vols]

        per_stage = {}
        # Rebuild and re-randomise cumulatively so each row is "output down to
        # here is destroyed".
        for i in range(len(CASCADE)):
            model = load_model(args.cbam_checkpoint)
            for st in CASCADE[:i + 1]:
                randomise_stage(model, st, conditions)
            rhos, tops = [], []
            for v, b in zip(vols, base):
                try:
                    m2 = S.cam(model, v, args.condition, args.target_class,
                               method, args.stage)
                except Exception:
                    continue
                r, t = similarity(b, m2)
                rhos.append(r)
                tops.append(t)
            per_stage["+".join(CASCADE[:i + 1])] = {
                "spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
                "spearman_sd": float(np.std(rhos)) if rhos else float("nan"),
                "top_decile_overlap_mean": float(np.mean(tops)) if tops else float("nan"),
            }
        results["methods"][method] = per_stage

        print(f"\n{method}")
        print(f"  {'randomised':<40s} {'spearman':>10s} {'top10% overlap':>15s}")
        for k, v in per_stage.items():
            print(f"  {k:<40s} {v['spearman_mean']:>10.3f} "
                  f"{v['top_decile_overlap_mean']:>15.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print("\nReading it: a method that PASSES shows correlation falling toward 0 "
          "as more of the model is destroyed. Correlation staying high means the "
          "map does not depend on what the model learned.")


if __name__ == "__main__":
    main()
