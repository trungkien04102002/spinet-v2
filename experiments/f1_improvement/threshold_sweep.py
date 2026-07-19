#!/usr/bin/env python3
"""
Experiment #0 (post-hoc threshold/calibration) — Step 3: post-hoc decision-rule
tuning on ALREADY-DUMPED probabilities/logits from `dump_logits.py`. No
training, no gradient steps — this only changes how we turn a frozen model's
probabilities into a class prediction.

WHY: RSNA's official error weights are Normal-Mild:Moderate:Severe = 1:2:4,
so plain argmax (which implicitly assumes equal cost for every class) is the
wrong decision rule for a model whose Severe AUC (ranking quality, ~0.899 for
Hybrid) is much better than its Severe F1 (thresholded decision quality,
~0.356) — a classic "good ranker, wrong operating point" gap.

Three post-hoc methods, BEFORE (argmax) vs AFTER:

  (a) Per-class decision-weight optimization via coordinate ascent.
      Score_c(x) = w_c * p_c(x); prediction = argmax_c Score_c(x).
      w is fit per condition by coordinate ascent on macro-F1 (grid search
      each w_c holding the others fixed, repeat until no improvement).
      This is the general form of "moving the classification threshold" for
      a >2-class softmax head (a per-class multiplicative decision weight is
      equivalent to shifting the log-odds boundary between classes).

  (b) Post-hoc logit adjustment (Menon et al., ICLR 2021 "Long-tail learning
      via logit adjustment"): score_c(x) = log p_c(x) - tau * log(pi_c),
      where pi_c is the TRAINING-set class prior (recomputed from the same
      patient split dump_logits.py used — never touches val labels, so this
      is not circular). log p_c is used as a stand-in for the raw logit: for
      a fixed x, log p_c(x) = logit_c(x) - logsumexp_c(logit(x)), and the
      subtracted term is a per-x constant that cancels out of any argmax
      comparison across classes, so the method is exact from probabilities
      alone (no need for the pre-softmax logits here).

  (c) Tau-normalization (Kang et al., ICLR 2020 "Decoupling representation
      and classifier"): rescale the classifier's decision by the inverse
      norm of each class's weight vector, ||w_c||^-tau. This DOES need
      access to the trained classifier weights (not just probabilities):
        - CBAM checkpoint: real linear heads (fc_spinal_canal etc, [3,512]
          weight matrices) -> ||w_c|| differs per class -> tau-norm is a
          real, distinct correction from (a)/(b). We load the checkpoint
          only to read the weight norms (no forward pass needed since we
          already have the dumped raw logits for CBAM), so this method is
          only run when raw logits + a CBAM state dict are both available.
        - Hybrid checkpoint: classification is cosine similarity against
          BiomedCLIP TEXT-PROMPT embeddings, which are L2-normalized by
          construction (||w_c|| = 1 for every class, every condition).
          Tau-normalization's entire mechanism (correcting classifiers whose
          minority-class weight vectors shrank during training) therefore
          CANNOT exist in this architecture — it is a mathematical no-op.
          We detect this and report it explicitly rather than fabricate a
          number; method (b) is the relevant post-hoc lever for a
          cosine-similarity / CLIP-style head.

Overfitting caveat: we only have ONE val split (seed 42, ~20% of patients).
Method (a) is fit and evaluated on it, which will look better than a true
held-out number. We report BOTH:
  - "in-split" (fit == eval on the full val set) — optimistic, upper bound.
  - "2-fold group cross-val within val" (split val patients into 2 groups,
    fit weights on group A / eval on group B and vice versa, average) — a
    more honest estimate of how much of the F1 gain is a real decision-rule
    shift vs. val-set-specific overfitting.

Usage:
    python3 experiments/f1_improvement/threshold_sweep.py \\
        --npz experiments/f1_improvement/logits/hybrid_seed42_val.npz \\
        --cbam-checkpoint checkpoints/v3_20260503/cbam_best.pth \\
        --output-dir experiments/f1_improvement/results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]
N_CLASSES = 3
# RSNA-style 1:2:4 error weights, indexed by TRUE class (Normal/Mild, Moderate, Severe).
COST_WEIGHTS = np.array([1.0, 2.0, 4.0])


# --------------------------------------------------------------------------
# Loading + metrics (definitions match scripts/aggregate_rsna_seeds.py so the
# argmax sanity check can be compared 1:1 against SOURCE_OF_TRUTH.md).
# --------------------------------------------------------------------------

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    probs = {c: d[f"probs_{c}"] for c in CONDITIONS}
    has_logits = f"logits_{CONDITIONS[0]}" in d.files
    logits = {c: d[f"logits_{c}"] for c in CONDITIONS} if has_logits else None
    labels = {c: d[f"labels_{c}"] for c in CONDITIONS}
    study_id = d["study_id"] if "study_id" in d.files else None
    meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
    return probs, logits, labels, study_id, meta


def per_class_metrics(labels, preds):
    valid = labels != -1
    labels_f, preds_f = labels[valid], preds[valid]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_f, preds_f, labels=list(range(N_CLASSES)), average=None, zero_division=0
    )
    acc = float((labels_f == preds_f).mean()) if len(labels_f) else float("nan")
    return {
        "precision": precision.tolist(), "recall": recall.tolist(),
        "f1": f1.tolist(), "support": support.tolist(), "acc": acc,
    }


def cost_weighted_error(labels, preds, weights=COST_WEIGHTS):
    """Mean RSNA-1:2:4-weighted misclassification rate:
    sum(w[y_true] * 1[pred!=true]) / sum(w[y_true]) over valid samples.
    0 = perfect: 1 = every sample wrong (weighted-average rate, NOT raw sum).
    """
    valid = labels != -1
    labels_f, preds_f = labels[valid], preds[valid]
    if len(labels_f) == 0:
        return float("nan")
    w = weights[labels_f]
    wrong = (preds_f != labels_f).astype(float)
    return float((w * wrong).sum() / w.sum())


def evaluate_decision_fn(probs_dict, labels_dict, decision_fn):
    """decision_fn(cond, probs[N,3]) -> preds[N]. Returns full metrics dict."""
    per_cond = {}
    for c in CONDITIONS:
        preds = decision_fn(c, probs_dict[c])
        pcm = per_class_metrics(labels_dict[c], preds)
        pcm["cost_weighted_error"] = cost_weighted_error(labels_dict[c], preds)
        per_cond[c] = pcm
    macro_f1 = float(np.mean([np.mean(per_cond[c]["f1"]) for c in CONDITIONS]))
    macro_recall = float(np.mean([np.mean(per_cond[c]["recall"]) for c in CONDITIONS]))
    macro_precision = float(np.mean([np.mean(per_cond[c]["precision"]) for c in CONDITIONS]))
    severe_f1 = float(np.mean([per_cond[c]["f1"][2] for c in CONDITIONS]))
    severe_recall = float(np.mean([per_cond[c]["recall"][2] for c in CONDITIONS]))
    severe_precision = float(np.mean([per_cond[c]["precision"][2] for c in CONDITIONS]))
    mean_acc = float(np.mean([per_cond[c]["acc"] for c in CONDITIONS]))
    cwe = float(np.mean([per_cond[c]["cost_weighted_error"] for c in CONDITIONS]))
    return {
        "per_condition": per_cond,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "severe_f1": severe_f1,
        "severe_recall": severe_recall,
        "severe_precision": severe_precision,
        "mean_acc": mean_acc,
        "cost_weighted_error": cwe,
    }


def argmax_decision(cond, probs):
    return probs.argmax(axis=1)


# --------------------------------------------------------------------------
# (a) Per-class decision-weight coordinate ascent
# --------------------------------------------------------------------------

def _macro_f1_for_weights(probs, labels, weights):
    preds = (probs * weights[None, :]).argmax(axis=1)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(N_CLASSES)), average=None, zero_division=0
    )
    return float(f1.mean())


def fit_weights_coord_ascent(probs, labels, n_rounds=4, grid=None, seed_weights=None):
    """Coordinate ascent: for each class in turn, grid-search its multiplicative
    decision weight (others fixed) to maximize macro-F1; repeat until no class
    improves. Returns (weights[3], best_macro_f1)."""
    valid = labels != -1
    p, y = probs[valid], labels[valid]
    if grid is None:
        grid = np.geomspace(0.1, 15.0, 60)
    w = np.ones(N_CLASSES) if seed_weights is None else seed_weights.copy()
    best = _macro_f1_for_weights(p, y, w)
    for _ in range(n_rounds):
        improved = False
        for c in range(N_CLASSES):
            cur_best_val, cur_best_f1 = w[c], best
            for cand in grid:
                trial = w.copy()
                trial[c] = cand
                f1 = _macro_f1_for_weights(p, y, trial)
                if f1 > cur_best_f1 + 1e-9:
                    cur_best_f1, cur_best_val = f1, cand
            if cur_best_val != w[c]:
                improved = True
            w[c] = cur_best_val
            best = cur_best_f1
        if not improved:
            break
    return w, best


def weighted_decision_fn(weights_by_cond):
    def fn(cond, probs):
        return (probs * weights_by_cond[cond][None, :]).argmax(axis=1)
    return fn


def stability_check_coord_ascent(probs, labels, study_ids, n_splits=2, seed=0):
    """2-fold GroupKFold (grouped by patient) within the val set: fit weights on
    fold A, evaluate on fold B, and vice versa. Returns mean held-out macro-F1
    (honest estimate) vs the in-split macro-F1 (optimistic estimate)."""
    valid = labels != -1
    p, y, g = probs[valid], labels[valid], study_ids[valid]
    if len(np.unique(g)) < n_splits:
        return None  # not enough distinct patients to split
    gkf = GroupKFold(n_splits=n_splits)
    held_out_f1s = []
    for fit_idx, eval_idx in gkf.split(p, y, groups=g):
        if len(np.unique(y[fit_idx])) < 2:
            continue
        w, _ = fit_weights_coord_ascent(p[fit_idx], y[fit_idx])
        f1 = _macro_f1_for_weights(p[eval_idx], y[eval_idx], w)
        held_out_f1s.append(f1)
    if not held_out_f1s:
        return None
    return float(np.mean(held_out_f1s)), held_out_f1s


# --------------------------------------------------------------------------
# (b) Post-hoc logit adjustment (Menon et al. 2020)
# --------------------------------------------------------------------------

def compute_train_class_priors(data_dir, seed, val_split):
    """Class priors from the TRAINING half of the exact same patient split
    dump_logits.py reproduced. Uses only the metadata CSV (no .npy loading
    needed) — fast and touches zero validation labels."""
    df = pd.read_csv(Path(data_dir) / "train_metadata.csv")
    unique_patients = df["study_id"].unique()
    train_patients, _ = train_test_split(unique_patients, test_size=val_split, random_state=seed)
    train_df = df[df["study_id"].isin(train_patients)]
    priors = {}
    for c in CONDITIONS:
        vals = train_df[c]
        vals = vals[vals != -1]
        counts = vals.value_counts().reindex(range(N_CLASSES), fill_value=0).to_numpy().astype(float)
        priors[c] = counts / counts.sum()
    return priors


def logit_adjust_decision_fn(priors_by_cond, tau):
    def fn(cond, probs):
        logp = np.log(np.clip(probs, 1e-12, 1.0))
        adj = logp - tau * np.log(np.clip(priors_by_cond[cond], 1e-12, 1.0))[None, :]
        return adj.argmax(axis=1)
    return fn


# --------------------------------------------------------------------------
# (c) Tau-normalization (Kang et al. 2020) — needs classifier weight access
# --------------------------------------------------------------------------

def load_cbam_fc_weight_norms(cbam_checkpoint_path):
    """Read ||w_c|| per class from the CBAM linear heads. Returns None if the
    checkpoint doesn't have the expected fc_* keys (e.g. a Hybrid checkpoint)."""
    ckpt = torch.load(cbam_checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    name_map = {
        "spinal_canal": "fc_spinal_canal",
        "left_foraminal": "fc_left_foraminal",
        "right_foraminal": "fc_right_foraminal",
    }
    norms = {}
    for cond, fc_name in name_map.items():
        w = state.get(f"{fc_name}.weight")
        if w is None:
            return None
        norms[cond] = w.norm(dim=1).numpy()  # [3]
    return norms


def tau_norm_decision_fn(logits_dict, weight_norms_by_cond, tau):
    def fn(cond, probs_unused):
        logits = logits_dict[cond]
        norms = weight_norms_by_cond[cond]
        scale = norms ** (-tau)
        scale = scale / scale.mean()
        return (logits * scale[None, :]).argmax(axis=1)
    return fn


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def fmt_row(name, res):
    return (f"{name:<38} MeanAcc={res['mean_acc']*100:5.1f}%  "
            f"MacroF1={res['macro_f1']:.3f}  SevereF1={res['severe_f1']:.3f}  "
            f"SevereRec={res['severe_recall']:.3f}  SeverePrec={res['severe_precision']:.3f}  "
            f"CostErr(1:2:4)={res['cost_weighted_error']:.3f}")


def print_per_condition(res):
    for c in CONDITIONS:
        pcm = res["per_condition"][c]
        f1 = pcm["f1"]
        rec = pcm["recall"]
        prec = pcm["precision"]
        print(f"    {c:<16} F1=[N/M {f1[0]:.3f} Mod {f1[1]:.3f} Sev {f1[2]:.3f}]  "
              f"Recall=[N/M {rec[0]:.3f} Mod {rec[1]:.3f} Sev {rec[2]:.3f}]  "
              f"Prec=[N/M {prec[0]:.3f} Mod {prec[1]:.3f} Sev {prec[2]:.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "rsna_preprocessed"))
    ap.add_argument("--cbam-checkpoint", type=str, default=None,
                     help="Needed only for method (c) tau-norm when the dumped "
                          "model IS the CBAM linear-head model.")
    ap.add_argument("--sanity-check-json", type=str, default=None,
                     help="Path to a *_best_metrics.json (e.g. "
                          "checkpoints/v3_20260503/hybrid/hybrid_best_metrics.json) "
                          "to diff the reconstructed argmax metrics against.")
    ap.add_argument("--tau-grid", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    print("=" * 78)
    print("Experiment #0 — threshold_sweep.py (post-hoc, no retraining)")
    print("=" * 78)

    probs, logits, labels, study_id, meta = load_npz(args.npz)
    print(f"Loaded: {args.npz}")
    print(f"Meta: {json.dumps(meta, indent=2)}")
    n_val = len(labels[CONDITIONS[0]])
    print(f"Val samples: {n_val}")
    for c in CONDITIONS:
        vals, counts = np.unique(labels[c][labels[c] != -1], return_counts=True)
        print(f"  {c}: class counts {dict(zip(vals.tolist(), counts.tolist()))}")

    # ---------------- STEP 2: sanity check ----------------
    print("\n" + "=" * 78)
    print("STEP 2 — SANITY CHECK: does argmax on the dumped probs reproduce "
          "known numbers?")
    print("=" * 78)
    before = evaluate_decision_fn(probs, labels, argmax_decision)
    print(fmt_row("ARGMAX (reconstructed)", before))
    print_per_condition(before)

    if args.sanity_check_json and Path(args.sanity_check_json).exists():
        ref = json.load(open(args.sanity_check_json))
        ref_severe_f1 = ref.get("avg_severe_f1")
        ref_acc = {c: ref["val_accuracies"][c] for c in CONDITIONS} if "val_accuracies" in ref else None
        ref_pcm = ref.get("per_class_metrics", {})
        ref_macro_f1 = float(np.mean([np.mean(ref_pcm[c]["f1"]) for c in CONDITIONS])) if ref_pcm else None
        print(f"\nReference file: {args.sanity_check_json}")
        print(f"  Reference Severe F1: {ref_severe_f1:.4f}   Reconstructed: {before['severe_f1']:.4f}   "
              f"diff={abs(ref_severe_f1 - before['severe_f1']):.4f}")
        if ref_macro_f1 is not None:
            print(f"  Reference Macro F1:  {ref_macro_f1:.4f}   Reconstructed: {before['macro_f1']:.4f}   "
                  f"diff={abs(ref_macro_f1 - before['macro_f1']):.4f}")
        if ref_acc is not None:
            for c in CONDITIONS:
                print(f"  Reference Acc[{c}]: {ref_acc[c]:.4f}   Reconstructed: {before['per_condition'][c]['acc']:.4f}")
        tol = 0.02
        ok = abs(ref_severe_f1 - before["severe_f1"]) < tol and (
            ref_macro_f1 is None or abs(ref_macro_f1 - before["macro_f1"]) < tol
        )
        print(f"\n  SANITY CHECK: {'PASS' if ok else 'FAIL'} (tolerance={tol})")
        if not ok:
            print("  STOPPING: reconstructed argmax metrics do not match the known "
                  "reference. Do NOT trust downstream threshold-tuning numbers "
                  "until this is resolved.")
            sys.exit(1)
    else:
        print("\n  (no --sanity-check-json given or file not found — skipping "
              "numeric diff; report the numbers above against SOURCE_OF_TRUTH.md "
              "manually.)")

    results = {"argmax": before}

    # ---------------- (a) coordinate-ascent per-class weights ----------------
    print("\n" + "=" * 78)
    print("(a) Per-class decision-weight coordinate ascent (maximize macro-F1)")
    print("=" * 78)
    weights_by_cond = {}
    for c in CONDITIONS:
        w, f1 = fit_weights_coord_ascent(probs[c], labels[c])
        weights_by_cond[c] = w
        print(f"  {c:<16} fitted weights (Normal/Mild, Moderate, Severe) = "
              f"[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]  in-fold macro-F1={f1:.3f}")
    after_a_insplit = evaluate_decision_fn(probs, labels, weighted_decision_fn(weights_by_cond))
    print("\n  IN-SPLIT (fit==eval on full val; OPTIMISTIC upper bound):")
    print("  " + fmt_row("coord-ascent weights (in-split)", after_a_insplit))
    print_per_condition(after_a_insplit)
    results["coord_ascent_weights_in_split"] = {
        **after_a_insplit,
        "fitted_weights": {c: weights_by_cond[c].tolist() for c in CONDITIONS},
    }

    print("\n  2-FOLD GROUP CROSS-VAL WITHIN VAL (fit on half A / eval on half B, "
          "grouped by patient — HONEST held-out estimate):")
    if study_id is not None:
        cv_macro_f1s = []
        cv_severe_f1s = []
        for c in CONDITIONS:
            r = stability_check_coord_ascent(probs[c], labels[c], study_id)
            if r is None:
                print(f"    {c:<16} (not enough distinct val patients to split — skipped)")
                continue
            mean_f1, per_fold = r
            print(f"    {c:<16} held-out macro-F1 per fold: {['%.3f' % x for x in per_fold]}  mean={mean_f1:.3f}")
            cv_macro_f1s.append(mean_f1)
        if cv_macro_f1s:
            print(f"\n  Held-out mean macro-F1 across conditions: {np.mean(cv_macro_f1s):.3f} "
                  f"(vs in-split {after_a_insplit['macro_f1']:.3f}, vs argmax {before['macro_f1']:.3f})")
        results["coord_ascent_weights_2fold_cv_macro_f1"] = cv_macro_f1s
    else:
        print("    (no study_id in npz — cannot group-split; skipped)")

    # ---------------- (b) logit adjustment ----------------
    print("\n" + "=" * 78)
    print("(b) Post-hoc logit adjustment (Menon et al. 2020): "
          "score_c = log p_c - tau * log(train_prior_c)")
    print("=" * 78)
    priors = compute_train_class_priors(args.data_dir, meta.get("seed", 42), meta.get("val_split", 0.2))
    for c in CONDITIONS:
        print(f"  {c:<16} train priors (N/M, Mod, Sev) = "
              f"[{priors[c][0]:.3f}, {priors[c][1]:.3f}, {priors[c][2]:.3f}]")
    logit_adj_results = {}
    for tau in args.tau_grid:
        res = evaluate_decision_fn(probs, labels, logit_adjust_decision_fn(priors, tau))
        print("  " + fmt_row(f"logit-adjust tau={tau}", res))
        logit_adj_results[str(tau)] = res
    best_tau = max(logit_adj_results, key=lambda k: logit_adj_results[k]["macro_f1"])
    print(f"\n  Best tau by macro-F1: {best_tau}")
    print_per_condition(logit_adj_results[best_tau])
    results["logit_adjustment"] = logit_adj_results
    results["logit_adjustment_best_tau"] = best_tau

    # ---------------- (c) tau-normalization ----------------
    print("\n" + "=" * 78)
    print("(c) Tau-normalization (Kang et al. 2020) — needs classifier weight access")
    print("=" * 78)
    if logits is None:
        print("  SKIPPED: no raw logits in this npz (re-run dump_logits.py — the "
              "current version saves logits_<cond> alongside probs_<cond>).")
    elif args.cbam_checkpoint and meta.get("model") == "cbam":
        norms = load_cbam_fc_weight_norms(args.cbam_checkpoint)
        if norms is None:
            print(f"  SKIPPED: {args.cbam_checkpoint} has no fc_* linear-head weights "
                  "(not a CBAM checkpoint?).")
        else:
            for c in CONDITIONS:
                print(f"  {c:<16} ||w_c|| (N/M, Mod, Sev) = "
                      f"[{norms[c][0]:.3f}, {norms[c][1]:.3f}, {norms[c][2]:.3f}]")
            tau_norm_results = {}
            for tau in args.tau_grid:
                res = evaluate_decision_fn(probs, labels, tau_norm_decision_fn(logits, norms, tau))
                print("  " + fmt_row(f"tau-norm tau={tau}", res))
                tau_norm_results[str(tau)] = res
            best_tau_norm = max(tau_norm_results, key=lambda k: tau_norm_results[k]["macro_f1"])
            print(f"\n  Best tau by macro-F1: {best_tau_norm}")
            print_per_condition(tau_norm_results[best_tau_norm])
            results["tau_normalization"] = tau_norm_results
            results["tau_normalization_best_tau"] = best_tau_norm
    else:
        print("  SKIPPED (mathematically a no-op for this architecture): the dumped "
              "model is Hybrid, whose classification score is cosine similarity "
              "against BiomedCLIP TEXT-PROMPT embeddings. Those prototypes are "
              "L2-normalized by construction, so ||w_c|| = 1 for EVERY class in "
              "every condition — there is no weight-norm imbalance for "
              "tau-normalization to correct. Pass --cbam-checkpoint together with "
              "a CBAM-model npz (--model cbam in dump_logits.py) to run a real "
              "tau-norm experiment instead.")
        results["tau_normalization"] = "skipped_no_op_for_cosine_head"

    # ---------------- BEFORE -> AFTER summary ----------------
    print("\n" + "=" * 78)
    print("SUMMARY: BEFORE (argmax) -> AFTER (best of each method)")
    print("=" * 78)
    print(fmt_row("BEFORE: argmax", before))
    print(fmt_row("AFTER:  coord-ascent (in-split, optimistic)", after_a_insplit))
    print(fmt_row(f"AFTER:  logit-adjust tau={best_tau}", logit_adj_results[best_tau]))
    if "tau_normalization" in results and isinstance(results["tau_normalization"], dict):
        print(fmt_row(f"AFTER:  tau-norm tau={results['tau_normalization_best_tau']}",
                       tau_norm_results[results["tau_normalization_best_tau"]]))

    d_severe_coord = after_a_insplit["severe_f1"] - before["severe_f1"]
    d_severe_logit = logit_adj_results[best_tau]["severe_f1"] - before["severe_f1"]
    d_macro_coord = after_a_insplit["macro_f1"] - before["macro_f1"]
    d_macro_logit = logit_adj_results[best_tau]["macro_f1"] - before["macro_f1"]
    print(f"\nSevere F1 delta:  coord-ascent(in-split) {d_severe_coord:+.3f}   "
          f"logit-adjust {d_severe_logit:+.3f}")
    print(f"Macro F1 delta:   coord-ascent(in-split) {d_macro_coord:+.3f}   "
          f"logit-adjust {d_macro_logit:+.3f}")

    biggest = max(d_severe_coord, d_severe_logit)
    if biggest > 0.08:
        verdict = "BIG WIN"
    elif biggest > 0.03:
        verdict = "MODEST"
    else:
        verdict = "NEGLIGIBLE"
    print(f"\nVERDICT: post-hoc threshold/logit-adjustment recovers Severe F1 by "
          f"up to {biggest:+.3f} (in-split) -> {verdict}.")
    if study_id is not None and results.get("coord_ascent_weights_2fold_cv_macro_f1"):
        cv_macro = np.mean(results["coord_ascent_weights_2fold_cv_macro_f1"])
        print(f"CAVEAT: held-out (2-fold within-val) macro-F1 for coord-ascent is "
              f"{cv_macro:.3f} vs in-split {after_a_insplit['macro_f1']:.3f} vs argmax "
              f"{before['macro_f1']:.3f} — the gap between held-out and in-split is the "
              f"overfitting risk on this small val set.")

    # ---------------- save ----------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{meta.get('model', 'unknown')}_seed{meta.get('seed', 'na')}"
    out_path = out_dir / f"{tag}_threshold_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
