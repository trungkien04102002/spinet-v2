# Thesis Paper Execution Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each phase plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce all experimental results, tables, and comparison reports needed for Kien's Rank-C paper submission and master's thesis defense on spinal MRI grading with CBAM-enhanced SpineNetV2.

**Architecture:** Three sequential phases — each phase builds on the previous and produces a concrete, paper-ready artifact. Phases 1 and 2 can run in parallel; Phase 3 is blocked on advisor confirmation of protocol.

**Tech Stack:** Python 3, PyTorch, pandas, scikit-learn, 3D ResNet34 backbone, CBAM attention, Focal Loss, Linear Probing transfer protocol.

---

## Cross-cutting constraints

These rules apply to all three phases:

1. **Two-repo layout** — the evaluation infrastructure lives in `/Users/kienha/thesis-experiments/` (read-only for third-party model predictions + GT); the training code lives in `/Users/kienha/spinet-v2/` (CBAM, Focal, SPIDER transfer). Never mix code between them — the `THESIS_CONTEXT.md` in `thesis-experiments/` makes this explicit.
2. **Binary aggregation rule** (from `detect-abnormal/CLAUDE.md`): severity → binary via `Normal/Mild → 0`, `Moderate|Severe → 1`. Patient-level via ANY-rule across 5 vertebral levels. Apply the same rule when adding CBAM results so comparison is fair.
3. **Inner join study_id across all models** — do NOT report metrics on disjoint study sets. The existing `evaluate_detection_abnormal.py` already does this; extending it to include CBAM must preserve the join.
4. **3 conditions only** — `spinal_canal`, `left_foraminal`, `right_foraminal`. Subarticular is out-of-scope for the paper because SpineNetV2 has no Subarticular output.
5. **English in all written files; Vietnamese only in chat** — per project convention.
6. **Git commits** — each phase commits to a feature branch off `main` (or `train-attention` for spinet-v2). No force-push, no amend of pushed commits.

---

## Phase overview

### Phase 1 — Third-party baseline comparison (independent)

**Blocker:** None. Infra exists. Can start immediately.

**Goal:** Produce a single paper-ready comparison table + markdown report quantifying SpineNetV2 (vanilla) vs MedGemma vs Ning Shen on RSNA 2024, evaluated against ground truth. Per-condition precision / recall / F1 / accuracy + confusion matrices + statistical significance tests (McNemar pairwise).

**Artifact:** `/Users/kienha/thesis-experiments/detect-abnormal/outputs/baseline_comparison_report.md` + supporting CSVs.

**Detailed plan:** [2026-04-20-phase1-baseline-comparison.md](2026-04-20-phase1-baseline-comparison.md)

**Estimated effort:** 1 day (scripting + result inspection). No GPU required.

---

### Phase 2 — CBAM + Focal Loss experiments on RSNA (independent)

**Blocker:** None. RSNA checkpoints already exist.

**Goal:** Re-evaluate existing CBAM+Focal checkpoint under the same protocol as Phase 1 so CBAM can be added as a 4th column to the comparison table. Also perform the minimum ablation required for a Rank-C paper.

**Artifact:** `/Users/kienha/spinet-v2/experiments/cbam_rsna_results.md` + `/Users/kienha/thesis-experiments/detect-abnormal/prediction/spinenetv2_cbam_result.csv`.

**Ablation matrix** (4 variants × 3 seeds = 12 runs, ~8 GPU-hours on single 3090):
- baseline (no CBAM, CE loss)
- +CBAM only (CBAM, CE loss)
- +Focal only (no CBAM, Focal loss)
- +CBAM+Focal (full) ← the checkpoint you already have

**Detailed plan:** [2026-04-20-phase2-cbam-experiments.md](2026-04-20-phase2-cbam-experiments.md)

**Estimated effort:** 2–3 days (GPU training) + 1 day (analysis).

---

### Phase 3 — SPIDER Linear Probe external validation (blocked on advisor)

**Blocker:** Waiting for advisor Nhan Phan to confirm protocol interpretation ("zero-shot" = Linear Probing).

**Goal:** Validate that CBAM-enhanced features trained on RSNA transfer to SPIDER's different label space, via Linear Probing (freeze backbone+CBAM, train only new heads).

**Artifact:** `/Users/kienha/spinet-v2/experiments/spider_linear_probe_results.md`.

**Runs required:**
- CBAM + RSNA checkpoint + Linear Probe (freeze-backbone) on SPIDER (3 seeds)
- Baseline + RSNA checkpoint + Linear Probe on SPIDER (3 seeds)
- Optional: Full Fine-tune variants if advisor requests

**Detailed plan:** [2026-04-20-phase3-spider-linear-probe.md](2026-04-20-phase3-spider-linear-probe.md)

**Estimated effort:** 1–2 days (Linear Probe is fast, <1 hour per run).

---

## Execution order

1. **Parallel start:**
   - Phase 1 (no GPU needed, can run on laptop)
   - Phase 2 first training runs (start on GPU machine immediately)

2. **After Phase 1 + 2 complete:**
   - Produce unified comparison table: SpineNetV2-vanilla | MedGemma | NingShen | SpineNetV2-CBAM | SpineNetV2-CBAM+Focal
   - This is the "Table 1" of the paper.

3. **Phase 3 starts when advisor confirms.**
   - If advisor says "Linear Probe only": run 2 setups × 3 seeds.
   - If advisor says "also Full Fine-tune": add 2 more setups × 3 seeds.

## Definition of done for paper readiness

A phase is "done" when:
- Its artifact file exists at the documented path.
- All tables have per-class precision/recall/F1 (not just accuracy).
- Statistical tests are included where comparing two methods (McNemar for paired binary predictions).
- Git is clean; all scripts + results committed.
- The artifact can be dropped into the paper's Results section with minimal editing.

## Out of scope for this roadmap

Explicitly deferred to a future paper or future work section:
- Human-in-the-loop feedback / active learning
- Coordinate Attention (CA), SENet, or other attention variants
- Multi-scale feature fusion
- Modic changes classification on SPIDER
- Subarticular stenosis

Do not expand scope mid-execution. If the advisor or reviewer asks about any of these, cite them as future work.
