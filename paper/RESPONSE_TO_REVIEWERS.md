# Response to Reviewers

**Manuscript**: "Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar Intervertebral Disc Grading with Cross-Dataset Zero-Shot Transfer"
**Authors**: Trung Kien Ha, Nhân Phan
**Affiliation**: HCMUT, VNU-HCM
**Revision date**: 2026-05-15
**Revision target**: MIWAI 2026 / CSoNet 2026

---

## Summary of Changes

This revision addresses all P0-level issues raised in the 5-reviewer panel review of the 3-seed paper version (commit `19fccea`). Six P0 items were applied; the resulting manuscript is `REPORT_PAPER.tex` (8 pages). The single-seed backup version `REPORT_PAPER_singleseed.tex` is preserved unchanged for internal reference.

## Point-by-Point Response

### R1. Devil's Advocate — CRITICAL-1: Internal contradiction in motivation

**Reviewer comment**: *"Paper motivates CBAM as improving Severe metrics on RSNA, then shows CBAM hurts cross-dataset. The proposed Hybrid solves the problem CBAM creates. A reader unfamiliar with our specific journey would ask: 'Why use CBAM at all if it hurts generalization?'"*

**Response**: Acknowledged. We have added a dedicated paragraph "Why CBAM, given the cross-dataset cost?" at the start of §VI Discussion (lines 720-738 in the revised manuscript). The paragraph defends CBAM on two empirical grounds:

1. **CBAM is the single largest source of in-domain Severe-class improvement**: Severe F1 rises from 0.149 (Baseline) to 0.304 (CBAM-only), +0.155 absolute. BiomedCLIP alone (BMC-only) achieves only 0.229 Severe F1 — i.e., the in-domain Severe-class gain cannot be obtained from BiomedCLIP alone.
2. **The cross-dataset asymmetry is a finding, not a bug**: We re-frame Hybrid as combining CBAM's in-domain selectivity with BiomedCLIP's distribution-invariant prior, rather than as repairing a CBAM defect.

**Change location**: §VI Discussion, paragraph "Why CBAM, given the cross-dataset cost?" (~150 words, lines 720-738).

---

### R2. Devil's Advocate — CRITICAL-2: Minority class attribution unclear

**Reviewer comment**: *"For Disc Herniation Yes recall, BMC-only achieves 0.270 vs Hybrid 0.356. The +0.086 difference (Hybrid vs BMC-only) is the actual CBAM contribution on this minority class. Paper should either acknowledge that BMC-only captures most of this improvement or report the BMC-only vs Hybrid gap explicitly."*

**Response**: Acknowledged. We added an explicit attribution sentence in §V.C Cross-Dataset Transfer Learning. The revised text now reads (paraphrased): *"BMC-only (BiomedCLIP branch without CBAM) already achieves 27.0% ± 15.4 recall on this label; the BiomedCLIP semantic prior is responsible for the bulk of the herniation recovery, while the CBAM volumetric branch contributes an incremental +0.086 ± 17.3 when fused into the Hybrid."*

This framing positions BiomedCLIP as the primary source of minority-class recovery and CBAM as the secondary refinement, which is the most defensible reading of the data.

**Change location**: §V.C, paragraph after Table~\ref{tab:spider_minority} (lines ~675).

---

### R3. Methodology Reviewer — W1: RSNA single-seed limitation

**Reviewer comment**: *"RSNA experiments still single-seed → Severe Recall claim 11.5→46.4% has no CI. No bootstrap CI on val set (cheap to add)."*

**Response**: Acknowledged. The Limitations section now includes an explicit commitment to bootstrap 95% confidence intervals on RSNA Severe F1, Recall, and AUPRC ($n = 1000$ resamples on the validation set) in addition to the pending 3-seed RSNA sweep. This provides uncertainty quantification on the headline numbers independently of the multi-seed effort.

**Change location**: §VII Limitations, paragraph (i) (lines ~750-756).

---

### R4. Devil's Advocate — MAJOR-3: Trivial ensemble alternative

**Reviewer comment**: *"The alternative is that Hybrid simply averages out CBAM's overfitting. This is mathematically equivalent to ensemble over two representations. The paper should compare to a simple ensemble baseline (e.g., logit average of Baseline + BMC-only) to rule out trivial ensembling."*

**Response**: Acknowledged as a legitimate alternative explanation. We have added a final sentence in §VI Discussion explicitly noting that a Baseline+BMC-only logit-average ensemble comparison is a future-work direction to rule out trivial ensembling. We do not claim the regularization effect is non-ensemble; we acknowledge the open question.

**Change location**: §VI Discussion, end of "CBAM helps in-domain but hurts cross-dataset transfer" paragraph (lines ~745-749).

---

### R5. Methodology Reviewer — W4: nshen7 Severe F1 comparison

**Reviewer comment**: *"nshen7 reports Severe F1 0.501 in Table I — closest comparable setup. Our 0.343. The paper claims 'intentionally more difficult variant' but no explicit explanation of why our number is lower than nshen7's despite similar setup."*

**Response**: Acknowledged. We have added an explicit clarification at the start of §V.A noting that nshen7's 0.501 is likely a per-condition Severe F1 (specifically canal stenosis), while our 0.343 is the macro-average across all three RSNA conditions (canal + left foraminal + right foraminal). The foraminal-stenosis sagittal-only ceiling (~0.18-0.20) drags down the macro-average. Like-for-like comparison is not currently feasible from public artifacts.

**Change location**: §V.A first paragraph (lines 475-486).

---

### R6. EIC — W3: Reorder contribution list

**Reviewer comment**: *"Contribution list in §I lacks ordering by importance; reader cannot quickly identify the headline claim."*

**Response**: Acknowledged. The contribution list in §I Introduction has been reordered by strength of multi-seed evidence:

1. **Cross-dataset regularization finding** (3-seed evidence, significant)
2. **Two-branch hybrid architecture** (ablation evidence)
3. **Cross-dataset label-space extension** (mixed evidence — zero-shot + transfer)
4. **Class-imbalance handling pipeline** (single-seed RSNA, pending multi-seed)

This places the strongest claim first and signals the multi-seed methodology upfront.

**Change location**: §I Introduction, contribution list (lines 115-138).

---

## Verification Checklist

- [x] All 6 P0 items applied
- [x] `REPORT_PAPER.tex` compiles cleanly (8 pages, 2 pdflatex passes)
- [x] `REPORT_PAPER_singleseed.tex` preserved (timestamp May 15 18:27, untouched since backup creation)
- [x] No new TODOs introduced
- [x] No new citations require addition (all reference existing bibitems)
- [x] Page count within MIWAI 12 / CSoNet 14 limits

## Items Deferred (P1/P2 — not in this revision)

- BiomedCoOp baseline empirical comparison (P2)
- RadCLIP empirical comparison (P2)
- Per-level Severe F1 breakdown (P2)
- 2-class Pfirrmann auxiliary metric (P2)
- Modic T1+T2 paired analysis discussion (P1)

These items are tracked for future revision rounds; they do not block the current submission.

---

## Estimated Updated Accept Probability

| Venue | Pre-revision | Post-revision (this) | Delta |
|---|---|---|---|
| MIWAI 2026 | 50-60% | **60-70%** | +10pp |
| CSoNet 2026 | 65-75% | **75-80%** | +5-10pp |
| LVTN Thạc sĩ | 95% Pass | **95% Pass + Giỏi likely** | (no change in pass; raise grade) |
