# FINAL Review — Peer Reviewer 1 (Methodology & Statistics)

**Venue:** MIWAI 2026
**Paper:** "A Frozen BiomedCLIP Branch as a Cross-Dataset Regularizer for Imbalanced Lumbar Disc Grading and Zero-Shot Label-Space Extension"
**File reviewed:** `paper/lncs_hk252/main.tex` (read-only, 334 lines)
**Review type:** Final independent methodology/statistics review

---

## Recommendation: **ACCEPT (minor revision optional)**

The methodological and statistical reporting in this version is sound and notably careful for a workshop paper. Every quantitative claim I could recompute is internally consistent, and the most common failure mode in this class of paper — overstating significance from few seeds — has been correctly avoided. Remaining items are MINOR polish, none blocking.

## Scores (1–5, 5 best)

| Dimension | Score |
|---|---|
| Soundness of method | 4 |
| Statistical rigor / honesty | 4.5 |
| Internal numerical consistency | 5 |
| Reproducibility | 4 |
| Clarity of claims | 4 |
| **Overall** | **4 / 5** |

---

## Strengths

1. **Numerical consistency is exact.** I independently recomputed the load-bearing aggregates and all match to the stated precision:
   - Table 2 (zero-shot, l.252–263): hybrid all-8 mean F1 = **0.362** ✓, OTS all-8 = **0.394** ✓, disc-morphology hybrid = **0.587** ✓, OTS disc-morphology = **0.433** ✓, disc-morphology hybrid AUC = **0.812** ✓, all-8 hybrid AUC = **0.733** ✓. The "Mean (disc-morphology)" sub-row and the "Mean (all 8)" row are both honest arithmetic means of their respective per-label cells.
   - RSNA Severe ratios (l.238): F1 0.152→0.356 = 2.34× (stated "2.3×") ✓; recall 12.3→48.6 = 3.95× (stated "about 4×") ✓; fusion gain 0.527−0.482 = 0.045 = "4.5-point" (l.236) ✓; Mean Acc drop 81.0−72.4 = 8.6 (stated "8.6-point", l.238) ✓.
   - Cross-section agreement (abstract l.56 ↔ intro l.70–73 ↔ Table 1 ↔ §4.2 ↔ Summary l.323): Severe recall 12.3%→48.6%, Severe F1 0.152→0.356, Severe AUC 0.899, Mean F1 0.527, SPIDER 0.653 vs 0.646 vs 0.619, CBAM Δ −0.027, hybrid recover +0.034 — all consistent everywhere they appear.

2. **The descriptive-effect-size framing is now stated correctly.** The "≥2σ" / "pooled seed standard deviation" language (abstract l.56, intro l.70, Table 3 caption l.277, footnote l.295, discussion l.314) is consistently and explicitly labelled a *descriptive effect size, not a significance test* (footnote `$^\ast$`, l.295). This is the right thing to do with n=3 seeds and is exactly how it should be phrased. The pooled σ ≈ 0.012–0.013 claim is reproducible: RMS of the four F1 stds = 0.0092, and the |Δ|/σ = 3.8 for CBAM-vs-SpineNetV2 reproduces from a Spine+CBAM pooled σ ≈ 0.0072 (0.027/0.0072 = 3.7–3.8) ✓. The Hybrid–CBAM Δ=+0.034 indeed exceeds 2σ on any reasonable pooling. Parity claim (Hybrid−SpineNetV2 = +0.006 < 1σ, l.298) is correct and appropriately *not* spun as a win.

3. **The trivial-ensemble control (§4.5, l.314) is fair and on-point.** Averaging SpineNetV2 + BMC-only softmax (0.45 / 0.18) vs learned hybrid (0.53 / 0.36) is the correct null model to rule out "fusion is just logit-averaging," and the authors honestly report that the ensemble *wins* on AUC/AUPRC (ranking) while losing on thresholded Severe F1 — a nuanced, non-cherry-picked statement.

4. **Reproducibility basics are present:** patient-level 80/20 split with `random_state=42` (l.182), explicit val counts (1,942 IVDs/395 patients RSNA; 234 IVDs/34 patients SPIDER transfer; 218-patient SPIDER), prompt template disclosed verbatim with the a-priori/no-leakage statement (l.242), temperature now explicitly stated (logit scale **clamped at 100**, l.158 + eq. cosine), optimizer/lr/batch/epochs/early-stopping (l.175), metric definitions including macro-averaging rationale and the AUPRC-vs-AUC justification (l.204).

---

## Weaknesses (numbered, with severity)

1. **[MINOR] AUPRC "7.9× over chance" uses a baseline (0.042) that disagrees with the prevalence stated elsewhere (~0.05).**
   §4.2 l.238: "Severe AUPRC of 0.333 against the 0.042 prevalence baseline is a 7.9× gain." But l.204 and l.182 both state Severe prevalence "about 0.05." 0.333/0.05 = 6.7×, not 7.9×. Either the 0.042 is the actual measured val-split Severe prevalence (in which case state it as such and reconcile with the "~5%" rounding) or the multiplier should be ~6.7×. This is the only place where two stated numbers are in mild tension. Recommend: report the exact val Severe prevalence once and derive the multiplier from it.

2. **[MINOR] Same label name carries different numbers across experiments — risk of reader confusion, not an error.** "Disc herniation" appears as zero-shot F1 0.563 (Table 2, l.253) and as supervised-transfer recall 35.6%±7.9 / 0.36±0.08 (intro l.72, §4.6 l.301); "spondylolisthesis" appears as zero-shot F1 0.028 with AUC 0.807 (l.260) and as transfer recall 30% (l.301). These are genuinely different experiments (zero-shot cosine vs 15-epoch head-tuning), and the AUC-vs-F1 divergence is even used as a calibration argument (l.268) — so the numbers are *correct*, but a reader skimming will see "spondylolisthesis 0.028" and "spondylolisthesis 30%" and suspect a typo. Add one clause at first mention in §4.6 ("under supervised transfer, not the zero-shot setting of Table 2").

3. **[MINOR] Mean (all 8) is an unweighted mean over labels of very different class counts and difficulty (5-class Pfirrmann and 4-class Modic averaged in with binary labels).** This is defensible and the paper does the right thing by *also* reporting the disc-morphology sub-mean. But the all-8 mean mixes macro-F1 values whose chance levels differ wildly (binary ~0.33–0.5 vs 5-class ~0.1), so the single number is only loosely interpretable. The paper already mitigates this verbally (l.268); one sentence noting that the all-8 mean is a coarse summary dominated by the binary labels would close the gap. Does not affect any conclusion.

4. **[MINOR] n=3 seeds is thin and the paper leans on it for its headline causal story.** The authors are commendably honest (footnote l.295; limitation l.318 promises bootstrap CIs + more seeds). With 3 seeds the pooled-σ estimate is itself noisy, so |Δ|/σ = 3.8 should not be read as a hard threshold. The current "descriptive effect size" wording already guards against over-reading; I would only ask that the Summary (l.323) not let "shows CBAM … degrading cross-dataset F1" harden into a causal claim — keep the "descriptive" hedge attached. Severity is minor precisely because the framing is already correct in §3.5/§4.4.

5. **[MINOR] Trivial-ensemble control is reported on "two-seed mean" while the main tables are 3-seed (l.314).** Minor inconsistency in seed budget for the control; acknowledge it (it is a sanity check, so 2 seeds is acceptable) but state explicitly that the 0.53/0.36 hybrid comparison figures used for the contrast are the 3-seed means rounded (0.527→0.53, 0.356→0.36) so readers don't think the hybrid was also re-run at 2 seeds.

6. **[MINOR] Zero-shot Table 2 reports no variability.** Table 1 and Table 3 are mean±std over 3 seeds, but Table 2 (zero-shot) gives point estimates only. Zero-shot inference with a frozen model and fixed prompts is deterministic given one trained checkpoint, so this is fine — but since the *hybrid* checkpoint itself has 3 seeds, the zero-shot row could in principle show seed spread. A one-line note ("zero-shot evaluated on the seed-42 hybrid checkpoint; per-seed spread reported in supplement") would tidy reproducibility.

---

## Items explicitly checked and found OK (no action needed)

- Seeds {42, 123, 456} stated consistently in both Table 1 (l.214) and Table 3 (l.277) captions and abstract.
- Bolding rule in Table 3 ("best per row when gap > 2σ", l.277) is applied consistently: Mean F1 and Mean AUC hybrid carry `$^\ast$` (not bold-as-winner over SpineNetV2, correct since parity); Mean Accuracy and Mean AUPRC bold SpineNetV2 where it leads beyond noise — internally consistent with the stated rule.
- "more than twice the pooled seed standard deviation" (abstract/intro/discussion) reconciles with Δ=0.034 and σ≈0.012–0.013 (2σ≈0.018–0.026): 0.034 > 2σ holds. ✓
- Parameter budget (1.18M trainable / ~218M total, l.158) and architecture dims (W1 768×1024, W2 512×768, d=512, l.147–150) are internally consistent with concat of two 512-d branches into 1024.
- Eq. (3) CBAM, Eq. (4) attention pool, Eq. (7) cosine with temperature s — notation consistent; the zero-shot/supervised settings differ only in head c, as claimed in §3.1.

---

## Summary for editor

A careful, honest empirical paper. The statistics are reported with exactly the restraint n=3 seeds demands — descriptive effect sizes, no inferential overclaiming, an appropriate ensemble null model, and exact internal arithmetic. I found **zero** load-bearing numerical contradictions; the only genuine number tension is the AUPRC baseline 0.042-vs-0.05 (Weakness 1), which is cosmetic. Accept; the six MINOR items are optional polish.
