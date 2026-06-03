# Re-Review (Round 1) — Methodology & Statistics

**Reviewer 1 — Methodology & Statistics** (MIWAI 2026, LNAI, double-blind)
**Paper:** Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar IVD Grading and Cross-Dataset Zero-Shot Transfer
**Type:** Re-review of revised manuscript against the four first-round concerns (C1–C4).

## Recommendation
**Accept with Minor Revision.** The revision is a substantial and genuine improvement. The two CRITICAL first-round issues (single-seed headline; invalid significance language) are now resolved: RSNA Table 1 is 3-seed mean±std, and every cross-config claim is explicitly demoted to a *descriptive effect size* rather than a significance test. The honest disclosure that off-the-shelf BiomedCLIP beats the hybrid on the zero-shot mean is a credit to the authors and, handled correctly, does not sink the contribution. What remains is a small set of internal-consistency slips (one of which is a real number mismatch) and a few residual statistical-framing nits — all addressable in a minor revision without new experiments.

## Scores (0–100)
| Dimension | R0 | RR1 | Δ |
|---|---|---|---|
| Rigor | 42 | 70 | +28 |
| Statistical Validity | 35 | 66 | +31 |
| Reproducibility | 55 | 64 | +9 |
| Evaluation Design | 68 | 74 | +6 |
| **Overall** | **46** | **69** | **+23** |

Rigor and Statistical Validity rise sharply because the n=1 headline (the single biggest cap in R0) is gone and the "2σ = significant" language has been correctly reframed. They stop short of >75 only because of the residual number mismatch (W1) and the fact that n=3 still anchors the central regularizer claim (W4).

---

## Verification of the four first-round concerns

### C1 — Single-seed RSNA headline → 3-seed mean±std. **CLOSED.**
Table `tab:main` (lines 212–234) is now "mean ± std over 3 seeds {42, 123, 456}" (caption line 214), and the abstract (line 56), contributions (line 73), and Summary (line 322) all consistently say "three-seed mean / across three seeds." The std values are reported for all nine metrics. This directly retires R0-W1 (CRITICAL). The std magnitudes are plausible and informative — see W1/W2 below for the one number that did *not* get propagated and for a comment on std heterogeneity.

### C2 — Trivial-ensemble confound → real logit-averaging control. **CLOSED.**
§4.5 (line 313) now reports an actual control: "averaging the SpineNetV2 and BMC-only softmax outputs reaches only Mean F1 0.45 and Severe F1 0.18 (two-seed mean), well below the learned-fusion hybrid (0.53 and 0.36)." This is the exact experiment R0-W5 asked for, and it now empirically separates the learned concat-MLP fusion from a trivial logit average. The mechanistic reasoning is statistically coherent (see W3 — it is sound, with one caveat on rigor of the n=2 control).

### C3 — Zero-shot prompts undisclosed → format shown + fixed a priori. **CLOSED (lightly).**
§4.3 (line 242) now gives the prompt template verbatim — `"a magnetic resonance image of [label]"` with the binary example `"lumbar disc herniation"` vs `"no disc herniation"` — and states the prompts were "written a priori without tuning on SPIDER, so no target-label leakage occurs." This resolves the leakage concern (the most important half of R0-W6). Two minor gaps persist: the temperature `s` value is still not numerically given (Eq. line 155 / line 158 still says only "inherited from BiomedCLIP pretraining"), and the prompt *sets* for the multiclass labels (Pfirrmann 5-class, Modic 4-class) are not enumerated — see W6.

### C4 — Off-the-shelf BiomedCLIP comparison prose-only → real "OTS F1" column. **CLOSED, and improved by honesty.**
Table `tab:spider` (lines 244–265) now has a numeric **OTS F1** column for all eight labels plus the mean, replacing the unverifiable prose claim flagged in R0-W9. I independently recomputed both column means from the eight per-label values: OTS = 0.3935 (paper: 0.394 ✓) and Hybrid = 0.3622 (paper: 0.362 ✓). The authors also disclose (line 246 caption) that the table was "recomputed from cosine predictions," which addresses the reproducibility note that the old Table 2 numbers were not reproducible. The bolding rule ("higher F1") is correctly applied per row. This is now solid and, crucially, honest — the OTS mean (0.394) **exceeds** the hybrid (0.362), and the paper says so plainly. Whether that undermines the contribution is treated in W5; my judgment is **no**, provided the abstract is tightened.

**Bottom line:** all four concerns are adequately closed. The remaining items below are minor/major-polish, not blockers.

---

## Strengths (now solid)
- **3-seed RSNA reporting is internally consistent across abstract, contributions, Table 1, and Summary** — except for the single line-93 slip in W1. The Severe headline (12.3% → 48.6% recall; 0.152 → 0.356 F1; AUC 0.899) matches Table 1 exactly (lines 224/229/230 vs 56/73/238).
- **Statistical honesty is exemplary for a student paper.** The footnote at line 294 explicitly states "With three seeds these are descriptive effect sizes, not significance tests," distinguishes the <1σ Hybrid–SpineNetV2 parity from the >2σ Hybrid–CBAM gap, and the abstract (line 56) repeats "reported as a descriptive effect size rather than a significance test." This is exactly the fix R0-W2 demanded.
- **The OTS-vs-Hybrid disclosure (line 262, 267) is scientifically correct and self-critical.** Reporting that the baseline you are arguing for *loses on the headline mean* and re-scoping the claim to "selective transfer" is the opposite of cherry-picking.
- **Ranking-vs-thresholding decomposition is genuinely correct.** Lines 267 and 313 use the AUC-high / F1-low pattern (spondylolisthesis AUC 0.807, F1 0.03; Pfirrmann AUC 0.687, F1 0.16) to argue the embedding *orders* cases well while a fixed cosine threshold mis-calibrates the decision. This is the textbook-correct interpretation of a high-AUC/low-F1 divergence and is consistent throughout.
- **Evaluation metric selection remains sound** (macro over C=3, AUPRC alongside AUC with prevalence baseline stated, line 204) — carried over correctly from R0.
- **The logit-averaging control is the right experiment and it lands the right way** (ensemble < learned fusion on F1), earning the "not reducible to a logit average" claim (line 313).

---

## Weaknesses (residual)

### W1 — Number mismatch: Related Work cites the OLD single-seed RSNA figures (Mean F1 0.528, Severe F1 0.343). **CRITICAL (consistency).**
- **Issue:** Line 93 states the paper's own result as "(Mean F1 0.528, Severe F1 0.343)." These are the *pre-revision single-seed (seed-42)* values. The revised 3-seed Table 1 reports **Mean F1 0.527 ± 0.027** (line 223) and **Severe F1 0.356 ± 0.017** (line 229). So the Related-Work self-citation disagrees with the paper's own Table 1 in two places: 0.528 vs 0.527 (cosmetic, 1 LSD) and **0.343 vs 0.356 (material — a 0.013 gap, i.e., ~0.8σ, and on the wrong side: line 93 understates the revised Severe F1)**. This is a leftover from the C1 edit: the table and headline were updated to 3-seed means, but the embedded number in the positioning paragraph was not. A methodology reviewer will treat any internal number that disagrees with the primary table as a provenance red flag.
- **Location:** Line 93 ("Mean F1 0.528, Severe F1 0.343") vs Table 1 lines 223, 229.
- **Fix:** Change line 93 to "Mean F1 0.527, Severe F1 0.356" to match the 3-seed Table 1. Grep the whole manuscript for `0.528` and `0.343` to confirm no other stragglers.
- **Severity: CRITICAL** (it is a one-token fix, but it is a direct number contradiction in the submitted text, and it sits in the load-bearing "our setup is the harder variant" sentence).

### W2 — std heterogeneity across configs is large and unremarked; plausibility is mixed. **MINOR.**
- **Issue:** The std values are plausible *in magnitude* but strikingly *heterogeneous* across columns, which the text never flags. In Table 1 the CBAM-only column has 3–8× the std of the others on several rows: Mean F1 ±0.057 (vs Hybrid ±0.027, SpineNetV2 ±0.010), Mean Recall ±0.068, Severe F1 ±0.082, Severe AUPRC ±0.099 (lines 223–231). A per-seed std of ±0.099 on Severe AUPRC with n=3 means one seed is an outlier; the "best" markings and any cross-config comparison involving CBAM-only rest on a column whose variance is itself poorly estimated (df=2). This is not wrong to report, but the paper draws a sharp in-domain conclusion ("both branches lift Severe F1 to a similar degree — CBAM-only 0.262, BMC-only 0.268," line 313) from columns with ±0.082 and ±0.034 std respectively — i.e., those two means are statistically indistinguishable and the paper correctly does not over-separate them, but it also does not acknowledge the instability.
- **Location:** Table 1 lines 223–231; discussion line 313.
- **Fix:** One sentence noting CBAM-only shows the highest seed-to-seed variance (consistent with the overfitting story the paper already tells) would turn a latent weakness into supporting evidence. Optionally report median or min/max for the high-variance rows.
- **Severity: MINOR.**

### W3 — The C2 control is sound but rests on n=2, and is reported to coarse precision. **MINOR.**
- **Issue:** The logit-averaging control (line 313) is "two-seed mean" (0.45 / 0.18) compared against the 3-seed hybrid (0.53 / 0.36, rounded). The *direction* of the conclusion is safe — the ensemble (0.45) is ~3 std below the hybrid (0.527±0.027) on Mean F1 and ~10 std below on Severe F1, so n=2 vs n=3 does not threaten the qualitative result. But (a) the control uses fewer seeds than the table it is compared against, and (b) it is reported to two decimals (0.45/0.18/0.53/0.36) while Table 1 uses three, so the reader cannot tell whether 0.53 means 0.527 or a separately-computed 0.53. The "ensemble attains higher AUC/AUPRC (better ranking, worse thresholded Severe decisions)" claim (line 313) is statistically coherent and is the same ranking-vs-thresholding logic used correctly in §4.3 — but no AUC/AUPRC *numbers* are given for the ensemble, so this specific sentence is currently an unverifiable assertion (it is the same shape of gap R0-W9 flagged for OTS, now reintroduced in the ensemble control).
- **Location:** Line 313.
- **Fix:** Report the ensemble's actual AUC/AUPRC (the two numbers behind "higher AUC/AUPRC"), note the n=2 vs n=3 asymmetry in one clause, and either round the hybrid 0.53/0.36 to match Table 1 (0.527/0.356) or cite Table 1 directly.
- **Severity: MINOR** (the conclusion holds regardless; this is about verifiability, not correctness).

### W4 — The central "regularizer" claim still rests on a 0.034 effect at n=3; the σ-multiple is correctly labeled descriptive but the magnitude is overstated by the chosen denominator. **MAJOR.**
- **Issue:** The headline cross-dataset claim — Hybrid recovers above CBAM by +0.034, "more than twice the pooled seed standard deviation" (abstract line 56, contribution line 70, Table footnote line 294, discussion lines 297/313) — is now *correctly* framed as a descriptive effect size, which closes R0-W2. However, two residual statistical points remain:
  1. **The arithmetic of "more than twice" is on the edge.** With pooled σ ≈ 0.012–0.013 (footnote line 294), Δ=0.034 gives Δ/σ = 2.62 (at σ=0.013) to 2.83 (at σ=0.012) — i.e., "more than 2×" is true but barely, and is sensitive to which pooled σ is used. The paper states the range honestly, so this is not a misrepresentation, but "more than twice the pooled seed deviation" reads as more decisive than 2.6×.
  2. **The denominator is still per-seed scatter, not the SE of the difference.** As in R0-W2.2, the relevant uncertainty for a *difference of two means* is SE_Δ ≈ σ_pooled·√(2/3) ≈ 0.82·σ_pooled ≈ 0.010–0.011, so in SE-of-difference units Δ/SE_Δ ≈ 3.1–3.4. The footnote's demotion to "descriptive effect size" technically licenses comparing Δ to a raw σ, but a methodology reader will still note that the *correct* descriptive normalizer for a mean-difference is the SE of the difference. The CBAM-vs-SpineNetV2 |Δ|/σ=3.8 (line 297) has the same issue and is additionally fragile because, with df=2, a 3.8-σ-unit gap is *not* the slam-dunk the prose tone ("a sizeable +0.034," "telling result") implies.
- **Location:** Abstract line 56; contribution line 70; Table 2b footnote line 294; discussion lines 297, 313; Summary line 322.
- **Fix:** (a) Soften "more than twice the pooled seed deviation" to "about 2.6× the pooled per-seed std (≈3× the standard error of the difference)" — this is both more accurate and, helpfully, a *larger*-sounding multiple in SE units. (b) Add a single bootstrap or Welch-t 95% CI on the Hybrid−CBAM Δ to the camera-ready (the seeds exist; this is a 10-line script, not a new run) so the central claim has one real interval. The R0 reviewer asked for this; the revision answered with reframing rather than a CI, which is acceptable for acceptance but the CI would close it fully.
- **Severity: MAJOR** (the paper's #1 contribution is staked on this; it is now *defensible* but not yet *quantified* with a proper interval).

### W5 — Does OTS > Hybrid on the zero-shot mean undermine the contribution? Verdict: NO, but the abstract over-frames the zero-shot result. **MAJOR.**
- **Issue:** The "selective transfer / semantic proximity" framing (lines 262, 267) is a **legitimate scientific claim, not post-hoc rationalization** — the prediction (RSNA supervision helps where the SPIDER label is semantically close to the RSNA disc-morphology training schema, and not elsewhere) is structurally falsifiable and the data support it: the hybrid wins exactly on the three disc-morphology labels (bulging 0.613 vs 0.363, herniation 0.563 vs 0.532, narrowing 0.586 vs 0.403) and loses on the schema-distant ones (endplate, spondylolisthesis, Modic). That is a coherent, directional, pre-registered-in-spirit pattern. **However**, the abstract (line 56) still leads with "text-prompt zero-shot transfer reaches mean F1 = 0.362 across eight unseen labels" as a positive headline, with no mention that an off-the-shelf encoder with zero RSNA training reaches 0.394 on the same labels. A reader of the abstract alone would conclude the RSNA-trained hybrid is the better zero-shot transferer overall, which the paper's own Table 2 contradicts. The contribution is real (selective transfer on the semantically-near labels), but the *aggregate* zero-shot number is not a win and should not be presented as a standalone achievement in the abstract.
- **Location:** Abstract line 56; contribution line 72; vs Table 2 line 262, discussion line 267.
- **Fix:** In the abstract, reframe to the honest scoped claim, e.g., "zero-shot transfer is *selective*: the RSNA-trained hybrid beats off-the-shelf BiomedCLIP on the three disc-morphology labels closest to its training schema, while off-the-shelf is competitive on schema-distant labels (overall mean F1 0.362 vs 0.394)." This costs ~one line and converts a latent over-claim into a defensible finding. Do **not** drop the zero-shot contribution — the *mechanism* (label space becomes an inference input; rare-class recall recovery, e.g., Disc Herniation 0.36 vs 0) is the genuine novelty and survives intact.
- **Severity: MAJOR** (abstract-level over-framing of a result the body honestly walks back).

### W6 — Residual zero-shot/under-specification (temperature, multiclass prompt sets). **MINOR.**
- **Issue:** C3 disclosed the binary prompt template and the a-priori protocol (good), but (a) the temperature `s` (Eq. cosine line 155; "learned temperature inherited from BiomedCLIP," line 158) still has no numeric value, and (b) the multiclass prompt sets for Pfirrmann (5-class) and Modic (4-class) are not enumerated — the binary example does not tell the reader how five Pfirrmann grades were verbalized, which is exactly where prompt sensitivity bites hardest and where Modic F1 collapses to 0.018 (line 259).
- **Location:** Lines 155, 158, 242, Table 2 lines 258–259.
- **Fix:** Give the `s` value (one number) and add the Pfirrmann/Modic prompt strings (an appendix line or a footnote). Low effort, closes the last reproducibility gap on zero-shot.
- **Severity: MINOR.**

### W7 — SPIDER per-condition rare-class precision still overstated; sub-0.5 AUC now absent but Modic AUC 0.454 remains unexplained-as-metric. **MINOR.**
- **Issue:** Carried from R0-W7: rare-class recalls over a 234-IVD / 34-patient val split (line 272) at ≤5% prevalence are reported to 3 sig figs with large std (Disc Herniation 35.6%±7.9, Pfirrmann-4 0.534 vs 0.473, lines 300). Absolute positive counts are still not given, so the ±std is dominated by which 1–2 patients land in the fold. Separately, the revised Table 2 (line 259) reports Modic AUC 0.454 — i.e., below 0.5 — which the text attributes to a T2-FS imaging-physics effect (line 267); that is a *plausible substantive* explanation, and an improvement over R0 where sub-0.5 AUCs were unexplained, but a sub-0.5 macro-OvR AUC also warrants a one-clause note that it reflects anti-alignment of the cosine direction for that label, not just physics.
- **Location:** Lines 272, 300, 259, 267.
- **Fix:** Add absolute positive counts for the rare SPIDER classes; round rare-class recalls to integer percent; one clause noting sub-0.5 AUC = embedding anti-aligns with the Modic prompt.
- **Severity: MINOR** (down from MAJOR in R0 because the physics explanation now partially covers it).

---

## What is now resolved from R0 (for the editor's tracking)
| R0 weakness | R0 severity | RR1 status |
|---|---|---|
| W1 single-seed RSNA headline | CRITICAL | **Resolved** — 3-seed Table 1 (one stray number, new W1) |
| W2 invalid ">2σ" significance | CRITICAL | **Resolved** — explicitly demoted to descriptive effect size (residual framing nit, W4) |
| W5 trivial-ensemble confound | MAJOR | **Resolved** — real logit-average control reported (W3 = verifiability polish) |
| W6 prompts/temperature | MAJOR | **Mostly resolved** — template + a-priori shown; temp value still missing (W6) |
| W9 naked-BiomedCLIP not tabulated | MINOR→MAJOR | **Resolved** — OTS F1 column added, means verified |
| W3 split mechanism | MAJOR | Still asserted, function not named (out of scope of C1–C4; recommend one sentence) |
| W4 AUC/AUPRC averaging spec | MAJOR | Partially — sub-0.5 AUCs now explained substantively (W7) |
| W7 tiny SPIDER val precision | MAJOR | Partially — still 3-sig-fig, no abs counts (W7) |
| W8 reproducibility statement | MAJOR | Unchanged — no code-availability note (out of C1–C4 scope) |
| W10 mixed precision in Table 1 | MINOR | **Resolved** — Table 1 precision now uniform |

## Questions to Authors
1. Line 93 says "Mean F1 0.528, Severe F1 0.343," but Table 1 says 0.527 / 0.356. Which is correct, and please reconcile (W1).
2. For the regularizer claim: can you add a single bootstrap/Welch 95% CI on the Hybrid−CBAM SPIDER-F1 difference? The seeds already exist (W4).
3. What is the numeric temperature `s`, and what are the Pfirrmann/Modic prompt strings (W6)?
4. What AUC/AUPRC did the logit-averaging ensemble attain (the numbers behind "higher AUC/AUPRC," line 313) (W3)?
5. Absolute positive-case counts per rare SPIDER class in the 234-IVD val split (W7)?

## Verdict
A well-executed revision. The two CRITICAL R0 blockers are genuinely fixed, the C2/C3/C4 evidence is now in the manuscript, and the authors' self-correction on OTS > Hybrid is a sign of integrity, not weakness. Accept with minor revision conditioned on: (i) fixing the line-93 number mismatch [must-fix], (ii) re-scoping the abstract's zero-shot sentence to the selective-transfer finding [must-fix], and (iii) adding a CI on the central +0.034 regularizer effect plus the temperature value [should-fix].
