# Methodology Review

**Reviewer 1 — Methodology & Statistics** (MIWAI 2026, LNAI, double-blind)
**Paper:** Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar IVD Grading and Cross-Dataset Zero-Shot Transfer

## Recommendation
**Major Revision (weak reject in current form).** The core narrative rests on RSNA headline numbers from a single seed (n=1) and on ">2σ" significance claims computed over 3 seeds with no valid hypothesis test; the statistical scaffolding cannot currently support the abstract's confident framing.

## Scores (0–100)
| Dimension | Score |
|---|---|
| Rigor | 42 |
| Statistical Validity | 35 |
| Reproducibility | 55 |
| Evaluation Design | 68 |
| **Overall** | **46** |

Rigor is capped below 50 because the headline result is single-seed; per the review brief a single-seed headline cannot score >6/10 on rigor unless heavily defended, and the defense here is one sentence in Limitations.

## Strengths
- **Evaluation metric selection is genuinely sound.** Macro-averaging over the 3 severities (Section "Evaluation metrics," line 204) is the correct guard against the 85% Normal/Mild prevalence masking failure on Severe. Reporting AUPRC alongside AUC, and stating the AUPRC random baseline equals positive-class prevalence (~0.05 for Severe), is methodologically correct and better than most imbalanced-classification papers.
- **The honest reframing of the contribution.** The authors do not overclaim hybrid > baseline on SPIDER (lines 295, 313 explicitly say "we do not claim the hybrid surpasses the baseline"). Framing the result as *regularization/recovery* rather than raw accuracy gain is intellectually honest and matches the data.
- **The single-seed limitation is at least disclosed** (line 317) and a 3-seed RSNA sweep with bootstrap CIs is named as planned future work (also line 322).
- **Patient-level split is the right unit** to prevent IVD-level leakage, and the arithmetic is internally consistent (see W3).
- **The trivial-ensemble confound is voluntarily disclosed** (line 313) rather than hidden — good scientific practice even though it weakens the claim.

## Weaknesses

### W1 — Headline RSNA numbers are single-seed (n=1); abstract framing is over-confident
- **Issue:** Every RSNA headline figure — Severe Recall 46.4%, Severe F1 0.343, Mean F1 0.528 (Table tab:main, lines 220–229) — is from seed 42 only. Limitations (line 317) confirms "RSNA results use a single seed (42)." With n=1 there is no estimate of variance, so claims like "4.0×" Severe-recall improvement and "2.6-point fusion gain over the better single branch" (line 234) are point estimates with unknown sampling error. On a Severe class at ~5% prevalence in a 1,942-IVD validation set, the Severe positive count is on the order of ~100 IVDs per condition; a recall of 46.4% over so few positives has a binomial 95% CI of roughly ±10 points even before seed variance — easily wide enough that the 4.0× headline could be substantially smaller. The abstract (line 56) states these as established facts ("lifts Severe-class recall from 11.5% to 46.4% (4.0×)") with no hedge.
- **Location:** Abstract line 56; Table tab:main lines 220–229; Section exp2 lines 234–236; Limitations line 317.
- **Fix:** Either (a) run the 3-seed RSNA sweep that is already promised and report mean ± std for the headline table (strongly preferred — the SPIDER side already does this, so the asymmetry is an effort gap, not a feasibility gap), or (b) add bootstrap 95% CIs over the validation set for at least Severe Recall/F1 and soften the abstract to "in our single-seed run." The within-row "Bold = best" markings in tab:main are meaningless without variance and should be removed or justified.
- **Severity: CRITICAL.**

### W2 — The ">2σ" SPIDER significance claims are not a valid significance test
- **Issue:** Lines 282–283, 292, 295, 313 claim the Hybrid–CBAM gap on Mean F1 (0.653 vs 0.619, Δ=+0.034, pooled σ=0.012) "exceeds 2σ" and is "significant." Three problems:
  1. **n=3 is far too small for a 2σ heuristic.** With 3 seeds, the per-config std has ~2 degrees of freedom; the sampling distribution of the std itself is enormously wide (a 3-sample std can under- or over-estimate the true σ by a factor of ~2 routinely). A "2σ" threshold borrowed from large-sample intuition is not calibrated at df=2. A proper test would be a two-sample t-test (Welch) yielding a p-value with t_{~2 df} critical values around 4.3 for α=0.05 two-sided — i.e., far stricter than "2σ." The paper reports |Δ|/σ = 3.8 for CBAM-vs-baseline (line 295) and treats it as conclusive, but 3.8 std-units at df≈2 is borderline, not the slam-dunk the prose implies.
  2. **"Pooled σ" is the wrong denominator for a difference of means.** The relevant quantity for comparing two means is the standard error of the *difference*, SE_Δ = sqrt(σ_A²/n_A + σ_B²/n_B), not a pooled per-seed std σ. With n=3 each, SE_Δ ≈ σ_pooled·sqrt(2/3) ≈ 0.82·σ_pooled — so the effective threshold the authors should clear is larger in t-units than their "Δ/σ" ratio suggests. The current framing compares a mean difference against a single-run scatter, which understates uncertainty.
  3. **Seeds are paired (same data, same splits, {42,123,456}).** A *paired* test (paired t or sign test on the 3 matched seed-pairs) would be both more appropriate and more powerful, but with n=3 paired observations the sign test cannot reach p<0.05 at all (min two-sided p = 0.25), and a paired t at df=2 still needs t>4.30. The authors should state which test, paired or unpaired, and report the actual p-value or CI rather than an ad-hoc σ-multiple.
- **Location:** Table tab:spider_transfer_agg lines 282–283; footnote line 292; lines 295, 313; abstract line 56 and contribution bullet line 70.
- **Fix:** Replace every "> 2σ" / "|Δ|/σ = 3.8" statement with a named test and its result: paired (or Welch) t-test over the 3 seeds with the exact p-value and a 95% CI on Δ, and explicitly acknowledge that n=3 yields very low power (so a non-significant result is uninformative, and a "significant" one is fragile). Better: increase to ≥5 seeds. If the authors keep the σ-multiple language, they must call it a descriptive effect size, not significance. The asterisks/bolding in tab:spider_transfer_agg should be tied to the reported test.
- **Severity: CRITICAL.**

### W3 — Patient-level split: claim is internally consistent but leakage guarantee is not demonstrated
- **Issue:** Line 182 states a patient-level 80/20 split (random_state=42) giving 1,942 validation IVDs from 395 patients. Arithmetic checks out: ~2,000 patients × 20% ≈ 395–400 patients; 1,942/395 ≈ 4.9 IVDs/patient, consistent with 5 evaluated IVD levels minus occasional missing levels. So the *numbers* are coherent. **However**, the paper only asserts the split is patient-level; it never states the mechanism that guarantees all IVDs of a patient land in one fold (e.g., GroupShuffleSplit on study_id). Given the codebase uses `random_state=42` patient-level splits, this is plausible, but a reviewer cannot verify no IVD-level leakage from the text alone. The SPIDER supervised-transfer split (line 270: 234 val IVDs / 34 patients) is similarly asserted but with no statement on whether the 3 seeds reshuffle the patient grouping or only the training stochasticity.
- **Location:** Lines 182, 270, 214 (table caption).
- **Fix:** State the exact splitting function/grouping key (one sentence: "patients are grouped by study_id and split with GroupShuffleSplit, so no patient appears in both folds"). Clarify whether the 3 SPIDER seeds vary the data split or only init/sampling order — this materially affects what the ±std measures (if splits are fixed across seeds, the std captures only optimization noise, not split variance, which weakens the generalization claim).
- **Severity: MAJOR.**

### W4 — Macro-OvR AUC / AUPRC computation under-specified; SPIDER multiclass labels need clarity
- **Issue:** Line 204 says AUC is "one-vs-rest" and macro-averaged over C=3 severities — this is computable and standard, but the paper does not say whether OvR probabilities are renormalized, nor how ties/degenerate folds are handled. More importantly, in tab:spider (lines 256–258) AUC is reported for **multiclass** labels: Pfirrmann (5-class) AUC 0.492 and Modic (4-class) AUC 0.422. An AUC *below 0.5* for a multiclass macro-OvR metric is suspicious and needs explanation — for a 5-class problem, macro-OvR AUC of 0.492 is essentially "worse than the per-class chance level of 0.5," which can happen but should be flagged and the averaging scheme (macro vs weighted; per-class vs micro) stated. The AUPRC prevalence baseline is correctly stated for RSNA Severe (~0.05, line 204; 0.042 used at line 236 — minor inconsistency 0.05 vs 0.042), but no prevalence baseline is given for any SPIDER AUPRC, so the reader cannot judge whether SPIDER AUPRC numbers beat chance.
- **Location:** Lines 204, 236 (0.05 vs 0.042), 256–258 (sub-0.5 AUCs), tab:spider_transfer_agg AUPRC line 284.
- **Fix:** State the exact AUC/AUPRC averaging (sklearn `roc_auc_score(multi_class='ovr', average='macro')` or equivalent), reconcile 0.05 vs 0.042, explain the sub-0.5 Pfirrmann/Modic AUCs (likely a sign that zero-shot text prompts anti-align for those labels), and give the prevalence baseline for SPIDER AUPRC values so they are interpretable.
- **Severity: MAJOR.**

### W5 — Trivial-ensemble confound undermines the central "regularizer" claim
- **Issue:** The paper's lead contribution (line 70, abstract line 56) is that BiomedCLIP acts as a *cross-dataset regularizer* recovering CBAM's transfer loss. But line 313 concedes "a trivial-ensemble explanation (logit averaging of Baseline and BMC-only) cannot be fully ruled out." This is not a peripheral caveat — it is a direct alternative explanation for the *headline causal claim*. If simple logit-averaging of Base + BMC-only reproduces the Hybrid's SPIDER recovery, then "regularization" is the wrong mechanistic story and the architectural fusion-MLP contribution is not isolated. The experiment to rule this out (a Base⊕BMC logit-average baseline) is cheap — it requires no new training, only re-using already-trained Base and BMC-only checkpoints — so deferring it to "future work" is hard to justify for the paper's #1 contribution.
- **Location:** Line 313; contribution bullets lines 70–71; abstract line 56.
- **Fix:** Add the logit-averaging ensemble as a fifth column / row in the SPIDER analysis. If the Hybrid beats it, the regularizer claim is earned; if not, retitle the contribution honestly. This single experiment would substantially raise the paper's rigor score.
- **Severity: MAJOR.**

### W6 — Zero-shot protocol not fully specified (prompts + temperature)
- **Issue:** The zero-shot setting (Eq. cosine, line 155) uses temperature s "inherited from BiomedCLIP pretraining" (line 158) but the value is never given, and it is unclear whether s is the frozen BiomedCLIP logit scale or re-learned. More critically, **the text prompts {q_k} are never shown verbatim** for any of the 8 SPIDER labels (Table tab:spider). Zero-shot results are known to be highly prompt-sensitive; without the exact prompt strings, the mean F1 = 0.362 headline is not reproducible and the "selective transfer / semantic proximity" interpretation (line 265) cannot be checked. There is also no statement of how multi-class labels (Pfirrmann 5-class, Modic 4-class) are mapped to prompt sets, nor whether prompt ensembling/templating was used.
- **Location:** Lines 116, 155, 158, 240, tab:spider; no appendix.
- **Fix:** Add a table of the exact prompt template(s) per label (or an appendix), state the temperature s value, and note whether any prompt selection/tuning was done on SPIDER (if so, that is test-set leakage and must be disclosed). Without verbatim prompts the zero-shot section is not reproducible.
- **Severity: MAJOR.**

### W7 — SPIDER supervised-transfer validation set is very small (234 IVDs / 34 patients)
- **Issue:** Line 270: 234 val IVDs from 34 patients, across 8 conditions, several at ≤5% prevalence (line 297) and Modic types at ≤0.5% (line 298). At 34 patients and ~5% prevalence, a rare positive class may have only 1–3 positive patients in validation; per-condition recall figures like Disc Herniation 35.6%±7.9 or Spondylolisthesis 30.0% (line 298) are then computed over a handful of positives, so the ±std over 3 seeds is dominated by which 1–2 patients landed in the fold rather than model quality. Reporting these to 3 significant figures (0.534 vs 0.473 Pfirrmann-4, line 298) overstates precision.
- **Location:** Lines 270, 297–298.
- **Fix:** Report the absolute positive count per rare class in validation; round per-class recalls to integer percent; add CIs (Wilson/bootstrap) for the rare-class recalls, and temper claims like "the hybrid leads on rare-class recall" accordingly.
- **Severity: MAJOR.**

### W8 — Reproducibility: no code/data availability statement; missing several hyperparameters
- **Issue:** RSNA optimization hyperparameters are mostly given (AdamW, lr 1e-4, batch 32, wd 1e-4, ≤20 epochs, patience-5, focal γ=2.0, class weight ∝1/√n_c, oversample 3–5×, augmentation params — lines 169–175). Good. But: (a) no code/data availability statement at all (understandable under double-blind, but a "code will be released" note is expected); (b) the SPIDER transfer uses "15 epochs" (line 270) with no LR/optimizer/early-stopping stated — only RSNA hyperparameters are given; (c) oversample factor is a range "3–5" not a fixed value, so the run is not exactly reproducible; (d) the temperature s value (W6) is missing; (e) the BiomedCLIP checkpoint/version and image preprocessing for the 2.5D branch (resize interpolation, normalization stats) are partly given (line 135) but not the normalization constants. RSNA/SPIDER are public, which helps.
- **Location:** Lines 169–175 (RSNA hp present), 270 (SPIDER hp absent), throughout.
- **Fix:** Add a Reproducibility/Implementation-details paragraph with: the single oversample factor actually used, SPIDER transfer optimizer+LR+early-stop, temperature value, BiomedCLIP version string, and a (blinded) code-release commitment.
- **Severity: MAJOR.**

### W9 — Naked-BiomedCLIP zero-shot baseline referenced but not tabulated
- **Issue:** Line 265 claims "the hybrid wins on disc-related labels" against a Naked BiomedCLIP baseline, but no numbers for that baseline appear in tab:spider or anywhere. A comparative claim with zero reported comparison values is unverifiable.
- **Location:** Line 265, tab:spider.
- **Fix:** Add the Naked-BiomedCLIP per-label F1/AUC column to tab:spider, or remove the comparative claim.
- **Severity: MINOR (would be MAJOR if the claim is load-bearing — it partly supports the zero-shot contribution).**

### W10 — Mixed precision in reported figures
- **Issue:** Table tab:main mixes "81.4%" / "72.1%" (1 decimal) with "68.98%" / "69.19%" (2 decimals) in the same Mean Accuracy row (line 220), and F1 to 3 decimals. Inconsistent precision suggests numbers were transcribed from different runs/sources and reduces confidence in their provenance.
- **Location:** Line 220.
- **Fix:** Use uniform precision (e.g., one decimal for %, three for F1/AUC) across all configs.
- **Severity: MINOR.**

## Questions to Authors
1. The SPIDER side has 3 seeds; running RSNA at 3 seeds is the same pipeline. Why was the *headline* dataset left at n=1? Can you provide the 3-seed RSNA table before camera-ready?
2. For the ">2σ" claims: which test (paired vs unpaired), and what is the exact p-value at df=2? Do the seeds vary the data split or only initialization/sampling order?
3. Does logit-averaging of Base + BMC-only reproduce the Hybrid's SPIDER recovery? (No new training needed.)
4. What are the exact zero-shot prompt strings per label, and the temperature s? Was any prompt selected using SPIDER labels?
5. Why is macro-OvR AUC < 0.5 for Pfirrmann (0.492) and Modic (0.422)? Which averaging scheme and how are multiclass labels handled in OvR?
6. For rare SPIDER classes (Disc Herniation, Spondylolisthesis, Modic), how many positive validation cases exist (absolute count)?
7. Is 0.05 or 0.042 the correct Severe AUPRC baseline (both appear)?

## Reproducibility Checklist

| Item | Status |
|---|---|
| RSNA optimizer/LR/batch/WD/epochs/patience | Present (lines 169–175) |
| Focal γ, class-weight scheme | Present |
| Oversample factor | **Partial** — range "3–5", not a fixed value |
| Augmentation params | Present (line 175) |
| SPIDER transfer optimizer/LR/early-stop | **Missing** (only "15 epochs," line 270) |
| Train/val split mechanism (grouping key) | **Missing** — "patient-level" asserted, function not named |
| Seeds | Present for SPIDER {42,123,456}; RSNA single seed 42 |
| Multi-seed variance on headline (RSNA) | **Missing** — n=1 |
| Statistical test definition for significance | **Missing** — ad-hoc "2σ" only |
| Zero-shot prompt strings | **Missing** |
| Zero-shot temperature s value | **Missing** |
| BiomedCLIP version/checkpoint | **Partial** — "ViT-B/16" only, no version string |
| 2.5D branch preprocessing (resize/normalize constants) | **Partial** |
| Naked-BiomedCLIP baseline numbers | **Missing** (claim made line 265) |
| Hardware | Present (line 206) |
| Code/data availability statement | **Missing** (data is public; code release not stated) |
| AUC/AUPRC averaging implementation | **Missing** |

**Overall reproducibility:** RSNA supervised run is ~70% reproducible; the zero-shot and SPIDER-transfer experiments are **not currently reproducible** from the text (missing prompts, temperature, SPIDER hyperparameters, split mechanism).
