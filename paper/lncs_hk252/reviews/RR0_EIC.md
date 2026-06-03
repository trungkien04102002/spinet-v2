# EIC Re-Review (Round 2) — MIWAI 2026 (Springer LNAI)

## Status of Submission
Revised manuscript addressing the four Round-1 concerns. Verified against `main.tex` and `main.pdf` (12 pages exactly — at the hard limit).

## Verification of Round-1 Fixes
- **C1 (single-seed RSNA headline → 3-seed mean±std).** FIXED. Table~\ref{tab:main} (lines 212–234) now reports mean±std over seeds {42,123,456} for all four configs. Headline numbers updated accordingly (Severe recall 12.3→48.6%, Severe F1 0.152→0.356). This removes the prior internal inconsistency (the paper demanded >2σ for SPIDER but used single-seed for RSNA). Strong, credibility-restoring fix.
- **C2 (trivial-ensemble confound → real control).** FIXED, and FIXED HONESTLY. §4.5 (line 313) reports the Base+BMC softmax-average control: Mean F1 0.45 / Severe F1 0.18 vs learned-fusion 0.53/0.36, so the concat-MLP is not reducible to a logit average on the clinical (F1/Severe) axis. The authors also disclose the ensemble wins on AUC/AUPRC. This is the correct scholarly move.
- **C3 (zero-shot prompts undisclosed → disclosed).** FIXED. §4.3 (line 242) states the fixed a-priori prompt template ("a magnetic resonance image of [label]"), gives an example, and explicitly claims no SPIDER-target leakage. Adequate.
- **C4 (off-the-shelf BiomedCLIP not quantified → OTS column added).** FIXED, and FIXED HONESTLY. Table~\ref{tab:spider} (lines 244–265) now has an `OTS F1` column; the disclosed result is that OTS BiomedCLIP wins on overall mean F1 (0.394 vs 0.362), hybrid winning only on the 3 disc-morphology labels.

All four fixes land. Two of them surface honest negative results that the prior prose had implied a win.

## Overall Recommendation
**Minor Revision (accept-leaning).** All R1 concerns are resolved and the revision is materially more credible. The remaining work is *framing/positioning only* — no new experiments — to ensure a skim-reading reviewer extracts the one defensible contribution rather than reading the paper as an honest-but-no-clean-win mixed bag. On EIC criteria (venue fit, originality, significance, quality) it clears the MIWAI bar.

## Scores (0–100; Round-1 in parentheses)
- Originality: 72 (70) — first peer-reviewed BiomedCLIP application to RSNA 2024 lumbar + first RSNA→SPIDER text-prompt zero-shot label-space extension; hook intact and specific.
- Significance: 64 (66) — slightly down. The two new honest disclosures (OTS beats hybrid on overall zero-shot mean F1; trivial ensemble beats fusion on AUC/AUPRC) narrow the magnitude of the practical win, even as they raise the integrity of the paper. Net: more trustworthy, less impressive.
- Venue Fit: 86 (85) — foundation-model + attention + deployment-relevant medical task with an explicit second-reader framing is squarely on-topic for applied-AI LNAI.
- Clarity: 81 (80) — readable; honesty is consistent in the body. The residual issue is the title/abstract→body gradient (see W1).
- Overall: 73 (72) — above the reject line for a mid-tier applied-AI venue; the contribution is real once correctly framed.

## Strengths
- **Intellectual honesty now well above the mid-tier norm.** The paper twice reports a result that cuts against its own architecture: OTS BiomedCLIP wins overall zero-shot mean F1 (line 262/267), and the trivial ensemble wins AUC/AUPRC (line 313). It then carves out the precise axis on which the hybrid does win. This is exactly the behavior an EIC wants to reward, not penalize.
- **Defensible multi-seed core.** With RSNA now 3-seed, the paper's evidentiary backbone is internally consistent: every quantitative claim is mean±std, and the 2σ gating rule (Table~\ref{tab:spider_transfer_agg} footnote, line 294) is applied uniformly.
- **The regularizer story is genuinely interesting and multi-seed-supported.** CBAM helps in-domain Severe (0.262 vs 0.152) but degrades cross-dataset F1 below plain SpineNetV2 (0.619 vs 0.646, |Δ|/σ=3.8), and the frozen BiomedCLIP branch restores it (+0.034 > 2σ). This help-and-hurt asymmetry is a real, reproducible finding and is the paper's strongest contribution.
- **Metric discipline for imbalance.** AUPRC alongside AUC with prevalence baseline (line 204), macro-averaging justified, per-class Severe broken out.
- **Reproducibility signals.** Fixed split (random_state=42), named seeds, hardware+cost, trainable-parameter budget (1.18M of ~218M).
- **Honest positioning vs prior RSNA work** (lines 92–93): correctly explains why the 0.945/0.957 binary numbers are not comparable instead of cherry-picking them.

## Weaknesses / Concerns

### 1. Title + abstract still sell an architecture-superiority story the honest body walks back. **[MAJOR — and now the single most important fix]**
- Location: title (line 36), abstract (line 56) vs §4.3 (line 262/267), §4.5 (line 297, line 313), Table~\ref{tab:main} Mean-Accuracy row (line 222).
- Issue: This was W1 in R1 and is *more* acute after revision, because the body is now even more candid about where the hybrid loses. The title leads "Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings … and Cross-Dataset Zero-Shot Transfer" — promising both a winning architecture and a winning zero-shot method. But the revised body establishes: (a) hybrid is at *parity* with plain SpineNetV2 on SPIDER aggregate F1 and *loses* on AUPRC (line 286, line 297); (b) hybrid *loses* Mean Accuracy on RSNA (line 222, 8.6-pt drop); (c) OTS BiomedCLIP *beats* the hybrid on overall zero-shot mean F1 (line 262). A skim reviewer reading title+abstract+Table 2's "Mean (all 8)" row where OTS is bolded as best will perceive overclaim, even though every individual sentence is true. The gap between the title's promise and the body's honesty is the chief risk to how this paper lands.
- Fix (no new experiments): re-lead the title and abstract on the *one clean defensible headline* (see Reframing below). Demote the architecture-superiority phrasing.

### 2. The paper currently reads as three half-wins rather than one full win. **[MAJOR — positioning]**
- Location: contributions list (lines 68–74); §4.2 (RSNA Severe), §4.3 (zero-shot), §4.5 (transfer regularizer).
- Issue: Each of the three results carries an asterisk after the honest disclosures — RSNA Severe lift costs Mean Accuracy; zero-shot is beaten overall by OTS; supervised transfer is parity-not-win vs baseline. A reviewer can read the whole as "honest but no clear win." The contributions list (lines 68–74) is already ordered by evidence strength (regularizer first), which is good, but the abstract and title do not match that ordering, so the strongest, cleanest claim is buried.
- Fix: Pick ONE headline and make the other two explicitly *subordinate supporting evidence* for it, not co-equal claims. Concretely, lead everywhere with the regularizer finding (it is the only multi-seed, >2σ, novel-and-clean result) and frame RSNA-Severe and zero-shot as the two phenomena that the regularizer explanation predicts and unifies.

### 3. Table 2 bolds OTS as overall best, which visually concedes the headline to the baseline. **[MINOR — but high-leverage presentation]**
- Location: Table~\ref{tab:spider} "Mean (all 8)" row (line 262): `OTS 0.394` is bolded over `Hybrid 0.362`.
- Issue: This is honest and correct under the stated rule, but for a skim reviewer the single boldface number in the summary row of the zero-shot table says "the baseline wins." Combined with W1, this is the visual the reviewer remembers.
- Fix: Keep the OTS column (honesty), but restructure the table so the comparison the hybrid *wins* is the visual focus: group the 3 disc-morphology labels (where hybrid wins decisively, e.g. bulging 0.613 vs 0.363) and report a "Mean (disc-morphology)" sub-row in addition to "Mean (all 8)". Then the reader sees both the honest overall (OTS leads) and the selective win (hybrid leads on the semantically-near labels) — which is exactly the §4.3 prose argument (line 267), now made visible. No new numbers needed; they already exist in the per-row data.

### 4. Inconsistent bold conventions across the two main tables persist. **[MINOR]**
- Location: Table~\ref{tab:main} ("Bold = best per row", line 214) vs Table~\ref{tab:spider_transfer_agg} ("Bold = best per row when gap exceeds 2σ", line 276).
- Issue: Two definitions of bold in one paper. In Table 1, Mean Accuracy is bolded as best for SpineNetV2 (line 222) even though the paper argues accuracy is the wrong objective (line 238). The caption parenthetical helps, but the visual still rewards the dismissed metric.
- Fix: Apply the 2σ-gated rule to both tables, or de-emphasize (grey/italic) Mean Accuracy in Table 1 so the visual hierarchy matches the argument.

### 5. Single-example qualitative evidence; small architecture figure. **[MINOR]**
- Location: Figure~\ref{fig:arch} at 0.52\linewidth (line 162); Figure~\ref{fig:gradcam} n=1 at 0.42\linewidth (line 308).
- Issue: One Grad-CAM on one Severe case is thin support for the spatial-attention claim (line 304); the architecture diagram (the entry point for an applied-AI reader) is rendered at ~half text-width, against the project's own figure-readability guidance.
- Fix: Enlarge Figure 1 toward full text-width; if space permits after trimming whitespace, add 1–2 Grad-CAM panels. Page-neutral at the 12-page limit only if traded against existing whitespace — do not exceed 12 pages.

### 6. Minor framing nit in §4.3 wording. **[MINOR]**
- Location: line 242 — "the first reported cross-dataset zero-shot transfer (RSNA to SPIDER)."
- Issue: defensible but absolute; given OTS BiomedCLIP also does zero-shot SPIDER (it is in the same table), consider "first reported RSNA-supervised cross-dataset zero-shot label-space extension" to keep the novelty claim watertight.
- Fix: one-word qualifier.

## Length / Structure / Double-Blind
- **12-page limit:** at exactly 12 pages — RESPECTED but zero margin. Any added sub-row/figure must be offset by trimming. Watch this.
- **Double-blind:** CLEAN. Author block and acknowledgments correctly commented out (lines 41–51, 324–328); no identifying self-citation phrasing in the body.
- **Structure:** coherent and standard (Intro → Related → Method → Experiments → Summary). Contributions ordered by evidence strength (good). Reads well.

## Reframing Recommendation (the requested deliverable)

**Single clean defensible headline:**
> *Adding CBAM attention improves in-domain minority-class detection but provably degrades cross-dataset generalization; a frozen BiomedCLIP branch restores it — acting as a cross-dataset regularizer — while also unlocking zero-shot label-space extension.*

This is the only claim that is (a) multi-seed, (b) >2σ, (c) novel, and (d) not undercut by either honest disclosure. The OTS-wins-overall and ensemble-wins-AUC results do not contradict it — they sharpen it (the hybrid's value is regularization + the operating-point/F1 axis + the label-extension *capability*, not raw zero-shot dominance).

**Concrete, no-experiment changes:**
1. **Retitle** to lead on the regularizer + capability story, e.g.: *"When Attention Hurts Transfer: A Frozen BiomedCLIP Branch as a Cross-Dataset Regularizer for Lumbar Disc Grading and Zero-Shot Label-Space Extension."* This sets the reviewer's expectation to the result the paper actually proves, converting the honesty from a liability into the selling point.
2. **Rewrite the abstract's last 2 sentences** so the regularizer finding leads and RSNA-Severe + zero-shot are framed as supporting phenomena. Explicitly state up front, in the abstract, the two honest losses (OTS leads overall zero-shot mean F1; ensemble leads AUC/AUPRC) and immediately state the axis on which the hybrid wins (F1/Severe-F1 at the clinical operating point, + label-extension capability OTS-fusion cannot match in supervised transfer). Pre-empting the "but the baseline wins" reaction in the abstract is what flips a skeptical reviewer.
3. **Add the "Mean (disc-morphology)" sub-row to Table 2** (W3) so the selective win is visible, not only prose.
4. **State the capability framing for zero-shot explicitly:** the contribution of the zero-shot result is not "we beat OTS" (you do not, overall) but "an RSNA-supervised model retains a *queryable* label space and beats OTS on the semantically-near labels at zero marginal training cost." That is a true, defensible, applied-AI-relevant claim.

Net: the science does not change; the framing converts an "honest mixed bag" into "an honest, focused negative-and-positive result with a clear thesis." For a mid-tier applied-AI venue, an honest paper with one crisp reproducible finding is publishable; an honest paper that reads as three asterisked half-claims is the one that gets rejected for "no clear contribution."

## Editorial Decision
**MINOR REVISION (accept-leaning).** All four Round-1 concerns are resolved; the revision is more credible and more honest. No new experiments required. Conditional on the three framing fixes below.

### Top 3 highest-priority fixes
1. **Retitle + re-lead the abstract on the cross-dataset regularizer headline** (W1, W2), and surface the two honest losses in the abstract while naming the axis the hybrid wins on. This is the make-or-break fix.
2. **Add a "Mean (disc-morphology)" sub-row to Table~\ref{tab:spider}** (W3) so the hybrid's selective zero-shot win is visible alongside the honest overall OTS lead — no new numbers.
3. **Unify the bold convention across Tables 1 and 2** (W4) using the 2σ-gated rule, and de-emphasize Mean Accuracy in Table 1 so the visual hierarchy matches the argument.

(Secondary, page-budget permitting: enlarge Figure 1 / add Grad-CAM panels (W5); tighten the "first" claim wording (W6). Strictly hold to 12 pages.)
