# RR3 — Devil's Advocate Re-Review (MIWAI 2026)

Paper: *Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar IVD Grading and Cross-Dataset Zero-Shot Transfer*
Scope of this review: re-review after revisions addressing (3-seed RSNA, trivial-ensemble control, zero-shot prompt disclosure, off-the-shelf BiomedCLIP comparison). The revision added two honest disclosures (A: OTS BiomedCLIP > hybrid on overall zero-shot mean F1; B: trivial ensemble > hybrid on AUC/AUPRC). This review stress-tests whether the contributions survive those disclosures.

---

## Strongest Counter-Argument (the case for rejection)

The revisions are commendably honest, but in being honest they have quietly dismantled three of the paper's four headline contributions, leaving a paper whose title promises two things ("zero-shot transfer" and a winning "hybrid") that its own tables now refute.

**1. The zero-shot contribution is not the model's — it is BiomedCLIP's.** Table 2 shows that an off-the-shelf BiomedCLIP with **zero RSNA training** achieves a *higher* overall zero-shot mean F1 (0.394) than the authors' RSNA-trained hybrid (0.362). The "label-space extension" property — the ability to answer a new prompt vocabulary without retraining — is a property of the frozen BiomedCLIP text-image alignment, which the authors did not build and did not improve. Their RSNA training makes the overall zero-shot result *worse*. The contribution as stated in the abstract/intro ("the same RSNA-trained model transfers... reaching mean F1 = 0.362") is therefore a repackaging of a foundation-model capability, and on the aggregate number their intervention is net-negative. The 3-disc-label win (bulging/herniation/narrowing) is precisely the subset semantically closest to RSNA disc grading — i.e., the authors are reporting the favorable slice of an 8-label table where their model loses 5 of 8 labels and loses the mean. A skeptical reviewer reads this as cherry-picking a 3/8 subset to rescue a contribution the full table contradicts.

**2. "Just use the ensemble."** The paper's own control (Section 4.6) shows the trivial average of two branches achieves *higher* AUC and AUPRC than the learned concat-MLP fusion. AUC/AUPRC are the threshold-free ranking metrics — exactly the metrics one trusts most under severe imbalance, and exactly the metrics the paper itself elevates ("we report AUPRC because AUC can stay high while a model fails to rank Severe cases"). The hybrid's only claimed advantage is thresholded F1/Severe-F1, which depends on a single arbitrary operating point that the paper never justifies or tunes on held-out data. A reviewer can fairly say: the better-ranking, simpler, near-zero-parameter ensemble dominates on the principled metrics, and the learned 1.18M-parameter fusion buys only a fragile F1 win at an unspecified threshold. This undercuts the second contribution ("two-branch hybrid architecture... complementary").

**3. There is no clean cross-dataset headline.** Net result across both datasets: on RSNA the hybrid wins F1/Severe but *loses* Mean Accuracy by 8.6 points; on SPIDER supervised transfer it merely *matches* SpineNetV2 (ΔF1 = +0.006, within noise) and *loses* to SpineNetV2 on AUPRC. The only "win" the paper can defend with >2σ evidence is Hybrid > CBAM-only — but CBAM-only is the authors' own ablation, not a competitive baseline. So the strongest statistically-supported claim reduces to "our component B repairs the damage our component A caused," i.e., the hybrid is a regularizer that *restores* SpineNetV2-level performance after the authors added a module (CBAM) that hurt it. That is an argument *against* including CBAM at all, not for the hybrid. The single decisive win over a simpler alternative (plain SpineNetV2) on a principled aggregate metric does not exist.

In short: the most rigorous reading of the revised tables is that BiomedCLIP (not the authors) owns the zero-shot story, a trivial ensemble (not the authors' MLP) owns the ranking story, and plain SpineNetV2 (not the hybrid) is the simpler thing that already matches them on aggregate. The honesty is real; the headline is gone.

---

## Issue List

### CRITICAL

- **C1 — Title/abstract overclaim vs. own zero-shot table.** Title sells "Cross-Dataset Zero-Shot Transfer" and the abstract reports "mean F1 = 0.362 across eight unseen labels" as a contribution, but Table 2 shows OTS BiomedCLIP beats this (0.394) with no RSNA training. The headline zero-shot number is the authors' *losing* configuration on the aggregate. Either the framing must change (RSNA training *degrades* overall zero-shot but improves the disc-near subset) or the contribution must be demoted. As written it is an overclaim that the paper's own table refutes.

- **C2 — Learned fusion not justified over trivial ensemble on principled metrics.** The paper argues AUPRC/AUC are the right metrics under imbalance (Sec 4.1), then concedes (Sec 4.6) the trivial ensemble wins exactly those metrics. The defense ("preferable at the clinical operating point") rests on a thresholded F1 advantage at an *unspecified, untuned* operating point. With no operating-point selection protocol on held-out data, the F1 win is not defensible and "just use the ensemble" is the parsimonious conclusion. This threatens contribution #2.

### MAJOR

- **M1 — Disc-subset win is structurally confounded with semantic proximity.** The 3 labels the hybrid wins (bulging, herniation, narrowing) are the SPIDER labels closest to RSNA's disc-grading schema. The "selective transfer" narrative is plausible but the paper presents the 3-win subset prominently while the 8-label mean loss is stated almost in passing. Needs an explicit, prominent statement that the hybrid loses the aggregate and wins only a semantically-adjacent subset — and ideally a pre-registered definition of "semantically close" rather than a post-hoc grouping.

- **M2 — The only >2σ supervised-transfer win is over the authors' own ablation, not a baseline.** Hybrid > CBAM-only (+0.034) is the headline effect size, but CBAM-only exists only because the authors added CBAM. Framed honestly this is "B fixes A," which is an argument to drop A. The paper should confront whether a BMC-only-without-CBAM or plain-SpineNetV2+BiomedCLIP configuration would be the cleaner, simpler design — the data hint CBAM is a net liability cross-dataset.

- **M3 — Ranking-vs-threshold divergence is pervasive and under-explained.** Spondylolisthesis AUC 0.807 / F1 0.028; Pfirrmann AUC 0.687 / F1 0.155; ensemble wins AUC/AUPRC but loses F1. This recurring pattern means the cosine-threshold decision rule is badly miscalibrated. If the embedding *ranks* well but *thresholds* badly everywhere, the honest claim is "we have a calibration problem," and the F1-based wins (the paper's main evidence) are artifacts of an arbitrary threshold rather than evidence of a better representation.

- **M4 — Coherence: the paper is now four mixed results, not one story.** Contribution ordering in the intro is by "strength of multi-seed evidence," and the #1 contribution is now "CBAM hurts, BiomedCLIP repairs it" — a negative result about the authors' own module dressed as a finding. A reviewer can reasonably call this "no clear win": each dataset/metric has a different winner (SpineNetV2 on Accuracy/AUPRC; ensemble on AUC/AUPRC; OTS BiomedCLIP on zero-shot mean; hybrid only on RSNA F1/Severe-F1 and a single CBAM-relative gap).

### MINOR

- **m1 — Two-seed ensemble vs three-seed hybrid.** The trivial-ensemble control (Sec 4.6) is a *two-seed* mean compared against three-seed hybrid numbers; the comparison the paper most needs to win is run at lower statistical strength. Either re-run at 3 seeds or flag the asymmetry.
- **m2 — Abstract claims "exceeds CBAM-only by more than twice the pooled seed standard deviation" as a near-headline.** True, but it is a win over an ablation, not a baseline; in the abstract it reads as a competitive claim. Consider rewording to avoid implying superiority over a real comparator.
- **m3 — "Hybrid wins on every metric except Mean Accuracy" (Sec 4.2) is RSNA-only** but is easy to misread as a global claim; SPIDER Table 4 shows SpineNetV2 winning Accuracy *and* AUPRC. Scope the sentence.
- **m4 — AUPRC 7.9× / "2.3×" framing.** Multiplicative framing of small absolute F1 gains (0.152→0.356) is rhetorically inflated; absolute deltas should lead given the noise (±0.017–0.082).
- **m5 — Prompt disclosure is good but single-prompt.** Zero-shot uses one fixed template "a magnetic resonance image of [label]." No sensitivity analysis over prompt phrasing; a reviewer may ask whether OTS-vs-hybrid ordering is prompt-dependent, which would further weaken any zero-shot claim.

---

## What Genuinely Survives (fair credit)

- **S1 — RSNA Severe-class improvement is real and multi-seed.** Severe F1 0.152→0.356, Severe recall 12.3%→48.6%, Severe AUC 0.899, all over 3 seeds with stds reported. This is the paper's most defensible empirical result and is honestly caveated (8.6-pt accuracy cost stated, clinical-readiness disclaimer in Sec 4.7).
- **S2 — Statistical honesty is now exemplary.** Effect-size language instead of significance tests at n=3, explicit "we do not claim the hybrid surpasses SpineNetV2 here," 2σ bolding rule in Table 4, parity stated openly. This is better practice than most accepted MIWAI-tier papers.
- **S3 — The trivial-ensemble control was actually run and reported against interest.** Few authors run the control that undermines their fusion and then publish it. The concat-MLP-is-not-a-logit-average point (F1 0.53 vs 0.45) is a legitimate, if narrow, result.
- **S4 — Honest negative/mixed disclosures (A and B) are scientifically sound** even though they damage the headline. The OTS-beats-hybrid and ensemble-beats-fusion admissions are the right thing to report.
- **S5 — Positioning vs prior RSNA work is fair and well-argued** (Sec 2): the paper correctly refuses to compare its 3-class full-distribution numbers to others' binarized/balanced/ensemble setups.
- **S6 — Class-imbalance pipeline is principled and reproducible** (sqrt class weights with quantified gradient-ratio effect, focal loss with ignore_index, oversampling rationale, cost ~$2/run, single-GPU). Reproducibility is strong.
- **S7 — Calibration insight (ranks well, thresholds badly) is itself a publishable observation** if reframed as a finding rather than buried as a caveat.

---

## Honest accept-ability read (MIWAI tier)

MIWAI is a mid-tier (Springer LNAI) applied-AI venue, not a top archival conference; it rewards solid empirical work and tolerates modest/mixed results, and it values methodological honesty. Against that bar:

- The paper will **not** be rejected for lack of rigor — the revisions made it more rigorous than the median submission.
- It **is** vulnerable to a "no clear win / contribution overclaimed in title+abstract" critique. The fix is largely *framing*, not new experiments: demote the zero-shot mean-F1 claim, reframe contribution #1 as a regularization/negative-result finding (which the authors have already half-done), scope the "wins every metric" sentences, and lead with the one solid result (RSNA Severe-class recovery) plus the honest calibration finding.
- **My call:** borderline-accept / weak-accept at MIWAI tier *conditional on the abstract and title being de-hyped to match the tables* (C1) and the fusion-vs-ensemble justification being tightened or honestly downgraded (C2). As literally titled and abstracted today, a careful reviewer has fair grounds for major-revision because the headline ("zero-shot transfer," "hybrid") is contradicted by the paper's own honest tables. The science is acceptable; the packaging is one notch too strong.
