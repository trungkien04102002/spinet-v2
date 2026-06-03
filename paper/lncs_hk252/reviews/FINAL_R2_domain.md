# FINAL Review — Reviewer 2 (Domain Expert: Spine MRI / Radiology AI)

**Paper:** *A Frozen BiomedCLIP Branch as a Cross-Dataset Regularizer for Imbalanced Lumbar Disc Grading and Zero-Shot Label-Space Extension*
**Venue:** MIWAI 2026 (12-page LNCS limit)
**Reviewer role:** Independent final review, clinical/radiology lens
**Recommendation:** **ACCEPT with minor revisions** (weak-to-clear accept). The paper is clinically honest where it matters, the headline is not overclaimed, and the radiological reasoning is largely sound. Remaining issues are minor and mostly cosmetic/citation-level; one MAJOR honesty nuance on the SPIDER-T2-FS choice should be sharpened.

---

## 1. Overall assessment

This is a clinically literate submission. Unusually for a deep-learning-on-MRI paper, the authors resist the temptation to dress up a non-deployable model as a finished product. The headline (Severe recall 12.3% → 48.6%, three-seed mean) is reported *with* its accuracy cost (−8.6 pts Mean Accuracy, Table~\ref{tab:main}), *with* a 3-seed std, and *with* an explicit statement (Sec. "Clinical reading") that 48.6% is below the ~80% sensitivity a screening tool needs and is therefore positioned only as a second-reader / triage flag. That framing is exactly what a radiology-AI reader wants to see and is, in my judgement, the single strongest aspect of the paper.

The cross-dataset-regularizer narrative (CBAM helps in-domain Severe, hurts SPIDER transfer, frozen BiomedCLIP restores it) is technically coherent and, importantly, the authors downgrade it honestly to a "descriptive effect size, not a significance test" given n=3 seeds. The zero-shot selective-transfer story is radiologically defensible. I detail the specifics below.

---

## 2. Clinical honesty of the headline — STRONG (no critical issues)

The headline claims are calibrated, not inflated. Specifically:

- **48.6% Severe recall is correctly contextualized as sub-screening.** Sec. "Clinical reading" states the ~80% sensitivity floor and frames the tool as a second reader. This is correct: lumbar central canal / foraminal stenosis screening tools (e.g., Hallinan et al., Radiology 2021, ref `hallinan2021`) operate at much higher sensitivity, and a 48.6% recall tool used autonomously would miss every other Severe case. The second-reader framing is the only honest deployment claim available, and the authors make exactly that claim — no more.
- **The accuracy/recall trade-off is explained mechanistically** (SpineNetV2 "scores high accuracy only by predicting Normal/Mild almost everywhere"), which is true and is the correct reading of an 81.0% accuracy model on an 85% majority class. A naive constant-Normal/Mild predictor would score ~85% accuracy and 0 Severe recall; the paper correctly does not let Mean Accuracy be the headline.
- **AUPRC vs prevalence baseline is reported** (Severe AUPRC 0.333 vs 0.042 prevalence, "7.9× over chance"). Reporting AUPRC against prevalence on a 5% class is the methodologically correct thing to do and many clinical-AI papers omit it. Good.
- **The label-noise caveat** (Sec. "Limitations": RSNA Severe inter-rater agreement not released, so the ceiling reflects label noise as well as capacity) is the right hedge. Inter-rater variability on Pfirrmann/stenosis grading is well documented and the authors acknowledge it.

**MINOR (honesty polish):** The abstract says "Severe-class recall from 12.3% to 48.6%" without the std, while the body carries it. The 3-seed std on Severe recall is not shown anywhere (Table~\ref{tab:main} gives Severe **F1** ±0.017 and Severe AUPRC ±0.042 but **not** Severe Recall ±). Since 48.6% *is the literal headline number*, its variance should appear at least once. Add the Severe-Recall std to Table~\ref{tab:main} (currently Recall appears only as Mean Recall macro 0.592±0.026). Without it a clinical reader cannot tell whether one seed hit 60% and another 38%. **Recommend: add Severe Recall row with ±std.**

**MINOR:** "missing a Severe case costs more than a false alert" (Sec. exp2) is the right clinical intuition but is asserted without a cost reference. One sentence noting that missed severe stenosis can delay surgical decompression would ground it. Optional within page budget.

---

## 3. Cross-dataset-regularizer story — SOUND, with one honesty nuance (MAJOR)

The core claim — CBAM improves in-domain RSNA Severe (F1 0.152→0.262 alone, 0.356 fused) but *degrades* cross-center SPIDER transfer (Mean F1 0.619 vs SpineNetV2 0.646), and the frozen BiomedCLIP branch restores transfer to parity (0.653) — is internally consistent across Tables 1, 3 and the discussion. The interpretation that an attention module overfits center-specific texture/contrast patterns and fails to transfer across scanners/sites is **clinically and technically plausible**: RSNA (US, sagittal T2) and SPIDER (4 Dutch hospitals, T2-FS) differ in scanner vendor, field strength, and sequence weighting, and attention maps keyed to RSNA-specific intensity statistics are a believable source of domain-shift brittleness.

The statistical honesty here is good: the authors explicitly label these as descriptive effect sizes (n=3), report the Hybrid–SpineNetV2 gap as *parity* (Δ=+0.006, <1σ), and only assert the Hybrid–CBAM gap (+0.034, >2σ) and the CBAM-regression (|Δ|/σ=3.8) as the load-bearing findings. They even pre-empt the trivial-ensemble objection by showing softmax-averaging SpineNetV2+BMC reaches only F1 0.45 / Severe 0.18 vs the learned fusion's 0.53 / 0.36 (Sec. `sec:disc`). That is the right control and strengthens the "learned fusion, not averaging" claim.

**MAJOR (clinical-honesty nuance — needs one clarifying sentence):**
The "regularizer" story is partly *confounded by the T2-FS sequence choice* and this is under-acknowledged in the regularizer narrative. The paper trains on RSNA **plain sagittal T2** but evaluates SPIDER on **T2-FS** (fat-suppressed) only (Sec. exp1). Fat suppression substantially changes IVD, marrow, and epidural-fat signal relative to plain T2. So the SPIDER "domain shift" the regularizer is credited with absorbing is not purely center/scanner shift — it is **also a sequence-contrast shift (T2 → T2-FS)**. This matters clinically: the conclusion "CBAM overfits RSNA-specific patterns that do not transfer" is correct, but a large part of what "does not transfer" is fat-suppression contrast, not just hospital identity. The current text frames it almost entirely as "cross-center generalization." A reader could reasonably worry that the regularizer is mostly papering over a T2-vs-T2-FS mismatch that the authors *chose* (they used T2-FS "for consistency with the RSNA-trained encoder," which is itself slightly odd phrasing since T2-FS is *less* like RSNA's plain T2 than SPIDER's available plain-T1 would differ in the other direction). **Recommend:** one sentence acknowledging that the SPIDER shift conflates center *and* sequence-contrast (T2→T2-FS), and that disentangling them is future work. This does not weaken the finding; it makes it honest.

**MINOR:** "for consistency with the RSNA-trained encoder we use the T2-FS sagittal series only" — clarify *why* T2-FS is the consistent choice over SPIDER's T1. As written a radiologist will pause here, because plain T2 (RSNA) is contrast-wise closer to plain spin-echo than to fat-suppressed. If the real reason is that T2-FS was the more complete/registered series in SPIDER, say so.

---

## 4. Zero-shot selective-transfer claim — RADIOLOGICALLY CORRECT

The selective-transfer reading (Table~\ref{tab:spider}) is the most radiologically satisfying part of the paper and I largely endorse it:

- **Disc-morphology labels (bulging 0.613, herniation 0.563, narrowing 0.586) beat OTS BiomedCLIP** — correct, and the explanation (these are semantically closest to RSNA's canal/foraminal stenosis schema, which is *driven by* disc protrusion/bulge encroaching the canal and foramen) is anatomically sound. Stenosis and disc herniation/bulge are mechanistically the same lesion viewed through different label vocabularies, so RSNA supervision *should* transfer here. Good reasoning.
- **Endplate / spondylolisthesis: OTS competitive or better** — also correct. Endplate damage (Schmorl's nodes, irregularity) and spondylolisthesis (vertebral *slip*, an alignment/positional finding) are not in the RSNA stenosis/foraminal schema, so RSNA supervision adds little and the general BiomedCLIP prior wins. Radiologically consistent.
- **The ranking-vs-thresholding split (spondylolisthesis AUC 0.807 but F1 0.028)** is a genuinely insightful observation: the embedding *orders* slip cases well but the fixed cosine threshold mis-calibrates the decision. This is exactly the kind of nuance a domain reviewer wants and it is correctly diagnosed as calibration, not representation, failure.

**The Modic gap analysis (AUC 0.45 ≈ chance) is radiologically the correct call** and I want to commend it specifically. Modic changes are *vertebral endplate/subchondral marrow* signal changes whose **type (1 vs 2 vs 3) is defined by the T1/T2 signal pair** — Modic I is T1-hypo/T2-hyper (edema), Modic II is T1-hyper/T2-hyper/iso (fatty), Modic III is T1/T2-hypo (sclerosis). The paper correctly states that (a) Modic grading conventionally needs paired T1, and (b) SPIDER's T2-FS suppresses exactly the marrow-fat signal Modic II/III depend on. Both facts are accurate. A model given only T2-FS *cannot in principle* separate Modic types, so AUC≈0.45 is the expected, honest result — not a model failure to hide. This is correct radiology and correctly reported. (The `modic1988` citation is the right primary source.)

**MINOR:** The prompts are single fixed templates ("a magnetic resonance image of [label]"). For Pfirrmann (5-class) the zero-shot F1 is ~chance (0.155), which the paper honestly shows. It would help to note that ordinal grades (Pfirrmann 1–5, Modic 0–3) are poorly served by independent text anchors with no ordinal structure — the near-chance result is partly a *prompt-design* limit, not only a signal limit. One clause suffices.

---

## 5. Label definitions + Pfirrmann / Modic / SPIDER citations — CORRECT

I checked the clinical definitions and primary citations against the text:

- **Pfirrmann grades 1–5** (`pfirrmann2001`) — correct primary source (Pfirrmann et al., *Spine* 2001), correct grade count (5), correctly described as disc degeneration grading. ✓
- **Modic changes 0–3 (4 classes)** (`modic1988`) — correct primary source (Modic et al., *Radiology* 1988). NOTE: Modic's original 1988 paper described **types 1–3** (three types). The "type 0 = normal" convention used to make it a 4-class problem (as SPIDER does) is a later/dataset convention, not in Modic 1988. The paper's "0–3 (4 classes)" is fine *as the SPIDER labeling*, but citing only `modic1988` for a 4-class (0–3) scheme is a slight mismatch — Modic 1988 defines the 3 abnormal types, not the 4-class-with-normal encoding. **MINOR:** either say "Modic types 1–3 plus normal (Type 0), per the SPIDER labeling~\cite{vandergraaf2024spider}" or add the SPIDER cite alongside `modic1988`. Currently `modic1988` carries a claim (4 classes) it does not originate.
- **SPIDER dataset** (`vandergraaf2024spider`, van der Graaf et al., *Scientific Data* 2024) — correct primary source, correct description (lumbar MR segmentation dataset, multi-center NL). The "~1,400 IVDs / 218 patients / 4 Dutch hospitals / T1 + T2-FS" description matches the dataset. ✓
- **RSNA 2024** (`kaggle2024`) — appropriately cited as the competition. The "3 conditions × 3 severities" restriction (canal + L/R foraminal, excluding subarticular which needs axial) is a clinically defensible scoping decision and is *correctly justified* (subarticular/recess stenosis is best assessed on axial reformats). ✓ Good clinical judgment to exclude rather than fake it on sagittal.
- **Nigru et al.** (`nigru2024`) used as the "sagittal-only literature ceiling" (foraminal balanced acc ~0.69–0.70). The bib entry is thin (`{Nigru, Saud M. and others}`, no volume/pages) — verify this reference is real and complete before camera-ready. The 18–22% foraminal Severe recall "matching the ceiling" claim leans on this number, so the citation must be solid. **MINOR but verify.**

**MINOR:** `hallinan2021` (Hallinan et al., Radiology 2021 — the canonical deep-learning lumbar stenosis grading paper) is in the .bib but appears **uncited in the text**. This is the most directly comparable clinical prior work (automated central canal / lateral recess / foraminal stenosis grading on lumbar MRI) and should be cited in Related Work, both to satisfy CLAUDE.md rule 5 (cite named clinical standards/prior systems) and because reviewers will expect it. Its omission is the single most notable missing-reference issue. **Recommend citing it in Sec. 2 ("Lumbar disc grading" or a clinical-systems sentence).**

---

## 6. Contribution clarity & value to a clinical-AI audience — CLEAR & VALUABLE

The four contributions are ordered by evidence strength (regularizer finding → architecture → label-space extension → imbalance pipeline), which is honest and unusual. For a clinical-AI audience the valuable, novel pieces are:

1. **The label-space extension via text prompts** — genuinely useful clinically: a model that can answer a *new* grading vocabulary (SPIDER schema) without retraining a fixed head addresses a real friction (every institution/registry uses slightly different grading). This is the freshest contribution.
2. **The honest negative/mixed result** (CBAM hurts transfer; zero-shot is selective; Modic is intractable on T2-FS) — clinical-AI venues are short on papers willing to report where the method *fails*. This is a feature.

The "to our knowledge, first BiomedCLIP on RSNA 2024 + first zero-shot label-space extension for lumbar spine MRI" novelty claim (Sec. 1, Sec. "Cross-Dataset Zero-Shot") is plausible and appropriately hedged with "to our knowledge."

---

## 7. Overclaim check

I found **no serious overclaim**. The paper repeatedly self-limits (parity not superiority on SPIDER aggregate; descriptive effect sizes not significance; second-reader not screening; selective not universal zero-shot). If anything it is *conservatively* framed. The only place to watch:

- **MINOR overclaim risk:** Abstract calls the architecture a "two-branch hybrid model that fuses... for both supervised classification and zero-shot label-space extension" and reports zero-shot mean F1=0.362 prominently, but the body shows zero-shot all-8 mean F1 (0.362) is actually **below** off-the-shelf BiomedCLIP (0.394, Table~\ref{tab:spider}). The abstract reports the 0.362 without the OTS comparator, which could read as "our zero-shot is good" when the honest story (in the body) is "our zero-shot wins *only on disc-morphology* and loses overall to OTS." The body is honest; the abstract should not quote 0.362 in isolation. **Recommend:** in the abstract, append the selectivity ("gains concentrated on disc-morphology labels") — which it *does* already say ("with gains concentrated on disc-morphology labels semantically close to the training schema"). On re-read this is actually adequately hedged. Downgrade to: acceptable, but consider adding that OTS is competitive overall to fully match the body.

---

## 8. Itemized issues by severity

**CRITICAL:** none.

**MAJOR:**
- **M1.** The SPIDER domain shift conflates *center* shift with *sequence-contrast* shift (RSNA plain T2 → SPIDER T2-FS). The "cross-dataset regularizer / CBAM overfits center-specific patterns" narrative should acknowledge that part of what fails to transfer is fat-suppression contrast, not hospital identity. Add one clarifying sentence (Sec. exp1 and/or `sec:disc`). This is an honesty fix, not a result change.

**MINOR:**
- **m1.** Add Severe **Recall** ±std to Table~\ref{tab:main}; 48.6% is the literal headline and its variance is currently absent.
- **m2.** `hallinan2021` (Hallinan et al., Radiology 2021) is in .bib but uncited — cite it in Related Work as the canonical clinical stenosis-grading prior (also satisfies CLAUDE.md rule 5).
- **m3.** Modic "0–3 (4 classes)" cites only `modic1988`, which defines types 1–3, not the 4-class normal-inclusive encoding; add `vandergraaf2024spider` (or note "per SPIDER labeling") for the 4-class scheme.
- **m4.** Justify the T2-FS-only SPIDER choice over T1 ("for consistency with the RSNA-trained encoder" is unconvincing since plain-T2 RSNA is contrast-wise *less* like T2-FS); state the real reason.
- **m5.** Note that near-chance Pfirrmann/Modic zero-shot is partly a *prompt-design* limit (no ordinal structure in independent text anchors), not solely a signal limit.
- **m6.** Verify `nigru2024` is a complete, real citation (currently `{Nigru, Saud M. and others}`, no volume/pages); the foraminal-ceiling claim depends on it.
- **m7.** Abstract: consider noting OTS BiomedCLIP is competitive overall (mean-8 F1 0.394 > hybrid 0.362) so the abstract fully matches the body's honest selectivity story.

**Unused-but-present refs** (no action required for acceptance, but trim if page-tight): `mcsweeney2023`, `yang2026deciphermr`, `blankemeier2026merlin`, `koleilat2025biomedcoop`, `hong2025mscan`, `phaphuangwittayakul2026` — several appear unused. `yang2026deciphermr`/`blankemeier2026merlin` would actually be good one-line cites for the "native-3D MRI VLMs not available during this work" sentence in Related Work/Limitations.

---

## 9. Strengths (for the editor)

- Best-in-class **clinical honesty**: the headline is contextualized against the screening-sensitivity floor and framed as second-reader only.
- **Correct radiology** on the hard cases — the Modic-on-T2-FS impossibility argument is exactly right and well cited.
- **Honest statistics** — descriptive effect sizes over n=3, parity claimed where parity holds, trivial-ensemble control run.
- **Selective zero-shot** result is anatomically reasoned (disc-morphology ≈ stenosis schema) and not oversold.
- Sensible clinical scoping (drops subarticular stenosis rather than faking it on sagittal).

## 10. Verdict

**ACCEPT with minor revisions.** The science is honestly reported and the radiology is correct. Address M1 (disentangle center vs T2-FS contrast shift) and the citation fixes (m2, m3, m6), add the Severe-Recall std (m1), and the paper is solid for MIWAI 2026. No critical barriers; this is the rare imbalanced-medical-imaging paper that does not oversell.
