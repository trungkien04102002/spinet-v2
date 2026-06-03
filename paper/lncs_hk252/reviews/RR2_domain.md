# RR2 — Domain (Clinical / Spine-MRI) Re-Review

**Manuscript:** Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar IVD Grading and Cross-Dataset Zero-Shot Transfer (MIWAI 2026)
**Round:** Re-review (revised submission)
**Reviewer lens:** Spine MRI / radiology-AI domain expert
**Recommendation:** **Accept with minor revisions** (clinical honesty materially improved; remaining issues are MINOR/MAJOR-but-fixable in camera-ready)

---

## 1. Verdict on the revision

The revision is a substantial improvement in *clinical honesty*. The three changes that matter most to a domain reviewer all landed:

- **3-seed reporting** removes the previous single-run cherry-pick concern. The headline is now a mean (12.3% → 48.6% Severe recall) with std, not a best-of-N number.
- **Honest disclosure that OTS BiomedCLIP beats the hybrid on overall zero-shot mean F1 (0.394 vs 0.362)** is exactly the kind of negative result that builds reviewer trust. The authors did not bury it; it is stated in the abstract-adjacent narrative, the table caption, and the §4.3 text.
- **The trivial-ensemble control (§4.5)** pre-empts the most obvious "your fusion is just a logit average" objection and answers it with numbers.

This is now a paper I would defend to a clinical audience as *truthful about its limits*. The contribution still stands (see §6 below). My remaining concerns are about precision of clinical language and two radiological claims, not about integrity.

---

## 2. Is the headline still clinically honest? (Q1)

**Largely yes. No material overclaim.** Specifics:

- **Severe recall 12.3 → 48.6% (≈4×), 3-seed mean.** Honestly reported as a mean ± std, and the paper explicitly states this is "still below the sensitivity (typically above 80%) a screening tool would need" and frames the system as a **second-reader assistant** (§4.6 "Clinical reading"). This is the correct framing and the correct sensitivity benchmark — for a screening/triage tool the relevant operating point is high sensitivity, and 48.6% does not reach it. **Strength.**

- **The accuracy trade-off is disclosed, not hidden.** The 8.6-point Mean-Accuracy drop is shown in Table 1 (SpineNetV2 81.0 → Hybrid 72.4) and explained as the intended majority/minority rebalancing. Clinically defensible: in a miss-costly screening setting, trading specificity for sensitivity is the right direction. **Strength.**

- **MINOR — "Severe AUC 0.899" risks being read as deployment-grade.** AUC on a 5%-prevalence class is exactly the metric the authors *themselves* warn against earlier in §4.1 ("AUC can stay high while a model fails to rank Severe cases meaningfully"). The Summary (§5) leads with "AUC 0.899" while the body leads with the more honest F1/recall + AUPRC 0.333. Recommend the Summary foreground Severe **F1 0.356 / recall 48.6% / AUPRC 0.333 (7.9× over prevalence)** and treat AUC as secondary, to stay self-consistent with §4.1. Otherwise a skim-reader takes away "0.9 AUC = nearly solved," which the authors do not intend.

- **MINOR — "second-reader assistant" needs one caveat.** A 48.6%-sensitivity flagging tool used as a second reader can induce *automation bias* and *de-anchoring*: a reader who sees "no Severe flag" may down-weight their own suspicion, and the tool misses >50% of Severe cases. The current text presents second-reader use as obviously safe; one sentence acknowledging that a sub-80% sensitivity assistant must be positioned as *additive* (flags to confirm) and never as *rule-out* would make the clinical framing airtight. As written it is defensible but slightly optimistic.

- No overclaim found in the abstract. The phrase "matches SpineNetV2 ... rather than a significance test" and "descriptive effect size" language is appropriately hedged for n=3 seeds.

---

## 3. Zero-shot story: OTS-wins-overall / hybrid-wins-on-disc-labels (Q2)

The nuanced framing is **plausible and well-argued**, with two caveats.

### 3a. "Semantic proximity to RSNA training" explanation — PLAUSIBLE, with a wording risk
The claim is: the hybrid beats OTS on the three disc-morphology labels (bulging 0.613 vs 0.363, herniation 0.563 vs 0.532, narrowing 0.586 vs 0.403) because those are *closest to the RSNA training schema*, and loses on labels far from it (endplate, spondylolisthesis).

- **Technically coherent.** RSNA 2024's three evaluated conditions are canal stenosis and L/R foraminal narrowing — all driven by disc bulge/herniation/height-loss morphology. So a model fine-tuned to read disc morphology *should* transfer to SPIDER's bulging/herniation/narrowing. The mechanism is sound.
- **MINOR caveat — RSNA does not label "herniation" or "bulging" per se.** RSNA labels *stenosis severity*, not disc morphology directly. The transfer benefit is therefore via a *correlated* morphological substrate, not a same-label transfer. The current wording ("closest to the RSNA training schema") is fine but could be read as implying RSNA *contains* these labels. One clause — "RSNA grades the stenosis these morphologies cause, not the morphology itself" — would tighten it. Not blocking.
- **Spondylolisthesis AUC 0.807 but F1 0.028** is correctly attributed to threshold mis-calibration (good ranking, bad fixed cosine threshold). Radiologically sensible: spondylolisthesis is a *vertebral slip / alignment* finding, not an intra-disc signal finding, so a disc-centric embedding can still rank it (slips co-occur with disc/facet degeneration) yet a single fixed threshold mis-calibrates the rare positive. The ranking-vs-thresholding distinction is the right explanation. **Strength.**

### 3b. The T2-FS-vs-T2 Modic argument — RADIOLOGICALLY CORRECT (MAJOR strength, one precision fix)
"The Modic gap (AUC 0.45) is also an imaging-physics effect: SPIDER's fat-suppressed T2-FS alters the marrow signal Modic grading relies on."

- **This is correct and is the single most clinically literate sentence in the paper.** Modic changes are *defined by vertebral endplate/subchondral marrow signal*: Modic I = T1-hypo / T2-hyper (edema/inflammation), Modic II = T1-hyper / T2-iso-hyper (**fatty marrow conversion**), Modic III = T1/T2-hypo (sclerosis). Modic II — the most common type — is literally a *fat* signal. **Fat suppression (T2-FS) by design nulls exactly the signal that distinguishes Modic II**, and also flattens the marrow-edema contrast. So an RSNA encoder trained on non-fat-suppressed T2, applied to SPIDER T2-FS, *cannot* see the marrow-signal axis Modic depends on. AUC ≈ 0.45 (sub-chance) is consistent with that.
- **MINOR precision fix:** Strictly, Modic grading is classically a *combined T1 + T2* read (T1 is what separates Modic II fat from Modic I edema). The paper trains/evaluates on T2-FS only and never had T1 marrow contrast even in-modality. The honest framing is: *Modic is under-determined from a single T2-FS sequence regardless of model* — i.e., this is partly a **task-input mismatch (missing T1), not only a fat-suppression artifact.** The current sentence is correct but slightly under-sells the point; adding "and Modic grading conventionally requires the paired T1 sequence, absent here" would make it bulletproof and convert a MINOR weakness into a strength. The authors already use only T2-FS (§4.1) so this is internally consistent.

---

## 4. Prompt strings (Q3)

Disclosed prompts: template `"a magnetic resonance image of [label]"`, e.g. `"lumbar disc herniation"` vs `"no disc herniation"`, and (from the brief) `"pfirrmann grade 4 severe disc degeneration"`.

- **Disclosure itself is a strength** — a priori, no SPIDER tuning, stated to avoid target leakage. Good for reproducibility and integrity.
- **Clinically sensible, mostly.** "lumbar disc herniation," "disc bulging," "disc narrowing" are standard radiology-report phrasings and should align with PMC-15M caption language — consistent with the hybrid's success on those three.
- **MINOR — some prompts are likely *why* certain transfers fail and deserve a sentence:**
  - **"disc narrowing"** is ambiguous. Radiologists write **"disc space/height narrowing"** or **"loss of disc height"**; "disc narrowing" alone can be read as canal/foraminal narrowing. It still worked (0.586), but the ambiguity is worth noting.
  - **Modic prompts:** there is no good short text prompt for Modic I/II/III — these are *signal-pattern* descriptors ("type I endplate marrow edema," "type II fatty endplate change"), not lay terms, and are sparsely captioned in PMC. Part of the Modic failure is plausibly **prompt under-specification**, compounding the T2-FS physics. The paper attributes the Modic gap entirely to fat suppression; it should acknowledge prompt expressivity as a co-factor.
  - **"pfirrmann grade 4 severe disc degeneration"** — defensible: Pfirrmann IV = T2-hypointense disc, reduced height, indistinct nucleus/annulus boundary. But Pfirrmann is an *ordinal* 5-grade scale and treating five independent cosine prompts discards ordinality; the F1 0.155 (barely above OTS 0.152) reflects that a flat cosine head cannot exploit grade ordering. This is a **method limitation surfaced by the prompt design**, worth one sentence (currently only attributed to "threshold mis-calibration").
  - **"spondylolisthesis"** — as noted, an *alignment* finding poorly captured by a disc-centric crop and a single morphological prompt; the near-zero F1 is partly a prompt/anatomy mismatch, not just thresholding.
- **No prompt is clinically *wrong*.** None would make a radiologist object outright; the issues are *expressivity/specificity*, not correctness.

---

## 5. SPIDER label definitions + Pfirrmann/Modic citations (Q4)

Checked §4.1 against `references.bib`:

- **Pfirrmann grades 1–5 `\cite{pfirrmann2001}`** — CORRECT. Pfirrmann et al., *Spine* 2001;26(17):1873-1878. Canonical, correct grade count (5). ✓
- **Modic changes 0–3 (4 classes) `\cite{modic1988}`** — CITATION CORRECT (Modic et al., *Radiology* 1988;166(1):193-199). **MINOR nomenclature flag:** classical Modic is types **I/II/III** (3 types); "Modic 0" = *no* Modic change. So "0–3 (4 classes)" = {none, I, II, III} is a legitimate *classification-task* encoding (and matches SPIDER's released labels), but a purist reader may note that Modic himself described 3 types, with "0/none" added as the negative class. One half-sentence ("type 0 = no Modic change") would pre-empt the nitpick. The 1988 citation is for the original 3-type description; fine.
- **SPIDER `\cite{vandergraaf2024spider}`** — CORRECT (van der Graaf et al., *Scientific Data* 2024). Cohort description (≈218 patients, 4 Dutch hospitals, T1 + T2-FS) matches the published dataset. ✓
- **Label list** (Pfirrmann, Modic, spondylolisthesis, herniation, narrowing, bulging, upper/lower endplate damage) matches the SPIDER grading release. ✓
- **MINOR consistency note:** §4.1 lists Modic as a SPIDER label and the supervised-transfer table (§4.4) reports Modic, but Table 2 zero-shot lists "Modic (4-class)." Consistent across tables. Good. The eight zero-shot labels in Table 2 = bulging, herniation, narrowing, upper EP, lower EP, Pfirrmann, Modic, spondylolisthesis — matches "eight unseen labels." ✓ (Note endplate-damage is split upper/lower to reach 8, while §4.1 lists it as one category "upper/lower endplate damage"; harmless but the count bookkeeping is worth a glance.)

**Conclusion on Q4: citations are correct after the edits.** Only nomenclature-precision MINORs remain.

---

## 6. Does the contribution still stand now that OTS-overall beats hybrid? (Q5)

**Yes — and arguably the honest disclosure *strengthens* the paper.** The value proposition is clear if the reader parses the two-axis claim:

1. **In-domain (RSNA):** the hybrid is unambiguously the best config on the imbalance metrics that matter clinically (Severe F1 0.356 best, Severe AUPRC 0.333 best, 3 seeds). This is the primary, well-supported result. **The OTS-beats-hybrid result is on a *different* dataset/task (SPIDER zero-shot) and does not touch this.**

2. **Zero-shot SPIDER:** the honest reframe is *not* "we beat OTS overall" (they don't) but **"RSNA supervision adds transferable value precisely where the target is semantically close (disc morphology: +0.25 on bulging, +0.18 on narrowing), and OTS is a strong floor elsewhere."** That is a *legitimate, useful, and falsifiable* finding for a clinical-AI audience: it tells a practitioner *when* to expect RSNA-pretraining to help on a new label space and when to fall back to off-the-shelf foundation features. This is more useful than a blanket "we win" claim.

3. **The cross-dataset *regularizer* finding (the reframed lead contribution)** — CBAM helps in-domain but *hurts* SPIDER transfer (0.619 < 0.646 SpineNetV2), and BiomedCLIP restores it (0.653) — is independent of the OTS comparison and stands on its own as the most novel scientific point. For a domain audience this is the interesting message: *attention modules tuned on one center's protocol can overfit that center*, and a frozen foundation branch buys back generalization.

**Where the value prop is at risk (MAJOR-but-fixable):** the paper now carries *three competing "lead" framings* — (a) Severe-recall imbalance fix, (b) zero-shot label-space extension, (c) CBAM-overfits / BiomedCLIP-regularizes. The abstract and contribution list order them by "strength of multi-seed evidence" (regularizer first), but the *title* and Summary lead with the hybrid architecture and Severe gains. A clinical reader can finish unsure what the single takeaway is. **Recommend stating one sentence of the form:** "The clinically actionable result is the in-domain Severe-recall recovery; the scientific result is that a frozen VLM branch regularizes a center-overfitting attention module; zero-shot transfer is a capability demonstration, strongest on RSNA-adjacent disc labels." That single triage sentence would resolve the ambiguity.

---

## 7. Issue list (classified)

### CRITICAL
- None. No integrity, overclaim, or factual-radiology error rises to critical after this revision.

### MAJOR
- **M1 (value-prop clarity).** Three competing lead framings; no single explicit takeaway sentence. Add one triage sentence in the Summary distinguishing the *clinical* result (Severe recall) from the *scientific* result (BiomedCLIP-as-regularizer) from the *capability demo* (zero-shot). (§6 above.)
- **M2 (Modic argument completeness).** The T2-FS explanation is correct but incomplete: Modic grading conventionally requires the paired **T1** sequence (to separate Modic II fatty from Modic I edematous marrow), which is absent here. State that Modic is *under-determined from T2-FS alone regardless of model*, so AUC≈0.45 is partly a task-input limit, not only fat-suppression. Converts a borderline claim into a fully defensible one.

### MINOR
- **m1.** Summary (§5) leads with Severe AUC 0.899, contradicting §4.1's own warning that AUC over-flatters on 5%-prevalence classes. Foreground F1/recall/AUPRC instead.
- **m2.** "Second-reader assistant" framing should add one caveat about automation bias / not-a-rule-out given sub-80% sensitivity.
- **m3.** Clarify that RSNA grades *stenosis severity*, not disc morphology directly — the disc-label transfer is via a correlated morphological substrate ("closest to RSNA schema" could mislead).
- **m4.** Prompt expressivity is a co-factor in the Modic / Pfirrmann / spondylolisthesis failures (ordinal Pfirrmann collapsed to flat cosine; Modic has no good short prompt; spondylolisthesis is an alignment not disc finding). Currently attributed only to thresholding / fat suppression. One sentence.
- **m5.** "disc narrowing" prompt is ambiguous vs the standard "loss of disc height / disc-space narrowing"; note it.
- **m6.** Modic "0–3 (4 classes)" — add "type 0 = no Modic change" to pre-empt the I/II/III-vs-0–3 nomenclature nitpick. Citation itself correct.

---

## 8. Strengths (for the editor's balance)

- 3-seed reporting + descriptive-effect-size framing instead of significance claims at n=3 — methodologically and ethically correct.
- Honest OTS-beats-hybrid disclosure with a *mechanistic* (not defensive) explanation.
- Trivial-ensemble control (§4.5) directly refutes the "fusion = logit average" objection with numbers.
- AUPRC reported alongside AUC with the prevalence baseline stated — exactly right for imbalanced clinical data.
- The T2-FS/Modic and ranking-vs-thresholding analyses show genuine radiological literacy, rare in ML-venue submissions.
- Citations for Pfirrmann, Modic, SPIDER, BiomedCLIP, RadCLIP all verified correct.
- Sagittal-only scope honestly bounded (axial-dependent subarticular/foraminal reads deferred, ceiling benchmarked against Nigru et al.).

## 9. Bottom line
Clinically honest: **yes.** Contribution clear: **yes, once M1's one-sentence triage is added.** No CRITICAL issues; 2 MAJOR (both fixable in camera-ready by adding sentences, not by new experiments) and 6 MINOR. The honest negative result on zero-shot *improves* the paper's credibility for a clinical-AI audience. **Accept with minor revisions.**
