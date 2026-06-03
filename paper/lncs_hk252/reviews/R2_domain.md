# Domain Review

Reviewer: Peer Reviewer 2 — Domain Expert (lumbar spine MRI / radiology AI)
Venue: MIWAI 2026 (LNAI, double-blind), 12-page limit
Paper: "Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar IVD Grading and Cross-Dataset Zero-Shot Transfer"

## Recommendation + 1-line justification

**Weak Accept (borderline, conditional on minor revisions).** The clinical framing is unusually honest and the domain claims are largely correct and well-cited; the main domain concerns are a few label-definition imprecisions (Modic "0–3" vs "4-class", subarticular justification) and one over-reach in the zero-shot semantic narrative, all fixable within the page budget.

## Scores (0–100)

| Dimension | Score |
|---|---|
| Clinical Validity | 82 |
| Literature Coverage | 76 |
| Domain Contribution | 70 |
| Correctness of Claims | 78 |
| **Overall** | **76** |

## Strengths

1. **Clinically honest, non-overclaimed framing (rare and commendable).** Section "Clinical reading" (lines 315) states plainly that Severe recall 46.4% "is still below the sensitivity (typically above 80%) a screening tool would need" and explicitly positions the system as a *second-reader assistant that flags candidate Severe cases for radiologist confirmation*, not a deployment-ready autonomous tool. This is exactly the correct clinical posture for a model at this operating point, and the second-reader / triage-flag positioning is credible and defensible. The 80% sensitivity benchmark for a screening tool is a reasonable rule-of-thumb.

2. **Correct prioritization of the clinically meaningful failure mode.** The paper correctly identifies that minority-class (Severe) recall, not mean accuracy, is the metric that matters clinically (line 236: "the right clinical trade-off when missing a Severe case costs more than a false alert"). The deliberate acceptance of a 9.3-point Mean Accuracy drop in exchange for 4× Severe recall is the right trade-off and is justified transparently rather than hidden.

3. **Appropriate metric choice for imbalance.** Reporting AUPRC alongside AUC with the prevalence baseline made explicit (line 204: "AUPRC random baseline equals the positive-class prevalence (about 0.05 for Severe)") is methodologically sound and clinically literate — AUC alone is well known to be misleadingly optimistic on rare pathology.

4. **Honest SOTA positioning.** The "not directly comparable" argument (lines 92–93) is, on the merits, *correct and fair*, not a dodge (see Weakness 4 for the one caveat). Lin et al. 0.945 and Liu et al. 0.957 are indeed binary and/or single-level balanced subsets; collapsing to binary does eliminate the Moderate–Severe boundary, which is the clinically hardest and most consequential distinction (e.g., the threshold at which surgical consultation is considered). Binarizing also removes the very class imbalance that defines the problem. This is a legitimate and well-articulated distinction, and the paper commits to the harder setting rather than cherry-picking the easier one.

5. **Pfirrmann/Modic/SPIDER citations are present and correct.** Pfirrmann grades 1–5 → `pfirrmann2001` (Spine 2001), Modic changes → `modic1988` (Radiology 1988), SPIDER → `vandergraaf2024spider` (Scientific Data 2024) are all the canonical primary sources and are correctly attached (line 184). Datasets/standards are cited per good practice.

6. **Domain-plausible selective-transfer pattern.** The finding that disc-morphology labels (bulging F1 0.709, herniation 0.522, narrowing 0.388) transfer far better than Modic (0.117) / Pfirrmann (0.147) / spondylolisthesis (0.030) is *clinically and semantically coherent* (see detailed assessment below) and is one of the more interesting domain contributions.

## Weaknesses (numbered: issue / location / fix / severity)

### W1 — "Modic (4-class)" vs. "Modic 0–3": internal inconsistency in label definition
- **Issue:** Section "SPIDER" (line 184) defines "Modic changes 0--3" (which is the correct clinical convention: type 0 = no change, types I/II/III). But Table 2 (line 257) and Section "Supervised Transfer" (line 270) call it "Modic (4-class)". Modic *types* are classically I, II, III (three pathological types); "Modic 0–3" treats "no Modic change" as a fourth class, which is defensible as a classification target but the paper should not silently alternate between "0–3" and "4-class" without one sentence reconciling them. A spine radiologist reading "Modic 4-class" without the 0–3 definition may assume a non-standard scheme.
- **Location:** line 184 vs. lines 257, 270.
- **Fix:** State once, e.g., "Modic changes, encoded as 4 classes (0 = none, I/II/III)", and use "Modic (0–III)" or "Modic 4-class" consistently thereafter. One sentence.
- **Severity:** MINOR.

### W2 — Subarticular-exclusion justification is correct but under-supported by citation
- **Issue:** The restriction to 3 of 5 RSNA conditions — keeping canal stenosis + L/R foraminal narrowing, excluding L/R subarticular (lateral recess) stenosis as "best read on axial reformats" (lines 182, 304) — is *clinically correct*: subarticular/lateral-recess stenosis is indeed optimally assessed on axial T2, and foraminal narrowing on sagittal. This is genuine scientific scoping, not merely convenient. However, the claim is asserted without a supporting radiology citation, while a directly relevant one (`hallinan2021`, Radiology — "Deep Learning Model for Automated Detection and Classification of Central Canal, Lateral Recess, and Neural Foraminal Stenosis at Lumbar Spine MRI") already sits *unused* in the .bib (line 1).
- **Location:** lines 182, 304; unused ref at references.bib line 1.
- **Fix:** Cite `hallinan2021` (or a comparable RSNA/lateral-recess source) at the sentence justifying the sagittal-vs-axial split. This converts a bare assertion into a supported design choice and costs zero net references (the entry already exists). Per project rule 5, every named clinical standard/condition split should be cited.
- **Severity:** MINOR (but easy win, and strengthens Weakness-4-type objections from non-domain reviewers).

### W3 — T2-FS-only SPIDER restriction: justified, but conflates two different rationales
- **Issue:** SPIDER provides T1 and T2-FS sagittal series; the paper uses "the T2-FS sagittal series only" "for consistency with the RSNA-trained encoder" (line 184). RSNA 2024 uses sagittal **T2** (not fat-suppressed). T2-FS (fat-suppressed) and standard T2 differ substantially in contrast — fat-suppression markedly alters marrow signal, which is *exactly* the signal Modic changes are read on. So choosing T2-FS for "consistency" with a T2-trained encoder is arguably the *less* consistent choice, and may partly explain the catastrophic Modic transfer (0.117). The restriction to a single sequence is reasonable scoping for a 12-page paper, but the stated rationale ("consistency") is questionable and the domain-mismatch is not acknowledged as a confound.
- **Location:** line 184; Modic result line 257; explanation lines 264–265.
- **Fix:** Either (a) correct the rationale to the honest one ("we restrict to a single sequence to control series-count; T2-FS was chosen as the closest available, acknowledging a contrast gap vs. RSNA's non-fat-suppressed T2 that may further penalize marrow-signal labels such as Modic") or (b) add one clause to the Modic discussion noting that the T2-FS vs T2 contrast difference is a *second* cause of poor Modic transfer beyond the "semantic distance" explanation currently given. One–two sentences.
- **Severity:** MAJOR. This directly affects the correctness of the headline zero-shot interpretation: the paper attributes poor Modic/Pfirrmann transfer purely to *semantic distance in the pretraining corpus* (line 265), but a non-trivial part of the Modic failure is plausibly an *imaging-physics domain gap* (fat-suppressed marrow signal unseen at RSNA training). The single-cause "semantic proximity" story is incomplete.

### W4 — Selective-transfer explanation is partly sound, partly hand-wavy
- **Issue:** The disc-morphology-vs-rest split is clinically coherent: disc bulging/herniation/narrowing are *morphological* findings on the disc silhouette that a contrastive image-text model trained largely on captioned figures could plausibly latch onto, and they share semantics with RSNA's canal/foraminal "narrowing" vocabulary. Spondylolisthesis (vertebral *slip*, a positional/alignment finding) and Modic (vertebral *marrow* signal) are anatomically distinct from disc morphology, so weak transfer is expected. So far so good. **However:** (a) spondylolisthesis F1 = 0.030 *but AUC = 0.745* (line 258) — a high AUC with near-zero F1 means the ranking signal exists but the operating threshold is wrong; the text says "semantically distant ... measurable domain gap" but the AUC 0.745 actually contradicts a pure "the model can't see it" reading. This nuance is not discussed. (b) Pfirrmann AUC 0.492 and Modic AUC 0.422 are *below chance* (0.5) — a sub-0.5 AUC is not merely "a domain gap," it indicates the cosine-prompt ordering is *anti-correlated* with severity (likely a prompt-wording / ordinal-collapse artifact), which is a different and arguably more serious failure than the text implies.
- **Location:** lines 256–258, 264–265.
- **Fix:** Add one sentence distinguishing the two failure modes visible in the table: (i) spondylolisthesis — *rankable* (AUC 0.745) but un-thresholded (F1 0.030), i.e., a calibration not a perception failure; (ii) Pfirrmann/Modic — *sub-chance AUC*, i.e., the ordinal label is not linearly recoverable from a single cosine axis (prompt design limitation), not just "semantic distance." This makes the analysis sharper and more credible to a domain reader.
- **Severity:** MAJOR (the current single-axis "semantic proximity" explanation glosses over a sub-chance-AUC result that undermines it).

### W5 — Pfirrmann/Modic as flat multi-class ignores their ordinal clinical structure
- **Issue:** Pfirrmann (I–V) and Modic (0/I/II/III) are **ordinal** scales with clinically graded meaning (Pfirrmann IV→V is the herniation-risk boundary; Modic I vs II carries different prognosis). The paper treats them as flat 5-class / 4-class targets via argmax cosine (lines 256–257, 270) with macro-F1, which discards ordinality and is partly why a single cosine axis fails (W4). A domain reviewer expects at least acknowledgement that ordinal grading ≠ nominal classification, and that adjacent-grade confusion (Pfirrmann IV↔V) is clinically less harmful than distant confusion (II↔V).
- **Location:** lines 256–257, 298 (Pfirrmann Grade 4 metric reported in isolation).
- **Fix:** One sentence in Limitations acknowledging that Pfirrmann/Modic are ordinal and that flat macro-F1 / argmax-cosine ignores grade adjacency; optionally note that an ordinal-aware metric (e.g., mean absolute grade error or linearly-weighted kappa) is left to future work. No new experiments needed.
- **Severity:** MINOR.

### W6 — "First BiomedCLIP on RSNA 2024" / "first cross-dataset zero-shot" novelty claims
- **Issue:** Two "to our knowledge, first" claims (lines 76, 240). These are *plausibly* true for the specific combination, but RSNA 2024 is a heavily worked Kaggle dataset (thousands of public notebooks) and SPIDER↔external transfer is an active area. "To our knowledge, no prior peer-reviewed work" (line 76) is appropriately hedged with "peer-reviewed," which is the right qualifier. The zero-shot claim (line 240) is hedged with "to our knowledge." These are acceptable as written, but a domain reviewer notes the claim's strength rests entirely on the "peer-reviewed" and "to our knowledge" qualifiers — if a reviewer knows of a workshop/preprint counterexample it collapses. Low risk given the hedging.
- **Location:** lines 76, 240.
- **Fix:** Keep the hedges; optionally soften "uniquely enables" in the Summary (line 322) to "enables, to our knowledge for the first time," matching the hedged Intro claim. Minor wording.
- **Severity:** MINOR.

### W7 — Nigru "sagittal-only literature ceiling" used slightly loosely
- **Issue:** `nigru2024` is invoked three times as the "sagittal-only literature ceiling" (foraminal F1 ~0.84 binary, balanced accuracy ~0.69–0.70; lines 84, 304). The numbers are used consistently, but: (a) the bib entry for `nigru2024` is thin ("Nigru, Saud M. and others", no volume/pages/DOI), which for a *load-bearing* external-validation citation is weak for a domain venue; and (b) the paper equates *its own* foraminal Severe recall 18–22% with Nigru's *balanced accuracy* 0.69–0.70 (line 304) — these are different metrics on different label granularities (binary balanced-accuracy vs 3-class Severe recall), so "matching the ... ceiling" is an apples-to-oranges comparison even though the qualitative point (sagittal foraminal reading is hard) is valid.
- **Location:** lines 84, 304; references.bib lines 11–16.
- **Fix:** (a) Complete the `nigru2024` bibentry (volume/pages/DOI) — it is cited as the central external benchmark and must be verifiable. (b) Reword line 304 so it does not directly equate Severe-recall 18–22% with balanced-accuracy 0.69–0.70; say instead that both point to the *same qualitative ceiling* on sagittal foraminal reading rather than implying numeric equivalence.
- **Severity:** MAJOR for the incomplete bibentry (verifiability of the key external benchmark in a double-blind domain review); MINOR for the metric-mismatch wording.

### W8 — Inter-rater variability claim under-cited
- **Issue:** The opening sentence (line 64) grounds the whole motivation in "substantial inter-rater variability," cited only to `nigru2024`. Inter-rater variability of Pfirrmann/stenosis grading is a well-established radiology finding with primary sources; pinning it to a single thin external-validation paper undersells a claim that is foundational to the paper's clinical rationale.
- **Location:** line 64.
- **Fix:** Within page budget this is optional, but if room exists, add the canonical inter-rater source (Pfirrmann's own 2001 paper reports kappa, and is already in the bib) — i.e., cite `pfirrmann2001` here too, costing zero new references.
- **Severity:** MINOR.

## Missing References (if any, with reason)

Given the strict 12-page limit, I do **not** demand a broad survey. Two targeted points:

1. **`hallinan2021` (already in .bib, currently uncited).** Should be *cited* (not added — it's already there) at the subarticular/lateral-recess vs sagittal/foraminal justification (W2). It is the canonical deep-learning lumbar-stenosis paper covering exactly central canal / lateral recess / foraminal stenosis and directly supports the 3-of-5 condition split. Net references: zero change.

2. **No new references required.** Coverage of the load-bearing domain anchors (SpineNet/V2, Pfirrmann, Modic, SPIDER, BiomedCLIP, CBAM, focal loss, RSNA SOTA Lin/Liu) is adequate for a 12-page applied paper. I would *discourage* adding the currently-unused 3D-VLM references (`yang2026deciphermr`, `blankemeier2026merlin`, `koleilat2025biomedcoop`, `hong2025mscan`, `mcsweeney2023`, `phaphuangwittayakul2026`) unless they are actually cited — they currently sit dead in the .bib and either should be cited (1 sentence each, in Related Work, only if room) or left out. Dead bib entries are harmless to the compiled PDF but signal an unpolished submission if the .bib is shared.

## Questions to Authors

1. **(W3, central)** RSNA is sagittal T2 (non-fat-suppressed); SPIDER T2-FS is fat-suppressed. Did you verify that the poor Modic transfer (F1 0.117) is driven by *semantic distance* rather than the *fat-suppression marrow-signal contrast gap*? Would T1 or non-FS T2 (if available in SPIDER) plausibly transfer Modic better? This distinction changes the headline interpretation.

2. **(W4)** Spondylolisthesis shows AUC 0.745 but F1 0.030, and Pfirrmann/Modic show *sub-chance* AUC (0.492 / 0.422). How do you reconcile the "semantic proximity" explanation with (a) a rankable-but-mis-thresholded spondylolisthesis and (b) a *below-chance* Pfirrmann/Modic ordering? Is the latter a prompt-wording artifact rather than a domain gap?

3. **(W1)** Is Modic encoded as 0/I/II/III (4 classes including "no change") or as the three pathological types only? Please reconcile "Modic 0–3" (text) with "Modic 4-class" (tables).

4. **(Clinical)** For the second-reader positioning: at the 46.4% Severe-recall operating point, what is the corresponding Severe *precision* / false-alert rate? A second-reader that flags cases is only useful if its false-alert burden is acceptable to a radiologist; Table 1 gives Severe precision only implicitly via F1/AUPRC. A one-line statement of the Severe precision at this operating point would strengthen the clinical claim.

5. **(W5)** Did you consider an ordinal-aware evaluation (weighted kappa / mean grade error) for Pfirrmann and Modic? Adjacent-grade errors are clinically less harmful than distant ones, and flat macro-F1 penalizes them equally.

---

## SHORT SUMMARY FOR SYNTHESIZER

**Recommendation: Weak Accept (borderline), conditional on minor revisions.** Clinically the most honest paper in this batch — second-reader positioning is credible, the 46.4% Severe-recall-vs-80%-screening-threshold gap is stated openly, AUPRC/imbalance handling is domain-literate, and the SOTA "not directly comparable" argument (binary / single-level balanced subsets) is fair, not a dodge. Label citations (Pfirrmann 2001, Modic 1988, SPIDER 2024) are correct. The selective zero-shot transfer (disc morphology transfers, Modic/Pfirrmann/spondylolisthesis don't) is clinically plausible.

Top 3 issues by severity:
1. **MAJOR — T2-FS vs T2 confound (line 184, 264–265):** poor Modic transfer is attributed purely to "semantic distance," but SPIDER T2-FS (fat-suppressed) vs RSNA T2 is an imaging-physics marrow-signal domain gap that the paper doesn't acknowledge; the single-cause story is incomplete. The stated "for consistency" rationale for choosing T2-FS is itself questionable.
2. **MAJOR — sub-chance AUC undermines the zero-shot narrative (lines 256–258):** Pfirrmann AUC 0.492 / Modic 0.422 are *below* chance, and spondylolisthesis is AUC 0.745 but F1 0.030; the "semantic proximity" explanation glosses over the difference between un-thresholded-but-rankable vs anti-correlated failures. Also the central external benchmark `nigru2024` has an incomplete (unverifiable) bibentry and its 0.69–0.70 balanced-accuracy is equated apples-to-oranges with the paper's 18–22% Severe recall.
3. **MINOR — Modic "0–3" vs "4-class" inconsistency (line 184 vs 257/270)**, subarticular-exclusion lacks a citation despite `hallinan2021` sitting unused in the .bib, and Pfirrmann/Modic ordinality is ignored (flat macro-F1). All one-sentence fixes within the page limit.
