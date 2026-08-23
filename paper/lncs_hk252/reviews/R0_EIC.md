# EIC Review

## Summary of Submission
The paper presents a two-branch hybrid model for lumbar IVD grading on RSNA 2024 that fuses a trainable 3D-CBAM ResNet-34 with a frozen BiomedCLIP image-text encoder via a small concat-MLP, plus a cosine-similarity head that enables text-prompt zero-shot label-space extension. On RSNA the hybrid lifts Severe-class recall from 11.5% to 46.4% and Severe F1 from 0.149 to 0.343; on SPIDER it reaches mean F1 = 0.362 zero-shot and, under 3-seed supervised transfer, matches the baseline (F1 0.653 vs 0.646) while recovering above the CBAM-only variant (0.619), framed as a cross-dataset regularizer story.

## Overall Recommendation
**Minor Revision** — The work is a well-scoped, honest, venue-appropriate applied-AI contribution with a genuine novelty hook (BiomedCLIP zero-shot label-space extension for lumbar MRI); the main blemishes are framing/consistency issues and a single-seed RSNA headline, all fixable without new experiments.

## Scores (0-100)
- Originality: 70
- Significance: 66
- Venue Fit: 85
- Clarity: 80
- Overall: 72

## Strengths
- **Strong venue fit for MIWAI/LNAI applied-AI.** Combines foundation models (BiomedCLIP), attention (CBAM), and a deployment-relevant medical problem; the second-reader framing in Sec.~\ref{sec:disc} ("Clinical reading", line 315) matches the conference's applied, multi-disciplinary readership.
- **Genuine and clearly-bounded novelty claim.** Lines 76 and 240 claim first BiomedCLIP application to RSNA 2024 lumbar classification and first RSNA→SPIDER text-prompt zero-shot label-space extension. The claim is specific ("peer-reviewed", "same architecture") rather than an open-ended "novel."
- **Intellectual honesty is well above the mid-tier norm.** The authors explicitly decline to over-claim: hybrid is at parity with baseline on SPIDER aggregate ("we do not claim the hybrid surpasses the baseline here", line 295), the regularizer story is hedged with a stated alternative they cannot rule out (trivial logit-ensemble, line 313), and the clinical limitation (46.4% recall is below the ~80% a screening tool needs, line 315) is stated plainly.
- **Honest positioning against prior RSNA work.** The "Positioning against prior RSNA 2024 methods" paragraph (lines 92-93) correctly explains why the 0.945/0.957 binary numbers are not comparable, instead of cherry-picking them. This is the right scholarly move.
- **Appropriate metric discipline for the imbalance setting.** Reporting AUPRC alongside AUC with the prevalence baseline (line 204), and macro-averaging justified explicitly, is exactly right for a paper whose whole point is the 5% Severe class.
- **Reproducibility signals.** Fixed splits (random_state=42), named seeds {42,123,456}, hardware and cost (line 206), and a clear trainable-parameter budget (1.18M of ~218M, line 158).

## Weaknesses / Concerns

1. **The title/abstract sell an accuracy-improvement story the SPIDER results do not support.**
   - Location: title (line 36), abstract (line 56), vs. Sec.~\ref{sec:spider_transfer} (line 295) and Sec.~\ref{sec:disc} (line 313).
   - Issue: The title leads with "Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings" and the abstract foregrounds Severe-class lifts, but the paper's most defensible multi-seed finding is the *regularizer* result (hybrid restores baseline-level transfer that CBAM erodes), and the hybrid is explicitly NOT better than a plain ResNet-34 on SPIDER aggregate. A reader expecting a uniformly winning architecture will feel oversold by the framing even though the body is honest. From a readership-relevance angle, this mismatch is the single biggest risk to how the paper lands.
   - Proposed fix: No new experiments. Rebalance the abstract so the "regularizer + label-space extension" framing (already the ordering of contributions in lines 68-74) leads, and the RSNA Severe lift is presented as the in-domain benefit rather than the headline. One or two sentences.
   - Severity: MAJOR

2. **Single-seed RSNA headline numbers undercut the paper's own significance.**
   - Location: Table~\ref{tab:main} (lines 212-232), limitation noted at line 317.
   - Issue: Every RSNA number in the abstract and Table 1 (Severe F1 0.149→0.343, recall 11.5%→46.4%, the 2.6-point fusion gain) comes from a single seed (42), while SPIDER gets 3 seeds. Because the paper repeatedly argues that small effects (e.g. +0.034 on SPIDER) require >2σ to be credible, applying single-seed point estimates to the headline RSNA claims is internally inconsistent and weakens the significance score. The authors flag this honestly but it remains the paper's central evidentiary soft spot.
   - Proposed fix: A 3-seed RSNA sweep is the ideal fix but may not be feasible pre-camera-ready. At minimum, label Table 1 explicitly as single-seed in the caption and soften "2.6-point fusion gain indicates the two sources are complementary" (line 234) to acknowledge it is one seed. If even one extra RSNA seed is runnable within the rebuttal window, report it for at least Severe F1.
   - Severity: MAJOR

3. **Zero-shot baseline comparison is asserted but not quantified in a table.**
   - Location: line 265 ("Compared against a Naked BiomedCLIP baseline ... the hybrid wins on disc-related labels") and Table~\ref{tab:spider}.
   - Issue: The zero-shot section's interpretive punchline depends on beating off-the-shelf BiomedCLIP, but Table 2 reports only the hybrid's per-label F1/AUC — the "Naked BiomedCLIP" numbers are described in prose with no figures. A reader cannot verify the central claim of that subsection.
   - Proposed fix: Add a "Naked BiomedCLIP" column to Table~\ref{tab:spider} (it is 2 extra numbers per row, fits the existing float). No new experiment if those runs already exist; if not, at least give the mean for the disc group in text.
   - Severity: MAJOR

4. **Bold-best convention is inconsistent between the two main tables and may mislead.**
   - Location: Table~\ref{tab:main} caption ("Bold = best per row", line 214) vs. Table~\ref{tab:spider_transfer_agg} caption ("Bold = best per row when the gap ... exceeds 2σ", line 274).
   - Issue: Two different definitions of "bold" in two tables of the same paper. In Table 1, "Mean Accuracy 81.4% Base" is bolded as best even though the paper argues that high accuracy is the wrong objective here (line 236) — bolding it visually rewards the metric the authors then dismiss. This is a clarity/relevance issue for a skim-reading reviewer.
   - Proposed fix: Use one bolding rule across both tables (the 2σ-gated one is the more defensible), and either footnote or de-emphasize Mean Accuracy in Table 1 so the visual hierarchy matches the argument.
   - Severity: MINOR

5. **Several cited references appear to be unused or only weakly integrated, risking a padded-bibliography impression.**
   - Location: references.bib contains hallinan2021, mcsweeney2023, yang2026deciphermr, blankemeier2026merlin, koleilat2025biomedcoop, hong2025mscan, phaphuangwittayakul2026 — none of these keys appear cited in main.tex (grep of \cite calls shows the body cites nigru2024, jamaludin2017, windsor2022, woo2018cbam, he2016resnet, zhang2023biomedclip, lin2017focal, lin2024cbam, datascarcity2024, lu2024radclip, vandergraaf2024spider, pfirrmann2001, modic1988, aktan2025, kaggle2024, nshen2024).
   - Issue: With splncs04 + \bibliography these uncited entries simply will not appear, so it is harmless to the PDF, but the related-work section is then thinner than the bib suggests. Notably, 3D MRI/CT VLMs (yang2026deciphermr, blankemeier2026merlin) are mentioned conceptually ("native 3D MRI vision-language models were not public", line 90/317) but not cited at that point, and M-SCAN (hong2025mscan) is a directly relevant multi-view RSNA competitor left undiscussed.
   - Proposed fix: Within the 12-page limit, swap a sentence into Related Work to cite the 3D-VLM future-direction refs at line 90 and acknowledge M-SCAN as a multi-view alternative; or remove the unused keys to avoid the impression of citation inflation. No length increase needed (one-for-one sentence swap).
   - Severity: MINOR

6. **Architecture figure is small and the only qualitative result is a single Grad-CAM example.**
   - Location: Figure~\ref{fig:arch} at width=0.52\linewidth (line 162); Figure~\ref{fig:gradcam} at width=0.42\linewidth (line 308), n=1.
   - Issue: For an applied-AI readership the architecture diagram is the entry point and is rendered at roughly half text-width; a single Grad-CAM on one Severe case (line 304) is weak evidence for the "CBAM spatial-attention hypothesis." This is a presentation/persuasiveness concern, not a correctness one.
   - Proposed fix: Enlarge Figure 1 toward full text-width (per the project's own figure-readability rule), and if space allows after trimming, show 2-3 Grad-CAM panels (one per condition) rather than one. Page-neutral if traded against whitespace.
   - Severity: MINOR

## Questions to Authors
1. Are the "Naked BiomedCLIP" zero-shot numbers (line 265) available? If so, can they be added as a column to Table 2 for the camera-ready?
2. Is even a single additional RSNA seed feasible before camera-ready, to put an uncertainty bar on the Severe F1 headline?
3. The regularizer interpretation hinges on ruling out a trivial Baseline+BMC logit ensemble (line 313). Is that ablation runnable, or should the claim be downgraded to "consistent with" a regularization effect?
4. Why is Mean Accuracy bolded as "best" in Table 1 (line 220) when the paper argues it is the wrong objective? Is that intentional?

## Confidential Comments to Editor
My honest read: this clears the MIWAI bar. It is not a top-tier breakthrough — the hybrid does not beat a plain ResNet-34 on the cross-dataset aggregate, and the headline RSNA gains are single-seed — but for a mid-tier applied-AI venue the combination of a real novelty hook (first BiomedCLIP + text-prompt zero-shot label-space extension on lumbar RSNA→SPIDER), unusually honest reporting, sound imbalance-aware metrics, and clear reproducibility puts it comfortably above the reject line. The work is squarely on-topic for the LNAI readership.

My main editorial worry is framing rather than substance: the title and abstract promise an architecture-superiority story that the body then carefully walks back to a regularizer/parity story. A skim-only reviewer could read that as overselling and an in-depth reviewer as refreshing honesty — the gap between those two reactions is exactly what the authors should close in revision (Weakness 1). The single-seed RSNA issue (Weakness 2) is the one place a methodology-focused co-reviewer may push harder than I can from the EIC chair; if they demand multi-seed RSNA and the authors cannot deliver, this could slip toward Major Revision, but on the EIC criteria alone (fit, originality, significance, relevance, quality) Minor Revision is the right call. I would accept conditional on the abstract reframing, the Naked-BiomedCLIP column, and an explicit single-seed disclosure on Table 1.
