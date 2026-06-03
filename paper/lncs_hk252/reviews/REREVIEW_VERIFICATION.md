# Verification Re-Review (Editor) — MIWAI 2026 Final Revision

**Manuscript:** `paper/lncs_hk252/main.tex` (334 lines) + `references.bib`
**Role:** Editor traceability check after two prior 5-reviewer rounds.
**Scope:** Confirm each accumulated concern is resolved in the *current* text; catch any new error from the latest edits. READ-ONLY.

---

## Traceability Matrix

| # | Concern | Where checked | Resolved? | Evidence / quote |
|---|---------|---------------|-----------|------------------|
| C1 | Single-seed RSNA → 3-seed mean±std; abstract/intro/summary say "three seeds"; numbers consistent | Table `tab:main` (L214–234), abstract L56, intro L73, summary L323, caption L214 | **Y** | Table caption: "mean $\pm$ std over 3 seeds \{42, 123, 456\}". Every row carries $\pm$std (e.g. Hybrid Mean F1 `0.527 $\pm$ 0.027`). Abstract: "across three seeds the hybrid lifts Severe-class recall from 12.3\% to 48.6\%". Intro L73 + summary L323 both say "three-seed". Consistent. |
| C2 | Trivial-ensemble control reported in §4.5 (ensemble F1 0.45 / Severe 0.18 vs hybrid 0.53 / 0.36, two-seed) | §4.5 L314 | **Y** | "averaging the SpineNetV2 and BMC-only softmax outputs reaches only Mean F1 0.45 and Severe F1 0.18 (two-seed mean), well below the learned-fusion hybrid (0.53 and 0.36)". Numbers exact. Also honestly notes ensemble attains higher AUC/AUPRC. |
| C3 | Zero-shot prompt template disclosed + "a priori, no leakage" | §4.3 L242 | **Y** | "Each class uses a fixed prompt ``a magnetic resonance image of \emph{[label]}'' (e.g.\ ``lumbar disc herniation'' vs.\ ``no disc herniation''), written a priori without tuning on SPIDER, so no target-label leakage occurs." |
| C4 | Off-the-shelf BiomedCLIP: OTS F1 column + disc-morphology sub-row; prose honestly states OTS mean F1 0.394 > hybrid 0.362 | Table `tab:spider` (L248–266), prose L268 | **Y** | Table has dedicated `OTS F1` column (L250). Disc-morphology block (bulging/herniation/narrowing, L252–254) + `Mean (disc-morphology)` sub-row 0.433/0.587 (L262) + `Mean (all 8)` row with **OTS 0.394 bolded** vs Hybrid 0.362 (L263). Prose: "its mean F1 over all eight labels (0.394) edges the hybrid (0.362): RSNA supervision helps only where the label is semantically close." Honest. |
| Reframe | Title leads on "cross-dataset regularizer"; abstract scopes zero-shot to disc-morphology; no overclaim | Title L36, abstract L56 | **Y** | Title: "A Frozen BiomedCLIP Branch as a **Cross-Dataset Regularizer** for Imbalanced Lumbar Disc Grading and Zero-Shot Label-Space Extension". Abstract scopes ZS: "with gains concentrated on disc-morphology labels semantically close to the training schema" and explicitly "positions the frozen BiomedCLIP branch as a cross-dataset regularizer". No surpass-claim over SpineNetV2 (states "matches"). |
| CLIP cite | `radford2021clip` in .bib AND cited in §2 "CLIP-style" | bib L34–40, main L90 | **Y** | bib entry present (Radford et al., ICML 2021). §2 L90: "BiomedCLIP~\cite{zhang2023biomedclip} pretrains a CLIP-style~\cite{radford2021clip} image-text model". |
| Paired t-test | abstract/contribution/footnote/§4.4/§4.5 report "paired t-test p=0.011" for Hybrid>CBAM; NO stale '>2σ'/'descriptive effect size' for that claim | L56, L70, L295, L298, L314 | **Y** | 5 occurrences of `$p=0.011$` for the Hybrid-vs-CBAM claim. Footnote L295: "Hybrid vs.\ CBAM is significant (Mean F1 $\Delta=+0.034$, $p=0.011$, 95\% CI $[0.018,0.049]$)". `grep "descriptive effect"` → **0 hits**. `grep "naked"` → **0 hits**. The Hybrid>CBAM recovery claim never uses σ-language. |
| AUPRC gain | "6.7x" against "~0.05 prevalence" (not old 7.9x/0.042) | §4.2 L238, §4.1 L204 | **Y** | L238: "Severe AUPRC of 0.333 against the $\approx$0.05 prevalence baseline is a 6.7$\times$ gain over chance." Arithmetic: 0.333/0.05 = 6.66 ≈ 6.7. L204 also states prevalence "about 0.05 for Severe". `grep 7.9\|0.042` → **0 hits**. |
| nshen7 | No dangling "discussed in Section" promise | §2 L93 | **Y** | "The closest like-for-like reference point is the public 3-class baseline of nshen7~\cite{nshen2024}." No forward promise. `grep "discussed in Section"` → **0 hits**. |
| Severe headline | 12.3→48.6% recall, Severe F1 0.152→0.356, AUC 0.899 consistent; L93 says Mean F1 0.527 / Severe F1 0.356 (not old 0.528/0.343) | L56, L73, L93, L229–230, L238, L316, L323 | **Y** | All instances of 12.3/48.6, 0.152/0.356, 0.899 agree across abstract, intro, table, §4.2, discussion, summary. L93: "evaluated as a single model (Mean F1 0.527, Severe F1 0.356)". `grep 0.528\|0.343` → **0 hits**. |

---

## Independent New-Issue Scan

**Cross-references.** All 14 `\ref{}` targets (`sec2–5`, `sec:objective`, `sec:disc`, `fig:arch`, `fig:dist`, `fig:gradcam`, `tab:main`, `tab:spider`) map to a defined `\label{}`. No dangling/broken refs. Every float is referenced in prose (Fig arch L124, Fig dist L186, Fig gradcam L305, Tab main L210, Tab spider L242/L268, Tab spider_transfer_agg referenced via §4.4 prose). `tab:spider_transfer_agg` label exists (L278) and the table is discussed in §4.4 L298 / §4.5 L314 by content, though not via `\ref` — acceptable (it is the only table in its subsection and prose quotes its numbers directly).

**Citations.** All 17 `\cite` keys (aktan2025, datascarcity2024, he2016resnet, jamaludin2017, kaggle2024, lin2017focal, lin2024cbam, lu2024radclip, modic1988, nigru2024, nshen2024, pfirrmann2001, radford2021clip, vandergraaf2024spider, windsor2022, woo2018cbam, zhang2023biomedclip) resolve to bib entries. No undefined citations. (Bib contains a few unused entries — hallinan2021, mcsweeney2023, yang2026deciphermr, blankemeier2026merlin, koleilat2025biomedcoop, hong2025mscan, phaphuangwittayakul2026 — harmless; bibtex only emits cited keys.)

**Residual σ-language audit.** `grep sigma\|pooled\|2\\sigma`: remaining uses are all legitimate and refer to the *CBAM-hurts-transfer* claim (not the Hybrid>CBAM regularizer recovery, which now uses the t-test):
- L70 contribution: CBAM "lowers aggregate F1 by $0.027$ (beyond the pooled seed standard deviation)" — descriptive, appropriate for a one-directional degradation observation.
- L277 caption: "best per row when the gap to second place exceeds $2\sigma$ (the pooled standard deviation)" — this is the *bolding rule* for the table, legitimate.
- L298 / L314: CBAM-only vs SpineNetV2 "$|\Delta|/\sigma = 3.8$" — effect-size descriptor for the degradation; the *recovery* claim immediately following uses "paired $t$-test $p=0.011$". No stale/misapplied σ for the inferential Hybrid>CBAM claim.
- L127 / L175: σ appears as CBAM kernel notation and Gaussian-noise $\sigma=0.05$ — unrelated.

**Internal number contradictions.** None found. Spot-checks:
- Hybrid Severe F1 0.356 = Table L229, abstract, L73, L93, L238, L314 ✓
- Hybrid Mean F1 0.527 = Table L223, abstract, L93, L236 ✓
- SPIDER aggregate 0.653 / 0.646 / 0.619 consistent across abstract L56, intro L70, Table L285, §4.4 L298, §4.5 L314 ✓
- Δ=+0.034, p=0.011, 95% CI [0.018,0.049] consistent (L56, L70, L295, L298, L314) ✓
- Disc Herniation Yes recall: intro L72 "$0.36 \pm 0.08$ vs $0$"; §4.4 L301 "35.6\%" — consistent (0.356 ≈ 0.36). BMC-only 27.0% / hybrid 35.6% (L301) ✓
- ZS mean F1 0.362 (abstract L56, intro L72, §4.3 L242, Table L263) ✓
- Trainable params: "about 1.18\,M" (L158, L163) consistent.

**Double-blind leak.** Author identities fully commented out (L42–47); active block is `Anonymous Author(s)` / "Affiliation withheld" (L49–51). Acknowledgments with HCMUT/VNU-HCM commented out (L325–329). No leaks in body, captions, or footnotes. The commented camera-ready block contains real names/emails but is `%`-prefixed and will not render — standard practice, not a leak in the compiled PDF.

**Minor (non-blocking) observations.**
1. `tab:spider_transfer_agg` is never invoked via `\ref` (only discussed by content). Optional: add "Table~\ref{tab:spider_transfer_agg}" in §4.4 for CLAUDE.md rule 7 strictness. Not a blocker.
2. Several bib entries are uncited (listed above) — cosmetic, no compile/visual impact.

---

## Verification Verdict

**All 10 tracked concerns are FULLY CLOSED.** Every prior-round fix is present and internally consistent in the current text; the latest edits introduced **no new numerical contradiction, no broken cross-ref, no stale "descriptive effect size"/"naked"/">2σ" wording on the inferential Hybrid>CBAM claim, and no double-blind leak**. The 6.7×/0.05 AUPRC pair, the 0.527/0.356 headline, the OTS-honest SPIDER table, the disclosed zero-shot prompt, the trivial-ensemble control, and the paired t-test (p=0.011, CI provided) are all in place and mutually consistent.

**Residual blockers:** None.

**Final decision: ACCEPT (camera-ready ready).** Two optional cosmetic touch-ups remain (add a `\ref` to `tab:spider_transfer_agg`; prune unused bib entries) — neither gates acceptance.
