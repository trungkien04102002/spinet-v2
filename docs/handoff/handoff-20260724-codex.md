# Handoff — 2026-07-24 (for Codex to continue LVTN phase 3)

Single entry point to resume the thesis (LVTN kì cuối). Read this first, then
`docs/LVTN_phase3/00_TRACKER.md` (the canonical follow file). Do NOT read the
raw Claude session transcript.

## Repos / branches
- `~/spinet-v2` — research + paper + tracker. Branch `biomedclip-integration` (HEAD `eef7467`).
- `~/spine-labeling-app` — the labeling software (FastAPI + React/Cornerstone3D, MySQL, TotalSpineSeg). Branch `main` (HEAD `cfdc7a2`), remote `trungkien04102002/spine-labeling-app`.
- Run app: `cd ~/spine-labeling-app && ./run.sh both` (BE :8000, FE :5173) · stop `./run.sh stop`.
- App needs `backend/.env` (gitignored) with `TOTALSPINESEG_BIN=/Users/kienha/totalspineseg/venv/bin/totalspineseg` and `SEG_DEVICE=cpu`.

## Done in the last session (2026-07-22 → 24)
1. **Fixed "Full grading" bug** (app `dea54b1`): TotalSpineSeg CLI not on PATH + `pixel_spacing` was a tuple (must be scalar mean of in-plane axes). Verified E2E.
2. **Unified the viewer grading UI** (app `cfdc7a2`): one `✨ Run AI` button + one editable 11-label "Abnormality grades" panel (was two overlapping panels). `/grade_full` now persists the segmentation mask + saves a v0 annotation, so the existing edit→`/annotations`→`correction_log` pipeline works on all 11 labels. Added "About this tool" guide (Oxford-demo style) + "Doctor feedback" section. Backend 63 tests pass; FE lint+build clean.
   - Known perf limit: "Run AI" is ~8–15 min on CPU (nnU-Net). Fine on GPU; async-job/progress UX is future polish.

## The big buckets (thesis scope — tracker §0)
Pillars: (1) labeling **software**, (2) **SOTA comparison** for grading, (3) **improve F1**, (4) doctor **feedback loop = FUTURE WORK**.

Mapping of the four items the user asked about:
- **Axial** → sub-item of pillar 3 (F1), the *last/hardest* lever (item #4 in tracker §3). Do T1-foraminal + gated fusion first.
- **Build software** → pillar 1, ~95% done (P0–P3). Remaining: polish + demo video.
- **Enable mask editing** → ALREADY built (viewer "Edit mask" + `saveMask`, mask round-trips). Only polish left.
- **Doctor feedback → model relearns** → pillar 4, FUTURE WORK (simple version); capture already built, no retrain this semester.
- ⚠️ Not in the user's list but part of scope: **SOTA comparison** (code ready, `experiments/sota_comparison/run_all.sh`, deferred run) and the already-staged F1 levers: **#0 threshold ✅**, **#1 T1-foraminal prep ✅**, **#2/#3 multi-view models ✅** — all just need GPU training (1 seed → 3 seeds).

## Item 1 — Axial: technical + papers (source: `docs/LVTN_phase3/MULTIVIEW_RESEARCH.md`)
- **Convention across top RSNA solutions:** central-canal ← Sag-T2, foraminal ← Sag-T1, subarticular/lateral-recess ← **Axial-T2**. Late fusion (ranks 1/9/10) beats early/transformer fusion (ranks 3/4).
- **Training crops:** use RSNA `train_label_coordinates.csv` (axial x,y,instance for subarticular) → per-disc axial crops, no projection. Verify column layout on the local file.
- **Inference alignment (the hard part):** DICOM geometry (ImagePositionPatient / ImageOrientationPatient / PixelSpacing → line-plane intersection with axial slices) or a trained axial-keypoint predictor. ~3–5 days, DICOM-geometry debugging. Pitfalls: series↔level messy, T12/L1 & L5/S1 coverage gaps.
- **Feed into a gated leader-supporter fusion** (already coded in `experiments/multiview/`): per-condition learned gate over sequence embeddings, init to clinical prior.
- **Papers:** M-SCAN cross-attention (arXiv:2503.01634, AUROC 0.971 axial+sagittal); GMU gated multimodal (Arevalo et al. 2017, arXiv:1702.01992); task-specific weighted fusion (arXiv:2307.00885, reusable contribution-viz figure); brendanartley RSNA 2nd place (axial branch built but **dropped — naive axial didn't help** → argues for selective/gated fusion); 3rd place (condition-specific concat + CenterNet keypoints for axial→level); PMC11913898 (sagittal-only AI ≈ radiologist-with-axial for canal).
- ⚠️ **Honest caveat:** no clean isolated ablation shows "axial → Severe-F1 gain." Mechanistic support is for **subarticular**. The stronger lever is **T1-for-foraminal**. Report as an ablation row (T2-only → +T1/flat → +gated → +axial), don't over-claim.

## Item 4 — Doctor feedback loop: technical + papers (source: `docs/LVTN_phase3/FEEDBACK_LOOP_RESEARCH.md`)
- **Sits at:** human-in-the-loop + active learning + label-correction/noise + continual learning.
- **Simplest credible loop (1×4090):** gather N corrections from `correction_log` → light noise filter (Cleanlab) → build fine-tune set (label = corrected grade) → **batch offline** fine-tune: freeze backbone + frozen BN stats, head-only, small LR, early stop, + **EWC** penalty vs original distribution → eval before/after on a **frozen held-out** (per-class F1/recall + weighted Cohen's kappa) **+ forgetting check** → swap only if both pass. Trigger = correction-count threshold (50–100), not perf-drop (no live GT).
- **Papers:** Budd/Robinson/Kainz 2021 HITL survey (arXiv:1910.02923); MONAI Label (Diaz-Pinto et al., arXiv:2203.12362 — closest prior-art tool); uncertainty/active-learning Gal et al. ICML 2017 (arXiv:1703.02910); **EWC** Kirkpatrick et al. PNAS 2017 (arXiv:1612.00796); **LwF** Li & Hoiem (TPAMI 2018); **Confident Learning / Cleanlab** Northcutt et al. JAIR 2021; cold-start AL can be worse than random — COLosSAL MICCAI 2023 (arXiv:2307.12004); VIOLA-AI deployment (arXiv:2505.09380).
- ⚠️ Several citations gathered from abstracts/snippets — re-verify author/year/venue before quoting. Pfirrmann inter-rater kappa ≈ 0.5–0.8 (pin a specific source). Ready-to-adapt Future-Work paragraph is in FEEDBACK_LOOP_RESEARCH.md §6.

## Immediate next actions (ordered)
1. Fix TotalSpineSeg PATH is done; app runs. No open bugs.
2. GPU batch (each 1 command, seed 42): train #1 T1-foraminal, train #2/#3 multi-view (concat + gated), SOTA `run_all.sh`.
3. Then: lock F1 numbers → write method/experiments chapters → polish app + demo video.
4. Feedback loop stays FUTURE WORK (write it up, don't build retrain).
