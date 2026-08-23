# LVTN Phase 3 — Labeling Software Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

> **SCOPE UPDATE 2026-07-06:** MIWAI paper submitted. Final-semester scope = **software (P1–P3, priority 1) + external SOTA comparison for grading (new work item) + optional result improvement (priority 2)**. The **doctor-feedback loop (P4) is moved to FUTURE WORK** (too large for one semester; advisor ranked it priority 2). Official timeline: hand-off 15–17 Jul, **15 weeks 20 Jul → 31 Oct 2026**, defense 02–06 Nov 2026. Rough split: software P1–P3 ~6–8 wks, SOTA ~2 wks (parallel on Vast), report + slides + demo video ~3–4 wks (interleaved), ~1 wk buffer. English title: "AI-Assisted Medical Image Annotation for Low Back Pain".

**Goal:** Build a local web app where a doctor opens a lumbar MRI, sees AI segmentation + abnormality grading overlays, corrects them, and exports the result.

**Architecture:** Separate repo `spine-labeling-app`. FastAPI backend orchestrates two independent models (TotalSpineSeg for anatomy masks, the Phase-2 CBAM+BiomedCLIP model for per-disc grading + Grad-CAM), returning one JSON "results contract". React + Cornerstone3D frontend renders/edits overlays. MySQL stores metadata + annotations; filesystem stores volumes + masks. Inference runs locally or on a remote Vast.ai server via a config switch.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy + PyMySQL, PyTorch, SimpleITK/nibabel/pydicom; React (Vite) + Cornerstone3D; MySQL 8.x.

## Global Constraints
- Modality: lumbar sagittal MRI only. Demo "patients" = RSNA 2024 / SPIDER volumes (no real PHI).
- Two models run in parallel, never chained (seg output is NOT grading input).
- No auth, no cloud deploy (single-user local; demo can be pre-recorded).
- MySQL holds metadata/annotations only; big binaries (volumes, masks) live on filesystem (DB stores paths).
- Phase-2 model is reused by copying its architecture module + `.pth` checkpoint into the app (self-contained); TotalSpineSeg via pip.
- Scope of this plan = P0–P3 (a working labeling tool). P4 (feedback loop) is gated on the advisor; P5 (eval + thesis writing + demo) is a roadmap section.

## Repo layout (target)
```
spine-labeling-app/
  backend/
    app/ (main.py, config.py, db.py, models_db.py, schemas.py, routers/, inference/)
    models/ (grading_hybrid.py copied, weights/phase2.pth)
    tests/
  frontend/  (Vite React: src/pages, src/components, src/lib/api.ts)
  data/      (imported demo studies; gitignored)
  serve_models.py  (optional Vast inference server)
  README.md
```

---

## PHASE P0 — Foundation & skeleton (~1 week)

### Task 0.1: Create app repo + structure
**Files:** new repo `spine-labeling-app/` with the layout above; `README.md`; `.gitignore` (ignore `data/`, `*.pth`, `__pycache__`, `node_modules`).
- [ ] Create dir + `git init`; add README stating goal + run instructions placeholder.
- [ ] Add `.gitignore`.
- [ ] Commit `chore: scaffold spine-labeling-app repo`.

### Task 0.2: Backend skeleton + config + health endpoint
**Files:** `backend/app/main.py`, `backend/app/config.py`, `backend/requirements.txt`, `backend/tests/test_health.py`
**Produces:** `GET /health -> {"status":"ok"}`; `Settings` with `inference_mode` (`local`|`vast`), `vast_url`, `mysql_dsn`.
- [ ] Write `test_health.py`: `client.get("/health")` asserts 200 + `{"status":"ok"}`.
- [ ] Run `pytest backend/tests/test_health.py` → FAIL (no app).
- [ ] Implement `config.py` (pydantic-settings reading env) + `main.py` (FastAPI app + `/health`).
- [ ] Run pytest → PASS.
- [ ] Commit `feat: FastAPI skeleton + health + settings`.

### Task 0.3: MySQL schema + DB session
**Files:** `backend/app/db.py` (engine/session from `mysql_dsn`), `backend/app/models_db.py` (SQLAlchemy models), `backend/tests/test_db.py`
**Produces:** tables `patients(id, name, created_at)`, `studies(id, patient_id, modality, volume_path, display_path, created_at)`, `annotations(id, study_id, version, kind, payload_json, mask_path, created_at)`, `correction_log(id, study_id, field, old, new, at)`.
- [ ] Write `test_db.py`: create a patient + study, query back, assert fields (use a test MySQL schema or SQLite in-memory for the unit test).
- [ ] Run → FAIL.
- [ ] Implement models + `Base.metadata.create_all`.
- [ ] Run → PASS.
- [ ] Commit `feat: MySQL models + session`.

### Task 0.4: Results-contract schemas (Pydantic)
**Files:** `backend/app/schemas.py`, `backend/tests/test_schemas.py`
**Produces:** `SegmentationResult{mask_uri:str, labels:dict[int,str]}`, `GradingItem{level:str, condition:str, severity:str, score:float, bbox:list[float]|None, heatmap_uri:str|None}`, `InferResult{study_id, segmentation, grading:list[GradingItem], model_version}`.
- [ ] Write `test_schemas.py`: parse a sample dict → objects; assert round-trip `.model_dump()` matches.
- [ ] Run → FAIL.
- [ ] Implement the Pydantic models.
- [ ] Run → PASS.
- [ ] Commit `feat: results-contract schemas`.

### Task 0.5: Demo dataset import script
**Files:** `backend/scripts/import_demo.py`
**Consumes:** DB models (0.3). **Produces:** CLI that copies N RSNA/SPIDER volumes into `data/<study_id>/` and inserts patient+study rows.
- [ ] Implement script: args `--src <dir> --n 5`; copy volume file, insert rows.
- [ ] Run on a few sample volumes; verify `GET`-able later; check DB has rows (manual `SELECT`).
- [ ] Commit `feat: demo dataset import`.

### Task 0.6: Frontend skeleton (Vite React) + API client + routing
**Files:** `frontend/` (Vite React+TS), `frontend/src/lib/api.ts`, `frontend/src/pages/{PatientList,Viewer}.tsx`, router.
**Produces:** two routes (`/` list, `/viewer/:studyId`); `api.getHealth()` shows backend status on the list page.
- [ ] `npm create vite@latest` (react-ts); add react-router; `api.ts` with base URL from env.
- [ ] Two stub pages + route; list page calls `/health` and renders status.
- [ ] Run `npm run dev`, confirm both routes load + health shows "ok".
- [ ] Commit `feat: frontend skeleton + routing + api client`.

**P0 deliverable:** backend runs (`/health`, DB, schemas), demo data importable, frontend shell with 2 routes talking to backend.

---

## PHASE P1 — Backend inference behind the API (~2 weeks)

### Task 1.1: Phase-2 grading wrapper
**Files:** copy `spinet-v2/spinenet/models/grading_hybrid.py` → `backend/models/grading_hybrid.py`; copy checkpoint → `backend/models/weights/phase2.pth`; `backend/app/inference/grading.py`; `backend/tests/test_grading.py`
**Produces:** `run_grading(volume_path) -> list[GradingItem]` (loads model once, preprocesses to `(1,9,112,224)`, returns per-condition severity + score + Grad-CAM heatmap path).
- [ ] Write `test_grading.py`: run on one sample volume; assert returns ≥1 `GradingItem` with a valid severity in the label set.
- [ ] Run → FAIL.
- [ ] Implement: load `GradingModelWithCBAM`/hybrid, reuse preprocessing from spinet-v2 dataloader logic (percentile clip + rescale + crop), produce grades; Grad-CAM via `viz/grad_cam` logic (target layer3/cbam3) → save heatmap PNG.
- [ ] Run → PASS (may need GPU; if none, run on CPU with a tiny sample).
- [ ] Commit `feat: phase-2 grading inference wrapper`.

### Task 1.2: TotalSpineSeg wrapper
**Files:** `backend/app/inference/segmentation.py`, `backend/tests/test_seg.py`, add `totalspineseg` to requirements.
**Produces:** `run_segmentation(volume_path) -> mask_path` (NIfTI labelmap) + `labels: dict[int,str]`.
- [ ] Write `test_seg.py`: run on one sample; assert a mask file is produced with >1 unique label.
- [ ] Run → FAIL.
- [ ] Implement: call TotalSpineSeg on the volume, save labelmap, map its label ids → names (vertebrae/discs).
- [ ] Run → PASS.
- [ ] Commit `feat: TotalSpineSeg wrapper`.

### Task 1.3: Volume I/O + web conversion
**Files:** `backend/app/inference/volume_io.py`, `backend/tests/test_volume_io.py`
**Produces:** `load_volume(path)` (DICOM series or `.mha`/NIfTI → numpy + spacing) and `to_web_form(path) -> display_path` (a Cornerstone3D-friendly form: NIfTI or multiframe; pick one and standardize).
- [ ] Write test: load a RSNA DICOM dir + a SPIDER `.mha`; assert shape/spacing non-empty; `to_web_form` produces a readable file.
- [ ] Run → FAIL.
- [ ] Implement with SimpleITK/pydicom/nibabel.
- [ ] Run → PASS.
- [ ] Commit `feat: volume load + web conversion`.

### Task 1.4: Upload endpoint
**Files:** `backend/app/routers/studies.py`, `backend/tests/test_upload.py`
**Consumes:** 0.3 DB, 1.3 volume_io. **Produces:** `POST /studies/{id}/upload` (multipart) → stores volume, converts, updates `studies.volume_path`+`display_path`.
- [ ] Write test: upload a small volume → 200 + DB paths set + files exist.
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Commit `feat: study upload endpoint`.

### Task 1.5: InferenceBackend interface + LocalBackend + /infer
**Files:** `backend/app/inference/base.py` (`InferenceBackend` ABC with `infer(volume_path) -> InferResult`), `backend/app/inference/local.py` (`LocalBackend` runs 1.1 ∥ 1.2), `backend/app/routers/infer.py`, `backend/tests/test_infer.py`
**Produces:** `POST /studies/{id}/infer -> InferResult` (runs seg + grading in parallel — `asyncio.gather`/threadpool — persists a v0 `annotations` row).
- [ ] Write test: infer on an imported study → valid `InferResult` (segmentation.mask_uri set, ≥1 grading item) + a v0 annotation row.
- [ ] Run → FAIL. Implement (parallel run, assemble `InferResult`, save). Run → PASS.
- [ ] Commit `feat: local inference backend + /infer`.

### Task 1.6: RemoteVastBackend + model server + config switch
**Files:** `backend/app/inference/remote.py` (`RemoteVastBackend` posts volume to `vast_url`), `serve_models.py` (standalone FastAPI exposing `/seg` + `/grade` on the Vast box), `backend/tests/test_remote.py`
**Produces:** same `InferResult` via HTTP; `Settings.inference_mode` selects Local vs Remote.
- [ ] Write test: mock `vast_url` with a stub server; assert `RemoteVastBackend.infer` returns a valid `InferResult`.
- [ ] Run → FAIL. Implement remote client + `serve_models.py`; wire the factory that picks backend by config. Run → PASS.
- [ ] Commit `feat: remote Vast inference backend + model server`.

**P1 deliverable:** `POST /infer` returns a valid results-contract for an imported study, local or via Vast.

---

## PHASE P2 — Frontend viewer, read-only (~2–3 weeks)

> Frontend tasks verify **visually** (documented manual check) plus a lightweight component/render test where practical (Vitest). Each task ends with a commit.

### Task 2.1: Patient list + upload UI
**Files:** `frontend/src/pages/PatientList.tsx`, `frontend/src/lib/api.ts` (add `getPatients`, `uploadStudy`).
- [ ] Fetch + render patients table (id, date, #studies); upload control posts to `/studies/{id}/upload`; row click → `/viewer/:id`.
- [ ] Verify: import demo data, list shows rows, click opens viewer route.
- [ ] Commit `feat: patient list + upload`.

### Task 2.2: Cornerstone3D viewport
**Files:** `frontend/src/components/Viewer/CornerstoneViewport.tsx`, cornerstone init in `frontend/src/lib/cornerstone.ts`.
- [ ] Init Cornerstone3D; load the study `display_path`; render a stack/volume viewport; enable slice scroll, zoom/pan, window/level tools.
- [ ] Verify: MRI renders, scroll through slices, W/L works.
- [ ] Commit `feat: cornerstone3d viewport`.

### Task 2.3: Segmentation overlay
**Files:** `frontend/src/components/Viewer/SegOverlay.tsx`.
**Consumes:** `InferResult.segmentation` (mask_uri, labels).
- [ ] Load mask as a Cornerstone3D labelmap segmentation; color per label; toggle visibility + opacity slider.
- [ ] Verify: masks overlay aligned on the MRI; toggle/opacity work.
- [ ] Commit `feat: segmentation overlay`.

### Task 2.4: Abnormality overlay (heatmap + box)
**Files:** `frontend/src/components/Viewer/AbnormalityOverlay.tsx`.
**Consumes:** `InferResult.grading[].{heatmap_uri,bbox,level,severity}`.
- [ ] Render Grad-CAM heatmap (image overlay, opacity) + a RectangleROI/Ellipse annotation per abnormal disc at `bbox`.
- [ ] Verify: heatmap + boxes show on correct slices/discs.
- [ ] Commit `feat: abnormality heatmap + box overlay`.

### Task 2.5: Grade table panel + jump-to-disc
**Files:** `frontend/src/components/Viewer/GradePanel.tsx`.
- [ ] Render per-disc table (level × condition × severity + score); click a row → set viewport slice to that disc's `bbox` slice + highlight its box.
- [ ] Verify: click row jumps to disc.
- [ ] Commit `feat: grade table + jump-to-disc`.

### Task 2.6: "Run AI" wiring
**Files:** `frontend/src/pages/Viewer.tsx` (compose 2.2–2.5), `api.infer(studyId)`.
- [ ] "Run AI" button → `POST /infer` (spinner) → populate overlays + table.
- [ ] Verify: end-to-end — open study, Run AI, overlays + table appear.
- [ ] Commit `feat: run-AI wiring (read-only viewer complete)`.

**P2 deliverable:** open a study, Run AI, see anatomy masks + abnormality heatmap/boxes + grade table (read-only).

---

## PHASE P3 — Editing + save + export (~2–3 weeks)

### Task 3.1: Mask editing tools
**Files:** `frontend/src/components/Viewer/SegEditTools.tsx`.
- [ ] Enable Cornerstone3D BrushTool/EraserTool/Scissors on the labelmap; toolbar to pick tool + brush size + active label.
- [ ] Verify: paint/erase edits the mask (too-wide region can be trimmed).
- [ ] Commit `feat: mask editing tools`.

### Task 3.2: Abnormality box editing
**Files:** extend `AbnormalityOverlay.tsx`.
- [ ] Make ROI boxes interactive: drag corner handles to resize, drag body to move; add/delete a box.
- [ ] Verify: resize/move a box (shrink a too-wide region).
- [ ] Commit `feat: abnormality box editing`.

### Task 3.3: Grade/label editing
**Files:** extend `GradePanel.tsx`.
- [ ] Severity dropdown per row; edit label name; changes held in local state.
- [ ] Verify: change a severity → reflected in state.
- [ ] Commit `feat: grade/label editing`.

### Task 3.4: Undo/redo
**Files:** `frontend/src/lib/history.ts`, wire into edits.
- [ ] Command/history stack for mask/box/grade edits; Undo/Redo buttons + Ctrl+Z/Y.
- [ ] Verify: edit → undo restores → redo re-applies.
- [ ] Commit `feat: undo/redo`.

### Task 3.5: Save corrections (versioned)
**Files:** backend `PUT /studies/{id}/annotations` in `routers/studies.py` + `backend/tests/test_annotations.py`; frontend `api.saveAnnotations`.
**Produces:** persists a new `annotations` row (version+1, kind='corrected', payload=edited grading, mask_path=edited mask) + `correction_log` diffs.
- [ ] Write backend test: PUT edited payload → new version row; GET latest returns it.
- [ ] Run → FAIL. Implement backend. Run → PASS.
- [ ] Frontend "Save" posts edited masks + boxes + grades; verify a reload shows saved edits.
- [ ] Commit `feat: save versioned corrections`.

### Task 3.6: Export
**Files:** backend `GET /studies/{id}/export` in `routers/studies.py` + `backend/tests/test_export.py`; frontend export button.
**Produces:** a zip/files: original volume, labeled image (overlay burned onto slices as PNG), mask file (NIfTI), grades (CSV + JSON).
- [ ] Write backend test: export → returns the expected artifacts (assert files present + CSV has rows).
- [ ] Run → FAIL. Implement (render overlays server-side or accept a client-rendered PNG). Run → PASS.
- [ ] Frontend button downloads the export.
- [ ] Commit `feat: export original + labeled + masks + grades`.

**P3 deliverable (= core product done):** a doctor can open a study, run AI, correct masks/boxes/grades, save, and export. This is the demoable thesis software.

---

## ROADMAP (not detailed here — separate plans)

### P4 — Doctor-feedback learning loop (GATED on advisor; ~2–3 weeks)
Only start after the advisor answers `CAU_HOI_THAY.md`. Likely shape (to be planned then): collect saved corrections into a training set → fine-tune / retrain the Phase-2 grading model (batch, offline) → version + swap the model → show "before/after" improvement in the app. Gets its own spec + plan once scope is confirmed.

### P5 — Evaluation + thesis + demo (~2 weeks)
Qualitative evaluation on sample cases (segmentation overlay sanity + grading agreement on a few volumes), write the LVTN chapters (software chapter + updated conclusion), record the demo video, and update the defense slides. Thesis-writing, not a code plan.

---

## Self-review notes
- **Spec coverage:** patient mgmt (0.5,2.1), viewer (2.2), seg overlay+edit (2.3,3.1), abnormality heatmap+box+edit (2.4,3.2), grade table+edit (2.5,3.3), run-AI (2.6), save/versioning (3.5), export (3.6), MySQL+filesystem (0.3), results-contract (0.4), local+Vast inference (1.5,1.6), separate repo + copied Phase-2 model (0.1,1.1). Feedback-loop + eval = roadmap (P4/P5) as designed.
- **Parallelism:** P0 tasks are mostly sequential (0.2→0.3→0.4, 0.6 independent). In P1, 1.1/1.2/1.3 are independent (can parallelize) before 1.5 joins them. P2 tasks 2.3/2.4/2.5 depend on 2.2 but are independent of each other.
- **Verification:** backend tasks are TDD (pytest); frontend tasks use documented manual/visual checks (+ optional Vitest render tests) since they wire a GUI library — this is the honest granularity for greenfield UI.
