# LVTN Phase 3 — AI-Assisted Lumbar MRI Labeling Software (Design Spec)

**Date:** 2026-07-06
**Author:** Hà Trung Kiên (2470723) · Advisor: TS. Phan Trọng Nhân
**Status:** Design approved (brainstorm) — pending spec review → implementation plan
**Related:** `docs/LVTN_phase3/CAU_HOI_THAY.md` (open questions for advisor)

## 1. Goal

A local, web-based **AI-assisted labeling workstation** for lumbar-spine MRI. The doctor opens a patient's MRI, the app auto-produces (1) anatomical **segmentation** overlays and (2) **abnormality** highlights + a per-disc grading table, and the doctor can **correct** the labels/regions and **export** the result. Corrections are stored so a later phase can retrain the model (feedback loop — deferred, see §11).

This is Phase 3 of the thesis. It reuses Phase 1 (off-the-shelf segmentation) and Phase 2 (the trained CBAM + BiomedCLIP grading model); it does not train new models for the tool itself.

## 2. Scope

**In scope**
- Modality: **lumbar MRI only** (sagittal), reusing RSNA 2024 / SPIDER volumes as demo "patients".
- Two AI outputs shown **in parallel, independently** (not chained):
  - Anatomical segmentation (vertebrae, discs, ...) from **TotalSpineSeg**.
  - Per-disc grading + abnormality highlight from the **Phase 2 model** (CBAM-3D ResNet-34 + frozen BiomedCLIP).
- Doctor interaction: view, edit masks (brush/eraser), edit abnormality box (drag to resize/move), relabel/adjust severity, save, export.
- Patient/study management + persistence.

**Out of scope (YAGNI for a one-semester demo)**
- X-ray and CT modalities.
- Real doctor user study (evaluation is qualitative on sample cases + advisor/committee demo).
- Authentication / multi-user / access control (single-user local app).
- Cloud deployment (local only; demo can be pre-recorded).
- Chaining TotalSpineSeg output into the grading model.

## 3. Users & Deliverable

- **User:** a single operator (student/doctor) running locally.
- **Deliverable:** a working local demo (recordable as video) evaluated by the advisor and defense committee.
- **Timeline:** early July → end October 2026. ~1–2 months for the software (P0–P3), then P4/P5.

## 4. Architecture (3 tiers)

```
[ Frontend (local browser) ]      [ Backend API ]            [ Inference ]
 React + Cornerstone3D       <-->  FastAPI (Python)   <-->   local GPU
 - patient list / upload            - REST endpoints          OR remote Vast.ai
 - MRI viewer + overlays            - orchestrates 2 models   (config switch)
 - grade table + editing            - MySQL + filesystem
 - export
```

- **Frontend** and **backend** run locally. **Inference** runs either in-process (if the machine has a GPU) or on a **remote Vast.ai model server** reached over HTTP — chosen by a config flag. The inference layer is abstracted behind an interface so local vs remote is transparent to the backend.
- The two models run **in parallel** on the same input volume; their outputs are merged only for display, never fed into each other.

## 5. Components

### 5.1 Backend (FastAPI)
- `POST /patients` (register), `GET /patients`, `GET /patients/{id}` — patient/study CRUD (minimal).
- `POST /studies/{id}/upload` — upload an MRI volume (DICOM series or `.mha`/NIfTI), convert to a web-friendly form for the viewer, store on filesystem, record metadata in MySQL.
- `POST /studies/{id}/infer` — run TotalSpineSeg + Phase 2 grading (parallel), return structured JSON (the "results contract", §7).
- `PUT /studies/{id}/annotations` — save doctor corrections (edited masks + edited boxes + edited grades), versioned.
- `GET /studies/{id}/export?format=...` — export original + labeled image + masks + grades.
- **Inference abstraction:** `InferenceBackend` interface with `LocalBackend` and `RemoteVastBackend` implementations.

### 5.2 Frontend (React + Cornerstone3D)
- **Screen 1 — Patient list:** table (patient ID, date, #studies), upload button, open → viewer.
- **Screen 2 — Viewer/labeling workspace:**
  - Center: Cornerstone3D viewport — slice scroll, zoom/pan, window/level.
  - Overlays: segmentation masks (color per structure, toggle, opacity) + abnormality highlight (Grad-CAM heatmap for display + an editable box/ellipse ROI).
  - Right panel: per-disc grade table (level × condition × severity), clicking a row jumps to that disc; editing controls.
  - Actions: Run AI (spinner), Save, Export, Undo/Redo.
- **Editing:**
  - Anatomy mask → brush / eraser / scissors (Cornerstone3D segmentation tools).
  - Abnormality region → draggable box/ellipse handles (resize/move).
  - Grade/label → dropdown edit.

### 5.3 Data store
- **MySQL 8.x** (existing on dev machine): metadata + annotations + correction log. Tables: `patients`, `studies`, `annotations` (AI + corrected, versioned), `correction_log`. Driver: PyMySQL via SQLAlchemy.
- **Filesystem:** the heavy binaries — original MRI volumes, converted display volumes, mask files (NIfTI/PNG). MySQL stores only paths/pointers, never blobs.

## 6. Feature list (Core / Optional)

**Core:** patient list + upload; MRI viewer (scroll/zoom/window-level); seg mask overlay (toggle/opacity); abnormality heatmap + editable box; per-disc grade table with jump-to-disc; Run AI; mask edit (brush/eraser); box edit (drag); grade edit (dropdown); undo/redo; save; export (original + labeled + masks + grades).

**Optional:** search/filter/delete patients; slice thumbnails; before/after compare; Screen 3 (model/feedback management — tied to P4).

## 7. Results contract (model ↔ UI JSON)

A single JSON schema returned by `/infer` and consumed by the frontend, decoupling models from UI. Sketch:
```json
{
  "study_id": "...",
  "segmentation": { "mask_uri": "...", "labels": {"1": "L1", "2": "L1-L2 disc", ...} },
  "grading": [
    {"level": "L4-L5", "condition": "canal_stenosis", "severity": "Severe",
     "score": 0.87, "bbox": [x,y,w,h,slice], "heatmap_uri": "..."}
  ],
  "model_version": "phase2-v3"
}
```
Corrections are stored in the same shape with a `corrected_by`/`version` tag.

## 8. Data flow

1. Upload MRI → backend converts + stores → MySQL row + files.
2. Run AI → backend runs TotalSpineSeg ∥ Phase 2 (local or Vast) → returns results JSON.
3. Frontend overlays masks + heatmap + boxes, fills grade table.
4. Doctor edits mask/box/grades → Save → backend writes versioned annotation.
5. Export → original image + labeled image (overlay burned in) + mask files + grades CSV/JSON.

## 9. Tech stack

- Backend: Python 3.x, FastAPI, SQLAlchemy + PyMySQL, PyTorch (existing models), SimpleITK/nibabel/pydicom for volume I/O + conversion.
- Frontend: React + Cornerstone3D (viewer + segmentation tools).
- DB: MySQL 8.x. Store: local filesystem.
- Inference: local GPU or remote Vast.ai FastAPI model server (HTTP), selected by config.

## 9.1 Repository layout

The software lives in a **separate repository** (e.g. `spine-labeling-app`), independent of the `spinet-v2` research repo. Rationale: it is a distinct deliverable with a JS frontend that does not belong in the research codebase, and it must be demoable / deployable to Vast.ai on its own.

Model reuse across repos:
- **TotalSpineSeg** — installed as a pip package (no source coupling).
- **Phase 2 grading model** — the architecture module (`grading_hybrid.py`) and trained checkpoint (`.pth`) are **copied into the app repo** so it is self-contained and portable (preferred over pointing `PYTHONPATH` at `spinet-v2`, which breaks when the app moves to another machine / Vast).

`spinet-v2` remains the place where the model is trained; the app only consumes a copied checkpoint.

## 10. Phase decomposition

| Phase | Goal | Est. |
|---|---|---|
| **P0** | Scope lock + demo dataset selection (RSNA/SPIDER) + repo skeleton (backend/frontend) + results-contract JSON + MySQL schema | ~1 wk |
| **P1** | Backend: wrap both models behind `/infer` (local first), return valid results JSON; volume upload + conversion; Vast remote option | ~2 wk |
| **P2** | Frontend viewer: load MRI, scroll/zoom/window-level, read-only overlay of masks + heatmap + grade table (jump-to-disc) | ~2–3 wk |
| **P3** | Interaction: mask edit (brush/eraser), abnormality box edit (drag), grade edit, undo/redo, save (versioned), export | ~2–3 wk |
| **P4** | *(conditional on advisor)* Feedback loop: corrections → dataset → retrain/fine-tune Phase 2 → "model improved" cycle | ~2–3 wk |
| **P5** | Qualitative evaluation on sample cases + LVTN chapters + demo video + defense slides | ~2 wk |

## 11. Open questions (deferred to advisor)

Tracked in `docs/LVTN_phase3/CAU_HOI_THAY.md`:
- Feedback loop: which model relearns (recommend Phase 2 grading only), learning mechanism (batch retrain vs online), and whether it is required this semester or future work.
- Whether the parallel (non-chained) two-model architecture is acceptable.

## 12. Risks

- **Frontend effort (biggest):** React + Cornerstone3D learning curve could eat the budget. Mitigation: build on the library's built-in viewer/segmentation tools (do not hand-roll rendering); keep a **Streamlit fallback** if P2/P3 slips.
- **Volume format conversion:** RSNA is DICOM, SPIDER is `.mha`; the viewer needs a consistent web-friendly form. Handle in P1.
- **Inference weight/latency:** if local GPU is insufficient, use the Vast remote backend (already designed as a switch).
- **Feedback loop (P4)** is the riskiest/optional part; gate it on the advisor's decision so it does not block the core demo.
