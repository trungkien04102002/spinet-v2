# Mendeley Dataset Analysis: Lumbar Spine MRI Family (k57fr854j2 + variants)

Compiled 2026-05-06.

## TL;DR

**Verdict: GO** — but NOT with the bare `k57fr854j2/2` dataset (raw MRI only,
no severity labels). The advisor's link points to a **family of 5 linked
Mendeley datasets** sharing the same 515-patient cohort. Two of them carry
labels directly compatible with our work:

- **x6ggzp2ycn/1** (Pfirrmann grade 1–5, 515 mid-sagittal T2 slices, 1,545
  disc assessments, 2024) — **directly comparable to our SPIDER Pfirrmann
  head**. Linked PLOS ONE 2024 paper (Natalia et al.) reports Mean Acc
  88.1% — clean comparator number.
- **9j9hmy8kbn/1** (Multi-Disorder masks: Normal IVD, LDB, SS, DDD, TDB,
  Spondylolisthesis, 1,038 sagittal images, 8,991 annotations, 2025) —
  comparable to SPIDER's Spondylolisthesis + DDD/herniation labels.

Effort to add as a 3rd evaluation dataset: ~6–8h (download + preprocess +
zero-shot eval). Rank-C confidence boost if successful: ~5–10%.

---

## Dataset family map

| ID | Title | Year | Size | Labels | Format | Use for our paper |
|---|---|---|---|---|---|---|
| `k57fr854j2/2` | Lumbar Spine MRI Dataset (raw) | 2019 | 515 pat. / 48,345 slices | **None** (raw MRI only) | DICOM, T1+T2, sag+ax, 320×320 | Source images only |
| `zbf6b4pttk/2` | Label Image Ground Truth (segmentation) | 2019 | axial subset | 4 anatomical regions: IVD, PE (Posterior Element), TS (Thecal Sac), AAP | PNG masks | Different task (segmentation) — skip |
| `s6bgczr8s2/2` | **Radiologist Notes (free-text)** | 2019 | 515 pat. | **Free-text** narrative reports | Text format | NLP-only, ~not-feasible without report-extraction LLM |
| `9j9hmy8kbn/1` | Multi-Disorder Annotations | 2025 | 1,038 sag. images / 8,991 annotations | Per-IVD presence/absence: Normal IVD (4,131), LDB (2,033), SS (1,921), DDD (633), TDB (151), Spondylolisthesis (122) | PNG masks 384×384 | **Compatible with SPIDER Spondy + DDD subsets** |
| `x6ggzp2ycn/1` ⭐ | **Pfirrmann grades + IVD height (PLOS ONE 2024)** | 2024 | 515 mid-sag T2 slices / 1,545 disc assessments | **Pfirrmann grade 1–5** + IVD mid-height (mm) | DICOM + PNG 384×384 + .mat | **Directly comparable to SPIDER Pfirrmann head** |

---

## Bibliographic info

### Raw dataset
- **Authors**: Sud Sudirman, Ala Al Kafri, Friska Natalia, Hira Meidia, Nunik Afriliana, Wasfi Al-Rashdan, Mohammad Bashtawi, Mohammed Al-Jumaily
- **Institutions**: Universitas Multimedia Nusantara, Liverpool John Moores University, Irbid Specialty Hospital (Jordan)
- **Period**: Sep 2015 – July 2016
- **DOI**: 10.17632/k57fr854j2.2
- **License**: CC BY 4.0
- **Year**: 2019

### Pfirrmann subset (x6ggzp2ycn/1)
- **Authors**: Sud Sudirman, Friska Natalia
- **DOI**: 10.17632/x6ggzp2ycn.1
- **License**: CC BY 4.0
- **Year**: 2024

### Linked publication (Pfirrmann eval ground truth comparator)
- **Natalia, Sudirman, Ruslim, Al-Kafri 2024**, "Lumbar spine MRI annotation with intervertebral disc height and Pfirrmann grade predictions", **PLOS ONE**
- **DOI**: 10.1371/journal.pone.0302067
- **Method**: DeepLabv3 + ResNet-50 (segmentation) → IVD height (geometry) → self-similar color correlogram features → 6 ML classifiers (best one for Pfirrmann)
- **Reported**: Pfirrmann **Mean Acc 88.1%** on 1,545 discs (1-5 grade scale)
- **Per-grade Acc**: G1 68.3%, G2 92.6%, G3 79.0%, G4 82.7%, G5 76.2%
- **Imbalance**: G1 3.9% / G2 64.9% / G3 10.8% / G4 19.0% / G5 1.4%
- **Split**: 80:20 train:test; training further 60:20 train:val
- **No F1/AUC reported**

---

## MRI specifications

- **Modality**: T2-weighted mid-sagittal (for Pfirrmann subset); raw dataset has T1 + T2 sag+ax
- **Scanner**: 1.5-Tesla Siemens
- **Plane**: sagittal (Pfirrmann subset); raw has both sag + ax
- **Spine region**: lumbar (last 3 vertebrae + last 3 IVDs L3-L4, L4-L5, L5-S1)
- **Resolution**: 320×320 (raw); 384×384 (Pfirrmann processed PNG)
- **Slice thickness**: 4 mm axial (4.4 mm centre-to-centre)
- **Pixel spacing**: 0.6875 mm
- **Pixel depth**: 12-bit
- **No segmentation labels for vertebrae/IVDs in raw dataset** (separate dataset zbf6b4pttk/2)

---

## Compatibility with our task

| Our task | Mendeley label | Mappable? | Effort |
|---|---|---|---|
| RSNA: spinal_canal_stenosis 3-class | x6ggzp2ycn/1: Pfirrmann 1-5 | NO (different condition) | — |
| RSNA: left/right_foraminal 3-class | x6ggzp2ycn/1: Pfirrmann 1-5 | NO | — |
| SPIDER: **Pfirrmann 1-5** | x6ggzp2ycn/1: Pfirrmann 1-5 | **YES (direct)** | Low |
| SPIDER: Spondylolisthesis (binary) | 9j9hmy8kbn/1: Spondylolisthesis (binary) | **YES** | Medium (need to align 122 cases) |
| SPIDER: Disc Herniation (binary) | 9j9hmy8kbn/1: LDB (binary) | **YES (LDB ≈ herniation/bulge)** | Medium |

**Key insight**: For SPIDER zero-shot evaluation, Mendeley provides a **third
external dataset** (after RSNA train, SPIDER test) that's directly
compatible with our Pfirrmann head. This is exactly what advisors mean by
"cross-cohort generalization".

---

## Decision recommendation: GO (selective)

### Recommended evaluation plan

1. **Download x6ggzp2ycn/1** (Pfirrmann subset only) — DICOM + PNG ~few hundred MB
2. **Preprocess** sagittal T2 mid-slice to our 9-slice format (use middle slice + 4 above / 4 below from raw DICOM if available; or upsample from single mid-slice with axial padding)
3. **Run zero-shot SPIDER Pfirrmann head** on Mendeley → report Pfirrmann F1, Acc, AUC, AUPRC
4. **(Optional) Retrain SPIDER → fine-tune on Mendeley Pfirrmann subset** for ~2-3h on Vast.ai
5. **Compare numbers vs Natalia 2024 PLOS ONE 88.1% Acc**

### Effort breakdown
- Download + extract: 30 min
- Preprocessing pipeline: 2-3h (need to handle 1.5T DICOM, generate 9-slice volumes)
- Zero-shot eval: 1h (just inference)
- Retrain (optional): 2-3h Vast
- Total: **6-9h**

### Rank-C boost estimate: +5–10%
- Adds a third independent dataset
- Direct numerical comparator (Natalia 88.1%) for our SPIDER Pfirrmann head
- Demonstrates cross-cohort generalization
- If we beat 88.1%: strong claim
- If we match (~85-90%): defensible
- If we underperform: still useful as honest external validation

### Caveats / risks
1. **Single-slice mid-sagittal vs our 9-slice**: x6ggzp2ycn/1 ships only the mid-sagittal slice. Our model expects (9, 112, 224). Need either (a) get 9-slice volume from raw DICOM in k57fr854j2/2 (extra ~30 min coordination) or (b) replicate the single slice 9× (degrades information).
2. **Pfirrmann taxonomy match**: Standard Pfirrmann 1–5 (Pfirrmann et al. 2001). Should match SPIDER Pfirrmann.
3. **Class imbalance**: G2 dominates (64.9%) — ours and theirs both. Means F1 macro will be sensitive.
4. **Pfirrmann is the SPIDER side, not the RSNA side** — Mendeley does NOT improve our RSNA numbers. It's a SPIDER cross-validation.

---

## Comparable usage in literature

The PLOS ONE 2024 Natalia et al. paper IS the published baseline using this
Pfirrmann subset. Other recent papers using `k57fr854j2/2` raw data:

1. Various transfer-learning CNN papers for binary spinal stenosis detection (no consistent labeling — used the bare images with manually relabelled subsets).
2. YOLOv5/v6/v7 disc herniation detection studies.
3. SegNet 4-region segmentation studies.

None of these use the Pfirrmann subset directly except Natalia 2024.

---

## Sources

- [Mendeley k57fr854j2/2 (raw)](https://data.mendeley.com/datasets/k57fr854j2/2)
- [Mendeley x6ggzp2ycn/1 (Pfirrmann)](https://data.mendeley.com/datasets/x6ggzp2ycn/1)
- [Mendeley 9j9hmy8kbn/1 (Multi-Disorder)](https://data.mendeley.com/datasets/9j9hmy8kbn/1)
- [Mendeley s6bgczr8s2/2 (Radiologist Notes)](https://data.mendeley.com/datasets/s6bgczr8s2/2)
- [Mendeley zbf6b4pttk/2 (Segmentation labels)](https://data.mendeley.com/datasets/zbf6b4pttk/2)
- [Natalia 2024 PLOS ONE (Pfirrmann annotation paper)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0302067) — DOI 10.1371/journal.pone.0302067

---

## Final summary

The Mendeley link the advisor sent (`k57fr854j2/2`) is a 2019 raw lumbar
spine MRI cohort (515 patients, 48,345 slices, T1+T2 sag+ax) with **no
severity labels in the raw release**. However, the same authors (Sudirman,
Natalia, Al-Kafri) released **two follow-up labelled subsets** that are
directly compatible with our SPIDER pipeline: `x6ggzp2ycn/1` (Pfirrmann
grade 1–5 on 1,545 discs) and `9j9hmy8kbn/1` (Multi-Disorder masks
covering Spondylolisthesis, LDB, DDD, SS). The Pfirrmann subset has an
existing PLOS ONE 2024 baseline (Natalia et al., Mean Acc 88.1%) we can
directly compare against. Recommended action: **GO with the Pfirrmann
subset as a third evaluation dataset for our SPIDER Pfirrmann head**;
expected effort 6–9h; expected Rank-C confidence boost +5–10%. The bare
`k57fr854j2/2` is **not directly usable** because it contains no severity
labels; the radiologist notes (`s6bgczr8s2/2`) are free-text and would
require an additional NLP extraction step that's out of scope for this
paper.
