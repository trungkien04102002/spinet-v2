# SPIDER Zero-Shot Evaluation Setup

> Documentation of the SPIDER dataset preparation, validation strategy, and zero-shot evaluation methodology used in Phase 3 of the EMBC 2027 paper.

## 1. Datasets in this experiment

| Role | Dataset | Used for | Sample size |
|---|---|---|---|
| **Training source** | RSNA 2024 Lumbar Spine | Train Hybrid (CBAM + BiomedCLIP) backbone | 1942 IVDs (val), patient-level 80/20 split, `random_state=42` |
| **Zero-shot test** | SPIDER (Zenodo) | Evaluate label-space generalization without retraining | 1439 IVDs / 208 patients (T2-only) |

**Critical separation**: the Hybrid model never sees SPIDER images or labels during training. SPIDER is used only for inference at evaluation time.

## 2. SPIDER raw data overview

SPIDER is a public lumbar spine MRI dataset from Zenodo (`zenodo.org/doi/10.5281/zenodo.8009679`) with 257 patients × up to 3 series each (544 series total) collected from 4 hospitals. Each series has:
- One `.mha` MRI volume (T1, T2, or T2_SPACE)
- One `.mha` segmentation mask (vertebrae 1-N, IVDs 201-207, spinal canal 100)
- Per-IVD radiological gradings: Modic (4-class), UP/LOW endplate (binary), Spondylolisthesis (binary), Disc_herniation (binary), Disc_narrowing (binary), Disc_bulging (binary), Pfirrman_grade (5-class)

The dataset's official split is `training` (218 patients) / hidden test (39 patients). The hidden test labels are not public, so we use the entire public training portion as our zero-shot evaluation set (no SPIDER training of any kind happens in our pipeline).

## 3. Filtering decisions

After loading `spider/overview.csv` (447 rows after de-duplication), we apply three filters:

### 3.1 Drop non-lumbar series (3 rows)
`BodyPartExamined` is inconsistent across hospitals. We keep `LSPINE`, empty values, and `MRI LWK*` (Dutch *Lendenwervelkolom* = lumbar). We drop:
- 2 `TSPINE` (thoracic)
- 1 `MRI RUGPOLI ZEVE` (unclear, dropped to be safe)

### 3.2 Modality: T2 only (drop T1 + T2_SPACE)
- **T1 dropped** because the Hybrid backbone was trained on T2-weighted RSNA data; mixing modalities would introduce a domain shift orthogonal to the zero-shot label question we want to study.
- **T2_SPACE dropped** because it is a 3D isotropic acquisition (0.9 mm cubic voxels) with very different appearance from 2D T2 sagittal stacks; only 41 series have it, all with corresponding T2 already available.

After modality filter: **210 T2 series**.

### 3.3 Patients without T2 image
Some patients have only a T1 image (no T2). We skip these. This loses 49 patients.

After all filters: **210 T2 series × ~7 IVDs each = 1439 IVD samples in 208 unique patients**. Note the masks for some patients exist only as `<pid>_t1.mha` even when the T2 image is present (48 such cases). We verified that T1 and T2 from the same patient share identical coordinate systems (size, origin, direction, spacing all match), so a T1 mask correctly localizes IVDs in the T2 image.

## 4. Orientation handling (the most important fix)

SPIDER `.mha` volumes come from 4 hospitals with multiple scanner manufacturers. The voxel-to-physical mapping (the SimpleITK *direction matrix*) is inconsistent across patients. Two examples:

```
patient 1   shape after sitk.GetArrayFromImage = (578, 448, 50)   slice axis = 2
patient 107 shape after sitk.GetArrayFromImage = (17, 512, 512)   slice axis = 0
```

The grading model expects an input volume of shape `(9, 112, 224)` corresponding to `(sagittal_slices, superior_inferior, posterior_anterior)`. If the array axes are not in this canonical order, the model sees rotated/permuted anatomy and predictions are meaningless.

### 4.1 The fix
We standardize every volume (and its mask) via SimpleITK's `DICOMOrient` with the orientation code **`PIR`**:

```python
image = sitk.ReadImage(path)
image = sitk.DICOMOrient(image, 'PIR')
volume = sitk.GetArrayFromImage(image)
# Now numpy axes are guaranteed to be:
#   axis 0 = R direction = lateral (sagittal slice axis)
#   axis 1 = I direction = superior -> inferior (image vertical, head at index 0)
#   axis 2 = P direction = anterior -> posterior (image horizontal width)
```

We verified across all 210 T2 series that this produces axis 0 as the slice axis (8-50 slices, matching the largest physical spacing of 3-5 mm), with consistent in-plane orientation (head at top, anterior on the left side of the displayed image) — matching the standard medical sagittal view that the RSNA training data also follows.

## 5. IVD extraction with mask

For each (patient, IVD level), we use the SPIDER ground-truth mask to localize the disc, then crop a 3D volume of fixed physical extent around the disc center. We deliberately use ground-truth masks rather than running a detector because:

1. The zero-shot question is about label-space generalization, not about whether a detector trained on Genodisc transfers to SPIDER. Using GT masks isolates the variable of interest.
2. The original SpineNetV2 grading head and our Hybrid grading head both expect *pre-cropped* IVD volumes — they are not detection models. The full pipeline detects IVDs first, then runs the grading network on each crop.

### 5.1 Crop dimensions
The crop is centered on the disc bounding-box centroid:
- **Sagittal axis (axis 0)**: 9 consecutive slices centered on the disc's lateral midline. This matches the RSNA training data, where the IVD volume is `[center-4, ..., center, ..., center+4]` instances around the IVD's central slice.
- **Superior-inferior axis (axis 1)**: physical extent **60 mm** (converted to voxels per scanner spacing). Captures the disc plus 1-2 vertebrae of context above and below.
- **Posterior-anterior axis (axis 2)**: physical extent **120 mm** (2:1 aspect ratio with axis 1). Captures disc width plus surrounding tissue.

The 60 × 120 mm choice mirrors the RSNA preprocessing pipeline, which uses a 240 × 120 pixel crop at typical 0.5 mm/pixel spacing ≈ 120 mm × 60 mm. Matching the physical receptive field eliminates a distribution-shift confound when applying the RSNA-trained model to SPIDER.

### 5.2 Resampling and normalization
After cropping:
- Axis 0 is resampled to exactly 9 slices via linear interpolation (in case the bounding-box-centered selection clipped at the volume boundary).
- Axes 1 and 2 are bilinearly resized to (112, 224) per slice using OpenCV.
- Pixel intensities are clipped to the 1st-99th percentile per volume and rescaled to `[0, 1]`.

Final tensor shape: `(9, 112, 224)` `float32` in `[0, 1]`. Identical format to the RSNA dataloader.

## 6. Train / validation / test split

Three datasets, three different uses:

### 6.1 RSNA (training source)
80/20 patient-level split with `random_state=42`. Used for training all three backbones (Baseline, CBAM, Hybrid). Validation set: 1942 IVDs from held-out patients. Best epoch chosen on validation Avg Severe F1.

### 6.2 SPIDER zero-shot test (current phase)
**No split needed**: the entire 1439-sample dataset is treated as a single test set, because no SPIDER data is used for training. We do not select a model based on SPIDER metrics — the `best_model_hybrid_fixed_e7.pth` checkpoint was chosen purely on RSNA validation; SPIDER eval reports its zero-shot performance.

### 6.3 SPIDER transfer learning (Phase 4, future)
For Phase 4 (transfer learning ablation), we will use SPIDER's official `subset` field:
- `training` (360 series, 80% / 20% patient-level for train / val)
- `validation` (87 series, used as held-out test)

The `validation` subset is held-out for transfer-learning evaluation; it is not used in Phase 3 zero-shot.

## 7. Zero-shot evaluation methodology

For each of the 8 SPIDER diseases:

1. **Encode class prompts** via the frozen BiomedCLIP text encoder. Example prompts (template `med` = "a magnetic resonance image of {label}"):
   - `Disc_herniation`: `0 -> "a magnetic resonance image of no disc herniation"`, `1 -> "a magnetic resonance image of lumbar disc herniation"`
   - `Pfirrman_grade`: 5 prompts, one per grade (1-5).

2. **Encode test volumes** via the trained Hybrid image encoder. The Hybrid path:
   - 3D ResNet34 + CBAM backbone -> 1024-d feature
   - 9 sagittal slices each go through frozen BiomedCLIP image encoder (ViT-B/16) -> 9 × 512-d features, attention-pooled to one 512-d feature
   - Concatenate (1024 + 512) -> 2-layer MLP projection (`image_projection`) -> 512-d image embedding, L2 normalized.

3. **Predict** via cosine similarity. For each test sample, compute `sims[i, c] = image_emb[i] . text_emb[c]`. Predicted class = `argmax(sims, axis=1)`.

4. **Metrics** per disease: F1 macro, balanced accuracy, AUC (binary diseases only — computed on `sims[:, 1] - sims[:, 0]`).

The model is `eval()` mode throughout; no gradient is computed; no parameter is updated; no SPIDER-specific data ever touches the network weights.

## 8. What is "trained" vs what is "frozen pretrained"

The Hybrid checkpoint contains **only 7 trainable tensors** (verified by inspection):

| Component | Origin | Trained on SPIDER? |
|---|---|---|
| `logit_scale` | Trained from CLIP-style init `log(1/0.07)` | No — RSNA only |
| `slice_pool.scorer.weight/bias` | Trained: scores 9 sagittal slices for attention pooling | No — RSNA only |
| `image_projection.0.weight/bias` (1024 -> 768) | Trained MLP layer 1 | No — RSNA only |
| `image_projection.3.weight/bias` (768 -> 512) | Trained MLP layer 2 | No — RSNA only |
| CBAM backbone (3D ResNet34 + CBAM blocks) | Loaded from separate `best_model_attention_sqrt_cw_e20.pth` | No — RSNA only |
| BiomedCLIP image encoder (ViT-B/16) | Frozen pretrained from HuggingFace | No — pretrained on PubMed image-text pairs |
| BiomedCLIP text encoder (PubMedBERT + projection) | Frozen pretrained from HuggingFace | No — pretrained on PubMed image-text pairs |

This setup makes the experiment a clean test of label-space generalization: the model has learned to align image features to a *biomedical text space* using RSNA's three diseases, and at evaluation time we ask whether it can decode unseen disease names in the same text space.

## 9. Outputs and artifacts

After running `eval_zeroshot_spider.py`, the following files are produced under `experiments/paper_results/spider_zeroshot/`:

| File | Content |
|---|---|
| `results.csv` | One row per (disease, prompt_template, slice_strategy) with F1 macro, balanced accuracy, per-class P/R/F1, AUC |
| `best_metrics.json` | Full structured summary (timestamp, checkpoints used, summary by tier, all per-disease results) |
| `best_metrics.txt` | Human-readable summary table |

Per-sample predictions (saved separately for advisor review) are documented below in Section 10.

## 10. Reproducibility

All code paths and CLI commands:

```bash
# 1) Prepare cropped IVD volumes (CPU, ~2 min on M-series, ~5 min elsewhere)
python3 prepare_spider_zeroshot.py
# -> rsna_preprocessed_spider/volumes/<patient>_<ivd>.npy   (1439 files)
# -> rsna_preprocessed_spider/spider_zeroshot_test.csv      (test labels)

# 2) Run zero-shot eval (CPU ~24 min on M-series, GPU ~2-3 min)
python3 eval_zeroshot_spider.py \
    --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid_fixed_e7.pth \
    --cbam-checkpoint   checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth
# -> experiments/paper_results/spider_zeroshot/{results.csv, best_metrics.{json,txt}}
```

Code paths:
- `spider_dataloader.py` -> filtering, orientation, crop, resampling
- `prepare_spider_zeroshot.py` -> drives the dataloader, saves `.npy` cache, builds test CSV
- `eval_zeroshot_spider.py` -> loads Hybrid, encodes prompts, encodes volumes, computes metrics
- `spinenet/models/grading_hybrid.py` -> `SpineNetHybrid` class, `encode_image` / `encode_text`
- `spinenet/models/biomedclip_wrapper.py` -> frozen BiomedCLIP image+text encoders
