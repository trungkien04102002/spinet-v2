# Báo cáo Kiến trúc Hệ thống SpineNetV2

**Học viên:** Hà Trung Kiên (2470723)
**Đề tài:** Ứng dụng AI trong gán nhãn ảnh y tế cho bệnh đau thắt lưng
**Ngày cập nhật:** 2026-04-20

> Tài liệu này mô tả kiến trúc hệ thống **training** và **inference** cho phần cải tiến SpineNetV2 trên dataset RSNA 2024, nhằm báo cáo định kỳ với thầy hướng dẫn.

---

## 1. Tổng quan

Hệ thống em xây dựng dựa trên fork của **SpineNetV2** (Windsor et al., 2024), tập trung vào cải tiến phần **grading** (phân loại mức độ tổn thương) cho 3 điều kiện bệnh lý chính trên RSNA 2024:

- Hẹp ống sống (Spinal Canal Stenosis)
- Hẹp lỗ liên hợp trái (Left Neural Foraminal Narrowing)
- Hẹp lỗ liên hợp phải (Right Neural Foraminal Narrowing)

Mỗi điều kiện có 3 lớp: **Normal/Mild, Moderate, Severe**.

**Hai đóng góp chính** của em so với SpineNetV2 gốc:

1. **Kiến trúc**: Thêm module CBAM (Convolutional Block Attention Module) vào sau mỗi ResNet stage.
2. **Loss function**: Thay Cross-Entropy bằng Focal Loss + Uncertainty Loss để xử lý class imbalance (Severe chỉ chiếm ~6% dữ liệu).

Phụ trợ: data augmentation + oversampling cho lớp minority.

---

## 2. Kiến trúc SpineNetV2 Upstream (base pipeline)

SpineNetV2 gốc là một pipeline **4-stage** xử lý MRI cột sống:

```
[MRI Volume (3D)]
        ↓
[1. VFRResNetDetector]       → Detect corner + centroid của vertebrae
        ↓
[2. AppearanceModel + ContextModel]  → Label từng vertebra (T12, L1, ...)
        ↓
[3. Extract IVV volumes]     → Crop ảnh IVV (Intervertebral Disc) giữa 2 đốt
        ↓
[4. GradingModel]            → Phân loại tổn thương cho từng IVV
        ↓
[Output: DataFrame với 11 tasks grading mỗi IVV]
```

**Calling convention** trong code upstream:
- `detect_vb(volume, pixel_spacing)` → `vert_dicts` (danh sách vertebra detected)
- `get_ivds_from_vert_dicts(vert_dicts, volume)` → `ivd_dicts` (crop IVV volumes)
- `grade_ivds(ivd_dicts)` → DataFrame (11-task grading)

**Em chỉ can thiệp vào Stage 4 (GradingModel)**, các stage 1-3 giữ nguyên vì detection/labelling đã hoạt động tốt.

---

## 3. Phần em cải tiến (GradingModel)

### 3.1 Backbone: 3D ResNet-34

Input: 1 volume IVV có shape **(9 slices sagittal × 112 × 224)**, float32, normalized [0, 1].

```
Input [1, 9, 112, 224]
    ↓
Conv3D(1→64, kernel=3×7×7, stride=1×2×2)
BN3D + ReLU + MaxPool3D
    ↓ [64, 9, 28, 56]
Layer1: 3 × BasicBlock(64 channels)          ← CBAM1 gắn sau stage này
    ↓ [64, 9, 28, 56]
Layer2: 4 × BasicBlock(128 channels)         ← CBAM2
    ↓ [128, 9, 28, 56]   (stride=1, giữ detail)
Layer3: 6 × BasicBlock(256 channels, stride=2)  ← CBAM3
    ↓ [256, 9, 14, 28]
Layer4: 3 × BasicBlock(512 channels, stride=2)  ← CBAM4
    ↓ [512, 9, 7, 14]
AdaptiveAvgPool3D → flatten → [512]
    ↓
3 classification heads:
  - fc_spinal_canal:   Linear(512, 3)
  - fc_left_foraminal: Linear(512, 3)
  - fc_right_foraminal: Linear(512, 3)
```

**Điểm đặc biệt**: stride chỉ áp dụng cho chiều không gian (H, W), KHÔNG áp dụng cho chiều depth (D=9 slices) — vì 9 slices sagittal là thông tin mỏng, không nên giảm.

### 3.2 Module CBAM (đóng góp của em)

CBAM = Channel Attention + Spatial Attention (Woo et al., ECCV 2018). Mục tiêu: model tự học **"feature nào quan trọng"** và **"vùng không gian nào quan trọng"** trên ảnh.

```
Input feature map [B, C, D, H, W]
        ↓
[Channel Attention]
    AvgPool3D → Shared MLP → σ
    MaxPool3D → Shared MLP → σ
    (Combine + sigmoid) → weights [B, C, 1, 1, 1]
        ↓ (multiply)
Refined1 = Input × ChannelAttention
        ↓
[Spatial Attention]
    Chan-wise AvgPool → concat → Conv3D(2→1, k=7) → σ
    Chan-wise MaxPool → (same)
    → weights [B, 1, D, H, W]
        ↓ (multiply)
Output = Refined1 × SpatialAttention
```

4 module CBAM được gắn sau mỗi ResNet stage, tổng thêm khoảng **1 triệu tham số** vào model 22M.

### 3.3 Loss function (đóng góp của em)

**Vấn đề**: Severe chỉ chiếm ~6% dữ liệu → model baseline dùng CE Loss bị collapse sang lớp đa số, Recall Severe ≈ 0.

**Giải pháp em áp dụng**:

1. **Focal Loss** (Lin et al., ICCV 2017):
   ```
   FL(p_t) = -α(1 - p_t)^γ · log(p_t)
   ```
   - γ = 2.0: giảm loss cho example dễ, tập trung học example khó
   - ignore_index = -1 (cho missing label)

2. **Uncertainty Loss** (Kendall et al., 2018) — multi-task weighting tự động:
   ```
   L_total = Σ (1/(2σ_i²)) · L_i + log(σ_i)
   ```
   Model tự học weight cho 3 task (3 conditions) qua tham số σ_i có thể học.

3. **Augmentation + Oversampling**:
   - Flip, rotation, brightness, noise
   - Oversample 5x cho lớp Moderate và Severe

---

## 4. Training Pipeline

### 4.1 Data pipeline (RSNA 2024)

```
[DICOM files từ RSNA 2024]
        ↓
prepare_rsna_data.py (~8GB)
        ↓
[Preprocessed .npy files]
  rsna_preprocessed/volumes/<study_id>/<series_id>_<level>.npy
  + train_metadata.csv (mapping study_id → labels)
        ↓
RSNAPreprocessedDataset (PyTorch Dataset)
        ↓
[Volume (9, 112, 224) + labels dict]
```

Mỗi sample là 1 IVV volume đã được normalize về shape chuẩn.

### 4.2 Train/validation split

- **80/20 split by patient** (không phải by sample để tránh data leakage)
- `random_state=42` (cố định để reproduce)
- Train: ~1580 bệnh nhân | Val: ~400 bệnh nhân

### 4.3 Training setup

| Hyperparameter | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 30 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Backbone | Frozen (load pretrained từ ckpt1.pt) |
| Trainable | CBAM (~1M params) + 3 heads (~5K params) |
| Early stopping | patience 15 |

### 4.4 Hai biến thể training

**Biến thể 1: Baseline** (`train_rsna_baseline.py`)
- Model: `GradingModelBaseline` (không có CBAM)
- Loss: Cross-Entropy (+ optional class weights)
- Output: `checkpoints/best_model.pth`

**Biến thể 2: Cải tiến** (`train_rsna_attention.py`)
- Model: `GradingModelWithCBAM`
- Loss: Focal Loss + Uncertainty Loss
- Augmentation: medium (flip/rotation/brightness/noise)
- Oversampling: 5x cho Moderate + Severe
- Output: `checkpoints/best_model_attention.pth`

### 4.5 Sơ đồ luồng training

```
[Load pretrained backbone từ ckpt1.pt]
        ↓
[Freeze backbone, init CBAM + heads random]
        ↓
For each epoch:
    For each batch (volumes, labels):
        1. Forward: volumes → backbone (frozen) → CBAM (train) → heads (train)
        2. Compute per-task loss (FocalLoss per condition)
        3. Aggregate: Uncertainty Loss tự học weight 3 task
        4. Backward → AdamW step
    
    Validate on val set:
        - Accuracy per condition
        - Per-class Precision/Recall/F1
        - Save best checkpoint nếu val loss giảm
```

---

## 5. Inference Pipeline

### 5.1 Full SpineNetV2 inference (upstream, end-to-end)

Dùng cho ảnh MRI nguyên bản (chưa preprocess):

```
[Raw MRI volume + pixel_spacing]
        ↓
SpineNet.detect_vb(volume, pixel_spacing)
        ↓
[vert_dicts: corner + centroid + label mỗi vertebra]
        ↓
SpineNet.get_ivds_from_vert_dicts(vert_dicts, volume)
        ↓
[ivd_dicts: crop volumes (9, 112, 224) per IVV]
        ↓
GradingModel(ivd_volumes)  ← Đây là model em train
        ↓
[DataFrame: per-IVV prediction]
  columns: study_id, level, spinal_canal, left_foraminal, right_foraminal
  values:  Normal/Mild | Moderate | Severe
```

### 5.2 Inference chỉ phần em train (từ checkpoint)

Áp dụng khi đã có IVV volumes đã preprocess (từ RSNA):

```
[Load checkpoint: best_model_attention.pth]
        ↓
Build GradingModelWithCBAM + load_state_dict
        ↓
model.eval()  (disable dropout, BN running stats)
        ↓
For each IVV volume:
    1. volume.shape = (9, 112, 224)
    2. Add batch dim + channel dim: (1, 1, 9, 112, 224)
    3. Forward: output = model(volume)
    4. output là dict với 3 key:
       - 'spinal_canal': logits [1, 3]
       - 'left_foraminal': logits [1, 3]
       - 'right_foraminal': logits [1, 3]
    5. Softmax + argmax → class index (0=Normal, 1=Moderate, 2=Severe)
```

### 5.3 Script inference có sẵn

- `test_spinenet.py` — end-to-end inference với MRI gốc
- `test_spinenet_on_rsna.py` — apply upstream pipeline lên DICOM RSNA
- `test_rsna_preprocessed.py` — test checkpoint của em trên preprocessed data

---

## 6. Kết quả hiện tại

### 6.1 So sánh 3 baseline trên RSNA 2024 (Table 1 — patient-level)

| Model | SC F1 | LF F1 | RF F1 | Average F1 |
|---|---|---|---|---|
| Ning Shen (Kaggle SOTA) | 81.20% | 79.43% | 81.48% | **80.70%** |
| SpineNetV2 upstream | 78.79% | 77.30% | 76.40% | **77.50%** |
| MedGemma (VLM, zero-shot) | 31.17% | 13.86% | 10.75% | **18.59%** |

→ SpineNetV2 có F1 competitive (chỉ kém Ning Shen ~3%) và code public → chọn làm target cải tiến.

### 6.2 Baseline trained của em vs CBAM+Focal (Table 2 — per-IVV, sẽ làm)

**Baseline (ResNet34 từ ckpt1.pt, CE Loss, no CBAM)** — từ `experiments/rsna/base_line.txt`:

| Condition | Overall Acc | Severe Recall | Severe F1 |
|---|---|---|---|
| Spinal Canal | 88.78% | 31.2% | 40.8% |
| Left Foraminal | 78.18% | **0.0%** | **0.0%** |
| Right Foraminal | 78.55% | **0.0%** | **0.0%** |

→ Baseline miss hoàn toàn lớp Severe ở foraminal conditions.

**CBAM + Focal Loss** — kết quả sơ bộ từ log training:

| Condition | Severe Recall | Ghi chú |
|---|---|---|
| Spinal Canal | **~83.5%** | Cải thiện từ 31.2% → 83.5% |
| Left Foraminal | **~31-42%** | Cải thiện từ 0% → 31-42% |
| Right Foraminal | **~31-42%** | Cải thiện từ 0% → 31-42% |

→ Đóng góp chính của em: **recover lớp Severe** — có ý nghĩa lâm sàng lớn vì không bỏ sót ca nặng.

Em đang làm lại phần extract chính xác metric từ checkpoint `best_model_attention.pth` để điền đầy bảng so sánh (Phase 2 Stage A trong plan).

---

## 7. Hướng mở rộng (đang làm)

### 7.1 External validation trên dataset SPIDER

- SPIDER là dataset MRI cột sống độc lập, public.
- Label khác RSNA (pfirrmann 5-class, spondylolisthesis binary, disc_herniation binary).
- Chiến lược: **Linear Probing** — freeze backbone + CBAM (đã train trên RSNA), chỉ train heads mới cho 3 task SPIDER.
- Mục tiêu: chứng minh feature CBAM học được từ RSNA **transfer được** sang task khác.
- Code đã sẵn (`train_spider.py`, `spider_dataloader.py`, `grading_spider.py`), chỉ chờ confirm protocol với thầy rồi chạy.

### 7.2 Các việc còn lại

1. **Phase 2 Stage A**: Extract chính xác metric của CBAM+Focal checkpoint → điền bảng Table 2.
2. **Phase 2 Stage B (optional)**: Ablation riêng CBAM only và Focal only để chứng minh 2 component đều cần.
3. **Phase 3**: Chạy SPIDER Linear Probe — chờ thầy confirm interpretation của "zero-shot".
4. **Viết paper**: Sau khi có đầy Table 1 + Table 2 → target hội nghị Rank-C (EMBC, SPIE Medical Imaging, ICIP, hoặc MICCAI Workshop).

---

## 8. Câu hỏi em mong thầy hướng dẫn

1. **Về SPIDER**: Thầy muốn em làm **Linear Probe only** (freeze tất cả, chỉ train head mới), hay thêm **Full Fine-tune** để có upper-bound comparison? (Xem message trước em gửi thầy.)
2. **Về scope paper**: Em có nên thêm **ablation CBAM-only vs Focal-only** không, hay chỉ cần **baseline vs full**? Em nghiêng về "full ablation" cho mạnh, nhưng tốn thêm ~2 giờ GPU.
3. **Về venue**: Thầy có target hội nghị cụ thể nào không (EMBC 2026? SPIE? MICCAI Workshops?) để em chuẩn bị đúng format + deadline?
4. **Per-level evaluation (Table 1)**: Em đã thử đánh giá baseline ở mức per-level như thầy gợi ý. SpineNetV2 upstream bị giảm F1 từ 77.5% → 43% do không có per-level output. Thầy có muốn em giữ patient-level cho Table 1 (fair với bản chất của 3 model khác regime) và dùng per-level cho Table 2 không?

---

## Phụ lục A — Files code chính

| File | Vai trò |
|---|---|
| `spinenet/models/grading_baseline.py::GradingModelBaseline` | ResNet34 baseline (no CBAM) |
| `spinenet/models/grading_attention.py::GradingModelWithCBAM` | ResNet34 + CBAM (proposed) |
| `spinenet/models/attention.py::CBAM` | CBAM module (Channel + Spatial Attention) |
| `spinenet/losses.py::FocalLoss` | Focal Loss với ignore_index |
| `spinenet/losses.py::UncertaintyLoss` | Multi-task weighting (Kendall et al.) |
| `spinenet/augmentation.py::get_training_augmentation` | Augmentation pipeline |
| `spinenet/augmentation.py::OversamplingDataset` | Wrapper oversample minority class |
| `rsna_preprocessed_dataloader.py` | Load .npy preprocessed volumes |
| `train_rsna_baseline.py` | Training script baseline |
| `train_rsna_attention.py` | Training script CBAM+Focal |
| `test_rsna_preprocessed.py` | Inference script trên RSNA val |

## Phụ lục B — Checkpoint

| File | Mô tả | Size |
|---|---|---|
| `~/.spinenet/weights/ckpt1.pt` | Pretrained ResNet34 3D medical backbone (upstream) | ~90 MB |
| `checkpoints/best_model.pth` | Baseline em train (ResNet34 + CE Loss) | ~100 MB |
| `checkpoints/best_model_attention.pth` | CBAM + Focal Loss em train | ~100 MB |

## Phụ lục C — Workflow diagram tóm tắt

```
┌──────────────────────────────────────────────────────────────┐
│  DATA                                                         │
│  ┌──────────────┐    ┌─────────────────────────────────────┐ │
│  │ RSNA 2024    │ →  │ Preprocess: DICOM → (9,112,224).npy  │ │
│  │ (~2000 BN)   │    └─────────────────────────────────────┘ │
│  └──────────────┘                                             │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│  TRAINING                                                     │
│  ┌─────────────────┐     ┌───────────────────────────────┐  │
│  │ Load ckpt1.pt   │ →   │ Freeze backbone               │  │
│  │ (medical pre-   │     │ Train: CBAM + 3 heads          │  │
│  │  trained R34)   │     │ Loss: Focal + Uncertainty     │  │
│  └─────────────────┘     │ Aug: medium + oversample 5x   │  │
│                          │ 30 epochs, AdamW, lr 1e-3     │  │
│                          └──────────────┬────────────────┘  │
└─────────────────────────────────────────┼────────────────────┘
                                          ↓
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ checkpoints/best_model_attention.pth                    │ │
│  │ - Backbone (frozen state from ckpt1.pt)                 │ │
│  │ - 4 × CBAM modules (learned on RSNA)                    │ │
│  │ - 3 × Linear heads (3-class each)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│  INFERENCE                                                    │
│  ┌────────────────┐   ┌──────────────────────────────────┐  │
│  │ Load checkpoint│ → │ Forward: (9,112,224) → 3 × logits │  │
│  │ model.eval()   │   │ Softmax → argmax → prediction     │  │
│  └────────────────┘   └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

*Tài liệu này em có cập nhật thường xuyên. Full code: https://github.com/trungkien04102002/spinet-v2*
