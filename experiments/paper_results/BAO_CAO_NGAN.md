# Báo cáo Ngắn — Kiến trúc Training & Inference

**Kien, 2026-04-20** | SpineNetV2 cải tiến trên RSNA 2024

---

## Cải tiến

Cải tiến phần **grading** (phân loại tổn thương) của SpineNetV2 cho 3 điều kiện trên RSNA 2024: Hẹp ống sống, Hẹp lỗ liên hợp trái, Hẹp lỗ liên hợp phải (mỗi cái 3 lớp: Normal/Mild, Moderate, Severe).

Hai đóng góp:
1. **CBAM** (Channel + Spatial Attention) gắn sau mỗi ResNet stage.
2. **Focal Loss + Uncertainty Loss** để xử lý class imbalance (Severe ~6% dữ liệu).

Mục tiêu: tăng Recall lớp Severe (baseline bị collapse → Recall ≈ 0 cho foraminal).

---

## Kiến trúc

### Backbone: 3D ResNet-34

Input: IVV volume **(9 slices × 112 × 224)**, float32 [0,1].

```
Input [1, 9, 112, 224]
  → Conv3D(1→64) + BN + ReLU + MaxPool
  → Layer1 (64 ch, 3 blocks)   → CBAM₁
  → Layer2 (128 ch, 4 blocks)  → CBAM₂
  → Layer3 (256 ch, 6 blocks)  → CBAM₃
  → Layer4 (512 ch, 3 blocks)  → CBAM₄
  → GlobalAvgPool → flatten [512]
  → 3 heads: fc_spinal_canal, fc_left_foraminal, fc_right_foraminal
              mỗi head Linear(512 → 3 classes)
```

Stride chỉ áp cho H,W — giữ nguyên depth D=9 (slices sagittal mỏng, không giảm).

### CBAM module

```
feature [B,C,D,H,W]
  → Channel Attention (AvgPool + MaxPool → shared MLP → σ) → scale
  → Spatial Attention (chan-wise pool → Conv3D(k=7) → σ)  → scale
  → refined feature
```

4 CBAM thêm ~1M params vào backbone 22M.

### Loss

- **Focal Loss** (γ=2.0, ignore_index=-1) cho từng task
- **Uncertainty Loss** (Kendall 2018) tự học weight 3 task qua σᵢ learnable
- **Augmentation** (flip/rotation/brightness/noise) + **Oversampling 5x** cho Moderate+Severe

---

## Training Pipeline

```
[RSNA DICOM] + [train_label_coordinates.csv (x,y tâm IVV, level, condition)]
                                  ↓
          Crop IVV dựa vào GT coordinates (không dùng detector)
                                  ↓
           Resample về shape chuẩn → [.npy (9,112,224) mỗi IVV]
                                  ↓
                    80/20 split by patient (seed=42)
                                  ↓
      Load pretrained backbone (ckpt1.pt) → freeze
                                  ↓
      For 30 epochs:
          forward → FocalLoss per task → UncertaintyLoss aggregate → AdamW step
          validate + save best checkpoint
                                  ↓
               [checkpoints/best_model_attention.pth]
```

**Note quan trọng về crop IVV**: Khi **training**, em dùng `train_label_coordinates.csv` (file tọa độ GT do RSNA cung cấp) để cắt vùng IVV từ volume gốc. Cách này giúp model grading được học trên IVV định vị chính xác, tách bạch hẳn với bài toán detection. Khi **inference** trên ảnh mới không có coords → dùng detector của SpineNetV2 upstream (`detect_vb`) để sinh coords trước, rồi mới grade. Đây là setup chuẩn trong các paper grading medical imaging — cho phép đánh giá phần cải tiến (grading) không bị nhiễu bởi lỗi detection.

**Hyperparams**: AdamW, lr=1e-3, wd=1e-4, batch=32, 30 epochs, ReduceLROnPlateau patience=5, early stop 15.

**Trainable**: chỉ CBAM (~1M) + 3 heads (~5K). Backbone frozen.

**Hai biến thể training**:
- `train_rsna_baseline.py` → `best_model.pth` (ResNet34 + CE Loss, không CBAM)
- `train_rsna_attention.py` → `best_model_attention.pth` (ResNet34 + CBAM + Focal)

---

## Inference Pipeline

### Option A — Full pipeline (ảnh MRI gốc)

```
MRI volume + pixel_spacing
  → SpineNet.detect_vb()           [upstream detector]
  → SpineNet.get_ivds_from_vert_dicts()  [crop IVV]
  → GradingModelWithCBAM(ivd)      [em train]
  → DataFrame per-IVV prediction
```

### Option B — Từ IVV volume đã preprocess (test RSNA val)

```
Load checkpoint → GradingModelWithCBAM + load_state_dict → model.eval()
  → volume (1,1,9,112,224) → forward
  → 3 logits dict {spinal_canal: [1,3], left_foraminal: [1,3], right_foraminal: [1,3]}
  → softmax + argmax → class (0=Normal, 1=Moderate, 2=Severe)
```

Script: `test_rsna_preprocessed.py`.

---

## Kết quả hiện tại

### Table 1 — 3 baseline (patient-level, F1)

| Model | SC | LF | RF | Avg |
|---|---|---|---|---|
| Ning Shen (SOTA) | 81.2% | 79.4% | 81.5% | **80.7%** |
| SpineNetV2 upstream | 78.8% | 77.3% | 76.4% | **77.5%** |
| MedGemma (zero-shot VLM) | 31.2% | 13.9% | 10.8% | 18.6% |

→ Chọn SpineNetV2 để cải tiến (F1 competitive + code public).

### Table 2 — Baseline em train vs CBAM+Focal (per-IVV, RSNA val)

| Variant | Condition | Overall Acc | Severe Recall | Severe F1 | Macro F1 |
|---|---|---|---|---|---|
| Baseline (ResNet34, CE) | Spinal Canal | **88.78%** | 31.2% | 40.8% | 46.6% |
| + CBAM + Focal | Spinal Canal | 85.23% | **83.5%** | **51.2%** | **59.1%** |
| Baseline | Left Foraminal | **78.18%** | 0.0% | 0.0% | 37.9% |
| + CBAM + Focal | Left Foraminal | 58.15% | **42.3%** | **27.1%** | **46.0%** |
| Baseline | Right Foraminal | **78.55%** | 0.0% | 0.0% | 38.2% |
| + CBAM + Focal | Right Foraminal | 64.11% | **31.1%** | **26.8%** | **48.9%** |

→ Đóng góp chính: khôi phục lớp Severe (0 → 31-83%) — có ý nghĩa lâm sàng. Overall Accuracy giảm là trade-off hợp lý cho screening task.

---

## Tiếp theo

1. Hoàn tất Table 2 (extract chính xác metric CBAM+Focal).
2. SPIDER external validation bằng Linear Probing (chờ thầy confirm protocol).
3. Viết paper target Rank-C (EMBC / SPIE / ICIP / MICCAI Workshop).
