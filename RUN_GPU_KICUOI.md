# GPU run cheatsheet — kì cuối (SOTA + F1 #1/#2/#3)

Copy-paste từng dòng, chạy lần lượt, tự track. Giống style cũ.
Mọi lệnh chạy từ repo root `spinet-v2`, branch `biomedclip-integration`.
Số liệu = seed 42 (mặc định). File output + scp ở cuối.

```bash
# tmux (rớt SSH không mất job)
tmux new -s gpu          # tạo / tmux a -t gpu để vào lại
tmux set -g mouse on
# tmux set -g mouse off
```

## 0. Setup box (1 lần, trên box TRẮNG vừa SSH vào)

```bash
# repo public → kéo mình vast_setup.sh về, nó tự clone repo + venv + cài deps
wget https://raw.githubusercontent.com/trungkien04102002/spinet-v2/biomedclip-integration/vast_setup.sh
bash vast_setup.sh --branch biomedclip-integration

cd spinet-v2
git branch --show-current                   # phải = biomedclip-integration
nvidia-smi                                   # xác nhận thấy RTX-4090

# vast_setup cài deps VÀO spinenet-venv. Nhiều template Vast có sẵn trong python
# hệ thống nên python3 trần chạy luôn — nhưng test 1 dòng cho chắc:
python3 -c "import torch, SimpleITK, pydicom, open_clip; print('OK', torch.cuda.is_available())"
#   → in "OK True"  : chạy thẳng, KHỎI activate.
#   → ModuleNotFoundError: source spinenet-venv/bin/activate   (rồi test lại)
```

## 1. DATA + WEIGHTS (prereq — thiếu là job chạy sai/ from scratch)

```bash
# (a) RSNA preprocessed 8GB — CHO SOTA
chmod +x 2_download_preprocessed.sh
./2_download_preprocessed.sh 1DCndO_ppTDMGT1H19b8XIqfureMqABi_

# (b) Pretrained backbone (~/.spinenet/weights) — nền chung (fallback).
chmod +x 3_download_weights.sh
./3_download_weights.sh 1GCmJ0OuNdw9c1E4giLwyA9EK4uwWL6HT
ls -lh ~/.spinenet/weights/                 # phải thấy weights.pt

# (b2) CBAM checkpoint đã train (best_model_attention.pth) — CHO warm-start #1/#2/#3.
#      md5-verified = phase2_cbam.pth trên Drive. #1/#2/#3 train TIẾP từ đây.
mkdir -p checkpoints/rsna
python3 -m gdown "https://drive.google.com/uc?id=1HRGzEk0UM1fHPkWlOh31psApah7S_02i" -O checkpoints/rsna/best_model_attention.pth
ls -lh checkpoints/rsna/best_model_attention.pth    # ~243MB

# (c) RSNA T1 crops 17GB — CHO #1/#2/#3. Đã zip + up Drive → tải thẳng:
chmod +x 6_download_t1.sh
./6_download_t1.sh                          # file ID đã baked sẵn trong script
ls rsna_preprocessed_t1/volumes | wc -l     # kỳ vọng ~19689
#
# (Fallback nếu link Drive hỏng — tái sinh từ raw Kaggle:)
# kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
# python3 experiments/f1_improvement/prep_t1_crops.py
```

## 2. SMOKE TEST (bắt buộc — 2 script train này CHƯA chạy GPU thật bao giờ)

```bash
CK=checkpoints/rsna/best_model_attention.pth       # warm-start từ CBAM đã train
# --class-weight-mode để smoke-test luôn nhánh có class weight (nhánh này CHƯA từng
# chạy GPU thật — trước đây weight nằm trên CPU sẽ crash ngay batch đầu, đã fix).
python3 experiments/f1_improvement/train_t1_foraminal.py --fast-dev --cbam-checkpoint $CK \
    --class-weight-mode effective --select-by severe_f1
python3 experiments/multiview/train_multiview.py --fusion concat --fast-dev --allow-missing-t1 \
    --cbam-checkpoint $CK --class-weight-mode effective --select-by severe_f1
python3 experiments/multiview/train_multiview.py --fusion gated  --fast-dev --allow-missing-t1 \
    --cbam-checkpoint $CK --class-weight-mode effective --select-by severe_f1
# OOM? hạ --batch-size (T1 default 32, multiview default 16).
# LƯU Ý: quên --cbam-checkpoint → banner "⚠️ NOT warm-starting"; sai path → dừng báo lỗi.
# Smoke test giờ cắt còn 24 train / 12 val (class-stratified) → xong trong ~1 phút.
# Class weights in ra ở fast-dev KHÔNG phản ánh imbalance thật (subset đã stratify);
# chỉ cần thấy nó FINITE và có in ra là đạt.
# Đã verify chạy được trên CPU máy Mac 2026-08-22: cả 3 lệnh exit=0.
```

## 3. SOTA (chạy được ngay khi có data (a) — kick trước, làm (b)(c) song song)

```bash
# ~40–80 phút, 2 model × seed 42. Ghi log vào experiments/sota_comparison/logs/
bash experiments/sota_comparison/run_all.sh
# (3-seed nếu muốn số chắc hơn:)
# SOTA_SEEDS="42 123 456" bash experiments/sota_comparison/run_all.sh
```

## 4. F1 #1 — T1-foraminal (ứng viên win to nhất)

```bash
CK=checkpoints/rsna/best_model_attention.pth
python3 experiments/f1_improvement/train_t1_foraminal.py --cbam-checkpoint $CK \
    --class-weight-mode effective --select-by severe_f1 \
    --epochs 30 --batch-size 32 --lr 1e-3 \
    2>&1 | tee experiments/f1_improvement/run_t1.log
# ⚠️ BẮT BUỘC 2 flag trên. Lần chạy 2026-07 thiếu chúng → CE trần, model đoán
#    toàn Normal, Severe F1 = 0.0000 suốt 10 epoch. Xem docs/LVTN_phase3/IMPROVE_F1_PLAN.md.
# → checkpoints/t1_foraminal/best_model_t1_foraminal_cbam.pth
# → checkpoints/t1_foraminal/t1_foraminal_cbam_best_metrics.json (+ _log.csv + .txt)
```

## 5. F1 #2 + #3 — multi-view fusion

```bash
CK=checkpoints/rsna/best_model_attention.pth
# ⚠️ --allow-missing-t1 là BẮT BUỘC. 51 hàng T2 (10 left, 41 right) không có crop
#    T1 tương ứng; thiếu cờ này dataloader raise FileNotFoundError và GIẾT job giữa
#    epoch 1. Kiểm rồi: 50/51 hàng đó vốn đã có nhãn -1 (loss bỏ qua sẵn), đúng 1 hàng
#    có nhãn thật bị thay bằng volume zeros → mất 1 mẫu / 9748.

# #2 concat
python3 experiments/multiview/train_multiview.py --fusion concat --cbam-checkpoint $CK \
    --allow-missing-t1 --class-weight-mode effective --select-by severe_f1 \
    --epochs 30 --batch-size 16 --lr 1e-3 \
    2>&1 | tee experiments/multiview/run_concat.log
# → experiments/multiview/checkpoints/best_model_multiview_concat.pth (+ _best_metrics.json/.txt + _log.csv)

# #3 gated leader-supporter (đúng ý thầy)
python3 experiments/multiview/train_multiview.py --fusion gated --cbam-checkpoint $CK \
    --allow-missing-t1 --class-weight-mode effective --select-by severe_f1 \
    --epochs 30 --batch-size 16 --lr 1e-3 \
    2>&1 | tee experiments/multiview/run_gated.log
# → .../best_model_multiview_gated.pth  +  multiview_gated_best_metrics.json
```

**Đọc kết quả cho đúng (2 điều đã rà ra 2026-08-22):**
- **`gated` có capacity THẤP HƠN `concat`.** Gate là tổ hợp lồi `w_t2·e_t2 + w_t1·e_t1`
  (softmax, sum=1) → vector 512 chiều; concat giữ cả hai → 1024 chiều. **concat thắng gated
  là bình thường, không phải bug.** Giá trị của gated là *giải thích được*
  (`model.get_gate_weights()` trả trọng số leader/supporter từng ca) — đúng thứ thầy muốn nhìn.
- **Encoder T1 đang bị đóng băng ở trọng số học từ T2.** `freeze_backbone(True)` (mặc định)
  chỉ để `fc_*` + `cbam*` train; conv backbone giữ nguyên. T1/T2 khác contrast rõ rệt, nên nếu
  epoch 5 Severe F1 vẫn thấp thì **`--unfreeze-backbone --lr 1e-4` không phải "leo thang khẩn cấp"**
  mà nhiều khả năng mới là setup đúng cho một phép đổi chuỗi ảnh.

## 6. Theo dõi trong lúc chạy

```bash
watch -n 30 nvidia-smi
df -h            # dung lượng đĩa (T1 17GB + data 8GB dễ đầy)
du -sh .
ls -lh checkpoints/t1_foraminal/ experiments/multiview/checkpoints/
tail -f experiments/sota_comparison/logs/brendanartley_seed42.log
```

## 7. Kéo kết quả về Mac (chạy trên MÁY LOCAL, sửa PORT/IP theo instance)

```bash
# NHẸ — metrics + log (cái Claude cần để ráp bảng report). Kéo cả thư mục cho gọn:
scp -P <PORT> -r root@<IP>:/root/spinet-v2/checkpoints/t1_foraminal/'*'.{json,csv,txt}    experiments/f1_improvement/results/
scp -P <PORT> -r root@<IP>:/root/spinet-v2/experiments/multiview/checkpoints/'*'.{json,csv,txt}   experiments/multiview/
scp -P <PORT> -r root@<IP>:/root/spinet-v2/experiments/f1_improvement/run_t1.log             experiments/f1_improvement/
scp -P <PORT> -r root@<IP>:/root/spinet-v2/experiments/multiview/run_*.log                   experiments/multiview/
scp -P <PORT> -r root@<IP>:/root/spinet-v2/experiments/sota_comparison/comparison_table.*    experiments/sota_comparison/
scp -P <PORT> -r root@<IP>:/root/spinet-v2/experiments/sota_comparison/logs                  experiments/sota_comparison/

# CHECKPOINTS (nặng — chỉ kéo nếu cần eval/viz thêm)
scp -P <PORT> root@<IP>:/root/spinet-v2/checkpoints/t1_foraminal/best_model_t1_foraminal_cbam.pth                     checkpoints/t1_foraminal/
scp -P <PORT> root@<IP>:/root/spinet-v2/experiments/multiview/checkpoints/best_model_multiview_concat.pth             experiments/multiview/checkpoints/
scp -P <PORT> root@<IP>:/root/spinet-v2/experiments/multiview/checkpoints/best_model_multiview_gated.pth              experiments/multiview/checkpoints/
```

## Ghi chú
- Chỉ **model multi-view mới** train thật. SOTA/threshold là 1 lượt. Model cũ KHÔNG train lại.
- seed khác 42 → tên file thêm hậu tố `_seed<N>` (vd `best_model_multiview_gated_seed123.pth`).
- Xong seed 42, nếu #1 thắng thì mới mở rộng 3 seed cho số cuối cùng.
- Kéo JSON về xong → báo Claude ráp bảng vào report LVTN.
```
