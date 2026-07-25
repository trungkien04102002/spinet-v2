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

## 0. Setup box (1 lần)

```bash
bash vast_setup.sh --branch biomedclip-integration
cd spinet-v2
git pull                                   # lấy fix mới nhất (run_all.sh PYTHONPATH)
pip3 install -r requirements.txt
git branch --show-current                  # phải = biomedclip-integration
nvidia-smi                                  # xác nhận thấy GPU
```

## 1. DATA + WEIGHTS (prereq — thiếu là job chạy sai/ from scratch)

```bash
# (a) RSNA preprocessed 8GB — CHO SOTA
chmod +x 2_download_preprocessed.sh
./2_download_preprocessed.sh 1DCndO_ppTDMGT1H19b8XIqfureMqABi_

# (b) Pretrained backbone (~/.spinenet/weights) — CHO #1/#2/#3
#     Thiếu cái này → train from scratch (chỉ warning, KHÔNG lỗi) → số vô nghĩa.
chmod +x 3_download_weights.sh
./3_download_weights.sh 1GCmJ0OuNdw9c1E4giLwyA9EK4uwWL6HT
ls -lh ~/.spinenet/weights/                 # phải thấy weights.pt

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
python3 experiments/f1_improvement/train_t1_foraminal.py --fast-dev
python3 experiments/multiview/train_multiview.py --fusion concat --fast-dev
python3 experiments/multiview/train_multiview.py --fusion gated  --fast-dev
# OOM? hạ --batch-size (T1 default 32, multiview default 16).
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
python3 experiments/f1_improvement/train_t1_foraminal.py \
    --epochs 30 --batch-size 32 --lr 1e-3
# → checkpoints/t1_foraminal/best_model_t1_foraminal_cbam.pth
# → checkpoints/t1_foraminal/t1_foraminal_cbam_best_metrics.json
```

## 5. F1 #2 + #3 — multi-view fusion

```bash
# #2 concat
python3 experiments/multiview/train_multiview.py --fusion concat \
    --epochs 30 --batch-size 16 --lr 1e-3
# → experiments/multiview/checkpoints/best_model_multiview_concat.pth
# → experiments/multiview/checkpoints/multiview_concat_best_metrics.json

# #3 gated leader-supporter (đúng ý thầy)
python3 experiments/multiview/train_multiview.py --fusion gated \
    --epochs 30 --batch-size 16 --lr 1e-3
# → .../best_model_multiview_gated.pth  +  multiview_gated_best_metrics.json
```

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
# METRICS JSON (nhẹ — cái Claude cần để ráp bảng report)
scp -P <PORT> root@<IP>:/root/spinet-v2/checkpoints/t1_foraminal/t1_foraminal_cbam_best_metrics.json                 experiments/f1_improvement/results/
scp -P <PORT> root@<IP>:/root/spinet-v2/experiments/multiview/checkpoints/multiview_concat_best_metrics.json          experiments/multiview/
scp -P <PORT> root@<IP>:/root/spinet-v2/experiments/multiview/checkpoints/multiview_gated_best_metrics.json           experiments/multiview/
scp -P <PORT> root@<IP>:/root/spinet-v2/experiments/sota_comparison/comparison_table.md                              experiments/sota_comparison/
scp -P <PORT> root@<IP>:/root/spinet-v2/experiments/sota_comparison/comparison_table.tex                             experiments/sota_comparison/

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
