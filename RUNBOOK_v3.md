# RUNBOOK v3 — RSNA Retrain Workflow (2026-05-03)

Chia 3 phần:
- **PART 1**: LOCAL MACHINE (Mac của bạn) — chuẩn bị
- **PART 2**: SSH MACHINE (Vast.ai) — chạy training
- **PART 3**: VỀ LOCAL — pull results + fill report

Mỗi command = **1 dòng duy nhất**, copy-paste an toàn.

Branch: `biomedclip-integration`, latest commit: `e2c7736`.

## PRIORITY (cập nhật 2026-05-03)

Chỉ chạy **4 run quan trọng**, skip 3 run optional để tiết kiệm ~6h:

| Step | Run | Priority | Time |
|---|---|---|---|
| 2.9 | Baseline | ✅ DONE | done |
| 2.10 | CBAM | ✅ DONE/IN-PROGRESS | done |
| 2.11 | Hybrid full | **MUST** | 2h |
| 2.12 | Hybrid cbam_only ablation | **MUST** (Q6) | 2h |
| 2.13 | Hybrid biomedclip_only ablation | **SKIP** — Run 8 thay thế | — |
| 2.14 | Hybrid gated fusion | **SKIP** — defense lý thuyết Q-JS-3b đã đủ | — |
| 2.15 | Hybrid modality_dropout | **SKIP** — không cần cho Rank-C | — |
| 2.16 | Linear probe BiomedCLIP | **MUST** (Q-JS-1) | 15min |
| 2.17 | Linear probe ImageNet ViT | **MUST** (Q-JS-2) | 15min |
| 2.20 | SPIDER Phase 4 | OPTIONAL | 3.5h |

→ **Total còn lại: ~4.5h** (4 run MUST). SPIDER nếu có time thêm 3.5h.

---

## PART 1 — LOCAL MACHINE (Mac của bạn)

### 1.1. Verify branch ready (chạy trên local Mac)

```bash
cd ~/spinet-v2
```

```bash
git status
```

```bash
git log --oneline -5
```

→ Phải thấy commit `63326d6` ở top. Nếu khác, báo tôi.

### 1.2. (Optional) Verify preprocessed data trên local

```bash
ls rsna_preprocessed/volumes/ | head -3
```

→ Nếu có data, OK. Local chỉ dùng để verify, không train.

### 1.3. Chuẩn bị 2 GDRIVE ID

Bạn cần có sẵn:
- **GDRIVE_RSNA**: ID file `rsna_preprocessed.zip` (~8GB) trên Google Drive
- **GDRIVE_SPINENET**: ID file `ckpt1.pt` (SpineNet pretrained weights)

→ Nếu chưa biết ID nào, search trong file `2_download_preprocessed.sh` hoặc hỏi tôi check.

---

## PART 2 — SSH MACHINE (Vast.ai)

### 2.1. SSH vào máy

```bash
ssh -p 39311 root@220.130.209.122 -L 8080:localhost:8080
```

### 2.2. Clone repo

```bash
cd /workspace 2>/dev/null || cd ~
```

```bash
git clone -b biomedclip-integration https://github.com/trungkien04102002/spinet-v2.git
```

```bash
cd spinet-v2
```

### 2.3. Setup môi trường

```bash
./vast_setup.sh --branch biomedclip-integration
```

### 2.4. Activate venv + verify GPU

```bash
source spinenet-venv/bin/activate
```

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

```bash
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

→ Phải thấy `cuda: True` và GPU name. Nếu `False`, báo tôi.

### 2.5. Bật tmux để session không mất khi SSH disconnect

```bash
tmux new -s train
```

(Sau đó nếu mất kết nối, SSH lại rồi `tmux attach -t train` để vào lại session.)

### 2.6. Download RSNA preprocessed data (~8GB, ~10-20 phút)

Thay `YOUR_GDRIVE_ID_HERE` bằng GDRIVE_RSNA của bạn:

```bash
./2_download_preprocessed.sh YOUR_GDRIVE_ID_HERE
```

Verify:

```bash
ls rsna_preprocessed/volumes/ | head
```

```bash
wc -l rsna_preprocessed/train_metadata.csv
```

→ Phải có ~9748 sample.

### 2.7. Download SpineNet pretrained weights

Thay `SPINENET_GDRIVE_ID` bằng GDRIVE_SPINENET:

```bash
./3_download_weights.sh SPINENET_GDRIVE_ID
```

Verify:

```bash
ls -la ~/.spinenet/weights/ckpt1.pt
```

### 2.8. Tạo folder output v3 (tách hoàn toàn khỏi old runs)

```bash
mkdir -p checkpoints/v3_20260503/baseline checkpoints/v3_20260503/cbam checkpoints/v3_20260503/hybrid checkpoints/v3_20260503/linear_probe checkpoints/v3_20260503/spider experiments/v3_20260503
```

Verify:

```bash
ls checkpoints/v3_20260503/ experiments/v3_20260503/
```

### 2.9. Train RSNA baseline (~1.5h)

```bash
python3 train_rsna_baseline.py --epochs 25 --batch-size 64 --lr 1e-3 --save-dir checkpoints/v3_20260503/baseline 2>&1 | tee experiments/v3_20260503/run_baseline.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/baseline/
```

```bash
cat checkpoints/v3_20260503/baseline/baseline_best_metrics.txt
```

→ **Paste output cat cho tôi để fill Bảng 1A/1C Base column.**

### 2.10. Train CBAM (~2h, **CẦN cho Hybrid input**)

```bash
python3 train_rsna_attention.py --epochs 25 --batch-size 64 --lr 1e-3 --use-focal --use-uncertainty true --class-weight-mode sqrt --augmentation medium --oversample-factor 5 --save-dir checkpoints/v3_20260503/cbam 2>&1 | tee experiments/v3_20260503/run_cbam.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/cbam/
```

```bash
cat checkpoints/v3_20260503/cbam/attention_best_metrics.txt
```

→ **Paste output cat cho tôi để fill Bảng 1A/1C Ours column.**

### 2.11. Hybrid FULL run (~2h, main result cho paper)

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --epochs 20 --batch-size 32 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_full.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/hybrid/
```

```bash
cat checkpoints/v3_20260503/hybrid/hybrid_best_metrics.txt
```

→ **Paste output cat cho tôi để fill Bảng 2 (Theme 3 main).**

### 2.12. Hybrid CBAM-only ablation (~2h, Q6 + Q10)

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --ablate-branch cbam_only --epochs 20 --batch-size 32 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_cbam_only.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/hybrid/
```

```bash
cat checkpoints/v3_20260503/hybrid/hybrid_cbam_only_best_metrics.txt
```

→ **Paste cho tôi để fill Bảng JS-1 (CBAM-only column).**

### 2.13. Hybrid BiomedCLIP-only ablation — **SKIP** (~2h, Q6 + Q10)

> **Skip lý do:** Run 8 (Linear probe BMC) đã cover "BMC một mình", rigorous hơn vì frozen hoàn toàn. Defense thầy: "Em đã làm linear probe BMC trên RSNA rồi (Run 8)."

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --ablate-branch biomedclip_only --epochs 20 --batch-size 32 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_bmc_only.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/hybrid/
```

```bash
cat checkpoints/v3_20260503/hybrid/hybrid_biomedclip_only_best_metrics.txt
```

→ **Paste cho tôi để fill Bảng JS-1 (BMC-only hybrid-head column).**

### 2.14. Hybrid Gated Fusion — **SKIP** (~2h, Q5 advisor request — đổi MLP)

> **Skip lý do:** Q-JS-3b đã có math defense — gated fusion ≡ concat-MLP với learned gate trên 2 single vector. Defense thầy: "Lý thuyết đã chứng minh equivalent, không cần empirical compare."

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --fusion gated --epochs 20 --batch-size 32 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_gated.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/hybrid/
```

```bash
cat checkpoints/v3_20260503/hybrid/hybrid_gated_best_metrics.txt
```

→ **Paste cho tôi để fill Q-JS-3b defense (concat-MLP vs gated comparison).**

### 2.15. Hybrid Modality Dropout — **SKIP** (~2h, Q10 robustness)

> **Skip lý do:** Robustness study cho deployment, không cần cho Rank-C accuracy benchmark. Defense thầy: "Paper focus accuracy, robustness study deferred to future work."

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --modality-dropout 0.15 --epochs 20 --batch-size 32 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_mdrop.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/hybrid/
```

```bash
cat checkpoints/v3_20260503/hybrid/hybrid_mdrop0.15_best_metrics.txt
```

→ **Paste cho tôi để fill Q10 robustness section.**

### 2.16. Linear probe BiomedCLIP (~15 phút)

```bash
python3 train_linear_probe.py --backbone biomedclip --save-dir checkpoints/v3_20260503/linear_probe 2>&1 | tee experiments/v3_20260503/run_lp_biomedclip.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/linear_probe/
```

```bash
cat checkpoints/v3_20260503/linear_probe/linear_probe_biomedclip_best_metrics.txt
```

→ **Paste cho tôi để fill Bảng JS-1 (BMC-only linear probe column) + Q-JS-3 defense.**

### 2.17. Linear probe ImageNet ViT (~15 phút)

```bash
python3 train_linear_probe.py --backbone imagenet_vit --save-dir checkpoints/v3_20260503/linear_probe 2>&1 | tee experiments/v3_20260503/run_lp_imagenet.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/linear_probe/
```

```bash
cat checkpoints/v3_20260503/linear_probe/linear_probe_imagenet_vit_best_metrics.txt
```

→ **Paste cho tôi để fill Q-JS-2 defense (BMC vs ImageNet pretrained).**

### 2.18. Verify outputs (theo plan SKIP — 4 run MUST)

```bash
ls -la checkpoints/v3_20260503/hybrid/*.pth checkpoints/v3_20260503/linear_probe/*.pth
```

→ Theo plan MUST: phải có **4 file `best_model_*.pth`** (1 hybrid full + 1 hybrid cbam_only + 2 linear probe). Nếu chạy thêm Run 5/6/7 (skip) thì sẽ có thêm.

### 2.19. Tar tất cả results

**Option A (gọn — recommended):** Bỏ `checkpoint_epoch_*.pth` periodic snapshots, chỉ giữ `best_model_*.pth` + metrics + logs:

```bash
tar -czvf v3_results.tar.gz --exclude='checkpoint_epoch_*.pth' --exclude='checkpoint_attention_epoch_*.pth' --exclude='checkpoint_hybrid_epoch_*.pth' experiments/v3_20260503/ checkpoints/v3_20260503/
```

**Option B (đầy đủ):** Bao gồm cả periodic snapshots (~1-2GB):

```bash
tar -czvf v3_results.tar.gz experiments/v3_20260503/ checkpoints/v3_20260503/
```

```bash
ls -lh v3_results.tar.gz
```

→ Option A: tar ~300-500MB. Option B: ~1-2GB. Nếu > 5GB có gì sai.

### Nội dung tar bao gồm:

- `experiments/v3_20260503/run_*.log` — toàn bộ log training (mỗi run 1 file)
- `experiments/v3_20260503/RESULTS_LOG.md` — backup raw cat output (nếu append)
- `checkpoints/v3_20260503/baseline/` — best_model.pth + baseline_best_metrics.{txt,json} + baseline_log.csv
- `checkpoints/v3_20260503/cbam/` — best_model_attention.pth + attention_best_metrics.{txt,json} + attention_log.csv
- `checkpoints/v3_20260503/hybrid/` — best_model_hybrid*.pth + hybrid*_best_metrics.{txt,json} + hybrid*_log.csv
- `checkpoints/v3_20260503/linear_probe/` — best_model_linear_probe_*.pth + linear_probe_*_best_metrics.{txt,json} + linear_probe_*_log.csv

→ **1 tar = đủ hết. SCP 1 lần là done.**

### 2.20. (Optional) SPIDER experiments — chỉ chạy nếu có SPIDER data

Nếu chưa có SPIDER data thì **skip**. Có thể làm sau.

```bash
ls spider/ 2>/dev/null && echo "SPIDER data found" || echo "SPIDER data NOT found, skip"
```

Nếu có SPIDER, chạy:

```bash
python3 train_spider_hybrid.py --rsna-checkpoint checkpoints/v3_20260503/hybrid/best_model_hybrid.pth --freeze-backbone --save-dir checkpoints/v3_20260503/spider --epochs 15 --batch-size 16 --lr 1e-3 2>&1 | tee experiments/v3_20260503/run_spider_phase4.log
```

Verify:

```bash
ls -lh checkpoints/v3_20260503/spider/
```

```bash
cat checkpoints/v3_20260503/spider/spider_hybrid_best_metrics.txt
```

→ **Paste cho tôi để fill Bảng 2 (SPIDER retrain F1 column).**

### 2.21. Detach tmux + thoát SSH

```bash
# Detach tmux: Ctrl+b rồi d (không type "Ctrl+b d" — phải là 2 phím riêng)
exit
```

---

## PART 3 — VỀ LOCAL MACHINE (Mac của bạn)

### 3.1. Pull tar về local (chạy trong terminal local Mac, KHÔNG phải SSH)

```bash
cd ~/Downloads
```

```bash
scp -P 39311 root@220.130.209.122:~/spinet-v2/v3_results.tar.gz ./
```

→ Hoặc nếu Vast clone vào `/workspace`:

```bash
scp -P 39311 root@220.130.209.122:/workspace/spinet-v2/v3_results.tar.gz ./
```

### 3.2. Extract về local repo

```bash
cd ~/spinet-v2
```

```bash
tar -xzvf ~/Downloads/v3_results.tar.gz
```

### 3.3. Verify outputs đã về

```bash
ls -la experiments/v3_20260503/
```

```bash
ls -la checkpoints/v3_20260503/baseline/ checkpoints/v3_20260503/cbam/ checkpoints/v3_20260503/hybrid/ checkpoints/v3_20260503/linear_probe/
```

### 3.4. Cat tất cả best_metrics để paste cho assistant

Chạy lệnh này, copy toàn bộ output paste cho tôi:

```bash
for f in checkpoints/v3_20260503/*/*best_metrics.txt; do echo "=== $f ==="; cat "$f"; echo; done
```

→ Tôi sẽ fill các slot `[TODO]` trong v3 report (Bảng 1A, 1C) với numbers thực tế.

### 3.5. (Sau khi fill xong report) commit v3 results

```bash
cd ~/spinet-v2
```

```bash
git add experiments/v3_20260503/ THESIS_REPORT_DRAFT_v3.md
```

```bash
git commit -m "Add v3 retrain results + fill report TODO slots"
```

```bash
git push origin biomedclip-integration
```

---

## TROUBLESHOOTING

### Nếu SSH disconnect giữa training:

```bash
ssh -p 39311 root@220.130.209.122 -L 8080:localhost:8080
```

```bash
tmux attach -t train
```

→ Training tiếp tục từ chỗ disconnect. Không mất.

### Nếu out of memory (OOM):

Giảm batch size từ 32 → 16 cho hybrid runs:

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --epochs 20 --batch-size 16 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_full.log
```

### Nếu cần resume từ checkpoint sau crash:

```bash
python3 train_rsna_hybrid.py --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth --resume checkpoints/v3_20260503/hybrid/checkpoint_hybrid_epoch_10.pth --epochs 20 --batch-size 32 --lr 1e-4 --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee -a experiments/v3_20260503/run_hybrid_full.log
```

### Nếu muốn check progress đang chạy ở tmux từ SSH khác:

```bash
ssh -p 39311 root@220.130.209.122
```

```bash
tmux ls
```

```bash
tmux attach -t train
```

→ Detach: Ctrl+b rồi d.

---

## TIME ESTIMATE

| Step | Time | Cumulative |
|---|---|---|
| 2.1-2.8 setup + download | ~30 phút | 0.5h |
| 2.9 baseline | 1.5h | 2.0h |
| 2.10 CBAM | 2.0h | 4.0h |
| 2.11-2.15 hybrid x5 | 10h | 14.0h |
| 2.16-2.17 linear probe x2 | 0.5h | 14.5h |
| 2.20 SPIDER (optional) | 3.5h | 18.0h |

→ Đặt máy Vast.ai budget **20h** để có buffer.

→ Có thể launch lúc tối, sáng pull về.

---

## CHECKLIST

Trước khi launch, confirm:

- [ ] Máy Vast.ai GPU >= 16GB VRAM (verify Step 2.4)
- [ ] Disk free >= 20GB
- [ ] Có GDRIVE_RSNA và GDRIVE_SPINENET IDs
- [ ] Branch `biomedclip-integration` commit `63326d6` đã clone (verify Step 2.2)
- [ ] tmux đã bật (Step 2.5)
- [ ] (Optional) SPIDER data có sẵn nếu muốn Phase 3+4

Done = launch sequential từ Step 2.9.
