# THESIS_MASTER — điểm vào DUY NHẤT cho luận văn (LVTN kì cuối)

> Cập nhật **2026-08-22**. Đây là file canonical: mọi session sau **đọc file này TRƯỚC**,
> rồi mới mở file chi tiết được link bên dưới. Không đọc transcript session (.jsonl).
>
> **Trạng thái mới nhất (22/08, cuối ngày):** trụ 2 (SOTA) **XONG**. Trụ 3: đợt GPU 22/08
> đã chạy xong 4 run, **box Vast đã huỷ** — mọi artifact cần thiết đã kéo về máy.
> Kết quả: `--fusion gated` **THUA** `concat_mlp` (0.2954 vs 0.3212); cấu hình thắng là (a).
> Phát hiện mới quan trọng nhất: lỗi là **ordinal** — Severe recall 13.8% trong khi Moderate
> recall 75%, tức ca nặng bị hút vào Moderate. Số từng epoch + hướng đi tiếp (A→E→C→D, B cuối)
> ghi ở `experiments/f1_improvement/RESULTS_LOG.md`.
> **đọc file đó trước khi tin bảng tóm tắt bên dưới**. Paper MIWAI camera-ready đã nộp 13–14/08.
>
> Repo: `~/spinet-v2` nhánh `biomedclip-integration` @ `6d47d46` · `~/spine-labeling-app` nhánh `main` @ `5e3c5d6`.
> Timeline: giao đề 15–17/07 · thực hiện **15 tuần 20/07 → 31/10/2026** · **bảo vệ 02–06/11/2026**.
> HW: 1× RTX-4090 24GB (Vast). Đề tài: *"Ứng dụng AI trong gán nhãn ảnh y tế cho bệnh đau thắt lưng"*
> (EN: "AI-Assisted Medical Image Annotation for Low Back Pain").

---

## (a) Scope + 4 trụ

Chốt với thầy (2026-07-06, tinh chỉnh sau meeting 2026-07-19):

| # | Trụ | Vai trò | Trạng thái tổng |
|---|---|---|---|
| 1 | **Software** gán nhãn MRI hỗ trợ bác sĩ (`spine-labeling-app`) | Ưu tiên 1 — deliverable chính | 🟢 ~95% |
| 2 | **So sánh SOTA** cho grading (PP mình vs PP đã công bố, cùng split RSNA) | Bắt buộc — khác ablation nội bộ | 🟢 **XONG 25/07** — 2 model × 3 seed |
| 3 | **Improve F1** (threshold, fusion, ordinal loss) | Ưu tiên 2, nhưng thầy đẩy lên trọng tâm 19/07 | 🟡 đợt GPU 22/08 XONG · #0 threshold = gain duy nhất · gated THUA · **hướng tiếp: A→E→C→D** (xem RESULTS_LOG) · axial khuyến nghị BỎ |
| 4 | **Feedback bác sĩ → model học lại** | **FUTURE WORK** (capture đã build, không build retrain) | ⚪ write-up only |

Bối cảnh: **paper MIWAI ĐÃ NỘP** (em first author, thầy corresponding) → kì cuối = luận văn.
Đánh giá tự thân: **đủ pass** (paper đã nộp + method mới + software + SOTA); trụ 4 không cần để pass.

**Định hướng thầy 2026-07-19 (đổi trọng tâm):** multi-view + T1/T2 "leader–supporter" có trọng số
lên làm trọng tâm cải tiến; feedback quay lại scope nhưng **bản đơn giản** (10–20 sample, không EWC nặng);
threshold/calibration vẫn làm trước vì free.

---

## (b) Trạng thái từng phần (done / doing / todo)

### Trụ 1 — Software `~/spine-labeling-app` (🟢 ~95%)
| Hạng mục | TT | Commit |
|---|---|---|
| P0 backend FastAPI + MySQL + schema | ✅ | — |
| P1 inference: TotalSpineSeg (venv riêng) ∥ CBAM grading, `/infer` | ✅ | — |
| P2 viewer Cornerstone3D (`CornerstoneViewport`/`Legend`/`GradeTable`/`lib/history` undo-redo) | ✅ | — |
| P3 sửa nhãn + mask, versioned annotation + `correction_log`, export CSV/JSON/NIfTI | ✅ | — |
| Full 11-nhãn grading (SpineNet gốc vendored + CBAM cho canal/foraminal) | ✅ | app `690dcb6` |
| Fix "Full grading" E2E (TotalSpineSeg PATH + `pixel_spacing` scalar), verified 479s CPU | ✅ | app `dea54b1` |
| Unify UI: 1 nút `✨ Run AI` + 1 bảng 11 nhãn editable + About + Doctor-feedback section | ✅ | app `cfdc7a2` |
| Upload `.zip` DICOM series (`extract_dicom_zip`, zip-slip guard) | ✅ | app `3c88178` |
| **Async job + progress cho Run AI** (`POST /grade_full/start` + `GET /jobs/{id}`) | ✅ | app `018c1df` |
| **Còn lại** | polish UI (dropdown per-condition) + **quay demo video** | ⬜ |

Test: backend **66 passed / 1 skipped**; FE lint+build clean. Repo **public**.
Chạy: `cd ~/spine-labeling-app && ./run.sh both` (BE :8000, FE :5173) · stop `./run.sh stop`.
Cần `backend/.env` (gitignored): `TOTALSPINESEG_BIN=/Users/kienha/totalspineseg/venv/bin/totalspineseg`, `SEG_DEVICE=cpu`.

⚠️ **Nói ĐÚNG trong report:** app dùng **SpineNet gốc cho 8 nhãn** + **fine-tune CBAM cho canal/foraminal L/R**.
KHÔNG nói "app chạy hoàn toàn bằng model fine-tune".

### Trụ 2 — SOTA comparison (🟢 XONG 25/07)

**Đã chạy thật: 2 model × 3 seed {42,123,456}, cùng split, `eval_samples=1942` khớp cả 2 bên.**
Bảng: `experiments/sota_comparison/comparison_table.{md,tex}` (commit `71b7700`).

| Metric | SpineNetV2 | brendanartley | transformer | **Hybrid (ta)** |
|---|---|---|---|---|
| Mean Accuracy | 81.0 | **83.6** | 83.4 | 72.4 |
| Mean F1 macro | 0.420 | **0.532** | 0.527 | 0.527 ±0.027 |
| Severe F1 | 0.152 | 0.271 | 0.257 | **0.356** |
| Severe Recall | 12.3 | 26.0 | 25.9 | **48.6** |

⚠️ **Phải nói rõ khi đưa vào báo cáo:** baseline ngoài train bằng **CrossEntropy trần, không class
weight, không oversample, không augmentation, from scratch**; Hybrid của mình có đủ bộ chống mất cân
bằng + warm-start. Khoảng cách Severe là *công thức của mình vs kiến trúc của họ*, không phải
kiến trúc vs kiến trúc. Ta **thua sát nút Mean F1** (0.527 vs 0.532, trong 1 std) và **thua đậm
Mean Accuracy** — chuẩn bị sẵn câu trả lời cho hội đồng.

<details><summary>Ghi chú cũ (trước khi chạy)</summary>
- 2 baseline **đã code + smoke-test pass**, drop-in vào harness sẵn có (cùng split, cùng metric JSON schema):
  `experiments/sota_comparison/models/grading_brendanartley.py` (2.5D CNN+BiLSTM+attention-pool, 13M)
  và `models/grading_transformer.py` (ResNet18/lát → TransformerEncoder, 17.5M).
- Split cố định đã export: `experiments/sota_comparison/splits/seed{42,123,456}_*`.
- **Chạy 1 lệnh:** `bash experiments/sota_comparison/run_all.sh` (~2–4h, 6 run) → `make_table.py` tự sinh
  **bảng SỐ** + **bảng NĂNG LỰC định tính**. Commit `633be1f`.
- Hoãn chạy tới sau multi-view (để so với model tốt nhất của mình).
- Fairness đã chốt TRƯỚC khi chạy: **nếu SOTA thắng F1 thô cũng OK** — novelty là label-space flexibility.
</details>

### Trụ 3 — Improve F1 (🟡)

> ⭐ **Nhật ký chạy GPU chi tiết: [`../../experiments/f1_improvement/RESULTS_LOG.md`](../../experiments/f1_improvement/RESULTS_LOG.md)**
> — box specs, lệnh chạy, config đã verify, bảng Severe F1 từng epoch, phân tích, lệnh `scp` kéo
> artifact. **Đọc file đó để biết số mới nhất**; mục dưới đây chỉ là tóm tắt.
| # | Việc | TT | Kết quả / commit |
|---|---|---|---|
| **0** | Threshold / calibration / logit-adjustment / τ-norm (không train) | ✅ | Hybrid seed42: macro **+0.019 held-out** (0.528→0.564 in-split), cost 0.314→0.285. **Fix CANAL Severe (0.50→0.57), KHÔNG cứu foraminal Severe (0.28→0.29)** → foraminal là vấn đề **representation** → cần #1. CBAM: +0.039. `ca2137a`/`d5f6a8e`/`bc9472a` |
| **1** | T1-foraminal (chấm foraminal trên Sag-T1 thay vì T2) | 🔵 **ĐANG CHẠY GPU 22/08** | Seed 42, 25 epoch, ~203s/epoch. **Epoch 13: Severe F1 0.3010** (vượt khoảng T2 0.27–0.29), precision 0.19–0.22. ⚠️ foraminal-only, **KHÔNG so được với Mean F1 0.527**. Biến động epoch lớn (ep9 tụt 0.164) → **cần 3 seed mới dám khẳng định** |
| **2** | Two-branch late fusion (T2 + T1, concat) | 🟡 smoke-test GPU xong, **chờ chạy thật** | Quan trọng hơn #1: #1 *thay* T2 bằng T1, #2 *giữ cả hai* và ra đủ 3 điều kiện → Mean F1 so được với số cũ |
| **3** | Gated leader–supporter fusion (GMU per-condition) — **đúng ý thầy** | 🟡 smoke-test GPU xong, **chờ chạy thật** | ⚠️ Gate là **tổ hợp lồi** (512-dim) vs concat (1024-dim) → **capacity thấp hơn, thua concat về F1 là bình thường**. Giá trị = interpretability (`get_gate_weights()`) |
| **4** | Axial-T2 branch | ⬜ stretch/future | — |

**🔴 Phát hiện gốc:** `rsna_dataloader.py:127-144` (`is_sagittal_t2`) lọc **Sag-T2 cho MỌI head**, kể cả foraminal —
trong khi foraminal **phải** chấm trên **Sag-T1** (mỡ quanh rễ thần kinh). Đang chấm sai chuỗi ảnh đúng chỗ F1 tệ nhất.

**Chẩn đoán F1 (3-seed):** foraminal Severe F1 0.27–0.29, bottleneck = **PRECISION** (0.22–0.23), recall ổn →
over-call Severe. Gap AUC–F1 lớn (Severe AUC 0.899 vs F1 0.356). spinal_canal Moderate thấp = **seed-instability**
(seed 123 recall sập 0.083), không phải yếu kiến trúc.

### Trụ 4 — Feedback loop (🟡 ĐÃ QUAY LẠI SCOPE 2026-07-26 — user sẽ làm)
- **⭐ Đọc [`FEEDBACK_LOOP_METHODS.md`](FEEDBACK_LOOP_METHODS.md) — khảo sát phương pháp + phần đã code xong.**
- **Đã có:** capture (`correction_log`, annotation versioned) + `backend/app/feedback/` (build_dataset + retrain_head).
- **Đã vá 2026-07-26** (3 guard, app test 91 pass/1 skip): đóng băng BatchNorm · replay buffer `--replay-dir` ·
  cổng keep/discard đổi từ accuracy sang **macro-F1** (accuracy che được việc lớp Severe sập về 0).
- **Held-out ĐÃ ĐÓNG BĂNG 26/07** (app `53080da`): `backend/data/holdout_frozen/`, 200 dòng, sha256 `62ab6df1…`,
  bốc **chỉ từ 395 bệnh nhân validation seed-42** (`--split-seed 42`) — nếu bốc từ cả 9748 dòng thì ~80% là data
  model đã học ⇒ đo trí nhớ chứ không phải cải thiện. MANIFEST.json đã vào git làm bằng chứng.
  ⚠️ Bộ này cân bằng lớp cố ý → **đừng so số với paper**, chỉ so trước/sau của chính nó.
- **Còn lại:** nút Áp dụng/Hoàn tác đã có API (`/model/activate`, `/model/revert`) — chưa gắn nút lên UI.
- **Không build (→ Future Work):** EWC/LwF/LoRA, Tip-Adapter cache, cleanlab, McNemar/bootstrap/BWT chính quy, drift detection.
- **Phạm vi user chốt:** "có cách tiếp cận, at least là có làm, và có impact, còn hơn không có gì" — không đánh giá nặng.
  ⚠️ Với 20–50 mẫu **không chứng minh được cải thiện có ý nghĩa thống kê** → phát biểu phải là "cơ chế an toàn, không quên",
  không phải "model tốt lên". Dự phòng nếu không có bác sĩ thật: **correction mô phỏng** từ held-out.

---

## (c) Report / paper — file nào là bản chính

| Deliverable | Bản chính | Trạng thái |
|---|---|---|
| **Paper hội nghị MIWAI 2026** | `paper/lncs_hk252/main_v3.tex` → `main_v3.pdf` (**12 trang, đúng hạn mức**) | ✅ **ĐÃ NỘP** (EasyChair `miwai2026`). Số liệu = **3-seed**. Build: `pdflatex main_v3 → bibtex → pdflatex ×2`, `pdfinfo` PHẢI = 12 trang. ⚠️ KHÔNG sửa `main_pair.tex` (bản thầy) / `main.tex` (cũ) |
| **Báo cáo thực tập 2** | `paper/internship2_new/` → `main.pdf` → nộp dưới tên `Internship2_report_2470723.pdf` | ✅ đã nộp. Số liệu = **single-seed 42** |
| **LUẬN VĂN (LVTN)** | `paper/lvtn_overleaf/` (`main.tex` + `chapter/chap01–07.tex`, 1452 dòng) | 🔴 **BẢN CŨ 2026-05-20, chưa viết cho kì cuối.** Đây là file phải viết tiếp |
| Bản nháp/phái sinh (đừng nhầm) | `paper/REPORT_PAPER*.tex`, `REPORT_DETAILED_VI*.tex`, `summary_v3*.tex` | tham chiếu/số liệu, không phải bản nộp |

**Số liệu — 3 bộ, TUYỆT ĐỐI không trộn:**
- **single-seed (42)** = báo cáo thực tập + slide: Mean F1 `0.420→0.528`, Severe Recall `11.5→46.4%`, Severe F1 `0.149→0.343`. Canonical: `paper/summary_v3_4config.tex`.
- **3-seed {42,123,456}** = paper MIWAI: Mean F1 `0.420→0.527`, Severe Recall `12.3→48.6`, Severe F1 `0.152→0.356`. Canonical: `experiments/v3_rsna_multiseed/SOURCE_OF_TRUTH.md`.
- SPIDER 3-seed dùng chung: Hybrid F1 `0.653`, CBAM `0.619`, Base `0.646` (p=0.011 vs CBAM; p=0.57 parity vs SpineNetV2).

**🔴 LỖI CHƯA SỬA trong LVTN:** `paper/lvtn_overleaf/` chap05 + abstract ghi Mean F1 `0.343 → 0.474` —
**không khớp nguồn nào** (0.343 là Severe F1 của Hybrid bị chép nhầm). Severe Recall CBAM/BMC (0.428/0.296)
và Mean AUPRC cũng sai. **Phải sửa khi viết lại** — hướng đã nghiêng: khớp bộ single-seed của báo cáo thực tập.

**Còn phải viết cho LVTN kì cuối:**
1. Chương Phương pháp: thêm **multi-view T1/T2 leader–supporter** (kiến trúc + lý do lâm sàng).
2. Chương Thực nghiệm: **bảng SOTA** + **bảng ablation F1** (T2-only → +T1 → +gated → +axial) + threshold/calibration.
3. Chương Phần mềm: kiến trúc app, 11 nhãn (SpineNet 8 + fine-tune 3), luồng sửa nhãn → correction_log, export.
4. Kết luận: Future Work = feedback loop (đoạn mẫu sẵn) + axial.
5. Sửa lỗi số liệu ở (b) trên + áp `paper/QUY_TAC_VIET_BAO_CAO_15.md` (15 quy tắc viết của thầy).

---

## (d) Slide

- **Deck hiện có:** `Master_Thesis_Presentation_Kien/presentation_spine.tex` → `.pdf` (**47 trang**, ~28 slide nội dung + 7 phụ lục).
  Build: `pdflatex → biber presentation_spine → pdflatex ×2`.
- **Đây là deck bảo vệ THỰC TẬP 2 (15 phút, single-seed).** Cho bảo vệ LVTN 02–06/11 phải làm bản mới/mở rộng.
- Kèm theo: `SCRIPT_THUYET_TRINH_15P.md` (kịch bản nói), `QA_HOI_DONG_DEFENSE.md` (20+ mục Q&A),
  `DEFENSE_METRICS_VI.md` (giải thích metric + khái niệm).
- ⚠️ Slide có frame zero-shot per-label **stale** (Disc bulging 0.709, Pfirrmann AUC 0.492 = dưới mức ngẫu nhiên = bug);
  số đúng theo log = 0.613 / 0.687 / Mean 0.362. Sửa nếu trình cạnh paper.
- Trạng thái: 🔴 **chưa bắt đầu deck LVTN** — làm ở GĐ6 (~12–25/10).

---

## (e) Kế hoạch còn lại + ưu tiên

**P0 — ĐÃ XONG 22/08.** Đợt GPU đã chạy #1, #2, (a) concat_mlp rebase, (b) gated. Box đã huỷ.
Kết quả và lệnh đầy đủ ở `experiments/f1_improvement/RESULTS_LOG.md`.
⚠️ **Nhánh multi-view (`experiments/multiview/`) là NGÕ CỤT** — nó dùng CBAM cho cả hai nhánh,
không có BiomedCLIP/text anchor/cosine head, tức là THAY THẾ Hybrid bằng model yếu hơn.
Đừng chạy lại `train_multiview.py`.

**P1 — hướng tiếp theo, xếp theo giá trị/GPU-hour** (chi tiết + dẫn chứng ở RESULTS_LOG):

| | Việc | Chi phí |
|---|---|---|
| **A** | In ma trận cosine giữa 3 text anchor của BiomedCLIP | ~2 phút, **chạy CPU trên Mac được** |
| **E** | `python3 train_rsna_hybrid.py ... --slice-strategy dynamic` | 0 dòng code, 1 run |
| **C** | Class Distance Weighted CE trong `spinenet/losses.py` (GIỮ head cosine) | vài dòng + 1 run |
| **D** | OGM-GE (arXiv:2203.15332) chống sụp nhánh | vừa |
| **B** | threshold tuning trên cấu hình thắng — làm **CUỐI**, vì nó cộng dồn | 0 train |

**Không làm axial** (gợi ý #1 của thầy): lời giải giải nhì RSNA 2024 chỉ dùng sagittal,
biến thể axial của họ chỉ +0.01-0.02 CV và không giúp ensemble; định tuyến chuỗi ảnh của
mình đã trùng với họ.

**Số cuối cho luận văn:** cấu hình thắng phải chạy 20 epoch × 3 seed cho **cả nó lẫn đối
thủ** = 6 run, ~5 giờ, 0 dòng code.

**P2 — Viết report LVTN** (chương Phương pháp + Thực nghiệm trước, cuốn chiếu theo số về).

**P3 — Polish app + quay demo video** (dropdown per-condition, async progress, video ~5 phút).

**P4 — Slide LVTN + tập bảo vệ.**

**Lịch 15 tuần (từ 00_TRACKER §4):**

| GĐ | Mốc | Việc chính |
|---|---|---|
| GĐ1 | 20/07–02/08 | GPU batch (F1 #1/#2/#3 + SOTA); rà viewer |
| GĐ2 | 03–16/08 | Chốt bảng SOTA; bắt đầu chương Phương pháp + Thực nghiệm |
| GĐ3 | 17–30/08 | Khóa số F1; polish UI + quay demo |
| GĐ4 | 31/08–20/09 | Viết bulk report; optional dataset-builder feedback |
| GĐ5 | 21/09–11/10 | Polish report, hình/bảng, review với thầy |
| GĐ6 | 12–25/10 | Slide + demo video final + tập bảo vệ |
| Buffer | 26–31/10 | Dự phòng, nộp |
| Bảo vệ | 02–06/11 | |

---

## (f) Multi-view T1/T2/axial + Feedback — tóm tắt

### Multi-view (chi tiết: [`MULTIVIEW_RESEARCH.md`](MULTIVIEW_RESEARCH.md), Q&A thầy: [`CAU_HOI_THAY.md`](CAU_HOI_THAY.md) Tham khảo C)
- **Route chuỗi ảnh theo điều kiện** (chuẩn protocol + gần như mọi lời giải top RSNA):
  canal ← **Sag-T2** · foraminal ← **Sag-T1** (mỡ quanh rễ) · subarticular ← **Axial-T2**.
- **Kiến trúc:** mỗi chuỗi 1 encoder riêng (3D ResNet34 + CBAM) → embedding per-đĩa →
  **gated leader–supporter (GMU)**: gate học trọng số từng chuỗi **theo từng bệnh**, init theo prior lâm sàng.
- **Tiết kiệm train:** đóng băng encoder, chỉ train **gate + head** (vài phút). T2 đã có; T1 train **1 lần**; axial để future.
- **Ablation báo cáo:** T2-only → +T1 (concat) → +gated → (+axial).
- ⚠️ **Caveat trung thực (nên chủ động nói):** không có ablation sạch chứng minh axial → tăng Severe-F1;
  giá trị riêng của axial là **subarticular** — nhãn này **chưa có** trong 11 nhãn hiện tại. RSNA hạng 2 build axial rồi **bỏ**.
  Đòn chắc ăn hơn hẳn là **T1-foraminal**.
- Papers: GMU (arXiv:1702.01992) · M-SCAN cross-attention AUROC 0.971 (arXiv:2503.01634) ·
  weighted task-specific fusion (arXiv:2307.00885) · RSNA LumbarDISC (arXiv:2506.09162) · Lee grading AJR 2010.

### Feedback loop (chi tiết: [`FEEDBACK_LOOP_RESEARCH.md`](FEEDBACK_LOOP_RESEARCH.md))
- Vòng lặp khả thi nhất: correction_log → lọc nhiễu (Cleanlab) → build fine-tune set →
  **batch offline**, freeze backbone + frozen BN, head-only, LR nhỏ, + EWC → eval before/after trên
  **held-out đóng băng** (per-class F1/recall + weighted kappa) **+ forgetting check** → swap nếu cả 2 đạt.
  Trigger = đếm số correction (50–100), không dùng perf-drop (không có ground truth live).
- Papers: Budd 2021 HITL survey (arXiv:1910.02923) · MONAI Label (arXiv:2203.12362) · EWC Kirkpatrick PNAS 2017 ·
  LwF Li & Hoiem · Confident Learning Northcutt JAIR 2021 · cold-start COLosSAL (arXiv:2307.12004) · VIOLA-AI (arXiv:2505.09380).
- ⚠️ Nhiều citation lấy từ abstract/snippet — **verify tác giả/năm/venue trước khi trích trong luận văn**.
  Pfirrmann inter-rater kappa ≈ 0.5–0.8 cần pin nguồn cụ thể.

---

## File chi tiết (đọc khi cần, không đọc mặc định)

| File | Nội dung |
|---|---|
| [`00_TRACKER.md`](00_TRACKER.md) | Tracker chi tiết theo tuần + dashboard (nguồn của mục b/e) |
| ⭐ [`../../experiments/f1_improvement/RESULTS_LOG.md`](../../experiments/f1_improvement/RESULTS_LOG.md) | **KẾT QUẢ CHẠY GPU thật (22/08+)** — số mới nhất của #1/#2/#3, từng epoch, kèm phân tích. Đọc file này để biết trụ 3 đang ở đâu |
| [`BAO_CAO_THAY_20260822.md`](BAO_CAO_THAY_20260822.md) | Báo cáo cho thầy: 4 gợi ý (axial, T1/T2 leader-supporter, pipeline, mất cân bằng) + khảo sát tài liệu đã verify |
| [`IMPROVE_F1_PLAN.md`](IMPROVE_F1_PLAN.md) | Chẩn đoán F1 + xếp hạng 5 hướng cải thiện |
| [`MULTIVIEW_RESEARCH.md`](MULTIVIEW_RESEARCH.md) | Research multi-view đầy đủ + kiến trúc fusion |
| [`FEEDBACK_LOOP_METHODS.md`](FEEDBACK_LOOP_METHODS.md) | **Khảo sát phương pháp feedback-loop (2026-07-26) + phần đã code + việc còn lại** |
| [`FEEDBACK_LOOP_RESEARCH.md`](FEEDBACK_LOOP_RESEARCH.md) | Literature feedback-loop (bản cũ) + đoạn Future Work mẫu |
| [`CAU_HOI_THAY.md`](CAU_HOI_THAY.md) | Talking points + Q&A đã chuẩn bị cho thầy |
| [`DESIGN_SPEC.md`](DESIGN_SPEC.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Thiết kế + plan P0–P3 của app |
| `../handoff/handoff-20260724-codex.md` | Handoff gần nhất (spinet-v2) |
| `~/spine-labeling-app/docs/handoff-20260724-app-session.md` | Handoff gần nhất (app) |
| `../../paper/QUY_TAC_VIET_BAO_CAO_15.md` | 15 quy tắc viết báo cáo của thầy |
| `experiments/sota_comparison/README.md` | Cách chạy SOTA |
