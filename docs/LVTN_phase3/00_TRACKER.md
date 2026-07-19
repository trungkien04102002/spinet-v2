# LVTN kì cuối — TRACKER tổng (file follow chính)

> Cập nhật 2026-07-19. Đây là file DUY NHẤT để theo dõi toàn bộ luận văn kì cuối:
> trạng thái, vấn đề đang gặp, việc improve, kế hoạch, tiến độ. Các file chi tiết
> nằm cùng thư mục — link ở cuối. Cập nhật cột trạng thái mỗi khi xong một việc.

## 0. Bối cảnh
- **Paper MIWAI: ĐÃ NỘP** (em first author, thầy corresponding). Kì cuối = luận văn.
- **Scope chốt:** (1) software gán nhãn MRI, (2) so sánh SOTA cho grading, (3) improve kết quả (F1) nếu kịp, (4) feedback-loop → **FUTURE WORK**.
- **Timeline:** giao đề 15–17/07 · thực hiện 15 tuần **20/07 → 31/10/2026** · bảo vệ **02–06/11**.
- **HW:** 1× RTX-4090 (24GB). **Đủ pass?** Có — paper đã nộp + method mới + software + SOTA.

## 1. Dashboard trạng thái

| Workstream | Trạng thái | Ghi chú |
|---|---|---|
| **Software** (spine-labeling-app) | 🟢 ~90% | P0–P3 gần xong hết (build trước cả kì). Còn: check P2 viewer đầy đủ, polish UI, quay demo video. Repo public. |
| **SOTA comparison** | 🟡 Code xong, chờ GPU | 2 baseline (brendanartley 2.5D+LSTM, transformer-agg) + run_all.sh + make_table.py đã verify. Chạy 1 lượt `bash experiments/sota_comparison/run_all.sh` trên 4090. |
| **Improve F1** | 🟡 Plan xong, chưa làm | Move #1 (threshold/calibration) gần như free, chưa chạy. Xem §2–§3. |
| **Feedback-loop** | ⚪ Future work | Capture đã xong+tested (thỏa thầy). Không build phần retrain. Optional: dataset-builder ~1 ngày. |
| **Report luận văn** | 🔴 Chưa bắt đầu | Viết sau khi có số SOTA + F1. |
| **Slides + demo video** | 🔴 Chưa | Cuối kì. |
| **Meeting chốt scope với thầy** | 🟡 Chuẩn bị xong | Talking points ở CAU_HOI_THAY.md. |

## 2. Vấn đề đang gặp (chẩn đoán từ metrics 3-seed thật)
- **F1 vẫn thấp ở lớp hiếm.** Cụ thể **foraminal Severe** F1 chỉ 0.27–0.29; bottleneck là **PRECISION** (0.22–0.23) — model **over-call Severe**, recall thật ra ổn.
- **Gap AUC–F1 lớn:** Severe AUC 0.899 nhưng F1 0.356 → phần lớn là **vấn đề threshold/operating-point**, KHÔNG phải thiếu representation. (AUC = xếp hạng tốt; argmax equal-cost = quyết định sai cho lớp hiếm.)
- **RSNA official metric weight lỗi 1:2:4** → argmax mặc định vốn đã sai cho task này.
- **spinal_canal Moderate F1 thấp = seed-instability** (seed 123 recall sập 0.083 vs ~0.4 seed khác), không phải yếu kiến trúc → cần ổn định seed (thêm epoch / early-stop tốt hơn / chọn seed robust).
- **foraminal** có headroom representation thật (AUPRC 0.21–0.23 vs 0.56 của canal).

## 3. Kế hoạch improve F1 (ranked gain ÷ effort)

| # | Việc | Retrain? | Effort | Trạng thái |
|---|---|---|---|---|
| **1** | Per-class **threshold optimization + calibration** trên checkpoint 3-seed sẵn có: coordinate-descent max macro-F1 / match cost 1:2:4; + logit adjustment / Balanced Softmax; + τ-normalization classifier weights | **Không** | Vài giờ | ⬜ chưa |
| 2 | **cRT**: freeze backbone, retrain head với class-balanced sampling (Kang 2020) | head | ~1 ngày | ⬜ |
| 3 | **SSL contrastive pretrain** trên RSNA disc ROI → fine-tune (bằng chứng cùng cohort: Severe recall 73.4%, arXiv:2602.05738) | full | vài ngày | ⬜ |
| 4 | **LDAM-DRW / Balanced-Softmax loss** thay/bổ sung focal | full | 1–3 ngày | ⬜ |
| 5 | **Multi-view + axial T2** (ceiling cao nhất, M-SCAN AUROC 0.971) | full | 1–2 tuần | ⬜ |

- **Làm #1 trước** (free + khớp chẩn đoán). Kết quả cũng là diagnostic: nếu #1 gỡ được nhiều F1 → gần free win; nếu không → chuyển sang #3/#5 (representation).
- Stackable: Balanced-MixUp, seed-ensemble + threshold. **Tránh:** naive TTA, pseudo-label (amplify majority). **Đã thử (đừng lặp):** focal, uncertainty weighting, oversampling, aug, CBAM+BiomedCLIP.
- Mọi số mới → bảng before/after riêng, giữ nguyên protocol 3-seed + frozen split.

## 4. Kế hoạch theo giai đoạn (15 tuần)

| Giai đoạn | Mốc | Việc chính |
|---|---|---|
| **GĐ1** | ~20/07–02/08 (T1–2) | Chạy SOTA trên 4090 (1 lượt); **F1 move #1 threshold** (free win); rà P2 viewer đầy đủ chưa |
| **GĐ2** | ~03–16/08 (T3–4) | F1 #2 cRT + #3 SSL (nếu #1 chưa đủ); chốt bảng SOTA; bắt đầu viết chương Phương pháp + Thực nghiệm |
| **GĐ3** | ~17–30/08 (T5–6) | Khóa số F1; polish UI + quay demo video; tiếp tục report |
| **GĐ4** | ~31/08–20/09 (T7–9) | Viết bulk report (đủ chương), tích hợp SOTA+F1+software; optional dataset-builder feedback (~1 ngày) |
| **GĐ5** | ~21/09–11/10 (T10–12) | Polish report, hình/bảng, vòng review với thầy |
| **GĐ6** | ~12–25/10 (T13–14) | Slides + demo video final + tập bảo vệ |
| **Buffer** | ~26–31/10 (T15) | Dự phòng, nộp cuối |
| **Bảo vệ** | 02–06/11 | |

## 5. Việc tiếp theo (ordered checklist)
- [ ] **F1 #1:** build script sweep threshold/logit-adjustment/τ-norm → chạy trên checkpoint 3-seed → bảng Severe/macro F1 mới (+ metric cost 1:2:4). *(Claude làm được ngay, chỉ cần chạy inference lấy logits val.)*
- [ ] **SOTA:** `git pull` + `bash experiments/sota_comparison/run_all.sh` trên 4090 → điền 2 cột SOTA. *(User bấm 1 lượt.)*
- [ ] Chốt scope với thầy (dùng CAU_HOI_THAY.md).
- [ ] Rà software P2 viewer + quay demo.
- [ ] (tùy) F1 #2/#3 nếu #1 chưa đủ.
- [ ] Viết report → slides → demo.

## 6. Câu hỏi mở cho thầy
Xem `CAU_HOI_THAY.md`. Trọng tâm: SOTA sâu tới đâu (adapt vs full retrain), yêu cầu software tối thiểu, xác nhận feedback→future work, data demo public đủ chưa.

## 7. File chi tiết (cùng thư mục)
- `IMPROVE_F1_PLAN.md` — chẩn đoán F1 + kế hoạch improve đầy đủ (nguồn của §2–§3).
- `FEEDBACK_LOOP_RESEARCH.md` — literature feedback-loop + đoạn future-work mẫu.
- `CAU_HOI_THAY.md` — talking points chốt scope với thầy.
- `IMPLEMENTATION_PLAN.md` — plan software P0–P3 gốc.
- `DESIGN_SPEC.md` — thiết kế software.
- SOTA: `experiments/sota_comparison/README.md` (+ memory `project-sota-comparison`).
