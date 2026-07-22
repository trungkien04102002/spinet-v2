# LVTN kì cuối — TRACKER tổng (file follow chính)

> Cập nhật 2026-07-19. Đây là file DUY NHẤT để theo dõi toàn bộ luận văn kì cuối:
> trạng thái, vấn đề đang gặp, việc improve, kế hoạch, tiến độ. Các file chi tiết
> nằm cùng thư mục — link ở cuối. Cập nhật cột trạng thái mỗi khi xong một việc.

## ⭐ MEETING THẦY 2026-07-19 — định hướng mới (ưu tiên đọc)
Thầy cho 7 ý, làm đổi trọng tâm cải tiến model:
1–2. **Multi-view + T1/T2 "leader–supporter" có trọng số** (TRỌNG TÂM MỚI). Y khoa: **T1 sag = leader cho foraminal**; **T2 sag + axial = leader cho ống sống/subarticular**. Mỗi điều kiện có chuỗi leader (trọng số cao) + supporter. → thêm nhánh **sagittal-T1 + axial-T2** vào model. (chi tiết kiến trúc: đang chờ agent research, sẽ điền §3.)
3. **Pipeline** — chuẩn hóa lại sơ đồ đa chuỗi vào → grading ra.
4. **Mất cân bằng + chỉnh thông số + highlight bất thường** → threshold/calibration (move #1) + Grad-CAM heatmap trong app.
5. **"Input/output cũ sai, hướng level bác sĩ"** → bác sĩ đọc cả study đa chuỗi. **QUYẾT: hướng A** — giữ pipeline per-IVD làm 1 nhánh, THÊM nhánh T1/axial + leader-supporter; framing doctor-level trình bày trong report (không rebuild study-level bản B kì này).
6. Improve recall/acc/F1 (xuyên suốt).
7. **Feedback bác sĩ — QUAY LẠI SCOPE, bản ĐƠN GIẢN:** đưa bác sĩ thử **10–20 sample** → cơ chế học đơn giản từ đó → cải thiện. "Đơn giản cũng được" → KHÔNG cần EWC/forgetting nặng; cỡ dataset-builder + fine-tune nhẹ show before/after.

**Đổi so với plan trước:** multi-view lên trọng tâm; feedback-loop back-in bản đơn giản; threshold vẫn làm trước (free). §2–§3 đã cập nhật theo research multi-view (2026-07-19).

**🔴 PHÁT HIỆN GỐC (research multi-view):** pipeline RSNA lọc **Sagittal T2 cho MỌI head** (`rsna_dataloader.py:127-144 is_sagittal_t2`) — kể cả foraminal. Nhưng foraminal PHẢI chấm trên **Sagittal T1** (protocol RSNA + Lee grading: mỡ quanh rễ tk là dấu hiệu T1). → **đang chấm foraminal bằng sai chuỗi ảnh**, đúng chỗ Severe F1 tệ nhất. Fix rẻ nhất = đổi nhánh foraminal sang T1. (Nguồn: RSNA LumbarDISC arXiv:2506.09162; brendanartley/3rd-place đều route foraminal←T1, canal←T2, subarticular←axial.)

## 0. Bối cảnh
- **Paper MIWAI: ĐÃ NỘP** (em first author, thầy corresponding). Kì cuối = luận văn.
- **Scope chốt:** (1) software gán nhãn MRI, (2) so sánh SOTA cho grading, (3) improve kết quả (F1) nếu kịp, (4) feedback-loop → **FUTURE WORK**.
- **Timeline:** giao đề 15–17/07 · thực hiện 15 tuần **20/07 → 31/10/2026** · bảo vệ **02–06/11**.
- **HW:** 1× RTX-4090 (24GB). **Đủ pass?** Có — paper đã nộp + method mới + software + SOTA.

## 1. Dashboard trạng thái

| Workstream | Trạng thái | Ghi chú |
|---|---|---|
| **Software** (spine-labeling-app) | 🟢 ~95% | P0–P3 XONG (FE+BE, các session trước): P2 viewer Cornerstone3D (CornerstoneViewport + Legend + GradeTable + lib/history undo/redo), seg∥grading /infer, sửa+export. Còn: polish UI + quay demo video. Repo public. |
| **SOTA comparison** | 🟢 Code xong, HOÃN chạy | Thầy không nhắc SOTA trong meeting → hạ ưu tiên, làm SAU multi-view (để so với model tốt nhất). Ready: `bash experiments/sota_comparison/run_all.sh` (default seed 42; 1 lệnh, chạy lúc nào cũng được). |
| **Improve F1** | 🟡 Plan xong, chưa làm | Move #1 (threshold/calibration) gần như free, chưa chạy. Xem §2–§3. |
| **Feedback-loop** | ⚪ Future work | Capture đã xong+tested (thỏa thầy). Không build phần retrain. Optional: dataset-builder ~1 ngày. |
| **Report luận văn** | 🔴 Chưa bắt đầu | Viết sau khi có số SOTA + F1. |
| **Slides + demo video** | 🔴 Chưa | Cuối kì. |
| **Meeting chốt scope với thầy** | 🟡 Chuẩn bị xong | Talking points ở CAU_HOI_THAY.md. |

## 2. Vấn đề đang gặp (chẩn đoán từ metrics 3-seed + research)
- **🔴 Sai chuỗi ảnh cho foraminal:** đang chấm foraminal trên Sagittal T2, đúng ra phải T1 (xem block ⭐ trên). Nghi là nguyên nhân gốc của foraminal Severe kém.
- **F1 thấp ở lớp hiếm:** **foraminal Severe** F1 0.27–0.29, bottleneck = **PRECISION** (0.22–0.23) — over-call Severe, recall ổn.
- **Gap AUC–F1 lớn:** Severe AUC 0.899 vs F1 0.356 → cũng có phần **threshold/operating-point** (argmax equal-cost sai cho lớp hiếm; RSNA official weight lỗi 1:2:4).
- **spinal_canal Moderate F1 thấp = seed-instability** (seed 123 recall sập 0.083 vs ~0.4), không phải yếu kiến trúc.

## 3. Kế hoạch model-improvement (staged, đúng ý thầy: T1/T2 leader-supporter + axial)

| # | Việc | Retrain? | Effort | Ghi chú | TT |
|---|---|---|---|---|---|
| **0** | **Threshold/calibration** trên checkpoint sẵn: dump logits val 1 lần → coordinate-descent max macro-F1 / match cost 1:2:4 + logit-adjustment + τ-norm | **Không** | Vài giờ | **✅ XONG (Hybrid seed42, sanity khớp paper 0.528/0.343/72.1%):** macro **+0.019 held-out** (in-split 0.528→0.564, Severe F1 +0.030, cost 0.314→0.285). Fix CANAL Severe (0.50→0.57) nhưng KHÔNG cứu foraminal Severe (0.28→0.29) → foraminal = representation problem → cần #1. Logit-adj đẩy Severe recall 0.46→0.83 (đổi precision). τ-norm vô nghĩa với Hybrid (cosine). CBAM cũng cùng story (+0.039 held-out). Local (BiomedCLIP có cache, không cần 4090). | 🟢 |
| **1** | **T1-foraminal fix (diagnostic):** thêm filter `is_sagittal_t1`, chấm foraminal trên crop T1, so precision/F1 vs T2 | train nhẹ | 1–2 ngày | **Ứng viên win lớn nhất** — cô lập "sai chuỗi" vs "cần fusion" | ⬜ |
| **2** | **Two-branch late fusion** (T2 branch = hybrid hiện có + T1 branch mới, concat → head) | có (model mới) | ~1 tuần | Đúng cái 1/2/9/10th place ship; đã là contribution | ⬜ |
| **3** | **Gated leader-supporter fusion** (GMU per-condition: trọng số học được T1/T2/axial theo từng điều kiện, init theo prior lâm sàng) | có | ~1 tuần | **Chính là ý thầy #1/#2**; ablation concat vs gated vs hard-route | ⬜ |
| **4** | **Axial T2 branch** (stretch): crop axial theo `train_label_coordinates.csv`, đưa vào gated fusion cho canal/subarticular | có | 3–5 ngày | Rủi ro: align axial→level messy; để cuối | ⬜ |

**Thứ tự:** #0 (free) + #1 (rẻ, khả năng win to) trước → nếu #1 xác nhận sequence là gốc → #2 → #3 (ý thầy) → #4 nếu kịp.
**Lưu ý train (từ §câu hỏi seed):** chỉ **model multi-view mới** là train thật (1 seed thử → 3 seed cho số cuối). SOTA/threshold/feedback là 1-lượt/free. Model cũ KHÔNG train lại. SPIDER chỉ có T1+T2 (không axial) → multi-view SPIDER là bản rút gọn, optional.
**Bằng chứng bổ trợ:** M-SCAN cross-attention (AUROC 0.971, arXiv:2503.01634); GMU (Arevalo 2017 arXiv:1702.01992); 2nd-place: axial naive KHÔNG giúp → cần gated (không blend đều). Alternate nếu fusion chưa đủ: SSL contrastive (arXiv:2602.05738, Severe recall 73.4%), LDAM-DRW. **Tránh:** naive TTA, pseudo-label. **Đã thử:** focal, uncertainty, oversampling, aug, CBAM+BiomedCLIP.

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
- [ ] **#0 Threshold (free):** dump logits val 1 lần/checkpoint → sweep threshold/logit-adjustment/τ-norm → bảng F1 mới (+ cost 1:2:4). *(Claude làm được ngay.)*
- [ ] **#1 T1-foraminal fix:** thêm filter `is_sagittal_t1`, prep crop T1 cho foraminal, train nhẹ, so vs T2. *(ứng viên win lớn nhất.)*
- [ ] **#2→#3 multi-view fusion** (late fusion → gated leader-supporter) nếu #1 xác nhận sequence là gốc.
- [ ] **SOTA:** `git pull` + `bash experiments/sota_comparison/run_all.sh` trên 4090. *(User bấm 1 lượt.)*
- [ ] **Feedback đơn giản:** dataset-builder từ correction_log + 1 fine-tune nhẹ (10–20 sample bác sĩ) → before/after.
- [ ] Rà software P2 viewer + quay demo.
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
