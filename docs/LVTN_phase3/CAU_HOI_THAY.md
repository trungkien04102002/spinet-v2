# Chốt scope với thầy — LVTN kì cuối (meeting talking points)

> Paper MIWAI: **ĐÃ NỘP** (em first author, thầy corresponding). Kì cuối tập trung luận văn.
> File này là bản DUY NHẤT để chuẩn bị họp (đã gộp MEETING_PREP.md cũ vào phần Tham khảo bên dưới).

## Scope đề xuất chốt (4 ý)
1. **Hoàn thiện pipeline + software** gán nhãn MRI hỗ trợ bác sĩ — **ưu tiên 1**.
2. **Thêm so sánh SOTA** cho phần grading (PP của mình vs PP đã công bố) — khác với ablation nội bộ đã có.
3. **Cải thiện thêm kết quả** nếu còn thời gian — ưu tiên 2 (optional).
4. **Feedback bác sĩ → model học lại** → xin **hạ xuống FUTURE WORK** (khó + không kịp 1 kì; thầy đã nói ưu tiên 2 và "làm feedback BS trước, khoan hướng tự động").

## Câu cần thầy xác nhận / trả lời (theo thứ tự ưu tiên)
1. **Scope tổng:** xác nhận **đủ tiêu chí pass** với: paper đã nộp + software + SOTA (+ tune thêm nếu kịp)?
2. **Dữ liệu demo:** software demo trên **data public (RSNA/SPIDER)** đủ chưa, hay thầy muốn có ca lâm sàng thật? (data thật → dính PHI + xin data, sẽ đội thời gian).
3. **Đánh giá software:** demo **chạy được (quay video)** là đủ, hay cần bác sĩ thật dùng thử / đo usability?
4. **Yêu cầu tối thiểu của phần mềm:** xem/overlay seg + highlight bất thường + sửa nhãn/mask + export — đủ chưa? cần thêm gì? (đối chiếu bảng component ở Tham khảo A).
5. **SOTA:** so với PP nào? Cần **chạy lại 1–2 PP trên đúng split RSNA của mình** (fair) hay **bảng định vị từ số đã công bố + caveat** là đủ?
6. **Feedback-loop:** OK hạ xuống future work chứ ạ? (nếu thầy vẫn muốn làm → đề xuất chỉ Phase-2 grading học lại, batch retrain offline).
7. **Kiến trúc:** 2 model chạy **song song, độc lập** (TotalSpineSeg + Phase-2 grading, chưa nối) — thầy OK không?
8. **Tiến độ:** 15 tuần thì thầy muốn review/gặp mỗi mấy tuần? (để em chủ động book lịch).

## Mốc thời gian (chính thức)
- Giao đề: 15–17/07/2026 · Thực hiện: **15 tuần, 20/07 → 31/10/2026** · Bảo vệ: **02–06/11/2026**.

---

## Tham khảo A — Component chuẩn của công cụ gán nhãn ảnh y tế
Để đối chiếu app đang có/thiếu gì (trả lời câu 4). Trạng thái là MỤC TIÊU app hướng tới, không phải đã xong (hiện mới P0/P1.1).

| Nhóm | Component | Kế hoạch |
|---|---|---|
| Danh sách | Worklist Patient→Study, search/filter | Có (P2) |
| Xem | 2D viewer: W/L, zoom, pan, scroll slice (Cornerstone3D) | Có (P2) |
| Xem nâng cao | MPR (axial/sag/coronal), MIP, 3D render | Cân nhắc bỏ |
| Overlay | Segmentation labelmap + legend | Có (P1/P2) |
| Overlay | Abnormality box / heatmap (Grad-CAM) | Có bbox, heatmap để sau |
| Sửa | Brush/erase mask, grade dropdown | Có (P3) |
| Sửa | Undo/redo | Có (P3) |
| Lưu | Versioned + audit log (`correction_log`) | Có (P0 DB) |
| Xuất | Mask (NIfTI), grades (CSV/JSON), ảnh overlay | Có (P3) |
| Chất lượng | Multi-reader / consensus, inter-rater (DICE) | Không kì này |
| Chất lượng | Đo đạc (RECIST, caliper, angle) | Không kì này |

**Khái niệm nên nắm khi thầy hỏi:** DICOM SR (chuẩn lưu annotation — hỏi có cần theo không, hay JSON/CSV nội bộ đủ); MPR/MIP; window/level; ground truth (1 người = "ý kiến", nhiều người + consensus = "ground truth"); inter-rater agreement đo bằng DICE (~0.8–0.9 cấu trúc rõ, 0.6–0.8 ranh giới mờ).

---

## Tham khảo B — Feedback-loop (đã hạ FUTURE WORK, để nói khi thầy hỏi)
"Feedback" = những gì bác sĩ SỬA so với gợi ý AI (đã bắt được qua `correction_log` + annotation versioned). Về ML chủ yếu là **label correction** (sửa mức độ grade) → model học lại hợp lý = **Phase-2 grading**; mask chỉ lưu làm data, không retrain TotalSpineSeg.

**Vòng lặp khả thi (nếu làm):** bác sĩ sửa grade → gom N ca → cắt crop + nhãn severity → batch fine-tune Phase-2 offline → đo before/after trên test giữ riêng → swap model. Đề xuất **batch offline** (tránh catastrophic forgetting, dễ đánh giá) thay vì online.

**Keywords:** human-in-the-loop / active learning medical imaging; learning from label corrections; continual learning + catastrophic forgetting; batch retraining vs online fine-tuning; label noise / inter-observer variability; model drift / versioning.

**Papers nền:** Budd et al. 2021 *Survey on Active Learning and HITL for Medical Image Analysis* (arXiv 1910.02923); *Comprehensive survey on deep active learning in medical image analysis* 2024 (arXiv 2310.14230); *Physician-in-the-Loop Active Learning in Radiology AI Workflows* AJR 2025.

---

## Tham khảo C — Multi-view T1/T2/Axial cho grading (Q&A đã chuẩn bị)

> Thầy nhiều khả năng hỏi quanh: mỗi chuỗi ảnh dùng cho gì, axial khai thác được gì,
> lấy đặc trưng bằng cách nào, ghép ra sao, và có phải train lại nhiều không.
> Trạng thái thật: **T2 đã có đầy đủ; T1 đang thêm (prep xong, chờ GPU train 1 lần);
> Axial CHƯA làm (stretch/future).** Chi tiết đầy đủ: `MULTIVIEW_RESEARCH.md`.

**Q1. Mỗi chuỗi ảnh mang đặc trưng khác nhau, đúng không? Dùng cho grading nào?**
Đúng — chuẩn protocol MRI cột sống, khớp cách top RSNA route chuỗi ảnh:
- **Sag-T2** (dịch/CSF sáng): Pfirrmann, hẹp đĩa, **hẹp ống sống trung tâm (canal)**, thoát vị.
- **Sag-T1** (mỡ sáng): **hẹp lỗ liên hợp (foraminal)** — dấu hiệu là *mất mỡ quanh rễ* → chỉ rõ trên T1; + Modic/marrow, endplate.
- **Axial-T2** (mặt cắt ngang): **hẹp ngách bên (subarticular)**, diện tích/hình dạng mặt cắt ống sống.

**Q2. Axial tận dụng được gì? Chỉ mỗi subarticular à?**
Không chỉ subarticular, nhưng phải phân biệt thông tin **MỚI** vs **xác nhận lại**:
- **Subarticular**: axial là mặt phẳng chính → thông tin MỚI (sagittal không thấy).
- **Canal, herniation**: axial có bổ trợ (diện tích mặt cắt; kiểu/hướng thoát vị) NHƯNG **phần lớn trùng tín hiệu đã có trong Sag-T2** → gain F1 nhỏ.
- **Pfirrmann, hẹp đĩa, spondylo, endplate, marrow**: axial gần như không thêm gì (đây là dấu hiệu mặt phẳng sagittal).
→ Chỗ **duy nhất** axial mang thông tin không dư thừa là **subarticular** — mà subarticular **chưa nằm trong 11 nhãn hiện tại**. Muốn khai thác axial đúng nghĩa ⇒ phải **thêm nhãn subarticular** (mở rộng label space); nếu không, axial chỉ đóng vai **supporter cho canal** trong fusion.

**Q3. Em dùng cái gì / cách nào để LẤY ĐẶC TRƯNG từ T1/T2/axial?**
Mỗi chuỗi có **1 encoder riêng: 3D ResNet34 + CBAM** (channel + spatial attention).
CBAM tự học vùng đặc trưng của từng chuỗi (mỡ lỗ liên hợp trên T1, CSF ống sống trên T2, ngách bên trên axial) → mỗi encoder xuất **1 embedding per-đĩa**.

**Q4. Ghép các đặc trưng đó lại để grading thế nào?**
**Gated leader–supporter fusion (GMU)**: một gate nhỏ học **trọng số từng chuỗi theo từng bệnh**, khởi tạo theo prior lâm sàng (T1↑ foraminal, T2↑ canal, axial↑ subarticular), rồi head cho ra grade. Ablation báo cáo: T2-only → +T1 (concat) → +gated → (+axial). Model concat + gated **đã code** ở `experiments/multiview/`.

**Q5. Có phải train lại nhiều không? (em muốn tiết kiệm thời gian)**
Có 2 mức, chọn theo mức chịu train:
- **Late/decision fusion — 0 training**: chạy model đã có độc lập, route output theo prior (foraminal←T1, canal←T2, subarticular←axial). Chính là cách RSNA hạng 2.
- **Gated fusion, ĐÓNG BĂNG encoder — train cực nhẹ** (khuyến nghị): freeze encoder đã train, chỉ train **gate + head** (vài phút, không đụng backbone). T2 đã có (free); T1 chỉ train **1 lần** (crop đã prep); axial cần encoder mới → **để future**.

**Papers chống lưng:** GMU gated fusion (Arevalo 2017, arXiv:1702.01992); M-SCAN attention fuse sag+axial, AUROC 0.971 (arXiv:2503.01634); weighted task-specific fusion (arXiv:2307.00885); RSNA 2nd-place (late fusion độc lập, axial build rồi bỏ vì không tăng CV); RSNA 3rd-place (concat theo điều kiện + CenterNet keypoints axial→level).

**⚠️ Caveat thành thật (nên chủ động nói):** **không có ablation sạch** chứng minh axial → tăng Severe-F1; giá trị riêng của axial là subarticular (chưa có nhãn). Đòn chắc ăn hơn hẳn là **T1-foraminal**. → Trình axial ở mức ablation/future, không over-claim.
