# Chốt scope với thầy — LVTN kì cuối (meeting talking points)

> Paper MIWAI: **ĐÃ NỘP** (em first author, thầy corresponding). Kì cuối tập trung luận văn.

## Scope đề xuất chốt (4 ý)
1. **Hoàn thiện pipeline + software** gán nhãn MRI hỗ trợ bác sĩ — **ưu tiên 1**.
2. **Thêm so sánh SOTA** cho phần grading (PP của mình vs PP đã công bố) — khác với ablation nội bộ đã có.
3. **Cải thiện thêm kết quả** nếu còn thời gian — ưu tiên 2 (optional).
4. **Feedback bác sĩ → model học lại** → xin **hạ xuống FUTURE WORK** (khó + không kịp 1 kì; thầy đã nói ưu tiên 2 và "làm feedback BS trước, khoan hướng tự động").

## Câu cần thầy xác nhận / trả lời
1. **SOTA:** thầy muốn so với PP nào? Cần **chạy lại 1–2 PP trên đúng split RSNA của mình** (fair, đúng nghĩa SOTA) hay **bảng định vị từ số đã công bố + caveat** là đủ?
2. **Software requirement:** thầy kỳ vọng phần mềm tối thiểu có gì? (xem/overlay seg + highlight bất thường + sửa nhãn/mask + export — đủ chưa? cần thêm gì?)
3. **Feedback-loop:** OK hạ xuống future work chứ ạ? (nếu thầy vẫn muốn làm → cơ chế: gom batch retrain hay online? model nào học lại — đề xuất chỉ Phase-2 grading)
4. **Kiến trúc:** 2 model chạy **song song, độc lập** (TotalSpineSeg + Phase-2 grading, chưa nối) — thầy OK không?
5. Xác nhận **đủ tiêu chí pass** với scope: paper đã nộp + software + SOTA (+ tune thêm nếu kịp)?

## Mốc thời gian (chính thức)
- Giao đề: 15–17/07/2026 · Thực hiện: **15 tuần, 20/07 → 31/10/2026** · Bảo vệ: **02–06/11/2026**.
