# Câu hỏi cho thầy — LVTN kì cuối (Phase 3: software + feedback-loop)

> File dồn các quyết định chưa chắc trong lúc lên plan. Hỏi thầy 1 lượt sau khi plan xong.

## 1. Cơ chế feedback bác sĩ → model học lại (phần novelty, đang PARK)
- **Model nào học lại?**
  - Đề xuất của em: **chỉ model grading/bất thường Phase 2 (CBAM+BiomedCLIP)** học lại (model mình kiểm soát, nhỏ, retrain được). Mask segmentation bác sĩ sửa thì **chỉ lưu làm data mới + export**, KHÔNG retrain TotalSpineSeg (model ngoài, nặng).
  - Thầy muốn cả segmentation cũng học lại không?
- **Cơ chế học lại?** retrain định kỳ theo batch (gom feedback rồi train lại) vs incremental/online fine-tune từng ca? (rủi ro catastrophic forgetting nếu online)
- **Có bắt buộc làm trong kì này, hay để future work / demo tối giản?** (đây là phần rủi ro + tốn thời gian nhất)

## 2. Kiến trúc 2 model chạy song song
- Hiện chốt: **TotalSpineSeg (segmentation) + SpineNetV2/Phase 2 (grading) chạy ĐỘC LẬP, song song**, overlay cùng ảnh — KHÔNG nối output seg làm input grading (chưa nối được + thời gian có hạn).
- Thầy OK hướng parallel này, hay muốn nối thành pipeline nối tiếp?

<!-- Thêm câu hỏi mới bên dưới trong quá trình lên plan -->
