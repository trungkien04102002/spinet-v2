# Chuẩn bị bảo vệ luận văn thạc sĩ

Bảo vệ 02–06/11/2026, trình bày khoảng 20–30 phút (thầy góp ý 19/10). Deck: `presentation_spine.tex` / `.pdf`.
Luận văn đã nộp ngày 18/10 (commit `ef23e38`). Số liệu dưới đây khớp luận văn, là trung bình 3 lần chạy.

File này gom các câu nói khi trình bày và câu trả lời khi hội đồng hỏi. Có câu mới thì thêm vào đúng mục.

---

## 1. Mạch trình bày

Giới thiệu đề tài → **Thách thức** → Mục tiêu → Phạm vi → Cơ sở lý thuyết (CBAM, BiomedCLIP) → Công trình liên quan → Dữ liệu → Chọn mô hình phân vùng → Chọn mô hình chấm nhãn → Pipeline → Kiến trúc Hybrid (hình, rồi công thức) → Xử lý mất cân bằng lớp → Huấn luyện và suy diễn → Kết quả RSNA → Grad-CAM → Zero-shot SPIDER → SPIDER có tinh chỉnh → Phần mềm → Giao diện → **Video demo** → Đóng góp → Hạn chế → Công bố → Hướng phát triển.

- 4 thách thức ở đầu, mỗi cái có phần kết quả trả lời ở sau. Khi tới kết quả, nói "đây là thách thức 1/2/3…".
  - Thách thức 1 (mất cân bằng) → Kết quả RSNA.
  - Thách thức 2 (nhãn cố định) → Zero-shot SPIDER.
  - Thách thức 3 (tổng quát hóa) → SPIDER có tinh chỉnh.
  - Thách thức 4 (khoảng cách tới người dùng) → Phần mềm và demo.
- Slide Thách thức đứng trước slide Bộ dữ liệu, nên khi nói thêm câu: "trên dữ liệu RSNA 2024, lớp Severe chỉ chiếm khoảng 4%".
- Slide mục lục giữa các phần dùng làm mốc để chuẩn bị chuyển phần.

## 2. Câu nói cho từng slide

### Kiến trúc Hybrid: slide công thức

> "Sau hai nhánh, em có hai vector đặc trưng: f_cbam từ nhánh CBAM và f_bmc từ nhánh BiomedCLIP. Em ghép hai vector này lại, đưa qua một MLP hai lớp để ra vector ảnh chung f_img, rồi chuẩn hóa độ dài về 1. Đây là phần duy nhất được huấn luyện, hai nhánh giữ cố định.
>
> Bước tiếp theo, thay vì một lớp phân loại thông thường, em so vector ảnh với vector câu mô tả của từng mức độ, ví dụ 'hẹp ống sống mức nặng'. Mức nào có độ tương đồng cô-sin cao nhất thì được chọn. Nhờ vậy, muốn chấm nhãn mới chỉ cần thêm câu mô tả, không phải huấn luyện lại. Đó là cơ chế zero-shot."

Thứ tự đúng: **ghép → MLP → vector chung → chuẩn hóa → so cô-sin với câu mô tả → softmax → chọn mức cao nhất.**

### Grad-CAM

Gọi đây là **một ca minh họa** cơ chế chú ý, không nói là chứng minh.

### Video demo

Mở sẵn video trong QuickTime trước buổi bảo vệ, tới slide Demo thì Cmd+Tab sang phát. Link YouTube trên slide chỉ để dự phòng.

## 3. Câu hỏi có thể gặp

### Về công thức

- **f_bmc lấy từ đâu?** BiomedCLIP mã hóa từng lát ảnh (3 lát giữa khối), rồi gộp các lát bằng trọng số chú ý: lát nhìn rõ tổn thương hơn được tính nặng hơn (luận văn mục 4.3).
- **s là gì?** Hệ số nhiệt độ, giãn điểm cô-sin trước khi đưa qua softmax.
- **φ là gì?** Hàm kích hoạt GELU giữa hai lớp của MLP. MLP đi 1024 → 768 → 512.
- **t_{k,c} là gì?** Vector câu mô tả của mức độ c thuộc nhãn k, do bộ mã hóa văn bản của BiomedCLIP tạo ra, đã chuẩn hóa độ dài.

### Về xử lý mất cân bằng (slide không ghi thông số)

- Focal Loss γ = 2.
- Trọng số lớp theo nghịch đảo căn bậc hai số mẫu.
- Oversampling: lặp lại 3 lần các mẫu có nhãn Moderate/Severe.
- Augmentation: lật trước–sau, xoay ±10°, độ sáng/tương phản 0.2, nhiễu Gaussian 0.05.
- Uncertainty weighting (Kendall 2018) cân bằng 3 nhóm bệnh.

### Về kết quả

- **Grad-CAM một ca thì chứng minh được gì?** "Đây là ví dụ minh họa trực quan; bằng chứng định lượng là Severe Recall tăng từ 0.123 lên 0.486 ở bảng kết quả."
- **Đợt Thực tập 2 em báo Severe Recall 46.4%, sao giờ là 0.486?** "Đợt trước là kết quả một lần chạy; luận văn lấy trung bình ba lần chạy để chắc chắn hơn, và số này khớp với bài báo đã đăng."
- **Accuracy giảm (0.810 → 0.724) thì sao?** Mô hình nền đạt accuracy cao nhờ đoán lớp đa số. Trong lâm sàng, bỏ sót ca nặng nguy hiểm hơn báo nhầm, nên luận văn ưu tiên Recall và F1 của lớp Severe.
- **Severe Recall 0.486 vẫn sót một nửa ca nặng?** Đúng, nên hệ thống chỉ dùng như người đọc thứ hai, bác sĩ vẫn là người chốt.
- **Trên SPIDER, Hybrid 0.653 so với SpineNetV2 0.646 là gần như ngang?** Đúng, thông điệp là Hybrid giữ được mức mô hình nền khi đổi dữ liệu, còn CBAM tụt xuống 0.619. Riêng thoát vị đĩa đệm, SpineNetV2 và CBAM bỏ sót mọi ca, Hybrid đạt Recall 0.636.
- **Zero-shot F1 0.362 có thấp không?** Thấp so với có huấn luyện, nhưng mô hình chấm được 8 nhãn của bộ tiêu chí khác, bệnh viện khác, mà không huấn luyện lại. Luận văn không xem đây là bộ phân loại hoàn chỉnh cho SPIDER.
- **Sao bảng ống sống không có cột CBAM?** Gộp cả 3 nhãn thì Hybrid vẫn hơn; lợi thế của Hybrid nằm ở hai nhãn lỗ liên hợp.
- **Ba lần chạy có đủ ý nghĩa thống kê không?** Chưa đủ để kết luận chặt chẽ, luận văn đã nêu trong phần hạn chế.

### Về chọn mô hình

- **Ning Shen tốt hơn sao không dùng?** F1 cao hơn khoảng 3 điểm nhưng là tổ hợp nhiều mô hình lớn (Faster R-CNN phát hiện đĩa đệm + Swin Transformer phân loại, mỗi nhóm bệnh một mô hình), khó duy trì và khó tích hợp. Mã nguồn công khai trên GitHub, chưa công bố thành bài báo. Em chọn làm mốc so sánh vì có mã chạy lại được.
- **Phân vùng chỉ chọn TotalSpineSeg, không cải tiến?** Đúng, đóng góp nằm ở mô-đun chấm nhãn; phân vùng là bước nền để định vị đĩa đệm.

### Về bài báo

- Nói chắc chắn được: "Bài báo được xuất bản trong bộ Lecture Notes in Artificial Intelligence của Springer, được Scopus và EI index; bộ LNCS xếp Q2 trên SCImago ở nhóm Computer Science. Năm 2026 hội nghị nhận 75/200 bài."
- Chỉ nói "hội nghị rank C" khi đã tự kiểm tra MIWAI trên portal.core.edu.au (chưa xác minh được).

### Về phần mềm

- Phần mềm mới lưu chỉnh sửa của bác sĩ, mô hình chưa học lại được từ đó (hướng phát triển 2).
- Chưa có đánh giá chính thức của bác sĩ trên ca thật (hướng phát triển 1).
