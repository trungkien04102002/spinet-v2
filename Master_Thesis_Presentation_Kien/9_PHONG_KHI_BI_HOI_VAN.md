# Ghi chú phòng khi bị hỏi vặn (không cần đọc)

Tách ra khỏi file hỏi–đáp chính để đỡ rối. Chỉ mở khi hội đồng hỏi trúng một trong các điểm dưới
đây. Không tự nêu khi trình bày.

---

## Phần 0. Những chỗ có thể bị bắt lỗi: đọc kỹ trước

Đây là các chỗ trong luận văn hoặc slide **không khớp hoàn toàn** với code hoặc kết quả thật. Luận
văn đã nộp nên không sửa, nhưng phải có câu trả lời sẵn.

### 0.1 Phần mềm đang chạy cấu hình CBAM, không phải Hybrid ⚠️ quan trọng nhất

- **Sự thật:** chương 6 và slide kiến trúc phần mềm (trang 26) ghi ô "Chấm nhãn: Hybrid". Nhưng code
  app (`spine-labeling-app/backend/app/inference/grading.py`) nạp `GradingModelWithCBAM` từ
  `phase2_cbam.pth`, tức **nhánh CBAM** của Hybrid. 11 nhãn = 3 nhãn RSNA từ mô hình CBAM + 8 nhãn
  từ SpineNetV2 gốc.
- **Nếu hội đồng hỏi "phần mềm dùng mô hình nào?":**
  > "Dạ, phiên bản phần mềm hiện tại chạy cấu hình CBAM, tức nhánh CBAM của Hybrid, cho ba nhãn
  > ống sống và lỗ liên hợp, cộng với SpineNetV2 gốc cho tám nhãn còn lại. Em chưa tích hợp nhánh
  > BiomedCLIP vào phần mềm vì nhánh này thêm 86 triệu tham số và một môi trường thư viện riêng.
  > Trong luận văn em ghi là Hybrid là chưa chính xác, em xin nhận. Tích hợp đầy đủ Hybrid là bước
  > tiếp theo, kiến trúc phần mềm tách mô hình ra nên chỉ cần thay mô-đun chấm nhãn."
- **Đừng chủ động nói** nếu không bị hỏi, nhưng **đừng nói "phần mềm chạy Hybrid"** khi demo.
- Hệ quả tốt: cấu hình CBAM vẫn hơn hẳn SpineNetV2 ở lớp Severe (Severe Recall 0.288 so với 0.123,
  Severe F1 0.262 so với 0.152).

### 0.2 Bảng SPIDER theo nhãn thiếu dòng Pfirrmann

- **Sự thật:** Mean F1 0.653 ở trang 24 tính **đủ 8 nhãn**, nhưng bảng chi tiết (luận văn và phụ lục
  trang 52) chỉ có 7 dòng. Pfirrmann (3 seed): SpineNetV2 **0.592**, CBAM 0.540, BMC-only 0.439,
  Hybrid **0.544**.
- **Trả lời:** "Pfirrmann có trong Mean F1. Ở nhãn này SpineNetV2 cao hơn Hybrid, 0.592 so với 0.544;
  em sơ suất không đưa dòng này vào bảng chi tiết. Pfirrmann là thang 5 mức theo độ mất nước của đĩa
  đệm, mô hình nền đã được huấn luyện gốc trên nhãn này nên có lợi thế."
- **Điều này không đổi kết luận** vì luận văn chỉ nói Hybrid "ngang" SpineNetV2 trên SPIDER.

### 0.3 SPIDER có tinh chỉnh chia train/val theo đĩa đệm, không theo bệnh nhân

- **Sự thật:** `train_spider*.py` gọi `train_test_split` trên danh sách 1.174 đĩa đệm, nên các đĩa
  đệm của cùng một bệnh nhân có thể nằm ở cả train lẫn val.
- **Trả lời:** "Ở thí nghiệm SPIDER có tinh chỉnh, em chia 80/20 theo đĩa đệm. Cả bốn cấu hình dùng
  đúng cùng một phép chia nên so sánh giữa chúng vẫn công bằng, nhưng con số tuyệt đối có thể lạc
  quan hơn so với chia theo bệnh nhân. Trên RSNA, thí nghiệm chính, em chia theo bệnh nhân."
- RSNA thì chia theo bệnh nhân (`train_test_split` trên danh sách bệnh nhân), không rò rỉ.

### 0.4 Số bệnh nhân tập val RSNA

- **Sự thật:** luận văn ghi "train khoảng 1.800, val khoảng 225 bệnh nhân". Code chia 80/20 ngẫu nhiên
  theo bệnh nhân trên ~1.973 ca: **~1.578 train / 395 val bệnh nhân**, 1.942 đĩa đệm val (seed 42).
- **Trả lời:** "Em chia 80/20 theo bệnh nhân, tập val là 395 bệnh nhân với gần 2.000 đĩa đệm. Con số
  225 trong luận văn em ghi chưa đúng." Thí nghiệm chọn mô hình (trang 14) dùng **150 bệnh nhân**,
  đó là thí nghiệm khác.

### 0.5 Độ lệch chuẩn dòng Severe Recall

- Bảng ghi Hybrid 0.486 ± 0.016, SpineNetV2 0.123 ± 0.005 (chia $n = 3$). Các dòng khác chia
  $n - 1 = 2$. Tính lại cùng kiểu: ± **0.019** và ± **0.006**.
- **Trả lời:** "Dòng đó em tính nhầm kiểu độ lệch chuẩn; tính lại cho đồng bộ là ±0.019. Trung bình
  và kết luận không đổi."

### 0.6 Ning Shen: mô tả kiến trúc và thứ hạng

- **Sự thật** (theo repo `github.com/nshen7/rsna-2024`): Faster R-CNN phát hiện đĩa đệm + Swin
  Transformer V2 phân loại, trên lát **dọc**. **Không có giải.**
  - Slide trang 10 ghi đúng ("Faster R-CNN + Swin Transformer").
  - Chương 3 ghi sai ("nnU-Net + CNN 2D trên lát axial").
  - Chương 5, phần Thảo luận chung, ghi "giải pháp giải nhất RSNA 2024": **sai** (giải nhất là đội
    Avengers). Chỗ khác trong luận văn ghi đúng "một giải pháp hàng đầu".
  - Link Kaggle trong tài liệu tham khảo thật ra mở bài "6th place" của đội khác.
- **Khi nói:** chỉ gọi là "một giải pháp mạnh, có mã nguồn công khai trên RSNA 2024". **Không nói
  "giải nhất".**
- **Nếu bị hỏi:** "Dạ em ghi nhầm ở phần thảo luận; đây là một giải pháp công khai mạnh trên RSNA,
  mã nguồn trên GitHub của tác giả Ning Shen, không phải giải nhất. Em chọn làm mốc vì chạy lại được."

### 0.7 Bảng AUC theo điều kiện (luận văn, không có trên slide)

- Bảng AUC/AUPRC theo điều kiện trong luận văn lấy từ một lần xuất cũ hơn: ghi Severe AUC Hybrid
  0.963 / 0.858 / 0.837 (trung bình 0.886). File kết quả chính thức của cùng lần chạy cho
  0.962 / 0.869 / 0.858 (trung bình **0.896**, khớp bảng theo lớp và slide).
- **Nếu bị hỏi:** "Số chuẩn là 0.896; bảng đó em lấy từ lần xuất trước, xu hướng như nhau: Severe
  tăng ở cả ba nhãn, Moderate lỗ liên hợp giảm."

### 0.8 Grad-CAM

- Một ca, chỉ là **minh họa**. Thử nghiệm sau luận văn cho thấy bản đồ nhiệt trên dữ liệu này không
  hơn phép đoán "giữa ảnh", vì khối ảnh vốn cắt căn giữa vào đĩa đệm.
- **Không bao giờ nói** "Grad-CAM chứng minh mô hình nhìn đúng chỗ".
- **Hình không có trong luận văn** (lấy từ bài báo MIWAI). Nếu hỏi: "Hình này em đưa thêm trong bài
  báo để minh họa, luận văn chỉ báo cáo bảng số."
- **Chữ trên slide không khớp hình:** slide ghi "không CBAM dàn trải, có CBAM tập trung", nhưng hình
  cho thấy bên không CBAM vùng đỏ lại gọn hơn. Nếu thầy chỉ ra, nhận luôn: "Dạ đúng, câu chữ em viết
  chưa chính xác. Hai bản đồ đều nằm ở vùng đĩa đệm và ống sống; điểm khác là xác suất Severe 0.26 so
  với 0.53."

### 0.9 SpineNetV2 gốc **có** xử lý mất cân bằng (luận văn viết ngược)

- Luận văn chương 4 (mục phân tích khoảng trống) viết: *"SpineNetV2 huấn luyện bằng cross-entropy
  thông thường, không có cơ chế riêng cho lớp hiếm."*
- Paper gốc (Windsor 2022, mục 4.4.3) ghi: mỗi nhãn trong Genodisc mất cân bằng nặng, *"hence a
  balanced cross entropy loss is used in all cases"*, tức **cross-entropy có cân bằng lớp**.
- Cấu hình SpineNetV2 **trong thí nghiệm của em** đúng là cross-entropy thường: giữ nguyên thân mạng,
  chỉ học lại đầu ra trên RSNA, không trọng số lớp (log: `class_weight_mode: none`, 4,617 tham số học).
- **Nếu thầy chỉ ra:** "Dạ đúng, câu đó em viết chưa chính xác. Bản gốc trên Genodisc có dùng
  cross-entropy cân bằng lớp. Ý em là cấu hình mô hình nền khi em học lại đầu ra trên RSNA dùng
  cross-entropy thường. Phần cải thiện của Hybrid vì vậy đến từ cả kiến trúc lẫn quy trình xử lý mất cân
  bằng; luận văn cũng trình bày hai thứ này như một gói."
- **Câu hỏi nối tiếp có thể gặp:** "Sao không cho SpineNetV2 dùng cùng quy trình cân bằng để so cho công
  bằng?" → "Đó là thí nghiệm đối chứng em chưa chạy. Bằng chứng hiện có là BMC-only và Hybrid dùng
  **cùng** quy trình cân bằng, mà Hybrid vẫn hơn rõ (Severe F1 0.268 lên 0.356), nên kiến trúc có đóng
  góp riêng." Xem thêm Phần 7d.

---

## G. ⚠️ Câu khó nhất có thể bị hỏi: khối 9 lát có chứa lỗ liên hợp không?

- **Code làm gì:** mẫu được tạo từ tọa độ trên chuỗi **T2**, mà tọa độ trên T2 là của **ống sống**
  (đường giữa). Khối 9 lát vì vậy **căn giữa vào đường giữa**, và nhãn lỗ liên hợp trái, phải được gắn
  vào chính khối đó. Lỗ liên hợp nằm lệch sang hai bên.
- **Đã đo thêm sau luận văn** (báo cáo Phase 3 gửi thầy hướng dẫn, **không có trong luận văn**): lỗ liên
  hợp lệch trung bình khoảng 4 lát so với tâm; khoảng **32% (trái) và 47% (phải) nằm ngoài cửa sổ 9
  lát**. Ngoài ra, nhãn lỗ liên hợp trong RSNA được bác sĩ chấm trên chuỗi **T1**, trong khi mô hình chỉ
  đọc T2.
- **Cách trả lời nếu bị hỏi:** "Dạ đúng, đây là hạn chế em phát hiện sau khi nộp. Khối ảnh căn giữa theo
  ống sống, nên một phần lỗ liên hợp nằm ngoài 9 lát; nhãn lỗ liên hợp lại được chấm trên T1. Đó là lý
  do chính lỗ liên hợp kém hơn ống sống, ngoài việc thiếu mặt cắt ngang. Em đã thử một mô hình riêng đọc
  T1 cho lỗ liên hợp, AUPRC tăng khoảng 0.04–0.05, nhưng chưa đưa vào luận văn."
- **Đừng tự nêu** nếu không bị hỏi. Đừng nói "mô hình nhìn thấy lỗ liên hợp rõ".

---

## Số tham số trên slide phụ lục

- Trang 41 ghi CBAM-3D ResNet-34 21.3M; số đúng là 63.5M (21.3M là của bản 2D). Nếu bị hỏi: "Dạ con
  số đúng là khoảng 63.5 triệu vì kernel 3D, em ghi nhầm trên slide."
- Trang 43 ghi CBAM thêm ~1% tham số; thực tế 46,264 tham số, chưa tới 0.1%.

---

## SpineNetV2: paper và code lệch nhau

- **Chi tiết paper và code lệch nhau, nên biết:** paper chỗ ghi ResNet34, chỗ ghi "ResNet32", và nói chỉ
  lớp đầu dùng kernel 3D. Nhưng **code và trọng số công bố** dùng kernel $3 \times 3 \times 3$ ở mọi lớp
  (63.5 triệu tham số), stride chỉ trên chiều cao và rộng. Paper cũng mô tả đầu ra là MLP 2 lớp; trong
  luận văn, cấu hình SpineNetV2 và CBAM gắn **một đầu tuyến tính** cho mỗi nhãn. Nếu hỏi: "Em theo đúng
  code và trọng số nhóm tác giả công bố."

---

## Số bệnh nhân RSNA ghi trong luận văn

- Luận văn (chương 5, mục bộ dữ liệu) ghi "tập train khoảng 1,800 bệnh nhân, tập val khoảng 225". Thực tế
  thí nghiệm: 1,973 bệnh nhân, chia theo bệnh nhân 80/20, khoảng 1,578 / 395 (seed 42: 1,942 đĩa đệm val).
- Nếu bị hỏi: "Dạ con số thí nghiệm là 1,973 bệnh nhân chia 80/20 theo bệnh nhân, 395 bệnh nhân val; dòng
  mô tả trong chương 5 em ghi chưa khớp, em sẽ sửa trong bản cuối."
- SPIDER: luận văn ghi khoảng 1,520 mẫu đĩa đệm là của cả bộ dữ liệu; code tinh chỉnh dùng 1,174 đĩa đệm.
