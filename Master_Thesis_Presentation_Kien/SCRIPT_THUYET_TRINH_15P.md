# Script thuyết trình bảo vệ — 15 phút (nói nhanh, một mình)

> Mục tiêu: ~14–15 phút để có buffer hỏi đáp. Cột thời gian là mốc tích lũy gợi ý.
> Nguyên tắc: **slide làm nền, mình nói** — không đọc chữ trên slide; mỗi slide nhấn 1–2 ý.
> Văn phong: design-first (vai trò trước, công cụ sau), không nói "ghép CBAM với BiomedCLIP".

---

## MỞ ĐẦU (0:00–2:30)

### [0:00] Slide tiêu đề
"Em chào quý thầy cô. Em là Hà Trung Kiên. Hôm nay em trình bày đề tài *Ứng dụng AI trong gán nhãn ảnh y tế cho bệnh đau thắt lưng*, dưới sự hướng dẫn của thầy Phan Trọng Nhân."

### [0:15] Giới thiệu đề tài
"Đau thắt lưng là bệnh lý rất phổ biến; chẩn đoán hiện dựa nhiều vào việc bác sĩ đọc thủ công ảnh MRI cột sống — vừa tốn thời gian, vừa phụ thuộc kinh nghiệm, và **dễ bỏ sót ca nặng**."
**Câu nối quan trọng (nói thêm):** "Trong lâm sàng, **bỏ sót một ca nặng nguy hiểm hơn nhiều so với sai ở lớp nhẹ** — nên mục tiêu của em không phải là accuracy tổng thể, mà là **tăng recall/F1 cho lớp bệnh nặng**." *(câu này nối thẳng tới kết quả Severe Recall ở cuối bài)*

### [0:40] Ví dụ các dấu hiệu bất thường
"Đây là ví dụ Normal/Mild so với Severe trên 3 nhóm bệnh của RSNA. Có thể thấy vùng tổn thương rất nhỏ và khó phân biệt — đó là lý do bài toán khó."

### [1:00] Bài toán và mục tiêu  ⭐ (làm rõ scope)
"Bối cảnh là một hệ thống hỗ trợ chẩn đoán gồm **hai mô-đun: phân vùng cấu trúc giải phẫu, và đánh giá mức độ bất thường (grading)**."
**Nói rõ để tránh bị hỏi "segmentation hay grading":** "Vì tên đề tài là *gán nhãn*, em xin làm rõ: gán nhãn ở đây gồm **hai mức** — nhãn cấu trúc giải phẫu qua phân vùng, và nhãn bệnh lý/mức độ qua grading. **Phần em trực tiếp tinh chỉnh sâu là mô-đun grading.**"
"Giai đoạn này em tập trung grading với **hai bài toán cốt lõi: (1) mất cân bằng lớp Severe** (chỉ chiếm ~4%); **(2) mở rộng nhãn zero-shot**."
"Từ đó, **bốn mục tiêu**: xây pipeline 2 giai đoạn và chọn mô hình nền; tinh chỉnh grading để giải quyết hai bài toán trên; và kiểm chứng tổng quát hóa sang dữ liệu khác."

### [2:05] Phạm vi đề tài
"Phạm vi: chỉ MRI thắt lưng, chủ yếu mặt cắt Sagittal. Giai đoạn thực tập tập trung **chọn model + tinh chỉnh grading**; tích hợp pipeline đầy đủ và phần mềm để dành cho Đồ án tốt nghiệp."

---

## DỮ LIỆU & CHỌN MÔ HÌNH (2:30–4:50)

### [2:30] Các bộ dữ liệu sử dụng
"Đề tài dùng 3 bộ dữ liệu: cho **grading** có **RSNA 2024** (tập chính, phân 4 mức Normal đến Severe) và **SPIDER** (8 nhãn bệnh, để kiểm chứng tổng quát hóa); cho **phân vùng** có **Lumbar Spine Segmentation** của Chu — 23 ca MRI T2, làm tập kiểm thử độc lập."

### [3:05] Lựa chọn mô hình phân vùng  *(Nhóm thực nghiệm 1)*
"Em so 4 mô hình phân vùng. **TotalSpineSeg đạt Precision 97.87% và ASSD thấp nhất** — định vị thân đốt sống chính xác để trích khối liên đốt sống (IVV). SpineNetV2 Recall cao nhưng Precision thấp, dễ phân vùng thừa. → **Em chọn TotalSpineSeg.**"
> **Gloss thuật ngữ (nếu thầy hỏi):** *Dice* = độ chồng lấp giữa mặt nạ dự đoán và mặt nạ chuẩn (0–1, càng cao càng tốt); *Precision/Recall* tính ở mức voxel (Precision = không phân vùng lan ra nền; Recall = phủ đủ vùng cột sống); *ASSD* = sai lệch đường biên trung bình tính bằng mm, **càng nhỏ càng tốt**. Chi tiết + ví dụ ở `DEFENSE_METRICS_VI.md` (Phần B).

### [3:35] Minh hoạ output phân vùng
"Đây là output thực tế của TotalSpineSeg — phân vùng và gán nhãn từng đốt sống; mặt nạ này dùng định vị ROI cho mô-đun grading sau."

### [3:55] Lựa chọn mô-đun đánh giá bất thường  *(vẫn Nhóm 1)*
"Cho mô-đun grading, em so SpineNetV2, MedGemma, và giải pháp Ning Shen. **Em chọn SpineNetV2** làm backbone: chuyên biệt cột sống, nhiều nhãn, mã nguồn mở. Ning Shen điểm cao hơn nhưng chỉ 5 nhãn và chưa qua bình duyệt; MedGemma Recall quá thấp (~12%)."

### [4:25] Thách thức: mất cân bằng lớp (RSNA)
"Nhưng nhìn phân bố nhãn của SpineNetV2 trên RSNA: **lớp Severe chỉ ~4%** — mất cân bằng rất nặng. Đây chính là lý do Recall lớp Severe rất thấp, và là động lực cho thiết kế ở phần sau." *(SPIDER distribution để ở phụ lục, chỉ mở nếu được hỏi.)*

---

## MÔ HÌNH ĐỀ XUẤT (4:50–9:15)

### [4:50] Pipeline tổng thể
"Luồng mục tiêu: MRI → phân vùng → mặt nạ định vị ROI → grading → phần mềm. Nét liền là phần đã làm trong thực tập; nét đứt là kế hoạch Đồ án tốt nghiệp. Hiện hai mô-đun chạy độc lập, grading nhận trực tiếp khối ảnh MRI."

### [5:25] Động lực thiết kế  ⭐ (design-first, theo pain point)
"Thiết kế của em xuất phát từ chính 2 vấn đề:
- Lớp **Severe rất hiếm, vùng tổn thương nhỏ** → cần **một nhánh chú ý** tập trung đúng vùng, thay vì xử lý đều toàn ảnh.
- **Nhãn mới / chuyển miền** → cần **một nhánh tri thức nền đa phương thức**, đặt ảnh và văn bản chung không gian để liên hệ với prompt.
- Mỗi nhánh đơn lẻ chưa đủ → **kết hợp (fusion)** để lấy cả đặc trưng 3D cục bộ lẫn prior y khoa tổng quát."
*(Tránh nói "em ghép 2 model có sẵn" — nói "thiết kế hai nhánh để giải đúng 2 pain point".)*

### [6:05] Attention branch  *(giải nhu cầu 1: khu trú tổn thương nhỏ)*
"Nhánh chú ý: mục tiêu là dồn trọng số vào đúng kênh và vùng không gian của tổn thương nhỏ — nhắm trực tiếp lớp Severe. Hiện thực bằng CBAM-3D: chú ý theo kênh rồi không gian, mở rộng cho khối ảnh 3D."

### [ẩn] Chi tiết Channel & Spatial Attention  *(SLIDE ĐÃ ẨN — KHÔNG trình chiếu)*
> Chỉ nói nếu hội đồng hỏi sâu về CBAM:
"Chi tiết hai khối: **Channel attention** gộp avg+max pool qua MLP chia sẻ rồi sigmoid (chú trọng kênh đặc trưng nào); **Spatial attention** pool theo kênh rồi conv $7{\times}7{\times}7$ rồi sigmoid (chú trọng vị trí không gian nào). Áp dụng tuần tự kênh trước, không gian sau. Công thức đầy đủ ở slide công thức trong appendix."

### [6:45] Multimodal branch  *(giải nhu cầu 2: bù dữ liệu hiếm + mở nhãn)*
"Nhánh tri thức nền: mục tiêu (a) **bù cho dữ liệu Severe quá ít** bằng đặc trưng học sẵn từ dữ liệu lớn, và (b) đặt ảnh–văn bản chung không gian để **mở nhãn mới bằng prompt**. Hiện thực bằng BiomedCLIP, học tương phản trên 15 triệu cặp; dự đoán bằng cosine giữa ảnh và prompt."

### [7:20] Kiến trúc Hybrid hai nhánh  *(ghép 2 nhánh lại → work thế nào)*
"Giờ ghép hai nhánh lại — chúng **bổ trợ nhau**: một bên cho đặc trưng cục bộ 3D, một bên cho prior y khoa tổng quát. **Fusion head** hợp nhất hai đặc trưng, rồi **đầu cosine** đối sánh với prompt văn bản. Nhờ đầu cosine–prompt này, mô hình vừa phân loại được nhãn đã học, vừa **thêm được nhãn mới zero-shot** chỉ bằng cách đổi prompt."

### [8:10] Giai đoạn huấn luyện  ⭐ (nói luôn phần xử lý mất cân bằng đã bỏ slide riêng)
"Trong huấn luyện: dữ liệu qua **augmentation và minority oversampling**; loss kết hợp **Focal Loss và Uncertainty Loss** — đây chính là **pipeline xử lý mất cân bằng**, ưu tiên các mẫu khó thuộc lớp Severe. **Cả hai nhánh đều frozen (CBAM-ResNet nạp pretrained, BiomedCLIP học sẵn); chỉ huấn luyện khối hợp nhất ~0.5M (fusion MLP + cosine head).**"
> **Ghi chú tái dùng checkpoint (nếu hỏi "CBAM có train không"):** CBAM được train riêng ở **bước 1** (cấu hình CBAM trong ablation, lưu thành `best_model_attention.pth`); **bước 2** Hybrid nạp đúng checkpoint đó rồi frozen, chỉ học khối hợp nhất. Nhờ tái dùng CBAM đã học sẵn nên Hybrid hội tụ rất nhanh.
*(Phần Focal/Uncertainty/oversampling: slide riêng đã bỏ → nói gọn ngay tại đây.)*

> **CHI TIẾT TỪNG KỸ THUẬT — để thủ (slide không có, nói dựa trên sơ đồ nếu thầy hỏi "mỗi cái để làm gì"):**
>
> - **Augmentation (tăng cường dữ liệu):** biến đổi *hình học* (lật ngang có hoán đổi nhãn trái/phải, xoay nhẹ) và *cường độ* (chỉnh sáng, tương phản, thêm nhiễu Gaussian). **Mục đích:** tăng đa dạng dữ liệu, chống overfitting, giúp lớp hiếm tổng quát hóa tốt hơn từ ít mẫu.
> - **Minority oversampling:** nhân số lần xuất hiện của mẫu lớp **Moderate + Severe** (hệ số ~5). **Mục đích:** mỗi batch có nhiều mẫu lớp hiếm hơn → mô hình "thấy" Severe thường xuyên, không bị lớp đa số Normal/Mild lấn át.
> - **Focal Loss ($\gamma=2$, kèm trọng số lớp):** nhân hệ số $(1-p_t)^{\gamma}$ để **giảm trọng số mẫu dễ** (lớp đa số model đã đoán đúng tự tin), **dồn việc học vào mẫu khó/hiếm** (Severe). Nhãn thiếu được bỏ qua (ignore\_index).
> - **Uncertainty Loss (Kendall):** **tự học trọng số** cho từng tác vụ (3 condition) qua tham số log-variance → tự cân bằng nhiều loss, khỏi dò tay trọng số.
>
> **Câu gói gọn:** *"Augmentation + oversampling cân bằng ở mức dữ liệu; Focal + Uncertainty cân bằng ở mức hàm mất mát — cả hai cùng kéo sự chú ý của mô hình về lớp Severe hiếm."*

### [8:55] Giai đoạn suy diễn  *(nói nhanh)*
"Khi suy diễn: chỉ forward, không loss; hai nhánh trích đặc trưng, hợp nhất, rồi qua các head sinh xác suất và nhãn."

---

## KẾT QUẢ (9:15–13:20)

### [9:15] Thiết lập thực nghiệm  ⭐ (khung 2 nhóm)
"Em có **hai nhóm thực nghiệm**: nhóm 1 — chọn mô hình nền (vừa trình bày); nhóm 2 — **xác thực đóng góp grading** (phần này), gồm ablation 4 cấu hình: SpineNetV2 / CBAM / BMC-only / Hybrid. Chạy trên RTX 4090, đánh giá macro R/P/F1/AUC/AUPRC và riêng lớp Severe."

### [9:45] Kết quả tổng hợp trên RSNA  ⭐ (chỉ nhấn 3 số)
"Ba con số chính: **Mean F1 macro 0.420 → 0.528**; **Severe Recall 11.5% → 46.4%** (~4 lần); **Severe F1 0.149 → 0.343**."
**Câu phải nói:** "Baseline có accuracy cao hơn (0.814 so với 0.721) vì nó **thiên về lớp đa số**. Nhưng trong y tế, **bỏ sót Severe rủi ro hơn** — nên Hybrid chấp nhận giảm accuracy tổng để **tăng mạnh recall và F1 lớp nặng**."
*(Đừng đọc cả bảng.)*
> **Gloss thuật ngữ (nếu thầy hỏi):** *Recall* = % ca Severe thật mà model bắt được (đừng bỏ sót bệnh); *Precision* = % lần báo Severe là đúng (tránh báo nhầm); *F1* = trung bình điều hòa của hai cái trên; *AUC* = chất lượng xếp hạng bất kể ngưỡng; *AUPRC* = như AUC nhưng riêng cho lớp hiếm, không bị lừa bởi lớp đa số. Chi tiết + ví dụ ở `DEFENSE_METRICS_VI.md` (Phần A).

### [10:40] Kết quả chi tiết theo lớp RSNA
"Bảng per-class khẳng định: ở lớp Severe, Hybrid thắng đồng thời cả 5 chỉ số so với baseline — không chỉ thắng trung bình mà thắng đúng lớp khó nhất."

### [11:10] Grad-CAM
"Trực quan hóa cho thấy trên các ca Severe, **vùng chú ý của mô hình trùng với vị trí tổn thương thật** — mô hình nhìn đúng chỗ, tăng độ tin cậy và khả năng diễn giải cho bác sĩ."

### [11:40] Mở rộng nhãn zero-shot trên SPIDER  ⭐ (nói cẩn thận)
"Đây là kiểm chứng **zero-shot thật sự**: em **không huấn luyện trên SPIDER**, chỉ đổi prompt văn bản để dự đoán 8 nhãn mới. Kết quả trung bình F1 ~0.362 — còn khiêm tốn, nhưng **chứng minh kiến trúc có khả năng mở nhãn mà không cần train lại**."
> **Defense note (nếu hỏi "nhãn nào tốt/kém"):** nhãn liên quan đĩa đệm (disc bulging, disc narrowing) chuyển giao **tốt nhất** vì gần ngữ nghĩa với dữ liệu BiomedCLIP đã học và với nhãn đã học trên RSNA; nhãn xa ngữ nghĩa (Modic, Pfirrmann, trượt đốt sống) còn yếu vì hiếm gặp trong dữ liệu tiền huấn luyện.
*(Phân biệt rõ với slide sau — slide sau là transfer learning, có train.)*

### [12:25] Kết quả tổng hợp trên SPIDER  ⭐ (phân biệt với zero-shot)
"Khi **cho phép tinh chỉnh (transfer learning)**, Hybrid đạt **Mean F1 0.653**, ngang baseline (0.646) và **không thoái hóa** khi chuyển miền. Đây là đánh giá khả năng chuyển miền, khác với zero-shot ở slide trước."

### [13:00] Kết quả chi tiết trên SPIDER  *(nói nhanh)*
"Chi tiết theo nhóm bệnh: Hybrid cải thiện rõ nhất ở Disc herniation và Disc bulging."

---

## KẾT LUẬN (13:20–15:00)

### [13:20] Đóng góp chính  ⭐
"Tóm lại 3 đóng góp:
1. **Đề xuất kiến trúc Hybrid hai nhánh** cho grading — nâng Severe Recall từ 11.5% lên 46.4%.
2. **Cơ chế mở rộng nhãn theo hướng zero-shot/vision-language** — thêm nhãn mới qua mô tả văn bản; trong báo cáo em kiểm chứng thêm bằng transfer learning sang SPIDER.
3. **Đánh giá tổng quát hóa liên tập dữ liệu** — Hybrid giữ hiệu năng ổn định.
Bên cạnh đó là khảo sát, chọn mô hình nền cho cả hai mô-đun."
*(Đóng góp 2 nói theo kiểu này để tránh bị bắt "zero-shot hay transfer?".)*

### [14:05] Hạn chế
"Hạn chế: hai mô-đun chưa tích hợp end-to-end; mới khai thác mặt cắt Sagittal, chưa dùng Axial."

### [14:25] Kế hoạch Đồ án tốt nghiệp
"Hướng tiếp: tích hợp pipeline end-to-end, xây phần mềm giao diện cho bác sĩ, tinh chỉnh tiếp grading."

### [14:55] Cảm ơn
"Em xin hết phần trình bày. Em cảm ơn quý thầy cô đã lắng nghe và rất mong nhận góp ý ạ."

---

## GHI CHÚ NHANH KHI BỊ HỎI (thủ sẵn)
- **Novelty?** → "Không phải thuật toán mới; là **đóng góp thiết kế + thực nghiệm**: kết hợp prior thị giác–ngôn ngữ y khoa frozen với backbone grading có chú ý để **đồng thời giải mất cân bằng và mở nhãn zero-shot** trên phân độ IVD thắt lưng. Điểm mới rõ nhất là **mở nhãn zero-shot** — các grading model trước có head cố định, không làm được."
- **Zero-shot hay transfer?** → "Cả hai: slide zero-shot là *không train*, chỉ prompt; slide SPIDER tổng hợp là *transfer learning* để kiểm chứng chuyển miền."
- **Sao chọn SpineNetV2 mà không Ning Shen điểm cao hơn?** → "Ning Shen chỉ 5 nhãn, chưa bình duyệt; SpineNetV2 nhiều nhãn, mã nguồn mở, phù hợp làm backbone tinh chỉnh."
- **Sao accuracy giảm?** → "Đánh đổi có chủ đích: baseline cao do thiên lớp đa số; y tế ưu tiên không bỏ sót ca nặng → tối ưu recall/F1 Severe."
- **Param/tốc độ?** → "Chỉ ~0.5M tham số được huấn luyện (cả CBAM-ResNet 21.3M lẫn BiomedCLIP 86.2M đều frozen); suy diễn ~62 ms/mẫu, không chậm hơn baseline."
