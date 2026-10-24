# Bộ hỏi–đáp bảo vệ luận văn: hỏi gì cũng giải thích được

Bảo vệ Thứ Hai 02/11/2026. Mọi số khớp luận văn đã nộp (commit `ef23e38`) và đã đối chiếu lại với
code, file kết quả trong repo.

**Cách dùng file này**
- Mỗi câu có **Trả lời** (câu nói được ngay, 1–3 câu) và **Vì sao / số liệu** (để hiểu, không cần
  đọc lên).
- Đi theo thứ tự slide, rồi tới các câu tổng quát về toàn bài.
- Công thức: xem `3_CONG_THUC_LUAN_VAN.md` (đủ 32 công thức + cách tính từng bảng).
- Metric chi tiết: `4_GIAI_THICH_METRIC_VA_CONG_THUC.md`.
- Nguyên tắc chung khi bị hỏi khó: **thừa nhận chỗ yếu trước, rồi nói nó ảnh hưởng tới kết luận
  đến đâu**. Đừng chối, đừng nói quá điều luận văn đã chứng minh.

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

---

## Phần 1. Giới thiệu (trang 4–7)

### Trang 4: Giới thiệu đề tài

**Hỏi: Bài toán cụ thể là gì? Đầu vào, đầu ra?**
- **Trả lời:** Đầu vào là ảnh MRI cột sống thắt lưng (DICOM, ưu tiên lát dọc T2). Đầu ra là mặt nạ
  phân vùng các cấu trúc, bảng mức độ bệnh theo từng tầng đĩa đệm, và nhật ký chỉnh sửa của bác sĩ.
- **Vì sao:** chương 1 mục "Đầu vào và đầu ra".

**Hỏi: Vì sao cần AI? Bác sĩ đọc vẫn được mà.**
- **Trả lời:** Mỗi ca có nhiều chuỗi, mỗi chuỗi nhiều lát, chấm nhiều tiêu chí cho 5 tầng đĩa đệm:
  tốn thời gian và kết quả dao động giữa người đọc. Hệ thống em làm để **rút ngắn thời gian đọc phim
  và đánh dấu ca nghi ngờ**, bác sĩ vẫn là người quyết định.

**Hỏi: Ảnh ví dụ trên slide lấy từ đâu?**
- **Trả lời:** Từ bộ RSNA 2024, hàng trên Normal/Mild, hàng dưới Severe cho 3 nhóm bệnh.

### Trang 5: Bốn thách thức

**Hỏi: Severe 4% là tính thế nào?**
- **Trả lời:** Đếm trên 3 nhãn được dùng của RSNA (ống sống, lỗ liên hợp trái, phải) ở mọi tầng:
  Normal/Mild khoảng 81%, Moderate khoảng 15%, Severe khoảng 4%.

**Hỏi: Bốn thách thức này giải quyết ở đâu?**
- **Trả lời:** (1) mất cân bằng → bảng RSNA trang 21; (2) nhãn cố định → zero-shot SPIDER trang 23;
  (3) tổng quát hóa → SPIDER có tinh chỉnh trang 24; (4) khoảng cách người dùng → phần mềm, demo.

**Hỏi: Vì sao "bỏ sót ca nặng nguy hiểm hơn báo nhầm"?**
- **Trả lời:** Báo nhầm thì bác sĩ xem lại ảnh là loại được ngay. Bỏ sót thì có thể không ai phát
  hiện, bệnh nhân chậm điều trị. Vì vậy em ưu tiên Recall của lớp Severe.

### Trang 6: Mục tiêu

**Hỏi: Tổng quát hóa sao không là mục tiêu riêng?**
- **Trả lời:** Vì nó không phải một thứ em xây, mà là một tính chất em kiểm chứng: em đưa mô hình sang
  bộ SPIDER của bệnh viện khác để xem có giữ được hiệu năng không.

**Hỏi: Mục tiêu 1 là phân vùng, em có cải tiến phân vùng không?**
- **Trả lời:** Không. Em khảo sát 4 mô hình, chọn TotalSpineSeg và dùng nguyên bản. Đóng góp nằm ở
  mô-đun chấm nhãn. Phân vùng là bước nền để định vị đĩa đệm.

### Trang 7: Phạm vi

**Hỏi: Vì sao chỉ MRI, không X-quang, CT?**
- **Trả lời:** MRI là chuẩn để xem mô mềm (đĩa đệm, tủy, rễ thần kinh), là nơi nhìn thấy hẹp ống
  sống, thoát vị. X-quang và CT chủ yếu cho xương.

**Hỏi: Sagittal, Axial, T1, T2, STIR khác nhau thế nào?**
- **Sagittal (mặt cắt dọc):** chia trái–phải, thấy cả dãy đốt sống, đĩa đệm; tốt cho thoái hóa đĩa,
  thoát vị, trượt đốt sống.
- **Axial (mặt cắt ngang):** vuông góc cột sống, thấy rõ ống sống, ngách bên, rễ thần kinh hai bên;
  tốt nhất cho lỗ liên hợp và ngách bên.
- **T1:** mỡ sáng, dịch tối; cần cho Modic.
- **T2 / STIR:** dịch sáng (dịch não tủy trắng, đĩa khỏe sáng); STIR là T2 xóa tín hiệu mỡ. Dùng
  cho Pfirrmann, thoát vị, hẹp.

**Hỏi: Vì sao chỉ dùng lát dọc?**
- **Trả lời:** SpineNetV2, mô hình nền, được thiết kế và huấn luyện sẵn trên lát dọc T2, và lát dọc
  thấy được cả ba nhãn RSNA em chấm. Đổi lại, lỗ liên hợp yếu hơn vì nhìn rõ nhất trên Axial; đó là
  hướng phát triển.

**Hỏi: RSNA có 5 nhãn, sao em dùng 3?**
- **Trả lời:** Hai nhãn hẹp ngách bên chủ yếu đánh giá trên mặt cắt ngang, mô hình em chỉ đọc lát dọc
  nên em không dùng.

---

## Phần 2. Cơ sở lý thuyết, công trình liên quan (trang 9–10)

### Trang 9: CBAM và BiomedCLIP

**Hỏi: CBAM là gì, khác self-attention thế nào?**
- **Trả lời:** CBAM là mô-đun chú ý nhẹ gắn vào CNN: chú ý theo kênh (chọn loại đặc trưng) rồi theo
  không gian (chọn vị trí). Self-attention so từng cặp vị trí, chi phí tăng theo bình phương số vị
  trí, rất nặng với khối 3D; CBAM chỉ tính một trọng số cho mỗi kênh và mỗi vị trí.
- Ví dụ số đầy đủ: `1_SCRIPT_BAO_VE_LVTN.md` khối 📘 dưới trang 9.

**Hỏi: "Kênh" cụ thể là cái gì? Cho ví dụ.**
- **Trả lời:** Sau mỗi tầng, ResNet cho ra nhiều "bản đồ" cùng kích thước với ảnh (thu nhỏ). Mỗi bản
  đồ là một **kênh**, do một bộ lọc tạo ra, sáng lên ở chỗ có một kiểu chi tiết mà bộ lọc đó bắt.
  - Số thật trong mô hình: tầng 1 có 64 kênh, tầng 2 có 128, tầng 3 có 256, tầng 4 có 512. Ở tầng
    cuối, mỗi kênh là một khối $9 \times 7 \times 14$ (9 lát giữ nguyên, chiều cao và rộng thu nhỏ
    16 lần từ $112 \times 224$).
  - Ví dụ để hình dung (mạng tự học, em không gán tên cho từng kênh):

    | Kênh | "Sáng lên" ở chỗ | Có ích cho chấm hẹp ống sống? |
    |---|---|---|
    | A | bờ sau đĩa đệm lồi ra | có, rất |
    | B | dịch não tủy (màu trắng trên T2) quanh tủy | có: dịch bị ép mất là dấu hiệu hẹp |
    | C | mỡ dưới da lưng | gần như không |
    | D | nhiễu, nền đen | không |

  - **Chú ý theo kênh** cho mỗi kênh một số từ 0 đến 1 rồi nhân vào cả kênh: A 0.9, B 0.85, C 0.1,
    D 0.05. Tức là chọn **loại chi tiết** nào đáng tin cho ca này. Nó không quan tâm chi tiết nằm ở
    đâu.

**Hỏi: "Không gian" cụ thể là gì? Chú ý theo không gian là theo cái gì?**
- **Trả lời:** Không gian là **vị trí** trong khối ảnh, tức bộ ba (lát, hàng, cột). Ở tầng cuối có
  $9 \times 7 \times 14 = 882$ vị trí. Chú ý theo không gian cho mỗi vị trí một số từ 0 đến 1, dùng
  chung cho mọi kênh.
  - Ví dụ một đĩa đệm hẹp lỗ liên hợp **bên trái**: các vị trí ở vùng lỗ liên hợp trái trên mấy lát
    phía trái được khoảng 0.9; ống sống ở lát giữa khoảng 0.5; thân đốt sống, cơ lưng, nền đen khoảng
    0.1.
  - Vì là 3D, mỗi lát có trọng số riêng: tổn thương chỉ nằm ở 1–2 lát vẫn được làm nổi lên.
- **Câu tóm gọn để nói:** "Kênh trả lời câu **xem cái gì** (loại chi tiết nào), không gian trả lời câu
  **xem ở đâu** (vị trí nào). Kênh trước để lọc ra chi tiết đúng loại, rồi không gian tìm chỗ của chi
  tiết đó."

**Hỏi: Trọng số chú ý học từ đâu? Có ai chỉ cho mô hình chỗ tổn thương không?**
- **Trả lời:** Không. Dữ liệu chỉ có nhãn mức độ cho cả đĩa đệm, không có nhãn vị trí. Trọng số chú ý
  học gián tiếp từ hàm mất mát phân loại: chú ý vào đâu mà chấm đúng hơn thì trọng số đó được giữ.

**Hỏi: CBAM có làm đổi kích thước đặc trưng không? Gắn ở đâu?**
- **Trả lời:** Không đổi. Nó chỉ **nhân** trọng số vào bản đồ đặc trưng, ra đúng kích thước cũ, nên
  gắn sau mỗi tầng trong 4 tầng ResNet mà không phải sửa phần còn lại. Nhờ vậy vẫn nạp được trọng số
  SpineNetV2. Gắn ở cả 4 tầng vì tầng đầu thấy chi tiết nhỏ (cạnh, đường viền), tầng sau thấy cấu trúc
  lớn (đĩa đệm, ống sống); lọc ở mọi mức.

**Hỏi: Lỡ CBAM chú ý sai chỗ thì sao? Làm sao biết nó nhìn đúng?**
- **Trả lời:** Có thể chú ý sai, vì không có nhãn vị trí để ép. Em **không** khẳng định nó nhìn đúng
  chỗ: luận văn không đo trực tiếp trọng số chú ý, Grad-CAM chỉ là minh họa một ca (xem 0.8). Bằng
  chứng em dùng là kết quả: thêm CBAM nâng Severe F1 từ 0.152 lên khoảng 0.26.

**Hỏi: Nếu kênh "nhiễu" bị nhân 0.05 thì có mất thông tin không?**
- **Trả lời:** Có giảm, và đó là mục đích: bớt nhiễu để tín hiệu tổn thương rõ hơn. Trọng số không
  về đúng 0 (sigmoid luôn lớn hơn 0), và mỗi ca tính lại một bộ trọng số, nên kênh bị giảm ở ca này
  vẫn có thể được giữ ở ca khác. Ngoài ra ResNet có đường tắt (residual) nên thông tin gốc vẫn đi qua
  các khối sau.

**Hỏi: Vì sao chọn CBAM mà không chọn SE-Net hay attention khác?**
- **Trả lời:** SE-Net chỉ có chú ý theo kênh, không chọn được vị trí. Tổn thương của em là một vùng
  nhỏ, nên cần chú ý không gian. CBAM có cả hai, rất nhẹ (thêm khoảng 1% tham số), gắn được vào
  ResNet mà vẫn nạp được trọng số SpineNetV2 đã huấn luyện.

**Hỏi: BiomedCLIP là gì? Khác CLIP?**
- **Trả lời:** BiomedCLIP là mô hình ảnh–văn bản y khoa của Microsoft, học theo cách của CLIP trên
  khoảng 15 triệu cặp ảnh và chú thích từ bài báo PubMed. Ảnh dùng ViT-B/16, văn bản dùng PubMedBERT.
  CLIP gốc học trên ảnh tự nhiên nên không hiểu thuật ngữ y khoa.

**Hỏi: CLIP học thế nào? Học tương phản là gì?**
- **Trả lời:** Mỗi lô có N cặp ảnh–chú thích. Mô hình kéo vector ảnh gần vector chú thích đúng của
  nó, đẩy xa N − 1 chú thích sai. Kết quả là ảnh và văn bản nằm chung một không gian, so được bằng
  cô-sin.

**Hỏi: Vì sao BiomedCLIP mà không MedCLIP, PMC-CLIP?**
- **Trả lời:** Trong ba mô hình, BiomedCLIP có dữ liệu học sẵn lớn nhất và đa dạng nhất (MRI, CT,
  X-quang, mô bệnh học), có mã nguồn và trọng số công khai.

**Hỏi: BiomedCLIP có học ảnh MRI cột sống không?**
- **Trả lời:** Không riêng cột sống, nhưng có ảnh MRI trong 15 triệu cặp. Vì vậy em không dùng nó một
  mình mà ghép với nhánh CBAM đã học trên chính ảnh cột sống. Kết quả BMC-only (chỉ BiomedCLIP) cho
  thấy một mình nó chưa đủ: Mean F1 0.482 so với Hybrid 0.527.

### Trang 10: Công trình liên quan

**Hỏi: Khoảng trống luận văn nhắm tới là gì?**
- **Trả lời:** Chưa có hướng nào vừa xử lý riêng lớp hiếm, vừa thêm được nhãn mới mà không huấn luyện
  lại, vừa tổng quát sang dữ liệu khác. SpineNetV2 nhãn cố định, yếu lớp hiếm; Ning Shen chính xác
  nhưng nặng, nhãn cố định; MedGemma linh hoạt nhưng độ chính xác thấp.

**Hỏi: Ning Shen là ai, làm gì?** → Xem 0.6. Trả lời: "Một giải pháp công khai mạnh trên RSNA 2024,
dùng Faster R-CNN phát hiện đĩa đệm và Swin Transformer phân loại, chia sẻ trên GitHub, chưa công bố
thành bài báo bình duyệt."

**Hỏi: MedGemma là gì?**
- **Trả lời:** Mô hình ngôn ngữ thị giác lớn của Google cho y khoa: đưa ảnh kèm câu hỏi, mô hình trả
  lời bằng chữ. Linh hoạt nhãn vì chỉ cần đổi câu hỏi, nhưng không chuyên cột sống: trên RSNA Recall
  chỉ 6–21%.

**Hỏi: SpineNetV2 là gì?**
- **Trả lời:** Mô hình của nhóm Oxford (Windsor 2022), mã nguồn mở. Phát hiện đốt sống bằng hồi quy
  trường vector, cắt khối quanh đĩa đệm (IVV), rồi một mạng 3D ResNet-34 chấm nhiều nhãn bệnh cùng
  lúc bằng nhiều đầu ra tuyến tính.

---

## Phần 3. Dữ liệu và chọn mô hình (trang 12–14)

### Trang 12: Bộ dữ liệu

**Hỏi: RSNA 2024 gồm gì?**
- **Trả lời:** Bộ dữ liệu cuộc thi của RSNA và ASNR, từ 8 trung tâm, 5 châu lục, hơn 50 bác sĩ chẩn
  đoán hình ảnh gán nhãn. Mỗi bệnh nhân có sagittal T2/STIR, sagittal T1, axial T2. Nhãn 5 tình trạng
  × 5 tầng (L1/L2 → L5/S1) × 3 mức.

**Hỏi: Vì sao Normal và Mild gộp chung?**
- **Trả lời:** RSNA gán nhãn sẵn như vậy. Về lâm sàng, Normal và Mild thường chưa cần can thiệp, nên
  ranh giới quan trọng là Moderate và Severe.

**Hỏi: Chia train/val thế nào? Có rò rỉ bệnh nhân không?**
- **Trả lời:** Chia 80/20 **theo bệnh nhân**, nên một bệnh nhân chỉ nằm ở train hoặc val. Mỗi seed
  một cách chia ngẫu nhiên khác; Hybrid nạp nhánh CBAM của **cùng seed**, nên không có ca val nào
  từng được nhánh CBAM thấy. Val khoảng 395 bệnh nhân (xem 0.4).

**Hỏi: Có tập test riêng không, hay chỉ val?**
- **Trả lời:** Tập test của cuộc thi RSNA bị ẩn nhãn nên không dùng được. Em báo cáo trên tập val giữ
  riêng theo bệnh nhân; checkpoint chọn trên chính tập val này (xem Phần 7), đó là một hạn chế. Bù
  lại, SPIDER là kiểm chứng trên dữ liệu ngoài.

**Hỏi: Tiền xử lý ảnh thế nào?**
- **Trả lời:** Từ chuỗi sagittal, cắt khối 3D quanh tâm từng đĩa đệm theo tọa độ có sẵn trong nhãn
  RSNA, đưa về $9 \times 112 \times 224$ (lát × cao × rộng) và chuẩn hóa về [0, 1], theo chuẩn của
  SpineNetV2.

**Hỏi: Vì sao 9 lát, 112 × 224?**
- **Trả lời:** Đây là kích thước đầu vào của SpineNetV2. Giữ đúng kích thước thì nạp được trọng số đã
  huấn luyện sẵn. Chín lát đủ phủ bề dày đĩa đệm và mô xung quanh mà chi phí vừa phải.

**Hỏi: SPIDER gồm gì?**
- **Trả lời:** 218 bệnh nhân từ Hà Lan (van der Graaf, Scientific Data 2024), ảnh MHA kèm mặt nạ 3D.
  8 nhãn: Pfirrmann (5 mức), Modic (4 mức), và 6 nhãn có/không (hẹp đĩa đệm, trượt đốt sống, tổn
  thương mặt trên, mặt dưới, thoát vị, phồng). Không nhãn nào trùng RSNA. Hiếm nhất: trượt đốt sống
  2.8%, thoát vị 4.7%.

**Hỏi: Pfirrmann, Modic là gì?**
- **Pfirrmann:** thang 1–5 đánh giá thoái hóa đĩa đệm theo tín hiệu T2 và chiều cao đĩa: độ 1 đĩa
  sáng, cao, khỏe; độ 5 đĩa đen, xẹp.
- **Modic:** biến đổi tủy xương sát mặt đốt sống. Type 1 (phù viêm: T1 tối, T2 sáng), type 2 (thoái
  hóa mỡ: T1 sáng, T2 sáng), type 3 (xơ cứng: tối cả hai). Phải đọc **T1 và T2 cùng nhau**.

**Hỏi: Lumbar Spine Segmentation là gì?**
- **Trả lời:** Bộ của Chu và cộng sự, Đại học Bern: 23 bệnh nhân MRI T2, có mặt nạ đốt sống từ T11
  tới L5, một mặt nạ chung cho cả thân đốt, gai sau, mỏm ngang. Em dùng làm tập kiểm thử độc lập cho
  phân vùng.

### Trang 13: Chọn mô hình phân vùng

**Hỏi: Bốn chỉ số nghĩa là gì?** → `3_CONG_THUC_LUAN_VAN.md` (5.1)–(5.4).

**Hỏi: TotalSpineSeg Recall chỉ 56%, sao vẫn chọn?**
- **Trả lời:** Nhãn chuẩn của bộ này tính cả gai sau và mỏm ngang, còn TotalSpineSeg chỉ khoanh thân
  đốt sống, nên Recall thấp là do **khác định nghĩa nhãn**, không phải bỏ sót. Trong hệ thống, mặt nạ
  chỉ dùng để định vị đĩa đệm rồi cắt khối ảnh, nên **khoanh đúng chỗ quan trọng hơn khoanh đủ**.
  Precision 97.87% và ASSD thấp nhất 6.85 mm cho thấy nó khoanh đúng chỗ.

**Hỏi: Sao không chọn SpineNetV2, Recall 91.79%?**
- **Trả lời:** Precision chỉ 51.76%, gần một nửa vùng khoanh là mô xung quanh; cắt khối ảnh từ đó sẽ
  lẫn mô bên cạnh. Và SpineNetV2 chỉ cho khung bao, không cho mặt nạ từng cấu trúc.

**Hỏi: MedCLIP-SAMv2 Recall cao nhất sao vẫn kém?**
- **Trả lời:** Recall cao vì khoanh rất rộng: Precision 29%, ASSD 19.5 mm, gấp gần 3 lần TotalSpineSeg.
  Đó là mô hình zero-shot, không học riêng cột sống.

**Hỏi: TotalSpineSeg làm gì cụ thể?**
- **Trả lời:** Hai mạng nnU-Net nối tiếp: mạng 1 phân vùng và tìm các đĩa đệm mốc; mạng 2 dùng thêm
  mặt nạ đĩa đệm lẻ để tách đốt sống chẵn/lẻ; rồi thuật toán gán tên từng đốt và đĩa. Bài gốc báo
  Dice 0.88 cho đốt sống, gán nhãn đúng 99%.

**Hỏi: 23 ca có ít quá không?**
- **Trả lời:** Đó là bộ có nhãn công khai phù hợp; mục đích là chọn mô hình có sẵn, không phải huấn
  luyện. Độ lệch chuẩn của TotalSpineSeg ±5.23% cho thấy kết quả ổn định giữa các ca.

### Trang 14: Chọn mô-đun đánh giá bất thường

**Hỏi: Sao phải quy về có/không bệnh?**
- **Trả lời:** Ba mô hình có đầu ra khác nhau: SpineNetV2 có thang riêng, MedGemma trả lời có/không,
  Ning Shen theo 3 mức RSNA. Quy về chung một câu hỏi "có bệnh không" (Moderate hoặc Severe = có) thì
  mới so được. Bệnh nhân có bệnh nếu ít nhất 1 trong 5 tầng có bệnh. Đánh giá trên 150 bệnh nhân.

**Hỏi: Số trên slide là gì?** → Trung bình cộng 3 nhóm bệnh (xem `3_CONG_THUC_LUAN_VAN.md` 5.2).

**Hỏi: F1 ở đây 77–81% mà bảng RSNA chỉ 0.42–0.53, mâu thuẫn không?**
- **Trả lời:** Không, hai bài toán khác nhau. Ở đây là có/không bệnh ở mức bệnh nhân (2 lớp, gộp
  Moderate và Severe, chỉ cần 1 tầng đúng). Bảng RSNA là phân 3 mức cho từng đĩa đệm, lớp Severe rất
  hiếm. Chính thí nghiệm này lộ ra vấn đề: F1 phát hiện 77.5% nhưng chấm đúng mức thì SpineNetV2 chỉ
  bắt 12.3% ca Severe.

**Hỏi: Ning Shen cao hơn 3 điểm, sao không chọn?**
- **Trả lời:** Ning Shen là tổ hợp nhiều mô hình lớn (Faster R-CNN + Swin Transformer, mỗi nhóm bệnh
  một mô hình), khó duy trì, khó tích hợp vào phần mềm và khó can thiệp kiến trúc. SpineNetV2 mã nguồn
  mở, một mạng chung cho nhiều nhãn, kiến trúc 3D ResNet-34 đơn giản để gắn CBAM, BiomedCLIP. Ba điểm
  F1 ở bài toán có/không đổi lấy khả năng phát triển là chấp nhận được.

**Hỏi: Đã so Hybrid với Ning Shen trên cùng bài toán 3 mức chưa?**
- **Trả lời:** Trong luận văn thì chưa, Ning Shen chỉ dùng ở thí nghiệm chọn mô hình. Ning Shen tự báo
  Severe F1 0.501 nhưng gộp chung các nhãn, không cùng cách tính với 0.356 của em, nên em không đặt
  hai số cạnh nhau.

**Hỏi: MedGemma tệ vậy là do prompt không?**
- **Trả lời:** Có thể prompt ảnh hưởng một phần, nhưng xu hướng nhất quán ở cả 3 nhóm: Precision chấp
  nhận được, Recall 6–21%, tức mô hình thiên về trả lời "bình thường". Đó là mô hình đa dụng chưa tinh
  chỉnh cho cột sống.

---

## Phần 4. Mô hình đề xuất (trang 16–19, phụ lục 40)

### Trang 16: Pipeline

**Hỏi: IVV là gì?**
- **Trả lời:** Intervertebral Volume: khối ảnh 3D cắt quanh một đĩa đệm (giữa hai đốt sống), kích
  thước $9 \times 112 \times 224$. Mỗi đĩa đệm chấm riêng.

**Hỏi: Số liệu chương 5 là của cả pipeline hay chỉ mô-đun chấm nhãn?**
- **Trả lời:** **Chỉ mô-đun chấm nhãn.** Trong thí nghiệm, khối ảnh cắt theo tọa độ đĩa đệm có sẵn
  trong nhãn RSNA, để mọi cấu hình nhận cùng đầu vào. Trong phần mềm, khối ảnh cắt từ mặt nạ
  TotalSpineSeg. Đánh giá cả chuỗi end-to-end chưa làm.

### Trang 17: Kiến trúc Hybrid

**Hỏi: Tóm tắt kiến trúc trong 30 giây?**
- **Trả lời:** Khối ảnh đĩa đệm đi vào hai nhánh. Nhánh CBAM: ResNet-34 3D gắn CBAM sau mỗi tầng, ra
  vector 512 chiều, học chi tiết 3D của đĩa đệm. Nhánh BiomedCLIP: 3 lát giữa, mã hóa từng lát, gộp
  bằng trọng số chú ý, ra vector 512 chiều, mang kiến thức y khoa chung. Ghép thành 1024, MLP hợp
  nhất ra 512, chuẩn hóa, rồi so cô-sin với vector câu mô tả từng mức; mức giống nhất được chọn.

**Hỏi: So cô-sin với vector văn bản của BiomedCLIP, vậy kiến thức của mô hình dồn hết vào BiomedCLIP,
nhánh CBAM có tác dụng gì?**
- **Trả lời ngắn:** Không. Vector văn bản chỉ là **đáp án mẫu cố định**, quyết định là ở vector ảnh,
  mà vector ảnh được tạo từ **cả hai nhánh**.
- **Giải thích:**
  - Chín câu mô tả (3 nhãn × 3 mức) mã hóa **một lần**, giữ cố định, thành 9 điểm mốc trong không
    gian. Bản thân chúng không nhìn ảnh, không biết gì về ca bệnh.
  - Việc chấm là xem **vector ảnh $\hat f_{img}$ rơi gần mốc nào**. Vector ảnh do Fusion MLP tạo ra từ
    $[f_{cbam}; f_{bmc}]$, tức chi tiết 3D của nhánh CBAM góp trực tiếp vào việc vector rơi vào đâu.
  - Ví von: câu mô tả như các ô đáp án A/B/C in sẵn trên đề; người làm bài là phần đọc ảnh (hai nhánh +
    MLP). Đề dùng chữ của BiomedCLIP không có nghĩa BiomedCLIP làm bài.
- **Bằng chứng số:** cấu hình **BMC-only** chính là Hybrid với cùng đầu so cô-sin, cùng 9 câu mô tả,
  chỉ tắt nhánh CBAM. Thêm nhánh CBAM vào:
  - Mean F1 0.482 → **0.527**.
  - Severe F1 0.268 → **0.356**.
  - Mean Recall 0.532 → **0.592**.

  Nếu kiến thức chỉ nằm ở BiomedCLIP thì hai cấu hình này phải ngang nhau.
- **Phần đúng của câu hỏi, nên thừa nhận:** BiomedCLIP cung cấp **không gian chung ảnh–văn bản**. Nhờ
  đó mới viết được câu mô tả cho nhãn mới, tức zero-shot. Nên phần "mở rộng nhãn" dựa vào BiomedCLIP,
  còn phần "chấm chính xác ca nặng" dựa nhiều vào CBAM.
- **Cẩn thận khi trích số:** cột **CBAM** trong bảng RSNA (0.477 / 0.262) là mô hình CBAM riêng, dùng
  đầu tuyến tính, **không** so với văn bản. Muốn chứng minh nhánh CBAM có đóng góp trong cơ chế so
  cô-sin thì dùng cặp **BMC-only và Hybrid**, đừng dùng cột CBAM.

**Hỏi: Chấm bằng cách so với vector văn bản của BiomedCLIP, vậy kết quả có bị phụ thuộc vào bộ mã
hóa văn bản đó không?**
- **Trả lời ngắn:** Có, và em nói rõ là phụ thuộc ở mức nào. Với nhãn đã học thì phụ thuộc ít. Với
  nhãn mới zero-shot thì phụ thuộc nhiều, đó là cái giá để mở rộng nhãn được.
- **Nhãn đã học (RSNA, và SPIDER có huấn luyện):**
  - Chín vector câu mô tả đóng vai trò giống **hàng trọng số của một lớp phân loại tuyến tính**, chỉ
    khác là được **cố định** bằng văn bản thay vì học tự do.
  - Fusion MLP được huấn luyện để kéo vector ảnh về đúng mốc. Mốc nằm đâu thì MLP học cách đưa ảnh về
    đó, nên chất lượng bộ mã hóa văn bản không quyết định nhiều.
  - Chỗ phụ thuộc thật: nếu hai câu, ví dụ "moderate" và "severe" của cùng một nhãn, cho ra hai vector
    rất gần nhau, thì hai mức đó khó tách hơn. Hệ số $s$ học được (tối đa 100) giúp khuếch đại khoảng
    cách nhỏ đó.
  - Bằng chứng không bị kéo xuống: trên RSNA, Hybrid (so với văn bản) cao hơn SpineNetV2 (đầu tuyến
    tính học tự do); trên SPIDER có huấn luyện, Hybrid đạt 0.653 so với 0.646.
- **Nhãn mới, zero-shot:** phụ thuộc **gần như hoàn toàn** vào việc bộ mã hóa văn bản hiểu câu mô tả
  đến đâu, và vào cách viết câu. Vì vậy F1 zero-shot trên SPIDER chỉ 0.362. Em trình bày zero-shot là
  khả năng **khởi đầu**: chấm được ngay khi chưa có dữ liệu, rồi có dữ liệu thì huấn luyện tiếp lên
  0.653.
- **Vì sao chấp nhận:**
  - Muốn thêm nhãn bằng một câu chữ thì bắt buộc phải có bộ mã hóa văn bản; không có nó thì quay về
    đầu tuyến tính cố định như SpineNetV2.
  - Bộ mã hóa văn bản giữ nguyên, mã hóa một lần, nên kết quả ổn định và lặp lại được.
  - Đây là PubMedBERT học trên văn bản y khoa, hiểu thuật ngữ như "stenosis", "herniation" tốt hơn mô
    hình văn bản thông thường.
- **Nói thật nếu hỏi sâu:** em chưa làm thí nghiệm thay 9 vector văn bản bằng 9 vector học tự do để đo
  chính xác mức phụ thuộc, và chưa dò tối ưu câu mô tả. Hướng cải thiện: viết nhiều câu cho mỗi mức rồi
  lấy trung bình (prompt ensembling), hoặc cho học thêm vài token trong câu (kiểu CoOp).

**Hỏi: Nhìn hình kiến trúc, những chỗ nào người xem dễ hiểu sai?** (để chủ động nói khi trình bày)
- **Hộp "text prompts → multimodal text encoder":** đó là **bộ mã hóa văn bản của BiomedCLIP**
  (PubMedBERT), giữ nguyên, mã hóa câu một lần. Không phải "mọi thứ chạy qua BiomedCLIP": nhánh ảnh
  CBAM vẫn đi vào Fusion head.
- **Hình không đánh dấu phần nào đóng băng:** người xem có thể nghĩ huấn luyện cả hơn 100 triệu tham
  số. Thực tế chỉ học bộ gộp lát, Fusion MLP, $s$ và 3 tham số $\sigma_k$, khoảng 1.18 triệu.
- **"MRI slices":** chỉ là **3 lát giữa**, không phải cả 9 lát.
- **Hộp "New label set, zero-shot (e.g. SPIDER)":** đừng để bị hiểu nhầm 0.653 là kết quả zero-shot.
  Zero-shot là 0.362; 0.653 là khi **có huấn luyện** trên SPIDER, vẫn cùng đầu so với văn bản.
- **Hai hộp xanh bên phải:** không phải mô hình xuất ra cả hai cùng lúc. Cùng một mô hình, chỉ đổi bộ
  câu mô tả đưa vào thì chấm bộ nhãn khác.

**Hỏi (nối tiếp): Vector ảnh từ CBAM vốn không cùng không gian với văn bản, sao so được?**
- **Trả lời:** Chính Fusion MLP làm việc đó: nó học cách chiếu $[f_{cbam}; f_{bmc}]$ vào không gian
  512 chiều của văn bản BiomedCLIP. Hàm mất mát kéo vector ảnh lại gần câu mô tả đúng mức, đẩy xa
  các câu sai. Sau khi học, $\hat f_{img}$ nằm trong cùng không gian với văn bản, dù một nửa đầu vào
  đến từ CBAM.

**Hỏi: Vì sao hai nhánh? Một nhánh không đủ?**
- **Trả lời:** Bảng RSNA trả lời: CBAM một mình Severe F1 0.262, BiomedCLIP một mình 0.268, ghép lại
  **0.356**. Và độ lệch giữa 3 lần chạy giảm (Severe F1 ±0.082 và ±0.034 xuống ±0.017). Trên SPIDER,
  mỗi nhánh một mình đều tụt dưới mô hình nền, ghép lại thì giữ được mức mô hình nền.

**Hỏi: Hybrid có phải chỉ là ghép trung bình hai mô hình (ensemble) không?**
- **Trả lời:** Em đã thử: lấy trung bình softmax của SpineNetV2 và BMC-only (seed 123) chỉ được Mean F1
  0.441 và Severe F1 0.181, còn Hybrid học cách trộn được 0.499 và 0.348. Ensemble có AUC nhỉnh hơn
  chút (0.851 so với 0.822) nhưng quyết định ở lớp Severe kém hẳn. Kết quả này có trong bài báo MIWAI,
  không có trong luận văn.

**Hỏi: Vì sao BiomedCLIP chỉ đọc 3 lát giữa?**
- **Trả lời:** BiomedCLIP là mô hình 2D, mỗi lát một lượt ViT, 9 lát tốn gấp 3. Ba lát giữa đi qua tâm
  khối ảnh; thông tin 3D đầy đủ đã có nhánh CBAM đọc cả 9 lát.

**Hỏi: Vì sao lát 112 × 224 đổi thành 224 × 224, méo ảnh không?**
- **Trả lời:** Có giãn gấp đôi chiều dọc để đúng đầu vào ViT-B/16. Em chấp nhận vì nhánh này giữ cố
  định, và nhánh CBAM vẫn đọc ảnh gốc. Cải thiện được bằng đệm viền thay vì kéo giãn.

**Hỏi: Slice attention pool hoạt động thế nào?** → `3_CONG_THUC_LUAN_VAN.md` (4.4).

**Hỏi: Vì sao ghép (concat) mà không cộng, hay dùng cổng (gating), cross-attention?**
- **Trả lời:** Cộng đòi hai vector cùng nghĩa ở từng chiều, mà chúng từ hai mạng khác nhau. Ghép rồi
  để MLP học cách trộn là đơn giản nhất, đủ cho dữ liệu cỡ vừa; cổng hay cross-attention thêm tham số,
  dễ học thuộc khi lớp Severe ít mẫu.

#### Nhóm câu hỏi về ghép hai vector ($f_{cbam}$, $f_{bmc}$ → $\hat f_{img}$)

Nhắc lại luồng: $f_{cbam}$ (512 số) và $f_{bmc}$ (512 số) → ghép thành 1024 số → Linear 1024→768 →
GELU → dropout 0.1 → Linear 768→512 → chuẩn hóa L2 → $\hat f_{img}$.

**Hỏi: Ghép (concat) cụ thể là làm gì? Cho ví dụ.**
- **Trả lời:** Chỉ là đặt hai vector nối đuôi nhau. Ví dụ thu nhỏ còn 3 số mỗi nhánh:
  $f_{cbam} = [0.8, 0.1, 0.5]$, $f_{bmc} = [0.2, 0.7, 0.3]$ → ghép $= [0.8, 0.1, 0.5, 0.2, 0.7, 0.3]$.
  Không cộng, không bỏ số nào. Sau đó MLP mới là nơi trộn hai nhánh với nhau.

**Hỏi: Ghép lại rồi qua MLP, lỡ bị mất đặc trưng ảnh (của một nhánh) thì sao?**
- **Trả lời:** Tách làm hai chỗ:
  - **Bước ghép không mất gì:** đủ 1024 số, như ví dụ trên.
  - **Bước MLP có nén** 1024 → 768 → 512, nên có bỏ bớt thông tin. Nhưng MLP được huấn luyện bằng
    chính hàm mất mát chấm nhãn, nên cái nó giữ là phần **có ích cho việc chấm mức độ**, cái nó bỏ là
    phần không liên quan. Nén có chủ đích, không phải mất ngẫu nhiên.
- **Bằng chứng không bị mất:** nếu MLP làm mất hẳn một nhánh thì Hybrid sẽ chỉ bằng nhánh còn lại.
  Thực tế Hybrid hơn cả hai: RSNA Mean F1 0.527 so với 0.477 (CBAM riêng) và 0.482 (BMC-only),
  Severe F1 0.356 so với 0.262 và 0.268. Trên SPIDER, CBAM riêng tụt xuống 0.619 còn Hybrid 0.653. Tức
  là thông tin của cả hai nhánh đều được dùng. (Cặp so sát nhất là BMC-only và Hybrid, vì cùng đầu
  so cô-sin; cột CBAM là mô hình riêng với đầu tuyến tính.)
- **Nói thật nếu hỏi sâu:** em chưa đo trực tiếp mỗi nhánh đóng góp bao nhiêu (ví dụ tắt từng nhánh
  khi suy luận trên mô hình đã học). Đó là phân tích có thể bổ sung.

**Hỏi: Làm sao giữ được đặc trưng tốt của từng nhánh?**
- **Trả lời:** Ba cách:
  1. **Đóng băng cả hai nhánh.** Đặc trưng của CBAM (đã học tốt trên ảnh cột sống) và của BiomedCLIP
     (kiến thức y khoa, cùng không gian với văn bản) không bị huấn luyện làm hỏng. Bước 2 chỉ học phần
     trộn, khoảng 1.18 triệu tham số.
  2. **MLP đủ rộng:** lớp ẩn 768, lớn hơn đầu ra 512, nên không bị "thắt cổ chai" quá hẹp.
  3. **Chống học thuộc:** dropout 0.1, và SupCon kéo các ca cùng mức độ lại gần nhau, nên MLP phải giữ
     đặc trưng phân biệt mức độ thật sự chứ không nhớ từng ca.

**Hỏi: Một nhánh có lấn át nhánh kia không? Hai vector có cùng thang đo không?**
- **Trả lời (nói thật):** Không cùng thang đo. $f_{bmc}$ được tạo từ các vector BiomedCLIP đã chuẩn
  hóa độ dài 1, còn $f_{cbam}$ là trung bình bản đồ đặc trưng sau ReLU, không chuẩn hóa, nên độ lớn có
  thể khác. Lớp tuyến tính đầu của MLP có trọng số riêng cho từng chiều trong 1024 chiều, nên học được
  cách phóng to hay thu nhỏ từng nhánh. Kết quả ablation ở trên cho thấy không nhánh nào bị bỏ qua.
  Cải tiến dễ làm: thêm LayerNorm cho từng nhánh trước khi ghép.

**Hỏi: Có thử cách ghép "thông minh" hơn, ví dụ cổng (gating), chưa?**
- **Trả lời:** Có, ngoài luận văn. Em thử gated fusion kiểu GMU (Arevalo 2017): mỗi chiều có một cổng
  từ 0 đến 1 quyết định lấy bao nhiêu từ mỗi nhánh. So cùng điều kiện (cùng khởi tạo, 20 epoch, một lần
  chạy): Severe F1 **0.295 so với 0.321** của ghép + MLP, thua ở cả ba nhãn, rõ nhất ở lỗ liên hợp.
  Lý do khả dĩ: cổng chỉ chọn tỷ lệ pha giữa hai vector 512 chiều, ít khả năng biểu diễn hơn MLP trên
  1024 chiều, mà Severe lại ít mẫu. Cross-attention em chưa thử.
- **Đừng nói** gated là kết quả trong luận văn; chỉ là một lần chạy để kiểm tra lựa chọn thiết kế.

**Hỏi: Sao đầu ra là 512, không giữ 1024? Sao lớp ẩn là 768?**
- **Trả lời:** 512 để bằng số chiều vector văn bản của BiomedCLIP, nhờ vậy so cô-sin trực tiếp với câu
  mô tả. 768 là mức ở giữa 1024 và 512 để nén dần; em không dò kỹ con số này.

**Hỏi: Sao chuẩn hóa L2 ở cuối?**
- **Trả lời:** Để vector ảnh dài 1 giống vector văn bản, khi đó tích vô hướng chính là cô-sin, chỉ còn
  so **hướng**. Độ "tự tin" do hệ số $s$ điều khiển riêng (xem `3_CONG_THUC_LUAN_VAN.md` 4.6).

**Hỏi: Vì sao dùng GELU mà không ReLU?**
- **Trả lời:** GELU mượt hơn ReLU ở quanh 0 (không cắt cứng về 0), là hàm kích hoạt mặc định trong
  Transformer, mà nhánh BiomedCLIP chính là Transformer (ViT). Khác biệt thực tế giữa hai hàm ở đây là
  nhỏ; em không làm thí nghiệm so riêng.

**Hỏi: Nếu một nhánh hỏng lúc chạy (ví dụ ảnh xấu làm BiomedCLIP ra đặc trưng vô nghĩa) thì sao?**
- **Trả lời:** Mô hình hiện tại chưa được huấn luyện cho tình huống thiếu nhánh. Code đã có tùy chọn
  "modality dropout" (khi học, ngẫu nhiên tắt một nhánh) để mô hình chịu được trường hợp này, nhưng bản
  trong luận văn chưa bật. Đây là hướng làm thêm.

**Hỏi: Zero-shot hoạt động thế nào? Muốn thêm nhãn mới làm gì?**
- **Trả lời:** Mỗi mức độ của mỗi nhãn viết thành một câu tiếng Anh, BiomedCLIP mã hóa thành vector.
  Mô hình chọn mức có câu giống vector ảnh nhất. Muốn thêm nhãn mới, ví dụ "disc herniation", chỉ cần
  viết câu "a magnetic resonance image of disc herniation" và "... no disc herniation", mã hóa một lần,
  không đổi gì trong mạng, không huấn luyện.

**Hỏi: Câu mô tả (prompt) viết thế nào? Đổi câu có đổi kết quả không?**
- **Trả lời:** RSNA dùng câu dạng "severe spinal canal stenosis"; SPIDER dùng mẫu "a magnetic resonance
  image of [nhãn]". Có, cách viết câu ảnh hưởng kết quả zero-shot; em chưa dò tối ưu câu mô tả, đó là
  hướng cải thiện.

**Hỏi: Cô-sin, $s$, softmax, argmax?** → `3_CONG_THUC_LUAN_VAN.md` (4.6), (4.9); và
`5_CHUAN_BI_BAO_VE_LVTN.md` mục "Về softmax và argmax".

**Hỏi: Khác SpineNetV2 ở đâu?**
- **Trả lời:** Ba chỗ: (1) gắn CBAM vào backbone; (2) thêm nhánh BiomedCLIP; (3) thay đầu tuyến tính
  softmax cố định bằng đầu cô-sin với câu mô tả. Cộng thêm quy trình xử lý mất cân bằng khi huấn luyện.

### Trang 18: Xử lý mất cân bằng

**Hỏi: Vì sao cần cả bốn kỹ thuật?**
- **Trả lời:** Mỗi cái lo một việc. Focal và trọng số lớp sửa **mức phạt**; oversampling tăng **số lần
  thấy** ca hiếm; augmentation làm các bản lặp **không giống hệt** để khỏi học thuộc; uncertainty cân
  **giữa 3 nhãn bệnh** khó dễ khác nhau.

**Hỏi: Focal Loss, trọng số căn, uncertainty?** → `3_CONG_THUC_LUAN_VAN.md` (4.7), (4.8), (4.10).

**Hỏi: Có thí nghiệm bỏ từng kỹ thuật (ablation) không?**
- **Trả lời:** Trong luận văn, ablation là theo kiến trúc (SpineNetV2 / CBAM / BMC-only / Hybrid). Em
  chưa tách riêng từng kỹ thuật cân bằng. Đây là điểm em có thể bổ sung.

**Hỏi: Oversampling cụ thể?**
- **Trả lời:** Đĩa đệm nào có ít nhất một nhãn Moderate hoặc Severe được lặp 3 lần mỗi epoch.

**Hỏi: Augmentation cụ thể? Sao không lật trái–phải?**
- **Trả lời:** Lật trước–sau (xác suất 0.5), xoay ±10°, độ sáng/tương phản ±20% (xác suất 0.5), nhiễu
  Gauss độ lệch chuẩn tối đa 0.05 (xác suất 0.3). Không lật trái–phải vì sẽ đổi lỗ liên hợp trái thành
  phải mà nhãn không đổi theo. Biên độ vừa phải vì MRI có hướng chụp chuẩn.

**Hỏi: Oversampling có làm lệch xác suất, mô hình báo Severe nhiều quá không?**
- **Trả lời:** Có xu hướng đó: Precision của Severe thấp (khoảng 0.27), mô hình báo nhầm khá nhiều.
  Em chấp nhận vì vai trò là người đọc thứ hai; báo nhầm thì bác sĩ loại được.

### Trang 19: Huấn luyện và suy diễn

**Hỏi: Vì sao huấn luyện hai bước, sao không học hết một lần?**
- **Trả lời:** Hai lý do. Một, số ca Severe quá ít để cập nhật hàng chục triệu tham số mà không học
  thuộc; chỉ học khoảng 1.18 triệu thì an toàn hơn. Hai, nếu cập nhật BiomedCLIP theo RSNA, không gian
  chung ảnh–văn bản sẽ bị kéo lệch về 9 câu của RSNA, câu mô tả nhãn mới khó so được, mất zero-shot.

**Hỏi: Những gì được học ở bước 2?**
- **Trả lời:** $w_a$ (gộp lát), Fusion MLP, hệ số $s$, 3 tham số $\sigma_k$. Tổng khoảng 1.18 triệu.
  CBAM-ResNet 21.3 triệu và BiomedCLIP 86.2 triệu giữ cố định.

**Hỏi: Siêu tham số?**
- **Trả lời:** AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$, batch 32, tối đa 20 epoch, dừng
  sớm sau 5 epoch không cải thiện. RTX 4090 trên Vast.ai, PyTorch 2.0.

**Hỏi: Thời gian huấn luyện, suy luận?**
- **Trả lời:** SpineNetV2 16.4 phút, CBAM 56 phút, Hybrid thêm 24.8 phút trên nền CBAM. Suy luận khoảng
  62 ms một đĩa đệm cho cả ba, nên nhánh BiomedCLIP gần như không làm chậm.

---

## Phần 5. Kết quả (trang 21–24)

### Trang 21: Kết quả RSNA

**Hỏi: Kết quả chính là gì?**
- **Trả lời:** Severe Recall từ 0.123 lên 0.486, khoảng 4 lần: mô hình nền bắt khoảng 1/8 ca nặng,
  Hybrid bắt gần một nửa. Severe F1 từ 0.152 lên 0.356. Mean F1 từ 0.420 lên 0.527. Hybrid cao nhất
  mọi dòng trong bảng.

**Hỏi: Các con số tính thế nào?** → `3_CONG_THUC_LUAN_VAN.md` mục 5.3 (có kiểm tra số).

**Hỏi: Accuracy giảm từ 0.810 xuống 0.724, vậy là tệ hơn?**
- **Trả lời:** Không. SpineNetV2 đạt accuracy cao vì gần như luôn đoán Normal/Mild (Recall lớp này
  0.963), mà lớp đó chiếm 81% dữ liệu. Đổi lại nó sót gần hết ca nặng. Phần accuracy mất đi là do
  Hybrid chịu đoán Moderate/Severe; đổi lại Precision của Normal/Mild tăng 6–11 điểm, tức khi Hybrid
  nói bình thường thì đáng tin hơn.

**Hỏi: AUC tăng ít (0.872 → 0.899), sao bảo cải thiện nhiều?**
- **Trả lời:** AUC chỉ đo xếp hạng và bị lớp đa số làm đẹp. Cải thiện lớn nằm ở quyết định (Recall,
  F1). AUC và AUPRC Severe cùng tăng (AUPRC 0.284 → 0.333) cho thấy mô hình thật sự xếp hạng ca nặng
  tốt hơn, không phải chỉ hạ ngưỡng.

**Hỏi: Severe AUC 0.899 mà F1 0.356, sao chênh vậy?**
- **Trả lời:** AUC đo xếp hạng, không cần ngưỡng; F1 đo quyết định sau argmax. Với lớp 4%, dù xếp hạng
  tốt vẫn còn nhiều ca không bệnh điểm cao, nên Precision thấp kéo F1 xuống. Severe AUPRC 0.333 là
  khoảng 8 lần đoán bừa (0.04).

**Hỏi: CBAM có AUC thấp hơn SpineNetV2 (0.803 so với 0.835), vậy CBAM có hại?**
- **Trả lời:** CBAM có độ dao động lớn giữa các lần chạy (±0.050). Nó tăng Recall, F1 Severe nhưng xếp
  hạng chưa ổn định. Ghép với BiomedCLIP thì AUC lên 0.838 và dao động giảm.

**Hỏi: 3 lần chạy có đủ ý nghĩa thống kê?**
- **Trả lời:** Chưa đủ để kết luận chặt chẽ, luận văn có nêu ở phần hạn chế. Nhưng khoảng cách ở lớp
  Severe rất lớn so với độ lệch: Severe Recall 0.486 ± 0.019 so với 0.123 ± 0.006, ba lần chạy của
  Hybrid đều trên 0.46.

**Hỏi: Thực tập 2 báo Severe Recall 46.4%, giờ 0.486?**
- **Trả lời:** Đợt trước là một lần chạy (seed 42 cho 46.4%). Luận văn lấy trung bình 3 lần chạy
  (46.4 / 49.7 / 49.7), khớp với bài báo.

**Hỏi: Lỗ liên hợp yếu hơn ống sống bao nhiêu? Vì sao?**
- **Trả lời:** Severe Recall ống sống 70%, lỗ liên hợp 31–38%. Lỗ liên hợp nằm ở hai bên, nhìn rõ nhất
  trên mặt cắt ngang; trên lát dọc chỉ thấy ở vài lát bên, nên đây nhiều khả năng là giới hạn của dữ
  liệu đầu vào hơn là kiến trúc.

**Hỏi: Lớp Moderate thì sao?**
- **Trả lời:** Recall Moderate tăng mạnh (trung bình khoảng 39 điểm), nhưng AUC Moderate ở lỗ liên hợp
  giảm, tức bắt được nhiều hơn nhưng xếp hạng kém hơn mô hình nền. Đây là điểm yếu em đã nêu.

**Hỏi: Chọn checkpoint thế nào? Có chọn trên tập val rồi báo cáo trên chính tập val không?**
- **Trả lời:** Có, em chọn epoch có Severe F1 trung bình cao nhất trên tập val, dừng sớm sau 5 epoch
  không cải thiện, rồi báo cáo trên tập đó. Đây là cách làm phổ biến khi tập test bị ẩn, nhưng có thể
  hơi lạc quan. Cả 4 cấu hình chọn cùng một cách nên so sánh vẫn công bằng.

### Trang 22: Grad-CAM

**Hỏi: Grad-CAM tính thế nào?** → `3_CONG_THUC_LUAN_VAN.md` mục 6.
**Hỏi: Một ca chứng minh được gì?**
- **Trả lời:** Đây chỉ là một ca minh họa trực quan. Bằng chứng định lượng là bảng kết quả. Em không
  khẳng định mô hình luôn nhìn đúng chỗ.

**Hỏi: Xác suất Severe 0.26 và 0.53 nghĩa là gì?**
- **Trả lời:** Đó là xác suất softmax mô hình gán cho mức Severe ở ca đó. Không CBAM: 0.26, argmax chọn
  mức khác, đoán sai. Có CBAM: 0.53, chọn Severe, đoán đúng.

### Trang 23: Zero-shot SPIDER

**Hỏi: Zero-shot ở đây là gì chính xác?**
- **Trả lời:** Lấy Hybrid đã huấn luyện trên RSNA, giữ nguyên toàn bộ trọng số, chỉ thay 9 câu mô tả
  RSNA bằng câu mô tả 8 nhãn SPIDER, rồi chấm trực tiếp. Không một bước huấn luyện nào trên SPIDER.

**Hỏi: F1 0.362 có thấp không?**
- **Trả lời:** Thấp so với có huấn luyện (0.653), và em không xem đây là bộ phân loại hoàn chỉnh. Điểm
  quan trọng là mô hình chấm được bộ nhãn của hệ tiêu chí khác, bệnh viện khác, không huấn luyện lại;
  SpineNetV2 và CBAM không làm được vì đầu ra cố định 3 mức RSNA.

**Hỏi: Vì sao ba nhãn đĩa đệm tốt, Modic tệ?**
- **Trả lời:** Phồng, thoát vị, hẹp đĩa nhìn thấy trực tiếp trên lát dọc T2 và gần với các nhãn RSNA,
  nên đặc trưng học được dùng lại được (F1 0.56–0.61). Modic cần đọc **T1 và T2 cùng nhau**, mô hình chỉ
  có T2, nên AUC 0.454, ngang đoán bừa: thông tin không có trong đầu vào. Pfirrmann khó vì 5 mức.

**Hỏi: Trượt đốt sống F1 0.028 mà AUC 0.807?**
- **Trả lời:** Mô hình xếp hạng đúng (AUC 0.807), nhưng nhãn chỉ 2.8% ca dương, nên argmax hầu như
  không bao giờ chọn "có". Chỉnh ngưỡng sẽ cải thiện F1, nhưng zero-shot nghĩa là không có dữ liệu để
  chỉnh.

**Hỏi: So với zero-shot của BiomedCLIP gốc (không huấn luyện gì) thì sao?**
- **Trả lời:** Trong luận văn em không báo cáo con số này. Nếu bị hỏi tiếp: "Đó là mốc so sánh hợp lý,
  em sẽ bổ sung." Đừng đưa ra số.

### Trang 24: SPIDER có tinh chỉnh

**Hỏi: Thiết lập?**
- **Trả lời:** Mỗi cấu hình lấy checkpoint tốt nhất trên RSNA, giữ cố định phần thân, học lại phần đầu
  ra cho 8 nhãn SPIDER. 1.174 đĩa đệm, chia 80/20, tối đa 15 epoch, 3 seed. (Chia theo đĩa đệm, xem 0.3.)

**Hỏi: Hybrid 0.653 so với 0.646 là ngang, vậy Hybrid có hơn không?**
- **Trả lời:** Đúng là ngang, chênh nhỏ hơn độ lệch giữa các lần chạy; ở AUPRC và Accuracy, SpineNetV2
  còn cao hơn. Thông điệp không phải Hybrid thắng, mà là Hybrid **giữ được** mức mô hình nền khi đổi
  bệnh viện, trong khi CBAM một mình tụt xuống 0.619 và BMC-only 0.613. Hybrid là cấu hình duy nhất vừa
  tốt nhất trên RSNA vừa không tụt trên SPIDER.

**Hỏi: Vì sao CBAM tụt?**
- **Trả lời:** Nhiều khả năng CBAM học quá sát đặc điểm riêng của ảnh RSNA, nên không chuyển giao tốt.
  Nhánh BiomedCLIP giữ cố định, học từ dữ liệu rất rộng, làm điểm tựa. Em chưa chứng minh cơ chế, đây
  là quan sát hành vi qua kết quả.

**Hỏi: Thoát vị đĩa đệm Recall 0.500 nghĩa là gì?**
- **Trả lời:** Recall trong bảng là trung bình Recall của hai lớp có và không. 0.500 = (1.0 + 0.0) / 2:
  mô hình đoán "không" cho mọi ca, không bắt được ca thoát vị nào, ở cả 3 lần chạy. Hybrid đạt 0.636,
  tức có bắt được một phần.

**Hỏi: Nhưng AUPRC thoát vị của Hybrid lại thấp hơn SpineNetV2 (0.248 so với 0.271)?**
- **Trả lời:** Đúng. Nghĩa là phần cải thiện chủ yếu đến từ việc Hybrid chịu dự đoán lớp hiếm, chưa
  phải xếp hạng tốt hơn. Luận văn có nói rõ điểm này.

**Hỏi: Pfirrmann đâu?** → Xem 0.2.

---

## Phần 6. Phần mềm (trang 26–28)

**Hỏi: Kiến trúc phần mềm?**
- **Trả lời:** Giao diện React với Cornerstone3D (thư viện hiển thị ảnh y tế trên trình duyệt); máy
  chủ FastAPI giữ cơ sở dữ liệu và gọi mô hình; ảnh và mặt nạ lưu trên ổ đĩa. TotalSpineSeg cài môi
  trường Python riêng, gọi qua dòng lệnh, vì hai mô hình cần phiên bản thư viện khác nhau.

**Hỏi: Phần mềm dùng mô hình nào?** → Xem 0.1. **Không nói "Hybrid".**

**Hỏi: 11 nhãn gồm những gì?**
- **Trả lời:** 3 nhãn từ mô hình của em: hẹp ống sống, hẹp lỗ liên hợp trái, phải (3 mức RSNA). 8 nhãn
  từ SpineNetV2 gốc: Pfirrmann, hẹp đĩa đệm, trượt đốt sống, tổn thương mặt trên, mặt dưới đốt sống,
  biến đổi tủy xương trên, dưới, thoát vị. Bảng 8 tầng từ T10–T11 tới L5–S1; 3 nhãn RSNA để trống ở
  tầng ngực vì dữ liệu huấn luyện chỉ có từ L1–L2.

**Hỏi: Một ca chạy bao lâu?**
- **Trả lời:** Trên máy có GPU khoảng 1 phút một ca (phân vùng chiếm phần lớn). Trên CPU lâu hơn nhiều.

**Hỏi: Chỉnh sửa của bác sĩ lưu thế nào, dùng làm gì?**
- **Trả lời:** Mỗi lần sửa một ô rồi lưu, hệ thống ghi cặp (giá trị AI, giá trị bác sĩ) kèm mã ca và
  tầng, gom thành lô cho cả hệ thống. Dùng để đánh giá nhãn nào AI hay sai, và sau này huấn luyện lại.

**Hỏi: Có tự học lại từ chỉnh sửa không?**
- **Trả lời:** Chưa, đó là hướng phát triển 2. Và em cố ý **không** để mô hình tự cập nhật: phải có quy
  trình đánh giá và người duyệt, vì vài chỉnh sửa sai có thể làm mô hình xấu đi mà không ai biết.

**Hỏi: Đã thử với bác sĩ thật chưa?**
- **Trả lời:** Mới thử ở mức chức năng; phần mềm đã có phiếu đánh giá cho bác sĩ, đánh giá chính thức
  trên ca thật là hướng phát triển 1.

**Hỏi: Bảo mật dữ liệu bệnh nhân?**
- **Trả lời:** Mỗi bác sĩ một phòng riêng, đăng nhập bằng mã PIN, chỉ thấy ca của mình. Dữ liệu thử
  nghiệm là bộ công khai đã ẩn danh. Triển khai thật cần thêm ẩn danh DICOM và chạy trong mạng bệnh
  viện; đó chưa phải phạm vi luận văn.

**Hỏi: Dấu cảnh báo "mô hình ít chắc chắn" tính thế nào?**
- **Trả lời:** Dựa trên xác suất mô hình: ô có xác suất của mức được chọn dưới 0.5 thì gắn dấu cảnh
  báo để bác sĩ chú ý (`GradeTable.tsx`, ngưỡng `LOW_CONFIDENCE = 0.5`).

---

## Phần 7. Kết luận (trang 30–33)

**Hỏi: Đóng góp lớn nhất là gì?**
- **Trả lời:** Kiến trúc Hybrid cùng quy trình cân bằng, nâng Severe Recall khoảng 4 lần, và cơ chế
  chấm nhãn bằng câu mô tả cho phép thêm nhãn mới không huấn luyện lại.

**Hỏi: Điểm mới so với bài báo khác?**
- **Trả lời:** Mỗi thành phần đã có (CBAM, BiomedCLIP, Focal), nhưng em kết hợp chúng cho chấm mức độ
  cột sống: CBAM mở rộng sang 3D, ghép với nhánh ảnh–văn bản giữ cố định, đầu cô-sin để mở rộng nhãn,
  và kiểm chứng chéo trên bộ dữ liệu khác. Bài báo được MIWAI 2026 chấp nhận.

**Hỏi: Hạn chế?**
- **Trả lời:** Sót khoảng một nửa ca nặng nên chỉ là người đọc thứ hai; chưa dùng Axial nên lỗ liên hợp
  yếu; phần mềm chưa học lại từ chỉnh sửa. Ngoài ra (trong luận văn): chỉ 3 lần chạy; dữ liệu công khai
  nước ngoài, chưa thử ảnh Việt Nam.

**Hỏi: Áp dụng ở Việt Nam được không?**
- **Trả lời:** Chưa kiểm chứng. Máy chụp, thông số, quần thể bệnh nhân khác. Cần dữ liệu trong nước để
  đánh giá và có thể tinh chỉnh; phần mềm đã sẵn đường thu chỉnh sửa của bác sĩ để làm việc đó.

**Hỏi: Hướng phát triển nào làm trước?**
- **Trả lời:** Đánh giá cùng bác sĩ (2 tuần) trước, vì nó cho biết nhãn nào cần cải thiện, rồi mới học
  lại từ chỉnh sửa (4 tuần), thêm Axial (4 tuần), sinh báo cáo (6 tuần).

### Trang 32: Công bố

**Hỏi: Hội nghị uy tín không?**
- **Trả lời:** MIWAI lần thứ 19, kỷ yếu xuất bản trong Lecture Notes in Artificial Intelligence của
  Springer, được Scopus và EI chỉ mục. Năm 2026 nhận 75/200 bài. Em là tác giả thứ nhất, thầy Nhân là
  tác giả liên hệ.
- Chỉ nói "rank C" nếu đã tự kiểm tra trên portal.core.edu.au.

**Hỏi: Bài báo khác luận văn chỗ nào?**
- **Trả lời:** Bài báo tập trung vào kiến trúc Hybrid và zero-shot; luận văn thêm hai thí nghiệm chọn
  mô hình, phần mềm, và phân tích chi tiết hơn.

---

## Phần 8. Câu hỏi tổng quát, mẹo

**Hỏi: Nếu được làm lại, em sẽ làm gì khác?**
- **Trả lời gợi ý:** Chia SPIDER theo bệnh nhân; tách tập test riêng khỏi tập chọn checkpoint; thêm
  Axial cho lỗ liên hợp; chạy nhiều seed hơn; tích hợp đầy đủ Hybrid vào phần mềm.

**Hỏi: Em tự làm phần nào?**
- **Trả lời:** Toàn bộ: thiết kế kiến trúc, mã huấn luyện và đánh giá, chạy thí nghiệm trên GPU thuê,
  phần mềm, và viết bài báo. Mô hình dùng lại: TotalSpineSeg (nguyên bản), trọng số SpineNetV2 và
  BiomedCLIP (khởi tạo / giữ cố định).

**Hỏi: Có so với SOTA không?**
- **Trả lời:** Trong luận văn, mốc so sánh là SpineNetV2 (mô hình nền) và các cấu hình ablation, cùng
  một dữ liệu, cùng cách chia. Ning Shen chỉ ở thí nghiệm chọn mô hình. So sánh trực tiếp với các giải
  pháp cuộc thi trên bài toán 3 mức là việc bổ sung được.

**Mẹo trả lời**
- Không biết thì nói: "Phần này em chưa làm / chưa kiểm tra, em xin ghi nhận để bổ sung."
- Câu hỏi về số: mở đúng trang phụ lục (bảng tra trang ở cuối `1_SCRIPT_BAO_VE_LVTN.md`).
- Câu hỏi về công thức: mở trang 40 (công thức Hybrid) hoặc 45 (CBAM-3D), nói theo thứ tự trên slide.
- Đừng nói quá: "Grad-CAM chứng minh", "Hybrid thắng trên SPIDER", "giải nhất RSNA", "phần mềm chạy
  Hybrid", "có ý nghĩa thống kê".
