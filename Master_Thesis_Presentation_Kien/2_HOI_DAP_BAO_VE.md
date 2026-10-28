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
  chỗ: luận văn không đo trực tiếp trọng số chú ý, Grad-CAM chỉ là minh họa một ca (file 9). Bằng
  chứng em dùng là kết quả: thêm CBAM nâng Severe F1 từ 0.152 lên khoảng 0.26.

**Hỏi: Nếu kênh "nhiễu" bị nhân 0.05 thì có mất thông tin không?**
- **Trả lời:** Có giảm, và đó là mục đích: bớt nhiễu để tín hiệu tổn thương rõ hơn. Trọng số không
  về đúng 0 (sigmoid luôn lớn hơn 0), và mỗi ca tính lại một bộ trọng số, nên kênh bị giảm ở ca này
  vẫn có thể được giữ ở ca khác. Ngoài ra ResNet có đường tắt (residual) nên thông tin gốc vẫn đi qua
  các khối sau.

**Hỏi: Vì sao chọn CBAM mà không chọn SE-Net hay attention khác?**
- **Trả lời:** SE-Net chỉ có chú ý theo kênh, không chọn được vị trí. Tổn thương của em là một vùng
  nhỏ, nên cần chú ý không gian. CBAM có cả hai, rất nhẹ (cả 4 khối CBAM chỉ 46 nghìn tham số, chưa tới 0.1%), gắn được vào
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

**Hỏi: Ning Shen là ai, làm gì?** → Chi tiết ở file 9. Trả lời: "Một giải pháp công khai mạnh trên RSNA 2024,
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

**Số liệu cần thuộc** (đếm trực tiếp từ dữ liệu đã tiền xử lý và code chia tập):

| | RSNA 2024 (huấn luyện chính) | SPIDER (kiểm tra ngoài, tinh chỉnh) |
|---|---|---|
| Bệnh nhân | **1,973** | **218** |
| Đĩa đệm | **9,748** (khoảng 5 tầng mỗi người, L1/L2 → L5/S1) | **1,174** đĩa đệm code dùng khi tinh chỉnh |
| Chia tập | **Theo bệnh nhân**, 80/20: khoảng **1,578 train / 395 val** bệnh nhân. Seed 42: 7,806 / **1,942** đĩa đệm | 80/20 **theo đĩa đệm** |
| Mỗi seed | Chia lại khác (42, 123, 456), số đĩa đệm val xê dịch quanh 1,950 | Cùng một phép chia cho cả 4 cấu hình |
| Severe | Ống sống 468 (4.8%), lỗ liên hợp trái 397 (4.1%), phải 379 (3.9%): **khoảng 4%** | Trượt đốt sống 2.8%, thoát vị 4.7% |
| Normal/Mild | Ống sống 87.7%, trái 77.6%, phải 78.0%: trung bình **khoảng 81%** | |
| Nhãn thiếu | 10 đĩa thiếu nhãn trái, 40 thiếu nhãn phải (ghi −1, bỏ qua khi tính loss) | |

Thí nghiệm chọn mô hình (trang 14) dùng riêng **150 bệnh nhân** của tập val RSNA. Bộ Chu (chọn mô hình phân
vùng, trang 13) có **23 bệnh nhân**.

**Câu nói gọn:** "Dạ RSNA có 1,973 bệnh nhân, gần 9,750 đĩa đệm. Em chia theo bệnh nhân 80/20, khoảng 1,580
bệnh nhân để học, 395 để đánh giá, tức gần 1,950 đĩa đệm val, chạy 3 lần với 3 cách chia."

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
  từng được nhánh CBAM thấy. Val khoảng 395 bệnh nhân (file 9).

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

**Hỏi: Lúc huấn luyện và lúc chạy thật, phần văn bản đi thế nào? Có phải định nghĩa câu trước?**
- **Trả lời:** Đúng, định nghĩa trước. Ba bước:
  1. **Viết sẵn 9 câu** (3 nhãn × 3 mức) trong code (`RSNA_PROMPTS`), ví dụ "normal or mild spinal canal
     stenosis", "moderate …", "severe spinal canal stenosis".
  2. **Mã hóa một lần** trước khi học: mỗi câu qua tokenizer và bộ mã hóa văn bản của BiomedCLIP (đóng
     băng) ra vector 512 số, chuẩn hóa dài 1. Lưu thành bảng: mỗi nhãn một ma trận $[3, 512]$.
  3. **Mỗi bước sau đó chỉ ảnh đi qua mạng:** ảnh → hai nhánh → hợp nhất → vector ảnh $[B, 512]$. Nhân với
     bảng văn bản: $[B, 512] \times [512, 3] = [B, 3]$ cô-sin → nhân $s$ → softmax → xác suất 3 mức.
- Lúc chạy thật y hệt, chỉ bỏ phần tính loss: bảng văn bản tính một lần, ảnh mới so với 3 vector rồi
  argmax.
- **Vì sao tính trước được:** câu không đổi và bộ mã hóa văn bản không học, nên vector văn bản cố định.
  Tính lại mỗi bước chỉ tốn thời gian mà ra y hệt.

**Hỏi: Có vector ảnh rồi, tính cô-sin xong thì ra nhãn thế nào? Tính bao nhiêu lần?**
- **Trả lời:** Vector ảnh không so với một câu mà với **3 câu** (3 mức) của một nhãn, rồi chọn câu giống
  nhất. Ví dụ ống sống:

  | Bước | Làm gì | Ví dụ |
  |---|---|---|
  | 1 | Cô-sin với 3 câu Normal/Mild, Moderate, Severe | 0.20 / 0.25 / 0.31 |
  | 2 | Nhân $s \approx 14.3$ | 2.86 / 3.57 / 4.43 |
  | 3 | Softmax ra xác suất, tổng bằng 1 | 0.127 / 0.260 / 0.613 |
  | 4 | Argmax | **Severe** |

- **Hình dung:** các vector đều là mũi tên dài 1; cô-sin đo **góc** giữa hai mũi tên, góc càng nhỏ càng
  giống. Chọn câu có mũi tên cùng hướng nhất với ảnh.
- **Vì sao nhân $s$:** ba cô-sin sát nhau, softmax thẳng thì ra 0.32 / 0.33 / 0.35, gần chia đều. Nhân $s$
  để giãn ra; $s$ cũng được học.
- **Bao nhiêu lần:** mỗi đĩa đệm 3 nhãn × 3 mức = **9 cô-sin**, ra 3 mức dự đoán. Code không lặp từng cái:
  một phép nhân ma trận $[B, 512] \times [512, 3]$ tính cả lô với 3 câu cùng lúc, làm cho 3 nhãn.
- **Dùng vào đâu:** mức dự đoán (argmax) để tính Recall/Precision/F1; xác suất để tính AUC/AUPRC và, lúc
  học, đưa vào Focal Loss.

**Hỏi: Vậy vector văn bản đứng yên, cái gì học?**
- **Trả lời:** Phía ảnh học. Fusion MLP (cùng gộp lát, $s$) được chỉnh để **kéo vector ảnh về gần câu
  đúng** và đẩy xa hai câu còn lại. Có thể hiểu 3 vector văn bản là 3 "cột mốc" cố định có nghĩa, mô
  hình học cách đặt ảnh vào gần cột mốc đúng.

**Hỏi: Thế khác gì lớp tuyến tính bình thường?**
- **Trả lời:** Về phép tính thì giống: lớp tuyến tính cũng là nhân vector đặc trưng với ma trận trọng số
  $[512, 3]$. Khác ở chỗ ma trận đó **khởi tạo ngẫu nhiên và phải học**, còn ở đây ma trận là **vector của
  câu mô tả**, có sẵn ý nghĩa và không cần học. Nhờ vậy thêm nhãn mới chỉ cần viết câu mới, không phải
  thêm đầu ra và học lại.

**Hỏi: Vậy mô hình bị giới hạn trong các câu mô tả đã viết?**
- **Trả lời:** "Dạ đúng, mỗi lần chạy mô hình chỉ **chọn** trong các câu đã định nghĩa, giống trắc nghiệm,
  không tự sinh ra mức mới. Lớp tuyến tính cũng vậy, chỉ ra được số lớp cố định, nhưng muốn thêm lớp thì
  phải sửa mạng và học lại. Ở đây muốn thêm nhãn chỉ cần viết thêm câu, mã hóa một lần. Tuy nhiên chất
  lượng zero-shot còn phụ thuộc cách viết câu và việc BiomedCLIP có hiểu thuật ngữ đó không, nên với
  nhãn mới em vẫn tinh chỉnh thêm (thí nghiệm SPIDER: zero-shot chạy được, tinh chỉnh tốt hơn rõ)."

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

#### Giải thích nhanh cả slide (đọc trước các câu hỏi bên dưới)

**Bối cảnh:** khoảng 81% đĩa đệm là Normal/Mild, Severe chỉ khoảng 4%. Học bình thường thì mô hình cứ
đoán Normal là đúng phần lớn, nên bỏ qua ca nặng: SpineNetV2 chỉ bắt được 12.3% ca Severe. Mỗi kỹ thuật
dưới đây lo một phần.

**1. Focal Loss: bớt học ca dễ.**
- Nhân thêm $(1-p_y)^\gamma$ với $\gamma = 2$; $p_y$ là xác suất mô hình gán cho mức đúng.
- Ví dụ: ca dễ, mô hình đã chắc 90% ($p_y = 0.9$) → $0.1^2 = 0.01$, phạt giảm **100 lần**. Ca khó,
  chỉ chắc 30% → $0.7^2 = 0.49$, phạt chỉ giảm một nửa.
- Mục đích: hàng nghìn ca Normal dễ không lấn át nữa, mô hình dồn sức vào ca khó, mà ca khó phần lớn
  là Moderate và Severe.

**2. Trọng số lớp (căn bậc hai): sai ở lớp hiếm thì phạt nặng hơn.**
- $\omega \propto 1/\sqrt{n}$: Normal **0.38**, Moderate **0.89**, Severe **1.73**. Sai một ca Severe
  bị phạt gấp khoảng 4.5 lần sai một ca Normal.
- Không dùng $1/n$ vì chênh khoảng 20 lần, mô hình báo Severe tràn lan và huấn luyện mất ổn định.
- Khác Focal: Focal chọn theo **độ khó**, trọng số lớp chọn theo **lớp hiếm**. Hai cái bổ sung nhau.

**3. Oversampling: cho thấy ca hiếm nhiều lần hơn.**
- Đĩa đệm có ít nhất một nhãn Moderate hoặc Severe được lặp **3 lần** mỗi epoch.
- Hai kỹ thuật trên chỉnh **mức phạt**; kỹ thuật này tăng **số lần nhìn thấy**.

**4. Augmentation: làm các bản lặp không giống hệt.**
- Lật trước–sau, xoay ±10°, độ sáng/tương phản ±20%, thêm nhiễu.
- Lặp nguyên một ca Severe 3 lần thì mô hình dễ học thuộc tấm ảnh đó; biến đổi nhẹ buộc nó học dấu
  hiệu bệnh thật.
- Không lật trái–phải, vì lỗ liên hợp trái sẽ thành phải mà nhãn không đổi theo.

**5. Uncertainty weighting: cân bằng giữa 3 nhãn bệnh, không phải giữa các lớp.**
- Ba bệnh khó dễ khác nhau: ống sống dễ hơn, lỗ liên hợp khó hơn (nhìn rõ nhất trên mặt cắt ngang mà
  mô hình không dùng). Cộng đều ba hàm mất mát thì bệnh có mất mát lớn kéo cả mô hình.
- $\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2}\mathcal{L}_k + \log\sigma_k$, mỗi bệnh một $\sigma_k$ tự
  học. Mất mát lớn → $\sigma_k$ tăng → trọng số giảm. Số hạng $\log\sigma_k$ chặn không cho $\sigma$
  tăng mãi, nếu không mô hình "bỏ" luôn bệnh đó.
- Ví dụ: $\mathcal{L}_{canal} = 0.2$, $\mathcal{L}_{foraminal} = 0.8$. Tối ưu tại $\sigma^2 = \mathcal{L}$
  → trọng số 2.5 và 0.625 → nhân với mất mát đều ra 0.5: mỗi bệnh **đóng góp ngang nhau**.
- Lợi ích: không phải dò tay trọng số cho 3 bệnh.

**Kết quả cả bộ kỹ thuật mang lại:**

| Chỉ số | SpineNetV2 | Hybrid |
|---|---|---|
| Severe Recall | 0.123 | **0.486** (gấp khoảng 4 lần) |
| Severe F1 | 0.152 | **0.356** |
| Mean Recall | 0.412 | **0.592** |
| Accuracy | 0.810 | 0.724 ⬇ |

- Accuracy giảm là cái giá chấp nhận được: SpineNetV2 cao nhờ hầu như luôn đoán Normal. Hybrid chịu
  đoán Moderate và Severe nên bắt được ca nặng.
- Mặt trái: Severe Precision khoảng 0.27, báo nhầm ca nặng khá nhiều. Với vai trò người đọc thứ hai thì
  chấp nhận được: báo nhầm thì bác sĩ gạt đi, bỏ sót mới nguy hiểm.

**⚠️ Nói thật nếu bị hỏi:** luận văn **không tách riêng** đóng góp của từng kỹ thuật. Các con số trên là
hiệu quả của **cả bộ kỹ thuật cộng với kiến trúc**. Câu nên nói: *"Mỗi kỹ thuật lo một việc khác nhau;
tách riêng từng cái là phần em có thể bổ sung."*

#### Các câu hỏi

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

**Hỏi: Bài dùng những hàm loss nào để huấn luyện?** (đã kiểm trong code và log huấn luyện)

| Mô hình | Hàm loss |
|---|---|
| **Hybrid** (chính) | Focal Loss ($\gamma = 2$, trọng số lớp căn bậc hai) + 0.1 × SupCon, cho từng nhãn bệnh; 3 nhãn gộp bằng Uncertainty weighting |
| BMC-only | Giống Hybrid (cùng code, chỉ tắt nhánh CBAM) |
| CBAM (bước 1, cũng là nhánh CBAM của Hybrid) | Focal Loss + trọng số căn bậc hai + Uncertainty weighting; không có SupCon (đầu tuyến tính) |
| SpineNetV2 (mô hình nền) | Cross-entropy thường, không trọng số |
| Hybrid fine-tune trên SPIDER | Cross-entropy có trọng số lớp |

- **Công thức tổng của Hybrid:**
  $\mathcal{L} = \sum_{k=1}^{3}\Big[\frac{1}{2\sigma_k^2}\big(\text{FL}_k + 0.1\,\text{SupCon}_k\big) + \log\sigma_k\Big]$,
  với $k$ là ống sống, lỗ liên hợp trái, lỗ liên hợp phải.
- **Vai trò từng cái:**
  - **Focal Loss:** loss phân loại chính. Bớt học ca dễ, kèm trọng số lớp để phạt nặng khi sai lớp hiếm.
  - **SupCon** (supervised contrastive): trong mỗi lô, kéo vector ảnh của các ca **cùng mức** lại gần
    nhau và đẩy ca **khác mức** ra xa. Nó giúp không gian vector ảnh gọn theo mức độ, nên so cô-sin với
    câu mô tả dễ hơn. Hệ số chỉ 0.1 vì mỗi lô có rất ít ca Severe, SupCon trên lớp hiếm dễ dao động; nó
    là loss phụ, không phải loss chính.
  - **Uncertainty weighting:** không phải một loss riêng, mà là cách **gộp** 3 loss của 3 nhãn bệnh
    với trọng số tự học.
- **Trọng số lớp thật trong log** (seed 123): Normal 0.391, Moderate 0.922, Severe 1.686. Con số 0.38 /
  0.89 / 1.73 trong file công thức là tính tay từ tỉ lệ làm tròn 81% / 15% / 4%, nên cùng bậc.
- **⚠️ Chi tiết nhỏ nếu bị hỏi sâu:** log của cấu hình CBAM (bước 1) ghi $\gamma = 1.8$ và lặp mẫu
  hiếm **5 lần**. Luận văn chỉ nêu $\gamma = 2$ và lặp 3 lần **cho cấu hình Hybrid** (chương 4 ghi rõ
  "cho cấu hình Hybrid"), và log Hybrid khớp đúng như vậy. Nếu bị hỏi: "Bước 1 huấn luyện nhánh CBAM
  dùng $\gamma = 1.8$ và lặp 5 lần; bước 2 Hybrid dùng $\gamma = 2$ và lặp 3 lần."

**Hỏi: Những gì được học ở bước 2?**
- **Trả lời:** $w_a$ (gộp lát), Fusion MLP, hệ số $s$, 3 tham số $\sigma_k$. Tổng khoảng 1.18 triệu.
  CBAM-ResNet 3D 63.5 triệu và bộ mã hóa ảnh BiomedCLIP 86.2 triệu giữ cố định.

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
  ra cho 8 nhãn SPIDER. 1.174 đĩa đệm, chia 80/20, tối đa 15 epoch, 3 seed. (Chia theo đĩa đệm, file 9.)

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

**Hỏi: Pfirrmann đâu?** → Chi tiết ở file 9.

---

## Phần 6. Phần mềm (trang 26–28)

**Hỏi: Kiến trúc phần mềm?**
- **Trả lời:** Giao diện React với Cornerstone3D (thư viện hiển thị ảnh y tế trên trình duyệt); máy
  chủ FastAPI giữ cơ sở dữ liệu và gọi mô hình; ảnh và mặt nạ lưu trên ổ đĩa. TotalSpineSeg cài môi
  trường Python riêng, gọi qua dòng lệnh, vì hai mô hình cần phiên bản thư viện khác nhau.

**Hỏi: Phần mềm dùng mô hình nào?** → Chi tiết ở file 9. **Không nói "Hybrid".**

**Hỏi: 11 nhãn gồm những gì?**
- **Trả lời:** 3 nhãn từ mô hình của em: hẹp ống sống, hẹp lỗ liên hợp trái, phải (3 mức RSNA). 8 nhãn
  từ SpineNetV2 gốc: Pfirrmann, hẹp đĩa đệm, trượt đốt sống, tổn thương mặt trên, mặt dưới đốt sống,
  biến đổi tủy xương trên, dưới, thoát vị. Bảng 8 tầng từ T10–T11 tới L5–S1; 3 nhãn RSNA để trống ở
  tầng ngực vì dữ liệu huấn luyện chỉ có từ L1–L2.

**Hỏi: Trong phần mềm, một ca đi qua các mô hình thế nào?**
- **Trả lời:** Hai đường chạy song song trên cùng khối ảnh sagittal T2, rồi ghép lại:
  1. **TotalSpineSeg** phân vùng cả khối, ra một bản đồ nhãn: mỗi điểm ảnh mang số hiệu của cấu trúc nó
     thuộc về (tủy sống, ống sống, từng đốt C1 → L5, xương cùng, từng đĩa đệm).
  2. **Tìm đĩa đệm:** lấy 5 nhãn đĩa thắt lưng (L1-L2 … L5-S1). Mỗi đĩa lấy **tâm** = trung bình tọa độ
     các điểm ảnh của đĩa đó, ra (lát, y, x).
  3. **Cắt khối** quanh tâm **giống hệt tiền xử lý RSNA**: 9 lát quanh lát tâm, khung 240 × 120 quanh
     (x, y), đổi về 224 × 112, chuẩn hóa phân vị 1–99% về 0–1 → khối `(9, 112, 224)`.
  4. **Mô hình của em** chấm khối đó, ra 3 nhãn: ống sống, lỗ liên hợp trái, phải.
  5. **Song song, SpineNetV2 gốc** chạy nguyên quy trình của nó trên cả khối (tự phát hiện đốt sống, gán
     tầng, cắt IVV, chấm), ra 8 nhãn còn lại. Ba nhãn ống sống/lỗ liên hợp của SpineNetV2 bị **bỏ**, thay
     bằng của mô hình em.
  6. **Ghép** thành bảng 11 nhãn theo từng tầng, hiện trên giao diện cùng mặt nạ phân vùng.

**Hỏi: TotalSpineSeg bên trong làm gì, cho ra gì?**
- **Trả lời:** Hai mạng nnU-Net nối tiếp (xem Phần 3). Đầu ra là **một ảnh nhãn cùng kích thước ảnh gốc**,
  mỗi điểm ảnh là một số: ví dụ 1 tủy sống, 2 ống sống, 41–45 đốt L1–L5, 50 xương cùng, các số 9x là đĩa
  đệm. App dùng nó vào hai việc: **hiển thị** mặt nạ cho bác sĩ xem, và **định vị đĩa đệm** để cắt khối
  cho mô hình chấm nhãn. Em dùng nguyên bản, không huấn luyện lại.

**Hỏi: SpineNetV2 bên trong làm gì?**
- **Trả lời:** Tóm gọn 3 bước (chi tiết Phần 7d): **phát hiện** thân đốt sống (U-Net dự đoán góc và tâm),
  **gán tầng** cho từng đốt (mạng ngoại hình + mạng ngữ cảnh + beam search), rồi **cắt IVV** quanh từng đĩa
  (xoay theo đốt sống, co giãn theo cỡ đốt) và **chấm** bằng ResNet-34 3D, mỗi nhãn một đầu ra. Trong app
  em gọi đúng ba hàm của thư viện: phát hiện đốt sống → lấy IVV → chấm.

**Hỏi: Mô hình của em bên trong làm gì?**
- **Trả lời:** Nhận khối `(9, 112, 224)`, qua ResNet-34 3D có CBAM ở 4 tầng: mỗi tầng CBAM chọn **kênh**
  (loại đặc trưng) rồi **vị trí** quan trọng; cuối cùng gộp trung bình ra vector 512 số; 3 đầu ra cho 3
  nhãn, mỗi đầu ra softmax 3 mức. Thân mạng khởi tạo từ trọng số SpineNetV2, học thêm trên RSNA với
  Focal Loss, trọng số lớp và Uncertainty weighting.

**Hỏi: Sao SpineNetV2 không dùng luôn vị trí đĩa của TotalSpineSeg?**
- **Trả lời:** Cách cắt IVV của SpineNetV2 cần các đốt sống do chính nó phát hiện (để xoay và co giãn theo
  cỡ đốt). Em giữ nguyên quy trình gốc để 8 nhãn đó chạy đúng như nhóm tác giả đã kiểm chứng. Còn mô hình
  của em học trên khối cắt kiểu RSNA (không xoay), nên dùng tâm đĩa từ TotalSpineSeg rồi cắt đúng kiểu đó.

**Hỏi: TotalSpineSeg phân vùng sai thì sao?**
- **Trả lời:** Nếu sai vị trí đĩa thì khối cắt lệch và nhãn chấm sai theo. Giao diện hiện mặt nạ và khung
  cắt từng đĩa nên bác sĩ thấy được, và sửa được nhãn. Em chưa đánh giá toàn chuỗi (phân vùng rồi chấm)
  trên dữ liệu có nhãn; đó là việc tiếp theo.

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

## Phần 7b. Câu hỏi sâu về phương pháp ("tại sao làm vậy, sao không làm khác")

Kiểu câu hội đồng hay hỏi để thử xem em có hiểu lựa chọn của mình không. Cách trả lời chung: **nói lý do
→ thừa nhận cái giá → nêu hướng cải thiện**. Không bịa số cho thứ chưa làm.

**Hỏi: Ba mức Normal < Moderate < Severe có thứ bậc, sao em coi như ba lớp độc lập?**
- **Trả lời:** Đúng, em dùng phân loại thường (softmax 3 lớp), nên đoán Severe thành Moderate bị phạt
  ngang với đoán Severe thành Normal, dù sai một bậc thì nhẹ hơn. Em giữ cách này để so công bằng với
  SpineNetV2 và cách gán nhãn của RSNA, và vì đầu cô-sin cần mỗi mức một câu mô tả riêng. Hướng cải
  thiện: dùng hàm mất mát có thứ bậc (ordinal loss) để sai càng xa thì phạt càng nặng.

**Hỏi: Đĩa đệm bị thiếu nhãn thì xử lý sao?**
- **Trả lời:** Nhãn thiếu được gán −1. Hàm loss bỏ qua các ô −1 (`ignore_index`), trọng số lớp và các
  chỉ số đánh giá cũng không tính chúng. Nếu cả lô không có nhãn nào cho một bệnh thì bệnh đó nghỉ lô
  đó. Em không đoán bù nhãn.

**Hỏi: Ba bệnh dùng chung một mô hình hay ba mô hình riêng? Vì sao?**
- **Trả lời:** Một mô hình chung, ba đầu ra (học đa nhiệm). Ba bệnh nhìn cùng một khối ảnh và liên
  quan nhau: đĩa đệm lồi có thể gây hẹp cả ống sống lẫn lỗ liên hợp, nên dùng chung đặc trưng thì
  nhãn bệnh này giúp nhãn bệnh kia, lại rẻ hơn ba lần. Cái giá là bệnh khó có thể lấn át bệnh dễ khi
  học, nên em dùng uncertainty weighting để cân.

**Hỏi: Sao chấm bằng argmax mà không chỉnh ngưỡng riêng cho Severe?**
- **Trả lời:** Argmax không cần thêm dữ liệu để chỉnh và giữ so sánh công bằng giữa bốn cấu hình. Nếu
  chỉnh ngưỡng trên tập val rồi báo cáo trên chính tập val thì kết quả sẽ đẹp giả. Muốn chỉnh ngưỡng
  đúng cách cần thêm một tập riêng. Bù lại, bảng đã có AUC và AUPRC, là các chỉ số không phụ thuộc
  ngưỡng. Trong thực tế, bệnh viện có thể chọn ngưỡng theo nhu cầu: hạ ngưỡng để bắt thêm ca nặng, chấp
  nhận báo nhầm nhiều hơn.

**Hỏi: Sao không dùng chỉ số chính thức của cuộc thi RSNA?**
- **Trả lời:** Cuộc thi chấm bằng log loss có trọng số (1 / 2 / 4 cho Normal / Moderate / Severe) trên
  cả ca chụp, đủ 5 nhãn × 5 tầng, trên tập test ẩn. Em dùng 3 trên 5 nhãn và chia tập riêng, nên số của
  em không so được với bảng xếp hạng. Em chọn Recall và F1 vì mục tiêu là bắt ca nặng. Log loss đo xác
  suất có "chuẩn" hay không (calibration); đó là chỉ số nên báo cáo thêm.

**Hỏi: Dùng cùng lúc Focal, trọng số lớp và lặp mẫu, có bị bù quá tay không?**
- **Trả lời:** Có nguy cơ, và đã có dấu hiệu: Severe Precision khoảng 0.27, tức mô hình báo nặng khá
  nhiều. Em đã giảm mức bù: trọng số lấy căn bậc hai thay vì $1/n$, bước Hybrid lặp mẫu 3 lần thay vì
  5. Mức còn lại là chấp nhận được với vai trò người đọc thứ hai, nơi bắt được ca nặng quan trọng hơn.
  Em chưa có thí nghiệm tách riêng từng kỹ thuật.

**Hỏi: Tập val có bị lặp mẫu hay biến đổi ảnh không? Số đo có bị "đẹp giả" không?**
- **Trả lời:** Không. Chỉ tập train được lặp và biến đổi. Tập val giữ nguyên phân bố thật, khoảng 4% là
  Severe (seed 42 có 1,942 đĩa đệm), nên số đo phản ánh đúng tỉ lệ thật. Trọng số lớp cũng tính từ tập
  train gốc, trước khi lặp.

**Hỏi: Siêu tham số chọn thế nào? Có dò lưới (grid search) không?**
- **Trả lời:** Phần lớn lấy mặc định của các bài gốc: $\gamma = 2$ của Focal Loss (Lin 2017), $s$ khởi
  tạo $1/0.07$ của CLIP, tỉ lệ thu gọn 16 và kernel 7 của CBAM, AdamW lr $10^{-4}$. Một vài giá trị như
  hệ số SupCon 0.1 và số lần lặp mẫu được chọn qua vài lần thử. Em chưa dò lưới kỹ vì chi phí GPU, nên
  kết quả có thể còn tăng nếu dò thêm. Không có giá trị nào được chọn dựa trên tập test.

**Hỏi: Vì sao cắt riêng từng đĩa đệm, không đưa cả ảnh cột sống vào?**
- **Trả lời:** Nhãn RSNA cho theo từng tầng đĩa đệm. Cắt khối quanh đĩa giúp mô hình nhìn đúng chỗ, đầu
  vào nhỏ, và mỗi ca chụp cho 5 mẫu (5 tầng). Đây cũng là cách SpineNetV2 làm. Cái giá là mất ngữ cảnh
  các tầng lân cận.

**Hỏi: RSNA và SPIDER khác máy chụp, khác bệnh viện, em xử lý lệch dữ liệu thế nào?**
- **Trả lời:** Mọi khối ảnh đều đưa về cùng kích thước $9 \times 112 \times 224$ và cường độ về 0–1, và
  khi học có biến đổi độ sáng, tương phản. Em không dùng kỹ thuật thích nghi miền chuyên biệt. Kết quả
  đo được: zero-shot đạt 0.362, có tinh chỉnh đạt 0.653, ngang SpineNetV2. Trên dữ liệu Việt Nam thì
  chưa thử.

**Hỏi: Nhãn RSNA có nhiễu không? Bác sĩ chấm có thống nhất không?**
- **Trả lời:** Ranh giới Moderate với Severe vốn dao động giữa các bác sĩ, và RSNA không công bố độ đồng
  thuận giữa người chấm. Vì vậy chỉ số Severe có một mức trần do chính nhiễu nhãn. Em không làm sạch nhãn
  mà dùng nguyên nhãn của bộ dữ liệu.

**Hỏi: Vì sao khởi tạo từ trọng số SpineNetV2, không học từ đầu?**
- **Trả lời:** Ca Severe quá ít để học một mạng 3D từ đầu. Khởi tạo từ mạng đã học trên MRI cột sống giúp
  hội tụ nhanh và ổn định hơn. CBAM gắn thêm không đổi kích thước đặc trưng nên vẫn nạp được trọng số
  cũ.

**Hỏi: Nhãn RSNA cố định, dùng đầu tuyến tính là đủ, sao phải dùng đầu cô-sin?**
- **Trả lời:** Để cùng một mô hình mở rộng được sang nhãn mới (SPIDER, zero-shot), đúng với mục tiêu 3 của
  luận văn. Trên RSNA, dùng đầu cô-sin không làm mô hình kém đi: Hybrid vẫn hơn SpineNetV2, vốn dùng đầu
  tuyến tính.

**Hỏi: Sao hệ số $s$ phải học, không cố định?**
- **Trả lời:** Cô-sin chỉ nằm trong khoảng −1 đến 1, khoảng cách giữa các mức rất nhỏ, nên cần $s$ để
  softmax phân biệt rõ. Cho mô hình tự học $s$ thì nó tự chọn mức "tự tin" phù hợp. Khởi tạo như CLIP và
  chặn tối đa 100 để huấn luyện không mất ổn định.

**Hỏi: Kết quả có lặp lại được không?**
- **Trả lời:** Code nằm trong repo, seed cố định (42, 123, 456) cho cả cách chia tập lẫn khởi tạo; mỗi
  seed chia bệnh nhân khác nhau. Trọng số BiomedCLIP công khai, log huấn luyện lưu đủ. Chạy trên GPU khác
  có thể lệch nhỏ.

**Hỏi: Sao không dùng Transformer 3D hay mô hình lớn hơn?**
- **Trả lời:** Dữ liệu khoảng 2,000 bệnh nhân, Severe chỉ khoảng 4%, nên mô hình lớn rất dễ học thuộc.
  ResNet-34 3D đã có sẵn trọng số cột sống. Phần Transformer (ViT của BiomedCLIP) thì em có dùng, nhưng
  giữ cố định. Thêm nữa là giới hạn phần cứng, chỉ một GPU RTX 4090.

**Hỏi: Có kiểm định thống kê cho chênh lệch giữa các cấu hình không?**
- **Trả lời:** Trong luận văn em báo cáo trung bình ± độ lệch chuẩn của 3 lần chạy. Bài báo MIWAI có thêm
  kiểm định t ghép cặp trên SPIDER: Hybrid hơn CBAM với $p = 0.011$. Với 3 lần chạy, p-value chỉ mang tính
  tham khảo (xem câu "3 lần chạy có đủ ý nghĩa thống kê?" ở Trang 21).

---

## Phần 7c. Thầy hỏi sâu vào xử lý ảnh, code, numpy

Bạn cùng lớp kể thầy "hỏi code với lý thuyết căng", "numpy hỏi tới bến", "cái gì nghĩ mình chưa hiểu
là hỏi hết". Phần này đi theo **đúng những gì code làm** (đã đọc `rsna_dataloader.py`,
`spider_dataloader.py`, `biomedclip_wrapper.py`, và chạy thử mô hình để in kích thước tensor).

### Bài em có phải xử lý ảnh không?

- **Trả lời:** Có. Bài thuộc **phân tích ảnh y khoa**, nhánh thị giác máy tính dùng học sâu. Đầu vào là
  ảnh MRI; pipeline có các bước xử lý ảnh thật: đọc DICOM, sắp lát, cắt vùng, nội suy đổi kích thước,
  chuẩn hóa cường độ, phân vùng (TotalSpineSeg), tăng cường ảnh. Phần đóng góp chính là mô hình học sâu
  phân loại, không phải bộ lọc cổ điển (lọc nhiễu, hình thái học), nên cần nắm chắc cả hai phía: **các
  bước tiền xử lý ảnh** và **bên trong mạng**.

### A. Ảnh đi từ file DICOM tới tensor thế nào? (RSNA, `rsna_dataloader.py`)

| Bước | Code làm gì | Ra gì |
|---|---|---|
| 1. Chọn chuỗi ảnh | Lọc theo mô tả chuỗi: có "t2" và "sag", không có "ax", "cor" | Chuỗi Sagittal T2 |
| 2. Đọc và sắp lát | `pydicom` đọc từng file; sắp theo `ImagePositionPatient[2]` (dự phòng: `InstanceNumber`) | Danh sách lát đúng thứ tự |
| 3. Chọn 9 lát | Lấy lát có số thứ tự đúng tọa độ nhãn làm tâm, lấy tâm −4 … tâm +4; tràn biên thì lặp lát ngoài cùng | 9 lát |
| 4. Cắt vùng | Mỗi lát cắt khung **240 × 120 pixel** quanh tọa độ $(x, y)$ của nhãn | 9 ảnh 120 × 240 |
| 5. Đổi kích thước | `cv2.resize(..., (224, 112), INTER_LINEAR)`: nội suy song tuyến, thu nhỏ 0.93 lần | 9 ảnh 112 × 224 |
| 6. Xếp chồng | `np.stack` | Mảng `(9, 112, 224)` float32 |
| 7. Chuẩn hóa | Lấy phân vị 1% và 99% **trên cả khối**, cắt ngoài khoảng đó (`np.clip`), rồi đưa về 0–1 | Giá trị trong [0, 1] |
| 8. Lưu | `np.save` thành `.npy`, khoảng 8 GB cho cả tập | Đọc lại nhanh khi huấn luyện |
| 9. Vào mạng | `np.load` → tensor → thêm chiều kênh và chiều lô | `[B, 1, 9, 112, 224]` |

**Hỏi: Sao chuẩn hóa bằng phân vị 1–99%, không min-max hay z-score?**
- **Trả lời:** Cường độ MRI không có đơn vị tuyệt đối như CT (Hounsfield), mỗi máy, mỗi lần chụp một
  thang khác, nên phải chuẩn hóa theo từng khối. Min-max thì một điểm quá sáng (mỡ, nhiễu, artefact)
  cũng kéo cả ảnh tối đi. Cắt 1% hai đầu bỏ các điểm ngoại lai đó trước, rồi mới co về 0–1. Z-score
  không giới hạn khoảng giá trị, còn em cần đầu vào 0–1 cho cả nhánh BiomedCLIP.

**Hỏi: Sao nội suy song tuyến (bilinear), không lân cận gần nhất (nearest)?**
- **Trả lời:** Với ảnh cường độ, bilinear cho ảnh mượt, không bị răng cưa khối. Nearest chỉ dùng cho
  **mặt nạ nhãn** (giá trị là số nhãn, không được trộn). Ở đây chỉ thu nhỏ 0.93 lần nên mất mát rất ít.

**Hỏi: Cắt theo pixel, vậy máy có độ phân giải khác nhau thì vùng cắt khác nhau à?** *(câu dễ bị bắt)*
- **Trả lời thật:** Đúng. RSNA cắt theo **pixel** (240 × 120), không theo **milimét**, nên ca có khoảng
  cách pixel khác nhau thì vùng cắt thật to nhỏ khác nhau một chút. Với SPIDER, em cắt theo **milimét**
  (120 × 60 mm, đọc khoảng cách pixel từ file) để khớp với RSNA, với giả định RSNA khoảng 0.5 mm/pixel.
  Cắt theo milimét cho cả RSNA là cải tiến nên làm.

**Hỏi: Đĩa đệm sát mép ảnh thì sao?**
- **Trả lời:** Khung cắt bị kẹp ở mép nên nhỏ hơn 240 × 120, rồi bị kéo giãn về 112 × 224, tức méo nhẹ.
  Theo chiều lát, nếu thiếu lát thì lặp lát ngoài cùng. Trường hợp này hiếm vì đĩa đệm thắt lưng nằm
  giữa ảnh.

**Hỏi: 9 lát này phủ bao nhiêu milimét theo chiều trái–phải?**
- **Trả lời:** Tùy độ dày lát của từng máy (lát dọc thường khoảng 3–4 mm), nên 9 lát phủ khoảng 3 cm,
  không cố định giữa các ca. Code không nội suy lại theo milimét theo chiều này.

### A2. Numpy: câu hỏi ngắn

**Câu nói gọn nếu thầy hỏi "numpy em dùng thế nào":** "Dạ numpy em dùng ở bước tiền xử lý. Mỗi đĩa đệm
em lấy 9 lát cắt dọc, mỗi lát cắt vùng quanh đĩa rồi resize về 112 × 224, xếp chồng thành mảng 3D
(9, 112, 224) float32, chuẩn hóa phân vị 1–99% trên cả khối rồi đưa về 0–1, lưu .npy. Lúc huấn luyện load
lên, đổi sang tensor, thêm chiều kênh và chiều lô thành [B, 1, 9, 112, 224]."

**Hỏi: (9, 112, 224) nghĩa là gì?**
- **Trả lời:** Mảng 3 chiều, đọc là (số lát, cao, rộng):
  - **9**: số lát cắt dọc (sagittal), đi theo hướng **trái → phải** của bệnh nhân, lát giữa nằm đúng vị trí nhãn.
  - **112**: chiều cao mỗi lát, hướng **trên → dưới** dọc cột sống, đủ chứa đĩa đệm và một phần hai thân đốt kề.
  - **224**: chiều rộng mỗi lát, hướng **trước → sau**, từ thân đốt qua ống sống tới phía sau.
  - Mỗi phần tử là cường độ sáng của một điểm ảnh, đã đưa về 0–1. Ví dụ `vol[4, 56, 112]` là điểm
    chính giữa của lát giữa.

**Hỏi: Các lệnh numpy chính?**

| Việc | Lệnh | Ý nghĩa |
|---|---|---|
| Cắt vùng | `img[y0:y1, x0:x1]` (cắt lát mảng) | Lấy khung 120 × 240 quanh tọa độ nhãn |
| Xếp 9 lát | `np.stack(slices)` | 9 ảnh 2D → mảng `(9, 112, 224)` |
| Lấy ngưỡng | `np.percentile(vol, [1, 99])` | Bỏ 1% điểm quá tối và 1% quá sáng (nhiễu) |
| Kẹp giá trị | `np.clip(vol, p1, p99)` | Giá trị ngoài khoảng bị kéo về biên |
| Đưa về 0–1 | `(vol - p1) / (p99 - p1)` | Tính trên cả mảng một lần, không cần vòng lặp |
| Lưu, đọc | `np.save` / `np.load` | Khỏi đọc lại DICOM mỗi lần huấn luyện |

**Hỏi: Vì sao là mảng 3D?** Vì mạng tích chập 3D cần nhìn cả quan hệ giữa các lát liền nhau (ví dụ đĩa
lồi kéo dài qua vài lát), không chỉ một ảnh phẳng.

**Hỏi: Vì sao float32 mà không float64?** Đủ chính xác cho ảnh, nhẹ bằng một nửa, GPU tính float32 nhanh
hơn, và trọng số mạng cũng là float32.

**Hỏi: Numpy khác tensor PyTorch chỗ nào?** Numpy chạy trên CPU, không tính đạo hàm. Tensor chạy được
trên GPU và có autograd để lan truyền ngược. Nên numpy dùng cho tiền xử lý, tensor dùng cho mô hình.
Đổi qua lại bằng `torch.from_numpy(vol)` và `tensor.numpy()`.

**Hỏi: Thêm chiều kênh và chiều lô là sao?** `Conv3d` chờ đầu vào 5 chiều [lô, kênh, sâu, cao, rộng].
Ảnh MRI là ảnh xám nên 1 kênh: `(9, 112, 224)` → `unsqueeze(0)` → `(1, 9, 112, 224)`; DataLoader
gộp B khối thành `[B, 1, 9, 112, 224]`, với B = 32.

### B. Kích thước tensor qua từng tầng (chạy thử thật với một khối)

| Vị trí | Kích thước `[B, C, D, H, W]` | Ghi chú |
|---|---|---|
| Đầu vào | `[1, 1, 9, 112, 224]` | 1 kênh xám, 9 lát |
| conv1 (kernel 3×7×7, stride (1,2,2)) | `[1, 64, 9, 56, 112]` | Stride chỉ trên H, W; **chiều lát giữ 9** |
| maxpool (stride (1,2,2)) | `[1, 64, 9, 28, 56]` | |
| layer1 + CBAM | `[1, 64, 9, 28, 56]` | CBAM không đổi kích thước |
| layer2 + CBAM | `[1, 128, 9, 28, 56]` | |
| layer3 + CBAM | `[1, 256, 9, 14, 28]` | |
| layer4 + CBAM | `[1, 512, 9, 7, 14]` | |
| GAP (`AdaptiveAvgPool3d(1)`) | `[1, 512, 1, 1, 1]` → $f_{cbam}$ 512 | Trung bình mỗi kênh trên 882 vị trí |

**Hỏi: Sao không giảm chiều lát (stride theo D)?**
- **Trả lời:** Chỉ có 9 lát, giảm nữa thì mất thông tin trái–phải. Đây là thiết kế của SpineNetV2:
  stride chỉ trên chiều cao và rộng, `stride=(1, s, s)` trong code.

**Hỏi: Một khối ảnh tốn bao nhiêu bộ nhớ?**
- **Trả lời:** $9 \times 112 \times 224 = 225{,}792$ số × 4 byte (float32) ≈ **0.86 MB**. Một lô 32 khối
  ≈ 27.6 MB cho đầu vào; phần lớn bộ nhớ GPU là đặc trưng trung gian của các tầng.

### C. Số tham số: tính tay được

- conv1: $64 \times 1 \times 3 \times 7 \times 7 = $ **9,408**.
- Một conv $3 \times 3 \times 3$ từ 64 lên 64 kênh: $64 \times 64 \times 27 = $ **110,592**. Bản 2D
  ($3 \times 3$) chỉ 36,864, tức **3D tốn khoảng 3 lần** vì kernel có thêm chiều sâu 3.
- **Toàn nhánh CBAM-3D ResNet-34: 63.5 triệu** (code đếm 63,507,969; log SpineNetV2 ghi 63,461,705
  khi chưa có CBAM).
- **CBAM cả 4 tầng: 46,264 tham số, chưa tới 0.1%.**
- Tổng mô hình Hybrid khoảng **260.6 triệu** (log), trong đó học **1,181,442**.

### D. Lý thuyết bên trong mạng, câu ngắn

- **Tích chập 3D khác 2D:** kernel trượt cả theo chiều lát, nên đặc trưng nhìn được quan hệ giữa các lát
  liền nhau (ví dụ đĩa lồi kéo dài qua vài lát).
- **BatchNorm:** chuẩn hóa đầu ra mỗi kênh theo trung bình và phương sai của lô, giúp học ổn định, cho
  phép tốc độ học lớn hơn. Lúc suy luận dùng trung bình tích lũy, không dùng của lô.
- **Residual (đường tắt):** cộng đầu vào vào đầu ra của khối, $y = F(x) + x$, để đạo hàm truyền qua mạng
  sâu không bị tắt dần.
- **GAP:** lấy trung bình mỗi kênh trên toàn khối, không cần tham số, không phụ thuộc kích thước ảnh.
- **Sigmoid trong CBAM, softmax ở đầu ra:** CBAM cần mỗi kênh một trọng số **độc lập** từ 0 đến 1, các
  kênh không cần cộng lại bằng 1, nên dùng sigmoid. Đầu ra chọn **một** mức trong ba, nên dùng softmax.
- **Vì sao đóng băng thì nhanh:** không cần tính đạo hàm và lưu đặc trưng trung gian cho hai nhánh lớn.
  Code chạy nhánh CBAM trong `torch.no_grad()`.

### D2. ResNet-34 là gì (câu nền tảng)

**Hỏi: ResNet là gì?**
- **Trả lời:** Mạng tích chập có **đường tắt (residual)**: mỗi khối học phần chênh lệch $F(x)$ rồi cộng
  lại với đầu vào, $y = F(x) + x$. Trước ResNet, mạng càng sâu càng khó học vì đạo hàm tắt dần; đường tắt
  cho đạo hàm đi thẳng về các tầng đầu, nên học được mạng rất sâu (He và cộng sự, 2016).

**Hỏi: Số 34 nghĩa là gì?**
- **Trả lời:** 34 tầng có trọng số: 1 tích chập đầu + 16 khối × 2 tích chập = 32, + 1 tầng đầu ra = 34.
  16 khối chia thành 4 tầng **[3, 4, 6, 3]**, số kênh 64 → 128 → 256 → 512.

**Hỏi: Một khối BasicBlock gồm gì?**
- **Trả lời:** Tích chập 3×3×3 → BatchNorm → ReLU → tích chập 3×3×3 → BatchNorm → cộng đường tắt → ReLU.
  Khi số kênh hoặc kích thước thay đổi, đường tắt đi qua một tích chập 1×1×1 để khớp kích thước rồi mới cộng.

**Hỏi: Sao chọn ResNet-34, không phải 18 hay 50?**
- **Trả lời:** Vì SpineNetV2 dùng đúng ResNet-34 3D và đã có trọng số học trên Genodisc. Em giữ nguyên
  để nạp được trọng số đó; đổi sang 18 hay 50 thì phải học lại từ đầu trên dữ liệu ít. ResNet-50 trở lên
  dùng khối Bottleneck (1×1, 3×3, 1×1), nặng hơn nhiều khi chuyển sang 3D.

**Hỏi: Bản 3D của em khác ResNet-34 gốc (2D) chỗ nào?**
- **Trả lời:** Ba chỗ: (1) kernel 3D, ví dụ 3×3×3, nên nhìn được quan hệ giữa các lát; (2) bước nhảy chỉ
  theo cao và rộng, `stride=(1, s, s)`, nên **9 lát giữ nguyên tới cuối**; (3) tầng 2 không giảm kích thước
  (stride 1), giữ chi tiết cho đĩa đệm nhỏ. Tích chập đầu là (3, 7, 7), bước nhảy (1, 2, 2).

**Hỏi: Đầu ra của ResNet trong bài em là gì?**
- **Trả lời:** Sau tầng 4 là khối đặc trưng `[512, 9, 7, 14]`, qua GAP thành **vector 512 chiều**. Vector
  này vào Fusion MLP cùng vector của BiomedCLIP (Hybrid), hoặc vào đầu ra tuyến tính 3 lớp (bản CBAM).

### E. Tiền xử lý của nhánh BiomedCLIP (`biomedclip_wrapper.py`)

| Bước | Làm gì |
|---|---|
| 1 | Lấy lát 4, 5, 6 (chỉ số 3:6) của khối đã chuẩn hóa 0–1 |
| 2 | `F.interpolate(..., (224, 224), bilinear)`: kéo 112 × 224 thành 224 × 224, giãn đôi chiều cao |
| 3 | Lặp 1 kênh xám thành 3 kênh RGB giống nhau |
| 4 | Chuẩn hóa bằng trung bình và độ lệch chuẩn RGB của CLIP |

**Hỏi: Sao lặp ảnh xám thành 3 kênh?** BiomedCLIP được học với ảnh 3 kênh, lớp đầu của ViT chờ đúng 3
kênh. Lặp 3 kênh giống nhau là cách chuẩn để dùng ảnh xám mà không đổi mô hình.

### F. Tiền xử lý SPIDER (`spider_dataloader.py`)

| Bước | Làm gì |
|---|---|
| 1 | Đọc ảnh và mặt nạ `.mha` bằng SimpleITK, xoay về cùng hướng chuẩn |
| 2 | Tìm đĩa đệm trong mặt nạ: nhãn 201–207; tâm = giữa hộp bao của vùng nhãn đó |
| 3 | Lấy 9 lát dọc quanh tâm |
| 4 | Cắt 60 mm (trên–dưới) × 120 mm (trước–sau), đổi milimét sang pixel bằng khoảng cách pixel của file |
| 5 | Nếu số lát khác 9 thì nội suy tuyến tính giữa hai lát gần nhất (`np.linspace`) |
| 6 | Đổi kích thước về 112 × 224, chuẩn hóa phân vị 1–99% như RSNA |

---

## Phần 7d. Mô hình nền SpineNetV2 (dựa trên paper gốc)

Nguồn: Windsor, Jamaludin, Kadir, Zisserman, *SpineNetV2: Automated Detection, Labelling and
Radiological Grading of Clinical MR Scans*, arXiv:2205.01683, 2022 (Oxford VGG, báo cáo kỹ thuật).
Thầy có thể hỏi "mô hình nền em dùng làm gì, học trên dữ liệu gì, em thay đổi chỗ nào".

**Hỏi: SpineNetV2 gồm những gì?**
- **Trả lời:** Bốn mạng nối tiếp:
  1. **Phát hiện đốt sống** bằng *Vector Field Regression*: một U-Net ra 13 kênh, gồm 5 kênh bản đồ
     nhiệt Gauss cho 4 góc và tâm mỗi đốt sống, và 8 kênh trường vector (mỗi góc 2 kênh x, y) "chỉ" từ
     góc về tâm. Các góc chỉ về cùng một tâm thì gom thành một đốt sống; gom giữa các lát nếu IoU > 0.25.
  2. **Mạng ngoại hình (appearance)**: nhìn một khối quanh đốt sống, đoán tầng (S1 … C3, 23 lớp).
  3. **Mạng ngữ cảnh (context)**: sửa dự đoán dựa trên các đốt sống lân cận.
  4. **Beam search** (ý tưởng từ mô hình ngôn ngữ): chọn dãy nhãn tầng hợp lý nhất, phạt dãy vô lý
     (hai đốt sống cùng tầng, thiếu tầng).
  Sau đó cắt **IVV** quanh từng đĩa đệm và đưa vào **mạng chấm nhãn**.

**Hỏi: SpineNetV2 cắt IVV thế nào? Em cắt có giống không?**
- **SpineNetV2:** tâm = trung điểm tâm hai đốt sống liền kề; **xoay khối** cho mặt dưới đốt sống trên
  nằm ngang; rộng gấp đôi đốt sống lớn hơn; tỉ lệ 2:1; lấy mọi lát có đốt sống; rồi lấy mẫu lại về
  $112 \times 224 \times 9$.
- **Luận văn (thí nghiệm RSNA):** cắt khung cố định 240 × 120 pixel quanh tọa độ nhãn RSNA, **không xoay**,
  9 lát quanh lát của tọa độ, rồi đổi về $112 \times 224$. Giống ở kích thước đầu ra và tỉ lệ 2:1; khác
  ở chỗ không xoay và không co giãn theo cỡ đốt sống. Tăng cường xoay ±10° bù một phần.
- **Vì sao không dùng cách cắt của SpineNetV2 cho RSNA:** RSNA đã cho sẵn tọa độ đĩa đệm do bác sĩ đánh
  dấu, dùng trực tiếp thì mọi cấu hình nhận cùng đầu vào, không phụ thuộc lỗi phát hiện.

**Hỏi: Mạng chấm nhãn của SpineNetV2 cụ thể là gì?**
- **Trả lời:** ResNet 3D làm thân, mã hóa mỗi IVV thành **vector 512 chiều**, rồi mỗi nhãn một đầu ra
  riêng. Một mô hình dùng chung cho mọi nhãn và mọi tầng.

**Hỏi: Trọng số SpineNetV2 học trên dữ liệu nào?**
- **Trả lời:** **Genodisc**: ảnh MRI thắt lưng T2 từ **6 trung tâm** quốc tế, khoảng **2,279 bệnh nhân**
  (2,819 lần chụp), mỗi ca do bác sĩ chẩn đoán hình ảnh chuyên khoa chấm. Phần phát hiện đốt sống còn học
  thêm trên OWS (ảnh toàn cột sống, 196 bệnh nhân). Đây là lý do khởi tạo từ SpineNetV2 hợp lý: thân mạng
  đã quen ảnh MRI thắt lưng T2.

**Hỏi: Nhãn của SpineNetV2 khác RSNA thế nào? Sao phải học lại đầu ra?**

| Nhãn | SpineNetV2 (Genodisc) | RSNA 2024 |
|---|---|---|
| Hẹp ống sống | 4 mức | 3 mức (Normal/Mild, Moderate, Severe) |
| Hẹp lỗ liên hợp trái / phải | có / không (2 lớp) | 3 mức |
| Pfirrmann, hẹp khe đĩa, tổn thương tấm tận, biến đổi tủy xương, trượt đốt sống, thoát vị | có | không có |

- Số lớp khác nhau nên đầu ra cũ không dùng được; giữ thân mạng, học lại đầu ra cho 3 nhãn RSNA.
- **Trong phần mềm**, 8 nhãn còn lại (Pfirrmann, thoát vị…) lấy từ **đầu ra gốc của SpineNetV2** (học trên
  Genodisc), em không huấn luyện lại.

**Hỏi: SpineNetV2 báo cáo kết quả thế nào? So với số của em được không?**
- **Trả lời:** Họ dùng **balanced accuracy** trên phần dữ liệu Genodisc giữ lại, ví dụ hẹp ống sống 4
  mức 64.9%, lỗ liên hợp trái / phải 84.8% / 82.4% (2 lớp), thoát vị 80.4%, Pfirrmann 70.9%. **Không so
  trực tiếp** với số của em được: khác dữ liệu, khác số lớp (lỗ liên hợp của họ chỉ 2 lớp, dễ hơn 3 mức),
  khác chỉ số.

**Hỏi: SpineNetV2 huấn luyện thế nào? Em có kế thừa gì?**
- **SpineNetV2:** học toàn mạng bằng Adam lr $10^{-4}$, dừng khi 10 epoch liền không cải thiện balanced
  accuracy trên val; **sau đó giữ cố định thân mạng**, học riêng từng đầu ra với lr $10^{-5}$. Tăng cường:
  dịch, xoay, lật, độ sáng và tương phản ±10%, 50% số mẫu thêm nhiễu Gauss 10%.
- **Luận văn:** cùng tinh thần "giữ thân mạng, học phần nhỏ phía sau". Tăng cường tương tự nhưng mạnh hơn
  (độ sáng và tương phản ±20%), không lật trái–phải, thêm Focal Loss, trọng số căn bậc hai, lặp mẫu hiếm.

**Hỏi: SpineNetV2 chạy nhanh cỡ nào?**
- **Trả lời:** Theo paper, trên GPU: phát hiện và gán tầng khoảng 2–3 giây, chấm nhãn khoảng 1 giây cho
  một ca thắt lưng. Hybrid suy luận khoảng 62 ms một đĩa đệm, tức khoảng 0.3 giây cho 5 đĩa đệm, cùng
  bậc.

**Hỏi: Vì sao không dùng phần phát hiện đốt sống của SpineNetV2 mà dùng TotalSpineSeg?**
- **Trả lời:** Ở bước chọn mô hình phân vùng (trang 13), TotalSpineSeg có Precision 97.87%, cao hơn hẳn
  SpineNetV2 (51.76%), và sai số bề mặt thấp nhất (6.85 mm so với 8.04 mm). Vùng tô chính xác thì khối cắt
  ra ít lẫn mô khác.

**Hỏi: Đi lại từng bước: SpineNetV2 gốc chấm nhãn thế nào, em sửa ở bước nào?**

| Bước | SpineNetV2 gốc (paper) | Mô hình nền trong luận văn (chưa sửa) | Sau khi em sửa (CBAM → Hybrid) |
|---|---|---|---|
| 1. Định vị đĩa | Tự phát hiện đốt sống, gán tầng | Dùng tọa độ đĩa do bác sĩ RSNA đánh dấu | Như cột giữa (app thì dùng TotalSpineSeg) |
| 2. Cắt khối | IVV xoay theo đốt sống, rộng gấp đôi đốt, về $112 \times 224 \times 9$ | Khung cố định 240 × 120, không xoay, về $9 \times 112 \times 224$ | Như cột giữa |
| 3. Thân mạng | ResNet-34 3D, học trên Genodisc | Giữ nguyên trọng số Genodisc, **đóng băng** | **+ CBAM-3D** sau 4 tầng (bước 1); **+ nhánh BiomedCLIP** 3 lát giữa (bước 2) |
| 4. Hợp nhất | Không có | Không có | Ghép 512 + 512, Fusion MLP về 512, chuẩn hóa |
| 5. Đầu ra | 8 đầu MLP 2 lớp cho 8 nhãn Genodisc | **3 đầu tuyến tính mới** cho 3 nhãn RSNA (chỉ học 4,617 tham số) | CBAM: 3 đầu tuyến tính. Hybrid: **cô-sin với câu mô tả**, nhân $s$ |
| 6. Loss | Cross-entropy cân bằng lớp | Cross-entropy thường, không trọng số | Focal + trọng số lớp căn bậc hai + Uncertainty; Hybrid thêm 0.1 × SupCon |
| 7. Mất cân bằng | Chỉ loss cân bằng | Không xử lý | + lặp mẫu hiếm (×5 bước 1, ×3 bước 2) + tăng cường (không lật trái–phải) |
| 8. Học | Học cả mạng (Adam, lr $10^{-4}$), rồi đóng băng thân, học riêng từng đầu ra (lr $10^{-5}$) | Đóng băng thân, học đầu ra | Bước 1 đóng băng thân, học CBAM + đầu ra; bước 2 đóng băng cả hai nhánh, học 1.18 triệu tham số |

- **Câu nói gọn:** "SpineNetV2 gốc: phát hiện đốt sống, cắt IVV, ResNet-34 3D ra vector 512, 8 đầu ra.
  Mô hình nền của em giữ thân mạng đó, đóng băng, thay đầu ra bằng 3 đầu cho RSNA, học bằng cross-entropy
  thường. Em sửa ba chỗ: thêm CBAM vào thân mạng, thêm nhánh BiomedCLIP và đầu cô-sin, và thêm quy trình
  xử lý mất cân bằng."

**Hỏi: Em có thử bật/tắt từng thành phần (ablation) không?**
- **Trả lời:** "Dạ có, theo **từng nhánh kiến trúc**, mỗi cấu hình chạy 3 lần trên RSNA:"

  | Cấu hình | CBAM | BiomedCLIP | Mean F1 | Severe Recall | Severe F1 |
  |---|---|---|---|---|---|
  | SpineNetV2 (nền) | tắt | tắt | 0.420 | 0.123 | 0.152 |
  | CBAM | **bật** | tắt | 0.477 | 0.288 | 0.262 ± 0.082 |
  | BMC-only | tắt | **bật** | 0.482 | 0.316 | 0.268 ± 0.034 |
  | Hybrid | **bật** | **bật** | **0.527** | **0.486** | **0.356 ± 0.017** |

  "Bật riêng từng nhánh đều tăng Severe F1 lên khoảng 0.26; bật cả hai lên 0.356 và ổn định hơn hẳn (độ
  lệch chuẩn 0.017 so với 0.082 và 0.034). Nghĩa là hai nhánh bổ sung cho nhau, không nhánh nào thừa."
- **Phần chưa làm, nói thẳng:** "Còn các kỹ thuật xử lý mất cân bằng (Focal, trọng số lớp, lặp mẫu, SupCon)
  em bật **cùng lúc**, chưa tắt riêng từng cái để đo đóng góp. Đây là thí nghiệm em sẽ bổ sung." Thêm ý
  nếu cần: cấu hình nền (không xử lý gì) chỉ bắt 12% ca Severe, nên cả cụm kỹ thuật này rõ ràng có tác dụng;
  chỉ chưa biết cái nào góp bao nhiêu.
- **Một ablation nhỏ khác đã có:** thử thay cách ghép (concat) bằng cổng (gated), không hơn (xem Phần 4).

**Hỏi: Em thay đổi gì so với SpineNetV2? (câu chốt)**
- **Trả lời:** Giữ thân ResNet 3D và trọng số Genodisc. Thêm bốn thứ: (1) CBAM-3D sau mỗi tầng; (2) nhánh
  BiomedCLIP giữ cố định; (3) thay các đầu ra cố định bằng đầu cô-sin so với câu mô tả, nhờ đó thêm được
  nhãn mới; (4) quy trình xử lý mất cân bằng cho RSNA.

---

## Phần 7e. Theo hướng của thầy phản biện (tối ưu, hệ thống, dữ liệu)

Thầy làm về tối ưu, lập lịch, blockchain, federated learning; không làm ảnh y tế. Nên giải thích bằng
lời thường, ít thuật ngữ y khoa, và chuẩn bị các câu về hàm mục tiêu, chi phí, dữ liệu.

**Hỏi: Bài toán tối ưu của em là gì? Tối ưu cái gì, theo biến nào?**
- **Trả lời:** Cực tiểu tổng loss $\mathcal{L} = \sum_{k=1}^{3}\big(\frac{1}{2\sigma_k^2}\mathcal{L}_k + \log\sigma_k\big)$,
  với $\mathcal{L}_k$ = Focal Loss trung bình của nhãn $k$ + 0.1 × SupCon. Biến tối ưu chỉ khoảng **1.18
  triệu tham số** (Fusion MLP, gộp lát, hệ số $s$) cộng 3 tham số $\sigma_k$; hai nhánh lớn đứng yên.
  Thuật toán AdamW, lr $10^{-4}$, weight decay $10^{-4}$, lô 32.

**Hỏi: Có đảm bảo hội tụ không?**
- **Trả lời:** Bài toán không lồi nên không có đảm bảo hội tụ về tối ưu toàn cục, giống mọi mạng nơ-ron.
  Em kiểm soát bằng thực nghiệm: theo dõi Severe F1 trên tập val, dừng sớm sau 5 epoch không cải thiện,
  và chạy 3 seed. Kết quả ổn định (Severe F1 0.356 ± 0.017), nên lời giải tìm được không phụ thuộc
  nhiều vào khởi tạo.

**Hỏi: Uncertainty weighting thực chất làm gì, dưới góc tối ưu?**
- **Trả lời:** Mỗi nhãn có một trọng số $1/(2\sigma_k^2)$ được học cùng mạng. Đạo hàm theo $\sigma_k^2$ bằng
  0 cho $\sigma_k^2 = \mathcal{L}_k$, tức trọng số tỉ lệ $1/\mathcal{L}_k$: nhãn loss lớn nhận trọng số nhỏ, để
  ba nhãn đóng góp cân nhau. Số hạng $\log\sigma_k$ là "phạt" để $\sigma_k$ không tăng mãi (nếu không, cách
  dễ nhất để loss nhỏ là cho mọi trọng số về 0).
- **Liên hệ hướng của thầy:** cùng tinh thần với các cách gộp có trọng số theo loss trong federated
  learning: dùng chính loss để quyết định mỗi phần đóng góp bao nhiêu, thay vì cố định tay.

**Hỏi: Sao không dò lưới (grid search) trọng số 3 nhãn cho chắc?**
- **Trả lời:** Mỗi tổ hợp trọng số là một lần huấn luyện khoảng 25 phút; dò lưới 3 chiều, mỗi chiều vài
  giá trị, đã là hàng chục lần chạy, nhân 3 seed. Uncertainty weighting học luôn trọng số trong một lần
  chạy, chỉ thêm 3 tham số.

**Hỏi: Chi phí tính toán? Có chạy được thực tế không?**
- **Trả lời:** Học bước 2 của Hybrid thêm khoảng **25 phút** trên một RTX 4090 (đo trên một lần chạy).
  Suy luận khoảng **62 ms một đĩa đệm**; một ca 5 đĩa đệm thắt lưng là dưới 1 giây cho phần chấm điểm.
  Trong app, cả quy trình (phân vùng + chấm) cho một ca mất khoảng một phút trên GPU, chủ yếu do phân vùng.
  Đóng băng hai nhánh lớn giúp học nhanh và chỉ cần một GPU.

**Hỏi: Độ phức tạp theo số nhãn?**
- **Trả lời:** Phần tốn nhất (hai nhánh ảnh) chạy **một lần** cho mỗi đĩa đệm, không phụ thuộc số nhãn.
  Mỗi nhãn thêm chỉ là một phép nhân ma trận $[512] \times [512, m]$ với $m$ câu mô tả, gần như không đáng kể.
  Vector văn bản tính một lần rồi lưu lại.

**Hỏi: Dữ liệu bệnh nhân và quyền riêng tư?**
- **Trả lời:** RSNA 2024 và SPIDER là bộ dữ liệu công khai, đã ẩn danh trước khi phát hành. App trong luận
  văn chạy thử với ca mẫu. Nếu triển khai thật thì cần ẩn danh thông tin trong file DICOM và chạy trong mạng
  nội bộ bệnh viện, không gửi ảnh ra ngoài.

**Hỏi: Nhiều bệnh viện muốn cùng huấn luyện mà không chia sẻ dữ liệu thì sao?**
- **Trả lời:** Federated learning phù hợp: mỗi bệnh viện học tại chỗ, chỉ gửi cập nhật trọng số về để gộp.
  Mô hình em hợp với cách này vì phần cần học nhỏ (khoảng 1.18 triệu tham số, vài MB mỗi lần gửi), hai
  nhánh lớn đứng yên. Em chưa làm, đây là hướng phát triển; khó khăn chính là dữ liệu mỗi nơi lệch nhau
  (máy chụp khác, tỉ lệ bệnh khác).

**Hỏi: Vòng phản hồi bác sĩ trong app là gì, có phải học trực tuyến không?**
- **Trả lời:** Không tức thì. Bác sĩ sửa nhãn, app ghi lại; khi đủ dữ liệu thì gom thành tập mới và huấn
  luyện lại theo đợt. Như vậy kiểm soát được chất lượng nhãn trước khi đưa vào mô hình.

---

## Phần 7f. Theo hướng của thầy chủ tịch (thị giác máy tính, ảnh y tế, hợp nhất đa nhiệm)

Thầy làm đúng mảng của bài: thị giác máy tính, phân vùng ảnh y tế với dữ liệu ít, chuẩn hóa và tăng cường
chuỗi ảnh y tế, mã hóa vị trí (positional encoding), hợp nhất đa phương thức và đa nhiệm. Thầy cũng có bài
ở MIWAI 2025 nên biết rõ hội nghị. Đây là người hỏi chuyên môn sâu nhất.

**Đã có sẵn, xem lại:**

| Chủ đề thầy có thể hỏi | Ở đâu trong file này |
|---|---|
| Vì sao ghép (concat) mà không cổng / cross-attention; đã thử gated | Phần 4, "Vì sao ghép (concat)…" và câu ngay sau về gated |
| Một mô hình đa nhiệm hay ba mô hình riêng | Phần 7b, "Một mô hình chung, ba đầu ra" |
| Tăng cường dữ liệu, sao không lật trái–phải | Phần 4, Trang 18, "Augmentation cụ thể?" |
| Chuẩn hóa phân vị 1–99%, nội suy, cắt theo pixel | Phần 7c, mục A |
| Chọn TotalSpineSeg, Recall 56% | Phần 3 |

**Hỏi: Gộp lát bên nhánh BiomedCLIP có dùng thông tin vị trí (positional encoding) không?**
- **Trả lời:** Không. Trọng số $\alpha_i = \text{softmax}(w_a^\top z_i)$ chỉ dựa vào nội dung từng lát, đổi
  thứ tự 3 lát thì kết quả y hệt. Em chấp nhận vì chỉ có 3 lát giữa, sát nhau, vai trò gần như ngang nhau.
  Thông tin theo chiều trái–phải (quan trọng cho lỗ liên hợp trái/phải) nằm ở nhánh CBAM: tích chập 3D
  trượt qua 9 lát theo đúng thứ tự. Thêm mã hóa vị trí cho phần gộp lát là một hướng cải thiện.

**Hỏi: Dữ liệu thiếu nhãn thì em xử lý thế nào? Có dùng bán giám sát không?**
- **Trả lời:** Nhãn thiếu ghi là −1 và bị bỏ qua khi tính loss (`ignore_index`), nên một đĩa đệm thiếu một
  nhãn vẫn học được từ hai nhãn còn lại. Em chưa dùng bán giám sát (ví dụ gán nhãn giả cho dữ liệu chưa
  nhãn); đó là hướng hợp lý vì lớp Severe rất ít.

**Hỏi: Dữ liệu ít và lệch, sao không học lại toàn bộ mạng?**
- **Trả lời:** Học lại khoảng 150 triệu tham số trên khoảng 10 nghìn đĩa đệm, lớp Severe chỉ 4%, rất dễ học thuộc.
  Em giữ hai nhánh đã học sẵn (SpineNetV2 học trên Genodisc, BiomedCLIP học trên 15 triệu cặp ảnh–chữ y
  khoa), chỉ học khoảng 1.18 triệu tham số ở phần hợp nhất. Đây là cách tận dụng tri thức có sẵn khi dữ
  liệu nhỏ.

**Hỏi: Ảnh RSNA từ nhiều máy khác nhau, cường độ lệch nhau thì sao?**
- **Trả lời:** MRI không có thang cường độ tuyệt đối như CT, nên em chuẩn hóa **theo từng khối**: lấy phân
  vị 1–99% của chính khối đó rồi đưa về 0–1. Cách này bỏ được điểm nhiễu quá sáng/tối và đưa các máy về
  cùng khoảng. Em chưa dùng chuẩn hóa mạnh hơn như khớp histogram giữa các máy.

**Hỏi: MIWAI là hội nghị cỡ nào?**
- **Trả lời:** Nói đúng mức: "Hội nghị quốc tế về AI, kỷ yếu xuất bản trong Springer LNAI, bài được phản
  biện." Không thổi phồng hạng hội nghị.

---

**Hỏi: Nhóm thầy hướng dẫn có bài MIWAI 2025 về đau thắt lưng bằng học tương phản. Bài em khác gì?**
- **Bối cảnh:** Le, Vo, Mai, Phan, "Revolutionizing Precise Low Back Pain Diagnosis via Contrastive
  Learning", MIWAI 2025. Thầy chủ tịch cũng có bài ở MIWAI 2025 nên có thể biết bài này. Bài MIWAI 2026
  của em có trích dẫn nó (`le2025lumbarclip`); luận văn thì không có trong danh mục tài liệu.
- **Trả lời:** "Dạ bài đó huấn luyện một mô hình kiểu CLIP **từ đầu** trên cặp ảnh MRI thắt lưng và văn bản
  để chẩn đoán đau thắt lưng. Bài em thì **giữ nguyên** BiomedCLIP đã học sẵn và mạng 3D của SpineNetV2,
  chỉ học phần hợp nhất nhỏ, rồi đọc kết quả bằng cô-sin với câu mô tả. Mục tiêu của em khác: chấm mức độ
  hẹp từng đĩa đệm khi dữ liệu rất lệch, và **mở rộng sang bộ nhãn mới** (SPIDER) bằng cách đổi câu mô tả,
  thay vì tăng độ chính xác trên cùng một bộ dữ liệu."
- Nếu bị hỏi sao luận văn không trích: "Dạ em có trích trong bài báo, em sẽ bổ sung vào luận văn bản cuối."

## Phần 7g. Theo hướng của thầy ủy viên (tối ưu tổ hợp, tin sinh học, học tương phản)

Thầy làm tối ưu tổ hợp, hình học tính toán, tin sinh học; bài gần đây dùng **học tương phản** và **phân cụm
trên không gian embedding**, rất gần SupCon của bài. Câu về tối ưu, độ phức tạp: xem Phần 7e. Câu về
kiểm định thống kê, 3 lần chạy: Phần 5 (Trang 21) và Phần 7b. Câu về bán giám sát: Phần 7f.

**Hỏi: SupCon cụ thể tính thế nào?**
- **Trả lời:** Trong một lô, với mỗi đĩa đệm (gọi là neo), các đĩa **cùng mức** của cùng nhãn bệnh là cặp
  dương, các đĩa khác mức là cặp âm. Tính cô-sin giữa mọi cặp vector ảnh (sau Fusion MLP, đã chuẩn hóa),
  chia nhiệt độ $\tau = 0.07$, rồi softmax trên tất cả các đĩa khác trong lô; loss là trung bình $-\log$ xác
  suất rơi vào cặp dương. Kết quả: ca cùng mức bị kéo lại gần, khác mức bị đẩy ra. Làm riêng cho 3 nhãn.

**Hỏi: Cặp dương lấy ở đâu? Có tăng cường ảnh để tạo cặp như SimCLR không?**
- **Trả lời:** Không. SimCLR (tự giám sát) tạo cặp dương bằng hai phiên bản tăng cường của cùng một ảnh.
  SupCon (có giám sát) dùng **nhãn**: mọi ca cùng mức trong lô đều là cặp dương. Em dùng nhãn vì đã có nhãn.

**Hỏi: Lô chỉ có đúng một ca Severe thì sao?**
- **Trả lời:** Thẳng thắn: ca đó không có cặp dương. Code em xử lý đơn giản là **bỏ SupCon của cả lô đó**
  (loss bằng 0), chỉ còn Focal Loss. Vậy SupCon chỉ có tác dụng ở các lô mà mức nào có mặt cũng có ít nhất
  2 ca; lặp mẫu hiếm ×3 giúp chuyện này xảy ra thường hơn. Đây cũng là lý do SupCon chỉ là phần phụ (hệ số
  0.1), phần chính vẫn là Focal Loss. Cách tốt hơn: chỉ bỏ những neo không có cặp dương thay vì cả lô, hoặc
  dùng bộ nhớ đệm (memory bank) giữ vector ca Severe từ các lô trước. Đó là hướng cải thiện.

**Hỏi: Nhiệt độ 0.07 lấy đâu ra? Sao không học như $s$?**
- **Trả lời:** 0.07 là giá trị mặc định của bài SupCon và CLIP gốc; em giữ cố định, không dò. $s$ ở đầu ra
  được học vì nó quyết định trực tiếp xác suất dự đoán; nhiệt độ của SupCon chỉ ở loss phụ nên em không
  thêm tham số.

**Hỏi: Có kiểm tra không gian embedding có tách cụm theo mức độ không (t-SNE, UMAP)?**
- **Trả lời:** Em đánh giá gián tiếp qua kết quả phân loại (vì đầu ra chính là cô-sin với câu mô tả, nếu
  không tách cụm thì F1 đã không lên). Vẽ t-SNE/UMAP theo mức độ là phần em nên bổ sung để thấy trực tiếp.

**Hỏi: Ba nhãn có thứ tự (nhẹ < vừa < nặng), sao không tận dụng thứ tự?**
- **Trả lời:** Đúng là nhãn có thứ tự. Hiện em coi là 3 lớp rời nhau, nên đoán Normal cho ca Severe bị phạt
  bằng đoán Moderate. Các cách tận dụng thứ tự (hồi quy thứ tự, loss có trọng số theo khoảng cách mức) là
  hướng cải thiện. Xem thêm Phần 7b.

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
