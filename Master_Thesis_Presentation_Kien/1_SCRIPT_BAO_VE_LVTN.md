# Kịch bản tập nói — Bảo vệ luận văn thạc sĩ (Thứ Hai 02/11/2026)

Deck: `presentation_spine.pdf` (52 trang; trang 1–39 trình bày, trang 40–52 phụ lục).
Thời lượng: thầy dặn 20–30 phút. Kịch bản này nhắm **khoảng 23 phút kể cả video demo**,
để còn dư nếu bị ngắt hỏi giữa chừng.

Cách đọc file:
- Chữ trong khung trích dẫn (`>`) là **lời nói**. Tập vài lần cho thuộc ý, lúc bảo vệ coi đây
  là ghi chú, đừng đọc nguyên văn. Câu bắc cầu cuối mỗi slide là chỗ dễ lộ ra "đọc giấy" nhất.
- Khối **📘 Giải thích công thức** là để anh hiểu, **không đọc lên**. Dùng khi hội đồng hỏi
  sâu vào công thức.
- Mốc giờ `[mm:ss]` là mốc tích lũy, ước tính theo tốc độ nói khoảng 170 âm tiết/phút. Bấm giờ
  thật một lần rồi chỉnh lại.
- Mọi số liệu khớp luận văn đã nộp (commit `ef23e38`): trung bình 3 lần chạy (seed 42, 123, 456).

Giải thích metric (Recall, AUC, AUPRC, Dice, ASSD…) và cách tính ở hai thí nghiệm chọn mô hình
nằm ở file riêng: `4_GIAI_THICH_METRIC_VA_CONG_THUC.md`.

---

## PHẦN 1 — GIỚI THIỆU `[00:00 → 04:20]`

### Trang 1 — Tiêu đề `[00:00]`

> Em kính chào quý thầy cô trong hội đồng. Em tên là Hà Trung Kiên, học viên cao học khóa
> 2024, chuyên ngành Khoa học Máy tính. Hôm nay em xin trình bày luận văn "Ứng dụng AI trong
> gán nhãn y tế cho bệnh đau thắt lưng", dưới sự hướng dẫn của thầy Phan Trọng Nhân.

### Trang 2 — Mục lục `[00:20]`

> Bài trình bày của em gồm bảy phần: giới thiệu, cơ sở lý thuyết, dữ liệu và lựa chọn mô hình,
> mô hình đề xuất, kết quả thực nghiệm, phần mềm hỗ trợ bác sĩ, và cuối cùng là kết luận.

*(Trang 3 là mục lục phần 1, bấm qua luôn.)*

### Trang 4 — Giới thiệu đề tài `[00:35]`

> Đau thắt lưng là một trong những bệnh phổ biến nhất hiện nay. Để chẩn đoán nguyên nhân, bác
> sĩ thường phải đọc ảnh MRI cột sống, và việc này hiện vẫn làm thủ công: mỗi ca có nhiều chuỗi
> ảnh, mỗi chuỗi nhiều lát, nên vừa tốn thời gian vừa phụ thuộc kinh nghiệm người đọc.
>
> Hình bên phải là ví dụ từ bộ dữ liệu RSNA 2024. Hàng trên là ca bình thường hoặc nhẹ, hàng dưới
> là ca nặng, cho ba nhóm bệnh: hẹp ống sống và hẹp lỗ liên hợp hai bên. Có thể thấy vùng khác
> biệt rất nhỏ, nên bỏ sót là chuyện dễ xảy ra. Vì vậy em hướng tới một hệ thống AI hỗ trợ bác sĩ
> gán nhãn tự động.
>
> Để làm được điều đó, có bốn khó khăn chính mà em sẽ trình bày ngay sau đây.

### Trang 5 — Các thách thức chính `[01:30]`

> Thứ nhất là mất cân bằng lớp. Biểu đồ bên phải là phân bố nhãn của RSNA 2024: lớp Severe, tức
> ca nặng, chỉ chiếm khoảng 4%. Mô hình học trên dữ liệu như vậy sẽ có xu hướng đoán ca nào cũng
> bình thường, mà trong lâm sàng, bỏ sót một ca nặng nguy hiểm hơn nhiều so với báo nhầm một ca
> nhẹ.
>
> Thứ hai là nhãn cố định. Các mô hình hiện nay được thiết kế cho đúng một bộ nhãn. Muốn chấm
> thêm một bệnh mới thì phải sửa đầu ra và huấn luyện lại.
>
> Thứ ba là khả năng tổng quát hóa. Mô hình học trên dữ liệu của bệnh viện này phải còn dùng được
> trên dữ liệu của bệnh viện khác.
>
> Và thứ tư là khoảng cách tới người dùng. Kết quả của mô hình phải hiện ra trên ảnh để bác sĩ
> xem và sửa được, chứ không chỉ là một con số trong bảng.
>
> Ở phần kết quả, em sẽ lần lượt quay lại từng thách thức này. Từ bốn thách thức đó, em đặt ra
> bốn mục tiêu tương ứng.

### Trang 6 — Mục tiêu đề tài `[02:40]`

> Mục tiêu thứ nhất là phân vùng tự động các cấu trúc cột sống trên MRI: đốt sống, đĩa đệm, ống
> sống và tủy sống. Với mục tiêu này, em khảo sát các mô hình có sẵn và chọn ra mô hình phù hợp
> nhất cho từng mô-đun.
>
> Mục tiêu thứ hai là chấm mức độ bất thường cho từng tầng đĩa đệm, và trọng tâm là lớp Severe
> hiếm gặp.
>
> Mục tiêu thứ ba là mở rộng không gian nhãn: chấm được một bộ nhãn mới mà không gắn thêm đầu ra,
> không huấn luyện lại.
>
> Mục tiêu thứ tư là xây dựng phần mềm tương tác để bác sĩ xem kết quả trên ảnh và sửa được mọi
> nhãn.
>
> Ngoài ra, khả năng tổng quát hóa được em kiểm chứng trên một bộ dữ liệu độc lập là SPIDER.
> Tiếp theo em xin nói rõ phạm vi của đề tài.

### Trang 7 — Phạm vi đề tài `[03:40]`

> Về dữ liệu, đề tài chỉ làm trên ảnh MRI cột sống thắt lưng, chủ yếu là mặt cắt dọc T2 hoặc
> STIR; mặt cắt ngang Axial chưa được khai thác. Về cấu trúc giải phẫu, em phân vùng đốt sống,
> đĩa đệm, ống sống và tủy sống. Về bệnh lý, trên RSNA em chấm hẹp ống sống và hẹp lỗ liên hợp
> hai bên; khi mở rộng sang SPIDER thì có thêm thang Pfirrmann, Modic, thoát vị, phồng đĩa đệm
> và trượt đốt sống.
>
> Phần tiếp theo em xin trình bày ngắn gọn cơ sở lý thuyết của hai thành phần chính trong mô hình.

---

## PHẦN 2 — CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN `[04:20 → 06:40]`

### Trang 9 — CBAM và BiomedCLIP `[04:20]`

> Thành phần thứ nhất là CBAM, một cơ chế chú ý gồm hai bước. Bước đầu là chú ý theo kênh, tức
> chọn xem đặc trưng nào quan trọng. Bước sau là chú ý theo không gian, tức chọn xem vị trí nào
> trên ảnh quan trọng. Với bài toán của em, tổn thương chỉ chiếm một vùng rất nhỏ quanh đĩa đệm,
> nên em muốn mạng dồn trọng số vào vùng đó thay vì xử lý đều toàn ảnh.
>
> Thành phần thứ hai là BiomedCLIP, một mô hình thị giác–ngôn ngữ được học trên 15 triệu cặp ảnh
> và văn bản y khoa, theo hướng CLIP. Điểm em cần ở BiomedCLIP là ảnh và câu mô tả nằm chung một
> không gian vector. Nhờ vậy, em có thể so một ảnh với một câu như "hẹp ống sống mức nặng", và
> muốn thêm nhãn mới thì chỉ cần viết thêm câu mô tả. Đây là nền tảng cho cơ chế zero-shot mà em
> trình bày ở phần sau.
>
> Bây giờ em xin đặt các hướng tiếp cận hiện có cạnh nhau để thấy khoảng trống mà luận văn nhắm
> tới.

#### 📘 Ví dụ về cơ chế chú ý CBAM (không đọc lên)

Công thức đầy đủ ở phụ lục trang 45, giải thích ký hiệu ở cuối file này. Phần dưới đây là ví dụ để
dễ hình dung và để trả lời câu "vì sao tốt hơn".

**Bối cảnh.** Sau mỗi tầng của ResNet-34, mạng có một bản đồ đặc trưng $F$ kích thước
$C \times D \times H \times W$: $C$ kênh, mỗi kênh là một khối 3D. Mỗi kênh có thể hiểu là một
"máy dò" một kiểu chi tiết: kênh này phản ứng với cạnh đĩa đệm, kênh kia phản ứng với dịch não
tủy sáng trong ống sống, kênh khác phản ứng với mỡ hoặc nhiễu. CBAM chỉnh $F$ qua hai bước, kênh
trước rồi tới không gian.

**Bước 1: chú ý theo kênh ("xem cái gì").** Mỗi kênh nhận một trọng số từ 0 đến 1.

- Cách tính: mỗi kênh được gộp cả khối thành 2 số (trung bình và lớn nhất), đưa qua một MLP nhỏ
  dùng chung (thu gọn 16 lần), cộng lại rồi qua sigmoid.
- Ví dụ với 4 kênh ở một đĩa đệm bị hẹp ống sống:

| Kênh | Phản ứng với | Trọng số $M_c$ | Sau khi nhân |
|---|---|---|---|
| 1 | bờ sau đĩa đệm lồi vào ống sống | 0.92 | giữ gần nguyên |
| 2 | dịch não tủy quanh tủy sống | 0.85 | giữ gần nguyên |
| 3 | mỡ dưới da phía lưng | 0.10 | bị làm mờ 10 lần |
| 4 | nhiễu nền | 0.05 | gần như bị tắt |

- Ý nghĩa: mạng **tự chọn loại đặc trưng** liên quan tới bệnh đang chấm. Hai kênh 1 và 2 chính là
  dấu hiệu bác sĩ nhìn khi đánh giá hẹp ống sống (đĩa lồi và dịch não tủy bị ép mất), nên được
  giữ. Mỡ và nhiễu bị giảm.
- Trọng số được tính lại cho **từng ảnh**: ca có tổn thương khác thì bộ trọng số khác.

**Bước 2: chú ý theo không gian ("xem ở đâu").** Mỗi vị trí $(d, h, w)$ trong khối 3D nhận một
trọng số từ 0 đến 1.

- Cách tính: tại mỗi vị trí, gộp tất cả kênh thành 2 số (trung bình và lớn nhất), ra 2 bản đồ 3D;
  tích chập $7 \times 7 \times 7$ rồi qua sigmoid, ra một mặt nạ 3D.
- Ví dụ với khối 9 lát của một đĩa đệm bị hẹp lỗ liên hợp **bên trái**:
  - Vùng lỗ liên hợp trái, ở mấy lát phía trái: trọng số khoảng 0.9.
  - Ống sống ở các lát giữa: khoảng 0.5.
  - Thân đốt sống, cơ lưng, nền đen: khoảng 0.1.
- Ý nghĩa: mạng **tự chọn vùng** cần nhìn. Vì mặt nạ là 3D, mỗi lát có trọng số riêng, nên tổn
  thương chỉ nằm ở 1–2 lát bên trái vẫn được tô đậm, không bị các lát giữa lấn át. Đây là điểm
  khác của bản CBAM-3D trong luận văn so với CBAM 2D gốc.

**Vì sao tốt hơn CNN thường: ví dụ "pha loãng tín hiệu".**

Cuối nhánh, CNN gộp trung bình toàn bộ khối thành một vector (global average pooling). Giả sử vùng
tổn thương chiếm 2% thể tích khối.

| | Tổn thương (2%) | Nền (98%) | Tổn thương chiếm bao nhiêu trong vector cuối |
|---|---|---|---|
| CNN thường (mọi vị trí trọng số 1) | 0.02 × 1 = 0.02 | 0.98 × 1 = 0.98 | **2%** |
| Có chú ý không gian (tổn thương 1, nền 0.1) | 0.02 × 1 = 0.02 | 0.98 × 0.1 = 0.098 | 0.02 / 0.118 ≈ **17%** |

Tín hiệu của tổn thương tăng khoảng 8 lần trong vector cuối, nên lớp phân loại dễ nhận ra ca nặng
hơn. Đó là lý do CBAM hợp với bài toán này: tổn thương nhỏ, phần lớn khối ảnh là mô bình thường.

**Các câu dễ bị hỏi**

- **Sao làm kênh trước rồi mới tới không gian?** Bài báo CBAM gốc (Woo và cộng sự 2018) đã thử
  các cách sắp xếp, và cách "kênh trước, không gian sau, nối tiếp" cho kết quả tốt nhất. Hiểu
  trực giác: chọn đúng loại đặc trưng trước, rồi mới tìm vị trí của đặc trưng đó.
- **Sao vừa dùng trung bình vừa dùng lớn nhất?** Trung bình phản ánh mức chung của cả kênh, lớn
  nhất bắt được điểm phản ứng mạnh nhất (thường là chỗ tổn thương). Tổn thương nhỏ dễ bị trung bình
  làm mờ, nên cần thêm max.
- **Tốn thêm bao nhiêu?** Khoảng 1% tham số (phụ lục trang 43), vì MLP thu gọn 16 lần và tích chập
  chỉ có 2 kênh vào, 1 kênh ra. CBAM gắn sau từng tầng mà không phải đổi kiến trúc ResNet, nên vẫn
  nạp được trọng số đã huấn luyện trước.
- **Khác self-attention của Transformer chỗ nào?** Self-attention so từng cặp vị trí với nhau, chi
  phí tăng theo bình phương số vị trí, rất nặng với khối 3D. CBAM chỉ tính một trọng số cho mỗi
  kênh và mỗi vị trí, nhẹ hơn nhiều.
- **Bằng chứng trong luận văn:**
  - Trên RSNA, chỉ thêm CBAM đã nâng Severe F1 từ 0.152 (SpineNetV2) lên khoảng 0.26.
  - Grad-CAM trang 22 minh họa vùng chú ý tập trung hơn, nhưng đó chỉ là **một ca**.
- **Giới hạn, nói thật nếu bị hỏi:** CBAM đơn lẻ tụt trên SPIDER (0.619 so với 0.646 của
  SpineNetV2). Có thể là CBAM học quá sát đặc điểm riêng của ảnh RSNA. Đó là lý do cần ghép thêm
  nhánh BiomedCLIP.
- **Đừng nói** "trọng số chú ý cho thấy mô hình nhìn đúng chỗ tổn thương". Các con số trong ví dụ
  trên là minh họa, luận văn không đo trực tiếp trọng số $M_c$, $M_s$.

### Trang 10 — Công trình liên quan `[05:30]`

> Bảng này tóm tắt bốn hướng. SpineNetV2 là mạng CNN 3D chuyên cho cột sống, là chuẩn tốt nhưng
> nhãn cố định và yếu ở lớp hiếm. Giải pháp của Ning Shen trên RSNA dùng Faster R-CNN kết hợp
> Swin Transformer, độ chính xác cao nhưng là tổ hợp nhiều mô hình lớn và chưa công bố thành bài
> báo. MedGemma đại diện cho hướng mô hình thị giác–ngôn ngữ zero-shot: rất linh hoạt về nhãn
> nhưng độ chính xác còn thấp. Còn CBAM giúp tập trung vào vùng tổn thương nhỏ.
>
> Nhìn chung, chưa có hướng nào vừa xử lý được lớp hiếm vừa linh hoạt về nhãn. Đó là chỗ kiến trúc
> của em muốn lấp. Nhưng trước khi đề xuất, em cần chọn mô hình nền, và việc đó bắt đầu từ dữ
> liệu.

---

## PHẦN 3 — DỮ LIỆU VÀ LỰA CHỌN MÔ HÌNH `[06:40 → 10:00]`

### Trang 12 — Các bộ dữ liệu `[06:40]`

> Em dùng ba bộ dữ liệu công khai. Bộ chính là RSNA 2024, chấm ba mức Normal/Mild, Moderate và
> Severe cho ba nhóm bệnh; em dùng bộ này để huấn luyện và đánh giá. Bộ thứ hai là SPIDER của Hà
> Lan, gồm 8 nhãn bệnh hoàn toàn khác RSNA; em dùng để kiểm tra khả năng tổng quát hóa. Bộ thứ ba
> là Lumbar Spine Segmentation, 23 ca MRI T2 có mặt nạ đốt sống, dùng làm tập kiểm thử độc lập khi
> so sánh các mô hình phân vùng.
>
> Em bắt đầu với thí nghiệm chọn mô hình phân vùng.

### Trang 13 — Lựa chọn mô hình phân vùng `[07:30]`

> Em so bốn mô hình trên 23 ca này, bằng bốn chỉ số: Dice đo độ chồng lắp giữa vùng dự đoán và
> nhãn chuẩn; Precision cho biết trong vùng mô hình khoanh có bao nhiêu phần đúng là đốt sống;
> Recall cho biết mô hình phủ được bao nhiêu phần đốt sống thật; còn ASSD là khoảng cách trung
> bình giữa hai đường biên, tính bằng milimét, càng nhỏ càng tốt.
>
> TotalSpineSeg có Dice cao nhất, 70.81%, nhưng điểm em quan tâm nhất là Precision 97.87% và ASSD
> thấp nhất, 6.85 mm. Recall của nó chỉ 56% vì nhãn chuẩn của bộ này tính cả gai sau và mỏm ngang,
> còn TotalSpineSeg chỉ khoanh thân đốt sống. Trong hệ thống của em, mặt nạ phân vùng chỉ dùng để
> định vị đĩa đệm rồi cắt khối ảnh, nên khoanh đúng chỗ quan trọng hơn khoanh đủ. Vì vậy em chọn
> TotalSpineSeg.
>
> Với mô-đun thứ hai là chấm mức độ bệnh, em làm một thí nghiệm chọn tương tự.

> **Nếu bị hỏi "sao không chọn mô hình Recall cao như SpineNetV2?"**: Recall 91.79% nhưng
> Precision chỉ 51.76%, tức gần một nửa vùng khoanh là mô xung quanh. Cắt khối ảnh từ mặt nạ đó sẽ
> lẫn nhiều mô bên cạnh.

### Trang 14 — Lựa chọn mô-đun đánh giá bất thường `[08:40]`

> Em so ba mô hình đại diện cho ba hướng: SpineNetV2, giải pháp của Ning Shen, và MedGemma. Vì
> ba mô hình có đầu ra khác nhau, em quy về một bài toán chung là có bệnh hay không: mức Moderate
> hoặc Severe được tính là có bệnh. Em đánh giá trên 150 bệnh nhân của tập validation RSNA, tính
> riêng cho ba nhóm bệnh rồi lấy trung bình.
>
> Ning Shen cao nhất với F1 80.7%. SpineNetV2 đứng thứ hai, 77.5%, chỉ kém khoảng ba điểm.
> MedGemma rất thấp, Recall chỉ khoảng 12%, tức bỏ sót gần chín trên mười ca có bệnh.
>
> Em chọn SpineNetV2 làm mô hình nền, vì nó chuyên cho cột sống, chấm được nhiều nhãn, mã nguồn
> mở và kiến trúc đủ đơn giản để tinh chỉnh. Ning Shen tốt hơn một chút nhưng là tổ hợp nhiều mô
> hình lớn, khó duy trì và khó tích hợp.
>
> Tuy nhiên, F1 ở đây chỉ đo có bệnh hay không. Khi chấm đúng ba mức độ, SpineNetV2 chỉ bắt được
> khoảng 12% số ca Severe. Đây chính là điểm em muốn cải thiện ở mô hình đề xuất.

---

## PHẦN 4 — MÔ HÌNH ĐỀ XUẤT `[10:00 → 14:30]`

### Trang 16 — Pipeline giải pháp `[10:00]`

> Đây là toàn bộ hệ thống em đề xuất, gồm hai mô-đun nối tiếp. Ảnh MRI đi vào TotalSpineSeg để
> phân vùng. Từ mặt nạ phân vùng, em định vị từng đĩa đệm và cắt ra một khối ảnh ba chiều quanh nó,
> gọi là IVV. Khối này đi vào mô-đun chấm nhãn Hybrid, và kết quả cuối cùng được đưa lên phần mềm
> cho bác sĩ.
>
> Đóng góp chính của em nằm ở mô-đun chấm nhãn, nên em xin đi vào kiến trúc của nó.

### Trang 17 — Kiến trúc Hybrid hai nhánh (hình) `[10:40]`

> Đầu vào là khối ảnh của một đĩa đệm, kích thước 9 lát, mỗi lát 112 nhân 224. Khối này đi vào
> hai nhánh song song.
>
> Nhánh phía trên là nhánh chú ý: mạng ResNet-34 ba chiều có gắn CBAM sau mỗi tầng. Nhánh này học
> hình dạng ba chiều của đĩa đệm và tập trung vào vùng tổn thương nhỏ. Đầu ra là vector f_cbam 512
> chiều.
>
> Nhánh phía dưới là nhánh đa phương thức, dùng bộ mã hóa ảnh của BiomedCLIP. Vì BiomedCLIP chỉ
> đọc ảnh hai chiều, em lấy các lát ở giữa khối, mã hóa từng lát, rồi gộp lại bằng trọng số chú ý,
> ra vector f_bmc cũng 512 chiều. Nhánh này mang kiến thức y khoa chung đã học từ dữ liệu rất lớn.
>
> Hai vector được ghép lại thành 1024 chiều, đi qua khối hợp nhất, rồi so với vector của các câu
> mô tả nhãn bằng độ tương đồng cô-sin; mức nào giống nhất thì được chọn. Hai ô bên phải cho thấy
> cùng một mô hình chấm được cả bộ nhãn đã học, như RSNA, lẫn một bộ nhãn mới như SPIDER; hai
> trường hợp chỉ khác nhau ở bộ câu mô tả đưa vào.
>
> Điểm quan trọng là danh sách câu mô tả nằm ngoài mạng. Muốn chấm bộ nhãn khác thì chỉ cần thay
> danh sách đó, không cần gắn thêm đầu ra hay huấn luyện lại.
>
> Có kiến trúc rồi, câu hỏi tiếp theo là huấn luyện thế nào khi lớp Severe chỉ có 4%.

#### 📘 Công thức Hybrid: phụ lục trang 40 (không trình bày, chỉ mở khi được hỏi)

Slide công thức đã chuyển xuống phụ lục, lúc trình bày bấm thẳng từ trang 17 sang trang 18. Nếu
hội đồng hỏi công thức, mở trang 40 và giải thích theo thứ tự trên slide: gộp lát → ghép hai nhánh
→ chuẩn hóa → so cô-sin → softmax → argmax. Giải thích kỹ phần gộp lát nằm ở
`4_GIAI_THICH_METRIC_VA_CONG_THUC.md`, mục 15.

**Công thức 1: hợp nhất (Fusion MLP)**

$$f_{img} = W_2\,\phi\big(W_1[f_{cbam}; f_{bmc}] + b_1\big) + b_2, \qquad \hat f_{img} = \frac{f_{img}}{\|f_{img}\|_2}$$

| Ký hiệu | Ý nghĩa | Kích thước |
|---|---|---|
| $f_{cbam}$ | vector đặc trưng từ nhánh CBAM (sau phép gộp trung bình toàn cục) | 512 |
| $f_{bmc}$ | vector đặc trưng từ nhánh BiomedCLIP (sau gộp lát) | 512 |
| $[\,\cdot\,;\,\cdot\,]$ | phép ghép hai vector nối đuôi nhau | 512 + 512 = 1024 |
| $W_1, b_1$ | trọng số và độ lệch lớp thứ nhất | $768 \times 1024$, 768 |
| $\phi$ | hàm kích hoạt GELU, một bản "mềm" của ReLU: giá trị âm bị ép gần 0 nhưng không cắt cụt đột ngột | — |
| $W_2, b_2$ | trọng số và độ lệch lớp thứ hai | $512 \times 768$, 512 |
| $\|f\|_2$ | độ dài (chuẩn L2) của vector: $\sqrt{f_1^2 + \dots + f_{512}^2}$ | số thực |

- Sau lớp thứ nhất có dropout 0.1 (khi huấn luyện, ngẫu nhiên tắt 10% nơ-ron để tránh học thuộc).
- **Vì sao đầu ra 512 chiều?** Để bằng số chiều vector văn bản của BiomedCLIP, hai loại vector mới
  so trực tiếp được với nhau.
- **Vì sao chuẩn hóa về độ dài 1?** Ví dụ vector $(3, 4)$ có độ dài 5, chia cho 5 được
  $(0.6, 0.8)$. Khi cả vector ảnh lẫn vector văn bản đều dài 1, tích vô hướng của chúng **đúng
  bằng cô-sin góc giữa hai vector**, tức chỉ còn đo *hướng*, không bị ảnh hưởng bởi độ lớn.
- Số tham số học: $W_1$ có $768 \times 1024 \approx 786$ nghìn, $W_2$ có $512 \times 768 \approx
  393$ nghìn, cộng lại khoảng **1.18 triệu**. Đó là gần như toàn bộ phần được huấn luyện (thêm
  $w_a$ 512 số, $s$, và 3 số $\sigma_k$).

**Công thức 2: chấm nhãn bằng cô-sin (cosine classifier)**

$$p_{k,c} = \frac{\exp(s\,\hat f_{img}^{\top} t_{k,c})}{\sum_j \exp(s\,\hat f_{img}^{\top} t_{k,j})}, \qquad \hat y_k = \arg\max_c p_{k,c}$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $k$ | nhãn bệnh: hẹp ống sống, hẹp lỗ liên hợp trái, hẹp lỗ liên hợp phải |
| $c$ | mức độ: Normal/Mild, Moderate, Severe |
| $t_{k,c}$ | vector của câu mô tả mức $c$ thuộc nhãn $k$, do bộ mã hóa văn bản BiomedCLIP tạo ra, đã chuẩn hóa độ dài 1. RSNA có 3 nhãn × 3 mức = **9 câu**, mã hóa **một lần** rồi giữ cố định |
| $\hat f_{img}^{\top} t_{k,c}$ | tích vô hướng = cô-sin, nằm trong $[-1, 1]$, càng lớn thì ảnh càng "giống" câu mô tả |
| $s$ | hệ số nhiệt độ, **học được**, giới hạn tối đa 100 |
| $\exp(\cdot)/\sum$ | softmax: biến điểm số thành xác suất, các mức cộng lại bằng 1 |
| $\arg\max_c$ | chọn mức có xác suất cao nhất |

- **Vì sao cần $s$?** Cô-sin giữa các câu mô tả thường chênh nhau rất ít. Ví dụ cô-sin với ba câu
  là 0.20 / 0.25 / 0.31. Không nhân $s$ thì softmax cho xác suất gần đều, khoảng 0.32 / 0.33 /
  0.35, nên mô hình rất "lưỡng lự". Nhân $s = 50$ thì điểm thành 10 / 12.5 / 15.5, và softmax
  cho khoảng **0.004 / 0.047 / 0.949**. Vậy $s$ làm khuếch đại khác biệt nhỏ để xác suất rõ ràng.
- **Vì sao gọi là zero-shot?** Công thức không có tham số nào gắn riêng với từng mức độ. Tất cả
  thông tin về nhãn nằm trong $t_{k,c}$. Đổi sang 8 nhãn SPIDER thì chỉ thay bộ $t$, công thức
  giữ nguyên.
- **So với SpineNetV2:** SpineNetV2 dùng đầu tuyến tính $\text{softmax}(W x + b)$, trong đó mỗi
  hàng của $W$ ứng với một mức độ cố định. Thêm nhãn mới thì phải thêm hàng mới vào $W$ và học
  lại. Ở Hybrid, "hàng" đó chính là vector câu mô tả $t_{k,c}$, nên lấy được miễn phí từ văn bản.

**Công thức 0: gộp lát của nhánh BiomedCLIP (dòng đầu trang 40)**

$$\alpha_i = \frac{\exp(w_a^\top z_i)}{\sum_j \exp(w_a^\top z_j)}, \qquad f_{bmc} = \sum_i \alpha_i z_i$$

- $z_i$: vector 512 chiều của lát thứ $i$. Nhánh này dùng **3 lát giữa** khối, mỗi lát nhân thành
  3 kênh và đổi kích thước về $224 \times 224$ cho đúng đầu vào ViT-B/16.
- $w_a$: vector 512 chiều học được, là "câu hỏi" mà mô hình đặt ra cho mỗi lát: lát này có đáng
  chú ý không.
- $\alpha_i$: trọng số của lát $i$, các $\alpha$ cộng lại bằng 1.
- Ví dụ: điểm $w_a^\top z_i$ của ba lát là 2, 1, 0, cho $\alpha \approx 0.67, 0.24, 0.09$. Lát
  đầu (nhìn rõ tổn thương hơn) đóng góp gần 70%, thay vì 33% nếu lấy trung bình đều.

### Trang 18 — Xử lý mất cân bằng lớp `[12:30]`

> Em dùng bốn kỹ thuật bổ sung cho nhau.
>
> Thứ nhất là Focal Loss kèm trọng số lớp. Công thức bên trái: so với cross-entropy thông thường,
> Focal Loss nhân thêm hệ số một trừ p mũ gamma. Mẫu nào mô hình đã đoán đúng chắc chắn thì hệ số
> này gần bằng 0, nên mẫu dễ gần như không còn đóng góp, và mô hình phải tập trung vào mẫu khó.
> Thêm vào đó là trọng số omega theo nghịch đảo căn bậc hai số mẫu, nên lớp Severe được tính nặng
> hơn khoảng bốn lần lớp đa số.
>
> Thứ hai là lấy mẫu lại: đĩa đệm nào có nhãn Moderate hoặc Severe được lặp lại ba lần trong mỗi
> epoch. Thứ ba là tăng cường dữ liệu: lật, xoay nhẹ, đổi độ sáng và thêm nhiễu, để các bản lặp
> không giống hệt nhau và mô hình không học thuộc.
>
> Thứ tư là Uncertainty weighting, công thức bên dưới. Em có ba nhãn bệnh, mỗi nhãn một hàm mất
> mát, và ba nhãn này khó dễ khác nhau. Thay vì cộng thẳng, mỗi nhãn có một tham số sigma tự học:
> nhãn nào khó hơn thì sigma tăng, trọng số của nó giảm, để hai nhãn lỗ liên hợp vốn khó không lấn
> át nhãn ống sống.
>
> Các kỹ thuật này kết hợp vào quá trình huấn luyện như sau.

#### 📘 Giải thích công thức (không đọc lên)

**Focal Loss có trọng số lớp**

$$\text{FL} = -\,\omega_y\,(1 - p_y)^{\gamma}\,\log(p_y)$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $y$ | mức đúng của mẫu |
| $p_y$ | xác suất mô hình gán cho mức đúng (từ softmax ở công thức cô-sin) |
| $-\log(p_y)$ | cross-entropy thông thường: $p_y$ càng nhỏ, phạt càng nặng |
| $(1 - p_y)^\gamma$ | hệ số "focal", với $\gamma = 2$ |
| $\omega_y$ | trọng số của lớp $y$ |

Ví dụ so sánh hai mẫu:

| | Mẫu dễ ($p_y = 0.9$) | Mẫu khó ($p_y = 0.2$) | Tỉ lệ khó/dễ |
|---|---|---|---|
| Cross-entropy $-\log p_y$ | 0.105 | 1.609 | ≈ 15 lần |
| Hệ số $(1-p_y)^2$ | 0.01 | 0.64 | |
| Focal Loss | 0.001 | 1.030 | **≈ 1000 lần** |

Ý nghĩa: với cross-entropy, hàng nghìn mẫu Normal/Mild dễ cộng lại vẫn lấn át vài chục mẫu Severe
khó. Focal Loss gần như "tắt" mẫu dễ, nên đạo hàm chủ yếu đến từ mẫu khó, mà mẫu khó thường là
Severe.

**Trọng số lớp theo căn bậc hai**

$$\omega_c = \frac{1/\sqrt{n_c}}{\frac{1}{3}\sum_{j=1}^{3} 1/\sqrt{n_j}}$$

- $n_c$: số mẫu của lớp $c$ trong tập huấn luyện. Mẫu số là trung bình của ba giá trị
  $1/\sqrt{n}$, để **trung bình ba trọng số bằng 1** (không làm thay đổi độ lớn chung của loss).
- Ví dụ với tỉ lệ 81% / 15% / 4%: $1/\sqrt{81} = 0.111$, $1/\sqrt{15} = 0.258$, $1/\sqrt{4} =
  0.500$; trung bình 0.290, nên $\omega \approx 0.38 / 0.89 / 1.73$. Severe nặng hơn Normal/Mild
  khoảng **4.5 lần**.
- **Vì sao căn bậc hai mà không lấy nghịch đảo trực tiếp?** Nghịch đảo trực tiếp $1/n$ cho tỉ lệ
  81/4 ≈ 20 lần (luận văn ghi "khoảng mười lăm lần", tùy nhãn). Trọng số quá chênh làm loss
  dao động mạnh mỗi khi gặp một mẫu Severe, huấn luyện mất ổn định. Căn bậc hai là mức giữa.

**Uncertainty weighting (Kendall và cộng sự 2018)**

$$\mathcal{L}_{total} = \sum_{k=1}^{3}\left(\frac{1}{2\sigma_k^2}\,\mathcal{L}_k + \log\sigma_k\right)$$

- $\mathcal{L}_k$: mất mát của nhãn $k$ = Focal Loss + 0.1 × SupCon (số hạng phụ kéo vector ảnh
  cùng mức độ lại gần nhau; có trong luận văn, không có trên slide).
- $\sigma_k$: tham số học được, hiểu là "độ bất định" của nhãn $k$.
- $\frac{1}{2\sigma_k^2}$: trọng số thật của nhãn $k$. Ví dụ $\sigma = 1$ cho trọng số 0.5;
  $\sigma = 2$ cho trọng số 0.125.
- $\log\sigma_k$: số hạng "phạt" để $\sigma$ không tăng mãi. Nếu không có nó, mô hình chỉ cần cho
  $\sigma \to \infty$ là loss về 0, tức bỏ học hết.
- Cân bằng giữa hai số hạng: nhãn nào có $\mathcal{L}_k$ lớn (khó, như lỗ liên hợp) thì tăng
  $\sigma_k$ có lợi hơn, nên trọng số tự giảm. Nhãn dễ hơn thì $\sigma_k$ nhỏ, trọng số lớn.

### Trang 19 — Huấn luyện và suy diễn `[13:50]`

> Bên trái là luồng huấn luyện. Em huấn luyện theo hai bước. Bước một là huấn luyện cấu hình CBAM
> trên RSNA với đầy đủ các kỹ thuật xử lý mất cân bằng vừa nói. Bước hai là huấn luyện Hybrid:
> nhánh CBAM nạp từ bước một và giữ cố định, nhánh BiomedCLIP cũng giữ cố định, chỉ học khối hợp
> nhất, khoảng 1.18 triệu tham số.
>
> Em giữ cố định hai nhánh vì hai lý do. Một là số ca Severe quá ít để cập nhật hàng chục triệu
> tham số mà không học thuộc. Hai là nếu cập nhật BiomedCLIP theo RSNA, không gian chung giữa ảnh
> và văn bản sẽ bị kéo lệch, và các câu mô tả nhãn mới sẽ khó so được.
>
> Bên phải là suy diễn: không còn tăng cường dữ liệu, không có hàm mất mát, khối ảnh chỉ chạy qua
> mạng một lần để ra xác suất và nhãn.
>
> Bây giờ em xin chuyển sang kết quả thực nghiệm.

---

## PHẦN 5 — KẾT QUẢ THỰC NGHIỆM `[14:30 → 19:00]`

### Trang 21 — Kết quả tổng hợp trên RSNA 2024 `[14:30]`

> Đây là câu trả lời cho thách thức thứ nhất, mất cân bằng lớp. Em so bốn cấu hình. SpineNetV2 là
> mô hình nền. CBAM chỉ có nhánh chú ý. BMC-only chỉ có nhánh BiomedCLIP. Hybrid có cả hai. Mỗi
> cấu hình chạy ba lần với ba seed khác nhau, số trong bảng là trung bình.
>
> Phần trên là các chỉ số trung bình. Hybrid cao nhất ở cả bốn: Mean F1 tăng từ 0.420 lên 0.527,
> Mean Recall từ 0.412 lên 0.592.
>
> Nhưng phần em muốn thầy cô chú ý nhất là lớp Severe ở phía dưới. Severe Recall tăng từ 0.123
> lên 0.486, tức gấp khoảng bốn lần: mô hình nền chỉ bắt được khoảng một trên tám ca nặng, Hybrid
> bắt được gần một nửa. Severe F1 tăng từ 0.152 lên 0.356, hơn gấp đôi. AUC và AUPRC của lớp Severe
> cũng cao nhất.
>
> Một điểm nữa: CBAM và BMC-only đều cải thiện Severe F1 lên khoảng 0.26, và khi ghép lại thì
> Hybrid đạt 0.356, tức hai nhánh bổ sung cho nhau.
>
> Đổi lại, Mean Accuracy giảm từ 0.810 xuống 0.724. Em xin giải thích: SpineNetV2 có accuracy cao
> vì đa số ca là Normal/Mild, đoán nhóm đó là phần lớn đúng, nhưng cũng vì thế mà bỏ sót gần hết
> ca nặng. Trong sàng lọc, em ưu tiên không bỏ sót ca nặng.
>
> Để xem cơ chế chú ý thực sự thay đổi điều gì, em có một ca minh họa.

### Trang 22 — Grad-CAM `[16:10]`

> Đây là một ca hẹp ống sống mức nặng ở tầng L4–L5. Grad-CAM tô màu vùng mà mô hình dựa vào để
> ra quyết định. Với mô hình có CBAM, vùng nóng phủ đúng chỗ đĩa đệm chèn vào ống sống, tức vùng
> tổn thương, và mô hình cho xác suất Severe 0.53, chấm đúng ca nặng này. Mô hình không có CBAM chỉ
> cho 0.26, tức đoán sai.
>
> *(Mẹo: nói mạnh vế "có CBAM phủ đúng vùng tổn thương + chấm đúng". Tránh tự nói vế "không CBAM dàn
> trải", vì hình bên trái cũng khá gọn; nếu thầy hỏi thì xem `2_HOI_DAP_BAO_VE.md` mục 0.8.)*
>
> Em xin nói rõ đây chỉ là một ca minh họa trực quan. Bằng chứng định lượng là bảng kết quả em vừa
> trình bày.
>
> Tiếp theo là thách thức thứ hai: chấm nhãn mới mà không huấn luyện lại.

#### 📘 Giải thích Grad-CAM (không đọc lên)

$$\alpha_k = \frac{1}{Z}\sum_{i,j}\frac{\partial y^{c}}{\partial A^{k}_{ij}}, \qquad L^{c}_{\text{Grad-CAM}} = \text{ReLU}\Big(\sum_k \alpha_k A^k\Big)$$

- $A^k$: bản đồ đặc trưng thứ $k$ ở một tầng tích chập (slide dùng tầng 3).
- $y^c$: điểm của lớp $c$ (ở đây là Severe).
- $\alpha_k$: đạo hàm của $y^c$ theo $A^k$, lấy trung bình trên mọi vị trí, cho biết bản đồ $k$
  quan trọng với lớp Severe đến đâu.
- Cộng các bản đồ theo trọng số, rồi qua ReLU để chỉ giữ vùng **làm tăng** điểm Severe. Phóng to
  về kích thước ảnh rồi tô màu.
- **Giới hạn:** một ca không chứng minh được gì. Thử nghiệm sau luận văn (`docs/LVTN_phase3/XAI_SURVEY.md`)
  cho thấy bản đồ saliency trên dữ liệu này không hơn phép đoán "giữa ảnh", vì khối ảnh vốn đã cắt
  căn giữa vào đĩa đệm. Vậy nên luôn gọi đây là **minh họa**.

### Trang 23 — Zero-shot trên SPIDER `[16:55]`

> Với Hybrid, muốn chấm nhãn mới thì chỉ cần đổi câu mô tả. Em thử với 8 nhãn của SPIDER: những
> nhãn này không có trong dữ liệu huấn luyện, và em không huấn luyện lại gì cả. SpineNetV2 và CBAM
> không làm được thí nghiệm này, vì đầu ra của chúng cố định ba mức của RSNA.
>
> Kết quả trung bình F1 là 0.362. Ba nhãn về hình thái đĩa đệm như phồng, thoát vị, hẹp khe đĩa
> đạt F1 khoảng 0.56 đến 0.61. Các nhãn nhiều mức như Pfirrmann và Modic, hoặc rất hiếm như trượt
> đốt sống, thì gần như chưa làm được.
>
> Em không xem đây là một bộ phân loại hoàn chỉnh cho SPIDER. Điều kết quả này cho thấy là cùng một
> mô hình, không đổi gì ở mạng, đã chấm được một bộ nhãn khác, của một bệnh viện khác.
>
> Còn nếu được huấn luyện lại trên SPIDER thì sao? Đó là thách thức thứ ba.

### Trang 24 — SPIDER có tinh chỉnh `[18:00]`

> Ở thí nghiệm này, em tinh chỉnh cả bốn cấu hình trên SPIDER để xem lợi thế có còn giữ được khi
> đổi dữ liệu không. Hybrid đạt F1 0.653, ngang với SpineNetV2 là 0.646. Còn CBAM tụt xuống 0.619,
> thấp hơn cả mô hình nền. Nghĩa là thêm nhánh BiomedCLIP giúp Hybrid giữ được mức của mô hình nền
> khi đổi bệnh viện, trong khi CBAM đơn lẻ thì không.
>
> Rõ nhất là ở nhãn thoát vị đĩa đệm, một nhãn hiếm. SpineNetV2 và CBAM đoán âm tính cho mọi ca,
> còn Hybrid đạt Recall 0.636.
>
> Đó là phần mô hình. Thách thức cuối cùng là đưa kết quả tới tay bác sĩ.

---

## PHẦN 6 — PHẦN MỀM HỖ TRỢ BÁC SĨ `[19:00 → 21:40]`

### Trang 26 — Mục tiêu và kiến trúc `[19:00]`

> Mục tiêu của phần mềm là rút ngắn thời gian đọc phim, không thay thế bác sĩ. Em đặt ra ba yêu
> cầu: bác sĩ thấy được AI nhìn vào đâu, thông qua mặt nạ tô màu ngay trên ảnh; mọi nhãn đều sửa
> được; và mỗi lần sửa được lưu lại thành một mẩu dữ liệu.
>
> Về kiến trúc: giao diện viết bằng React với thư viện Cornerstone3D để hiển thị ảnh y khoa, máy
> chủ FastAPI gọi hai mô hình phân vùng và chấm nhãn, cơ sở dữ liệu lưu ca chụp, nhãn và các lần
> hiệu chỉnh.

### Trang 27 — Giao diện `[19:40]`

> Hình trên là màn hình đọc phim, với mặt nạ phân vùng tô màu trên ảnh MRI. Hình dưới là bảng chấm
> điểm: mỗi ô là một nhãn ở một tầng đĩa đệm, màu nền theo mức độ, và bác sĩ bấm vào để sửa trực
> tiếp. Em xin mời thầy cô xem video demo.

### Trang 28 — Demo `[20:05]`

*(Cmd+Tab sang QuickTime đã mở sẵn, phát video. Nói kèm nếu video không có tiếng.)*

> Đầu tiên em tải một ca chụp lên. Hệ thống chạy phân vùng và chấm nhãn. Đây là kết quả phân vùng
> hiện trên ảnh. Đây là bảng nhãn. Bác sĩ thấy một nhãn chưa đúng thì bấm vào sửa, và chỉnh sửa
> được lưu lại.

*(Thời gian slide này phụ thuộc độ dài video; kịch bản dành khoảng 1 phút 30.)*

> Đó là toàn bộ phần mềm. Em xin chuyển sang phần kết luận.

---

## PHẦN 7 — KẾT LUẬN `[21:40 → 24:20]`

### Trang 30 — Đóng góp chính `[21:40]`

> Luận văn có bốn đóng góp. Một là kiến trúc Hybrid hai nhánh cùng quy trình xử lý mất cân bằng,
> nâng Severe Recall từ 0.123 lên 0.486. Hai là cơ chế mở rộng nhãn zero-shot, thêm nhãn mới bằng
> câu mô tả. Ba là kiểm chứng trên SPIDER, dữ liệu của một bệnh viện khác: Hybrid đạt F1 0.653,
> ngang SpineNetV2, và bắt được ca thoát vị đĩa đệm mà SpineNetV2 bỏ sót. Bốn là phần mềm hỗ trợ bác sĩ phân vùng, chấm 11 nhãn, sửa và lưu chỉnh sửa.
>
> Bên cạnh đó, luận văn vẫn còn một số hạn chế.

### Trang 31 — Hạn chế `[22:15]`

> Thứ nhất, mô hình vẫn bỏ sót khoảng một nửa số ca nặng, nên chỉ nên dùng như một người đọc thứ
> hai, bác sĩ vẫn là người quyết định. Thứ hai, mô hình chưa dùng mặt cắt Axial, trong khi lỗ liên
> hợp nhìn rõ nhất trên mặt cắt này. Và thứ ba, phần mềm mới lưu chỉnh sửa của bác sĩ, mô hình chưa
> học lại từ đó.

### Trang 32 — Công bố khoa học `[22:50]`

> Một phần kết quả của luận văn đã được chấp nhận và trình bày tại hội nghị MIWAI 2026, lần thứ
> 19, vào tháng 10 vừa rồi. Bài báo được xuất bản trong bộ Lecture Notes in Artificial
> Intelligence của Springer; em là tác giả thứ nhất và thầy Nhân là tác giả liên hệ.

### Trang 33 — Hướng phát triển `[23:10]`

> Từ các hạn chế trên, em có bốn hướng phát triển. Một là đánh giá phần mềm cùng bác sĩ trên ca
> thật, so nhãn bác sĩ chốt với nhãn mô hình để biết nhãn nào hay bị sửa. Hai là cho mô hình học
> lại từ các chỉnh sửa đó mà không quên kiến thức cũ, và mỗi phiên bản mới đều phải qua đánh giá
> trước khi dùng. Ba là bổ sung mặt cắt Axial để cải thiện hai nhãn lỗ liên hợp. Và bốn là sinh
> báo cáo chẩn đoán bằng ngôn ngữ tự nhiên, gợi ý đoạn kết luận cho bác sĩ.

### Trang 39 — Cảm ơn `[24:00]`

*(Bấm thẳng từ trang 33 sang trang 39, bỏ qua tài liệu tham khảo.)*

> Phần trình bày của em đến đây là kết thúc. Em xin cảm ơn quý thầy cô đã lắng nghe, và em rất
> mong nhận được góp ý của hội đồng.

---

## Phụ lục — chỉ mở khi được hỏi

| Câu hỏi | Mở trang | Ý chính |
|---|---|---|
| Công thức Hybrid (gộp lát, hợp nhất, cô-sin)? | 40 | Xem khối 📘 dưới trang 17 |
| Hybrid có bao nhiêu tham số, học bao nhiêu? | 41 | CBAM-ResNet 21.3M và BiomedCLIP 86.2M giữ cố định; chỉ học ~1.18M |
| Huấn luyện lâu không, suy luận nhanh không? | 42 | Baseline 16.4 phút, CBAM 56 phút, Hybrid **thêm** 24.8 phút; suy luận ~62 ms/mẫu cả ba |
| CBAM gắn ở đâu? | 43 | Sau mỗi stage của ResNet-34; thêm ~1% tham số |
| SpineNetV2 khác Hybrid chỗ nào? | 44 | Đầu softmax cố định so với đầu cô-sin với câu mô tả |
| Công thức CBAM? | 45 | Xem khối giải thích ngay dưới |
| Pipeline gốc SpineNetV2? | 46 | Phát hiện mốc → gom đốt sống → gán mức → cắt IVV → chấm |
| Phân bố nhãn SPIDER? | 47 | Trượt đốt sống 2.8%, thoát vị 4.7% là hiếm nhất |
| Thiết lập thực nghiệm? | 50 | AdamW, batch 32, RTX 4090 |
| Kết quả từng lớp? | 51 | Một lần chạy đại diện (seed 42); Severe: Hybrid thắng cả 5 chỉ số |
| Kết quả từng nhãn SPIDER? | 52 | Hybrid F1 cao nhất 4/7 nhóm |

#### 📘 Giải thích công thức CBAM-3D (trang 45, không đọc lên)

$$M_c(F) = \sigma\big(\text{MLP}(\text{AvgPool}_{3D}(F)) + \text{MLP}(\text{MaxPool}_{3D}(F))\big)$$
$$M_s(F') = \sigma\big(\text{Conv}_{7\times7\times7}([\text{AvgPool}_c(F'); \text{MaxPool}_c(F')])\big)$$
$$F' = M_c(F) \otimes F, \qquad F'' = M_s(F') \otimes F'$$

- $F$: bản đồ đặc trưng kích thước $C \times D \times H \times W$ (kênh × lát × cao × rộng).
- **Bước 1, chú ý theo kênh $M_c$**: với mỗi kênh, gộp toàn bộ khối $D \times H \times W$ thành
  một số bằng hai cách (trung bình và lớn nhất), đưa qua một MLP dùng chung (lớp ẩn $C/16$), cộng
  lại, rồi qua sigmoid $\sigma$ ra số từ 0 đến 1. Kết quả là **một trọng số cho mỗi kênh**:
  kênh nào mang đặc trưng hữu ích (ví dụ cạnh đĩa đệm) thì gần 1, kênh nhiễu thì gần 0.
- **Bước 2, chú ý theo không gian $M_s$**: với mỗi vị trí $(d, h, w)$, gộp theo chiều kênh bằng
  trung bình và lớn nhất, ra 2 bản đồ; tích chập $7 \times 7 \times 7$, rồi sigmoid. Kết quả là
  **một trọng số cho mỗi vị trí trong không gian 3D**: vùng có tổn thương gần 1, nền gần 0.
- $\otimes$: nhân từng phần tử, tức "tô đậm" kênh và vị trí quan trọng, "làm mờ" phần còn lại.
- **Khác bản gốc 2D**: $M_s$ là mặt nạ 3 chiều, mỗi lát có trọng số riêng. Nhờ vậy mạng đề cao
  được vùng nhỏ chỉ xuất hiện ở vài lát, ví dụ hẹp lỗ liên hợp ở một bên.
- **Vì sao vừa avg vừa max?** Trung bình nắm được thông tin chung của toàn kênh, max nắm được tín
  hiệu mạnh nhất (thường là chỗ tổn thương). Dùng cả hai thì ít mất thông tin hơn dùng một.

---

## Câu hỏi khó và cách trả lời

Bộ câu hỏi đầy đủ nằm ở `5_CHUAN_BI_BAO_VE_LVTN.md` mục 3. Dưới đây là các câu liên quan trực tiếp
tới công thức, chưa có trong file đó.

- **"s là học được hay đặt tay? Sao giới hạn 100?"** Học được. Mô hình học $\log s$, khởi tạo
  $s = 1/0.07 \approx 14.3$ giống CLIP gốc (`spinenet/models/grading_hybrid.py`). Giới hạn 100 cũng
  theo CLIP gốc, để softmax không quá "gắt" làm đạo hàm gần 0 và huấn luyện mất ổn định.
- **"Tại sao ghép (concat) mà không cộng hai vector?"** Cộng thì hai vector phải cùng nghĩa ở từng
  chiều, trong khi f_cbam và f_bmc đến từ hai mạng khác nhau, chiều thứ i của mỗi bên không tương
  ứng nhau. Ghép rồi cho MLP học cách trộn thì không cần giả định đó.
- **"Nhánh CBAM giữ cố định thì CBAM có được học không?"** Có, ở bước 1, khi huấn luyện cấu hình
  CBAM. Bước 2 chỉ dùng lại trọng số đó.
- **"Tại sao BiomedCLIP chỉ đọc 3 lát giữa mà không đủ 9?"** BiomedCLIP là mô hình 2D và mỗi lát
  tốn một lượt chạy ViT. Ba lát giữa nằm qua tâm đĩa đệm, nơi thấy tổn thương rõ nhất; chi tiết
  ba chiều đã có nhánh CBAM lo.
- **"Uncertainty weighting có thật sự cần không?"** Ba nhãn có độ khó khác nhau: lỗ liên hợp khó
  hơn ống sống trên mặt cắt dọc. Cộng thẳng thì nhãn khó có loss lớn sẽ chi phối đạo hàm.
  Uncertainty để mô hình tự cân, không cần dò tay ba trọng số.
- **"Checkpoint nào được báo cáo?"** Epoch có Severe F1 trung bình cao nhất trên tập validation;
  dừng sớm nếu 5 epoch không cải thiện, tối đa 20 epoch. Chi tiết ở file metric, mục 9.
