# Báo cáo tiến độ Phase 3 — cải thiện chỉ số grading (22/08/2026)

> Trả lời ba câu thầy đặt ra:
> **(1)** đã tận dụng được đặc trưng của Sagittal T1 / Sagittal T2 / Axial T2 tới đâu;
> **(2)** cơ chế Leader–Supporter hoạt động thế nào;
> **(3)** bài toán mất cân bằng: đã điều chỉnh những gì, highlight bất thường ra sao,
> và còn lại gì để cải thiện chỉ số.
>
> Mọi trích dẫn trong file này đã kiểm tra tồn tại thật. Số liệu thực nghiệm lấy từ
> [`../../experiments/f1_improvement/RESULTS_LOG.md`](../../experiments/f1_improvement/RESULTS_LOG.md)
> (nhật ký chạy GPU, ghi từng epoch).

---

## Phần 1 — Đã tận dụng được đặc trưng từng chuỗi ảnh tới đâu

### 1.1 Vấn đề phát hiện được

Bộ RSNA 2024 có ba chuỗi ảnh cho mỗi bệnh nhân, và theo protocol lâm sàng thì **mỗi
điều kiện được đọc trên một chuỗi khác nhau**:

| Điều kiện | Chuỗi ảnh chuẩn | Lý do giải phẫu |
|---|---|---|
| Hẹp ống sống (spinal canal) | **Sagittal T2/STIR** | dịch não tuỷ sáng trên T2, thấy rõ mức chèn ép ở đường giữa |
| Hẹp lỗ liên hợp (foraminal) | **Sagittal T1** | mỡ quanh rễ thần kinh sáng trên T1; lỗ liên hợp là cấu trúc **bên**, không phải giữa |
| Hẹp ngách bên (subarticular) | **Axial T2** | chỉ nhìn được trên mặt cắt ngang |

Hệ thống của em trước Phase 3 **lọc Sagittal T2 cho toàn bộ ba đầu ra**, kể cả hai đầu
foraminal (`rsna_dataloader.py`, hàm `is_sagittal_t2`). Nghĩa là foraminal đang được
chấm trên **sai chuỗi ảnh**, và đó đúng là điều kiện có chỉ số tệ nhất.

Đây không phải suy đoán. Công trình tham chiếu của Hallinan và cộng sự trên *Radiology*
[1] — hệ thống học sâu cho đúng ba loại hẹp này, 168 lượt trích dẫn — dùng
**Axial T2 và Sagittal T1**, không dùng Sagittal T2. Bài này **đã nằm trong danh mục
tham khảo của bài báo MIWAI của em**, tức là em trích nó nhưng chưa làm theo nó.

### 1.2 Trạng thái hiện tại theo từng chuỗi

| Chuỗi | Có trong dữ liệu | Đã dùng chưa | Kết quả |
|---|---|---|---|
| Sagittal T2/STIR | 1.974 chuỗi | ✅ dùng từ đầu | canal Severe F1 ~0.52 |
| **Sagittal T1** | 1.980 chuỗi | ✅ **mới khai thác trong Phase 3** | đã trích 19.689 crop theo từng bên; xem 1.3 |
| **Axial T2** | **2.340 chuỗi** (nhiều nhất) | ❌ **chưa dùng** | xem 1.4 |

### 1.3 Kết quả khai thác Sagittal T1 (thí nghiệm #1, đã chạy 22/08)

Chấm lại foraminal trên Sagittal T1 thay vì T2, khởi tạo từ mô hình CBAM đã huấn luyện,
đóng băng backbone, 25 epoch (dừng tay ở epoch 20), seed 42:

| | Foraminal Severe F1 |
|---|---|
| Chấm trên Sagittal T2 (hệ cũ) | 0.27 – 0.29 |
| **Chấm trên Sagittal T1 (mới)** | **mức hội tụ ~0.291**, epoch tốt nhất 0.3095 |

Precision của lớp Severe cũng bò từ 0.167–0.188 (epoch 10) lên 0.212–0.216 (epoch 20),
tức về đúng dải precision của hệ cũ (0.22–0.23).

**Em xin báo cáo trung thực ba giới hạn của con số này:**

1. **Đường cong dao động, không phải leo đều.** Tám epoch 13–20 nằm trong dải
   0.273–0.310, biên độ ±0.02–0.03, lớn hơn cả mức tăng giữa epoch 15 và 20 (+0.0076).
   Trung bình dải đó là **0.2914**, còn 0.3095 là **đỉnh của dao động** do cơ chế
   chọn epoch tốt nhất theo Severe F1 lưu lại.
2. **Một seed duy nhất (42).** Chưa đủ để khẳng định; cần 3 seed như các bảng khác.
3. **Đây là chỉ số chỉ trên foraminal**, vì nhãn canal không tồn tại trong dữ liệu T1.
   Nên **không so trực tiếp được với Mean F1 0.527** của ba điều kiện.

Kết luận thẳng thắn: đổi sang T1 cho **ngang tới hơi nhích**, chưa phải bước nhảy. Nhưng
việc sửa cho khớp protocol vẫn đúng, và nó là điều kiện cần cho thí nghiệm ở Phần 2.

### 1.4 Axial T2 — chưa làm, và đề xuất cách đóng gói

Dữ liệu đã sẵn: **2.340 chuỗi Axial T2**, nhiều hơn cả hai chuỗi sagittal. Quan trọng
hơn, bảng nhãn gốc RSNA có **10 cột hẹp ngách bên** (2 bên × 5 tầng) mà bộ tiền xử lý
hiện tại của em **đang bỏ đi** — em chỉ giữ 3 trong 5 điều kiện.

Hallinan [1] đạt κ = 0,92 cho ngách bên bằng Axial T2, cho thấy nhãn này học được.

**Đề xuất:** giá trị thật của axial là **thêm 2 điều kiện mới**, không phải nâng chỉ số
của 3 điều kiện cũ (axial gần như không mang thông tin về canal hay foraminal). Vì vậy
em xin trình bày axial theo hướng **mở rộng không gian nhãn 3 → 5 điều kiện**, đúng mạch
đóng góp của bài báo, thay vì hứa nó tăng F1.

Chi phí: phải viết mới bộ trích xuất crop axial, dataloader và đầu ra subarticular,
ước 3–5 ngày. Và Mean F1 trên 5 điều kiện **không so trực tiếp** với Mean F1 trên 3
điều kiện, nên phải báo cáo thành bảng riêng.

---

## Phần 2 — Cơ chế Leader–Supporter

### 2.1 Vì sao cần trọng số, thay vì ghép thẳng

Sau khi có cả hai chuỗi, cách đơn giản nhất là **ghép phẳng** (late fusion / concat):
lấy embedding của T2 và của T1, nối lại thành vector 1024 chiều, cho qua một lớp phân
lớp. Nhược điểm: mô hình phải tự ngầm học chuyện "chuỗi nào quan trọng hơn", và **trọng
số đó là cố định cho mọi bệnh nhân**.

Trên thực tế mức hữu ích của từng chuỗi **thay đổi theo từng ca**: có ca T1 rõ mỡ quanh
rễ, có ca ảnh T1 nhiễu hoặc lệch tầng thì T2 lại đáng tin hơn.

### 2.2 Cơ chế đã cài đặt (cổng GMU theo từng điều kiện)

Em dùng cổng kiểu Gated Multimodal Unit [2], đặt **riêng cho mỗi điều kiện**:

```
Với điều kiện c và mỗi chuỗi s ∈ {T2, T1}:
    a(c,s) = MLP( [ e_c ; emb_s ] )        # e_c: vector truy vấn học được của điều kiện c
    w(c,s) = softmax_s( a(c,s) )           # softmax trên các chuỗi, tổng = 1
    fused_c = Σ_s  w(c,s) · emb_s
    logits_c = head_c( fused_c )
```

Giải thích bằng lời:

- **`e_c` là "câu hỏi" của từng điều kiện.** Đầu foraminal và đầu canal hỏi hai câu khác
  nhau, nên chấm điểm hai chuỗi khác nhau.
- **`w` là trọng số Leader–Supporter.** Chuỗi có trọng số cao hơn là *leader*, chuỗi kia
  là *supporter*. Vì `w` tính từ embedding **của chính ca đó**, trọng số **thay đổi theo
  từng bệnh nhân**, không phải một hằng số toàn cục.
- **Khởi tạo theo tiên nghiệm lâm sàng.** Bias khởi tạo nghiêng về T1 cho foraminal
  (`[T2 0.0, T1 1.0]` trước softmax), nhưng vẫn là tham số học được, không khoá cứng.
  Smoke test cho gate foraminal ra `[T2 0.27, T1 0.73]`, khớp tiên nghiệm.
- **Canal không có cổng**: nó chỉ dùng T2, vì trong thiết kế này không có crop T1 nào
  tương ứng với đường giữa.

### 2.3 Giá trị chính là **giải thích được**, và em xin nói trước một giới hạn

Hàm `get_gate_weights()` trả về cặp trọng số (T2, T1) **cho từng ca**, nên có thể vẽ
phân bố trọng số theo điều kiện, hoặc chỉ ra những ca mà mô hình tự chuyển sang tin T1.
Đây là thứ em nghĩ thầy muốn nhìn thấy.

**Giới hạn cần nói trước:** cổng là một **tổ hợp lồi** (trọng số cộng lại bằng 1), nên
vector hợp nhất chỉ có **512 chiều**, trong khi ghép phẳng giữ cả hai và có **1024
chiều**. Về dung lượng biểu diễn, **cổng thấp hơn ghép phẳng**. Nếu kết quả cho thấy
gated thua concat về F1 thì đó là điều dự đoán được, không phải lỗi cài đặt; khi đó em
sẽ trình bày gated bằng giá trị *giải thích được*, còn concat bằng chỉ số.

Nếu cần mạnh hơn cổng vô hướng, bước tiếp theo là **cross-attention giữa các view** như
M-SCAN [3] làm trên đúng bộ 1.975 study em đang dùng (AUROC 0,971). Em xin lưu ý con số
đó là **chỉ cho hẹp ống sống**, tức điều kiện dễ nhất, nên không phải bằng chứng rằng
multi-view cứu được foraminal.

### 2.4 Trạng thái thực nghiệm

| Cấu hình | Trạng thái 22/08 |
|---|---|
| T2-only | mốc hiện có (Mean F1 0.527, Severe F1 0.356) |
| + T1, ghép phẳng (concat) | 🔵 **đang chạy GPU**, 20 epoch, ~100 phút |
| + T1, cổng leader–supporter (gated) | ⏳ chạy ngay sau |

Kết quả epoch 1 của cấu hình concat đã cho thấy bức tranh rõ: **canal Severe F1 0.5188
ngay epoch đầu** (vì warm-start từ mô hình cũ), còn hai đầu foraminal mới ở 0.095 và
0.145 (recall 0.062 / 0.108) vì chúng là đầu ra mới khởi tạo. Nên toàn bộ kết quả của
thí nghiệm này phụ thuộc vào việc hai đầu foraminal có leo lên được hay không:

| Foraminal đạt (mỗi bên) | Mean Severe F1 | So với 0.356 hiện tại |
|---|---|---|
| 0.28 | 0.360 | ngang |
| **0.30** | **0.373** | **vượt** |

Mà Phần 1.3 vừa cho thấy foraminal trên T1 chạm được 0.29–0.31. Nên đây là phép cộng có
cơ sở, không phải kỳ vọng suông.

---

## Phần 3 — Bài toán mất cân bằng, điều chỉnh thông số, highlight bất thường

### 3.1 Đã làm gì

Lớp Severe chiếm dưới 5% dữ liệu. Hệ hiện tại đã chồng **sáu** kỹ thuật:

| Kỹ thuật | Vai trò |
|---|---|
| Focal loss (γ = 2,0) | giảm trọng số các mẫu dễ |
| Trọng số lớp theo căn bậc hai | bù tần suất |
| Oversampling ×3 lớp thiểu số | tăng số lần lớp hiếm xuất hiện |
| Supervised contrastive (λ = 0,1) | kéo các mẫu cùng lớp lại gần nhau |
| Trọng số đa nhiệm theo độ bất định | cân ba nhiệm vụ mà không đặt tay |
| **Tinh chỉnh ngưỡng hậu kiểm** | đổi quy tắc quyết định, không cần huấn luyện lại |

Hiệu quả tổng: Severe F1 **0,152 → 0,356** (2,3 lần), Severe Recall **12,3% → 48,6%**
(4 lần) so với backbone gốc. Đây là kết quả đã có trong bài báo MIWAI.

### 3.2 Điều chỉnh thông số: kết quả và điều nó tiết lộ

Bước tinh chỉnh ngưỡng (thí nghiệm #0) đáng chú ý vì **không cần huấn luyện lại**, chỉ
đổi ngưỡng quyết định trên các checkpoint đã có:

| | Mean F1 macro | Chi phí RSNA (1:2:4) |
|---|---|---|
| CBAM | **+0,039** | — |
| Hybrid | **+0,019** trên held-out | 0,314 → 0,285 |

Cơ sở để làm bước này: chỉ số RSNA 2024 tự nó gán trọng số sai lệch
Normal-Mild : Moderate : Severe = **1 : 2 : 4**, tức bản thân hàm chi phí đã nói rằng
quy tắc argmax chi phí đều là **sai** cho bài toán này.

**Nhưng phát hiện quan trọng hơn nằm ở chỗ nó KHÔNG sửa được:** tinh chỉnh ngưỡng cứu
được lớp Severe của canal nhưng **không cứu được foraminal**. Điều đó loại trừ giả
thuyết "chỉ là sai ngưỡng", và chỉ ra phần còn thiếu nằm ở **biểu diễn đặc trưng** —
đúng là lý do em chuyển sang Phần 1 và Phần 2.

### 3.3 Khảo sát tài liệu: còn nên chỉnh gì nữa

Em có khảo sát để tránh làm trùng thứ đã hết dư địa. Kết quả đáng chú ý nhất là
MONICA [4]: benchmark **hơn 30 phương pháp long-tail trên 12 bộ dữ liệu y tế thuộc 6
lĩnh vực**, kết luận rằng **chất lượng khởi tạo / tiền huấn luyện quan trọng hơn các hàm
mất mát long-tail tinh vi**.

Điều này **trùng khớp với kết quả thực nghiệm của chính em** ở mục 3.2. Hai nguồn độc
lập cùng chỉ một hướng: **phần chỉ số còn lại phải lấy bằng biểu diễn tốt hơn (đúng
chuỗi ảnh, đa chuỗi), không phải bằng thêm mẹo mất mát.**

Nếu thầy muốn thử thêm về phía mất cân bằng, còn hai hướng chưa dùng, có bằng chứng trên
dữ liệu y tế và chi phí thấp:

| Hướng | Cơ sở |
|---|---|
| **GCL** — điều chỉnh logit bằng nhiễu Gauss [5] | phương pháp có điểm trung bình cao nhất trong MONICA |
| **cRT** — huấn luyện hai giai đoạn, tách biểu diễn khỏi bộ phân lớp | MONICA khuyến nghị rõ mô hình hai giai đoạn |

Ngoài ra vừa có công trình rất mới (19/08/2026) về **hồi quy có thứ bậc cho đúng bài toán
hẹp lỗ liên hợp trên MRI sagittal** [6]. Em đã xác minh công trình tồn tại nhưng chưa lấy
được toàn văn để đối chiếu số liệu; em sẽ đọc trước khi quyết có dùng ordinal hay không.

Một điểm kỹ thuật em đang theo dõi: `compute_class_weights` hiện gộp nhãn của **cả ba
điều kiện** rồi tính một bộ trọng số dùng chung. Canal có tỉ lệ Severe khác foraminal,
nên trọng số ra là số dung hoà (`Severe = 2,081` khi gộp, so với `2,252` khi tính riêng
foraminal). Nếu đến epoch 5 mà recall của foraminal vẫn rất thấp, em sẽ thử **trọng số
riêng cho từng điều kiện**.

### 3.4 Highlight bất thường

Phần này **đã có sẵn và đang dùng**: bản đồ Grad-CAM theo từng đĩa đệm, so sánh
có/không có CBAM (hình `grad_cam_nocbam_vs_cbam.png` trong báo cáo), và đã tích hợp vào
phần mềm gán nhãn để bác sĩ nhìn được vùng mô hình chú ý.

Với cấu hình gated ở Phần 2, em có thêm một tầng giải thích nữa: ngoài "mô hình nhìn vào
đâu" (Grad-CAM), còn có "mô hình tin chuỗi ảnh nào" (trọng số leader–supporter).

---

## Tóm tắt cho thầy

| Câu hỏi của thầy | Trả lời ngắn |
|---|---|
| Tận dụng T1/T2/axial chưa? | T2 ✅ từ đầu. **T1 ✅ mới làm trong Phase 3**, cho ngang tới hơi nhích. **Axial ❌ chưa**, dữ liệu sẵn 2.340 chuỗi + 10 cột nhãn subarticular đang bỏ trống |
| Leader–Supporter là gì | Cổng GMU theo từng điều kiện, học trọng số T2/T1 **theo từng bệnh nhân**, khởi tạo theo tiên nghiệm lâm sàng. Code xong, đang chạy. Giá trị chính là **giải thích được**; dung lượng biểu diễn thấp hơn ghép phẳng nên có thể thua về F1 |
| Mất cân bằng còn chỉnh được gì | Đã dùng 6 kỹ thuật (Severe F1 0,152 → 0,356). Tinh chỉnh ngưỡng thêm +0,019/+0,039 miễn phí. Literature nói dư địa phía hàm mất mát **đã cạn**; phần còn lại ở biểu diễn. Còn GCL + cRT nếu muốn thử |
| Highlight bất thường | ✅ Grad-CAM đã có, đã vào phần mềm |

**Việc còn lại của Phase 3:** chạy xong concat + gated (đang làm) → mở 3 seed cho cấu
hình tốt nhất → viết chương Thực nghiệm → quyết định axial theo thời gian còn lại.

---

## Tài liệu tham khảo (đã kiểm tra tồn tại)

[1] Hallinan, J.T.P.D., Zhu, L., Yang, K., Makmur, A., Algazwi, D.A.R., Thian, Y.L.,
Lau, S., Choo, Y.S., Eide, S.E., Yap, Q.V., Chan, Y.H., Tan, J.H., Kumar, N., Ooi, B.C.,
Yoshioka, H., Quek, S.T.: Deep learning model for automated detection and classification
of central canal, lateral recess, and neural foraminal stenosis at lumbar spine MRI.
*Radiology* **300**(1), 130–138 (2021). https://doi.org/10.1148/radiol.2021204289

[2] Arevalo, J., Solorio, T., Montes-y-Gómez, M., González, F.A.: Gated multimodal units
for information fusion. arXiv:1702.01992 (2017).

[3] Batra, A., Gumber, A., Kumar, A.: M-SCAN: a multistage framework for lumbar spinal
canal stenosis grading using multi-view cross attention. arXiv:2503.01634 (2025).

[4] Ju, L., Yan, S., Zhou, Y., Nan, Y., Xing, X., Duan, P., Ge, Z.: MONICA: benchmarking
on long-tailed medical image classification. arXiv:2410.02010 (2024).

[5] Li, M., Cheung, Y., Lu, Y.: Long-tailed visual recognition via Gaussian clouded logit
adjustment. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)*, pp. 6929–6938 (2022).

[6] Phadke, R.A., Salman, S.G., Salman, Z.G., Marupudi, A., Patel, K., Ong, J.,
Tavakkoli, A., Galhotra, S., Tripuraneni, A., Rizkalla, J., Lee, N.J.: Ordinal deep
learning for lumbar foraminal stenosis grading on sagittal MRI. *Journal of Imaging*
**12**(8), 388 (2026). https://doi.org/10.3390/jimaging12080388

### Mức độ xác minh

- [1] xác minh đầy đủ qua tóm tắt gốc (168 lượt trích dẫn); câu "Axial T2-weighted and
  sagittal T1-weighted images were used" và các giá trị κ lấy trực tiếp từ tóm tắt.
- [2][3][4][5] xác minh tiêu đề, tác giả, năm, mã định danh.
- [6] xác minh qua Crossref (tiêu đề, tác giả, tạp chí, tập/số/số bài, ngày xuất bản);
  **chưa lấy được toàn văn** nên chưa trích số liệu nào từ công trình này.
- **Chưa xác minh, chưa dùng:** bài viết giải pháp hạng nhất cuộc thi Kaggle RSNA 2024
  (trang chặn truy cập tự động); một bài trên *Frontiers in Medicine* 2026 về học sâu
  nhận biết giải phẫu cho phân độ thoái hoá thắt lưng.
