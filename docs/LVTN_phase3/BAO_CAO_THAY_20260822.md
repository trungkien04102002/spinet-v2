# Báo cáo tiến độ — giai đoạn cải thiện F1 (22/08/2026)

> Trả lời 4 gợi ý thầy đưa: (1) ảnh axial, (2) tín hiệu T1/T2 leader–supporter,
> (3) pipeline, (4) bài toán mất cân bằng + highlight bất thường.
> Kèm khảo sát tài liệu (mọi trích dẫn trong file này đã kiểm tra tồn tại thật; mục nào
> chưa kiểm được em ghi rõ ở cuối).

## Tóm tắt

Giai đoạn này em tìm ra **một chỗ lệch protocol trong pipeline của chính mình**: hệ thống
đang chấm hẹp lỗ liên hợp (foraminal) trên chuỗi **Sagittal T2**, trong khi chuẩn lâm sàng
và các công trình tham chiếu đều chấm trên **Sagittal T1**. Đây đúng là điều kiện có F1 tệ
nhất trong 3 điều kiện.

Em đã dựng xong dữ liệu và mã nguồn cho 3 thí nghiệm sửa việc này (T1-foraminal, late
fusion, gated leader–supporter). **Chưa có số mới** — cả 3 đang chờ chạy GPU (~6 giờ máy).

Khảo sát tài liệu cho một kết luận đáng chú ý: **phần F1 còn thiếu nhiều khả năng nằm ở
biểu diễn đặc trưng (đọc đúng chuỗi ảnh), không nằm ở hàm mất mát**. Hai nguồn độc lập
cùng chỉ hướng này (chi tiết mục 4).

---

## 1. Pipeline — đã tìm ra chỗ sai

`rsna_dataloader.py` lọc chuỗi **Sagittal T2 cho mọi đầu ra**, kể cả hai đầu foraminal
trái/phải. Nhưng trong bộ RSNA 2024, toạ độ nhãn foraminal nằm trên chuỗi **Sagittal T1**,
và trên slice khác với slice giữa dùng cho ống sống.

Cơ sở lâm sàng: lỗ liên hợp là cấu trúc **bên**, không phải giữa; mỡ quanh rễ thần kinh
hiện rõ trên T1. Công trình tham chiếu của Hallinan và cộng sự trên *Radiology* [1] — hệ
thống học sâu cho đúng ba loại hẹp này — dùng **Axial T2 và Sagittal T1**, không dùng
Sagittal T2.

**Đã làm:** trích xuất **19.689** crop T1 theo từng bên (`rsna_preprocessed_t1/`), khớp
99,9% với chỉ mục T2 hiện có.

**Ý nghĩa:** đây không phải một phép thử may rủi mà là **sửa cho khớp chuẩn**. Kể cả F1
không tăng, việc phát hiện và sửa chỗ lệch này vẫn là kết quả báo cáo được.

## 2. Leader–supporter (T1/T2 có trọng số)

Đã cài đặt đúng hướng thầy gợi ý: mỗi chuỗi một encoder riêng, rồi **cổng GMU theo từng
điều kiện** học trọng số cho từng chuỗi **trên từng bệnh nhân**, khởi tạo nghiêng theo tiên
nghiệm lâm sàng (T1 dẫn dắt cho foraminal). Cổng vẫn học được, không cố định.

Ba cấu hình để so sánh (bảng ablation cho chương Thực nghiệm):

| Cấu hình | Nội dung |
|---|---|
| T2-only | mốc hiện tại |
| + T1 (concat) | ghép phẳng hai embedding |
| + T1 (gated) | cổng leader–supporter |

**Một điểm em xin nói trước để tránh hiểu nhầm khi đọc kết quả:** cổng GMU là **tổ hợp
lồi** (trọng số cộng lại bằng 1) nên vector hợp nhất có 512 chiều, trong khi concat giữ cả
hai và có 1024 chiều. Về mặt dung lượng biểu diễn, **gated thấp hơn concat**. Nếu gated
thua concat về F1 thì đó là kết quả bình thường, không phải lỗi. Giá trị riêng của gated là
**giải thích được**: xuất ra được trọng số T1/T2 cho từng ca, dùng làm hình minh hoạ.

**Nếu gated không đủ tốt**, bước tiếp theo là thay cổng vô hướng bằng **cross-attention
giữa các view**, như M-SCAN [2] làm trên đúng bộ 1.975 study mà em đang dùng.

> Ghi chú trung thực: M-SCAN báo AUROC 0,971 nhưng **chỉ cho hẹp ống sống**. Ống sống vốn
> đã là điều kiện dễ nhất của em (AUPRC 0,56 so với 0,21–0,23 của foraminal), nên con số đó
> **không phải bằng chứng** rằng multi-view cứu được foraminal.

## 3. Ảnh axial — khả thi, và mở ra 2 nhãn mới

Em đã kiểm dữ liệu thật:

| Hạng mục | Số lượng |
|---|---|
| Chuỗi Axial T2 | **2.340** (nhiều hơn cả hai chuỗi sagittal) |
| Chuỗi Sagittal T1 | 1.980 |
| Chuỗi Sagittal T2/STIR | 1.974 |

Quan trọng hơn: bảng nhãn gốc của RSNA 2024 có **10 cột hẹp ngách bên (subarticular
stenosis)** — 2 bên × 5 tầng — mà bộ tiền xử lý hiện tại của em **đang bỏ đi**. Em chỉ giữ
3 điều kiện (ống sống + foraminal trái/phải) trong tổng số 5.

Hallinan và cộng sự [1] đạt κ = 0,92 cho ngách bên bằng **Axial T2**, cho thấy nhãn này học
được.

**Đề xuất cách đóng gói:** giá trị thật của axial là **thêm 2 điều kiện mới**, không phải
nâng F1 của 3 điều kiện cũ — axial gần như không mang thông tin về ống sống hay foraminal.
Vì vậy em xin trình bày axial theo hướng **mở rộng không gian nhãn** (3 → 5 điều kiện),
đúng mạch đóng góp của bài báo, thay vì hứa nó tăng F1.

**Chi phí:** phải viết mới bộ trích xuất crop axial, dataloader và đầu ra cho subarticular —
ước lượng 3–5 ngày. Và **Mean F1 trên 5 điều kiện không so trực tiếp được với Mean F1 trên
3 điều kiện**, nên phải báo cáo thành bảng riêng.

## 4. Bài toán mất cân bằng

Hiện tại pipeline đã dùng **sáu** kỹ thuật chồng lên nhau: focal loss (γ = 2,0), trọng số
lớp theo căn bậc hai, oversampling ×3, supervised contrastive, trọng số đa nhiệm theo độ bất
định, và tinh chỉnh ngưỡng hậu kiểm.

Khảo sát tài liệu cho kết quả em thấy đáng lưu ý nhất trong cả đợt này. MONICA [3] đánh giá
**hơn 30 phương pháp long-tail trên 12 bộ dữ liệu y tế thuộc 6 lĩnh vực**, và kết luận rằng
**chất lượng khởi tạo / tiền huấn luyện quan trọng hơn các hàm mất mát long-tail tinh vi**.

Điều này **trùng khớp với kết quả thực nghiệm của chính em**: bước tinh chỉnh ngưỡng (không
cần huấn luyện lại) sửa được lớp Severe của ống sống nhưng **không cứu được foraminal** — tức
là chỗ thiếu không nằm ở quy tắc quyết định mà nằm ở **đặc trưng**.

Hai nguồn độc lập cùng chỉ một hướng: **phần F1 còn lại phải lấy bằng biểu diễn tốt hơn
(đúng chuỗi ảnh, đa chuỗi), không phải bằng thêm mẹo mất mát.** Đây cũng chính là lý do em
ưu tiên mục 1–3 ở trên.

**Nếu thầy muốn thử thêm về phía mất cân bằng**, còn hai hướng chưa dùng và có bằng chứng
trên dữ liệu y tế, chi phí thấp:

| Hướng | Cơ sở |
|---|---|
| **GCL** (điều chỉnh logit nhiễu Gauss) [4] | phương pháp có điểm trung bình cao nhất trong MONICA |
| **cRT** (huấn luyện hai giai đoạn, tách biểu diễn khỏi bộ phân lớp) | MONICA khuyến nghị rõ mô hình hai giai đoạn |

Ngoài ra vừa có một công trình rất mới (19/08/2026) về **hồi quy có thứ bậc cho đúng bài
toán hẹp lỗ liên hợp trên MRI sagittal** [5]. Em đã xác minh công trình này tồn tại nhưng
chưa lấy được toàn văn để đối chiếu số liệu; em sẽ đọc trước khi quyết có dùng ordinal
không.

**Highlight bất thường:** phần này đã có sẵn — bản đồ Grad-CAM theo từng đĩa, đã dùng trong
báo cáo và trong phần mềm gán nhãn.

---

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Tinh chỉnh ngưỡng (không huấn luyện) | ✅ Xong — macro +0,019 (Hybrid), +0,039 (CBAM) |
| Trích xuất crop T1 | ✅ Xong — 19.689 crop |
| Mã nguồn 3 thí nghiệm T1 / concat / gated | ✅ Xong, đã kiểm thử trơn tru đầu-cuối |
| **Chạy GPU 3 thí nghiệm** | ⬜ **Chưa — đây là việc chặn mọi thứ phía sau** |
| So sánh SOTA ngoài | ✅ Xong — 2 mô hình × 3 seed, cùng split |
| Axial / subarticular | ⬜ Chưa bắt đầu |

**Em xin nói rõ: hiện chưa có con số F1 mới nào.** Kết quả F1 thật gần nhất là bước tinh
chỉnh ngưỡng. Toàn bộ công sức giai đoạn này đang ở dạng *cơ chế đã dựng xong nhưng chưa
chạy*.

## Việc tiếp theo

1. **Chạy 3 thí nghiệm trên GPU** (~6 giờ máy) → có bảng ablation T2-only → +T1 → +gated.
2. Nếu T1-foraminal thắng: mở rộng 3 seed để khoá số cuối cùng.
3. Viết chương Phương pháp + Thực nghiệm theo số thực tế.
4. Quyết định axial dựa trên thời gian còn lại và kết quả bước 1.

---

## Tài liệu tham khảo (đã kiểm tra tồn tại)

[1] Hallinan, J.T.P.D., Zhu, L., Yang, K., Makmur, A., Algazwi, D.A.R., Thian, Y.L., Lau,
S., Choo, Y.S., Eide, S.E., Yap, Q.V., Chan, Y.H., Tan, J.H., Kumar, N., Ooi, B.C.,
Yoshioka, H., Quek, S.T.: Deep learning model for automated detection and classification of
central canal, lateral recess, and neural foraminal stenosis at lumbar spine MRI.
*Radiology* **300**(1), 130–138 (2021). https://doi.org/10.1148/radiol.2021204289

[2] Batra, A., Gumber, A., Kumar, A.: M-SCAN: a multistage framework for lumbar spinal canal
stenosis grading using multi-view cross attention. arXiv:2503.01634 (2025).

[3] Ju, L., Yan, S., Zhou, Y., Nan, Y., Xing, X., Duan, P., Ge, Z.: MONICA: benchmarking on
long-tailed medical image classification. arXiv:2410.02010 (2024).

[4] Li, M., Cheung, Y., Lu, Y.: Long-tailed visual recognition via Gaussian clouded logit
adjustment. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)*, pp. 6929–6938 (2022).

[5] Phadke, R.A., Salman, S.G., Salman, Z.G., Marupudi, A., Patel, K., Ong, J., Tavakkoli,
A., Galhotra, S., Tripuraneni, A., Rizkalla, J., Lee, N.J.: Ordinal deep learning for lumbar
foraminal stenosis grading on sagittal MRI. *Journal of Imaging* **12**(8), 388 (2026).
https://doi.org/10.3390/jimaging12080388

[6] Cao, K., Wei, C., Gaidon, A., Aréchiga, N., Ma, T.: Learning imbalanced datasets with
label-distribution-aware margin loss. In: *Advances in Neural Information Processing Systems
(NeurIPS)*, vol. 32 (2019).

### Ghi chú về mức độ xác minh

- [1] xác minh đầy đủ (tóm tắt gốc, 168 lượt trích dẫn) — câu "Axial T2-weighted and
  sagittal T1-weighted images were used" và các giá trị κ lấy trực tiếp từ tóm tắt.
- [2][3][4][6] xác minh tiêu đề, tác giả, năm, mã định danh.
- [5] xác minh qua Crossref (tiêu đề, tác giả, tạp chí, tập/số/số bài, ngày xuất bản).
  **Chưa lấy được toàn văn** nên chưa trích dẫn số liệu nào từ công trình này.
- **Chưa xác minh, chưa dùng:** bài viết giải pháp hạng nhất của cuộc thi Kaggle RSNA 2024
  (trang bị chặn truy cập tự động); một bài trên *Frontiers in Medicine* 2026 về học sâu
  nhận biết giải phẫu cho phân độ thoái hoá thắt lưng.
