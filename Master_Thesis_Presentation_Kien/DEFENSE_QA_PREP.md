# Chuẩn bị câu hỏi phản biện — Báo cáo Thực tập 2

> Đề tài: Ứng dụng AI trong gán nhãn y tế cho bệnh đau thắt lưng
> HV: Hà Trung Kiên (2470723) — GVHD: TS. Phan Trọng Nhân
> File này gom các câu thầy/hội đồng có thể hỏi + câu trả lời chuẩn bị sẵn. Bổ sung dần khi discuss.

---

## A. Số liệu & cách tính metric

### A1. AUC của Modic và Pfirrmann trong bảng zero-shot tính thế nào? (vì sao trước đây để trống)
- Modic (4 lớp) và Pfirrmann (5 lớp) là bài toán **đa lớp**, nên không có AUC nhị phân trực tiếp như các nhãn 2 lớp khác.
- Đã tính **AUC macro one-vs-rest (OvR)**: với mỗi lớp, coi "lớp đó vs phần còn lại" thành bài nhị phân, dùng **điểm tương đồng văn bản–ảnh của BiomedCLIP** làm score, tính AUC rồi lấy trung bình các lớp.
- Nguồn: tính trực tiếp từ `experiments/paper_results/spider_zeroshot/predictions.csv` (các cột `*_sim_classK`) trên cùng tập 1439 mẫu của lần chạy zero-shot.
- Kết quả: **Modic 0.422**, **Pfirrmann 0.492**, trung bình 8 nhãn **0.699**.
- **Ý nghĩa (điểm cần nhấn):** hai giá trị gần 0.5 (≈ ngẫu nhiên) ⟶ khẳng định nhãn **xa ngữ nghĩa** với dữ liệu pretrain chuyển giao **kém** trong điều kiện zero-shot; ngược lại nhóm nhãn về đĩa đệm (bulging/herniation/narrowing) AUC ~0.8 vì gần ngữ nghĩa hơn. Đây chính là động lực phải **fine-tune** (mục kết quả có tinh chỉnh).

### A2. Vì sao Severe Recall trên RSNA có chỗ ghi 46.4%, chỗ khác liên quan tới 37%?
- **46.4%** = Recall **gộp (pooled)** toàn bộ mẫu Severe của cả 3 condition — đây là số dùng nhất quán trong Tóm tắt/Kết luận/bảng tổng hợp/bảng chi tiết.
- Con số ~37% (nếu nhắc) là **trung bình macro 3 condition** = (70.4 + 18.8 + 21.7)/3 — cách tính khác, không sai, nhưng để tránh nhầm lẫn báo cáo **thống nhất dùng pooled 46.4%**.
- Cải thiện theo từng condition: Spinal Canal +34.6%, L. Foraminal +18.8%, R. Foraminal +21.7%.

### A3. Vì sao RSNA chỉ chạy 1 seed?
- (Đã bỏ disclaimer "hạn chế GPU" khỏi slide/report.) Nếu hỏi: tập RSNA full train+val lớn, mỗi cấu hình huấn luyện tốn nhiều giờ GPU; phần đánh giá độ ổn định (multi-seed mean±std) đã được thực hiện ở nhánh transfer SPIDER. Hướng tiếp theo: chạy multi-seed cho RSNA.

### A4. Vì sao Mean Accuracy của Hybrid (0.721) THẤP hơn SpineNetV2 baseline (0.814)?
- **Accuracy là chỉ số gây hiểu lầm trên dữ liệu mất cân bằng nặng.** Lớp Normal/Mild chiếm ~85%, nên một mô hình chỉ cần đoán "Normal/Mild" cho hầu hết mẫu là đã đạt Accuracy ~81% — nhưng **bỏ sót gần 9/10 ca nặng** (baseline Severe Recall chỉ 11.5%).
- Hybrid **chủ động đánh đổi ~9% Accuracy** để bắt được ca nặng: Severe Recall 11.5% → 46.4% (gấp 4 lần). Đồng thời vẫn dẫn đầu các chỉ số ít bị lệch bởi mất cân bằng: **Mean F1 macro (0.528), Recall (0.592), Precision (0.517), AUC (0.837), AUPRC (0.526)** — thắng 5/6 chỉ số trung bình.
- Câu chốt: *"Trong sàng lọc lâm sàng, bỏ sót ca nặng nguy hiểm hơn nhiều so với báo động nhầm ca nhẹ; vì vậy em tối ưu F1 macro + Severe Recall thay vì Accuracy."*

### A5. Vì sao bảng chi tiết RSNA chia theo MỨC ĐỘ (Normal/Mild–Moderate–Severe), còn SPIDER chia theo BỆNH LÝ (Modic, thoát vị...)? Có mâu thuẫn không?
- Không — hai cách chia phản ánh đúng cấu trúc khác nhau của 2 bộ dữ liệu:
  - **RSNA**: 1 loại bài toán (hẹp) được phân thành **3 mức độ nặng** → chia theo mức độ để soi lớp Severe (nhóm cần ưu tiên phát hiện).
  - **SPIDER**: gồm **8 bệnh lý riêng biệt**, đa số nhị phân, không có thang severity chung → chia theo từng bệnh để xem khả năng **chuyển giao** (nhóm đĩa đệm chuyển giao tốt, nhóm xa ngữ nghĩa kém).
- RSNA không có "8 loại bệnh" để chia; SPIDER không có "thang nặng nhẹ chung" cho cả 8 nhãn. Ép cùng một trục sẽ sai bản chất dữ liệu.
- Hai bảng trả lời 2 câu hỏi khác nhau: RSNA → "có bắt được lớp nặng hiếm không?"; SPIDER → "bệnh nào chuyển giao tốt?".

---

## B. Kiến trúc & phương pháp

### B1. Vì sao chọn CBAM? (không phải attention khác)
- (điền sau khi discuss)

### B2. Vì sao chọn BiomedCLIP làm nhánh đông cứng (frozen)?
- (điền sau khi discuss)

### B3. MRI volume 9×112×224 nghĩa là gì?
- 9 lát cắt Sagittal liền kề quanh đĩa đệm, mỗi lát 112×224 px sau chuẩn hóa; gom thành khối 3D đầu vào CNN. (chi tiết bổ sung sau)

### B4. Vì sao ghép (concat) đặc trưng + dùng MLP head?
- (điền sau khi discuss)

---

## C. Dữ liệu & phạm vi

### C1. RSNA vs SPIDER khác nhau gì về nhãn? Vì sao 3 vs 8 nhãn?
- (điền sau khi discuss)

### C2. Vì sao đánh giá 3/5 condition của RSNA?
- (điền sau khi discuss)

---

## D. Hạn chế & hướng phát triển

### D1. Hai mô-đun (phân vùng + grading) chưa tích hợp — kế hoạch?
- ĐATN: dùng mặt nạ TotalSpineSeg làm input định vị ROI cho grading → pipeline end-to-end, rồi xây phần mềm giao diện bác sĩ.

### D2. Vì sao chưa dùng mặt cắt Axial?
- (điền sau khi discuss)

---

## TODO discuss với Claude
- Hoàn thiện B1–B4, C1–C2, D2.
- Rà câu hỏi "đóng góp khoa học của thực tập là gì" và cách trả lời ngắn gọn.
