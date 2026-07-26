# Feedback-loop: khảo sát phương pháp (deep-research 2026-07-26)

> ## ✅ ĐÃ CODE XONG 2026-07-26 — đọc mục này là đủ, phần dưới là nền lý thuyết
>
> **User chốt phạm vi:** không cần đánh giá mạnh (McNemar/bootstrap/BWT chính quy), mục tiêu là
> **CÓ cách tiếp cận, có làm, có impact** — hơn là không có gì. Đánh giá nặng → Future Work.
> **CHƯA viết gì vào luận văn** (`paper/lvtn_overleaf/`) — chạy xong hết mới viết report.
>
> ### Sửa gì (`spine-labeling-app/backend/app/feedback/retrain_head.py`)
> Code cũ đã có sẵn head-only fine-tune + eval before/after + cổng "chỉ lưu nếu tốt hơn".
> Thiếu đúng 3 chỗ, đã vá:
>
> | # | Lỗ | Vì sao chết | Đã sửa |
> |---|---|---|---|
> | 1 | `model.train()` làm **BN running stats vẫn chạy** — `freeze_backbone()` chỉ tắt gradient | Cập nhật BN bằng 20–50 mẫu = đúng cơ chế gây quên (arXiv:2011.08096) | `_freeze_batchnorm()`, gọi **SAU** `model.train()` mỗi epoch (train() sẽ bật lại BN) |
> | 2 | Train **chỉ trên correction** | Không có mẫu cũ → quên. Replay thắng EWC trong ảnh y tế | `build_train_dataset()` + cờ `--replay-dir` (mặc định 200 mẫu cũ, seed cố định) |
> | 3 | Cổng "improved" dùng **`mean_acc`** | **Đúng bẫy đã dính ở run T1 26/07: acc 76% mà Severe F1 = 0.0.** Model giết lớp Severe vẫn "cải thiện accuracy" | `evaluate()` trả thêm `macro_f1` + `severe_f1`; cổng đổi sang **macro-F1** |
>
> ### Test: `backend/tests/test_retrain_guards.py` — 11 test, phủ cả 3 guard
> Full suite: **91 passed / 1 skipped** (trước 80). Có test chứng minh guard là load-bearing
> (BN *có* trôi nếu bỏ guard) và test tái hiện đúng bẫy accuracy-che-Severe.
>
> ### Cách chạy
> ```bash
> cd ~/spine-labeling-app/backend
> ./.venv/bin/python -m app.feedback.retrain_head \
>     --checkpoint models/weights/phase2_cbam.pth \
>     --dataset-dir <correction-set> \
>     --holdout-dir <held-out ĐÓNG BĂNG> \
>     --replay-dir  <mẫu train gốc> \
>     --out-checkpoint models/weights/phase2_cbam_v2.pth
> ```
> In ra `Before / After / Delta` theo macro_f1 + severe_f1. Quên `--replay-dir` → có cảnh báo.
>
> ### ✅ Vòng 2 (cùng ngày) — luồng "bác sĩ bấm gửi" + khoá held-out + đổi phiên bản model
>
> **① Luồng GỬI (`backend/app/routers/feedback.py`)** — trước đây app *có* capture (`correction_log`)
> nhưng **không có đường gửi**: correction nằm im, bác sĩ không biết có được nhận không.
> Tách **lưu** (sửa hồ sơ bệnh nhân) khỏi **gửi** (đóng góp làm dữ liệu học) — bác sĩ chủ động bàn giao,
> không phải ống ngầm từ bệnh án vào training set.
> - `correction_log` + 2 cột `submitted_at`, `batch_id` (migration additive, đã test trên bảng có sẵn data)
> - `GET /corrections` (đang chờ / đã gửi / số lô) · `POST /corrections/submit` (đóng 1 lô, trả `batch-<ts>-<hash>`)
> - **Gửi ≠ train.** Chỉ xếp hàng; train offline; model mới phải được bấm duyệt mới vào phục vụ.
> - FE: panel "Doctor feedback" từ chữ tĩnh → panel thật (số đang chờ + nút Gửi + thông báo lô)
>
> **② Đổi/hoàn tác phiên bản model (`app/feedback/model_registry.py`)** — mọi checkpoint nằm cạnh nhau
> trong `models/weights/`, con trỏ `ACTIVE` (text) chọn cái đang phục vụ. **Baseline không bao giờ bị ghi đè**
> → hoàn tác luôn khả dụng, nên mới dám thử update. Chặn path traversal (tên đến từ HTTP).
> `GET /model/versions` · `POST /model/activate` · `POST /model/revert`. `retrain_head` ghi thêm
> sidecar `<ckpt>.pth.json` để UI đọc số before/after mà không phải nạp 240MB.
>
> **③ Khoá held-out (`app/feedback/freeze_holdout.py`)** — lấy mẫu **phân tầng theo tổ hợp nhãn**
> (rút đều tay dễ trượt sạch lớp Severe → mù đúng lớp cần đo), ghi `MANIFEST.json` có **sha256 + timestamp + seed**.
> `--verify` phát hiện được nếu bị sửa. Từ chối freeze đè (mọi số before/after cũ sẽ mất tính so sánh)
> và từ chối nuốt trọn source.
> ```bash
> python -m app.feedback.freeze_holdout --source-dir <data> --out-dir data/holdout_frozen --n 150
> python -m app.feedback.freeze_holdout --verify data/holdout_frozen
> ```
>
> **Test:** `test_feedback_flow.py` (17) + `test_retrain_guards.py` (11) + migration `correction_log` (1).
> **Full suite 108 passed / 1 skipped** (trước 80). FE lint + build sạch.
> **Đã verify chạy thật** trên DB thật: thấy đúng 2 correction anh tạo 07/07 (spider_100), migration không mất data,
> revert OK, path traversal trả 404.
>
> ### ⚠️ CÒN LẠI
> - **Chạy `freeze_holdout` thật một lần** — code xong rồi nhưng **chưa chạy**. Phải chạy TRƯỚC khi bác sĩ
>   đụng vào app. Nguồn data: crop RSNA ở `spinet-v2/rsna_preprocessed/` (cần convert sang shape
>   `build_dataset` hoặc trỏ thẳng nếu đã đúng format `volumes/` + `train_metadata.csv`).
> - Bấm thử nút Gửi trên UI (em cố ý không bấm hộ — đó là 2 correction thật của anh).
>
> ### Dự phòng nếu không mời được bác sĩ thật
> Lấy N ca model đoán sai trong held-out, "sửa" bằng nhãn thật → chạy nguyên pipeline.
> Ghi rõ **"correction mô phỏng"** trong luận văn. Vẫn kiểm chứng được cơ chế.
>
> ### Câu sẽ viết trong luận văn (sau, khi có số)
> *"Cơ chế cập nhật từ phản hồi bác sĩ đã được hiện thực và kiểm chứng end-to-end: với N ca được
> sửa, macro-F1 trên tập held-out đóng băng thay đổi từ X lên Y, không suy giảm ở các lớp cũ.
> Cỡ mẫu chưa đủ để kết luận ý nghĩa thống kê — đây là hạn chế đã biết và là hướng phát triển."*
>
> ---

> Kết quả chạy `academic-research-skills:deep-research` chế độ `lit-review`, 4 agent song song.
> Bổ sung cho [`FEEDBACK_LOOP_RESEARCH.md`](FEEDBACK_LOOP_RESEARCH.md) (bản 2026-07-18, thiên về
> "có nên làm không"). File này trả lời: **"làm thì làm bằng cách nào"**.
>
> Bối cảnh: app đã host, bác sĩ sửa mask + sửa nhãn severity → dùng để model học.
> Ràng buộc: **~20–50 ca được sửa**, cơ chế phải đơn giản.

---

## 0. Kết luận trước (đọc cái này nếu chỉ đọc 1 mục)

**Với 20–50 mẫu, KHÔNG thể chứng minh model tốt lên một cách có ý nghĩa thống kê.**
Đây không phải bi quan — đây là điều literature nói thẳng: hầu hết paper continual-learning y tế
dùng N=100–1000; vùng N=20–50 **chưa được validate trong ảnh y tế**. Ngưỡng tối thiểu để có
macro-F1 bảo vệ được là ~50–100 mẫu **cho riêng lớp thiểu số**.

→ **Đổi mục tiêu luận văn**, từ:
- ❌ "Feedback của bác sĩ làm model tốt lên" (không chứng minh nổi, hội đồng bắt ngay)

thành:
- ✅ **"Xây dựng và kiểm chứng CƠ CHẾ cập nhật an toàn"**: model học được từ correction,
  **không quên** kiến thức cũ (đo bằng BWT/forgetting), có cổng kiểm duyệt của người,
  có thể rollback. Cải thiện đo được thì báo cáo kèm CI; không đủ mạnh thì nói thẳng là
  hạn chế do cỡ mẫu.

Cái này **mạnh hơn** về mặt học thuật vì nó khớp đúng khung pháp lý FDA PCCP (mục 5).

---

## 1. Cập nhật trọng số bằng cách nào (N nhỏ)

Xếp theo **bằng chứng trong ảnh y tế**, không phải theo độ "kêu".

| Hạng | Phương pháp | Cơ chế | Cần bao nhiêu data | Rủi ro quên | Chi phí code |
|---|---|---|---|---|---|
| **1** | **Đóng băng BN + train head** | Giữ nguyên BN statistics toàn cục, chỉ train head/affine | 20–50 OK | Thấp | 1–2 dòng |
| **1** | **Replay buffer nhỏ** (40–500 mẫu cũ) | Trộn mẫu cũ vào batch khi train mẫu mới | 20–50 mới + buffer | Rất thấp | ~30 dòng |
| **2** | **cRT / τ-norm** (Kang ICLR2020) | Train lại RIÊNG classifier với class-balanced sampling | 20–50 | Thấp (backbone không đụng) | ~40 dòng |
| **2** | **LP-FT** (Kumar ICLR2022) | Linear-probe trước, rồi mới fine-tune | 20–50 | Trung bình | ~20 dòng |
| **3** | **LoRA / adapter** | Ma trận low-rank cắm vào layer, backbone đóng băng | ≥50 tốt hơn | Thấp | thư viện có sẵn |
| **3** | **EWC** (Kirkpatrick PNAS 2017) | Phạt thay đổi trọng số quan trọng, trọng số hóa bằng Fisher | Cần data cũ để tính Fisher | Trung bình | ~50–80 dòng |
| **4** | **LwF** (Li & Hoiem) | Distill output model cũ làm soft target | 20–50 | Thấp | ~20 dòng |
| ✗ | **Full fine-tune** | Train hết | ≥500 | **Rất cao** | — |
| ✗ | **Generative replay (GAN)** | Sinh lại data cũ | — | — | 100+ dòng, chưa validate y tế |

### Ba phát hiện đáng giá

**(a) Đóng băng BatchNorm là đòn rẻ nhất và có bằng chứng y tế mạnh nhất.**
Paper *"The unreasonable effectiveness of Batch-Norm statistics in addressing catastrophic
forgetting across medical institutions"* (arXiv:2011.08096): chỉ fine-tune **affine params của BN**
đã cho kết quả **ngang full fine-tune** trên 7 dataset y tế × 4 kiến trúc, hội tụ nhanh hơn.
Với batch nhỏ (N=20–50), BN statistics tính từ mini-batch cực nhiễu → đóng băng là bắt buộc.
Chi phí: `bn.eval()` — đúng nghĩa 1 dòng.

**(b) Replay ĐÁNH BẠI EWC trong ảnh y tế — ngược với trực giác từ CV tổng quát.**
Nghiên cứu incremental learning trên da liễu (PMC8469804) cho thấy replay-based **thắng EWC ở MỌI
hyperparameter và mọi memory size**. Buffer hiệu quả báo cáo: 500 ảnh → acc 86.04%, forgetting 0.0344;
chest X-ray peak ở buffer ~1000. Lý do nghi ngờ: data y tế mất cân bằng nặng, EWC không xử lý được.
→ **Đừng mặc định chọn EWC chỉ vì nó nổi tiếng.** Với 20–50 mẫu, Fisher matrix ước lượng không đáng tin.

**(c) Fine-tune có thể làm HỎNG feature đã pretrain.**
Kumar et al. ICLR 2022: full fine-tune được +2% in-distribution nhưng **−7% out-of-distribution** so với
linear probing; LP-FT (probe trước, fine-tune sau) được +1% ID và **+10% OOD**. Với update tí hon,
linear probe đơn thuần thường thắng.

---

## 2. Hướng KHÔNG cần gradient — rất hợp ca này

Đây là chỗ em nghĩ anh nên chú ý nhất, vì nó **triệt tiêu hoàn toàn rủi ro quên**.

**Tip-Adapter (Zhang et al., ECCV 2022)** — cache key-value **không train gì cả**: lưu embedding của
các mẫu few-shot làm "key", nhãn làm "value", lúc suy luận thì trộn kết quả zero-shot của CLIP với
kết quả tra cứu cache. 16-shot ImageNet: 62.03% **không train**, 65.51% nếu fine-tune 10 epoch.

**Vì sao hợp mô hình của anh:** hệ thống đã có **BiomedCLIP đóng băng**. Correction của bác sĩ
→ nạp thẳng vào cache → model "nhớ" ngay lập tức, **không đụng một trọng số nào**, và **xoá cache là
rollback tức thì**. Không thể có catastrophic forgetting vì không có gì bị ghi đè.

Cùng họ: **Prototypical Networks** (Snell et al. 2017) và Nearest-Class-Mean — cộng thêm prototype
cho ca bị sửa. **BiomedCoOp** (arXiv:2411.15232) học prompt cho BiomedCLIP: 4-shot 68.1% → 16-shot 77.1%.

> ⚠️ Caveat: Tip-Adapter mạnh khi CLIP zero-shot vốn đã khá. Trên task rất chuyên biệt
> (severity 3 lớp cột sống), zero-shot BiomedCLIP yếu → gain có thể khiêm tốn. Phải đo, đừng hứa trước.

---

## 3. Correction về MASK dùng thế nào cho model GRADING?

Đây là chỗ đề tài dễ hớ: app có **2 model** (TotalSpineSeg segmentation + CBAM grading), bác sĩ sửa mask
là sửa cho model seg — **không tự động giúp model grading**.

Ba cách nối, theo độ chắc chắn giảm dần:

1. **Mask sửa → định nghĩa lại vùng crop** (chắc nhất, rẻ nhất). Model grading ăn crop per-IVD
   (9,112,224). Mask đúng hơn → crop đúng hơn → input sạch hơn. Cơ chế rõ ràng, dễ giải thích.
2. **Attention supervision** — ép attention map của classifier khớp mask (CAMAL, LGM-ViT).
   Hợp với CBAM vì CBAM vốn là attention. Nhưng bằng chứng còn mỏng.
3. **Multi-task seg+cls chung backbone** — đắt, không nên trong khuôn khổ LVTN.

> ⚠️ **Bằng chứng thưa:** literature cho thấy mask hướng dẫn classification lúc **pre-training** thì có,
> còn "correction mask online cải thiện classifier" thì **gần như chưa ai chứng minh định lượng**.
> Nói trong luận văn phải cẩn thận, đừng khẳng định quá.

**Phân biệt cực quan trọng cho phần related-work** (hội đồng rất dễ hỏi):

| Kiểu | Ví dụ | Có cập nhật trọng số? |
|---|---|---|
| **Interactive refinement** | SAM, MedSAM, ScribblePrompt, nnInteractive | ❌ Không — chỉ điều kiện hoá suy luận theo click |
| **Click-conditioned training** | DeepGrow, DeepEdit | ✅ Có, nhưng offline, mô phỏng click lúc train |
| **Continual HITL** | MONAI Label, HybridMS | ✅ Có, theo chu kỳ |

→ Cái anh làm thuộc nhóm **thứ 3**. Đa số tool "AI hỗ trợ gán nhãn" đình đám thực ra chỉ là nhóm 1
(không học gì cả) — đây chính là **khoảng trống để anh định vị đóng góp**.

**Số liệu tham chiếu để so:** MedSAM giảm **82.4–82.9%** thời gian gán nhãn; HybridMS (Sci Rep 2025)
giảm 82% (10–15 phút → ~2.5 phút), Dice 0.9538 vs 0.9435 của MedSAM; ScribblePrompt giảm 28% + Dice +15%.
→ Nếu anh **đo thời gian gán nhãn tiết kiệm được**, đó là con số dễ đạt và dễ bảo vệ hơn nhiều so với F1.

---

## 4. Correction của bác sĩ KHÔNG phải ground truth

Đây là đòn phản biện mạnh nhất hội đồng có thể tung ra, nên phải chủ động xử lý.

**Nghiên cứu 3 người đọc trên MRI thắt lưng thoái hoá (PMC7011264):**
- Pfirrmann: Cohen's κ từng cặp **0.60–0.76**, tổng thể **κ = 0.68** (chỉ ở mức *moderate*)
- Gwet's AC2 có trọng số: 0.90 (cao hơn vì hiệu chỉnh prevalence)
- Spondylolisthesis κ > 0.80; thoái hoá diện khớp ~0.75

→ **κ=0.68 nghĩa là hai bác sĩ bất đồng ~32% số ca.** Một bác sĩ "sửa" nhãn của model không có nghĩa
model sai. Nếu nuốt hết correction vào train, có khi đang dạy model **thiên vị theo 1 người**.

**Cách xử lý (theo Confident Learning — Northcutt, Jiang & Chuang, JAIR 2021 / cleanlab):**
- Lọc correction mâu thuẫn với dự đoán mà model **đang rất tự tin** → đánh dấu để review, không nuốt thẳng
- Lý tưởng: ≥2 bác sĩ độc lập/ca. Không có thì **phải ghi vào phần Hạn chế**
- Versioned annotation + `correction_log` (app đã có) chính là cơ chế truy vết cần thiết

---

## 5. Khung pháp lý — món quà cho phần biện luận

**FDA, guidance chính thức 12/2024:** *"Marketing Submission Recommendations for a Predetermined
Change Control Plan for Artificial Intelligence-Enabled Device Software Functions"* (PCCP).
Cơ chế: nhà sản xuất **khai báo trước** phạm vi thay đổi được phép (tần suất, giới hạn data, protocol
validate, quy trình rollback) → sau đó update trong phạm vi đó chỉ cần **thông báo**, không xin duyệt lại.

**Điểm mấu chốt:** PCCP áp cho **batch update offline**, KHÔNG áp cho online learning liên tục.
EU AI Act (hiệu lực 01/08/2024) + MDR cũng đòi giám sát của người + hậu kiểm.

→ **Thiết kế "batch, offline, có người duyệt, rollback được" của anh KHÔNG phải là làm cho đơn giản —
nó là thiết kế ĐÚNG theo chuẩn quản lý.** Câu này nên nói thẳng trong luận văn và lúc bảo vệ. Nó lật
ngược một hạn chế thành một lựa chọn thiết kế có căn cứ.

> ⚠️ Verify lại trên fda.gov trước khi trích (agent lấy từ bản tóm tắt của hãng luật).

---

## 6. Khi nào retrain + chứng minh thế nào

**Trigger:** đếm số correction (N≥30) **+ người bấm nút**. KHÔNG dùng drift detection tự động
(CUSUM/EWMA) — quá phức tạp cho LVTN và không có ground truth trực tuyến để đo tụt hiệu năng.
Ghi rõ trong luận văn là đã cân nhắc và loại vì lý do gì.

**Protocol đánh giá tối thiểu để bảo vệ được:**

1. **Held-out ĐÓNG BĂNG** — khoá trước khi có correction nào. Không bao giờ train lên nó.
2. **McNemar's test** (bắt cặp, so model cũ vs mới trên cùng test set) — đúng bài cho N nhỏ.
3. **Bootstrap 95% CI** cho ΔF1 (10.000 lần lấy mẫu lại). **CI trùm số 0 thì phải nói là chưa kết luận được.**
4. **Backward Transfer / forgetting** (Lopez-Paz & Ranzato, NeurIPS 2017): đo hiệu năng trên phần cũ
   sau khi update. **BWT âm = quên = loại bản update.**
5. **Per-class F1**, không phải accuracy — lớp Severe hiếm, accuracy che hết.

**Câu kết luận nên nhắm tới:** *"Cơ chế cập nhật không gây quên (BWT ≥ 0) và có xu hướng cải thiện
lớp X (ΔF1 = +0.0YY, CI [−a, +b]); cỡ mẫu chưa đủ để kết luận có ý nghĩa thống kê."*
Đây là phát biểu **trung thực và không thể bắt bẻ**.

---

## 7. Đề xuất triển khai (xếp theo ưu tiên)

App đã có `backend/app/feedback/` (build_dataset + retrain_head). Phần thiếu **không phải phần train**
— mà là **phần đánh giá và phần cổng chặn**.

### Bậc 0 — bắt buộc (~2–3 ngày, phần lớn code đã có)
- [ ] Đóng băng BN + train head, class-balanced sampling (cRT-style)
- [ ] Replay buffer: trộn ~200 mẫu train cũ vào mỗi lần update
- [ ] Held-out đóng băng + McNemar + bootstrap CI + **BWT**
- [ ] Nút "Áp dụng / Hoàn tác" trong app (cổng người duyệt + rollback)

### Bậc 1 — nên có (~2–3 ngày, đây là chỗ ăn điểm)
- [ ] **Cache adapter kiểu Tip-Adapter trên BiomedCLIP đóng băng** — không train, không quên, rollback tức thì.
      So với nhánh gradient → thành một **ablation thật sự** thay vì một demo.
- [ ] Cổng lọc nhiễu nhãn: cờ correction mâu thuẫn với dự đoán model đang rất tự tin (cleanlab)
- [ ] Mask sửa → crop lại → chấm lại (nối 2 model, cơ chế dễ giải thích)

### Bậc 2 — future work, ghi trong luận văn, không code
- EWC/LwF (bằng chứng y tế yếu hơn replay), LoRA (cần ≥50 mẫu), drift detection tự động,
  generative replay, multi-task seg+cls.

### Nếu KHÔNG mời được bác sĩ thật
Phương án dự phòng bắt buộc phải có: **mô phỏng correction** từ ground truth held-out
(lấy N ca model đoán sai, "sửa" bằng nhãn thật) → chạy nguyên pipeline. Ghi rõ **"corrections mô phỏng"**
trong luận văn. Vẫn kiểm chứng được cơ chế; chỉ mất phần bằng chứng về người dùng thật.

---

## 8. Hạn chế của chính khảo sát này

1. **Vùng N=20–50 gần như không có bằng chứng trực tiếp trong ảnh y tế** — mọi khuyến nghị ở đây là
   ngoại suy từ N=100–1000. Phải ghi vào phần Hạn chế.
2. **Không có paper nào nghiên cứu đúng bài toán này** (correction từ bác sĩ, per-IVD, 3 lớp severity,
   backbone lai). Vừa là khoảng trống (tốt cho novelty) vừa là rủi ro (không có mốc để so).
3. **Correction từ nhiều bác sĩ khác nhau** (domain shift giữa người đọc) chưa ai nghiên cứu.
4. Vài citation trong bảng dưới **chưa verify xong** — phải mở nguồn gốc trước khi trích vào LVTN.

---

## 9. Bảng nguồn + trạng thái kiểm chứng

| Nguồn | Dùng cho | Trạng thái |
|---|---|---|
| Kirkpatrick et al., PNAS 2017 — EWC | Continual learning | ✅ |
| **Aljundi et al., ECCV 2018 — MAS** | Continual learning | ⚠️ **agent ghi nhầm thành "Ritter et al." — ĐÃ SỬA, verify lại** |
| Li & Hoiem — LwF | Distillation | ⚠️ ECCV 2016 hay TPAMI 2018? kiểm venue |
| Rebuffi et al., CVPR 2017 — iCaRL | Replay | ✅ |
| Lopez-Paz & Ranzato, NeurIPS 2017 — GEM/BWT | Chỉ số forgetting | ✅ arXiv:1706.08840 |
| arXiv:2011.08096 — BatchNorm chống quên | Đóng băng BN | ✅ |
| Kang et al., ICLR 2020 — cRT/τ-norm | Mất cân bằng | ✅ (đã dùng ở #0 threshold) |
| Menon et al., ICLR 2021 — logit adjustment | Mất cân bằng | ✅ |
| Ren et al., NeurIPS 2020 — Balanced Softmax | Mất cân bằng | ✅ |
| Kumar et al., ICLR 2022 — LP-FT | Fine-tune méo feature | ✅ arXiv:2202.10054 |
| Snell et al., 2017 — Prototypical Networks | Few-shot không gradient | ✅ |
| Zhang et al., ECCV 2022 — Tip-Adapter | Cache adapter | ✅ arXiv:2207.09519 |
| arXiv:2411.15232 — BiomedCoOp | Prompt tuning BiomedCLIP | ⚠️ preprint |
| Budd, Robinson & Kainz, Med Image Anal 2021 — HITL survey | Related work | ✅ arXiv:1910.02923, DOI 10.1016/j.media.2021.102062 |
| Diaz-Pinto et al. — MONAI Label | HITL framework | ✅ arXiv:2203.12362, Med Image Anal 2024 |
| Diaz-Pinto et al. — DeepEdit | Click-conditioned | ✅ arXiv:2305.10655 |
| Ma et al. — MedSAM | Interactive refinement | ✅ arXiv:2304.12306, Nat Commun 2024 |
| Wong et al. — ScribblePrompt | Interactive refinement | ✅ arXiv:2312.07381, ECCV 2024 |
| Mohamed Ali et al., Sci Rep 2025 — HybridMS | Case study có số | ⚠️ verify tác giả/số |
| arXiv:2311.13964 — Deep Interactive Seg taxonomy | Related work | ✅ |
| Northcutt, Jiang & Chuang, JAIR 2021 — Confident Learning | Nhiễu nhãn | ✅ arXiv:1911.00068 |
| **PMC7011264 — inter-rater MRI thắt lưng (κ=0.68)** | Correction ≠ ground truth | ✅ **số quan trọng, đã có nguồn gốc** |
| FDA PCCP guidance 12/2024 | Khung pháp lý | ⚠️ **lấy bản gốc fda.gov** |
| Rajkomar et al., BMJ Health Care Inform 2022 (PMC9156743) | MLOps lâm sàng | ⚠️ verify tác giả |
| Kumari et al., arXiv:2312.17004 — survey CL y tế | Related work | ✅ |
| González et al., arXiv:2405.13482 — survey CL y tế | Related work | ✅ |
| PMC8469804 — incremental learning da liễu | Replay > EWC | ✅ |

**Khai báo AI:** khảo sát này thực hiện có hỗ trợ của công cụ AI (Claude + web search). Mọi trích dẫn
phải được kiểm chứng lại tại nguồn gốc trước khi đưa vào luận văn.
