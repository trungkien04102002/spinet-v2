# Meeting prep — 20h 09/06/2026 (thầy Nhân)

> Trạng thái: **CHƯA sửa paper**, đây là note để bàn. Sửa sau khi meet chốt.
> Nếu MIWAI không extend → nộp **CSONet** (framing tổng quát càng hợp).

---

## NGUYÊN LÝ CHỐT của thầy: **Design-first, implementation-second**
Contribution của paper là **THIẾT KẾ (design)**, không phải công cụ. Nên:
1. Mô tả thiết kế bằng **từ tổng quát** trước (vai trò của từng khối).
2. Rồi mới nói **hiện thực** bằng công cụ/mô hình có sẵn (BiomedCLIP, CBAM, ResNet…) sau.
3. Không gắn chặt method vào **dataset cụ thể** (RSNA/SPIDER) hay **số nhãn cụ thể** (3/8).

→ 3 trục cần tổng quát hóa: **(A) component, (B) dataset, (C) số nhãn.**

---

## Từ vựng tổng quát ↔ hiện thực (dùng nhất quán toàn paper)
| Hiện thực (implementation) | Từ tổng quát (design) |
|---|---|
| BiomedCLIP (ViT-B/16) | **(frozen) multimodal branch** / vision-language foundation branch |
| CBAM | **channel-spatial attention** |
| CBAM-3D ResNet-34 | **volumetric attention encoder** (attention-equipped 3D CNN) |
| concat-MLP | **learned fusion head** |
| RSNA 2024 | **source dataset** |
| SPIDER | **target dataset** (label schema khác) |
| 3 conditions / 8 labels | **an arbitrary label set (size K)** |

Quy tắc: phần **Abstract / Intro / Contributions / mở đầu §3** dùng từ tổng quát (kèm 1 lần "implemented with …"); phần **§3 chi tiết + §4 experiments** mới dùng tên cụ thể (đó là chỗ hiện thực, hợp lý).

---

## (A) Trục COMPONENT — "multimodal branch", không "BiomedCLIP branch"

**Abstract** — before/after:
- *Before:* "fuses a 3D ResNet-34 with Convolutional Block Attention Modules (the CBAM-3D branch) and a frozen BiomedCLIP image-text encoder."
- *After:* "fuses a **volumetric attention branch** with a **frozen multimodal (vision-language) branch**, joined by a small learned fusion head; we implement these with a CBAM-3D ResNet-34 and BiomedCLIP, respectively."

**§3.3 sub-headers** — đổi tên khối theo VAI TRÒ, tên tool đưa vào body:
- "CBAM-3D ResNet-34 branch." → **"Volumetric attention branch."** (body: "We implement it as a 3D ResNet-34 with CBAM…")
- "BiomedCLIP branch." → **"Frozen multimodal branch."** (body: "We use the frozen BiomedCLIP ViT-B/16…")
- "Fusion head." → giữ (đã tổng quát).

**§1 contribution #2** — "frozen BiomedCLIP prior" → "frozen **multimodal** prior (implemented with BiomedCLIP)".

---

## (B) Trục DATASET — RSNA/SPIDER chỉ là ca kiểm chứng
- §3.1 (Problem Formulation) **đã tổng quát rồi** (encoder Φ dùng chung, "label set là input của inference") → chỉ cần **nâng** ý này lên Abstract/Contribution.
- Mỗi chỗ nêu RSNA→SPIDER: đóng khung **"we validate on …"**, không phải phạm vi của method.

---

## (C) Trục SỐ NHÃN — method chạy với K nhãn bất kỳ
- Cơ chế = cosine với **K text prompt**, K tùy ý → đổi tập nhãn (size nào cũng được) chỉ là đổi prompt.
- "3 conditions" / "8 labels" = ca đã test, không phải giới hạn.

**Contribution #3 — before/after (gộp B + C):**
- *Before:* "lets the RSNA-trained model grade eight unseen SPIDER labels zero-shot (mean F1 0.362)…"
- *After:* "Because labels are scored by matching the image embedding to text prompts, **a model trained on one schema can grade an arbitrary new label set — of any size — zero-shot**, with no new head or retraining. We validate this by transferring an RSNA-trained model to eight unseen SPIDER labels (mean F1 0.362); the same mechanism applies to any future label vocabulary."

---

## Readability / liên kết ý (thầy: "rườm rà, khó đọc")
1. **§3.4 (Training Objective / imbalance pipeline)** — thầy nói thẳng "đọc vô không dễ hiểu, **lẫn lộn imbalance + method + thông số kỹ thuật**". Đoạn này nhồi: công thức focal loss → class weight (15×→4×) → oversampling → augmentation → **hyperparameters (AdamW, lr 1e-4, batch 32, 20 epoch, patience-5)** — tất cả 1 đoạn.
   → **Fix:** tách rõ 3 ý: **(i) chiến lược xử lý imbalance** (focal + class weight + oversampling, mỗi cái 1 câu ngắn, nói *vì sao*), **(ii) augmentation** riêng, **(iii) training hyperparameters** gom riêng (1 câu cuối, hoặc đẩy xuống "implementation details") — KHÔNG trộn thông số kỹ thuật vào dòng mô tả concept.
2. **§3.3 (Architecture)** dày đặc (4 khối công thức liên tiếp, ít câu dẫn) → thêm topic sentence + mô tả vai trò trước công thức.
3. **§4.3 (zero-shot)** nhồi 5 finding trong 1 đoạn → tách ý / câu ngắn hơn.
4. **§5 "Interpreting CBAM"** đoạn quá dài → tách 2 ý (finding vs trivial-ensemble control).
5. **Bắc cầu giữa 3 thí nghiệm** (§4.2 in-domain → §4.3 zero-shot → §4.4 supervised): thêm câu nối "mỗi thí nghiệm trả lời câu hỏi gì".
- ⚠️ Đang **13 trang** → smoothing kiểu **tách/gọt cho dễ đọc, KHÔNG thêm chữ** (tách câu/đổi cấu trúc, không phình).

---

## Bài học từ paper tham khảo (LumbarCLIP, MIWAI 2025 — cùng nhóm thầy Nhân)
File: `01_Miwai 2025_paper LNCS 2025.pdf` (tr.20-32). Style này = gu thầy.

**1. TÁCH method thành 3 phần rõ (mình đang gộp → "rừng"):**
- Kiến trúc (design) → §3.3 dạng **role-first bullet**.
- Hàm mục tiêu (công thức) → §3.4 riêng, công thức tách bạch (họ tách I2T / T2I / total loss).
- **Hyperparameters (lr, batch, epoch, patience) → ĐẨY XUỐNG §4 "Training Configuration"**, KHÔNG để trong §3.4.
→ Đây chính là fix cho §3.4 "lẫn lộn imbalance + method + thông số" của mình.

**2. Component viết role-first** (vai trò tổng quát trước, tool sau):
> "**Image encoder:** A backbone (ResNet-50, ViT, or Swin) encodes…"
> "**Text encoder:** We used pretrained BioClinicalBERT…"
Áp cho mình: "**Volumetric attention branch:** implemented with 3D ResNet-34 + CBAM…" / "**Frozen multimodal branch:** implemented with BiomedCLIP…" / "**Fusion head:** a small learned MLP…"

**3. §3.4 mình nên rút còn:** 1 câu mở (4-part imbalance strategy) → focal (1 câu, vì sao) → class weight (1 câu) → oversampling (1 câu) → augmentation (1 câu); **bỏ hết hyperparameter** (đưa xuống §4).

**4. Câu mở §3 của họ ngắn gọn** ("In this study, we introduce LumbarCLIP, a framework that…") + Figure overview ngay → mình giữ kiểu này.

(Thầy dặn: tham khảo chỗ hay, KHÔNG cần làm giống y.)

---

## Câu hỏi chốt với thầy lúc meet
1. Mức tổng quát: dừng ở Abstract/Intro/Contribution + header §3.3, hay tổng quát sâu hơn?
2. §3.3 có đổi sub-header sang từ tổng quát không (Volumetric attention branch / Frozen multimodal branch)?
3. Ưu tiên gọt readability section nào trước (đề xuất: §3.3 + §4.3)?
4. MIWAI extend hay nộp CSONet → ảnh hưởng deadline sửa.
