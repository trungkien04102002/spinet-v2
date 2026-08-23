# HANDOFF — MIWAI paper revision (lưu trước compact, 2026-06-09)

## ⭐ CẬP NHẬT 2026-06-09 (mới nhất, sau khi đọc bài LumbarCLIP tham khảo)
- **Đã tạo `main_pair_v2.tex`** (file mới, `main_pair.tex` = backup, KHÔNG đụng). Build OK, 13 trang. Resolve **TOÀN BỘ 13 dấu `\textcolor{red}`** + feedback WhatsApp:
  design-first abstract/intro; câu "Our model realizes..." đưa lên design-level; contributions 4 bullet (fix cite he2016resnet, tổng quát số nhãn); thêm đoạn structure; §3.1 notation C×D×H×W (số dời §4.1); §3.3 role-first "Volumetric attention branch / Frozen multimodal branch"; §3.4 tách concept↔đẩy hyperparameter xuống §4; "Models compared"; framework→"Model".
- **Ảnh đã regenerate:** `grad_cam_severe_canal.png` (1 hàng canal, font to — qua `viz/grad_cam_rsna.py --canal-only`); `rsna/spider_class_distribution_en.png` (font 19–21, legend RSNA lên đỉnh — qua `figures/make_dataset_distribution_en.py`). v2 đã trỏ tới ảnh mới.
- **File hướng dẫn gõ cho meeting:** `CHANGES_FOR_MEETING.md` (before→after + lý do, từng mục A–H). ⚠️ Lên Overleaf nhớ **upload đè 3 PNG** + gõ theo section (không Ctrl+A).
- **12 trang:** chưa đạt (13), references còn ~17 mục tràn trang 13; cần thân bài ngắn ~½ trang → để dành cho pass generalization/readability sau meeting.
- **CÒN LẠI sau meeting:** generalization 3 trục sâu hơn + readability §3.3/§4.3/§5 + bridge câu (mục "CHƯA LÀM" bên dưới). Meet thầy 20h 09/06.


## Bối cảnh
- Paper MIWAI 2026 tại `paper/lncs_hk252/`. Bản đang làm = **`main_pair.tex`** (bản mới nhất của thầy Nhân). `main.tex` = bản cũ (backup, committed). Cả hai dùng `\bibliography{references}`.
- Thầy đang edit chung trên **Overleaf** → KHÔNG Ctrl+A paste cả file; dán theo từng section. Hiện build local **13 trang** (cần về 12; restructure sẽ giúp; nếu MIWAI không extend → nộp **CSONet**).
- Meeting với thầy **20h 09/06/2026**.

## FEEDBACK CỦA THẦY (đầy đủ)
1. **Design-first, implementation-second:** đóng góp là THIẾT KẾ; mô tả thiết kế bằng từ tổng quát trước, rồi mới nói hiện thực bằng tool có sẵn. VD "BiomedCLIP branch" → "multimodal branch" (impl: BiomedCLIP); "CBAM-3D" → "volumetric attention branch".
2. **Tổng quát hóa 3 trục — đừng gắn chặt:**
   - (A) component (multimodal branch ↔ BiomedCLIP, attention ↔ CBAM)
   - (B) dataset (RSNA/SPIDER = "validate on", không phải phạm vi)
   - (C) số nhãn (method chạy K nhãn bất kỳ; 3 RSNA / 8 SPIDER chỉ là ca test)
   - Lý do: method là zero-shot expansion → dùng dataset/nhãn nào cũng được.
3. **Cite mọi concept/method/dataset** được nhắc tên (đã thêm RSNA, SPIDER, CBAM, Grad-CAM, CLIP).
4. **§3.4 là "rừng"** — lẫn imbalance + method + hyperparameter + công thức + ghi chú trong 1 đoạn → cấu trúc lại (đọc dễ như abstract).
5. **Text rườm rà, khó đọc, ý chưa liên kết** → restructure.
6. **Tham khảo** `01_Miwai 2025_paper LNCS 2025.pdf` (LumbarCLIP, cùng nhóm thầy) — chỗ hay, không copy.
7. **Kỷ luật của tôi (đã thống nhất):** method viết tổng quát NHƯNG evidence giữ scoped "we validate on RSNA→SPIDER" (mới test 1 cặp, không over-claim).

## BÀI HỌC TỪ LumbarCLIP (giải pháp cho §3.4 "rừng")
Họ TÁCH 3 lớp: **Kiến trúc** (§3.3, role-first bullet) / **Hàm mục tiêu** (§3.4, công thức tách) / **Hyperparameters** (§4 "Training Configuration"). 
→ Fix §3.4 mình: **đẩy lr/batch/epoch/patience xuống §4**; §3.4 chỉ giữ concept imbalance (focal/weight/oversampling, mỗi cái 1 câu "vì sao"). Component viết "role: implementation".

## ĐÃ LÀM (đã apply vào `main_pair.tex` + ghi trong `CHANGES_TO_PASTE.md`)
1. §1 motivation reframe design-first + cite `woo2018cbam`, `zhang2023biomedclip`, `radford2021clip`.
2. §1 contributions: 4 bullet dẫn bằng giá trị (finding/design/capability/recipe).
3. §4.5: thêm `\cite{selvaraju2017gradcam}`.
4. `references.bib`: thêm entry `selvaraju2017gradcam` (ĐÂY là thay đổi references.bib DUY NHẤT còn giữ; full-name expansion đã REVERT).
5. Abstract: "SpineNetV2-style" → "standard geometric and intensity augmentation"; "Severe F1 to 0.356" → "from 0.152 to 0.356".
6. §6 Summary reframe design-first + bỏ "three-seed".
7. Title (main_pair.tex đã là của thầy: "A Two Branch Hybrid Multimodal Model Enhancing Imbalanced Lumbar Disc Grading and Zero-Shot Label Extension").

## CHƯA LÀM (chờ chốt sau meeting 20h — user đã OK hướng, chờ scope)
- **§3.4**: tách concept ↔ đẩy hyperparameter xuống §4.
- **§3.3**: đổi role-first ("Volumetric attention branch / Frozen multimodal branch / Fusion head").
- **Generalize 3 trục** xuyên Abstract/Contributions/§4.3 (giữ evidence scoped).
- **Readability**: tách câu §3.3/§4.3/§5 + câu bắc cầu giữa 3 thí nghiệm.
- ⚠️ Giữ 12 trang: restructure kiểu tách/gọt, KHÔNG thêm chữ.

## references.bib — vụ full tên tác giả (HOÃN)
- Venue đã official hết (BiomedCLIP→NEJM AI, SpineNetV2→Sci Reports, Woerner→LNCS, +DOI, Liu metadata, Grad-CAM).
- Tên tác giả đang để **et al** (baseline). Full-name từng bị tràn 13 trang → đã revert. Thầy OK "ref dài thì dùng et al". → khi nội dung ngắn lại mới thêm full tên cho ref ngắn (≤9 tg), giữ et al cho CLIP(12)+BiomedCLIP(24).

## FILE MAP
- `main_pair.tex` — bản thầy + 7 edit trên.
- `main.tex` — bản cũ committed (title đã update, commit 83843da).
- `references.bib` — baseline + Grad-CAM.
- `CHANGES_TO_PASTE.md` — 7 block dán Overleaf (before/after, "tìm gì → dán gì").
- `MEETING_PREP_20260609.md` — kế hoạch revise đầy đủ (3 trục, readability, bài học LumbarCLIP, before/after, câu hỏi cho thầy).
- `01_Miwai 2025_paper LNCS 2025.pdf` — paper tham khảo (tr.20-32).

## RÀNG BUỘC / SỞ THÍCH
- Chat tiếng Việt, file tiếng Anh. Đừng bịa số/citation (verify CrossRef). 
- Mỗi lần sửa LaTeX → tự build + render trang cho user xem.
- Overleaf chung với thầy → dán theo section, không Ctrl+A.

## BƯỚC TIẾP THEO khi resume
User đã đồng ý để tôi viết 1 version revise đầy đủ trên `main_pair.tex` + note ra. Đang chờ user chốt **scope** (full một lượt hay ưu tiên §3 method trước) → rồi apply 4 nhóm "CHƯA LÀM" + build + xuất note before/after.
