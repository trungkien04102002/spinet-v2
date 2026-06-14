# Hướng dẫn sửa paper cho meeting (09/06/2026) — main_pair → main_pair_v2

> **Cách dùng:** Mỗi mục có **VỊ TRÍ → BEFORE → AFTER → VÌ SAO**.
> File đầy đủ đã viết sẵn ở `main_pair_v2.tex` (build OK, 13 trang). File này để **anh tự gõ lại từng chỗ trên Overleaf** và **giải thích được cho thầy** (không phải copy máy móc).
> Nguyên tắc xuyên suốt của thầy: **thiết kế trước (từ tổng quát) → hiện thực sau (tool); đừng liệt kê đồ có sẵn; tổng quát hóa, đừng gắn chặt RSNA/SPIDER/số nhãn.**

---

## A. ABSTRACT — viết lại theo design-first

**VỊ TRÍ:** đoạn Abstract.

**BEFORE (đang liệt kê tool ngay):**
> ...we present a two-branch hybrid multimodal model, that fuses a 3D ResNet-34 with Convolutional Block Attention Modules (the CBAM-3D branch) and a frozen BiomedCLIP image-text encoder. The CBAM-3D branch captures... while the BiomedCLIP branch supplies...

**AFTER (mô tả vai trò trước, tool đẩy về 1 câu sau):**
> We address these problems by design rather than by a single component. We propose a two-branch model in which a **volumetric attention branch** captures cross-slice anatomical context and a **frozen multimodal (vision-language) branch** supplies a semantic prior learned from large-scale biomedical pretraining; a small trainable head fuses the two streams into a text-aligned embedding. Because predictions are made by matching this embedding to text prompts, **the label set becomes an input to inference**, so the same trained model can grade a new label set with no extra head and no retraining. **We implement the two branches with a CBAM-equipped 3D ResNet-34 and a frozen BiomedCLIP encoder**, and we validate the design on the RSNA 2024 and SPIDER datasets. With focal loss, ... raises Severe-class recall from 12.3% to 48.6% and Severe F1 from 0.152 to 0.356 on RSNA 2024...

**VÌ SAO:** tên khối nói **vai trò** (volumetric attention / multimodal branch); tool gom vào *một* câu "We implement... with...". Câu "label set becomes an input to inference" = nói thẳng đóng góp **design** (zero-shot expansion), không gắn chặt RSNA/SPIDER.

---

## B. §1 INTRODUCTION

### B1. Bỏ 2 dấu `[done] <trich dan tham khao>`
**VỊ TRÍ:** đoạn motivation (đoạn 2), sau `~\cite{zhang2023biomedclip}` và sau `~\cite{radford2021clip}`.
→ **Xóa** cả 2 `\textcolor{red}{[done]...}`. (Citation đã có sẵn, chỉ là ghi chú nhắc việc — bỏ đi.)

### B2. Câu thầy gạch đỏ `<xem lai cau nay>` (thầy quote trong chat 23:44)
**VỊ TRÍ:** cuối đoạn motivation.

**BEFORE (liệt kê tool trong Intro — đây là chỗ thầy chê):**
> Our model realizes this as a CBAM-equipped 3D ResNet-34 fused with a frozen BiomedCLIP~\cite{zhang2023biomedclip} encoder through a small trainable head.

**AFTER (đưa lên mức design, tool dời xuống §3):**
> These two ideas form the two branches of our model, joined by a small trainable fusion head; we give the component-level realization in Section~\ref{sec3}.

**VÌ SAO:** Intro chỉ nói **ý thiết kế** (hai nhánh + head hợp nhất). Tên model cụ thể (CBAM-3D, BiomedCLIP) để dành §3 — đúng "thiết kế trước, hiện thực sau".

### B3. Contributions — `<kiem tra lai tung y>` (sửa cả 4 gạch đỏ)
**VỊ TRÍ:** khối `\begin{itemize}`.

- **Bullet 1** `<xem lai y nay>` → viết lại đúng như anh giải thích trong chat (attention overfit theo tập huấn luyện, foundation model ổn định lại khi đổi dataset):
> **A cross-dataset transfer finding.** On in-domain RSNA data, channel-spatial attention improves the minority Severe metrics; but when the model is transferred to a different dataset (SPIDER), the same attention falls below a plain SpineNetV2 backbone. Adding a frozen multimodal branch brings transfer performance back up (+0.034 over the attention-only variant, p=0.011). We read this as attention overfitting the training dataset, with the foundation model stabilizing the representation under dataset shift.

- **Bullet 2** `<o doan tren dung 17, ma o day lai co 2>` → **bỏ `he2016resnet`**, chỉ giữ `\cite{woo2018cbam}` cho khớp với đoạn motivation (trên chỉ cite 1):
> **A model design that reconciles minority sensitivity with cross-center robustness.** We couple a volumetric attention encoder~\cite{woo2018cbam} with a **frozen multimodal prior**~\cite{zhang2023biomedclip} through a small learned fusion; an ablation shows the two branches are complementary rather than redundant.

- **Bullet 3** → tổng quát hóa **số nhãn** (trục C của thầy), bằng chứng vẫn scoped:
> **Label-space extension without retraining.** Because labels are scored by matching the image embedding to text prompts, a model trained on one schema can grade **a new label set of arbitrary size** with no new head and no retraining. We validate this by transferring an RSNA-trained model to eight unseen SPIDER labels (mean F1 = 0.362), which the fixed-head baselines cannot do.

- **Bullet 4** `<kiem tra y cuoi voi thiet lap thi nghiem>` → giữ, số khớp Table 1 (12.3→48.6, đúng experimental setup):
> **An imbalance recipe that recovers the minority class.** Focal loss~\cite{lin2017focal}, square-root class weighting, minority oversampling, and augmentation together raise the near-zero Severe recall of standard training about 4× (12.3% to 48.6%) on RSNA 2024.

### B4. `<bo sung cau truc paper>`
**VỊ TRÍ:** ngay sau `\end{itemize}` (chỗ gạch đỏ).
**AFTER (thêm đoạn cấu trúc):**
> The rest of this paper is organized as follows. Section~\ref{sec2} reviews related work on lumbar disc grading and on vision-language models for medical imaging. Section~\ref{sec3} formulates the grading task and its zero-shot extension and describes the two-branch model. Section~\ref{sec4} reports in-domain results on RSNA 2024, zero-shot and supervised transfer to SPIDER, and a qualitative analysis. Section~\ref{sec:disc} discusses the cross-dataset behaviour and the limitations, and Section~\ref{sec5} concludes.

---

## C. §3 PROPOSED METHOD

### C1. §3.1 Problem Formulation — `<thay cac so cu the bang notation>`
**VỊ TRÍ:** dòng `Let $\mathcal{V} = \mathbb{R}^{1 \times 9 \times 112 \times 224}$ ... (here $d = 512$)`.

**AFTER (dùng ký hiệu tổng quát, số cụ thể dời xuống §4):**
> Let $\mathcal{V} = \mathbb{R}^{C \times D \times H \times W}$ be the space of per-IVD volumetric crops (one channel, $D$ sagittal slices of $H \times W$ pixels; the concrete dimensions are given in Section~\ref{exp1}).

Và bỏ "(here $d=512$)" — chỉ ghi "an L2-normalized embedding of dimension $d$". (Giá trị $1{\times}9{\times}112{\times}224$ và $d=512$ chuyển vào §4.1 "Training configuration".)

**VÌ SAO:** method tổng quát hóa (trục dataset/kích thước); số đo cụ thể là chi tiết thực nghiệm. (§3.2 cũng đổi `V \in \mathbb{R}^{1\times9\times112\times224}` → `V \in \mathcal{V}`.)

### C2. §3.3 — đổi tên section + viết **role-first** (bắt chước cách LumbarCLIP, KHÔNG copy)
**VỊ TRÍ:** `\subsection{Hybrid Two-Branch Framework <dung thong nhat framework hay model>}`.

- **Tên section** → thống nhất dùng "Model" (khớp tiêu đề bài): `\subsection{Two-Branch Hybrid Model}`.
- **Câu mở** (design-level):
> Figure~\ref{fig:arch} shows the model. The same per-IVD volume is encoded by two complementary branches whose outputs are merged by a small trainable head. We describe each block by its role first, then by its concrete realization.

- **Đổi heading 2 nhánh sang VAI TRÒ, tool đưa vào câu body:**
  - `\textbf{CBAM-3D ResNet-34 branch.}` → `\textbf{Volumetric attention branch.}` + mở đầu body: *"This branch encodes the 3D structure of a disc and focuses on the small region a lesion occupies. We implement it as a 3D ResNet-34 with ... a CBAM block~\cite{woo2018cbam} after each of the four stages ..."*
  - `\textbf{BiomedCLIP branch.}` → `\textbf{Frozen multimodal branch.}` + body: *"This branch supplies a semantic prior from large-scale image-text pretraining and provides the text-aligned space used for zero-shot extension. We implement it with the frozen BiomedCLIP ViT-B/16 encoder ..."*
  - `\textbf{Fusion head.}` → giữ.
  - `\textbf{Classification heads.}` → đổi `\textbf{Task adaptation.}` (nói rõ đây là chỗ thích ứng task: linear head cho RSNA / cosine cho zero-shot).

**VÌ SAO:** đúng yêu cầu "multimodal branch thay BiomedCLIP branch", vai trò nói trước, tool nói sau. (Trong `main_pair_v2.tex` các công thức CBAM/slice-pool/fusion giữ nguyên, chỉ đổi câu dẫn.)

### C3. §3.4 — gỡ "rừng": tách concept ↔ đẩy hyperparameter xuống §4
**VỊ TRÍ:** `\subsection{Training Objective and Class-Imbalance Workflow}` (đoạn thầy chụp bảo "1 rừng").

- **Tên section** → `Training Objective and Class-Imbalance Strategy` (bỏ chữ "Workflow").
- **Tách thành 4 đoạn ngắn, mỗi cái 1 ý + 1 lý do** (đọc như abstract):
  1. *Câu mở:* "We handle class imbalance with a strategy that acts at three points: the loss, the gradient weighting, and the sampler."
  2. **Focal loss** (giữ công thức (8)) + 1 câu vì sao: "(1−p_t)^γ down-weights easy samples so the gradient concentrates on the hard minority cases; missing labels (−1) excluded via ignore_index."
  3. **Class weighting:** "w_c ∝ 1/√n_c ... reduces the Severe-to-Normal gradient ratio from ~15× to ~4×, keeps training stable."
  4. **Minority oversampling:** "factor 3–5 at the sampler ... complementary to class weighting (oversampling changes how often a Severe sample is seen; w_c reweights its gradient once in the batch)."
  5. **Augmentation:** "adds geometric/intensity transforms common in spinal MRI. We give the exact transforms and all training hyperparameters in Section~\ref{exp1}."
- **XÓA khỏi §3.4** câu: *"We train with AdamW (lr 10⁻⁴, batch 32, weight decay 10⁻⁴) for up to 20 epochs with patience-5..."* và chi tiết augmentation cụ thể → **chuyển xuống §4.1** (xem D2).

**VÌ SAO:** đây là fix chính thầy muốn — §3.4 chỉ giữ **chiến lược + công thức**; **thông số kỹ thuật (lr/batch/epoch/patience + augment cụ thể) dồn xuống §4** (đúng cách LumbarCLIP: §3.4 chỉ công thức, hyperparameter ở §4.1 "Training Configuration").

---

## D. §4 EXPERIMENTS

### D1. §4.1 đổi tên → "Experimental Setup" (gộp các phần setup rời rạc)
`\subsection{Environment Settings}` → `\subsection{Experimental Setup}`.

### D2. Thêm "Training configuration" (chỗ nhận hyperparameter từ §3.4) + số cụ thể từ §3.1
**VỊ TRÍ:** trong §4.1.
**AFTER (thêm đoạn này):**
> **Training configuration.** Each per-IVD crop is a 1×9×112×224 tensor, and all encoders output d = 512-dimensional embeddings. Only the slice-attention pool and the fusion MLP are trained (about 1.18M of the roughly 260M total parameters); both backbones stay frozen. We train with AdamW (learning rate 10⁻⁴, weight decay 10⁻⁴, batch size 32) for up to 20 epochs with patience-5 early stopping. Augmentation follows common spinal-MRI practice~\cite{windsor2022}: horizontal flip with left/right label swap, rotation ±10°, brightness/contrast ±20%, and Gaussian noise (σ=0.05, p=0.3). All runs use a single NVIDIA RTX 4090 (Vast.ai), ~16GB peak memory, ~$2 per hybrid run. Code, three-seed checkpoints, and per-run logs released upon acceptance.

> **(TÙY CHỌN — thầy gợi ý dùng BẢNG)** Có thể thay phần optimizer+augment bằng bảng nhỏ cho gọn. Nhưng ở đây bảng tốn ~14 dòng dọc trong khi prose chỉ ~4 dòng, mà bài đang sát giới hạn trang → tôi để prose. Nếu meeting chốt cắt nội dung chỗ khác thì đổi sang bảng `tab:config` (đã viết sẵn dạng comment trong file backup). Bảng mẫu:
> | Setting | Value |
> |---|---|
> | Optimizer | AdamW (wd 1e-4) |
> | Learning rate | 1e-4 |
> | Batch size | 32 |
> | Epochs / patience | 20 / 5 |
> | Trainable params | ~1.18M / ~260M |
> | Augmentation | flip(+L/R swap), rot ±10°, bright/contrast ±20%, noise σ=0.05 (p=0.3) |

### D3. `<bo sung mo ta cac model duoc dung de so sanh>`
**VỊ TRÍ:** chỗ gạch đỏ trong §4.1.
**AFTER (thêm đoạn):**
> **Models compared.** We compare four configurations that share the same 3D ResNet-34 backbone and per-IVD input: *SpineNetV2*, the plain grading backbone of~\cite{windsor2022}; *CBAM-only*, that backbone with channel-spatial attention but no multimodal branch; *BMC-only*, the frozen multimodal branch with no attention; and the full *Hybrid* that fuses both branches. Comparing these four lets us separate the contribution of each branch.

### D4. Rút gọn mô tả 2 dataset + phóng to hình (`<2 hinh chu nho... cat bot doan mo ta>`) — ✅ ĐÃ XONG
- Đoạn **RSNA** và **SPIDER**: cắt bớt câu thừa (xem bản v2 — mỗi đoạn còn ~4 dòng, vẫn đủ split/số IVD/schema).
- **Ảnh đã regenerate font to:** chạy lại `figures/make_dataset_distribution_en.py` (font 19–21, legend RSNA đưa lên đỉnh ngang để hết chồng nhãn). Ghi đè `rsna_class_distribution_en.png` + `spider_class_distribution_en.png` (tên file không đổi → .tex giữ nguyên).
- **⚠️ Lên Overleaf:** nhớ **upload đè 2 file PNG mới** này (vì Overleaf giữ bản cũ font nhỏ).

---

## E. §4.5 QUALITATIVE — Grad-CAM (`<chu nho, show 1-2 hinh quan trong>`) — ✅ ĐÃ XONG
- **Ảnh đã regenerate 1 hàng (ca canal — ca đại diện rõ nhất):** chạy `viz/grad_cam_rsna.py --canal-only --output paper/lncs_hk252/figures/grad_cam_severe_canal.png` (font to, figsize 13×4.4). File mới: **`grad_cam_severe_canal.png`** (ảnh 3 hàng cũ `grad_cam_severe.png` giữ làm backup).
- **Sửa trong .tex (đã làm ở v2 — anh gõ lên Overleaf):**
  - `\includegraphics[width=0.5\linewidth]{grad_cam_severe.png}` → `\includegraphics[width=\linewidth]{grad_cam_severe_canal.png}`
  - **Caption** → "...for a representative Severe spinal-canal case. Left to right: input mid-sagittal slice, CBAM-only branch, full Hybrid."
  - **Câu §4.5** "...for three Severe validation cases..." → "...for **a representative Severe spinal-canal case**..."; "On the canal case the activation..." → "**The activation concentrates...**"; "On the foraminal conditions..." → "On the foraminal conditions **(not shown)**...".
  - Thêm `\cite{selvaraju2017gradcam}`.
- **⚠️ Lên Overleaf:** nhớ **upload file `grad_cam_severe_canal.png`** lên thư mục figures.

---

## F. §6 SUMMARY (đã sửa từ trước, giữ nguyên)
Đã reframe design-first + bỏ "three-seed". Câu hiện tại OK, không cần đổi thêm.

---

## G. VẤN ĐỀ 12 TRANG (báo để bàn với thầy)
- `main_pair_v2.tex` build OK, **13 trang** (bằng đúng `main_pair.tex` hiện tại của thầy — không tệ hơn).
- Đã thu Grad-CAM 3 hàng → 1 hàng (giúp references dồn lên đầu trang 12), nhưng references vẫn còn ~17 mục tràn trang 13.
- Muốn về **12 trang** cần thân bài ngắn thêm **~nửa trang** — vượt mức "gọt vài câu". Đây chính là phần **generalization + cắt rườm rà (mục CHƯA LÀM)** sẽ tự thu lại sau khi chốt với thầy.
- Nội dung sẽ còn đổi sau meeting nên tôi **chưa ép 12 trang** — chốt xong mới gọt 1 lần.

---

## H. CHECKLIST các `\textcolor{red}{}` đã resolve trong v2
| # | Red comment | Trạng thái |
|---|---|---|
| 1 | `[done]` ×2 (cite biomedclip/clip) | ✅ xóa marker |
| 2 | `<xem lai cau nay>` (Our model realizes...) | ✅ viết lại design-level (B2) |
| 3 | `<kiem tra lai tung y>` contributions | ✅ review 4 bullet (B3) |
| 4 | `<xem lai y nay>` bullet 1 | ✅ viết theo giải thích của anh |
| 5 | `<dung 17 vs 2 cite>` bullet 2 | ✅ bỏ he2016resnet |
| 6 | `<bo sung cau truc paper>` | ✅ thêm đoạn structure (B4) |
| 7 | `<thay so bang notation>` §3.1 | ✅ C×D×H×W (C1) |
| 8 | `<framework hay model>` §3.3 | ✅ thống nhất "Model" (C2) |
| 9 | §3.4 "1 rừng" (chat) | ✅ tách 4 đoạn + đẩy hyperparam xuống §4 (C3,D2) |
| 10 | `<bo sung model so sanh>` | ✅ đoạn "Models compared" (D3) |
| 11 | `<dung bang tom tat / tiet kiem space>` | ✅ gộp setup; bảng để option (D2) |
| 12 | `<2 hinh chu nho / cat mo ta dataset>` | ✅ text đã cắt + **ảnh regenerate font to** (D4) |
| 13 | `<chu nho / show 1-2 hinh>` Grad-CAM | ✅ cite + **ảnh regenerate 1 hàng canal** (E) |

> **Tất cả 13 comment đã resolve.** 2 mục ảnh đã chạy lại script (Grad-CAM 1 hàng + 2 hình phân bố font to). **Khi lên Overleaf nhớ upload đè 3 file PNG:** `grad_cam_severe_canal.png`, `rsna_class_distribution_en.png`, `spider_class_distribution_en.png`.

---

## I. SAU REVIEW PANEL (AI) — sửa over-claim, ƯU TIÊN Ý THẦY

> Panel AI chấm v2 = **Minor Revision (~72%, pass MIWAI/Rank C)**, không ai reject. Nêu mấy chỗ "vống" ở contributions/Summary/§2. Đã lọc: **chỉ giữ cái khớp ý thầy**; cái nghịch ý thầy (nhồi số vào contribution 3) đã bỏ. **Abstract KHÔNG đụng** (thầy sửa rồi).

### I1. §1 Contribution 1 — thêm "parity" + hedge (khớp lời thầy "ngang SpineNetV2")
**Tìm bullet 1, đổi câu cuối:**
- **CŨ:** "...brings transfer performance back up ($+0.034$ over the attention-only variant, $p=0.011$). We read this as attention overfitting the training dataset, with the foundation model stabilizing the representation under dataset shift."
- **MỚI:** "...brings transfer performance back up **to SpineNetV2 parity** ($+0.034$ over the attention-only variant; $p=0.011$, **exploratory at three seeds**). We read this pattern as attention overfitting the training dataset while the foundation model stabilizes the representation under dataset shift---**a behavioral observation on one RSNA→SPIDER pair, not a mechanistic proof**."
- *Lý do:* đúng mô tả của thầy (transfer trở lại **ngang SpineNetV2**); không khoe thắng baseline (thực ra hòa, $p=0.57$).

### I2. §1 Contribution 3 — viết TỔNG QUÁT (đúng ý thầy "đừng phụ thuộc dataset")
**Tìm bullet 3 (Label-space extension), thay cả bullet:**
- **CŨ:** "...can grade a new label set of arbitrary size with no new head and no retraining. We validate this by transferring an RSNA-trained model to eight unseen SPIDER labels (mean F1 = 0.362), which the fixed-head baselines cannot do."
- **MỚI:** "Because labels are scored by matching the image embedding to text prompts, **the label set becomes an input to inference**: the same trained model can grade a new label schema of arbitrary size with no new classifier head and no retraining---something fixed-head baselines cannot do at all. **Section~\ref{sec4} validates this by transferring an RSNA-trained model to a different dataset and label schema with no fine-tuning.**"
- *Lý do:* thầy nói ý quan trọng nhất là contribution 3 → viết **tổng quát**, **không gắn số/“8 SPIDER labels”**. Số trung thực (0.362 vs 0.394) vẫn còn nguyên ở §4.3 body, không mất gì.

### I3. §2 — thêm trích LumbarCLIP (bài tham khảo thầy gửi, cùng nhóm/cùng venue)
**Tìm câu RadCLIP trong đoạn "Vision-language models", chèn ngay sau:**
- **THÊM:** "Closest in application, a CLIP-style model has been trained from scratch on paired lumbar MRI for low-back-pain diagnosis under class imbalance~\cite{le2025lumbarclip}; we differ by keeping the encoder frozen and by studying cross-dataset zero-shot label-space extension rather than in-distribution accuracy on a single corpus."
- **references.bib:** upload entry `le2025lumbarclip` (Le, Vo, Mai, Phan 2025 — MIWAI 2025, DOI 10.1007/978-981-95-4960-3_32). Metadata đã verify qua arXiv 2509.20813.
- ⚠️ *Là bài của chính nhóm thầy → hỏi thầy có muốn cite không; double-blind thì cite ngôi thứ 3 vẫn ổn.*

### I4. §6 Summary — quy kết đúng (recipe + kiến trúc, không phải "the model")
**Tìm câu mở Summary, đổi:**
- **CŨ:** "...for lumbar IVD grading on RSNA 2024. It markedly improves the minority Severe class (Recall ~4×, F1 2.3×, AUC 0.899)..."
- **MỚI:** "...for lumbar IVD grading on RSNA 2024. **Relative to a standard cross-entropy baseline, the full pipeline** markedly improves the minority Severe class (Recall ~4×, F1 2.3×, AUC 0.899)..."
- *Lý do:* 4× là baseline-trần → full pipeline (recipe + kiến trúc), không quy hết cho "the model".

> **P1 (để sau meeting, không chặn nộp):** thêm dòng bảng "SpineNetV2 + recipe" tách recipe vs kiến trúc; chạy thêm 2 seed cho p-value; cite Hallinan 2021 + McSweeney 2023 (đã có .bib); tách §2 "rừng"; đổi Discussion lead "CBAM" → "attention branch".
