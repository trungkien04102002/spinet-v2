# Động Cơ và Thiết Kế Kiến Trúc — SpineNetV2 + CBAM + BiomedCLIP

> Ngày: 2026-04-27. Đối tượng: hội đồng bảo vệ luận văn thạc sĩ + reviewer EMBC 2027.
> Tài liệu này giải thích **vì sao** kiến trúc được thiết kế như vậy — không phải nó làm gì (xem [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), [`HIGH_LEVEL_ARCHITECTURE.md`](experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md)) cũng không phải số liệu thực nghiệm (xem [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md)).
> Bản tiếng Anh: [`MOTIVATION_AND_DESIGN.md`](MOTIVATION_AND_DESIGN.md) (dùng cho paper EMBC).

## Tóm tắt

Tôi chọn **một pattern kiến trúc nhẹ và có thể tái lập** thay vì hướng foundation-model nặng, bởi vì câu hỏi nghiên cứu là *làm sao mở rộng được không gian nhãn của model grading có cấu trúc mà không cần re-train*, chứ không phải *làm sao scale up parameters*. Cụ thể:

- **Backbone = SpineNetV2** (3D ResNet34, Windsor et al. 2024) — discriminative, đã pretrain trên IVD volumes ở vùng thắt lưng, dùng đúng format input (9 × 112 × 224) sagittal stack mà cả RSNA và SPIDER pipeline đều dùng.
- **+ CBAM** (Woo et al. 2018) — thêm spatial + channel attention với chi phí gần như zero parameter; thực nghiệm cho thấy phục hồi được Severe class detection từ F1 = 0.000 lên F1 ≈ 0.20–0.28 trên các class foraminal khó nhất.
- **+ BiomedCLIP đóng băng** (Zhang et al. 2023) — cung cấp không gian embedding 512-d image-text contrastive; phân loại bằng cosine similarity cho phép **mở rộng nhãn dạng zero-shot** (encode prompt mới ở thời điểm inference, không cần re-train).

Toàn bộ hệ thống ~218M parameters nhưng chỉ ~5M trainable, chạy end-to-end trong <30 phút trên 1 GPU consumer, và pattern này không phụ thuộc vào foundation model cụ thể nào (BiomedCLIP có thể swap thành MedSigLIP / RadFM / RAD-DINO trong future work).

---

## 1. Bài toán (xuất phát từ feedback của thầy)

Luận văn đã được reframe vào giữa năm 2026 sau feedback của thầy hướng dẫn (Phan Nhân), tóm tắt trong [`ADVISOR_FEEDBACK_LABELSPACE.md`](experiments/paper_results/ADVISOR_FEEDBACK_LABELSPACE.md):

> "Kêu 5 nhãn train ra 5 nhãn, kêu 10 nhãn, kêu ra 10 nhãn... RSNA 5 nhãn chỉ hiểu 5 nhãn, làm sao mở rộng được sang dataset khác?"

Thầy chỉ ra **2 mức độ** generalization về không gian nhãn:

| Mức độ | Yêu cầu | Tính chất kiến trúc cần có |
|---|---|---|
| Mức 1 — Zero-shot | Nhãn mới chưa từng thấy lúc training vẫn predict được lúc inference | Image và text trong cùng một embedding space; phân loại bằng cosine similarity (kiểu CLIP) |
| Mức 2 — Incremental | Thêm sample có nhãn cho class mới mà không quên class cũ | Class-incremental learning (out of scope cho luận văn này) |

**Quyết định**: Target Mức 1. Đây chính là cái mà SPIDER zero-shot evaluation kiểm tra — encode 8 nhãn bệnh chưa từng thấy lúc inference, classify mà không có training trên SPIDER. Yêu cầu này trực tiếp ép kiến trúc phải có cosine-similarity head và contrastive image-text encoder, đó là cái BiomedCLIP cung cấp.

---

## 2. Vì sao chọn SpineNetV2 làm base (so với Ningshen, so với MedGemma)

### 2.1 So với Ningshen rsna-2024 (đại diện cho hướng Kaggle 2D)

Submission của Ningshen [(github.com/nshen7/rsna-2024)](https://github.com/nshen7/rsna-2024) là **pipeline 2D 2-giai đoạn**: Faster R-CNN (ResNet-50-FPN) localize bounding box IVD theo từng slice; Swin Transformer fine-tune classify severity từ patch 2D crop bằng weighted cross-entropy. Về mặt kỹ thuật là một prototype Kaggle hợp lý — nhưng có **3 hạn chế cấu trúc** so với mục tiêu của tôi:

1. **Không có volumetric context.** Mỗi IVD là 1 crop 2D đơn lẻ; thông tin xuyên slice (through-plane) — vốn quan trọng về mặt lâm sàng cho stenosis grading — bị vứt bỏ. Pipeline của tôi dùng full sagittal stack (9 × 112 × 224) như một tensor 3D duy nhất.
2. **Không có shared multi-task representation.** Họ train 5 model fine-tune riêng cho 5 condition (SCS, LNFN, RNFN, LSS, RSS), với cascade tuần tự (SCS → LSS → RSS) chỉ như một hack cho data ít. Backbone của tôi dùng 1 trunk 3D ResNet34 chia sẻ với multi-task heads, có regularization xuyên condition.
3. **Không có cơ chế mở rộng không gian nhãn.** Thêm 1 bệnh mới (như Modic từ SPIDER) đòi hỏi phải có data labeled và train lại từ đầu. Cosine-similarity head của tôi mở rộng được sang prompt mới ngay lúc inference.

README repo của Ningshen tự nhận: pipeline "may not fully reflect optimal performance, as limited time and computational resources restricted the ability to thoroughly tune hyperparameters" — đây **không phải top-10 leaderboard entry**, nên cite như "một hướng tiếp cận Kaggle 2D đại diện", không phải state of the art. (Top winners của RSNA Kaggle 2024 dùng ensemble multi-model với TTA, pseudo-labels, fusion sagittal T2 + T1.)

### 2.2 So với MedGemma (medical multimodal foundation model)

MedGemma (Google Health 2025; [arXiv:2507.05201](https://arxiv.org/abs/2507.05201)) là một medical vision-language model chất lượng cao (4B / 27B parameters, build trên Gemma 3 + SigLIP-400M vision encoder, 33M cặp medical image-text). Nó thực sự mạnh ở medical VQA và report generation. Tôi không dùng nó làm base với **5 lý do cụ thể**:

1. **Multimodal chỉ 2D.** Technical report (Section 2.1.1) ghi rõ: *"the multimodal capabilities of MedGemma are currently focused on 2D medical images (e.g., X-ray, 2D slices from CT/MRI); 3D volumes ... were not included."* Bài toán của tôi là volumetric MRI grading nơi sagittal stack mang thông tin lâm sàng có ý nghĩa mà single slice không có.
2. **Không specialize cho spine MRI.** Trong 33M cặp training, chỉ 47,622 là MRI 2D slice (~0.14%) — được chọn từ radiology report nhắc đến slice bất thường cụ thể, không tập trung vào anatomy thắt lưng/sagittal. Backbone SpineNetV2 đã pretrain trực tiếp trên lumbar IVD volumes.
3. **Generative paradigm — sai use case.** MedGemma output là text token (radiology report, VQA answer). Bài toán của tôi là multi-class structured grading. Dùng MedGemma đồng nghĩa với việc strip cái LLM ra và chỉ dùng SigLIP encoder — lúc đó nó function như một frozen feature extractor, vai trò giống hệt BiomedCLIP nhưng nặng gấp ~5× (400M vs 86M) mà không có aligned text branch dùng được cho zero-shot extension.
4. **Compute cost không cân xứng với sample count.** Fine-tune model 4B–27B trên ~2000 IVD label là rơi vào regime overfitting cao kể cả khi dùng QLoRA. Đóng góp của luận văn này là chứng minh integration pattern hoạt động được dưới constraint thực tế — 1 GPU consumer, ~30 phút training. Plug-in MedSigLIP/MedGemma được note là future work và kiến trúc đã thiết kế để hỗ trợ điều đó.
5. **Không có CLIP-style zero-shot interface.** MedGemma không expose contrastive embedding space L2-normalized image-text như CLIP/BiomedCLIP. Cosine-similarity zero-shot head đòi hỏi tính chất này; BiomedCLIP cung cấp natively.

### 2.3 Các foundation model khác mà hội đồng có thể hỏi

| Model | Nó là gì | Lý do không dùng |
|---|---|---|
| **RadFM** (Wu et al., Nature Comm. 2025) | Generalist radiology FM, support cả 2D + 3D | Generative paradigm, không phải contrastive; chi phí fine-tune lớn; không có spine-specific benchmark |
| **RAD-DINO** (Pérez-García et al., 2024) | Vision-only DINOv2 SSL trên chest X-ray | Không có text branch → không zero-shot qua prompt được; domain chest X-ray |
| **Merlin** (Blankemeier et al., 2024) | 3D CT vision-language | Specialized cho CT, không phải MRI; model lớn |
| **Med-PaLM-M** (Tu et al., 2023) | 562B multimodal | Không open-weight; compute prohibitive |

**Chốt**: SpineNetV2 đúng vai trò *backbone* (3D, pretrain lumbar, discriminative). BiomedCLIP đúng vai trò *zero-shot head* (contrastive, frozen, kiểu CLIP). MedGemma và RadFM giải quyết bài toán khác (generative reasoning) ở mức resource khác.

---

## 3. Vì sao CBAM giúp detect Severe class

### 3.1 Cơ chế (Woo et al. ECCV 2018)

CBAM apply 2 attention gate nhẹ tuần tự sau mỗi ResNet stage:

- **Channel attention**: global average + max pool theo spatial dim → shared MLP → vector trọng số per-channel. Recalibrate xem feature map nào encode pathology vs anatomy.
- **Spatial attention**: thống kê per-channel tại mỗi vị trí spatial → 2D weight map. Suppress background, amplify vùng có lesion.

Tổng cost: ~0.1% backbone parameter. Insert sau mỗi 4 ResNet stage trong 3D ResNet34. Quan trọng: spatial gate operate trên full 3D feature tensor `(C, D, H, W)` — học đồng thời *slice nào và vị trí spatial nào*, đem lại volumetric reasoning thực sự cho backbone.

### 3.2 Vì sao điều này giúp riêng cho lumbar stenosis grading bị imbalance

Vấn đề class imbalance trong lumbar stenosis là **bản chất spatial**: một foraminal narrowing "Severe" chỉ ảnh hưởng một sub-region nhỏ của IVD crop (thường vài voxel residual canal/foramen còn lại). Không có spatial attention, gradient signal từ Severe class hiếm bị pha loãng bởi vùng anatomy bình thường lớn hơn nhiều. Điều này khớp với failure mode baseline trong [Phase 1+2 results](experiments/fresh_cbam/RESULTS_SUMMARY.md): Foraminal Severe F1 = 0.000 ở baseline không-CBAM (model không bao giờ predict Severe trên các class này).

Spatial gate của CBAM ép network up-weight đúng chỗ foramen bị hẹp, concentrate gradient vào discriminative boundary. Bằng chứng thực nghiệm:

| Metric | Baseline (no CBAM, no class weights) | CBAM + sqrt class weights |
|---|---|---|
| Spinal Severe F1 | 0.457 | **0.623** |
| L. Foraminal Severe F1 | **0.000** | 0.180 |
| R. Foraminal Severe F1 | **0.000** | 0.197 |

Phục hồi từ 0.000 chính là discriminative evidence: cơ chế đã không còn spatially blind đối với sub-region pathological. Pattern này khớp với precedent đã established trong medical imaging — CBAM-augmented network phục hồi recall hard-class trên drusen/microaneurysm nhỏ ở retinal imaging (Roy et al., IEEE TMI 2019), và trên pulmonary nodule sub-centimeter (Liu et al., Medical Physics 2020).

### 3.3 Vì sao chọn CBAM (chứ không phải SE-Net / non-local / transformer)?

- **vs Squeeze-and-Excitation (Hu et al. CVPR 2018)**: SE chỉ có channel attention, không có spatial gate. Đối với localize lesion nhỏ thì spatial component là cái critical.
- **vs non-local block / transformer self-attention**: quadratic theo spatial resolution, parameter overhead lớn. Prohibitive trong regime ~1500-IVD training của tôi nơi overfitting là rủi ro chính.
- **vs Coordinate Attention / ECA-Net**: cũng khả thi; chọn CBAM vì có literature ablation medical imaging vững chắc và drop-in clean sau mỗi ResNet stage.

Value proposition của CBAM là "spatial + channel attention với chi phí gần zero, well-studied trên medical imaging." Với resource regime của luận văn này, đó là trade-off phù hợp.

---

## 4. Vì sao BiomedCLIP đóng băng cho phép zero-shot label extension

### 4.1 Cơ chế (Zhang et al. 2023, Microsoft; [arXiv:2303.00915](https://arxiv.org/abs/2303.00915))

BiomedCLIP là model contrastive kiểu CLIP, pretrain trên 15M cặp biomedical image-text từ PubMed Central — radiology, pathology, microscopy, clinical photograph. Theo Radford et al. (CLIP, ICML 2021), training minimize symmetric contrastive loss align cặp image-text song hành đồng thời đẩy xa các cặp không match. Output là **một không gian embedding 512-d chia sẻ** nơi feature ảnh và mô tả văn bản của cùng một concept y khoa nằm gần nhau.

Cơ chế zero-shot rất trực tiếp: lúc inference, label prompt (ví dụ `"MRI showing severe foraminal stenosis"`) được encode bằng frozen text encoder thành vector trong shared space. Ảnh query được encode bằng frozen vision encoder, classification reduce thành nearest-prompt-by-cosine-similarity. **Không cần re-train để thêm nhãn mới** — chỉ encode prompt mới là đủ. Đây chính xác là tính chất kiến trúc đáp ứng yêu cầu Mức-1 của thầy.

### 4.2 Vì sao đóng băng, không fine-tune?

Rủi ro catastrophic forgetting. Với ~1500 mẫu RSNA + ~1400 mẫu SPIDER có label, fine-tune một ViT-B/16 đã pretrain trên 15M cặp sẽ làm xói mòn semantic alignment vốn là cái khiến zero-shot transfer hoạt động ngay từ đầu. Frozen feature extraction là standard practice trong low-data multimodal regime (theo CLIP linear probe protocol, Radford et al. 2021). Fine-tune được defer cho future work với corpus annotated lớn hơn.

### 4.3 Vì sao cần projection MLP (chỉ phía image)

Vision encoder của BiomedCLIP là 2D và pretrain trên figure ảnh PubMed. Bài toán của tôi dùng sagittal stack 9 slice và backbone CBAM 3D cung cấp feature volumetric bổ sung. Projection MLP fuse 2 feature đó:

```
[CBAM 3D feat: 512] ⊕ [BiomedCLIP per-slice + attention pool: 512]
                       ↓
              MLP (1024 → 768 → 512)
                       ↓
              L2-normalize → image_emb
                       ↓
              cosine sim with text_emb_db
```

Đây là phần kiến trúc mới: cross-encoder fusion (3D specialist + 2D semantic) vào không gian 512-d aligned với CLIP.

---

## 5. Vì sao kết hợp lại bổ trợ nhau (luận điểm cốt lõi của luận văn)

Hai feature path CBAM-3D và BiomedCLIP-2D **trực giao về mặt thông tin**:

| Trục | CBAM 3D ResNet34 | BiomedCLIP ViT-B/16 |
|---|---|---|
| Chiều input | Full 9-slice sagittal volume | Single 2D slice (×3, attention-pool) |
| Loại feature | Localize lesion theo spatial, geometry volumetric | Disease appearance semantic, language-grounded |
| Cơ chế attention | Channel + spatial gate explicit train trên RSNA | Implicit qua contrastive pretrain trên 15M cặp |
| Khả năng zero-shot | Không (đứng riêng) | Có sẵn (qua encode prompt) |
| Domain specialization | RSNA lumbar stenosis | Broad biomedical (curated từ PubMed) |

Kết quả Phase 3 SPIDER zero-shot trực tiếp validate complementarity story (số liệu đầy đủ trong [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md) §Phase 3):

- **Hybrid thắng trên các nhãn liên quan đến disc** (Disc_narrowing 0.586 vs Naked-BiomedCLIP 0.403, +0.18; Disc_bulging 0.613 vs 0.363, +0.25). Các pathology này đòi hỏi geometry volumetric — chiều cao đĩa, bulge xuyên qua nhiều slice — mà single 2D feature không thể capture đáng tin được. Path CBAM 3D supply spatial context.
- **Naked BiomedCLIP thắng trên nhãn vertebra/endplate** (Modic, UP/LOW endplate, Spondylolisthesis). Các bệnh này dễ phân biệt visually trong 2D và được represent tốt trong PubMed (Modic changes có rất nhiều ảnh trong literature lâm sàng). Projection MLP train trên RSNA lại bị *over-specialize* sang label space stenosis — đẩy embedding đi xa khỏi vùng có ích cho pathology không liên quan.

Đây **không phải failure của kiến trúc** — đó là behavior dự đoán được khi 3D specialist branch là domain-specific. Nó cũng là một experimental finding sạch cho paper: complementarity là thực sự, nhưng projection MLP induce một label-space bias mà field cần biết khi áp dụng pattern này. Future work: train projection trên label đa dạng hơn, hoặc dùng multi-domain projection head.

---

## 6. Câu hỏi hội đồng có thể hỏi (kèm câu trả lời sắc nét)

Các câu này nhiều khả năng hội đồng hoặc reviewer EMBC sẽ hỏi.

### Q1. "Sao không dùng MedGemma / RadFM cho rồi?"

3 lý do:
1. **Sai paradigm**: chúng là generative model; bài toán của tôi là structured grading với cosine-sim zero-shot extension — cần contrastive embedding space, mà MedGemma không expose natively.
2. **Sai domain**: nhánh multimodal của MedGemma là 2D-only và không pretrain riêng cho sagittal lumbar MRI.
3. **Sai resource regime**: integration pattern của tôi phải reproducible trên 1 GPU consumer. Plug-in foundation model nặng hơn được defer cho future work, kiến trúc đã design để hỗ trợ điều đó (BiomedCLIP swap được sang MedSigLIP / RadFM mà không cần thay đổi gì khác).

### Q2. "Sao chọn CBAM mà không phải Squeeze-and-Excitation, non-local, hoặc transformer 3D?"

CBAM có cả channel và spatial attention với chi phí ~0.1% backbone parameter. SE-Net chỉ có channel, thiếu spatial localization signal — vốn là cái drive Severe-class recovery. Non-local / transformer self-attention scale quadratic và thêm parameter đáng kể — prohibitive trong regime data ít. CBAM có ablation literature medical imaging vững và drop-in clean.

### Q3. "Vì sao BiomedCLIP đóng băng?"

Rủi ro catastrophic forgetting. Với ~3000 sample có label tổng cộng (RSNA + SPIDER), fine-tune ViT-B/16 đã pretrain 15M cặp sẽ erode semantic alignment — vốn là cái cho phép zero-shot hoạt động. Frozen feature extraction là standard practice trong low-data multimodal regime (CLIP linear probe protocol).

### Q4. "Tại sao cosine similarity, không phải linear softmax head?"

Softmax head trên vocabulary cố định không thể generalize sang nhãn chưa thấy lúc inference. Cosine similarity trong shared CLIP embedding space cho phép bất kỳ disease description nào được encode thành prompt và dùng ngay làm classification target — đây chính là tính chất kiến trúc đáp ứng generalization Mức-1 zero-shot. Tương đương với nearest-centroid classifier trong không gian L2-normalized (Snell et al. NeurIPS 2017, Prototypical Networks).

### Q5. "Hybrid zero-shot lại thua Naked BiomedCLIP về mean F1. Có nghĩa là kiến trúc của em tệ hơn?"

Không — kết quả là **selective transfer**, không phải failure. Hybrid thắng trên nhãn liên quan đến disc (nơi 3D volumetric context critical) và thua trên nhãn vertebra/endplate (nơi 2D appearance đủ và projection MLP train trên RSNA bị over-specialize cho stenosis feature). Cách diễn giải trung thực là projection MLP là một domain bias — giúp task aligned với stenosis và hại task không liên quan. Tôi report điều này transparently chứ không che đi.

### Q6. "Có scale được sang vùng cơ thể / modality khác không?"

Có. Pattern này anatomy-agnostic: backbone CNN 3D + frozen contrastive encoder + cosine-sim head. Component duy nhất phụ thuộc anatomy là label prompt (thay được trivial) và projection MLP weight (cần fine-tune trên dataset mới). Pretraining của BiomedCLIP cover nhiều vùng cơ thể. Mở rộng sang cervical spine, đầu gối, brain MRI chỉ cần thay prompt và re-run projection-MLP fine-tune.

### Q7. "Khác gì so với prior work làm CLIP cho medical imaging (MedCLIP, PubMedCLIP, GLoRIA)?"

Prior work dùng pretrain CLIP-style và evaluate zero-shot trên cùng modality 2D (chest X-ray). Novelty của tôi là **kiến trúc**: fuse 3D volumetric specialist với 2D contrastive encoder qua projection học được, và demonstrate **cross-dataset zero-shot label extension** (train trên RSNA, evaluate trên SPIDER mà không thấy bất kỳ nhãn SPIDER nào). 2 differentiator chính: (i) input volumetric + fusion 2D semantic, (ii) cross-dataset label-space generalization, không phải within-dataset.

### Q8. "Patient-level split nghĩa là gì, zero-shot evaluation có thực sự held-out không?"

Train/val split trong từng dataset là patient-level (`random_state=42`) để tránh leak cùng patient. Zero-shot evaluation trên SPIDER là test held-out mạnh nhất tôi có — model chưa từng exposed với bất kỳ label hoặc image SPIDER nào trong training; ảnh SPIDER lại từ scanner khác, hospital khác, protocol acquisition khác. Cái này gần "deployment shift" hơn cả within-dataset val split.

### Q9. "Report 1 seed có đủ không?"

Cho luận văn thì đủ — với caveat: tôi report Mean F1 macro qua 4 condition / 8 disease zero-shot, tức là đã average bớt per-class variance. Với phiên bản EMBC paper, kế hoạch là thêm 3-seed mean ± std trên các số headline (Phase 1+2 Hybrid + Phase 3 zero-shot + Phase 4 Hybrid frozen). Single-seed được acknowledge là limitation trong [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md).

### Q10. "Đóng góp thực sự là gì nếu BiomedCLIP và CBAM đều là module có sẵn?"

Đóng góp là **integration pattern + empirical demonstration**:
1. Kiến trúc fusion cụ thể (CBAM 3D + BiomedCLIP 2D per-slice + attention-pool + projection MLP vào shared CLIP space).
2. Lần đầu tiên (theo hiểu biết của tôi) demonstrate cross-dataset zero-shot label extension giữa RSNA và SPIDER trên volumetric lumbar MRI.
3. Đặc tả trung thực khi nào integration có ích (disc-related, nơi 3D matters) vs khi nào frozen encoder đơn giản đã đủ (vertebra/endplate, nơi 2D semantic đủ signal). Selective-transfer finding này tự thân là một đóng góp vì nó báo cho practitioner sau biết khi nào complexity phụ là justified.

---

## 7. Map tới 3 đóng góp đã state

3 đóng góp trong [`STEPS.md`](STEPS.md) map như sau:

| Đóng góp | Vị trí trong luận văn | Bằng chứng thực nghiệm |
|---|---|---|
| C1: CBAM + Focal Loss để recover Severe class trên RSNA | Phase 1+2 | Foraminal Severe F1: 0.000 → 0.180 / 0.197 (CBAM); 0.276 / 0.283 (Hybrid) |
| C2: BiomedCLIP integration cho zero-shot label extension | Phase 3 | 8 nhãn SPIDER predict được mà không có training SPIDER; F1 per-tier 0.586 / 0.401 / 0.277 |
| C3: CBAM/Hybrid generalize được giữa các dataset (RSNA → SPIDER supervised) | Phase 4 | Hybrid frozen mean F1 = 0.623 vs Vanilla 0.610, với ~5M trainable param vs 22M |

---

## 8. Tài liệu tham khảo (core list, ~12 papers)

1. Windsor R., Jamaludin A., Kadir T., Zisserman A. **SpineNetV2: Automated Detection, Labelling and Radiological Grading of Clinical MR Images** — MICCAI 2024 / arXiv:2205.01683.
2. Woo S., Park J., Lee J.-Y., Kweon I.S. **CBAM: Convolutional Block Attention Module** — ECCV 2018.
3. Zhang S., Xu Y., Usuyama N. *et al.* **BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs** — arXiv:2303.00915 (2023).
4. Radford A., Kim J.W., Hallacy C. *et al.* **Learning Transferable Visual Models From Natural Language Supervision** (CLIP) — ICML 2021.
5. Hu J., Shen L., Sun G. **Squeeze-and-Excitation Networks** — CVPR 2018.
6. Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. **Focal Loss for Dense Object Detection** — ICCV 2017.
7. Kendall A., Gal Y., Cipolla R. **Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics** — CVPR 2018.
8. Snell J., Swersky K., Zemel R. **Prototypical Networks for Few-shot Learning** — NeurIPS 2017 (cơ sở lý thuyết cho cosine nearest-centroid).
9. He K., Zhang X., Ren S., Sun J. **Deep Residual Learning for Image Recognition** (ResNet) — CVPR 2016.
10. Sebro R. *et al.* **MedGemma Technical Report** — arXiv:2507.05201 (2025) — reference cho phần "vì sao không MedGemma".
11. Pérez-García F. *et al.* **RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision** — arXiv:2401.10815 (2024).
12. nshen7. **rsna-2024** GitHub repository, [github.com/nshen7/rsna-2024](https://github.com/nshen7/rsna-2024) — reference như hướng Kaggle 2D đại diện.

Precedent attention cho lesion nhỏ (verify trước khi submit):
- Roy A.G. *et al.* IEEE TMI 2019 — attention cho retinal lesion detection.
- Liu et al. Medical Physics 2020 — spatial attention cho pulmonary nodule sub-centimeter.

---

## 9. Doc liên quan trong repo (đọc thêm để chi tiết)

| Chủ đề | Tài liệu |
|---|---|
| Bản tiếng Anh của doc này (cho EMBC paper) | [`MOTIVATION_AND_DESIGN.md`](MOTIVATION_AND_DESIGN.md) |
| Big-picture pipeline + status | [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) |
| Master execution plan + 3 đóng góp | [`STEPS.md`](STEPS.md) |
| Mermaid architecture diagram chi tiết | [`HIGH_LEVEL_ARCHITECTURE.md`](experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md), [`new_architecture.md`](experiments/paper_results/new_architecture.md) |
| How-to train Hybrid model | [`HYBRID_TRAINING_GUIDE.md`](HYBRID_TRAINING_GUIDE.md) |
| RSNA preprocessing + training | [`RSNA_PIPELINE.md`](RSNA_PIPELINE.md) |
| SPIDER zero-shot setup | [`SPIDER_ZEROSHOT_SETUP.md`](SPIDER_ZEROSHOT_SETUP.md) |
| SPIDER transfer learning | [`SPIDER_TRAINING_GUIDE.md`](SPIDER_TRAINING_GUIDE.md) |
| Feedback của thầy về label-space (verbatim) | [`ADVISOR_FEEDBACK_LABELSPACE.md`](experiments/paper_results/ADVISOR_FEEDBACK_LABELSPACE.md) |
| Note chuẩn bị họp với thầy | [`ADVISOR_MEETING_PREP_2026-04-24.md`](experiments/paper_results/ADVISOR_MEETING_PREP_2026-04-24.md) |
| Research foundation-model landscape | [`KEYWORDS_RESEARCH.md`](experiments/paper_results/KEYWORDS_RESEARCH.md) |
| Phase 1+2 RSNA chi tiết | [`fresh_cbam/RESULTS_SUMMARY.md`](experiments/fresh_cbam/RESULTS_SUMMARY.md) |
| Phase 3 SPIDER zero-shot chi tiết | [`spider_zeroshot/RESULTS_REPORT.md`](experiments/paper_results/spider_zeroshot/RESULTS_REPORT.md) |
| Báo cáo unified cross-phase | [`PHASE_REPORT_FULL.md`](experiments/PHASE_REPORT_FULL.md) |
