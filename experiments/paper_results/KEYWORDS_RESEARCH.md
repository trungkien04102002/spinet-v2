# Keywords Research Guide — BiomedCLIP + Hybrid Architecture

> Guide để em research kiến thức cần cho pipeline hybrid CBAM + BiomedCLIP. Đọc theo thứ tự Tier 1 → 5. Check box sau khi đọc xong.

---

## TIER 1 — Foundation (Phải hiểu TRƯỚC mọi thứ khác)

### [ ] 1. CLIP (Contrastive Language-Image Pre-training)

**1 câu**: Model của OpenAI học được mối quan hệ giữa ảnh và text qua contrastive learning, cho phép zero-shot classification.

**Search queries**:
- `CLIP Radford 2021 ICML`
- `CLIP learning transferable visual models`
- `CLIP contrastive language image`

**Câu hỏi phải trả lời được sau khi đọc**:
1. CLIP có mấy encoder? (→ 2: image + text)
2. Loss function của CLIP là gì? (→ contrastive, InfoNCE)
3. Zero-shot classification của CLIP hoạt động thế nào? (→ encode label text → similarity)
4. Temperature parameter dùng để làm gì?

**Nguồn đọc**:
- Paper: arXiv:2103.00020
- Blog explain: Lil'Log, OpenAI blog về CLIP
- YouTube: "CLIP explained" (Yannic Kilcher hoặc AI Coffee Break)

**Time**: ~1 giờ (đọc blog trước, paper sau nếu cần sâu)

---

### [ ] 2. BiomedCLIP

**1 câu**: Phiên bản y khoa của CLIP, do Microsoft train trên 15 triệu cặp (ảnh medical, text caption) từ PubMed Central.

**Search queries**:
- `BiomedCLIP Microsoft 2023 arXiv`
- `PubMedBERT vit_base_patch16_224 HuggingFace`
- `medical CLIP PMC-15M`

**Câu hỏi phải trả lời**:
1. BiomedCLIP khác CLIP thế nào? (→ training data medical, text encoder PubMedBERT)
2. Image encoder của BiomedCLIP là gì? (→ ViT-B/16 2D)
3. Text encoder là gì? (→ PubMedBERT)
4. Model có handle 3D không? (→ không, 2D only)

**Nguồn đọc**:
- Paper: arXiv:2303.00915
- HuggingFace model card: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Code example trên HuggingFace

**Time**: ~45 phút

---

### [ ] 3. VLM vs LLM

**1 câu**: VLM encode ảnh + text ra embedding (CLIP, BiomedCLIP), LLM generate text (GPT, Gemini, MedGemma) — 2 kiểu model khác mục đích.

**Search queries**:
- `vision language model vs LLM difference`
- `CLIP vs GPT4V`
- `discriminative vs generative VLM`

**Câu hỏi phải trả lời**:
1. BiomedCLIP có sinh được text không? (→ KHÔNG, chỉ ra embedding)
2. MedGemma có giống BiomedCLIP không? (→ KHÔNG, MedGemma là LLM generative)
3. Khi nào dùng VLM? Khi nào dùng LLM?

**Nguồn đọc**:
- Blog "What is a Vision Language Model?" trên HuggingFace
- YouTube: "CLIP vs GPT4V" comparison

**Time**: ~30 phút

---

## TIER 2 — Design Mechanics (hiểu để design đúng pipeline)

### [ ] 4. Cosine Similarity + Contrastive Loss (InfoNCE)

**1 câu**: Cosine đo độ "giống" giữa 2 vector, InfoNCE loss đẩy cặp đúng gần nhau, cặp sai ra xa.

**Search queries**:
- `cosine similarity explained`
- `InfoNCE loss contrastive learning`
- `CLIP loss formula temperature`

**Câu hỏi phải trả lời**:
1. Cosine similarity range là gì? (→ [-1, 1], tốt nhất = 1)
2. Tại sao phải L2 normalize trước cosine? (→ để loại bỏ magnitude, chỉ quan tâm direction)
3. Temperature làm gì trong loss? (→ scale logits, thấp → sharp distribution)
4. InfoNCE formula nhìn thế nào?

**Nguồn đọc**:
- Blog: "Understanding Contrastive Learning" trên towards-data-science
- Sebastian Raschka's book chapter on contrastive learning

**Time**: ~45 phút

---

### [ ] 5. Projection Head

**1 câu**: Lớp Linear nhỏ sau encoder để "dịch" feature từ không gian encoder sang không gian chung với encoder khác.

**Search queries**:
- `CLIP projection head`
- `multimodal alignment projection layer`
- `SimCLR projection head`

**Câu hỏi phải trả lời**:
1. Image projection và text projection trong CLIP có giống nhau không?
2. Khi nào cần projection trainable? Khi nào không?
3. Nếu load pretrained CLIP, projection có cần train lại không?

**Nguồn đọc**:
- CLIP paper Section 2.2 (có diagram projection head)
- `open_clip_torch` source code (xem `encode_image`, `encode_text` có project không)

**Time**: ~30 phút

---

### [ ] 6. Zero-shot vs Linear Probe vs Few-shot

**1 câu**: Zero-shot = không train gì; Linear Probe = freeze backbone, train head; Few-shot = train với ít sample.

**Search queries**:
- `zero-shot classification CLIP explanation`
- `linear probe evaluation protocol`
- `few-shot learning medical imaging`

**Câu hỏi phải trả lời**:
1. Zero-shot của CLIP dùng gì để predict? (→ text prompt + similarity)
2. Linear probe có trainable params không? (→ có, nhưng chỉ head)
3. Few-shot khác Linear probe thế nào?
4. Accuracy expected: zero-shot < few-shot < linear probe < full fine-tune?

**Nguồn đọc**:
- CLIP paper Section 3 (zero-shot evaluation)
- Blog "CLIP zero-shot vs linear probing"

**Time**: ~30 phút

---

### [ ] 7. Frozen vs Trainable Components

**1 câu**: Freeze = không update weights (`requires_grad=False`), giữ nguyên kiến thức pretrain; Trainable = update qua backprop.

**Search queries**:
- `PyTorch freeze layer requires_grad`
- `transfer learning feature extractor`
- `catastrophic forgetting fine-tuning`

**Câu hỏi phải trả lời**:
1. Tại sao freeze encoder khi transfer learning?
2. Unfreeze nhiều → rủi ro gì? (→ catastrophic forgetting, overfit small data)
3. LoRA là gì? (→ advanced, add trainable adapter vào frozen model)
4. BatchNorm trong frozen encoder cần làm gì đặc biệt? (→ set eval mode)

**Nguồn đọc**:
- PyTorch tutorial "Transfer Learning"
- Paper LoRA: arXiv:2106.09685 (optional, chỉ đọc abstract)

**Time**: ~30 phút

---

## TIER 3 — Cho Kiến Trúc Hybrid (của em)

### [ ] 8. 2D Encoder → 3D Medical Data Adaptation

**1 câu**: BiomedCLIP là 2D, muốn dùng cho 3D volume phải adapt qua slice selection + aggregation.

**Search queries**:
- `BiomedCLIP 3D adaptation medical imaging`
- `2D encoder 3D volume aggregation`
- `slice selection CT MRI deep learning`

**PAPER QUAN TRỌNG**: `Lian 2026 Scientific Reports data-efficient 3D medical vision-language model 2D encoder`

**Câu hỏi phải trả lời**:
1. Làm sao đưa 3D volume (9 slice × 112 × 224) vào encoder 2D?
2. Slice selection strategies? (→ center slice, representative, attention-based)
3. Aggregation options? (→ mean pool, max pool, attention pool)
4. Lian et al. dùng approach nào?

**Nguồn đọc**:
- Paper Lian et al. Scientific Reports 2026 (đã gửi)
- Review papers về 3D medical classification

**Time**: ~1 giờ

---

### [ ] 9. Feature Fusion Techniques

**1 câu**: Combine features từ nhiều nguồn (2 encoder khác nhau) thành 1 representation unified.

**Search queries**:
- `feature fusion deep learning concatenation`
- `late fusion vs early fusion multimodal`
- `cross-attention fusion transformer`

**Câu hỏi phải trả lời**:
1. Concat fusion là gì? Ưu / nhược?
2. Cross-attention fusion khác thế nào?
3. Early fusion vs late fusion?
4. Pipeline em dùng fusion gì?

**Nguồn đọc**:
- Blog: "Feature Fusion Methods in Deep Learning"
- Paper: BERT, T5 cross-attention (optional, hiểu concept)

**Time**: ~30-45 phút

---

### [ ] 10. CBAM Recap (em đã biết, nhưng refresh để defense)

**1 câu**: Channel Attention + Spatial Attention module plug-and-play cho CNN.

**Search queries**:
- `CBAM Woo ECCV 2018 channel spatial attention`
- `attention mechanism CNN medical imaging`

**Câu hỏi phải trả lời**:
1. Channel attention trả lời câu hỏi gì? (→ "feature channel nào quan trọng")
2. Spatial attention trả lời gì? (→ "vị trí nào quan trọng")
3. CBAM gắn ở đâu trong ResNet? (→ sau mỗi stage)
4. Tại sao CBAM tốt cho imbalanced medical data?

**Nguồn đọc**:
- CBAM paper: arXiv:1807.06521
- Đọc lại code `spinenet/models/attention.py` của em

**Time**: ~30 phút (refresh)

---

## TIER 4 — Cho Writing Paper (đọc khi viết)

### [ ] 11. Continual / Incremental Learning (backup plan)

**1 câu**: Kỹ thuật học label mới mà không quên label cũ (catastrophic forgetting).

**Search queries**:
- `EWC elastic weight consolidation Kirkpatrick 2017`
- `Learning without Forgetting Li Hoiem 2017`
- `iCaRL incremental learning`
- `continual learning medical imaging survey`

**Câu hỏi phải trả lời**:
1. Catastrophic forgetting là gì?
2. EWC hoạt động thế nào? (→ penalty preserve important weights)
3. LwF khác EWC thế nào?
4. Khi nào em cần incremental learning?

**Nguồn đọc**:
- Survey: "Continual Learning: A Survey" Masana et al.
- EWC paper: PNAS 2017

**Time**: ~1 giờ (cho paper Future Work section)

---

### [ ] 12. Medical Imaging Foundation Models

**1 câu**: Landscape các VLM y khoa để biết chọn đúng model, cite đúng paper.

**Search queries**:
- `medical imaging foundation model review 2024`
- `MedCLIP PubMedCLIP BioViL comparison`
- `medical CLIP variants`

**Models phải biết**:
| Model | Paper | Dùng cho |
|---|---|---|
| BiomedCLIP (Microsoft) | Zhang 2023 | **Em dùng** |
| MedCLIP | Wang 2022 EMNLP | Chest X-ray |
| PubMedCLIP | Eslami 2023 | VQA medical |
| GLoRIA | Huang 2021 ICCV | Fine-grained medical |
| BioViL | Boecking 2022 | Chest X-ray bilingual |

**Time**: ~30 phút (chỉ đọc abstract, không cần đọc full)

---

### [ ] 13. Class Imbalance Medical Imaging (em biết, recap)

**1 câu**: Severe class rare → model collapse → phải balance bằng Focal Loss, oversampling, class weight.

**Search queries**:
- `focal loss Lin 2017 ICCV`
- `class imbalance deep learning strategies`
- `medical imaging severe class recall`

**Câu hỏi phải trả lời**:
1. Focal Loss formula và γ (gamma) parameter
2. Oversampling ưu / nhược?
3. Class weight trong CrossEntropy?
4. Trade-off accuracy vs recall trong screening task?

**Time**: ~30 phút (refresh)

---

## TIER 5 — Tham khảo (đọc khi cần code)

### [ ] 14. open_clip_torch Library Usage

**1 câu**: Library Python để load và dùng CLIP / BiomedCLIP với API đơn giản.

**Search queries**:
- `open_clip_torch tutorial`
- `create_model_from_pretrained BiomedCLIP`
- `open_clip_torch encode_image encode_text`

**Code phải biết**:
```python
from open_clip import create_model_from_pretrained, get_tokenizer

model, _ = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = get_tokenizer('hf-hub:microsoft/...')

# Encode text
tokens = tokenizer(["a photo of moderate stenosis"])
text_features = model.encode_text(tokens)

# Encode image (2D)
image_features = model.encode_image(image_tensor)
```

**Time**: ~30 phút (đọc README + code example)

---

### [ ] 15. Prompt Engineering for Zero-shot

**1 câu**: Cách viết label text prompts để tăng accuracy zero-shot.

**Search queries**:
- `CLIP prompt engineering classification`
- `zero-shot prompt template medical imaging`
- `CLIP prompt ensembling`

**Câu hỏi phải trả lời**:
1. Prompt "a photo of X" khác "X" thế nào?
2. Ensemble nhiều prompt → tăng accuracy?
3. Medical-specific prompts (cite Tiu et al. CheXzero)?

**Time**: ~30 phút

---

## Roadmap Đọc Theo Ngày

### Ngày 1 (2-3 giờ) — Foundation
- [ ] CLIP
- [ ] BiomedCLIP
- [ ] VLM vs LLM

### Ngày 2 (2-3 giờ) — Mechanics
- [ ] Cosine + InfoNCE
- [ ] Projection head
- [ ] Zero-shot vs Linear Probe
- [ ] Frozen / Trainable

### Ngày 3 (2-3 giờ) — Hybrid architecture
- [ ] 2D → 3D adaptation (+ paper Lian et al.)
- [ ] Feature fusion
- [ ] CBAM recap

### Ngày 4 (1-2 giờ) — Writing context
- [ ] Continual learning (skim)
- [ ] Medical VLM landscape (skim)
- [ ] Class imbalance (skim)

### Ngày 5+ — Khi code
- [ ] open_clip_torch
- [ ] Prompt engineering

---

## Tổng thời gian ước tính

**Tối thiểu** (hiểu cơ bản): ~8 giờ / ~3 ngày  
**Đầy đủ** (ready defense): ~12-15 giờ / ~5 ngày

---

## Sau khi đọc xong, em defense được các câu này

1. **CLIP là gì?** → Dual encoder, contrastive training, zero-shot via text.
2. **BiomedCLIP khác CLIP?** → Medical data pretrain, PubMedBERT text encoder.
3. **Pipeline em có gì mới?** → Hybrid CBAM + BiomedCLIP image encoder, fusion trainable.
4. **Tại sao freeze BiomedCLIP?** → Preserve 15M pairs pretrain, avoid forgetting.
5. **Zero-shot label mới hoạt động sao?** → Encode text qua BiomedCLIP text encoder frozen.
6. **Projection head làm gì?** → Align CBAM feature vào không gian BiomedCLIP text.
7. **Slice selection là gì?** → Chọn representative 2D slice từ 3D volume cho BiomedCLIP image encoder.
8. **Lian et al. 2026 cite để làm gì?** → Precedent cho dùng 2D BiomedCLIP trên 3D medical data.
9. **Nếu zero-shot fail thì sao?** → Fallback few-shot fine-tune projection layer với 10-20 sample.
10. **Contribution novelty?** → Hybrid dual encoder (spine-specific + medical foundation) cho 3D lumbar grading.

---

## Tools suggestion

| Việc | Tool |
|---|---|
| Tìm paper + verify citation | Google Scholar |
| Đọc paper online | arXiv, Papers with Code |
| Explain code API | HuggingFace docs |
| Visual learning | YouTube (Yannic Kilcher, 3Blue1Brown attention) |
| Quick summary | ChatGPT hoặc Claude (khi hiểu đã rồi, summary nhanh) |
| Note-taking | Notion / Obsidian / simple markdown |

---

## Progress tracker

Track bằng cách check `[x]` vào các checkbox trên. Khi xong Tier 1 → move Tier 2. Không skip.

**Update lần cuối**: 2026-04-21
**Đã check**: 0 / 15 topics
