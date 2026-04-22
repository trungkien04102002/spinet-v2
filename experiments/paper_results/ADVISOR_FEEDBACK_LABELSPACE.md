# Feedback từ Thầy Nhan Phan — Vấn đề Label Space Scalability

**Ngày nhận feedback**: 2026-04-21  
**Context**: Sau khi em gửi báo cáo kiến trúc training + inference (ARCHITECTURE.md, BAO_CAO_NGAN.md)

---

## 1. Note gốc của thầy

```
PROBLEM VỚI NHÃN
- Kêu 5 nhãn train ra 5 nhãn, kêu 10 nhãn, kêu ra 10 nhãn
- RSNA 5 nhãn chỉ hiểu 5 nhãn, làm sao mở rộng được
- Khi mình đưa 7 nhãn, có union 7 + 5 => 12 thành
- 2 high level:
  - Level đầu là kiểu: ví dụ nhãn 13, 14, 15 chưa train bao giờ, 
    nhưng vẫn có xác suất predict được
  - Level tiếp theo: Nếu không zero-shot được, thì nó sẽ học được 
    nhãn 13, 14, 15
  - Train tăng cường
```

---

## 2. Diễn giải feedback

### Vấn đề thầy chỉ ra

Architecture hiện tại của em **rigid** về label space:
- Em train **3 heads cố định** (`fc_spinal_canal`, `fc_left_foraminal`, `fc_right_foraminal`), mỗi head `Linear(512, 3)`.
- **Không thể mở rộng** sang label mới mà không thêm heads mới + train từ đầu.
- Khi union dataset khác (vd: SPIDER có pfirrmann/spondy/herniation → 3 label mới), tổng label thành 6+ → phải thêm 3 heads mới → không scalable.
- Nếu có dataset thứ 3 với 7 label nữa → union thành 12+ → càng rigid.

### Thầy muốn gì — 2 level

| Level | Yêu cầu | Kỹ thuật tương ứng |
|---|---|---|
| **Level 1 — Zero-shot** | Label 13, 14, 15 chưa train bao giờ vẫn predict được | CLIP-style: image-text alignment, label embedding qua text encoder |
| **Level 2 — Incremental learning** (fallback) | Nếu zero-shot không được, model vẫn học label mới mà không quên cũ | Continual learning (EWC / LwF / Memory Replay) |

---

## 3. Em đã hiểu sai trước đó

Trong lần trao đổi trước về **SPIDER external validation**, thầy đã nói:

> *"Option 1 thì vẫn có thể zeroshot với các nhãn mà không có train thêm so với việc có train thêm thì kết quả như thế nào."*

Em hiểu thành **Linear Probe Evaluation** (freeze backbone, train linear head mới). Nhưng thực tế thầy nói **zero-shot đúng nghĩa ML** (không train head mới nào cả, label mới vẫn predict được qua cơ chế embedding).

→ Feedback lần này **confirm lại** thầy muốn zero-shot đúng nghĩa, không phải Linear Probe.

---

## 4. Kiến trúc hiện tại vs yêu cầu thầy

| Aspect | Hiện tại | Yêu cầu thầy | Gap |
|---|---|---|---|
| Output head | 3 Linear heads cố định | Flexible label space | ❌ |
| Add label mới | Phải thêm Linear head + train scratch | Zero-shot hoặc incremental | ❌ |
| Label representation | Integer class index (0/1/2) | Embedding vector (semantic) | ❌ |
| Scale sang dataset khác | Rebuild heads | Dùng cùng kiến trúc, thay label set | ❌ |

---

## 5. Ba options giải quyết

### Option A — Giữ scope hiện tại + Future Work section

**Làm**: Không đổi architecture. Trong paper thêm section "Future Work" bàn về label scalability, cite CLIP, BiomedCLIP, EWC, LwF.

**Ưu**:
- Giữ toàn bộ công em đã làm (CBAM + Focal + results đã có).
- Thesis defense kịp deadline.
- Paper Rank-C không cần scope quá lớn.

**Nhược**:
- Thầy **có thể không accept** vì thầy đã flag rõ là problem.
- Defense thầy có thể hỏi sâu chỗ này.

**Thời gian**: 0 ngày (chỉ viết).

---

### Option B — Thêm learnable label embedding matrix (support Level 2)

**Làm**: Thay fixed Linear heads bằng:

```python
# Cũ:
self.fc_spinal_canal = nn.Linear(512, 3)
self.fc_left_foraminal = nn.Linear(512, 3)
self.fc_right_foraminal = nn.Linear(512, 3)

# Mới:
num_labels = 9  # 3 conditions × 3 classes, có thể tăng
self.label_embeddings = nn.Parameter(torch.randn(num_labels, 512))

# Forward:
output_logits = feature @ self.label_embeddings.T  # [B, 512] @ [512, num_labels] = [B, num_labels]
```

Khi thêm label mới (vd từ SPIDER):
```python
new_label_count = 5
self.label_embeddings = nn.Parameter(torch.cat([
    self.label_embeddings.data,
    torch.randn(new_label_count, 512)
], dim=0))
```

Train incremental với **EWC** hoặc **LwF** để không quên label cũ.

**Ưu**:
- Support Level 2 (incremental learning).
- Architecture flexible, có thể demonstrate concept cho thầy.
- Giữ được phần lớn công em đã làm (CBAM backbone vẫn vậy).

**Nhược**:
- Chưa support Level 1 (zero-shot thuần túy) — label mới vẫn random init, cần ít train.
- Phải redo training, invalidate một phần kết quả cũ.

**Thời gian**: ~1-2 tuần code + train + report.

---

### Option C — Full CLIP-style với text encoder (support Level 1 + 2)

**Làm**: Dùng text encoder (BiomedCLIP hoặc tương tự) encode label text thành embedding:

```python
from transformers import AutoTokenizer, AutoModel

# Frozen text encoder
text_encoder = AutoModel.from_pretrained("microsoft/BiomedCLIP-...")

label_texts = [
    "normal or mild spinal canal stenosis",
    "moderate spinal canal stenosis",
    "severe spinal canal stenosis",
    # ... nhiều hơn
]
label_embeddings = text_encoder(label_texts)  # [num_labels, 512]

# Training: contrastive loss between image feature and label embedding
# Inference: cosine similarity → argmax
```

Label mới (vd "mild disc herniation") → chỉ cần encode text, không cần train gì.

**Ưu**:
- True zero-shot (Level 1).
- Đáp ứng đúng yêu cầu thầy.
- Kiến trúc hiện đại, paper có weight cao.

**Nhược**:
- Scope lớn: cần text encoder, contrastive training, alignment loss.
- Tốn time, có thể không kịp thesis defense.
- Phần lớn công CBAM + Focal có thể phải rework.

**Thời gian**: ~1-2 tháng.

---

## 6. Câu hỏi clarify cho thầy

Em muốn gửi message này cho thầy trước khi làm gì:

> *Dạ thầy, em hiểu feedback của thầy rồi. Thầy muốn kiến trúc em flexible hơn với label space (support zero-shot hoặc incremental learning).*
> 
> *Em có 3 hướng khả thi, muốn xin ý kiến thầy chọn scope phù hợp với timeline thesis:*
> 
> **1. Option A**: Giữ scope hiện tại (CBAM + Focal trên 3 fixed label). Thêm section Future Work bàn về label scalability, cite CLIP + continual learning.  
> *Thời gian: 0 ngày. Paper Rank-C feasibility cao.*
> 
> **2. Option B**: Thay Linear heads bằng **learnable label embedding matrix**. Support Level 2 (incremental learning với EWC/LwF). Giữ backbone + CBAM.  
> *Thời gian: ~1-2 tuần. Compromise giữa scope và timeline.*
> 
> **3. Option C**: Full **CLIP-style** với text encoder (BiomedCLIP). Support Level 1 (true zero-shot) + Level 2.  
> *Thời gian: ~1-2 tháng. Đáp ứng đầy đủ yêu cầu thầy nhưng có thể không kịp thesis.*
> 
> *Thầy có deadline cho em không và thầy muốn em đi theo option nào ạ?*

---

## 7. Recommendation của em (personal opinion)

Dựa vào:
- Deadline thesis thường không kéo dài được
- Paper Rank-C không yêu cầu novelty cao
- Công CBAM + Focal đã làm có value thực

→ **Nghiêng về Option A** (giữ scope), nhưng **mở đường cho B** nếu thầy nhấn mạnh:
- Nếu thầy accept Option A → Future Work + đúng track thesis
- Nếu thầy muốn ít nhất Level 2 → Option B, ~1-2 tuần ok
- Chỉ **không chọn C** trừ khi thầy commit deadline thesis kéo dài

---

## 8. Reference kỹ thuật (để defend khi thầy hỏi)

### Level 1 (Zero-shot)
- **CLIP** (Radford et al., 2021): Image-text contrastive learning, zero-shot classification via text prompt.
- **BiomedCLIP** (Zhang et al., 2023): Medical version of CLIP, pre-trained on PubMed image-text pairs.
- **Medical LVLM** (Moor et al., 2023; Tu et al., 2023): Various medical vision-language models.

### Level 2 (Incremental/Continual Learning)
- **EWC — Elastic Weight Consolidation** (Kirkpatrick et al., PNAS 2017): Penalty preserves important weights of old tasks.
- **LwF — Learning without Forgetting** (Li & Hoiem, TPAMI 2017): Distill old model output as soft target.
- **iCaRL** (Rebuffi et al., CVPR 2017): Exemplar replay buffer for class-incremental learning.
- **Medical imaging continual learning**: Lenga et al. (MICCAI 2020), Perkonigg et al. (Medical Image Analysis 2021).

*Cần verify các citation trên Google Scholar trước khi dùng trong paper.*

---

## 9. Action items

- [ ] Gửi message clarify cho thầy Nhan (soạn ở mục 6)
- [ ] Chờ phản hồi → decide Option A / B / C
- [ ] Nếu Option A: update paper Future Work section
- [ ] Nếu Option B: design label embedding architecture + test
- [ ] Nếu Option C: research BiomedCLIP integration, timeline extension với thầy
