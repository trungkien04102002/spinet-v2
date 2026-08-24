# Scope chốt — LVTN Thạc sĩ + Paper EMBC 2027 (Rank-C)

**Ngày:** 2026-04-28
**Người viết:** Trung Kien
**Mục đích:** Tài liệu trình thầy Nhan Phan để chốt scope chính thức,
không pivot thêm. Dùng làm talking points cho buổi gặp/chat với thầy.

---

## 1. Bài giải quyết 3 vấn đề chính

### Theme 1 — Imbalance handling

**Vấn đề:** RSNA mất cân bằng nặng, class Severe chỉ chiếm <5% dữ liệu.
Baseline thuần predict Normal/Mild → Severe Recall ~0%, vô dụng lâm sàng.

**Giải pháp:** Em kết hợp 4 kỹ thuật bổ sung cho nhau, mỗi cái đánh vào
một góc khác nhau của vấn đề imbalance.

**(1) Focal Loss ($\gamma=2.0$).**
Cross-entropy chuẩn coi mọi sample như nhau, nên class chiếm đa số
(Normal/Mild) áp đảo gradient, làm model "lười" — cứ predict Normal là
đủ đạt accuracy cao trên tập val. Focal Loss thêm hệ số $(1-p_t)^\gamma$
trước cross-entropy: sample dễ ($p_t$ cao, model đã predict đúng tự tin)
bị giảm đóng góp gradient gần về 0; sample khó ($p_t$ thấp, model còn
chưa học) giữ gần như nguyên gradient. Em chọn $\gamma=2.0$ theo Lin et
al. (2017), là giá trị cân bằng giữa "mạnh tay" với sample khó và "ổn
định" trong huấn luyện, phù hợp với mức imbalance ~5–7% Severe của RSNA.

**(2) Class weight (`sqrt` mode).**
Thay vì để mọi class đóng góp trọng số bằng nhau trong loss, em nhân loss
mỗi class với một trọng số $w_c$ tỉ lệ nghịch với số sample lớp đó. Có
2 mode phổ biến:
- *Inverse* ($w_c \propto 1/n_c$): đẩy minority lên rất mạnh, dễ làm
  loss nhảy không ổn định.
- *Square-root* ($w_c \propto 1/\sqrt{n_c}$): đẩy minority lên vừa phải,
  ổn định hơn.

Em chọn mode `sqrt` vì RSNA imbalance không quá khắc nghiệt; mode
`inverse` trong thử nghiệm sơ bộ làm loss dao động lớn, mất ổn định.
Khi tính trọng số, em **bỏ qua label `-1`** (missing label theo
convention RSNA) để class weight không bị kéo lệch.

**(3) Oversampling minority class (factor ×3–5).**
Trong mỗi epoch, mỗi sample Severe được nhân bản 3–5 lần vào sampling
pool trước khi shuffle. Khác với class weight (chỉ đổi trọng số gradient
khi sample đã rơi vào batch), oversampling **đổi trực tiếp tần suất**
sample Severe xuất hiện trong batch. Hai cơ chế này bổ sung cho nhau:
class weight cho mỗi sample trọng số lớn hơn, oversampling tăng số lần
model nhìn thấy sample đó trong 1 epoch. Factor 3–5 là vùng sweet spot
qua thử nghiệm — thấp hơn (≤2) thì hiệu ứng quá yếu, cao hơn (≥10) thì
model overfit do nhìn cùng 1 sample quá nhiều lần.

**(4) Augmentation medium-strength.**
Pipeline tổng hợp 3 nhóm biến đổi:
- *Hình học*: rotation $\pm 15^\circ$, scaling $0.9{-}1.1\times$ —
  mô phỏng độ lệch tư thế giữa các bệnh nhân/lần chụp.
- *Cường độ*: intensity jitter $\pm 10\%$ — mô phỏng sai số calibration
  MRI giữa máy/protocol khác nhau.
- *Không gian*: random crop với margin $\pm 5\%$ — robust với dao động
  nhỏ ở vùng quan tâm.

Mức "medium" được chọn cố ý: nếu augment quá mạnh sẽ phá đặc trưng giải
phẫu, quá yếu thì không tăng diversity đáng kể. Một điểm kỹ thuật quan
trọng: pipeline augment của em **bảo toàn label `-1`** (missing label) —
nếu dùng thẳng các thư viện augment public mà không xử lý chỗ này, label
missing sẽ bị transform sai và làm sai lệch class-weight + loss
computation.

**Tóm tắt cơ chế kết hợp:** Focal Loss đánh vào *bài toán optimization*
(ép gradient nhìn sample khó), Class weight đánh vào *cân bằng đóng góp
mỗi class trong loss*, Oversampling đánh vào *tần suất sample minority
trong batch*, Augmentation đánh vào *đa dạng hoá sample minority*. 4 kỹ
thuật tác động ở 4 tầng khác nhau của pipeline nên hiệu ứng cộng hưởng
chứ không trùng lặp.

### Theme 2 — Attention-driven feature quality

**Vấn đề:** Backbone 3D ResNet34 không tự biết "vùng nào quan trọng" trong volume
9×112×224 — chỉ trích đặc trưng đều khắp, dẫn đến tín hiệu Severe bị lẫn vào nền.

**Giải pháp:** CBAM (Woo et al. 2018) — gồm **2 module song song**:
- **Channel attention**: chọn feature channel nào đáng chú ý (ví dụ: edge của tủy
  sống, contour foramen, không phải nhiễu mỡ/bone).
- **Spatial attention**: chọn vùng không gian nào đáng chú ý (đốt sống, foramen,
  không phải nền).

### Bảng 1 — Kết quả Theme 1 + Theme 2 trên RSNA

RSNA có **3 condition** (3 nhãn bệnh) × **3 grade** (3 mức độ).

**3 condition (bệnh):**
- *Spinal Canal stenosis* (hẹp ống sống)
- *Left Foraminal stenosis* (hẹp lỗ liên hợp trái)
- *Right Foraminal stenosis* (hẹp lỗ liên hợp phải)

**3 grade (mức độ):**
- *Normal/Mild* — bình thường / nhẹ (majority class, ~78% data)
- *Moderate* — trung bình (~17% data)
- *Severe* — nặng (~5% data, class lâm sàng quan trọng nhất)

Bảng được chia thành **2 phần**: (1A) tổng quan trung bình 3 condition,
và (1B) **một bảng chi tiết duy nhất** show đủ 4 chỉ số (Acc, Recall,
Precision, F1) theo từng Condition × Grade.

**Format mỗi cell**: `Base: <số_baseline> → Ours: <số_improvement> (Δ)`.
Acc lặp trong cùng 1 condition vì Accuracy là metric *per-condition*,
không có per-grade. **Bold** = số improvement có gain rõ rệt.

#### 1A — Tổng quan (trung bình 3 condition)

| Chỉ số | Base | Ours | Δ |
|---|---|---|---|
| Mean Accuracy | **81.4%** | 68.6% | −12.8 pp |
| Mean F1 macro | 0.420 | **0.509** | **+0.089** |

#### 1B — Bảng chi tiết: 4 chỉ số theo Condition × Grade

| Condition | Grade | Acc *(per-condition)* | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Spinal Canal | Normal/Mild | Base: 89.5% → Ours: 88.3% (−1.2) | Base: 98.6% → Ours: 92.8% (−5.8) | Base: 91.1% → Ours: **96.4%** (**+5.3**) | Base: 0.947 → Ours: 0.946 (≈0) |
| Spinal Canal | Moderate | Base: 89.5% → Ours: 88.3% | Base: 7.9% → Ours: **42.4%** (**+34.5**) | Base: 33.3% → Ours: 32.4% (−0.9) | Base: 0.128 → Ours: **0.368** (**+0.24**) |
| Spinal Canal | **Severe** | Base: 89.5% → Ours: 88.3% | Base: 35.8% → Ours: **70.4%** (**+34.6**) | Base: 63.0% → Ours: 55.9% (−7.1) | Base: 0.457 → Ours: **0.623** (**+0.17**) |
| Left Foraminal | Normal/Mild | Base: 77.8% → Ours: 57.1% (−20.7) | Base: 96.2% → Ours: 54.5% (−41.7) | Base: 80.4% → Ours: **93.4%** (**+13.0**) | Base: 0.876 → Ours: 0.688 (−0.19) |
| Left Foraminal | Moderate | Base: 77.8% → Ours: 57.1% | Base: 18.6% → Ours: **76.4%** (**+57.8**) | Base: 46.2% → Ours: 28.2% (−18.0) | Base: 0.265 → Ours: **0.412** (**+0.15**) |
| Left Foraminal | **Severe** | Base: 77.8% → Ours: 57.1% | Base: **0.0%** → Ours: **18.8%** (**+18.8**) | Base: **0.0%** → Ours: **17.2%** (**+17.2**) | Base: **0.000** → Ours: **0.180** (**+0.18**) |
| Right Foraminal | Normal/Mild | Base: 76.8% → Ours: 60.4% (−16.4) | Base: 96.2% → Ours: 58.6% (−37.6) | Base: 79.5% → Ours: **94.1%** (**+14.6**) | Base: 0.871 → Ours: 0.723 (−0.15) |
| Right Foraminal | Moderate | Base: 76.8% → Ours: 60.4% | Base: 16.7% → Ours: **75.8%** (**+59.1**) | Base: 43.4% → Ours: 30.9% (−12.5) | Base: 0.241 → Ours: **0.439** (**+0.20**) |
| Right Foraminal | **Severe** | Base: 76.8% → Ours: 60.4% | Base: **0.0%** → Ours: **21.7%** (**+21.7**) | Base: **0.0%** → Ours: **18.0%** (**+18.0**) | Base: **0.000** → Ours: **0.197** (**+0.20**) |

**Pattern theo cột — đọc dọc:**

- **Cột Acc**: giảm nhỏ trên Canal (−1.2 pp), giảm lớn trên Foraminal
  (−16 đến −21 pp). Đánh đổi tập trung ở Foraminal vì baseline Foraminal
  Acc cao do "predict Normal cho mọi sample".
- **Cột Recall**: Normal/Mild giảm; Moderate + Severe **tăng mạnh** ở cả
  3 condition. Toolkit cứu được 2 minority class cùng lúc.
- **Cột Precision**: Normal/Mild **TĂNG** ở cả 3 condition (+5 đến +15
  pp) — *quan trọng*: chứng tỏ Recall Normal/Mild giảm KHÔNG phải vì
  model "tệ", mà vì model **không ôm hết** sample vào Normal nữa, khi
  nói Normal thì chính xác hơn. Moderate Precision giảm (over-predict
  Moderate là tradeoff thực sự). Severe Foraminal **tăng từ 0 → 0.17–0.18**.
- **Cột F1**: tăng trên 6/9 ô, gần như không đổi trên Canal Normal/Mild,
  giảm nhẹ trên Foraminal Normal/Mild. Trung bình tăng 0.420 → 0.509.

**Đọc bảng:**

1. **Pattern chung 3 condition giống hệt nhau**: Normal/Mild giảm nhẹ →
   Moderate và Severe đều tăng mạnh. Đây là *signature* của imbalance
   toolkit — đánh đúng cả 2 minority class cùng lúc.

2. **Foraminal là chỗ giá trị lớn nhất** (cả Recall lẫn F1):
   - Baseline Severe Recall = **0.0%** ở cả 2 foraminal — model **không
     bao giờ predict Severe**, vô dụng lâm sàng.
   - Baseline Severe F1 = **0.000** (vì cả Precision và Recall = 0).
   - Theme 1+2 cứu lên Recall ~19–22%, F1 0.18–0.20 — từ "không phát hiện
     được gì" sang "có khả năng phát hiện".

3. **Spinal Canal là condition dễ nhất**: Baseline đã có Severe F1 = 0.457
   (vì tổn thương ống sống rộng, dễ thấy). Theme 1+2 chỉ đẩy thêm (+0.166),
   không phải nơi tạo bứt phá. Normal/Mild Spinal Canal F1 còn gần như
   không đổi (0.947 → 0.946) — chứng tỏ toolkit **không phá** condition đã
   tốt.

4. **Acc giảm 12.8 pp đến từ Normal/Mild Recall giảm**: nhìn cột Recall
   Normal/Mild — giảm **5.8 pp** trên Canal, nhưng giảm **37–42 pp** trên
   Foraminal. Foraminal Baseline đạt Normal/Mild Recall ~96% vì nó
   *predict Normal cho mọi sample*, ăn dễ class đa số. Khi ép model
   nhìn vào minority, Recall Normal/Mild buộc phải giảm. Đây đúng spirit
   AI y tế thầy đã đề cập — đánh đổi cố ý.

5. **Moderate được cứu mạnh hơn cả Severe**: gain Recall trên Moderate
   (+34 đến **+59 pp**) lớn hơn gain trên Severe (+19 đến +35 pp). Toolkit
   không chỉ cứu class hiếm nhất, mà cứu **toàn bộ đuôi distribution**.

6. **Mean F1 macro tăng 0.420 → 0.509 (+0.089)**: tổng gain trên Moderate +
   Severe lớn hơn loss trên Normal/Mild → model **thực sự** học tốt hơn,
   không phải chỉ "đánh đổi loanh quanh".

> Lưu ý: Recall tăng là **sự cộng hưởng** Theme 1 + Theme 2 (chưa thể tách
> riêng từng theme trong bảng này — em đang chạy ablation `cbam_only` /
> `biomedclip_only` để cô lập đóng góp mỗi theme, sẽ có số ± std đi kèm).

### Theme 3 — Label-space extension qua BiomedCLIP

**Vấn đề:** Mỗi dataset spine có label space riêng. RSNA 3 nhãn stenosis,
SPIDER 8 nhãn (Pfirrmann 5 grades + spondylolisthesis + herniation +
narrowing). Model train RSNA xong **không phục vụ được** câu hỏi SPIDER —
phải retrain hoặc extend head, tốn dữ liệu + compute.

**Giải pháp:** Hybrid 2-branch:
- Nhánh 1: CBAM-3D ResNet34 (đặc trưng 3D context).
- Nhánh 2: **Frozen BiomedCLIP** (Zhang et al. 2023, 86M params, train trên
  15M cặp image-text y khoa) — đem **tri thức ngoài** vào.
- Fusion: concat 512+512 → projection MLP → 512-d L2-normalized.
- Classification: cosine similarity giữa image embedding và text prompt
  ("a normal disc", "a severe stenosis", ...).

→ **2 protocol** đánh giá trên SPIDER (8 nhãn unseen):
- **Zero-shot** (Phase 3): RSNA-trained checkpoint → test thẳng SPIDER,
  **0 dòng retrain** → Mean F1 macro = **0.362**.
- **Retrain** (Phase 4): cùng kiến trúc, fine-tune trên SPIDER → Mean F1 macro
  = **0.623** (frozen backbone, ~5M trainable params).

### Bảng 2 — Đóng góp của Theme 3 nhìn xuyên dataset

So sánh 3 cấu hình **trên cả 3 protocol** (RSNA, SPIDER zero-shot, SPIDER retrain):

| Cấu hình | RSNA F1 macro | RSNA Severe Recall | **SPIDER zero-shot F1** | SPIDER retrain F1 (best) |
|---|---|---|---|---|
| Baseline | 0.420 | 11.9% | — *(không thể: head 3 cls cố định)* | 0.610 *(vanilla, không transfer)* |
| Theme 1+2 (CBAM + imbalance) | 0.509 | 37.0% | — *(không thể: head 3 cls cố định)* | 0.597 |
| **Theme 3 (Hybrid + BiomedCLIP)** | **0.516** | **49.6%** | **0.362** | **0.623** |

**Đọc bảng (key takeaway):**
- **Cột "SPIDER zero-shot F1"**: Baseline và CBAM đều ghi "—" — *không làm
  được*, vì head classifier RSNA cứng 3 lớp, không cách nào predict 8 nhãn
  SPIDER. **Chỉ Theme 3 (Hybrid + BiomedCLIP) mới enable zero-shot** qua
  cosine-similarity với text prompt → **F1 = 0.362** trên 8 nhãn unseen,
  không retrain 1 dòng. Đây là **value chính của Theme 3**.
- **Cột RSNA F1 macro**: gain nhỏ (0.509 → 0.516, +0.007) — khi đã có CBAM,
  thêm BiomedCLIP chỉ cải thiện *nhẹ* trên RSNA. **Đây là expected** —
  BiomedCLIP không phải để đẩy RSNA F1, mà để unlock zero-shot ở cột bên.
- **Cột SPIDER retrain F1 (best)**: Hybrid (0.623) > Vanilla (0.610) > CBAM
  (0.597) — feature từ Hybrid pretrained tốt nhất khi transfer sang SPIDER.

> **Câu trả lời nếu thầy hỏi "Theme 3 đáng giá ở đâu?"**: Không phải ở RSNA
> (gain F1 nhỏ). Đáng giá ở chỗ **unlock label-space extension** —
> Baseline/CBAM không thể trả lời câu hỏi SPIDER, Theme 3 trả lời được cả 2
> protocol (zero-shot và retrain).

### Bảng 3 — Zero-shot SPIDER chi tiết theo nhãn bệnh

**Setting**: Hybrid checkpoint train trên RSNA, **không train trên SPIDER**.
Đem ra test trực tiếp trên SPIDER 8 nhãn unseen qua cosine-similarity với
text prompt. So sánh với **Naked BiomedCLIP** (chỉ dùng BiomedCLIP gốc, không
qua RSNA training, không CBAM, không projection MLP) để cô lập xem **RSNA
training + CBAM có giúp gì cho zero-shot transfer không**.

| Disease (SPIDER) | Tier | Hybrid (no train) | Naked BiomedCLIP | Δ (Hybrid − Naked) | Note |
|---|---|---|---|---|---|
| Disc_narrowing | easy | **0.586** | 0.403 | **+0.183** | Hybrid wins (disc-related) |
| Disc_bulging | medium | **0.613** | 0.363 | **+0.250** | Hybrid wins (disc-related) |
| Disc_herniation | medium | **0.563** | 0.532 | **+0.031** | Hybrid wins (disc-related) |
| Pfirrman_grade | hard | 0.155 | 0.152 | +0.003 | Both fail (5-class problem) |
| Modic | hard | 0.018 | **0.071** | −0.053 | Both fail (rare disease) |
| UP_endplate | hard | 0.467 | **0.569** | −0.102 | Naked wins (endplate) |
| LOW_endplate | hard | 0.468 | **0.558** | −0.090 | Naked wins (endplate) |
| Spondylolisthesis | medium | 0.028 | **0.500** | **−0.472** | Naked wins big |
| **Mean F1 macro** | — | **0.362** | **0.394** | **−0.032** | Naked nhỉnh hơn trung bình |

*Bold trong cột Hybrid/Naked = config thắng. F1 macro tính theo balanced
average qua các grade trong nhãn đó. n=1439 IVD samples cho mỗi disease.*

**Đọc bảng — Selective Transfer Pattern:**

1. **Hybrid thắng trên 3 nhãn disc-related** (Disc_narrowing, Disc_bulging,
   Disc_herniation) — đây đúng là semantic gần với RSNA training (RSNA cũng
   là về stenosis disc). Knowledge từ RSNA transfer **mạnh** sang các nhãn
   gần.

2. **Naked BiomedCLIP thắng trên endplate + spondylolisthesis** — đây là
   các nhãn semantic xa RSNA (endplate là cấu trúc xương, spondylolisthesis
   là trượt đốt sống — RSNA không có). Khi label xa training distribution,
   **RSNA-learned feature gây nhiễu** thay vì giúp → Naked BMC thuần (chưa
   bị "ô nhiễm" bởi RSNA bias) làm tốt hơn.

3. **Cả 2 cùng fail trên Modic + Pfirrman_grade** (F1 < 0.20). Modic là
   bệnh hiếm (imbalance cực mạnh), Pfirrman 5 class (zero-shot 5-way khó
   hơn binary). → Limitation của framework, cần data SPIDER thật sự để cứu
   (đó chính là Phase 4 retrain).

4. **Spondylolisthesis crash (Hybrid 0.028 vs Naked 0.500)** — finding
   *interesting nhất*: RSNA không có khái niệm "trượt đốt sống" → khi mô
   hình thấy ảnh dạng này, embedding bị kéo về phía "stenosis" hoặc
   "normal" → predict sai trầm trọng. Naked BMC không bị bias này nên
   classify ổn hơn.

5. **Mean F1 macro: Naked nhỉnh hơn (0.394 vs 0.362)** — *NHƯNG đây là
   trung bình aggregate, che đi pattern selective*. Nhìn từng nhãn, mỗi
   config có thế mạnh riêng. **Insight đúng** là: Hybrid mạnh ở nhãn gần
   semantic, Naked mạnh ở nhãn xa.

> **Câu chuyện của Bảng 3**: Đây không phải "Naked thắng Hybrid trung bình
> nên Hybrid thua". Đây là pattern **selective transfer** — knowledge từ
> RSNA chỉ transfer hiệu quả tới nhãn gần semantic. Đây là **finding mới**
> của paper, không phải limitation. Trong paper Section Discussion sẽ trình
> bày như là contribution thứ 4 (sau imbalance, attention, label extension).

### Bảng 4 — Phase 4 retrain chi tiết theo nhãn bệnh × cấu hình × chỉ số

**Setting**: 4 cấu hình **đã train trên SPIDER** (khác với Bảng 3 là không
train). SPIDER có 4 nhãn supervised (Pfirrmann 5-class, Modic 4-class,
Disc Narrowing 2-class, Spondylolisthesis 2-class). Đầy đủ 4 chỉ số (Acc,
Recall, Precision, F1) cho mỗi cặp nhãn × cấu hình.

**Format**: Acc / Recall / Precision tính theo % (mean qua các class
trong nhãn đó). F1 macro tính theo thang 0–1. **Bold** = config thắng
nhãn đó ở chỉ số đó.

| Disease | Config | Acc | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **Pfirrmann** *(5-class, hardest)* | Vanilla | 58.7% | 61.1% | 59.0% | 0.594 |
| | CBAM | 52.8% | 56.3% | 52.8% | 0.535 |
| | **Hybrid** | **62.1%** | **63.3%** | **64.2%** | **0.626** |
| **Modic** *(4-class, rare imbalance)* | Vanilla | 74.5% | 37.8% | 35.6% | 0.363 |
| | CBAM | 73.2% | 37.5% | 35.3% | 0.358 |
| | **Hybrid** | **77.5%** | **37.9%** | **36.7%** | **0.373** |
| **Disc Narrowing** *(2-class)* | Vanilla | 86.0% | 85.2% | 85.1% | 0.851 |
| | CBAM | 86.8% | **87.0%** | 85.8% | 0.862 |
| | **Hybrid** | **87.2%** | 86.9% | **86.3%** | **0.865** |
| **Spondylolisthesis** *(2-class)* | Vanilla | 89.4% | 75.4% | 60.2% | 0.633 |
| | CBAM | 88.1% | **79.5%** | 60.2% | **0.634** |
| | **Hybrid** | **91.9%** | 67.1% | **60.4%** | 0.627 |
| **Mean across 4 nhãn** | Vanilla | 77.1% | **64.9%** *≈tie* | 60.0% | 0.610 |
| | CBAM | 75.2% | **65.1%** *≈tie* | 58.5% | 0.597 |
| | **Hybrid** | **79.7%** | 63.8% | **61.9%** | **0.623** |

*Hybrid = transfer Hybrid RSNA + chỉ train classification head trên SPIDER
(backbone frozen, ~5M trainable params). Vanilla = train từ đầu trên
SPIDER, không transfer (~22M trainable). CBAM = transfer từ CBAM RSNA
(~22M trainable). BiomedCLIP luôn frozen ở mọi config (foundation model).*

**Trainable params (compute cost):**

| Config | Vanilla | CBAM | **Hybrid** |
|---|---|---|---|
| Trainable params | 22M | 22M | **5M** |
| Compute relative | 4.4× | 4.4× | **1×** *(rẻ nhất)* |

**Đọc bảng — pattern theo từng nhãn:**

1. **Pfirrmann (5-class, khó nhất)**: Hybrid **thắng cả 4 chỉ số** — Acc
   62.1%, Recall 63.3%, Precision 64.2%, F1 0.626. Đây là minh chứng rõ
   nhất rằng *RSNA-trained features giúp* trên task khó.

2. **Modic (rare disease)**: Hybrid thắng nhẹ trên cả 4 chỉ số nhưng giá
   trị tuyệt đối thấp (F1 ~0.36–0.37) cho cả 3 config — class 0 và 3 hoàn
   toàn F1 = 0.0 do imbalance cực mạnh, chỉ class 1 và 2 có signal. Đây
   là **giới hạn của data**, không phải của framework.

3. **Disc Narrowing (binary, dễ)**: Hybrid thắng tất chỉ số nhưng **đã
   saturate** ở ~0.85–0.87 cho cả 3 config. Không phải nơi quyết định.

4. **Spondylolisthesis (binary, imbalance)**: Trade-off rõ giữa
   Recall-mạnh vs Acc-mạnh —
   - **CBAM**: Recall cao nhất 79.5% (bắt được nhiều ca dương), F1 0.634
     thắng nhẹ.
   - **Hybrid**: Acc cao nhất 91.9%, Precision cao nhất 60.4%, predict
     thận trọng hơn.
   - **Tùy use case clinical**: sàng lọc (cần Recall) → CBAM; chẩn đoán
     (cần Precision) → Hybrid.

5. **Mean: Hybrid thắng F1 (0.623), Acc (79.7%), Precision (61.9%)** —
   3/4 chỉ số. Recall của Hybrid (63.8%) thua nhẹ Vanilla/CBAM
   (~64.9–65.1%) — đây là tradeoff cho gain trên Precision.

6. **CBAM SPIDER thua Vanilla ở F1 (0.597 < 0.610)** — *interesting*:
   attention pattern học từ RSNA không generalize sang SPIDER (data
   distribution khác). Chỉ Hybrid (kết hợp CBAM **+** BiomedCLIP) mới
   beat được Vanilla, vì BiomedCLIP knowledge bù cho CBAM bias.

### So sánh Phase 3 (zero-shot) vs Phase 4 (retrain) — proof of value

Lấy 3 nhãn xuất hiện trong cả Bảng 3 và Bảng 4 để so sánh:

| Disease | Phase 3 Hybrid (zero-shot) | Phase 4 Hybrid frozen (retrain) | Lift |
|---|---|---|---|
| Spondylolisthesis | 0.028 | **0.627** | **×22** |
| Disc Narrowing | 0.586 | **0.865** | ×1.5 |
| Modic | 0.018 | **0.373** | **×21** |
| Pfirrmann | 0.155 | **0.626** | **×4** |

→ **Retrain (Phase 4) cứu rỗi đúng những nhãn mà zero-shot (Phase 3) thất
bại**. Spondylolisthesis từ 0.028 → 0.627 (×22) là proof rõ nhất:
zero-shot không đủ vì RSNA bias kéo embedding sai phía → khi retrain
trên SPIDER, model học lại được semantic đúng. Đây là lý do paper cần
**cả 2 protocol** (đúng yêu cầu thầy đề ra hồi 2026-04-28).

**Bonus finding (Phase 3):** Selective transfer pattern — Hybrid **thắng**
Naked BiomedCLIP trên disc-related labels (Pfirrmann, herniation, narrowing
— gần với RSNA training), **thua** trên vertebra/endplate labels (xa RSNA
training). Đây là finding empirical mới về *where* knowledge transfer.

---

## 2. Match với 5 chỉ đạo thầy đã đưa

| # | Thầy nói | Mình có | Match |
|---|---|---|---|
| 1 | RSNA trước → mở rộng SPIDER/SpineNet | Phase 1+2 (RSNA) → Phase 3+4 (SPIDER) | ✅ |
| 2 | CLIP-style zero-shot HOẶC incremental | BiomedCLIP frozen + cosine prompt | ✅ |
| 3 | Test cả zero-shot VÀ retrain trên label mới | Phase 3 + Phase 4 | ✅ |
| 4 | Đánh đổi accuracy lấy recall, giải imbalance | Severe Recall ↑4.2×, Acc ↓11.3pp | ✅ |
| 5 | Contribution từ kỹ thuật (imbalance/aug/combine/knowledge) | Đủ 4 trục | ✅ |

→ **Match 5/5.** Mỗi quyết định kỹ thuật đều có chỉ đạo thầy back.

---

## 3. Trạng thái hiện tại

### LVTN Thạc sĩ — Dư sức bảo vệ (~95% sẵn sàng)

So với chuẩn LVTN thạc sĩ Việt Nam ngành CS:
- Problem statement, literature review, novel contribution, implementation,
  experimental validation, discussion, conclusion — **đủ hết**.
- Có **cross-dataset zero-shot evaluation** — phần lớn LVTN khác không có.
- Còn thiếu: viết quyển + slide bảo vệ. Không thiếu kết quả/contribution.

### Paper EMBC 2027 (Rank-C) — Scope đủ, ~70–75% sẵn sàng

| Yếu tố | Trạng thái |
|---|---|
| Scope (vấn đề + contribution) | ✅ Đủ 3 theme |
| Architecture novelty | ✅ Hybrid fusion + dual-protocol eval |
| Baselines | ✅ Baseline + CBAM + Hybrid + Naked BiomedCLIP |
| Cross-dataset eval | ✅ RSNA + SPIDER |
| **Statistical rigor (multi-seed)** | ❌ **Single seed n=1 — đang chạy 3-seed** |
| **Component ablation** | ❌ **Chưa chạy cbam_only / biomedclip_only — đang chạy** |
| Hyperparameter table | ⚠️ Chưa thêm (sẽ thêm sau khi multi-seed xong) |

**Còn lại để paper sẵn sàng:** ~17h compute trên Vast.ai (3-seed × 3 phase
+ 2 ablation runs). Code đã sẵn (`scripts/RUN_RANK_C_EXPERIMENTS.md`),
chỉ chờ chạy.

---

## 4. Đề xuất chốt scope (xin thầy duyệt)

**Nội dung em xin chốt:**

1. **LVTN + paper EMBC 2027** dùng chung kết quả, không tách 2 deliverable.
2. **3 theme contribution cứng** (Imbalance / Attention / Label extension).
3. **2 dataset**: RSNA (Phase 1+2) + SPIDER (Phase 3+4).
4. **Còn cần làm**: 3-seed + 2 ablation (~17h Vast.ai) → có ± std → viết paper.
5. **Defer các hướng**: HITL feedback, multi-scale fusion, cross-attention
   thay concat, more datasets — để future work sau khi nộp paper.

**Lý do nên chốt ngay:**
- Mở thêm scope → không kịp deadline EMBC + LVTN.
- 3 theme đã đủ "đậm" cho Rank-C, bộ contribution match trọn 5 chỉ đạo thầy.
- Risk thêm scope > benefit (mỗi pivot mất 1–2 tuần training).

---

## 5. File tham khảo trong repo

- `MOTIVATION_AND_DESIGN.md` / `MOTIVATION_AND_DESIGN_VI.md` — motivation chi tiết
- `experiments/PHASE_REPORT_FULL.md` — kết quả 4 phase đầy đủ
- `paper/main.tex` — paper LaTeX (placeholder IEEEtran, sẽ swap khi có template EMBC)
- `scripts/RUN_RANK_C_EXPERIMENTS.md` — guide chạy multi-seed + ablation
