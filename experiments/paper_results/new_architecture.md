# Pipeline Revised — Hybrid CBAM + BiomedCLIP

> Revised sau feedback thầy Nhan Phan (2026-04-21). Thay đổi chính so với version trước:
> 1. Thêm BiomedCLIP image encoder (không chỉ text encoder)
> 2. Show rõ text projection (đã có trong BiomedCLIP, frozen)
> 3. Hybrid 2 image encoder với fusion layer → contribution rõ ràng hơn

---

## 1. Kiến trúc đầy đủ (Hybrid Dual-Encoder)

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart TB
    V[3D IVV Volume<br/>9 x 112 x 224]

    V --> CBAM[CBAM 3D ResNet34<br/>đã train RSNA<br/>FROZEN]
    V --> SS[Slice Selection<br/>chọn 3 slice trung tâm]

    SS --> BMCI[BiomedCLIP Image Encoder<br/>ViT-B/16 2D<br/>FROZEN<br/>pretrain PubMed 15M cặp]
    BMCI --> AGG[Mean Pool 3 slice features]

    CBAM --> FCBAM[CBAM feature 512-dim<br/>spine-specific]
    AGG --> FBMC[BiomedCLIP image feature 512-dim<br/>medical generic]

    FCBAM --> CAT[Concat 1024-dim]
    FBMC --> CAT

    CAT --> PROJ[Image Projection Head<br/>Linear 1024 to 512 + GELU + Linear<br/>TRAINABLE]
    PROJ --> NI[L2 Normalize]

    T[Label text<br/>moderate spinal canal stenosis] --> TOK[Tokenizer]
    TOK --> BMCTE[BiomedCLIP Text Encoder<br/>PubMedBERT + text projection<br/>FROZEN]
    BMCTE --> NT[L2 Normalize]

    NI --> SIM[Cosine Similarity x temperature]
    NT --> SIM
    SIM --> OUT[Predicted label]

    style V fill:#e1f5ff
    style CBAM fill:#fff4e1
    style BMCI fill:#ffe1e1
    style BMCTE fill:#ffe1e1
    style PROJ fill:#fff4e1
    style OUT fill:#e1ffe1
```

---

## 2. 3 phần chính của pipeline

### Phần A — Image Path (Hybrid Dual-Encoder)

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    V[Volume 9x112x224] --> E1[CBAM 3D<br/>FROZEN<br/>spine domain]
    V --> SS[Slice Selection]
    SS --> E2[BiomedCLIP Image<br/>FROZEN<br/>medical generic]
    E1 --> M[Merge]
    E2 --> M
    M --> P[Image Projection<br/>TRAINABLE]
    P --> OUT[image emb 512]

    style E1 fill:#fff4e1
    style E2 fill:#ffe1e1
    style P fill:#fff4e1
    style OUT fill:#e1ffe1
```

**Đóng góp mới**: hybrid 2 encoder — CBAM spine-specific + BiomedCLIP medical generic → feature giàu hơn BiomedCLIP gốc.

---

### Phần B — Text Path (giữ nguyên BiomedCLIP)

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    T[Label text] --> TOK[Tokenizer]
    TOK --> TE[BiomedCLIP Text Encoder<br/>PubMedBERT + internal projection<br/>FROZEN]
    TE --> OUT[text emb 512]

    style TE fill:#ffe1e1
    style OUT fill:#e1ffe1
```

**Lưu ý**: BiomedCLIP text encoder đã có text projection internal (đã pretrain). Em không add projection trainable ở text side vì:
- Text encoder đã frozen (không đổi qua training)
- Projection internal đã đủ để đưa text vào không gian chung với image

---

### Phần C — Alignment (cosine similarity scaled)

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    IE[image emb 512 norm] --> SIM[dot product]
    TE[text emb 512 norm] --> SIM
    SIM --> SCALE[x temperature]
    SCALE --> L[Logits B x N_labels]
    L --> A[Argmax]
    A --> PRED[Prediction]

    style L fill:#fff4e1
    style PRED fill:#e1ffe1
```

**Same as BiomedCLIP inference**. Khi argmax → label có similarity cao nhất.

---

## 3. Trainable vs Frozen

| Component | Params | Train hay Frozen? | Lý do |
|---|---|---|---|
| CBAM 3D ResNet34 | ~22M | **Frozen** | Đã train RSNA tốt, không muốn destroy |
| BiomedCLIP Image Encoder | ~86M | **Frozen** | Pretrain 15M cặp, không touch |
| BiomedCLIP Text Encoder | ~110M | **Frozen** | Pretrain 15M cặp, không touch |
| **Image Projection Head** | **~500K** | **TRAINABLE** | Học cách fuse + align 2 image feature |
| **Temperature scale** | **1** | **TRAINABLE** | Contrastive scaling |
| Total | ~218M | **0.2% trainable** | Rất nhẹ, train nhanh |

---

## 4. So sánh với BiomedCLIP gốc

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    subgraph OLD [BiomedCLIP gốc]
        V1[2D Image] --> E1[ViT-B/16]
        E1 --> I1[image emb]
        T1[Text] --> TE1[PubMedBERT]
        TE1 --> TT1[text emb]
        I1 --> S1[cosine]
        TT1 --> S1
    end

    subgraph NEW [Pipeline của em]
        V2[3D Volume] --> C1[CBAM 3D]
        V2 --> S2[Slice selection]
        S2 --> B1[BiomedCLIP 2D]
        C1 --> F1[Fusion]
        B1 --> F1
        F1 --> P1[Projection]
        P1 --> I2[image emb]
        T2[Text] --> TE2[BiomedCLIP Text]
        TE2 --> TT2[text emb]
        I2 --> S3[cosine]
        TT2 --> S3
    end

    style C1 fill:#fff4e1
    style B1 fill:#ffe1e1
    style F1 fill:#fff4e1
    style P1 fill:#fff4e1
```

**Khác biệt chính**:
- BiomedCLIP gốc: 1 image encoder 2D generic
- Pipeline em: 2 image encoder (spine-specific + generic) + fusion + 3D support

---

## 5. 4 Contribution của paper

1. **CBAM attention** cải thiện feature quality ở lớp Severe (từ công trước)
2. **Focal + Uncertainty Loss** xử lý class imbalance (từ công trước)
3. **Hybrid dual-encoder image path**: CBAM 3D + BiomedCLIP 2D, fusion trainable → **NOVEL**
4. **Zero-shot label extension** via BiomedCLIP text encoder frozen → **NOVEL**

---

## 6. Lợi thế của hybrid approach

| Scenario | BiomedCLIP gốc only | CBAM only | **Hybrid (em)** |
|---|---|---|---|
| Zero-shot label mới | ✅ | ❌ | ✅ |
| 3D native | ❌ (2D only) | ✅ | ✅ |
| Spine-specific attention | ❌ | ✅ | ✅ |
| Medical foundation knowledge | ✅ (15M pairs) | ❌ | ✅ |
| Severe recall recovery | ❌ | ✅ | ✅ |

→ **Hybrid có đủ 5/5 ưu điểm**.

---

## 7. Risks + Mitigation

| Risk | Mitigation |
|---|---|
| Fusion layer khó converge | 2-stage training: train projection first với features frozen, sau đó optional LoRA fine-tune |
| BiomedCLIP 2D slice aggregation mất 3D info | Slice selection chọn đúng slice có thông tin nhất (IVV center), giống Lian et al. 2026 |
| Zero-shot accuracy thấp trên 3D | Fallback: few-shot fine-tune projection với 10-20 sample |
| Scope tăng lên 3 tuần | Claude Code support + AI agents parallelize tasks |

---

## 8. Precedent justify approach

- **Hybrid encoder**: Fusion of foundation model + domain-specific encoder is standard in medical VLM literature (cite: Lian et al. 2026, Wang et al. MedCLIP 2022)
- **2D encoder cho 3D data**: Lian et al. 2026 đã demo BiomedCLIP 2D work cho 3D medical — trên 4 A800 chỉ mất ~2 ngày
- **Frozen text + trainable image projection**: Standard CLIP zero-shot protocol (Radford et al. 2021)

---

## Summary

Pipeline này giải quyết đủ 3 điểm feedback của thầy:

✅ **Thêm BiomedCLIP image encoder** — không lãng phí half của BiomedCLIP nữa  
✅ **Text projection rõ ràng** — đã show internal projection của BiomedCLIP text (frozen)  
✅ **Novelty rõ ràng** — hybrid dual-encoder là contribution original, không chỉ thay Linear head

Scope ước tính: **3 tuần** (thay vì 2 tuần) do thêm fusion module + ablation 3 variants (CBAM only / BiomedCLIP only / Hybrid).
