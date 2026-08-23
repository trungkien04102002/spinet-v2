# SpineNetV2 + BiomedCLIP — Project Overview

> **Mục đích**: high-level view của toàn bộ pipeline để dễ theo dõi tổng thể, không đi vào số liệu chi tiết. Mỗi phase có 1 diagram + 1-2 dòng giải thích + status.
>
> Phù hợp để forward thầy review. Số liệu chi tiết xem trong các report riêng:
> - `experiments/fresh_cbam/RESULTS_SUMMARY.md` (Phase 1+2 RSNA results)
> - `experiments/paper_results/spider_zeroshot/SUMMARY_FOR_ADVISOR.md` (Phase 3 SPIDER zero-shot)
> - (Phase 4 chưa có)

## Color legend (Mermaid)

| Color | Meaning |
|---|---|
| 🟢 green | Done |
| 🟠 orange | TODO |
| 🟡 yellow | Pretrained / loaded from previous phase |
| 🔵 blue | Trained in this phase |
| ⚫ dark gray | Frozen pretrained (BiomedCLIP) |

## Big picture

Mở rộng SpineNetV2 (Windsor et al. 2024) với 2 cải tiến:
1. **CBAM attention** thêm vào 3D ResNet34 backbone → cải thiện grading trên RSNA
2. **Hybrid integration** (CBAM backbone + frozen BiomedCLIP) → enable **zero-shot label extension** sang SPIDER mà không cần re-train

```mermaid
flowchart LR
    RSNA[(RSNA 2024<br/>1942 IVDs<br/>3 stenosis labels)]
    SPIDER[(SPIDER<br/>1439 IVDs<br/>8 disease labels<br/>khác RSNA)]

    RSNA -->|train| M1[Baseline SpineNet]
    RSNA -->|train| M2[CBAM SpineNet]
    RSNA -->|train| M3[Hybrid SpineNet]

    M3 -->|"zero-shot<br/>(no SPIDER training)"| SPIDER
    M1 -->|fine-tune| SPIDER
    M2 -->|fine-tune| SPIDER

    classDef done fill:#43A047,stroke:#1B5E20,color:#fff
    classDef todo fill:#FB8C00,stroke:#E65100,color:#fff
    class M1,M2,M3 done
```

## Phase 1 + 2 — Train 3 RSNA models ✅ DONE

**Goal**: produce 3 checkpoints to show progressive improvement on RSNA validation set.

**Status**: All 3 trained. Best Hybrid Avg Severe F1 = 0.353.

### High-level architectures

> 📎 **Hybrid architecture chi tiết** (3 sources):
> - [`experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md`](experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md) — Mermaid diagram + giải thích từng phần + 2 feedback của thầy đã address
> - [`experiments/hybrid_architecture.png`](experiments/hybrid_architecture.png) — visual PNG full diagram
> - [`HYBRID_TRAINING_GUIDE.md`](HYBRID_TRAINING_GUIDE.md) — forward pass text step-by-step + hyperparameters
>
> 📎 **Code**: [`spinenet/models/grading_hybrid.py`](spinenet/models/grading_hybrid.py)

```mermaid
flowchart TD
    subgraph Baseline["Baseline SpineNet"]
        IB[Volume<br/>9 × 112 × 224] --> RB[3D ResNet34]
        RB --> HB[3 fc heads<br/>spinal/L_fora/R_fora<br/>3 classes each]
    end

    subgraph CBAM["CBAM SpineNet"]
        IC[Volume] --> RC[3D ResNet34<br/>+ CBAM blocks]
        RC --> HC[3 fc heads]
    end

    subgraph Hybrid["Hybrid SpineNet"]
        IH[Volume] --> RH[CBAM<br/>backbone]
        IH --> BH[BiomedCLIP<br/>per-slice<br/>FROZEN]
        RH --> CH[concat 1024]
        BH -->|attn pool 9 slices| CH
        CH --> PH[Projection MLP<br/>1024→768→512]
        PH --> EH[image_emb<br/>L2-normalized]
    end

    classDef frozen fill:#616161,stroke:#212121,color:#fff
    classDef trained fill:#1E88E5,stroke:#0D47A1,color:#fff
    class BH frozen
    class RB,HB,RC,HC,RH,PH,EH trained
```

### Pipeline

```mermaid
flowchart LR
    DATA[(RSNA<br/>9×112×224 .npy)] --> T1[train Baseline<br/>30 epochs]
    DATA --> T2[train CBAM<br/>+ sqrt class weights<br/>20 epochs]
    DATA --> T3[train Hybrid<br/>+ SupCon + sqrt CW<br/>20 epochs]

    T1 --> CK1[best_model_baseline.pth]
    T2 --> CK2[best_model_attention.pth]
    T3 --> CK3[best_model_hybrid.pth]

    classDef done fill:#43A047,stroke:#1B5E20,color:#fff
    class CK1,CK2,CK3 done
```

## Phase 3 — Zero-shot SPIDER evaluation ✅ DONE

**Goal**: kiểm tra Hybrid (chỉ train trên RSNA) có thể predict các label SPIDER chưa từng thấy hay không, **không fine-tune**.

**Status**: Hybrid + Naked BiomedCLIP ablation đã chạy. Finding: selective transfer (Hybrid thắng disc-related, thua vertebra-related).

### Zero-shot pipeline

```mermaid
flowchart LR
    SP[(SPIDER<br/>1439 IVDs<br/>9×112×224)]

    SP --> EI[encode_image<br/>Hybrid]
    EI --> IE[image_emb<br/>512-d L2 norm]

    LBL[("8 disease prompts<br/>e.g. 'lumbar disc herniation'")]
    LBL --> ET[encode_text<br/>BiomedCLIP frozen]
    ET --> TE[text_emb per class<br/>512-d L2 norm]

    IE --> CS[cosine similarity]
    TE --> CS
    CS --> AM[argmax]
    AM --> P[predicted class]

    classDef trained fill:#1E88E5,stroke:#0D47A1,color:#fff
    classDef frozen fill:#616161,stroke:#212121,color:#fff
    class EI,IE trained
    class ET,TE frozen
```

### Ablation: Hybrid vs Naked BiomedCLIP

```mermaid
flowchart TD
    SP[(SPIDER 1439 IVDs)]

    SP --> H[Hybrid<br/>CBAM + BiomedCLIP + trained projection]
    SP --> N[Naked BiomedCLIP<br/>only BiomedCLIP, mean-pool, NO training]

    H --> R1[Per-disease F1]
    N --> R2[Per-disease F1]

    R1 --> CMP{compare}
    R2 --> CMP
    CMP --> ANS["Selective transfer:<br/>RSNA training helps<br/>disc-related,<br/>hurts vertebra-related"]

    classDef done fill:#43A047,stroke:#1B5E20,color:#fff
    class H,N,ANS done
```

## Phase 4 — SPIDER transfer learning ⏳ TODO

**Goal**: 3-way comparison trên SPIDER fine-tune để chứng minh (a) CBAM > Vanilla, (b) BiomedCLIP integration giữ được zero-shot capability mà không hy sinh accuracy nhiều, (c) fine-tune cứu được Phase 3 failure modes.

**Status**: code đang chuẩn bị, run trên Vast.ai (~7-8h GPU cho 3 runs).

### 3 model variants

| # | Model | Output mechanism | Init từ | Zero-shot sau Phase 4? |
|---|---|---|---|---|
| 1 | **Vanilla SPIDER** | 4 hardcoded fc heads | RSNA Baseline | ❌ |
| 2 | **CBAM SPIDER**    | 4 hardcoded fc heads | RSNA CBAM | ❌ |
| 3 | **Hybrid SPIDER**  | Cosine sim với text prompts | RSNA Hybrid | ✅ |

### Architecture

```mermaid
flowchart TD
    subgraph BL["1. Vanilla SPIDER"]
        IB[Volume] --> RB[3D ResNet34<br/>load from RSNA Baseline]
        RB --> HB["4 hardcoded heads<br/>random init"]
    end

    subgraph CB["2. CBAM SPIDER"]
        IC[Volume] --> RC[3D ResNet34 + CBAM<br/>load from RSNA CBAM]
        RC --> HC["4 hardcoded heads<br/>random init"]
    end

    subgraph HY["3. Hybrid SPIDER (preserves zero-shot)"]
        IH[Volume] --> RH[CBAM backbone<br/>load from RSNA Hybrid]
        IH --> BH[BiomedCLIP per-slice<br/>frozen]
        RH --> CH[concat 1024]
        BH --> CH
        CH --> PH[Projection MLP<br/>load + fine-tune]
        PH --> EH[image_emb 512]
        EH --> SIM[cosine sim<br/>with SPIDER text prompts]
    end

    classDef pretrained fill:#FBC02D,stroke:#F57F17,color:#000
    classDef trained fill:#1E88E5,stroke:#0D47A1,color:#fff
    classDef frozen fill:#616161,stroke:#212121,color:#fff
    class RB,RC,RH pretrained
    class HB,HC,PH,EH,SIM trained
    class BH frozen
```

### Fine-tune pipeline (3 runs)

```mermaid
flowchart LR
    CK1[RSNA Baseline ckpt] --> M1[Vanilla SPIDER]
    CK2[RSNA CBAM ckpt] --> M2[CBAM SPIDER]
    CK3[RSNA Hybrid ckpt] --> M3[Hybrid SPIDER]

    SP_TR[(SPIDER train<br/>~360 series)] --> M1
    SP_TR --> M2
    SP_TR --> M3

    M1 --> F1[Vanilla checkpoint]
    M2 --> F2[CBAM checkpoint]
    M3 --> F3[Hybrid checkpoint]

    SP_VAL[(SPIDER val<br/>~87 series)] --> EV[F1 macro<br/>per condition]
    F1 --> EV
    F2 --> EV
    F3 --> EV

    EV --> CMP[3-way comparison<br/>Vanilla vs CBAM vs Hybrid]

    classDef todo fill:#FB8C00,stroke:#E65100,color:#fff
    class M1,M2,M3,F1,F2,F3,EV,CMP todo
```

### 4 SPIDER conditions chọn cho Phase 4

| # | Label | Type | Anatomy | Phase 3 zero-shot status | Why pick |
|---|---|---|---|---|---|
| 1 | Pfirrmann        | 5-class | disc degeneration       | tied (0.155)        | clinical gold standard |
| 2 | Modic            | 4-class | vertebra inflammation   | FAIL (0.018)        | multiclass + zero-shot fail → fine-tune cứu |
| 3 | Disc_narrowing   | binary  | disc structure          | WIN (0.586)         | sanity: fine-tune ≥ zero-shot? |
| 4 | Spondylolisthesis| binary  | spinal alignment        | FAIL (0.028)        | alignment task → fine-tune cứu |

### Commands trên Vast.ai

```bash
# Run 1: Vanilla SPIDER
python3 train_spider.py --model baseline \
    --rsna-checkpoint checkpoints/best_model.pth \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3

# Run 2: CBAM SPIDER
python3 train_spider.py --model cbam \
    --rsna-checkpoint checkpoints/best_model_attention.pth \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3

# Run 3: Hybrid SPIDER (preserves zero-shot)
python3 train_spider_hybrid.py \
    --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --epochs 15 --batch-size 16 --lr 1e-4
```

## Phase 5 — Paper writing ⏳ TODO

**Goal**: viết paper EMBC 2027 (Rank-C) với 2 stories complementary.

```mermaid
flowchart TD
    P1[Phase 1+2:<br/>RSNA progressive improvement<br/>Baseline → CBAM → Hybrid]

    P3[Phase 3:<br/>Zero-shot SPIDER<br/>Hybrid vs Naked BiomedCLIP<br/>→ selective transfer]

    P4[Phase 4:<br/>Fine-tune SPIDER<br/>CBAM vs Vanilla<br/>→ recovers zero-shot fails]

    P1 --> P3
    P1 --> P4

    P3 --> ST["Story 1:<br/>BiomedCLIP integration<br/>enables zero-shot for<br/>related concepts"]
    P4 --> ST2["Story 2:<br/>CBAM helps when<br/>fine-tuning to new dataset"]

    ST --> PAPER[EMBC 2027 paper]
    ST2 --> PAPER

    classDef done fill:#43A047,stroke:#1B5E20,color:#fff
    classDef todo fill:#FB8C00,stroke:#E65100,color:#fff
    class P1,P3 done
    class P4,ST,ST2,PAPER todo
```

## Status summary

| Phase | What | Status | Where to find results |
|---|---|---|---|
| 1+2  | Train Baseline + CBAM + Hybrid on RSNA       | ✅ done | `experiments/fresh_cbam/RESULTS_SUMMARY.md` |
| 3    | Zero-shot SPIDER (Hybrid + Naked ablation)   | ✅ done | `experiments/paper_results/spider_zeroshot/SUMMARY_FOR_ADVISOR.md` |
| 4    | Transfer learning SPIDER (4 labels)          | ⏳ TODO | (Vast.ai GPU, ~5h) |
| 5    | Paper writing                                | ⏳ TODO | (after Phase 4) |

## Next concrete actions

1. Commit current code (Phase 3 fixes + reports)
2. Zip SPIDER dataset, upload Drive
3. Modify `grading_spider.py` to support 4 heads (currently 3)
4. Launch Vast.ai instance, restore RSNA checkpoints
5. Run Phase 4: 2 fine-tune runs (Baseline, CBAM)
6. Backup Phase 4 results to local
7. Update this overview with Phase 4 results
