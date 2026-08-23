# Liu et al. 2025 IJCIS — Full Extraction (Updated 2026-05-06)

## Access status
- **PDF downloaded successfully** via curl with browser User-Agent.
- **Local path**: `/tmp/aktan_2025.pdf` (3.6 MB, 33 pages)
- **Source**: https://link.springer.com/content/pdf/10.1007/s44196-025-01098-7.pdf
- **Open access**: Yes (CC BY 4.0, Springer Open Access)

## Bibliographic (corrected)
- **Title**: Automated Lumbar Spine Degenerative Classification Using Deep Learning: A Comprehensive Evaluation Based on RSNA 2024
- **Authors**: **Yonghong Liu, Javed Rashid, Salah Mahmoud Boulaaras, Muhammad Shoaib Saleem, Muhammad Faheem** (NOT "Aktan/Khan" — earlier search summary was wrong)
- **Journal**: International Journal of Computational Intelligence Systems
- **Volume / Issue / Article**: 18, 1, 328
- **Year**: 2025 (received April 3, 2025; accepted November 30, 2025; published December 17, 2025)
- **DOI**: 10.1007/s44196-025-01098-7

## CRITICAL: Task mismatch (read first)

This paper does **pixel-level segmentation of 6 anatomical regions**, NOT
severity grading. The paper itself states (page 21):

> "While the model effectively localizes and segments degenerative
> conditions, it does not directly grade severity levels, which are
> essential in clinical decision-making. This limitation arises because
> the RSNA 2024 dataset does not include standardized severity labels."

(The author's claim that RSNA 2024 lacks severity labels is incorrect — the
Kaggle taxonomy uses Normal/Mild, Moderate, Severe — but the paper does not
use those labels regardless.)

**The headline F1 = 95.65% is per-class pixel-Dice on 6 anatomical regions
(BG, LNFN, RNFN, LSS, RSS, SCS), not on severity grades.** It is NOT
directly comparable to our per-IVD severity F1.

## Dataset & split
- **Source**: RSNA 2024 LumbarDISC Kaggle dataset
- **Cohort**: 8,593 MRI series across 2,697 patients (Richards 2025)
- **6 segmentation classes**: Background (BG), Left Neural Foraminal Narrowing (LNFN), Right Neural Foraminal Narrowing (RNFN), Left Subarticular Stenosis (LSS), Right Subarticular Stenosis (RSS), Spinal Canal Stenosis (SCS)
- **Image preprocessing**: 256×256 px input, intensity normalised [0, 1]
- **Augmentation**: horizontal flips
- **Evaluation**: 5-fold cross-validation (no separate held-out test)
- **Split details**: 4 folds train, 1 fold validation per CV iteration

## Per-class metrics (Table 6, fully extracted)

Class-wise precision, recall, F1, accuracy, Dice on test set (from PDF page 22):

| Class | Precision | Recall | F1 Score | Accuracy | Dice |
|---|---|---|---|---|---|
| BG | 95.63% | 95.66% | **95.65%** | 95.25% | 95.69% |
| LNFN | 95.33% | 94.42% | 94.35% | 94.25% | 94.55% |
| RNFN | 94.91% | 94.94% | 94.92% | 94.25% | 94.73% |
| LSS | 93.29% | 92.62% | 93.46% | 93.13% | 93.71% |
| RSS | 90.33% | 91.65% | 93.99% | 95.25% | 94.77% |
| SCS | 92.06% | 93.24% | **94.65%** | 94.25% | 94.77% |
| **Average** | **93.25%** | **93.76%** | **94.5%** | **94.93%** | **94.61%** |

**Note**: These are pixel-level metrics for binary presence/absence per class
(one-hot encoded over 6 channels). The Background class dominates the average.

## Method details

- **Loss**: Combined Cross-Entropy + IoU (α = 0.5)
- **Optimizer**: Adam
- **Epochs**: 50
- **Augmentation**: horizontal flip + intensity normalisation
- **Backbone**: MobileNetV2 encoder + UNet decoder
- **Input/output**: 256×256×3 input → 256×256×6 output (softmax over 6 classes)
- **Model size**: ~18 MB (vs 95-110 MB for ResNet/DenseNet-UNet baselines per Table 11)
- **Training time**: ~2.3 hours per fold
- **Inference**: 12-15 ms/slice
- **Hardware**: not specified in detail (mentions "consumer-grade hardware")

## Ablation tables (Tables 7-10)

Table 7 (model variants):
| Variant | Precision | Recall | F1 | Acc | Dice |
|---|---|---|---|---|---|
| Base (MobileNetV2-UNet) | 95.25% | 95.76% | 95.65% | 94.93% | 94.61% |
| Without MobileNetV2 backbone | 90.24% | 89.47% | 90.11% | 91.24% | 87.06% |
| Without skip connections | 91.84% | 90.53% | 91.05% | 92.32% | 90.58% |

Table 10 (α weighting in CE+IoU loss): α=0.3 / 0.5 / 0.7 tested; α=0.5 best.

## Updated entry for SOTA Table 1

**Position as a segmentation comparator with explicit task-difference footnote:**

| Method | Year | Dataset | Backbone | F1 | AUC | Notes |
|---|---|---|---|---|---|---|
| Liu et al.~\cite{liu2025rsna} | 2025 | RSNA 2024 (5-fold CV, 8,593 series) | MobileNetV2-UNet | 0.957† | — | †Pixel-level segmentation of 6 anatomical regions, NOT severity grading |

This row is now in `paper/sections/05_results.tex` Table 1.

## Sources

- Springer DOI: 10.1007/s44196-025-01098-7
- PDF: https://link.springer.com/content/pdf/10.1007/s44196-025-01098-7.pdf
- Local extracted text: /tmp/aktan_2025.txt (1,569 lines)

## Final summary

PDF downloaded successfully via curl + extracted via pdftotext. The author
list is **Yonghong Liu, Javed Rashid, Salah Mahmoud Boulaaras, Muhammad
Shoaib Saleem, Muhammad Faheem** (the earlier "Aktan/Khan" was a search
artefact). The paper is **segmentation-only** (6 anatomical regions:
BG/LNFN/RNFN/LSS/RSS/SCS), not severity grading — explicitly stated by the
authors. Headline F1 95.65% is pixel-level Dice averaged over 6 classes
including Background. Per-class table (BG 95.65%, LNFN 94.35%, RNFN 94.92%,
LSS 93.46%, RSS 93.99%, SCS 94.65%) is fully extracted. Method:
MobileNetV2-UNet, 256×256, Adam, CE+IoU loss, 50 epochs, 5-fold CV. **The
appropriate framing in our SOTA Table 1 is "different task — segmentation
not classification" with a $\dagger\dagger$ footnote**, which is now
implemented in `paper/sections/05_results.tex`.
