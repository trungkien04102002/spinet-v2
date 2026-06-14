# Các thay đổi cần dán lên Overleaf (main_pair.tex + references.bib)

> Cách dán: tìm đúng đoạn → bôi đen → dán đè. **KHÔNG Ctrl+A cả file** (sẽ đè phần thầy sửa).
> Đây là các sửa theo feedback thầy: bớt "liệt kê đồ có sẵn", dẫn bằng điểm mạnh design; cite concept (CBAM, Grad-CAM).

---

## 1) §1 Introduction — đoạn motivation (đoạn 2)
**Tìm** đoạn bắt đầu `Existing graders address neither gap:` → **dán đè cả đoạn** bằng:

```latex
Existing graders address neither gap: trained from scratch on a single dataset, they overfit its minority-class statistics and lock in a fixed head, so accuracy drops on a new center and a new label schema is out of reach without retraining. We design for the two failures directly. For imbalance, we anchor the volumetric encoder with a \emph{frozen biomedical foundation model}~\cite{zhang2023biomedclip} whose pretraining on 15 million image-text pairs gives a stable prior where Severe examples are too scarce to learn from scratch, and we add channel-spatial attention~\cite{woo2018cbam} so the network focuses on the small region a Severe lesion occupies rather than the whole disc. For the fixed label space, we exploit the same model's shared image-text embedding~\cite{radford2021clip}, which lets a new label set enter inference as a text prompt instead of a retrained classifier head. Our model realizes this as a CBAM-equipped 3D ResNet-34 fused with a frozen BiomedCLIP encoder through a small trainable head. Our main contributions, centered on the cross-dataset transfer finding, are as follows.
```

---

## 2) §1 Introduction — contributions
**Tìm** khối `\begin{itemize} ... \end{itemize}` ngay sau đoạn trên → **dán đè cả khối** bằng:

```latex
\begin{itemize}
    \item \textbf{A cross-dataset transfer finding.} Channel-spatial attention sharpens in-domain Severe metrics but \emph{degrades} cross-center SPIDER transfer below a plain SpineNetV2 backbone; adding a frozen biomedical foundation branch restores it ($+0.034$ over the attention-only variant, $p=0.011$), which we read as the foundation model regularizing attention-induced overfitting.
    \item \textbf{A design that reconciles minority sensitivity with cross-center robustness.} We couple a volumetric attention encoder~\cite{woo2018cbam,he2016resnet} with a frozen BiomedCLIP~\cite{zhang2023biomedclip} prior through a small learned fusion; an ablation shows the two branches are complementary rather than redundant.
    \item \textbf{Label-space extension without retraining.} Matching the fused embedding to text prompts lets the RSNA-trained model grade eight unseen SPIDER labels zero-shot (mean F1 $0.362$), which the fixed-head baselines cannot do.
    \item \textbf{An imbalance recipe that recovers the minority class.} Focal loss~\cite{lin2017focal}, square-root class weighting, minority oversampling, and augmentation together lift the near-zero Severe recall of standard training about $4\times$ (12.3\% to 48.6\%) on RSNA 2024.
\end{itemize}
```

---

## 3) §4.5 Qualitative Results — thêm cite Grad-CAM
**Tìm** `shows Grad-CAM activation` → sửa thành:

```latex
shows Grad-CAM~\cite{selvaraju2017gradcam} activation
```

---

## 4) references.bib — thêm entry Grad-CAM
**Dán vào cuối `references.bib`** (không đụng entry khác):

```latex
@inproceedings{selvaraju2017gradcam,
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  title     = {{Grad-CAM}: Visual Explanations from Deep Networks via Gradient-Based Localization},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {618--626},
  year      = {2017},
  doi       = {10.1109/ICCV.2017.74}
}
```

---

## 5) Abstract — bỏ "SpineNetV2-style"
**Tìm** `and SpineNetV2-style augmentation,` → sửa thành:

```latex
and standard geometric and intensity augmentation,
```
(Chỉ sửa trong abstract. Chỗ §3.4 "follows SpineNetV2~\cite{windsor2022}" giữ nguyên vì đã cite + giải thích.)

---

## 6) Abstract — thêm mốc đầu cho Severe F1
**Tìm** `and raises Severe F1 to 0.356.` → sửa thành:

```latex
and raises Severe F1 from 0.152 to 0.356.
```
(0.152 = Severe F1 của SpineNetV2 baseline, Table 1 — cho song song với recall "12.3% to 48.6%".)

---

## 7) §6 Summary — reframe design-first + bỏ "three-seed"
**Tìm** đoạn `We presented a hybrid CBAM-3D and frozen-BiomedCLIP model...` → **dán đè cả đoạn:**

```latex
We presented a two-branch model that pairs volumetric attention with a frozen biomedical foundation encoder for lumbar IVD grading on RSNA 2024. It markedly improves the minority Severe class (Recall $\sim$4$\times$, F1 2.3$\times$, AUC 0.899) at a controlled accuracy trade-off and extends zero-shot to eight unseen SPIDER labels without retraining. On cross-dataset transfer, attention alone drops F1 below plain SpineNetV2 while the frozen foundation branch restores it, an effect we interpret as cross-dataset regularization. Future work includes a mechanistic representation analysis, a learned slice-pooling adapter~\cite{lu2024radclip}, and axial-T2 fusion.
```
(Bỏ "three-seed" ở Summary. GIỮ "mean ± std over 3 seeds" trong §4 bảng — đó là minh bạch phương pháp.)

---

## Ghi chú
- Title, "In this paper", keyword "low back pain", merge đoạn intro, cite RSNA/SPIDER ở §1 = **thầy đã tự sửa** trên Overleaf rồi, không cần dán lại.
- references.bib (full tên tác giả) để **làm sau** khi nội dung ngắn lại (hiện 13 trang).
- Đã verify Grad-CAM citation qua CrossRef (Selvaraju et al., ICCV 2017, pp. 618–626, doi 10.1109/ICCV.2017.74).
