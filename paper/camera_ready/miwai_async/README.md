# MIWAI 2026, paper 122 — asynchronous session bundle

Ha Trung Kien and Phan Trong Nhan, HCMUT, VNU-HCM.
Paper: *A Two-Branch Hybrid Model for Imbalanced Lumbar Disc Grading and Zero-Shot Label Extension.*

## What to upload

Portal <https://miwai26.cs-smu.ca/prepost/>, site password `2026MIWAIsaintmarys`.
The form asks for the paper title (pick it from the list), Paper ID `122`, and **the exact
address the invitation arrived at** — check the "to" line of the message, it is validated
against the organisers' records.

Deadline **7 October 23:59 AST**, which is about **08:00 on 8 October Vietnam time**. Upload
early: the organisers warn that files over 50 MB are slow.

| Upload slot | File | Limit | Status |
|---|---|---|---|
| Abstract | `MIWAI2026_Paper122_HaTrungKien_Abstract.txt` | txt UTF-8, 2,500 chars | 2,015 chars |
| Infographic | `MIWAI2026_Paper122_HaTrungKien_Infographic.pdf` | PDF or PNG, 30 MB | 0.53 MB, 1 page |
| Slides | `MIWAI2026_Paper122_HaTrungKien_Slides.pdf` | PDF, 30 MB | 1.89 MB, 25 pages |
| Video | `MIWAI2026_Paper122_HaTrungKien_Video.mp4` | MP4, 300 MB, 20 min | **not recorded yet** |

`..._Infographic.png` (1920x1080) is a fallback if the portal refuses PDF. Upload the PDF first.

## Recording the video

`TALK_SCRIPT.md` is the script: 15 sections, one per spoken slide, each with its own time
budget, summing to about 13 minutes. The organisers accept a plain voice-over on the slides,
and do not require the author's own voice or any webcam footage.

Talk over pages 1 to 15 only. Pages 16 to 19 are references, 20 is the thank-you slide, and
21 to 25 are backup slides held for the live question period on 10 October. The `# Questions`
section of the script is for that live session, not for the video.

If the recording comes out over 300 MB:

    ffmpeg -i raw.mov -c:v libx264 -preset slow -crf 23 -vf scale=1920:-2 \
           -c:a aac -b:a 128k -movflags +faststart \
           MIWAI2026_Paper122_HaTrungKien_Video.mp4

## Rebuilding

The infographic is built here; the slides are built one directory up and copied in.

    lualatex -interaction=nonstopmode infographic.tex
    pdftoppm -png -scale-to-x 1920 -scale-to-y 1080 infographic.pdf ig

Both commands write `infographic.pdf` / `ig-1.png`, so **rename the output** back to
`MIWAI2026_Paper122_HaTrungKien_Infographic.{pdf,png}` afterwards, or the renamed copy in this
folder goes stale while the build output sits beside it unnoticed.

Build inputs, not uploaded: `infographic.tex`, `mri_sagittal_panel.png`, `gradcam_nocbam.png`,
`gradcam_cbam.png`, `rsna_dist_compact.png`, `miwai_logo.png`.

The architecture drawing is the paper's own Fig. 1, used verbatim so the infographic, the slides
and the paper all show the same figure. PDFs are gitignored, so the two copies here are derived
and have to be remade after a fresh clone:

    cp ../figures/hybrid_arch_generic.pdf .
    pdfcrop --margins 2 hybrid_arch_generic.pdf hybrid_arch_crop.pdf

`pdfcrop` trims the figure's own margins, which is what lets it sit at 158mm here against the
118mm it gets in the paper.

## Live presentation

Saturday 10 October, 10:00 Halifax (UTC-3) — **20:00 Vietnam time**. Online, Track 1
Session 1, about 20 minutes per paper. The organisers send a connection-test link roughly a
week beforehand and the session chair confirms the running order.
