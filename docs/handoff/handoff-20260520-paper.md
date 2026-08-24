# Handoff Document

**Created:** 2026-05-20
**Topic:** Conference paper review (5-reviewer panel) + LNCS reformat + LVTN citation fixes
**Status:** Paper submit-ready (LNCS 13pp); pending venue decision

---

## Context Summary

Session làm 3 việc chính:
1. **LVTN citation fixes** — verify + sửa citation trong bản LVTN Overleaf (`Internship2.zip`)
2. **Conference paper review** — chạy skill `academic-paper-reviewer` 2 passes (5 reviewers/pass) trên `REPORT_PAPER.tex`
3. **LNCS reformat + trim** — convert IEEE→LNCS, trim 17→13 pages

---

## Completed Work

### LVTN citation fixes (bản Overleaf, files ở `/tmp/internship2/` + synced `paper/lvtn_overleaf/`)
- [x] 5 bib entries sửa: zhang2024biomedclip→NEJM AI 2025, windsor2022context→MICCAI 2022, warszawer→ISMRM 2025, monzon→ML4H 2024, **xóa** aktan2025lumbar (unused)
- [x] §2.8 chap02: sửa 5 cite sai (SAM→kirillov2023sam, MedSAM→ma2024medsam, CLIP→radford2021clip, BiomedCLIP→zhang2024biomedclip ×2); thêm 3 bib entries gốc
- [x] Med3D citation thêm (chen2019med3d) ở chap04
- [x] Lỗi font Greek "αν Χίυ" ở SpineNetV2 ref → đổi @article→@misc với howpublished
- [x] Lời cam đoan + AI disclosure thêm vào front-matter main.tex (LVTN)

### Conference paper (`paper/REPORT_PAPER.tex` IEEE 8pp + `paper/REPORT_PAPER_LNCS.tex` LNCS 13pp)
- [x] Pass 1 review (5 reviewers): applied 6 edits (3/5 conditions honest, synergy→complementary, second-reader framing, Modic citation, inter-rater sentence, Aktan→Liu author)
- [x] Pass 2 review (5 reviewers): applied 5 fixes (C1 pooled σ 0.010→0.007 math error, C2 Aktan/Liu Table row, C3 "first BiomedCLIP" abstract soften, C4 nshen7 footnote, C5 "50 annotators" cite)
- [x] LNCS reformat: documentclass→llncs, author block, keywords env
- [x] Trim 17→13 pages: bỏ Fig CM+PR+Naked+GradCAM, compress Table I (bỏ 3 rows), inline computational cost, merge Discussion paragraphs, compress equations + bibliography, bỏ mcsweeney2023 unused ref

---

## Current State / Files

| File | Pages | Format | Status |
|---|---|---|---|
| `paper/REPORT_PAPER.tex` / `.pdf` | 8 | IEEE | Backup, 0 errors |
| `paper/REPORT_PAPER_LNCS.tex` / `.pdf` | **13** | Springer LNCS | **Submit-ready**, 0 errors, 0 undefined cites |
| `paper/lvtn_overleaf/` (+ `/tmp/internship2/`) | ~76 | LVTN | Citation fixes applied, user copy→Overleaf |

---

## Pending Work / Next Steps

1. **Venue decision** (chưa quyết):
   - MIWAI 2026 LNAI (deadline 8/6/2026) — 13/12pp cần 1 extra-page fee — pass ~45-55%
   - SoICT 2026 CCIS (deadline ~9/2026) — 13/15pp OK, deadline trễ — pass ~65-70% ← **em recommend**
   - CSoNet 2026 — scope mismatch, skip
2. **Nếu muốn đẩy pass% lên 70%+** (P2, optional, cần GPU):
   - 3-seed RSNA full ablation (~16 GPU hrs, $5-10) — kill main weakness "single-seed RSNA"
   - Bootstrap 95% CI RSNA Severe (30 min Python, no GPU)
   - Logit-average ensemble test (reject "trivial ensemble" alternative — DA flagged)
   - Document zero-shot text prompts (appendix)
3. **LVTN**: user copy bản fixed citation lên Overleaf, recompile verify

---

## Key Decisions / Notes

- **5-reviewer panel verdict (pass 2):** Major revision (DA flagged single-seed RSNA + "recovery wording" CRITICAL). Tech contribution thực, honest reporting tốt. R1 verified 12/13 numbers exact (1 pooled-σ math error đã fix).
- **MIWAI uy tín:** B-/C tier, Scopus+LNAI indexed, đủ cho LVTN + CV, không impressive cho PhD top-tier.
- **13 pages sticky** — không trim text thêm được mà không hy sinh content. LNCS cho 12+2 extra = 14 OK.
- **Single-seed RSNA** là weakness lớn nhất chưa giải quyết (acknowledged in Limitations + bootstrap CI committed cho camera-ready).

### Single source of truth numbers (paper)
- RSNA single-seed 3 conditions: Severe Recall 11.5%→46.4%, Severe F1 0.149→0.343, Mean F1 0.420→0.528
- SPIDER 3-seed: Hybrid F1 0.653±0.014 vs Baseline 0.646±0.002 (parity), CBAM 0.619±0.010 (degrades), pooled σ CBAM-Baseline = 0.007
- Zero-shot SPIDER: mean F1 0.362 / 8 labels

---

## Resources
- 5 reviewer personas (pass 2): Area Chair EIC, Methodology R1, Domain R2 (radiology), VLM R3, Devil's Advocate
- ARS skill: `academic-paper-reviewer` (full mode, 5 reviewers + synthesizer)
- LVTN handoff trước: `docs/handoff/handoff-20260519-2320.md`
