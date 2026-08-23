# Paper draft

LaTeX source for the EMBC 2027 paper. Lives inside the codebase so figures and
tables can reference `experiments/` directly — single source of truth between
the empirical results and the manuscript.

## Build

```bash
cd paper
latexmk -pdf main.tex          # full build
latexmk -pdf -pvc main.tex     # watch mode (rebuild on save)
latexmk -C                     # clean build artifacts
```

Single-pass:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Layout

```
paper/
├── main.tex                  Main file, IEEEtran conference (placeholder)
├── sections/
│   ├── abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methods.tex
│   ├── 04_experiments.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
├── references.bib            BibTeX entries
├── .gitignore                LaTeX build artifacts
└── README.md                 This file
```

`main.tex` sets `\graphicspath{{../experiments/}{../experiments/paper_results/}{../experiments/paper_results/spider_zeroshot/}}`
so any PNG already in those folders can be `\includegraphics{name.png}` without
copying. Already-referenced: `hybrid_architecture.png`, `cross_phase_comparison.png`.

## Swapping in the official conference template

When the advisor sends the EMBC 2027 (or other venue) template:

1. Drop the `.cls`, `.bst`, and any sample `.tex` files into this folder.
2. Replace the first three lines of `main.tex` with the new `\documentclass{...}`
   and any required `\usepackage`s the template specifies.
3. Sections are modular (`\input{sections/...}`) — they should port over without
   edits unless the template requires different section macros.
4. `references.bib` is BibTeX-standard and works with most templates; if the
   template wants a different `\bibliographystyle{}`, change it in `main.tex`.

## Content sources (for filling in the TODOs)

- Motivation, related work, defense Q&A: [`../MOTIVATION_AND_DESIGN.md`](../MOTIVATION_AND_DESIGN.md) (EN), [`../MOTIVATION_AND_DESIGN_VI.md`](../MOTIVATION_AND_DESIGN_VI.md) (VI)
- Empirical numbers and tables: [`../experiments/PHASE_REPORT_FULL.md`](../experiments/PHASE_REPORT_FULL.md)
- Architecture diagrams: [`../experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md`](../experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md)
- RSNA detailed results: [`../experiments/fresh_cbam/RESULTS_SUMMARY.md`](../experiments/fresh_cbam/RESULTS_SUMMARY.md)
- SPIDER zero-shot detailed: [`../experiments/paper_results/spider_zeroshot/RESULTS_REPORT.md`](../experiments/paper_results/spider_zeroshot/RESULTS_REPORT.md)
