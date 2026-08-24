#!/bin/bash
# Build the report PDF (full pass: handles citations + ToC + cross-refs).
# Usage:  ./build.sh          (full build)
#         ./build.sh quick    (1 pdflatex pass only — text-only edits)
#         ./build.sh clean     (xoá file phụ trợ rồi build lại từ đầu — dùng khi lỗi "Extra }" / undefined refs)
set -e
cd "$(dirname "$0")"

if [ "$1" = "clean" ]; then
    echo "Cleaning aux files..."
    rm -f *.aux *.toc *.lof *.lot *.out *.bbl *.blg chapter/*.aux
fi

if [ "$1" = "quick" ]; then
    pdflatex -interaction=nonstopmode main.tex
else
    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
fi

echo ""
echo "==> Done: $(pwd)/main.pdf"
