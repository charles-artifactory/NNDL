#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PATH="/Library/TeX/texbin:$PATH"
command -v pandoc >/dev/null
command -v xelatex >/dev/null
mkdir -p tmp/pdfs
"${A2_PYTHON:-$project_dir/.venv/bin/python}" scripts/prepare_pdf_figures.py
# Match A1_submission: A4, 22 mm margins, 10 pt type and 1.15 line spacing.
# Generate and inspect the staged PDF before replacing the formal REPORT.pdf.
pandoc REPORT.md --from=markdown --standalone --toc --toc-depth=2 \
  -V toc-title=目录 \
  --pdf-engine=xelatex --resource-path=. \
  -V documentclass=article -V papersize=a4 -V fontsize=10pt \
  -V geometry:margin=22mm -V linestretch=1.15 \
  -V mainfont="${A2_MAIN_FONT:-Times New Roman}" \
  -V CJKmainfont="${A2_CJK_FONT:-Songti SC}" \
  -V CJKsansfont="${A2_CJK_FONT:-Songti SC}" \
  -V CJKmonofont="${A2_CJK_FONT:-Songti SC}" \
  -V monofont="${A2_MONO_FONT:-Menlo}" \
  --include-in-header=scripts/pdf-header.tex \
  --lua-filter=scripts/pdf-layout.lua \
  --output=tmp/pdfs/REPORT.pdf
printf 'PDF prepared at %s/tmp/pdfs/REPORT.pdf; inspect before replacing REPORT.pdf.\n' "$project_dir"
