#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PATH="/Library/TeX/texbin:$PATH"
command -v pandoc >/dev/null
command -v xelatex >/dev/null
mkdir -p tmp/pdfs
# These defaults match the system used to export the submitted PDF.
# Override the fonts on other operating systems; training does not need TeX.
pandoc REPORT.md --from=markdown --standalone --toc --toc-depth=2 \
  -V toc-title=目录 \
  --pdf-engine=xelatex --resource-path=. \
  -V documentclass=article -V papersize=a4 -V fontsize=10pt \
  -V geometry:margin=22mm -V linestretch=1.15 \
  -V mainfont="${A1_MAIN_FONT:-Times New Roman}" \
  -V CJKmainfont="${A1_CJK_FONT:-Songti SC}" \
  -V CJKsansfont="${A1_CJK_FONT:-Songti SC}" \
  -V CJKmonofont="${A1_CJK_FONT:-Songti SC}" \
  -V monofont="${A1_MONO_FONT:-Menlo}" \
  --include-in-header=scripts/pdf-header.tex \
  --output=tmp/pdfs/REPORT.pdf
printf 'PDF prepared at %s/tmp/pdfs/REPORT.pdf; inspect before replacing REPORT.pdf.\n' "$project_dir"
