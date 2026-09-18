#!/usr/bin/env bash
set -euo pipefail
paper_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
/usr/bin/google-chrome --headless --no-sandbox --disable-gpu --disable-background-networking --no-pdf-header-footer --user-data-dir=/tmp/c3-paper-v2-chrome --print-to-pdf="$paper_dir/MANUSCRIPT_KO.pdf" "file://$paper_dir/MANUSCRIPT_KO.html"
