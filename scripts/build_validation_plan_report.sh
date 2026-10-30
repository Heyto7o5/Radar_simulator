#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

.venv/bin/python scripts/build_validation_plan_report_assets.py
google-chrome \
  --headless \
  --no-sandbox \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$project_dir/docs/RD_VALIDATION_EXTENSION_PLAN.pdf" \
  "file://$project_dir/docs/RD_VALIDATION_EXTENSION_PLAN.html"

pdfinfo docs/RD_VALIDATION_EXTENSION_PLAN.pdf | grep -E 'Pages|Page size|File size'
