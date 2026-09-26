#!/usr/bin/env bash
# Build the distributable release zip (source only — weights stay external).
# Researchers: unzip, pip install -r requirements.txt, python cli.py demo
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../ai-model-fusion-lab-release.zip}"
rm -f "$OUT"
zip -r "$OUT" . \
  -x "*.pyc" -x "__pycache__/*" -x "*/__pycache__/*" \
  -x ".git/*" -x "*.npz" -x "*.safetensors" -x "*.bin"
echo "---- contents ----"
unzip -l "$OUT" | tail -5
echo "Release zip: $OUT ($(du -h "$OUT" | cut -f1))"
