#!/usr/bin/env bash
# Rebuild the stage deck and speaker notes PDFs from src/ (needs Google Chrome on macOS).
set -euo pipefail
cd "$(dirname "$0")"
python3 -I src/build_deck.py src .
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for f in emberz_stage_deck emberz_speaker_notes; do
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=8000 \
    --print-to-pdf="$PWD/$f.pdf" "file://$PWD/$f.html" 2>/dev/null
done
