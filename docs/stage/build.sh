#!/usr/bin/env bash
# Rebuild the stage decks and speaker notes PDFs (needs Google Chrome on macOS).
#   short (main):  src_short/ -> emberz_stage_deck.pdf, emberz_speaker_notes.pdf
#   full (backup): src/       -> emberz_stage_deck_full.pdf, emberz_speaker_notes_full.pdf
set -euo pipefail
cd "$(dirname "$0")"
python3 -I src/build_deck.py src_short . emberz_stage_deck emberz_speaker_notes
python3 -I src/build_deck.py src . emberz_stage_deck_full emberz_speaker_notes_full
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for f in emberz_stage_deck emberz_speaker_notes emberz_stage_deck_full emberz_speaker_notes_full; do
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=8000 \
    --print-to-pdf="$PWD/$f.pdf" "file://$PWD/$f.html" 2>/dev/null
done
