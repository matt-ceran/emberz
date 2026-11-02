#!/usr/bin/env bash
# Emberz video demo: preflight, pause heavy background jobs, place one AI test call, narrate it live.
#
#   bash demo/run_demo.sh [bank_fraud|tech_support|prize_giftcard|pharmacy_refill|appointment_confirm] [index]
#
# bank_fraud (default) = the scam demo; appointment_confirm / pharmacy_refill = benign controls (should stay quiet).
# Needs: the line server (line.app) up on EMBERZ_PORT (default 8010) with its tunnel, and .env.
# Secrets stay in .env; nothing here prints them.
set -uo pipefail
cd "$(dirname "$0")/.."
PERSONA="${1:-bank_fraud}"
IDX="${2:-0}"
PORT="${EMBERZ_PORT:-8010}"
LOG="${EMBERZ_LINE_LOG:-${TMPDIR:-/tmp}/emberz_line_live.log}"
set -a; . ./.env; set +a

ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0

echo "== Preflight"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/console/calls?token=$CONSOLE_TOKEN")
[ "$code" = 200 ] && ok "line server on :$PORT" || bad "line server on :$PORT (HTTP $code)"
code=$(curl -s -o /dev/null -w '%{http_code}' "$PUBLIC_BASE_URL/console/calls?token=$CONSOLE_TOKEN")
[ "$code" = 200 ] && ok "public tunnel $PUBLIC_BASE_URL" || bad "public tunnel (HTTP $code)"
if [ -f "$LOG" ]; then
  g=$(grep -o "guardian backend [a-z]*: model=[^ ]*" "$LOG" | tail -1)
  [ -n "$g" ] && ok "${g%%#*}" || echo "  ??   guardian backend not found in $LOG"
fi
[ "$FAILED" = 0 ] || { echo "Preflight failed; not dialing."; exit 1; }
[ "${DEMO_DRY:-0}" = 1 ] && { echo "DEMO_DRY=1: preflight only, not dialing."; exit 0; }

# Pause CPU-heavy background jobs (batch ASR) so live transcription stays real time; resume on exit.
PAUSED=$(pgrep -f "realset\.transcribe|voice\.pipeline|noise_pass" | tr '\n' ' ')
resume() { [ -n "$PAUSED" ] && kill -CONT $PAUSED 2>/dev/null && echo "== Resumed background jobs: $PAUSED"; }
trap resume EXIT
[ -n "$PAUSED" ] && kill -STOP $PAUSED && echo "== Paused background jobs: $PAUSED"

echo "== Dialing: AI caller persona '$PERSONA' #$IDX -> guarded line ${TELNYX_GUARDED_NUMBER:-}"
start=$(( $(wc -l < "$LOG" 2>/dev/null || echo 0) + 1 ))
printf '\n\n' | .venv/bin/python -m callers.signalwire.place_calls --personas "$PERSONA" --calls 1 \
  --start "$IDX" --live --max-calls 1 2>&1 | tail -1 | sed 's/^.*: HTTP/  HTTP/'

echo "== Live (Ctrl-C stops watching; the call continues)"
tail -n +"$start" -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  case "$line" in
    *"conference.participant.joined"*) echo "  $(date +%T)  leg joined the call" ;;
    *"deepfake result"*)               echo "  $(date +%T)  Telnyx AI-voice: ${line##*: }" ;;
    *"caretaker alert"*)               echo "  $(date +%T)  CARETAKER: ${line##*: }" ;;
    *"coach on turn"*)                 echo "  $(date +%T)  COACH whispered (${line##*call c_*: })" ;;
    *"media closed"*)                  echo "  $(date +%T)  call ended"; break ;;
    *Traceback*|*" ERROR "*)           echo "  $(date +%T)  !! $line" ;;
  esac
done

rec=$(ls -td recordings/c_* 2>/dev/null | head -1)
[ -n "$rec" ] && .venv/bin/python - "$rec" <<'EOF'
import json, sys
d = sys.argv[1]
print(f"== Transcript ({d})")
for l in open(f"{d}/turns.jsonl"):
    t = json.loads(l); print(f"  {t['turn']:>2} {'CALLER' if t['speaker']=='caller' else 'GRANDMA':8} {t['text'][:110]}")
print("== Events")
for l in open(f"{d}/events.jsonl"):
    e = json.loads(l); x = e["data"]
    if e["type"] == "alert":   print(f"  {e['at_ms']/1000:6.1f}s ALERT {x.get('level')} {x.get('category')}: {x.get('caretaker_note')}")
    elif e["type"] == "coach": print(f"  {e['at_ms']/1000:6.1f}s COACH \"{x.get('text')}\"")
    elif e["type"] == "voice": print(f"  {e['at_ms']/1000:6.1f}s AI-voice {x.get('model')} {x.get('ai_score')}")
    elif e["type"] == "caretaker_joined": print(f"  {e['at_ms']/1000:6.1f}s caretaker joined ({x.get('mode')})")
EOF
