#!/usr/bin/env bash
# Start a cloudflared quick tunnel to the line server (localhost:8000) and print its URL.
# Quick tunnels need no Cloudflare account; the URL changes on every start, so re-point the
# demo app afterwards:  .venv/bin/python -m line.repoint_demo_app --url <URL>   (dry run; add --apply)
# Docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/
set -euo pipefail

PORT="${1:-8000}"
LOG="${TMPDIR:-/tmp}/emberz-tunnel.log"
command -v cloudflared >/dev/null || { echo "cloudflared is not installed" >&2; exit 1; }

: >"$LOG"
cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}" >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' INT TERM EXIT

URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -n1 || true)"
  [ -n "$URL" ] && break
  kill -0 "$PID" 2>/dev/null || { echo "cloudflared exited; see $LOG" >&2; exit 1; }
  sleep 0.5
done
[ -n "$URL" ] || { echo "no tunnel URL after 30 s; see $LOG" >&2; exit 1; }

echo "PUBLIC_BASE_URL=$URL"
echo "Webhook URL:  $URL/webhooks/telnyx"
echo "Media WS:     ${URL/https:/wss:}/media/<call_id>"
echo "Next: export PUBLIC_BASE_URL=$URL and run: .venv/bin/python -m line.repoint_demo_app --url $URL"
echo "(tunnel running as pid $PID, log $LOG; Ctrl-C stops it)"
wait "$PID"
