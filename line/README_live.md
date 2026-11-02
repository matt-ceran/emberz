# Live line: run it

```bash
# 1. tunnel (prints the https URL) -> export it, point the Telnyx demo app's webhook at $PUBLIC_BASE_URL/webhooks/telnyx
line/tunnel.sh
export PUBLIC_BASE_URL=https://<name>.trycloudflare.com
export CONSOLE_TOKEN=<any secret>          # console + /actions fail closed without it
export GUARDIAN_BACKEND=matt               # or stub | standin | deployment
export GUARDIAN_MODEL='accounts/afceran33-zaqel7a5ap/models/emberz-guardian-v1#accounts/afceran33-zaqel7a5ap/deployments/emberz-v1'
export MATT_CONSOLE_URL=http://<matt-console-host>:<port>   # optional: forward console events
.venv/bin/python -m line.app               # uvicorn 127.0.0.1:8000
```

`line/app.py` builds the real app: TelnyxClient, CallFlow, per-call `LiveBrain` (ASR -> guardian -> policy
-> coach to the protected leg / caretaker alert / console), TimingScorer + w2v voice detector (only if
`voice/heads/w2v-head-v1.joblib` exists; `EMBERZ_VOICE_DETECTOR=0` turns it off), CallRecorder
(`recordings/<call_id>/`), CaretakerFlow (alert call, press 1 to listen, barge) and a summary at hang-up.
Profiles: `line/profiles.json` (gitignored; real numbers from .env; `EMBERZ_PROFILES` overrides).

## Matt bridge (`line/matt_bridge.py`)

- **Guardian `GUARDIAN_BACKEND=matt`**: uses Matt's `callguard/guardian/render.py` (`render_messages`,
  `parse_response`) if it is on this machine (`CALLGUARD_PATH`, repo root, or `../callguard*`), else our stand-in
  prompt with JSON-schema output. Model `GUARDIAN_MODEL` (unset: the stand-in model), key `GUARDIAN_API_KEY`
  or `FIREWORKS_API_KEY`. Timeouts/errors fall back to "no risk" (never stalls the call).
- **Console forwarder**: with `MATT_CONSOLE_URL` set, every console event is also POSTed to
  `$MATT_CONSOLE_URL/events` (background thread, 1.5 s timeout, dropped on failure; header
  `X-Console-Token: $MATT_CONSOLE_TOKEN` if set).
- **Buttons**: `POST $LINE_URL/actions/{call_id}/{listen|join|hangup|safe}?token=$CONSOLE_TOKEN`
  (or header `X-Console-Token`). listen = dial the caretaker now (press 1 -> monitor), join = barge
  (409 unless monitoring), hangup = end the call, safe = no more alerts/coach/caretaker dials for this call.
- Our own console stream is still at `WS /console/ws?token=...`, `GET /console/calls`.

## Simulate without a phone

```bash
.venv/bin/python -m line.simulate_call --scene all --text --fast          # stub guardian, scripted turns, ~2 s
.venv/bin/python -m line.simulate_call --scene bank                       # real audio: say -> mu-law -> ASR, real time
GUARDIAN_BACKEND=standin .venv/bin/python -m line.simulate_call --scene all --text
EMBERZ_AUDIO_TESTS=1 .venv/bin/python -m pytest -q tests/integration      # includes the real-audio scene
```
Scenes: bank (ends with "can you read me the code": alert + coach + caretaker monitor), grandpa (allowed
contact: silent), pharmacy (AI bot: flagged AI voice, no alert). Prints every console event and the ms from
the end of the trigger line's audio to the coach speak command.

Known: the ASR needs spare CPU. With the machine at load ~35 (other jobs), decoding fell ~60 s behind real
time. Run the live demo with other heavy jobs stopped.
