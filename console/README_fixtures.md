# Console backend: endpoints and events (for C1)

The console page talks to one backend. During development that backend is the **fixture player**, which replays recorded demo scenes. On live calls it's the line server. Both use the same router (`line/console_hub.py`), so a page that works against the fixtures works live.

## Run the fixture player

```bash
cd fireworks-AMD
.venv/bin/python -m console.fixture_player              # plays all 4 scenes once, in order, at real speed
.venv/bin/python -m console.fixture_player --speed 4    # 4x faster
.venv/bin/python -m console.fixture_player --loop       # forever
.venv/bin/python -m console.fixture_player --no-autoplay
```

It listens on `http://127.0.0.1:8001`. CORS is open (`*`), so a page served from another port (for example Vite on 5173) can call it. To restrict origins, set `CONSOLE_ALLOWED_ORIGINS=http://localhost:5173`.

## Endpoints

| | Path | What it does |
|---|---|---|
| WS | `/console/ws` | Every call. On connect you get the **replay** (all buffered events, in order), then live events. |
| WS | `/console/ws?call_id=X` | The same, for one call only. |
| GET | `/console/calls` | One summary row per call, newest first (see below). |
| GET | `/console/calls/{call_id}/events` | The full event list of one call (404 if unknown). |
| POST | `/console/calls/{call_id}/barge` | **The join button.** `200 {"call_id", "caretaker_state": "barged"}`. `409` if the caretaker isn't listening yet (or the call ended), `404` for an unknown call. A second press is a harmless 200. On success a `caretaker_joined {mode: "barge"}` event arrives on the WebSocket. |
| GET | `/fixtures` | Fixture player only: the scenes, with event counts. |
| POST | `/fixtures/play/{scene}?speed=2` | Fixture player only: replays a scene now and returns `{"call_id"}`. Each replay gets a fresh id (`fx_scene3_stranger_kid-2`, `-3` and so on). |

Each WebSocket message is one JSON `ConsoleEvent` (`contracts/models.py`). The page never needs to send anything.

**Token (live only).** The line server always needs `CONSOLE_TOKEN`: add `?token=...` to every URL (WS and HTTP) or send an `X-Console-Token` header on HTTP. A wrong or missing token gets 401 (WebSocket: close code 1008); if the server has no `CONSOLE_TOKEN` set, the console answers 503. Make the page read the token from its own URL (e.g. `?token=`) so the link Mom opens carries it. The fixture player needs no token.

**Reconnects.** On reconnect you get the whole replay again, so reset that call's view when its `call_started` arrives, or de-duplicate on `(call_id, type, at_ms)`.

## Event shape

```json
{"type": "turn", "call_id": "fx_scene1_bank_scam", "at_ms": 7450, "data": { ... }}
```

`at_ms` is milliseconds since the call started. The `data` shape depends on `type` (contracts/README.md):

| type | data |
|---|---|
| `call_started` | `{caller_number, guarded_number, profile: elder\|kid\|adult, caller_known: bool}` |
| `turn` | a `Turn`: `{call_id, turn, speaker: caller\|protected, text, t_start_ms, t_end_ms, gap_ms}`. After the caretaker barges in, their voice comes through on the `protected` side. |
| `voice` | a `VoiceSignal`: `{call_id, at_ms, ai_score: 0..1, model}`. This drives the **AI voice** badge (scenes use 0.91 and 0.94 for AI, below 0.1 for humans). It can come more than once, and the latest one wins. |
| `risk` | `{turn, response: GuardianResponse, latency_ms, display_risk, suggest_hangup, reasons}`. `response` is the raw model output: `risk` `none\|watch\|alert\|urgent`, `category`, `severity` 0-3, `stage`, `evidence[{turn, quote}]`, `action`, `coach_line`, `caretaker_note`. **Draw the risk line from `display_risk`** (the rule layer's verdict after smoothing, allowed contacts and the AI-voice rules); fall back to `response.risk` if it is missing. `suggest_hangup` (bool) and `reasons` (short rule codes) are optional extras. |
| `coach` | `{text, turn, sent}`: what the protected person heard (the caller didn't). `sent` is `false` if the line couldn't play it. |
| `alert` | `{level: alert\|urgent, category, caretaker_note, turn}` |
| `caretaker_joined` | `{mode: monitor\|barge}`. `monitor` means the caretaker pressed 1 and is listening. `barge` means they joined (button or fixture). |
| `summary` | `{text}`: 3 sentences, arrives about 1-2 s **after** `call_ended`. Every call with speech gets one, quiet calls included. It can be missing if the summary model failed. |
| `call_ended` | `{duration_ms, reason}` |

`/console/calls` row: `{call_id, n_events, last_at_ms, started (call_started data), max_risk (highest display_risk), alerts, ai_score, caretaker (null|monitor|barge), summary (text|null), ended (call_ended data|null)}`.

The join button can also get `502` if the phone system rejected the request; the caretaker stays in listen mode and can press again.

## Scenes (`console/fixtures/*.jsonl`, one ConsoleEvent per line)

| File | Story | What the page should show |
|---|---|---|
| `scene1_bank_scam.jsonl` (60 s) | AI voice posing as Grandma's bank asks her to read a one-time code | AI badge at ~5 s; risk none, watch, watch, **urgent**; coach and urgent alert at turn 7; caretaker listens (45 s), then barges in (52 s); call ends; summary |
| `scene3_stranger_kid.jsonl` (30 s) | Stranger asks Mia whether her parents are home and what street she lives on | Human voice; an alert, then an urgent alert with coach lines; caretaker monitoring from 21 s, so **the join button works here** until 29 s (or the whole time at `--speed` below 1) |
| `scene4_grandpa_allowed.jsonl` (19 s) | Grandpa (`caller_known: true`, on the allowed list) asks the same questions | Risk stays `none`. No coach, no alert. A calm summary. |
| `scene5_pharmacy_bot.jsonl` (22 s) | Pharmacy reminder bot | AI badge (0.94) but risk only `watch`. No alert, no coach. A calm summary. |

Scene 2 of the demo ("Mom joins") is the end of scene 1. Scene 6 is the closing overlay, so it has no event stream.

Every line validates as `contracts.models.ConsoleEvent`, and every `data` payload is checked per type. Run `.venv/bin/python -m pytest tests/caretaker` to check this.
