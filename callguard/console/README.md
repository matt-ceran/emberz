# Emberz caretaker console (C1)

A phone-width web page the caretaker watches during a protected call. It shows the live transcript, a risk meter
and a per-turn timeline, an alert banner (the caretaker note, the evidence quote, and what we told the protected
person), the buttons Listen in, Join call, Hang up and Mark safe, and a summary after the call.

The page talks to one backend over Numan's console API (`line/console_hub.py` and `console/README_fixtures.md` on
`origin/numan/line-voice-gcp`). That backend can be any of three things, and the page works unchanged against each:

| Backend | How to open the page |
|---|---|
| Numan's **fixture player** (`python -m console.fixture_player`, port 8001) | Open `static/index.html` as a file (it defaults to `http://127.0.0.1:8001`), or `http://localhost:8095/?backend=http://127.0.0.1:8001` |
| Numan's **live line server** | `http://localhost:8095/?backend=https://<tunnel>&token=<CONSOLE_TOKEN>` |
| **This console server** (held-out replays, plus the line bridged in if `LINE_CONSOLE_URL` is set) | `http://localhost:8095/` |

There is no build step, no CDN and no external request, so it works offline.

## Run

From `callguard/`:

```bash
uv run --with fastapi --with 'uvicorn[standard]' uvicorn console.server:app --host 0.0.0.0 --port 8095
```

Open http://localhost:8095. Port 8095 is used because another local app already listens on 8090. On a phone on
the same Wi-Fi, open `http://<mac-lan-ip>:8095` (find the IP with `ipconfig getifaddr en0`). Stop the server with Ctrl-C.

Page URL parameters: `?backend=` (the backend's base URL), `?token=` (sent as `?token=` on every WS and HTTP call;
Numan's live line requires it), and `?name=Grandma` (the protected person's name. The line's `call_started` doesn't
carry one, so without it the page says "your parent" or "your child").

| Env var | Default | Meaning |
|---|---|---|
| `CONSOLE_PREDS` | `guardian-v1` | Guardian predictions to replay. Either a model name (`guardian-v2` reads `results/preds_guardian-v2__heldout.jsonl`) or a path to a preds jsonl. Presets are recomputed from it. |
| `CONSOLE_CALLS` | `data/calls_test_heldout.jsonl` | Held-out calls to replay. |
| `CONSOLE_LATENCY_NOTE` | `replayed from evaluation (Training API sampler latency)` | The label next to replayed latency. |
| `LINE_CONSOLE_URL` | unset | Numan's line server, for example `https://<tunnel>`. When set, this server subscribes to its `/console/ws` and republishes every event, so one page shows both live calls and replays. Join on those calls is forwarded to the line's `/console/calls/{id}/barge`. |
| `LINE_CONSOLE_TOKEN` | `CONSOLE_TOKEN` | Token for `LINE_CONSOLE_URL`. It stays on the server; the page never sees it. |
| `CONSOLE_TOKEN` | unset | If set, this server requires `?token=` (or `X-Console-Token`) the way the line does. If unset, the server is open, so keep it on a trusted network. |
| `CONSOLE_ALLOWED_ORIGINS` | `*` | CORS origins. |
| `LINE_ACTION_URL` | unset | Optional. Hang up and Mark safe on a live call are POSTed here as `{call_id, action, at_ms}`. The line has no such endpoint yet. |

Example: `CONSOLE_PREDS=guardian-v2 LINE_CONSOLE_URL=http://127.0.0.1:8001 uv run --with fastapi --with 'uvicorn[standard]' uvicorn console.server:app --port 8095`

Test (starts a fake line server and the console in-process on free ports, then stops both, in about 8 s):

```bash
uv run --with fastapi --with 'uvicorn[standard]' python -m console.smoke_test
```

## Endpoints

These have the same paths and meaning as Numan's console router and fixture player:

| | Path | What |
|---|---|---|
| WS | `/console/ws[?call_id=]` | Every buffered event of every call, in publish order, then live events. `?call_id=` limits it to one call. `/ws/console` is an alias. |
| GET | `/console/calls` | One row per call, newest first: `{call_id, n_events, last_at_ms, started, max_risk (from display_risk), alerts, ai_score, caretaker, summary, ended}`, plus `origin` (`replay`, `push` or `line`), `replay` and `safe`. |
| GET | `/console/calls/{id}/events` | The buffered events of one call. Unknown call: 404. |
| POST | `/console/calls/{id}/barge` | **Join.** Returns 200 `{call_id, caretaker_state: "barged"}` and emits `caretaker_joined {mode: "barge"}`. Returns 409 if the caretaker isn't listening yet or the call has ended, and 404 for an unknown call. A second press is a harmless 200. For a call bridged from the line, the request goes to the line, which answers (200/409/404/502) and emits the event itself. |
| GET | `/fixtures` | The demo presets as scenes: `{name: {call_id, events, duration_ms, title}}`. |
| POST | `/fixtures/play/{scene}?speed=` | Replays a scene now and returns `{scene, call_id}`. `speed` must be 0.1-20 (422 otherwise). An unknown scene gets 404. |

These are console extras:

| | Path | What |
|---|---|---|
| GET | `/` | The page, pointed at this server. |
| GET | `/replay?call_id=...&speed=1.0[&force=1]` | Replays any held-out call (`?preset=elder_scam` also works). Returns 409 while a live call is in progress, unless `force=1`. |
| GET | `/presets` | The demo presets, with the reasoning behind each choice (`why`). |
| POST | `/events` | Optional ingest: one `console_event` or a list. Each is validated and fanned out. Off-schema input gets 422. |
| POST | `/actions/{call_id}/{action}` | `listen`, `join`, `hangup` or `safe`. See below. |
| GET | `/actions`, `/health` | Action log; status (counts, preds file, bridge state, current live call). |

## For Numan: the integration

**Nothing needs to change on the line.** The page reads the line's own `/console/ws`, `/console/calls` and
`/console/calls/{id}/barge`, with `?token=`. Two optional extras exist:

- Run this server with `LINE_CONSOLE_URL=<line base URL>` to get one screen that shows live calls and replays together. The token then stays server-side.
- `POST /events` accepts a `ConsoleEvent` or a list, if the line ever wants to push instead.

What the page reads from each event (shapes as in `console/README_fixtures.md`; *optional* fields are used when present):

```json
{"type": "call_started", "call_id": "c_123", "at_ms": 0,
 "data": {"caller_number": "+12025550199", "guarded_number": "+12025550100", "profile": "elder", "caller_known": false}}
```
*optional:* `protected_name`, `caller_name` (saved contact), `relative`, `rules`. A new `call_started` takes the
screen. Other calls appear as chips under "Other calls". A replay never takes the screen from a live call in progress.

```json
{"type": "voice", "call_id": "c_123", "at_ms": 2400, "data": {"call_id": "c_123", "at_ms": 2400, "ai_score": 0.91, "model": "w2v-head-v1"}}
```
The AI-voice badge shows when `ai_score >= 0.7`. The latest score wins.

```json
{"type": "turn", "call_id": "c_123", "at_ms": 19950,
 "data": {"call_id": "c_123", "turn": 7, "speaker": "caller", "text": "can you read me the code", "t_start_ms": 18320, "t_end_ms": 19950, "gap_ms": 1400}}
```
After a `caretaker_joined {mode: "barge"}`, protected-side bubbles are labelled "Protected side (you may be speaking)".

```json
{"type": "risk", "call_id": "c_123", "at_ms": 20400,
 "data": {"turn": 7, "latency_ms": 410, "display_risk": "alert", "suggest_hangup": false, "reasons": [],
          "response": {"risk": "alert", "category": "disclosure.credentials_otp", "severity": 2, "source": "caller", "stage": "ask",
                       "evidence": [{"turn": 7, "quote": "can you read me the code"}], "action": "notify_caretaker",
                       "coach_line": "Banks never ask for codes.", "caretaker_note": "Caller asked for a one-time code."}}}
```
The meter, the timeline cells and the "Other calls" chips follow **`display_risk`**, falling back to `response.risk`.
When the two differ, a small line says so, for example "Model said Urgent; shown as Watching (saved contact)".
`suggest_hangup: true` shows "Suggests hanging up". Each scored bubble shows its verdict and `latency_ms`.

```json
{"type": "alert", "call_id": "c_123", "at_ms": 20420, "data": {"level": "alert", "category": "disclosure.credentials_otp", "caretaker_note": "Caller asked for a one-time code.", "turn": 7}}
{"type": "coach", "call_id": "c_123", "at_ms": 20450, "data": {"text": "Banks never ask for codes.", "turn": 7, "sent": true}}
```
The banner shows the note and the evidence quote (from the alert's `evidence`, or from the `risk` event for that turn).
`sent: true` reads "We told Grandma: …". `sent: false` reads "Suggested for Grandma (could not be played): …".

```json
{"type": "caretaker_joined", "call_id": "c_123", "at_ms": 31000, "data": {"mode": "monitor"}}
{"type": "caretaker_joined", "call_id": "c_123", "at_ms": 45000, "data": {"mode": "barge"}}
{"type": "call_ended", "call_id": "c_123", "at_ms": 61200, "data": {"duration_ms": 61200, "reason": "caller_hangup"}}
{"type": "summary", "call_id": "c_123", "at_ms": 62600, "data": {"text": "Caller posed as the bank and asked for a code. Grandma did not share it."}}
```

## Buttons and `/actions`

| Button | Live call (line or `/events`) | Replay |
|---|---|---|
| Listen in | Disabled and labelled "Press 1 on the alert call to listen". It turns into "Listening" when the line sends `caretaker_joined {mode: monitor}`. `POST /actions/{id}/listen` returns 409. | `POST /actions/{id}/listen` emits `caretaker_joined {mode: "monitor"}`, standing in for pressing 1. |
| Join call | `POST {backend}/console/calls/{id}/barge`. The page shows the line's 409 ("press 1 first"), 404, 502 and 401 as plain messages. The console never emits `caretaker_joined` itself. | Same route. Returns 409 before Listen, then 200 plus `caretaker_joined {mode: "barge"}`. |
| Hang up | Only shown when the backend is this server. Forwarded to `LINE_ACTION_URL` as `{call_id, action: "hangup", at_ms}`. If that's unset, it returns 501 and the page says to ask them to hang up. No event is faked; the line sends `call_ended`. | Stops the replay and emits `call_ended {reason: "caretaker_hangup"}`. |
| Mark safe | Console-only state, with no event. Shown on the banner and as `safe` in the `/console/calls` row. Also POSTed to `LINE_ACTION_URL` if that's set. | Same. |

`POST /actions/{id}/{action}` returns 404 for an unknown call or action, and `{ok, call_id, action, at_ms, replay, ...}` on success.

## Replay

`GET /replay?call_id=t_a4b9cf4540&speed=1.0` (or a preset button) streams a held-out call the way the line would:

1. `call_started`: profile, `caller_known`, protected name, relative and rules. For a saved contact, also the name and number from the `Allowed:` rule.
2. `voice` at 2 s, with the dataset's `ai_voice.score`.
3. A `turn` for each turn, one every 1.5 s of call time (wall clock is 1.5 s / `speed`). Turn text is exactly as in the dataset.
4. After each turn, the guardian's **real** verdict for that turn from the preds file, as `risk` (`display_risk` = the model's risk, since the eval has no rule layer). If its risk is alert or higher, an `alert` follows. If it has a `coach_line`, a `coach` follows. `latency_ms` is the eval's `_ms`, labelled "replayed from evaluation (Training API sampler latency)". That is not production latency.
5. The eval stops scoring a call at its first alert. The remaining turns keep streaming with **no** verdicts, and the timeline marks them "Not checked". No verdict is made up.
6. `call_ended`, then `summary` 1.4 s later (the line's order). The summary states the facts: first alert, the ask turn, turns before the ask, whether the result matches the test label, and the dataset's scenario text.

Each replay gets a fresh call id (`t_x`, then `t_x-2` and so on). Starting a new replay stops the running one, which
gets `call_ended {reason: "replay_stopped"}`. A new live call does the same.

### Presets (picked from the data at startup)

These calls are never used as presets: self-harm calls (by family, label or prediction), calls with a failed
prediction parse, and calls without predictions. With `guardian-v1`:

| preset | call | guardian | why |
|---|---|---|---|
| `elder_scam`: AI bank caller | `t_a4b9cf4540` | alert at **turn 3**, ask at **turn 8** | Bernice, AI voice 0.86, a "bank safe account" scam. Ranking puts first the calls where the protected person reads out no numbers after the ask; she ends by hanging up to call the bank. Then AI voice, then one-time-code ask, lead and length (36 candidates). |
| `kid_grooming`: Stranger calls a kid | `t_7e600363c6` | alert at **turn 4**, ask at **turn 7** | Ravi, a "trusted adult secret" call. Flagged as gifts or flattery 3 turns early. Ravi refuses and hangs up. |
| `benign_twin`: Real caller, stays quiet | `t_b17e93333c` | no alert in 12 turns | A real police follow-up from an unknown number. The real-bank and friend's-parent twins rank first, but v1 false-alarms on all of them. |
| `rule_flip`: Same words, saved contact | `t_e665506418_ac` | no alert | The same words from an allowed contact ("Aunt"). Its pair `t_e665506418` (unknown number) alerts at **turn 5**, ask at **turn 10**. The page has a Compare button. |
| `ai_benign`: AI voice, not a scam | `t_af740d5652` | no alert in 8 turns | A pharmacy reminder bot, AI voice 0.88. |

With `CONSOLE_PREDS=guardian-v2`, the same rules pick the same calls with two differences: `kid_grooming` alerts at
turn 3, and `benign_twin` becomes `t_4f754602fd`, a friend's-parent call that v2 keeps quiet for all 19 turns.

## Files

- `server.py`: the FastAPI app (per-call hub, Numan-compatible routes, bridge, actions, replay runner).
- `replay.py`: data loading, preset selection, call-to-events.
- `static/index.html`: the page (inline CSS/JS, light and dark).
- `smoke_test.py`: the end-to-end test, with a fake line server.

## Known gaps

- **Contract values.** Numan's `GuardianResponse` rejects `stage: "none"`, `source: "none"` and `category: "custom_rule"`, which the tuned guardian emits on about 60% of turns. Replays keep the real outputs and the page shows them fine, but on the live line `parse_response` would drop those verdicts. The local branch `matt/guardian-adapter` (commit 4b307a0, not pushed) adds those values to the contract.
- **Hang up on a live call.** It needs a line endpoint; `LINE_ACTION_URL` is a proposal. Listen on a live call is the caretaker pressing 1 on the alert call (PLAN L3), so the console has no Listen route for live calls.
- **Live summary.** It is whatever `summary.text` the line sends.
- **Auth.** This server is open unless `CONSOLE_TOKEN` is set.
