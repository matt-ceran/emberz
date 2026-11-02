# callers/elevenlabs: ElevenLabs test callers (benign bots only)

Two benign bots, `pharmacy_refill` and `appointment_confirm`, run as ElevenLabs Agents and call the
guarded Telnyx number from the idle SignalWire number. They add a second commercial TTS family to the
live AI-voice test set (PLAN.md V2). **No scam persona ever goes to ElevenLabs**: its use policy bans
fraud and impersonation with no stated exception for testing (PLAN.md 2c). `agents.check_persona`
enforces an allowlist, every payload builder calls it, and the CLI exits with code 2 on anything else.
ElevenLabs audio is test data only, never training data (PLAN.md V1; the use policy's section 12 also
restricts using Output to train AI models); ledger rows carry `exclude_from_training: true`.

| File | What |
|---|---|
| `agents.py` | Allowlist, 3 default voices per bot, the agent config. Prompts and the 10 variants come read-only from `callers/agents/<persona>.json` |
| `place_calls.py` | `check` (read-only), `call` (dry run by default, `--live` to send) |
| `tests/callers_el/` | Allowlist, dry-run payloads, the live flow against fake clients |

Voices (ElevenLabs default voices; variant `v` uses voice `v % 3`):

| Bot | v0 | v1 | v2 |
|---|---|---|---|
| pharmacy_refill | Sarah (F, US) | Bill (M, US, older) | Alice (F, UK) |
| appointment_confirm | Matilda (F, US) | Chris (M, US) | Daniel (M, UK) |

TTS `eleven_flash_v2_5`, LLM `gpt-4o-mini`. 2 bots x 10 variants = 20 calls, 6 agents.

## How a call is routed (and why)

```
SignalWire REST: dial +1 304 648 1098 -> +1 323 516 1282 (guarded, Telnyx), inline SWML
   guarded line answers, plays the notice, rings the protected phone
   SWML `connect` -> sip:+13046481098@sip.rtc.elevenlabs.io:5061;transport=tls
                     digest auth + X- headers carrying the variant's values
ElevenLabs: imported "SIP trunk" number (inbound only) -> the assigned agent answers
   X-Callee-Name: Margaret  becomes  {{sip_callee_name}} in the prompt
```

ElevenLabs offers these phone routes
([overview](https://elevenlabs.io/docs/agents-platform/overview),
[docs index](https://elevenlabs.io/docs/llms.txt)): native Twilio, SIP trunk (provider guides for
Telnyx, Plivo, Bandwidth, Sinch, Vobiz and Exotel; Vonage via a WebSocket connector), and batch calls
on top of those. The Agents phone-number API only imports numbers (Twilio or SIP trunk: import, list,
get, update, delete); there is no endpoint to buy one. (ElevenLabs' separate Reception.ai product
provisions numbers, but it is a receptionist product, not Agents.) We have no Twilio, so the options were:

| Route | What it needs | Verdict |
|---|---|---|
| **A. SignalWire bridges to ElevenLabs** (chosen) | Import one SIP-trunk "number" in ElevenLabs (inbound only, just an identifier). On SignalWire, nothing: each call is one REST request with its SWML inline | **Touches no SignalWire resource.** Uses the caller number only as caller ID, as PLAN.md allows |
| B. ElevenLabs dials out over a SIP trunk to SignalWire ([outbound-call API](https://elevenlabs.io/docs/api-reference/sip-trunk/outbound-call)) | A new SignalWire **SIP credential** (`call_handler: passthrough`, `send_as` the caller number) in EmberDial's SignalWire project, next to its existing `ember-media1` endpoint | Creates a new resource on EmberDial's account and has an unknown (does SignalWire accept a digest user that differs from the From user?). Not built |
| C. ElevenLabs over a Telnyx SIP trunk ([guide](https://elevenlabs.io/docs/eleven-agents/phone-numbers/telephony/telnyx.md)) | A new Telnyx credential connection plus a second Telnyx number as caller ID | New resources on EmberDial's Telnyx account and $1+/month, and the caller would no longer be the SignalWire number PLAN.md picked. Not built |

Route A in the docs:
- SignalWire dial with inline SWML: `POST https://<space>/api/calling/calls`, `{"command":"dial","params":{from,to,caller_id,timeout,max_price_per_minute,swml}}`
  ([call commands](https://signalwire.com/docs/apis/rest/calls/call-commands)).
- SWML `connect` to a SIP URI with `username`, `password`, `headers: [{name, value}]` (names must start with `X-`), `codecs`, `max_duration`
  ([connect](https://signalwire.com/docs/swml/methods/connect)).
- ElevenLabs SIP endpoint `sip.rtc.elevenlabs.io` (TLS 5061, TCP 5060). The user part before the `@` must match the imported number exactly, `+` included. Inbound `X-Foo-Bar` headers become `{{sip_foo_bar}}`
  ([SIP trunking](https://elevenlabs.io/docs/agents-platform/phone-numbers/sip-trunking)).
- Import a number: `POST /v1/convai/phone-numbers`, `provider: sip_trunk`, `inbound_trunk_config.credentials`
  ([create](https://elevenlabs.io/docs/api-reference/phone-numbers/create)). Reassign the agent: `PATCH /v1/convai/phone-numbers/{id}` `{agent_id}`
  ([update](https://elevenlabs.io/docs/api-reference/phone-numbers/update)).
- Agents: `POST /v1/convai/agents/create`, `PATCH /v1/convai/agents/{id}`, `GET /v1/convai/agents?search=`
  ([create](https://elevenlabs.io/docs/api-reference/agents/create), [list](https://elevenlabs.io/docs/api-reference/agents/list)).
- Call end: `GET /api/voice/logs/{id}` on SignalWire ([voice log](https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log)). The conversation id: `GET /v1/convai/conversations?agent_id=` ([list](https://elevenlabs.io/docs/api-reference/conversations/list)).

The agents don't speak first (`first_message: ""`), because the line plays the Emberz notice before
the protected phone rings. The opening line is in the prompt, after "wait for a person to say
hello" (the same idea as `callers/common.py`). `initial_wait_time` is 60 s. The `skip_turn` tool
([docs](https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/skip-turn.md):
"After this tool is called, the assistant will not speak") lets the agent stay silent when it hears
the notice instead of answering a recording. The `end_call` tool hangs up after the goodbye, and
`max_duration_seconds` (the spec's 150 s + 30 s) caps every call. That cap counts from when the
guarded line answers, so the notice and the ringing come out of it: a pickup after ~30 s leaves
~150 s of conversation.

## What it creates and what it never touches

- **ElevenLabs (on `--live`, once):** up to 6 agents named `Emberz test caller: <bot> v<i> (<voice>)`
  and tagged `emberz-test-caller`, plus one imported number labelled
  `Emberz EL bridge (inbound SIP from SignalWire, test only)`. Their ids go to
  `data/callers/elevenlabs.json` (gitignored). Before each call, the bridge number is reassigned to
  that call's agent. The script only PATCHes agents with our name prefix and tag, and only the
  number with our label. It refuses to run if the caller number is already imported under another
  label. The 4 existing `EmberDial LOCAL practice` agents are never touched.
- **SignalWire:** creates nothing and modifies nothing. It only reads the caller number's inbound
  routing, in `check` (today: `laml_webhooks`, EmberDial's webhook, left as is).
- **Telnyx:** nothing here. The guarded number receives a normal inbound call.

## One-time setup (Numan, before the first live call)

1. Add a fresh SIP password to the repo `.env`. It is used only between SignalWire and ElevenLabs:
   ```sh
   echo "EMBERZ_EL_SIP_PASSWORD=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')" >> .env
   ```
2. The rest comes from what's already there: `SIGNALWIRE_CALLER_NUMBER` and `TELNYX_GUARDED_NUMBER`
   from the repo `.env`, and `ELEVENLABS_API_KEY` and `SIGNALWIRE_SPACE/PROJECT_ID/API_TOKEN` from the
   file `TELNYX_ENV_FILE` points to (set `ELEVENLABS_ENV_FILE` to use a different file).
   No secret is ever printed or copied.
3. `python -m callers.elevenlabs.place_calls check`: read-only, prints the plan and credits.
4. The first `call --live` creates the agents and imports the bridge number, then calls.

## Running

```sh
.venv/bin/python -m callers.elevenlabs.place_calls check                    # read-only
.venv/bin/python -m callers.elevenlabs.place_calls call                     # dry run, all 20 calls, passwords redacted
.venv/bin/python -m callers.elevenlabs.place_calls call --personas pharmacy_refill --variants 0 --live   # first call
#   check the recording and the transcript, then:
.venv/bin/python -m callers.elevenlabs.place_calls call --live              # the rest (one at a time)
```

Calls go one at a time. The script waits for each to end (it polls SignalWire's voice log) and
then pauses `--gap-s` (20 s). Options: `--max-calls N`, `--update-agents` (PATCH our agents to the
current spec), `--transport tcp` (debug only; TLS is the default).

Each placed call appends one row to `recordings/ai_calls.jsonl`, in the shared AI-caller shape:

```json
{"call_id": "<SignalWire call id>", "provider": "elevenlabs", "persona": "pharmacy_refill", "variant": 0,
 "voice": "Sarah:EXAVITQu4vr4xnSDxMaL", "engine": "elevenlabs-agents/eleven_flash_v2_5",
 "started_at": "2026-11-02T18:00:00+00:00",
 "platform": "elevenlabs", "started_utc": "2026-11-02T18:00:00+00:00", "label": "ai", "kind": "benign",
 "status": "queued", "route": "signalwire-sip-bridge", "from_last4": "1098", "to_last4": "1282",
 "el_agent_id": "agent_...", "exclude_from_training": true}
```

`platform`, `started_utc`, `label`, `kind`, `status` and `from_last4`/`to_last4` repeat the keys that
`callers/signalwire/place_calls.py` writes, so one reader handles both providers.

After each call, the script reads the ElevenLabs conversation (read-only) and writes to
`data/callers/elevenlabs_outcomes.jsonl`: the end status, the conversation id, `headers_arrived`
(whether `sip_emberz_persona`/`sip_emberz_variant` reached the agent), `call_duration_secs` and
the cost fields. If the headers didn't arrive, the agent ran on the variant-0 placeholders, so the
script **stops** after that call. That call's ledger row then claims a variant the agent didn't use:
join on `call_id` and drop rows where `headers_arrived` is false.

## Costs (checked 2026-11-02)

- **ElevenLabs Agents** ([agents pricing](https://elevenlabs.io/pricing/agents), fetched 2026-11-02;
  the page shows no year): Creator includes 275 call minutes a month, then $0.08 a minute, 10
  concurrent calls, burst calls at $0.16 a minute. "LLM usage is billed separately on top", deducted
  from credits; small for `gpt-4o-mini`. ElevenLabs adds no telephony fee; an external provider bills
  its own charges. The length is the connection time, with a 95% discount on silence over 10 s
  ([help](https://elevenlabs.io/docs/help-center/product/eleven-agents/how-much-does-eleven-agents-cost.md)).
  - Worst case: 20 calls x 180 s (`max_duration_seconds`) = **60 minutes**, under the 275 included.
  - This account (read-only `GET /v1/user/subscription`, 2026-11-02): legacy Creator, $22 next
    invoice, 62,201 of 300,000 credits used (237,799 left, resets 2026-11-06), current overage $0,
    and `allowed_to_extend_character_limit: false`, so usage can't run past the quota into overage.
    It's unconfirmed whether agent minutes on this credit plan come from the 275 minutes or from
    credits. Secondary guides (not official) quote 1,000 credits a minute; even then 60 minutes is
    ~60,000 of the 237,799 credits left. Each call's actual `el_cost` goes to the outcomes log.
- **SignalWire** ([voice pricing](https://signalwire.com/pricing/voice), fetched 2026-11-02): US
  local outbound is $0.008/min and SIP is $0.003/min, so a call is $0.011/min across its two legs.
  Worst case: 20 calls x 210 s (the SWML `max_duration`) = 70 min x $0.011 = **$0.77**.
  `max_price_per_minute: 0.05` rejects any oddly priced route. Check the first call's
  `charge_details` in the voice log.
- **Telnyx:** the guarded number's inbound minutes and media streaming, already in PLAN.md section 9.

## Not yet verified (needs the first live call)

- That SignalWire's SWML `connect` reaches `sip.rtc.elevenlabs.io` over TLS with digest auth, and
  audio flows both ways (PCMU/PCMA). If TLS fails, retry with `--transport tcp` to isolate it
  (ElevenLabs suggests testing without TLS first).
- That the `X-` headers arrive as `{{sip_*}}` variables (check the conversation's Phone Call tab).
- That the dial response `id` works with `GET /api/voice/logs/{id}` (its docs call the id a
  `segment_id`). If it 404s, the wait falls back to the ElevenLabs conversation reaching
  `done`/`failed`, then to a timeout of the spec's max duration + 180 s.
- That `skip_turn` actually keeps the agent quiet through the notice. Listen to the first call.
- Whether ElevenLabs accepts the import with no `outbound_trunk_config` (the create docs mark it
  optional and nullable).
- Whether agent minutes come from the 275 included minutes or from credits on this legacy plan.
