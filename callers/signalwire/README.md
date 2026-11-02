# callers/signalwire: AI test callers on SignalWire

SignalWire AI agents call the guarded Telnyx number as five personas (PLAN.md 2c, V2). They use no ElevenLabs
voices. Each call is a single `POST /api/calling/calls` whose SWML document runs an `ai` agent. The SWML is sent
inline, so nothing has to be created on SignalWire and no inbound number is reconfigured.

- **From:** `SIGNALWIRE_CALLER_NUMBER` (+1 304 648 1098). This is an idle number used only for outbound calls.
  Its inbound handler (`laml_webhooks`) is never touched.
- **To:** `TELNYX_GUARDED_NUMBER` (+1 323 516 1282). There is no flag to dial anything else.
- **Default:** a dry run that prints the exact request. `--live` places the calls, and `--max-calls` (default 1)
  caps how many.

## Personas and variants

The prompts and the 10 fill values per persona (names, banks, amounts) are read from `callers/agents/*.json`, which
this folder does not change. Each persona has 3 variants, and each variant uses a different TTS engine, pace and
opening line. Call *k* uses variant *k* mod 3 and a fill value not yet used with that variant. A caller name always
matches the voice's gender. 10 calls per persona give 50 different calls.

| Persona | Kind | v0 | v1 | v2 |
|---|---|---|---|---|
| bank_fraud | scam | Azure Andrew, normal | Google Neural2-F, slow | Cartesia Customer Support Man, brisk |
| tech_support | scam | Deepgram Aura arcas, normal | Polly Joanna (neural), slow | Azure Guy, brisk |
| prize_giftcard | scam | OpenAI nova (tts-1), brisk | Cartesia New York Man, normal | Polly Danielle (generative), slow |
| pharmacy_refill | benign | Google Neural2-C, normal | Deepgram Aura asteria, slow | OpenAI echo (tts-1), brisk |
| appointment_confirm | benign | Azure Jenny, normal | Polly Matthew (neural), brisk | Cartesia Helpful Woman, slow |

Across the 15 variants there are six engines (Azure, Google, Amazon Polly, OpenAI, Deepgram and Cartesia), and each
is used 2-3 times. Every voice is in SignalWire's Economy, Economy+ or Standard TTS tier, never Premium or Pro.
"Pace" sets three things:

- the agent's turn-taking (`end_of_speech_timeout` 500/800/1200 ms);
- its nudge timing;
- a spoken-style line in the prompt.

So the reply gap that V1's timing cue measures varies too.

## What one call does

```
SignalWire dials the guarded number --> our line answers, plays the Emberz notice, rings the protected phone
SWML main:
  play silence:5-7s     the agent is not listening yet, so it can't mistake the notice for the person
  ai                    wait_for_user: true -> stays silent until the protected person says "Hello?"
                        opens with the variant's line, follows the spec's call plan
                        hard_stop_time 240s (scams) / 150s (benign), inactivity 45 s,
                        nobody greets within 60 s -> times out
                        says "Have a good day, goodbye." then calls SWAIG end_call (data_map action hangup)
                        "pineapple" from the person ends the test at once
  hangup
```

## Commands

```sh
.venv/bin/python -m callers.signalwire.place_calls --summary              # 50 planned calls, one line each + cost
.venv/bin/python -m callers.signalwire.place_calls --personas bank_fraud --calls 1   # exact request JSON (dry run)
.venv/bin/python -m callers.signalwire.place_calls check                  # read-only: auth + caller number
# live (Numan; the line must be up and recording, and someone answers the protected phone):
.venv/bin/python -m callers.signalwire.place_calls --personas bank_fraud --calls 1 --live
.venv/bin/python -m callers.signalwire.place_calls --calls 10 --live --max-calls 50        # Enter before each call
.venv/bin/python -m callers.signalwire.place_calls --personas tech_support --start 3 --calls 7 --live --max-calls 7  # resume
.venv/bin/python -m callers.signalwire.place_calls logs                   # read-only: duration, charge, voice warnings
```

How live runs behave:

- **Pacing:** the script waits for Enter before each call (`q` stops). Use `--wait gap --gap-s 300` for unattended
  spacing.
- **Failed request:** the run stops at the first request that fails, and the failure is still logged.
- **Credentials:** read from the process environment, or from the file `TELNYX_ENV_FILE` points at (EmberDial's
  `.env`). Only the three `SIGNALWIRE_SPACE`, `SIGNALWIRE_PROJECT_ID` and `SIGNALWIRE_API_TOKEN` keys are read, and
  they are never printed.

`--mode url` makes SignalWire fetch the SWML from `PUBLIC_BASE_URL/callers/swml/{persona}/{variant}?fill=N`. That
needs `swml_router.py` mounted on the line server:
`app.include_router(callers.signalwire.swml_router.router)`. Inline mode needs none of this and is the default.

## Ledger (ground truth)

Each live call appends one row to `recordings/ai_calls.jsonl`. Its fields:

- `started_utc`, `platform: signalwire` and `label: ai`;
- the SignalWire `call_id` and `status`;
- `persona` and `kind` (scam or benign);
- `variant`, `fill`, `voice`, `voice_label`, `engine`, `voice_gender` and `tts_tier`;
- `pace`, `opening_text`, `ai_model` and `swml_sha`;
- the last 4 digits of each number.

`label: ai` is V1's AI-voice label, and `persona`/`kind` is the scenario label for the guardian live set. Each call
is also appended to `data/callers/calls.jsonl`, so `python -m callers.label_sheet match-sources` pairs recordings
with sources. **`recordings/` is gitignored.** If this ledger is to be committed as ground truth, `.gitignore` needs
an exception such as `!recordings/ai_calls.jsonl`.

## Cost

From [SignalWire voice pricing](https://signalwire.com/pricing/voice), read 2026-11-02:

- **AI Agent runtime:** $0.16/min, including STT, the LLM and standard TTS. Premium TTS costs extra; we use none.
- **Outbound US PSTN (local):** $0.008/min. Recent EmberDial voice logs bill "Outbound Voice Minutes" at
  $0.008/min, which matches.
- **All in:** about **$0.168 per call-minute**. SignalWire says customers typically see $0.163-0.168/min.

| | Minutes per call | Cost per call |
|---|---|---|
| Scam, typical (about 3 min) | 3.0 | ~$0.50 |
| Benign, typical (about 2 min) | 2.0 | ~$0.34 |
| **50 calls (30 scam + 20 benign), typical** | 130 | **~$22** |
| 50 calls, every call hits its hard stop (4:15 / 2:45), plus Standard-tier TTS billed on top (~1,500 chars × $0.0001) | 182 | ~$38 (upper bound) |

The Telnyx side (inbound leg, conference, streaming, the protected-phone leg) is separate. It is about $0.01/min
(PLAN.md 9).

## Verified (read-only, 2026-11-02)

- `place_calls check`: auth works, and +1 304 648 1098 is in the project as a longcode number with voice capability.
  Its inbound handler is `laml_webhooks` and was left unchanged.
- **Request bodies:** all 50 dry-run bodies validate against SignalWire's OpenAPI 3.1 spec, using `Calling.CallRequest`
  with the `dial` / `Calling.CallCreateParamsSWML` schemas
  ([openapi](https://signalwire.com/docs/openapi/signalwire-rest-api.json)). The tests use the vendored subset in
  `tests/callers_sw/dial_params_schema.json`.
- **SWML documents:** all 50 validate against the SWML JSON schema shipped in `signalwire-sdk==3.4.1`
  (`signalwire/schema.json`, checked with `jsonschema` in a scratch venv). That schema is stricter than the docs in
  two places, and we meet both:
  - `hard_stop_time`: the schema accepts only the `"<n>h<n>m<n>s"` form, while the docs say letters are ignored.
    `"240s"` means 240 s under both.
  - `attention_timeout`: the schema requires at least 10,000 ms.
- **Not verifiable without a call:**
  - that each voice id resolves on SignalWire. An unresolved voice is replaced, with a
    `relay_script_voice_unresolved` warning, and `place_calls logs` flags it.
  - that `wait_for_user` plus the leading silence handles our notice and ringing as intended.

## Sources

- [Call commands (dial)](https://signalwire.com/docs/apis/rest/calls/call-commands)
- [SWML `ai`](https://signalwire.com/docs/swml/reference/ai) ·
  [languages and voice strings](https://signalwire.com/docs/swml/reference/calling/ai/languages) ·
  [params](https://signalwire.com/docs/swml/reference/calling/ai/params) ·
  [data_map actions](https://signalwire.com/docs/swml/reference/calling/ai/swaig/functions/data-map) ·
  [play](https://signalwire.com/docs/swml/reference/calling/play) ·
  [errors](https://signalwire.com/docs/swml/reference/errors)
- [TTS voices](https://signalwire.com/docs/platform/voice/tts): the azure, gcloud, amazon-polly, openai, deepgram and
  cartesia sub-pages. The Cartesia ids come from SignalWire's own catalog table.
- [Phone numbers list](https://signalwire.com/docs/apis/rest/phone-numbers/list-phone-numbers) ·
  [voice logs](https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log) ·
  [pricing](https://signalwire.com/pricing/voice)
