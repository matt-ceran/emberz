# callers/: V2 real test calls

Real calls over the guarded Telnyx number, recorded by `line/recorder.py`, labeled blind by Matt.

| Set | How many | Who calls | Defined in |
|---|---|---|---|
| AI test callers | 5 agents x 10 calls | Vapi (variants 0-4) and Retell (variants 5-9) | `agents/*.json`, `vapi_create.py`, `retell_create.py` |
| Role-play guardian calls | 40 | two people per call | `roleplay/cards.json`, `roleplay/README.md` |
| Human voice calls (V1 human class) | ~50 | friends and teammates | `human_voice_calls.md` |

## AI test callers

Agents: `bank_fraud`, `tech_support`, `prize_giftcard` (scams) and `pharmacy_refill`, `appointment_confirm`
(benign bots that say they're automated). No grooming persona: kid scenarios are role-played by people,
tactics only. Every org name is made up. Voices: OpenAI and Deepgram on Vapi; OpenAI, Cartesia and MiniMax
on Retell. **No ElevenLabs** (PLAN.md V1); `common.check_spec` refuses it.

Each prompt frames the call as an authorized test with a consenting team member, asks only for made-up
codes and gift-card numbers, and ends on the safe word "pineapple".

The agents **wait for "hello"** (Vapi `firstMessageMode: assistant-waits-for-user` with no `firstMessage`,
Retell `start_speaker: user`). The line answers the caller first, plays the Emberz notice and only then
rings the protected phone, so a speak-first agent would say its opening line to nobody. The opening line
(the spec's `first_message`) is in the prompt, with an instruction to stay silent through the notice and
ringing (`common.CALL_START`). Silence timeouts are 60 s so ringing doesn't end the call.

Everything is a **dry run by default** and prints the request payloads as JSON:

```sh
.venv/bin/python -m callers.vapi_create create                  # assistant payloads
.venv/bin/python -m callers.vapi_create call --variants 0-4     # call payloads (25)
.venv/bin/python -m callers.retell_create create
.venv/bin/python -m callers.retell_create call --variants 5-9
```

To go live (Numan only, after the line is up and recording):

1. In `.env`: `VAPI_API_KEY`, `VAPI_PHONE_NUMBER_ID`, `RETELL_API_KEY`, `RETELL_FROM_NUMBER`, `GUARDED_NUMBER`.
2. `create --live` once per platform. The created ids go to `data/callers/` (gitignored).
   Retell first runs `voices` to resolve concrete voice ids for each provider and gender.
3. `call --agents bank_fraud --variants 0 --live`: one call first, then check the recording.
4. Then the rest. Calls go one at a time: the script waits for each to end, then pauses `--gap-s`.
   Every placed call is logged to `data/callers/calls.jsonl`.

Vapi's free numbers allow about 10 outbound calls a day per org, so 25 Vapi calls take 3 days unless we
import a number.

## After the calls

```sh
.venv/bin/python -m callers.label_sheet match-sources    # recording <-> ledger row, by start time -> source.json
.venv/bin/python -m callers.label_sheet sheet --all      # blind sheets -> recordings/to_label/<call_id>.csv + .md
# Hand Matt recordings/to_label/ only (no audio, events.jsonl or source.json). He fills the risk,
# category, stage and ask columns, then:
.venv/bin/python -m callers.label_sheet import <call_id> --csv recordings/to_label/<call_id>.csv --labeler matt
.venv/bin/python -m callers.label_sheet validate
git add recordings/labels && git commit   # BEFORE any model scores these calls (needs the .gitignore fix: recordings/* + !recordings/labels/)
.venv/bin/python -m callers.label_sheet validate --require-committed   # gate for M3
```

The sheet shows the transcript and the guardian's inputs (profile, rules, caller known). It never shows
events (model outputs, coach lines, alerts) or `source.json` (which agent or card placed the call).
