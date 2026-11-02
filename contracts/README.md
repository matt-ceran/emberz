# Contracts (I0 draft)

Pydantic models for every PLAN.md section 4/6 shape live in `models.py`; `python -m contracts.export` writes JSON Schema to `schemas/`.
`stub_guardian.score()` returns canned responses so the line and console can be built before the model exists.

Added beyond section 4 (to review at kickoff):
- `NoisyTurn`: G1 output row (`data/noisy_turns.jsonl`). G1 input `data/noise_input.jsonl` = `DatasetRow` lines.
- `ConsoleEvent.type` also has `voice` (AI-voice badge updates).
- `DatasetRow.caller_known` / `ai_voice`: the simulated live signals from M1.

`console_event.data` by type:
- `call_started`: `{caller_number, guarded_number, profile, caller_known}`
- `turn`: a `Turn`
- `voice`: a `VoiceSignal`
- `risk`: `{turn, response: GuardianResponse, latency_ms}`
- `alert`: `{level: alert|urgent, category, caretaker_note, turn}`
- `coach`: `{text, turn}`
- `caretaker_joined`: `{mode: monitor|barge}`
- `summary`: `{text}`
- `call_ended`: `{duration_ms, reason}`
