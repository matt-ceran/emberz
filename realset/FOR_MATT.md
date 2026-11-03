# Scoring your guardian on the real-audio set (realset)

Goal: your tuned guardian sees exactly the same inputs as the frontier baselines
(Kimi K3, Qwen3.8 Max, keyword stub), so the numbers in `results/realset.md` compare 1:1.

## The file: `data/realset/calls.jsonl`

Also at `~/Share/emberz/realset_calls.jsonl` on the VPS (Harbor Share sync). One JSON object per line:

```json
{"call_id": "ncsu_761222", "source": "ncsu_robocall" | "harpervalley_bank",
 "license": "...", "label": "scam" | "benign", "scam_type": "gov_ssa_irs" | "benign.bank.pay_bill" | ...,
 "audio_s": 16.84, "asr_wall_s": 37.1,
 "turns": [{"call_id": "...", "turn": 1, "speaker": "caller" | "protected", "text": "...",
            "t_start_ms": 0, "t_end_ms": 5120, "gap_ms": null}, ...]}
```

`turns` are `contracts.models.Turn` rows from our live ASR path: audio resampled to 8 kHz,
G.711 mu-law, 20 ms frames into `line.asr.CallTranscriber` (Nemotron streaming 0.6B, the
code the line server runs). Robocalls: everything is on `caller`, there are no `protected`
turns. HarperValley: the bank agent's channel is `caller`, the customer's is `protected`.

Do not use `label` / `scam_type` / `source` in the model input; they are for scoring only.

## Window rule (one request per caller turn)

For every turn `t` with `speaker == "caller"` (in order), build one guardian_request:

```python
from guardian.render import WINDOW, render_messages   # WINDOW = 8, PROMPT_SHA 806908cfba7d

turns = call["turns"]
for i, t in enumerate(turns):
    if t["speaker"] != "caller":
        continue
    win = turns[max(0, i + 1 - WINDOW): i + 1]          # last 8 turns, both speakers, ending at t
    req = {"call_id": call["call_id"], "profile": "elder", "rules": [], "caller_known": False,
           "ai_voice": None, "earlier": None,
           "turns": [{"turn": w["turn"], "speaker": w["speaker"], "text": w["text"]} for w in win]}
    messages = render_messages(req)                     # [system, user], byte-identical to training
```

Fixed for every call: `profile "elder"` (so the profile cannot leak the class), no caretaker
rules, unknown caller number, voice "unknown", `earlier` = nothing flagged. Exactly this is in
`realset/score.py::requests_for`, which you can import instead of copying.

Sampling: temperature 0. The baselines used `max_tokens 600`, `reasoning_effort "none"`.

## Decision rule

- Parse each reply with `guardian.render.parse_response`. Unparseable reply = not flagged (count it as an error).
- A call is **flagged** if ANY of its scored caller turns has `risk` in `{"alert", "urgent"}`.
- Scam recall = flagged scam calls / scam calls. False-alarm rate = flagged benign calls / benign calls.
- 95% CI: Wilson (`realset/metrics.py::wilson`).

## Easiest path: write your verdicts in our format and reuse our metrics

Write `data/realset/scores/<your-tag>.jsonl`, one line per scored turn:

```json
{"call_id": "ncsu_761222", "turn": 3, "model": "<your-tag>", "verdict": {"risk": "alert", ...} | null,
 "usage": {"prompt_tokens": 410, "completion_tokens": 90}, "latency_ms": 350}
```

then `python -m realset.metrics --models <your-tag> kimi-k3 qwen3p8-max keyword-stub`.
It rewrites `results/realset_results.jsonl` (ResultsRow), `results/realset_chart.svg`, and
`data/realset/summary.json` (recall by scam type, missed calls, false-alarm calls). Add your
tag's $/1M token price to `PRICES` in `realset/metrics.py` if you want a cost column.
