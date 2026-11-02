# Numan-side measured numbers (for the README)

_Updated 2026-11-02 ~14:40 PST. Every number below comes from a command run in this repo or a file named next to it; anything not yet measured says so._

## Live speech-to-text (NVIDIA Nemotron Speech Streaming 0.6B, sherpa-onnx INT8, local CPU)

Bench: `.venv/bin/python -m line.bench_asr all` (8 kHz mu-law 20 ms frames, as a Telnyx stream; LibriSpeech test wavs + a scripted scam dialog voiced with macOS `say`).

ASR_BENCH_PLACEHOLDER

Earlier measurement (PLAN.md 2b): **37 ms** to decode each 160 ms chunk, **64 ms** worst case with 4 streams (2 live calls) at once; bulk ~11x real time with 8 processes.

From Matt's run (TRAINING_RUN.md, 14:26): 49 held-out test calls voiced with macOS TTS -> 8 kHz -> Nemotron ASR give **median WER 16%** (mostly numbers written as words). On those ASR transcripts every risky call is still flagged at some point by our guardian (100%, vs 97% for Rune and DeepSeek).

## Noise pass (G1)

Not finished at the time of writing: `data/noisy_turns.jsonl` is still empty and `data/noisy_turns.stats.json` does not exist yet; the work directory `data/noisy_turns.jsonl.work/` holds 40 per-call render files (Kokoro voices, per-turn SNR 20-28 dB noise, speed 1.03-1.08x). Fill in calls / hours / WER from the stats file when it lands.

## AI-voice detector

Not available yet: `results/voice_roc.json` does not exist, so there is no ROC figure. `make_figures.py` draws `results/figures/voice_roc.png` automatically once it appears.

## Live line latency

LINE_LATENCY_PLACEHOLDER

## Guardian numbers to quote (from Matt's snapshot, `results/guardian_snapshot.json`)

- Same 120 held-out calls: fine-tuned guardian v1 false alarms **16.7%** (6/36 benign) vs Kimi K3 **25.0%** and DeepSeek V4.1 Flash **33.3%**; catch before the ask **86.5%** (64/74) vs 85.1% and 81.1%. Wilson 95% intervals overlap at n=36 benign.
- All 257: false alarms **53.2% -> 16.5%** (untuned base -> v1), rule-flip pairs **3/22 -> 19/22**.
- BothBosu (different-family calls): v2 recall **98%**, 0/100 false alarms.
- Guardian latency is Training API sampler latency, not a deployment.

## Figures

`results/figures/` (PNG at 2x + SVG): `headline_subset_120`, `headline_all_257`, `rule_flips`, `latency_p50`, `bothbosu`, plus `line_budget` / `voice_roc` when their inputs exist. One-page view for the video: `docs/results.html`.

Rebuild everything:

```
.venv/bin/python results/figures/build_snapshot.py
uv run --no-project --with matplotlib==3.9.2 python results/figures/make_figures.py
.venv/bin/python results/figures/build_html.py
```
