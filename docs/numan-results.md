# Numan-side measured numbers (for the README)

_Updated 2026-11-02 ~14:40 PST. Every number below comes from a command run in this repo or a file named next to it; anything not yet measured says so._

## Live speech-to-text (NVIDIA Nemotron Speech Streaming 0.6B, sherpa-onnx INT8, local CPU)

Bench: `.venv/bin/python -m line.bench_asr all` (8 kHz mu-law 20 ms frames, as a Telnyx stream; LibriSpeech test wavs + a scripted scam dialog voiced with macOS `say`).

- **Endpoint delay (audio time, compute excluded): p50 0.62 s, p95 0.74 s, max 0.76 s** over 26 utterance ends, 2 calls / 4 streams paced in real time, VAD silence 0.6 s (`.venv/bin/python -m line.bench_asr delay`, run 14:46 PST Nov 2). Cut reasons: 24 VAD, 2 speaker handoff.
- The wall-time numbers from that run are **not valid**: the Mac was saturated by concurrent jobs (load average ~26), the decoder logged "decoding is falling behind" and queued up to 4,796 frames. Re-run `delay` on an idle machine (or the VPS) before quoting a wall-time number.
- WER on the bench material: the `wer` run did not finish inside this time box on the loaded machine; run `.venv/bin/python -m line.bench_asr wer` when the CPU is free.

Earlier measurement (PLAN.md 2b): **37 ms** to decode each 160 ms chunk, **64 ms** worst case with 4 streams (2 live calls) at once; bulk ~11x real time with 8 processes.

From Matt's run (TRAINING_RUN.md, 14:26): 49 held-out test calls voiced with macOS TTS -> 8 kHz -> Nemotron ASR give **median WER 16%** (mostly numbers written as words). On those ASR transcripts every risky call is still flagged at some point by our guardian (100%, vs 97% for Rune and DeepSeek).

## Noise pass (G1)

Not finished at the time of writing: `data/noisy_turns.jsonl` is still empty and `data/noisy_turns.stats.json` does not exist yet; the work directory `data/noisy_turns.jsonl.work/` holds 40 per-call render files (Kokoro voices, per-turn SNR 20-28 dB noise, speed 1.03-1.08x). Fill in calls / hours / WER from the stats file when it lands.

## AI-voice detector

From `results/voice_v1.md` / `results/voice_results.jsonl` (wav2vec2 XLS-R-300m layer 9 features + logistic head `w2v-head-v1`): In-the-Wild clips from 25 speakers unseen in training, simulated 8 kHz mu-law phone path, first 4 s of speech, n = 2,829 (1,413 AI / 1,416 human): **EER 1.27% (95% CI 0.85-1.77%), AUC 0.9991 (0.9985-0.9996)**; 98.0% of AI clips caught at 1% false alarms. First 2 s: EER 1.55%. Clean audio: EER 0.92%. Caveats from that file: in-domain test, pre-2022 TTS, simulated phone path. Figure: `results/figures/voice_roc.png`.

## Live line latency

No end-to-end live-call latency has been measured yet (no recorded live call with timings in the repo). Stage numbers available now (`results/figures/line_budget.png`, inputs in `results/figures/line_latency.json`):

| Stage | p50 | Status |
|---|---|---|
| ASR endpoint (VAD 0.6 s) | 620 ms | measured, audio time |
| Guardian v2 | 1,030 ms | Training API sampler, not a deployment |
| Telnyx conference `speak` start | - | not measured |
| Sum of known stages | 1,650 ms | over the 1.5 s target until the guardian runs on a dedicated deployment |

## Guardian numbers to quote (from Matt's snapshot, `results/guardian_snapshot.json`)

- Same 120 held-out calls: fine-tuned guardian v1 false alarms **16.7%** (6/36 benign) vs Kimi K3 **25.0%** and DeepSeek V4.1 Flash **33.3%**; catch before the ask **86.5%** (64/74) vs 85.1% and 81.1%. Wilson 95% intervals overlap at n=36 benign.
- All 257: false alarms **53.2% -> 16.5%** (untuned base -> v1), rule-flip pairs **3/22 -> 19/22**.
- Our fine-tune on NVIDIA Nemotron 3.5 Super: all-257 false alarms **15.2%** (12/79), catch 89.2%, rule flips 17/22 (same family as the test-set writer: caveat).
- BothBosu (different-family calls): v2 recall **98%**, 0/100 false alarms.
- Guardian latency is Training API sampler latency, not a deployment.

## Figures

`results/figures/` (PNG at 2x + SVG): `headline_subset_120`, `headline_all_257`, `rule_flips`, `latency_p50`, `bothbosu`, `line_budget`, `voice_roc`. One-page view for the video: `docs/results.html`.

Rebuild everything:

```
.venv/bin/python results/figures/build_snapshot.py
uv run --no-project --with matplotlib==3.9.2 python results/figures/make_figures.py
.venv/bin/python results/figures/build_html.py
```
