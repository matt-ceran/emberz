# G1: Google Cloud bulk jobs (Numan)

Two one-shot jobs, each on its own VM that deletes itself when the job ends:

| Job | For | VM | Expected | Worst case (hard stop) |
|---|---|---|---|---|
| Noise pass | M1 (Matt) | spot n2-highcpu-96 | ~30 min / ~$1.05 for 300 calls; ~47 min / ~$1.60 for 500 | 90 min, $3.10 |
| AI-voice data | V1 | spot g2-standard-8 + 1 L4 | ~1-1.5 h / ~$0.50-0.80 | 4 h, $2.04 |

Project `unified-booster-512122-j5`, region us-central1, zone us-central1-a, bucket `gs://emberz-g1-512122j5`.
**Cap: $25 total.** Expected total for both jobs: about $2-4.

## Blocker first: the global CPU quota

`CPUS_ALL_REGIONS` is **16** (4 used by `gw-slice-1`), checked 2026-11-02. The 200 N2 vCPUs in
the plan are the *regional* quota; the global one caps us at 12 more vCPUs. So:
- the n2-highcpu-96 noise VM **cannot start** until the global quota is raised to at least 112
  (Console > IAM & Admin > Quotas > `compute.googleapis.com/cpus_all_regions`, ask for 128);
- the g2-standard-8 voice VM (8 vCPUs, 1 L4) **fits today** (GPU quota: 1 L4, spot allowed).

`launch_noise.sh` refuses to start (before uploading or creating anything) when the global or regional
quota is short (CPUS_ALL_REGIONS, CPUS, N2_CPUS, SSD_TOTAL_GB; plus L4 for the voice job).
Fallbacks if the increase is refused: `NOISE_MACHINE_TYPE=n2-standard-8 NOISE_MAX_RUN=6h` (~3 h for
300 calls, ~$0.70 spot; the default 90 min hard stop is far too short for it, and `launch_noise.sh`
refuses to start when its own estimate exceeds `NOISE_MAX_RUN`), or run `run_noise.py` on the Mac
when it is idle (~45 min for 300 calls with 12 workers).

## Run order

```
gcloud/check.sh                     # read-only: VMs, quotas, APIs, service account, bucket
gcloud/setup_bucket.sh              # once: bucket, service account emberz-g1, least-privilege grants
gcloud/check.sh                     # all "ok" except possibly the CPU quota

# noise pass (needs data/noise_input.jsonl from Matt: DatasetRow lines)
gcloud/launch_noise.sh              # validates the input, estimates wall time, checks quota, shows worst-case cost, asks "yes"
gcloud/watch.sh noise-MMDD-HHMM     # follows status; prints when done and the VM is gone
gcloud/fetch_noise.sh noise-MMDD-HHMM   # -> data/noisy_turns.jsonl (+ stats), validated as NoisyTurn

# AI-voice data (uses voice/pipeline.py from the V1 work)
gcloud/launch_voice.sh              # default steps download,manifests,features
gcloud/watch.sh voice-MMDD-HHMM
gcloud/fetch_features.sh voice-MMDD-HHMM [--small]   # -> data/voice_features/

gcloud/check.sh                     # at EVERY session end: must say "ok: no emberz VMs"
gcloud/teardown.sh                  # if anything emberz is left; --bucket also deletes the bucket
```

Both jobs can run at the same time once the quota allows (96 + 8 + 4 used = 108 vCPUs; SSD 30 + 150 of the 200 GB left).

## Teardown rule

1. Every VM is created with `--max-run-duration` and `--instance-termination-action=DELETE`:
   Google deletes it at that age (90 min noise, 4 h voice) whatever happens.
2. The VM deletes itself when its job ends **or fails** (`vm_startup.sh` exit trap). If the
   self-delete is refused it powers off (no CPU billing) and `teardown.sh` removes it.
3. `watch.sh` deletes the VM itself if the job wrote DONE but the VM is still there 3 min later.
4. Run `gcloud/check.sh` at every session end; run `gcloud/teardown.sh` if it shows any `emberz-` VM or disk.
5. `teardown.sh` only touches resources named `emberz-*` / labelled `emberz=g1`. `gw-slice-1` is not ours.

## How the noise pass works

`noise_pass/run_noise.py`, two phases so each worker holds one model:
1. **render**: Kokoro-82M renders each turn (caller and protected person get different voices;
   kid = lighter female voices, 1.1x speed; elder = older-sounding voices, 0.9x; voice ids from
   [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)), 24 -> 8 kHz,
   random level (-30..-22 dBFS), noise on 50% of turns (white/pink/brown, SNR 15-30 dB),
   optional packet loss, then G.711 mu-law. Kokoro has no child voices, so kid ASR noise is underestimated.
2. **asr**: mu-law decode, 8 -> 16 kHz, the same Nemotron 0.6B 160 ms int8 sherpa-onnx model as
   the live line (the VM checks the encoder's sha256 against the Mac's copy), 160 ms chunks.
   Calls go through in rounds of `--batch-calls` (default 200): render the round, transcribe it, append
its rows. The VM uploads the rows every minute (and once more if run_noise.py fails), so a run cut
short by preemption or the hard stop keeps its finished rounds, and `RUN_ID=<same id> launch_noise.sh`
resumes with only the missing calls. A call that fails (bad text, Kokoro error) is skipped and listed
under `failed` in the stats file instead of killing the run; the exit code is non-zero only when every
call fails. Rendered audio is cached per call in `<output>.work/` keyed on text + noise flags + seed.

Output: `data/noisy_turns.jsonl`, one `contracts.models.NoisyTurn` per turn (`text_clean`,
`text_asr`, `voice`, `wer`), plus `noisy_turns.stats.json` (timings, corpus WER, per-call peak memory).
Flags worth knowing: `--noise-prob`, `--snr-min/--snr-max`, `--loss-prob`, `--limit`, pass them via
`NOISE_EXTRA_ARGS="--noise-prob 0.8 --snr-min 5"`.

## Mac smoke test (2026-11-02, M5 Pro, shared with other agents)

Venv: `/private/tmp/claude-501/.../scratchpad/venvs/gcloud` (uv, Python 3.12; kokoro 0.9.4,
sherpa-onnx 2.0.1, torch 2.15.0). espeak-ng comes bundled with `espeakng-loader` (no brew), but
espeak-ng truncates data paths over ~160 bytes, so deep venv paths need `EMBERZ_ESPEAK_DATA`
pointing at a short *copy* of `espeakng_loader/espeak-ng-data` (the VM path is short; not needed there).

| Run | Calls / turns / audio | Render (worker-s per audio-s) | ASR (worker-s per audio-s) | Corpus WER |
|---|---|---|---|---|
| sample, 3 workers | 3 / 19 / 76.5 s | 0.25 (~1.0 s per turn) | 0.59 (~2.4 s per turn) | 0.5% |
| sample x8, 8 workers, Mac load ~14 | 24 / 152 / 644 s | 0.85 | 1.67 | 1.2% |
| sample, SNR 3-12 dB + 3% packet loss | 3 / 19 / 76.5 s | 0.27 | 0.60 | 0.9% |
| review rerun: sample, 2 workers, 2 rounds (`--batch-calls 2`), HF offline in workers | 3 / 19 / 76.5 s | 0.20 | 0.49 | 0.5% |

Peak memory per worker: ~1.9 GB (render and ASR) when the Mac had free memory; run_noise.py sizes
workers at 2 GB each, which caps an n2-highcpu-96 (96 GB) at 39 workers x 2 threads.

**Finding:** Kokoro speech through a phone channel is easy for Nemotron: 0.5-1.2% WER even at
3-12 dB SNR with packet loss. The noise pass will add few real ASR errors (mostly number/word-form
differences like "copay" -> "copy", "mom" -> "mum"). Real callers will be noisier than this.

## Estimate vs the plan's "~10-15 min, a few dollars"

Assumptions: ~120 s of audio per call (the plan's 300 calls = ~10 h), an n2 vCPU thread runs
~2.5x slower than an idle M5 Pro core (between the idle and loaded Mac numbers above), 2 threads
per worker give ~1.5x, ~6 min of setup (boot, uv + packages, 464 MB model, Kokoro weights).

| Machine (spot) | $/h | 300 calls | 500 calls |
|---|---|---|---|
| n2-highcpu-96 (memory-capped: 39 workers x 2 threads) | 2.06 | ~30 min, ~$1.05 | ~47 min, ~$1.60 |
| n2-standard-96 (96 workers) | 2.80 | ~21 min, ~$1.00 | ~31 min, ~$1.45 |
| n2-standard-8 (fits today's quota) | 0.23 | ~3 h, ~$0.70 | ~5 h, ~$1.15 |

On-demand is ~1.7x the spot price. **"A few dollars" holds; "10-15 min" is ~2x optimistic** on
n2-highcpu-96, because its 1 GB per vCPU caps the worker count. n2-standard-96 costs about the
same and is faster (`NOISE_MACHINE_TYPE=n2-standard-96`). These are estimates from Mac timings;
the stats file of the first VM run gives the real numbers.

Prices (USD/h, us-central1, Cloud Billing Catalog API, Compute Engine SKUs, read 2026-11-02):
N2 vCPU 0.031611 (spot 0.018960), N2 GiB 0.004237 (spot 0.002542); G2 vCPU 0.024988
(spot 0.014950), G2 GiB 0.002927 (spot 0.001751); L4 0.560040 (spot 0.335200).
Same numbers as https://cloud.google.com/compute/vm-instance-pricing and
https://cloud.google.com/compute/gpus-pricing (those pages render client-side, so the catalog API was used).

## AI-voice job

`voice_job/vm_run.sh` on the Deep Learning VM image family `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`
(CUDA 12.9, PyTorch 2.9, driver 580): waits for the GPU, finds the image's Python with torch, installs
`voice/requirements-gcp.txt` + `unzip`, then runs the V1 pipeline:

```
python -m voice.pipeline --data /data --out /data/voice_out --steps download,manifests,features
```

That downloads ASVspoof 5 parts T_aa, D_aa, E_aa (~22.6 GB, from Zenodo) and In-the-Wild (8.2 GB,
from Hugging Face) onto the VM's 150 GB disk, builds manifests and extracts wav2vec2 XLS-R 300M
features on the L4. Only `/data/voice_out` (features ~1-2 GB + small results) is synced to the
bucket (every 5 min, so a preempted run resumes); the audio dies with the VM. `VOICE_STEPS` can add
`head,eval,aasist` to finish V1 on the VM. Estimate: downloads 10-40 min (Zenodo speed varies),
features ~15-30 min for ~57k clips, plus setup; about 1-1.5 h, ~$0.50-0.80 spot.

## Files

| File | What |
|---|---|
| `config.sh` | project, zone, bucket, machine types, max-run durations, model URL + sha256, prices |
| `check.sh` | read-only status; run at every session end |
| `setup_bucket.sh` | one-time setup (creates things; Numan runs it) |
| `launch_noise.sh`, `launch_voice.sh` | create one VM each (spend money; Numan runs them) |
| `launch_common.sh` | preflight (quota, bucket, SA), code packing, cost line |
| `vm_startup.sh` | VM bootstrap: metadata, code, job, logs, self-delete on any exit |
| `job_config.py` | validates the VM's emberz-* metadata, prints shell exports |
| `watch.sh`, `fetch_noise.sh`, `fetch_features.sh`, `teardown.sh` | follow, fetch, clean up |
| `noise_pass/` | `run_noise.py`, `noise_lib.py`, `vm_run.sh`, pinned requirements, `sample_input.jsonl` |
| `voice_job/vm_run.sh` | the voice job body |

Tests: `.venv/bin/python -m pytest tests/gcloud` (no models or cloud needed; the shell scripts run
against a fake `gcloud`).

## Ledger

Append each run to `results/ledger-numan.md` (section G1): run id, machine, provisioning, start/end,
wall time (from `status.txt`), cost = hours x $/h, and that `check.sh` showed no emberz VMs afterwards.
