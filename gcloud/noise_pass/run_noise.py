"""G1 noise pass: DatasetRow calls -> Kokoro -> 8 kHz mu-law -> Nemotron ASR -> NoisyTurn rows.

Two phases so each worker holds only one model in memory:
  render: Kokoro-82M renders every turn, converts to phone audio (8 kHz G.711 mu-law),
          and saves one .npz per call in --work-dir.
  asr:    the same sherpa-onnx Nemotron streaming model as the live line transcribes
          each turn (mu-law decode, 8 -> 16 kHz, 160 ms chunks) and rows are appended
          to --output, one call at a time, so a preempted run can resume.

Usage (repo root on PYTHONPATH):
  python gcloud/noise_pass/run_noise.py --input data/noise_input.jsonl \
      --output data/noisy_turns.jsonl --model-dir models/sherpa-onnx-nemotron-...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for `contracts`

import noise_lib as nl  # noqa: E402
from contracts.models import DatasetRow  # noqa: E402

KOKORO_REPO = "hexgrad/Kokoro-82M"  # https://huggingface.co/hexgrad/Kokoro-82M
KOKORO_SR = 24000
PHONE_SR = 8000
ASR_SR = 16000
CHUNK = 2560  # 160 ms at 16 kHz, the model's chunk size

# Peak resident memory per worker (ru_maxrss / ps) on the Mac: up to ~1.9 GB for both render and
# ASR when the Mac was not under memory pressure (gcloud/README.md). Caps the worker count so a
# 96 GB VM never runs out of memory; spare cores become extra threads (plan_workers).
RENDER_GB_PER_WORKER = 2.0
ASR_GB_PER_WORKER = 2.0
# Calls per render+ASR round. Rows are written after each round, so a run cut short by
# --max-run-duration or a spot preemption keeps (and uploads) the finished rounds instead of
# losing everything still sitting in the render phase. Each round restarts the pools (~10-20 s).
DEFAULT_BATCH_CALLS = 200


# ------------------------------------------------------------------ worker state
_STATE: dict = {}


def _pin_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)


ESPEAK_MAX_PATH = 150  # espeak-ng keeps its data path in a 160-byte buffer (path_home, N_PATH_HOME)


def espeak_data_path() -> str:
    """Kokoro's G2P falls back to espeak-ng (bundled by espeakng-loader). espeak-ng silently
    truncates data paths longer than ~160 bytes and then calls exit(), killing the worker.
    Deep venv paths hit that, so EMBERZ_ESPEAK_DATA can point at a short COPY (not a symlink:
    phonemizer resolve()s it back to the long path) of the
    espeak-ng-data dir. Checked in the parent before any worker starts."""
    import espeakng_loader

    path = os.path.realpath(os.environ.get("EMBERZ_ESPEAK_DATA") or espeakng_loader.get_data_path())
    if len(path.encode()) > ESPEAK_MAX_PATH:
        raise SystemExit(
            f"espeak-ng data path is {len(path.encode())} bytes (> {ESPEAK_MAX_PATH}); copy "
            f"{espeakng_loader.get_data_path()} to a short path (cp -R, not a symlink) and set EMBERZ_ESPEAK_DATA to it"
        )
    return path


def _fix_espeak_path() -> None:
    import misaki.espeak  # noqa: F401  (sets the long default path on import; override it after)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_data_path(espeak_data_path())


def _check_init() -> None:
    if "init_error" in _STATE:
        raise RuntimeError(f"worker init failed: {_STATE['init_error']}")


def init_render(threads: int) -> None:
    """Pool initializer. multiprocessing.Pool respawns workers forever when an initializer
    raises, so the error is recorded and raised from the first task instead (stops the run)."""
    try:
        _init_render(threads)
    except Exception as e:  # noqa: BLE001
        _STATE["init_error"] = f"{type(e).__name__}: {e}"


def _init_render(threads: int) -> None:
    _pin_threads(threads)
    import torch
    from kokoro import KModel, KPipeline  # https://github.com/hexgrad/kokoro

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _fix_espeak_path()
    model = KModel(repo_id=KOKORO_REPO).eval()
    _STATE["pipes"] = {
        code: KPipeline(lang_code=code, repo_id=KOKORO_REPO, model=model) for code in ("a", "b")
    }
    _STATE["torch"] = torch


def init_asr(threads: int, model_dir: str) -> None:
    try:
        _init_asr(threads, model_dir)
    except Exception as e:  # noqa: BLE001
        _STATE["init_error"] = f"{type(e).__name__}: {e}"


def _init_asr(threads: int, model_dir: str) -> None:
    _pin_threads(threads)
    import sherpa_onnx

    # Constructor args: OnlineRecognizer.from_transducer in sherpa-onnx 2.0.1 (inspected installed
    # package); model files from https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html
    d = Path(model_dir)
    _STATE["rec"] = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(d / "tokens.txt"),
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"),
        num_threads=threads,
        sample_rate=ASR_SR,
        feature_dim=80,
        decoding_method="greedy_search",
    )


# ------------------------------------------------------------------ phase 1: render
def to_phone(audio24k: np.ndarray, rng: np.random.Generator, noise_prob: float, snr: tuple[float, float],
             loss_prob: float = 0.0):
    """24 kHz float -> 8 kHz G.711 mu-law bytes, with a random level, maybe noise, maybe packet loss."""
    import soxr

    x = soxr.resample(audio24k.astype(np.float32), KOKORO_SR, PHONE_SR)
    x = nl.set_level(x, rng.uniform(-30.0, -22.0))
    noise = None
    if rng.random() < noise_prob:
        snr_db = float(rng.uniform(*snr))
        color = float(rng.choice([0.0, 1.0, 2.0]))
        x = nl.add_noise(x, snr_db, rng, color)
        noise = {"snr_db": round(snr_db, 1), "color": color}
    x = nl.packet_loss(x, loss_prob, rng)
    return nl.lin2ulaw(nl.float_to_pcm16(x)), noise


def render_key(job: dict) -> str:
    """Hash of everything that changes a call's rendered audio (texts, speakers, profile, noise
    settings, seed). A cached render is reused only when this matches, so rerunning with other
    --noise-prob/--snr-*/--loss-prob/--seed flags never silently reuses old audio."""
    row = job["row"]
    parts = {
        "profile": row["profile"],
        "turns": [(t["turn"], t["speaker"], t["text"]) for t in row["turns"]],
        "seed": job["seed"], "noise_prob": job["noise_prob"], "snr": list(job["snr"]),
        "loss_prob": job.get("loss_prob", 0.0),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def render_call(job: dict) -> dict:
    _check_init()
    call_id = job["row"]["call_id"]
    try:
        return _render_call(job)
    except Exception as e:  # noqa: BLE001  one bad call must not kill a 500-call run
        return {"call_id": call_id, "error": f"render: {type(e).__name__}: {e}"}


def _render_call(job: dict) -> dict:
    row = DatasetRow.model_validate(job["row"])
    out_npz = Path(job["work_dir"]) / f"{nl.safe_name(row.call_id)}.npz"
    meta_path = out_npz.with_suffix(".json")
    key = render_key(job)
    if meta_path.exists() and out_npz.exists():
        cached = json.loads(meta_path.read_text())
        if cached.get("key") == key:
            return cached

    torch = _STATE["torch"]
    voices = nl.pick_voices(row.call_id, row.profile, job["seed"])
    rng = np.random.default_rng(nl.stable_rng("noise", row.call_id, job["seed"]).getrandbits(64))
    arrays, turns = {}, []
    t0 = time.perf_counter()
    for t in row.turns:
        voice = voices[t.speaker]
        speed = nl.speaker_speed(row.call_id, t.speaker, row.profile, job["seed"])
        text = t.text.strip()
        audio = np.zeros(0, dtype=np.float32)
        if text:
            pieces = []
            with torch.inference_mode():
                for res in _STATE["pipes"][nl.lang_code(voice)](text, voice=voice, speed=speed, split_pattern=None):
                    if res.audio is not None:
                        pieces.append(res.audio.numpy())
            if pieces:
                audio = np.concatenate(pieces)
        ulaw, noise = to_phone(audio, rng, job["noise_prob"], job["snr"], job.get("loss_prob", 0.0))
        arrays[f"t{t.turn}"] = ulaw
        turns.append({"turn": t.turn, "seconds": round(len(ulaw) / PHONE_SR, 3), "noise": noise, "speed": speed})
    np.savez(out_npz, **arrays)
    meta = {
        "call_id": row.call_id,
        "key": key,
        "voices": voices,
        "turns": turns,
        "audio_s": round(sum(x["seconds"] for x in turns), 3),
        "render_s": round(time.perf_counter() - t0, 3),
        "render_maxrss_mb": max_rss_mb(),
    }
    meta_path.write_text(json.dumps(meta))
    return meta


# ------------------------------------------------------------------ phase 2: ASR
def transcribe(ulaw: np.ndarray) -> str:
    """Same path as the live line: mu-law decode, 8 -> 16 kHz, 160 ms chunks, tail padding."""
    import soxr

    if len(ulaw) == 0:
        return ""
    x = soxr.resample(nl.pcm16_to_float(nl.ulaw2lin(ulaw)), PHONE_SR, ASR_SR).astype(np.float32)
    rec = _STATE["rec"]
    s = rec.create_stream()
    for i in range(0, len(x), CHUNK):
        s.accept_waveform(ASR_SR, x[i : i + CHUNK])
        while rec.is_ready(s):
            rec.decode_stream(s)
    s.accept_waveform(ASR_SR, np.zeros(int(0.8 * ASR_SR), dtype=np.float32))  # flush the last chunk
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    return rec.get_result(s).strip()


def asr_call(job: dict) -> dict:
    _check_init()
    try:
        return _asr_call(job)
    except Exception as e:  # noqa: BLE001
        return {"call_id": job["row"]["call_id"], "error": f"asr: {type(e).__name__}: {e}"}


def _asr_call(job: dict) -> dict:
    row = DatasetRow.model_validate(job["row"])
    base = Path(job["work_dir"]) / nl.safe_name(row.call_id)
    meta = json.loads(base.with_suffix(".json").read_text())
    t0 = time.perf_counter()
    with np.load(base.with_suffix(".npz")) as z:
        texts = {t.turn: transcribe(z[f"t{t.turn}"]) for t in row.turns}
    rows = nl.build_noisy_rows(row, texts, meta["voices"])
    return {
        "call_id": row.call_id,
        "lines": [r.model_dump_json() for r in rows],
        "asr_s": round(time.perf_counter() - t0, 3),
        "audio_s": meta["audio_s"],
        "render_s": meta["render_s"],
        "render_maxrss_mb": meta.get("render_maxrss_mb"),
        "asr_maxrss_mb": max_rss_mb(),
        "pairs": [(r.text_clean, r.text_asr) for r in rows],
    }


# ------------------------------------------------------------------ driver
def max_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 2**20 if sys.platform == "darwin" else rss / 2**10, 1)


def total_mem_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 8.0


def auto_workers(threads: int, gb_per_worker: float, cpus: int | None = None, mem_gb: float | None = None) -> int:
    cpus = cpus or os.cpu_count() or 1
    mem_gb = mem_gb if mem_gb is not None else total_mem_gb()
    by_cpu = max(1, cpus // threads)
    by_mem = max(1, int((mem_gb * 0.85 - 2.0) // gb_per_worker))
    return max(1, min(by_cpu, by_mem))


def plan_workers(requested: int, threads: int, gb_per_worker: float) -> tuple[int, int]:
    """(workers, threads per worker). When memory caps the worker count below the CPU count,
    the spare cores go to the workers as extra torch/onnx threads."""
    if requested:
        return requested, threads
    w = auto_workers(threads, gb_per_worker)
    return w, max(threads, (os.cpu_count() or 1) // w)


# Rough VM speed model (gcloud/README.md): idle M5 Pro core = 0.84 worker-s per audio-s
# (render 0.25 + ASR 0.59); an n2 vCPU thread ~2.5x slower; 2+ threads per worker ~1.5x faster.
# Kokoro speaks ~2.8 words/s (sample_input.jsonl: 217 words, 76.5 s). Setup ~6 min.
WORDS_PER_AUDIO_S = 2.8
N2_WORKER_S_PER_AUDIO_S = {1: 2.1, 2: 1.4}
SETUP_MIN = 6.0
GIB_PER_VCPU = {"highcpu": 1, "standard": 4, "highmem": 8}


def estimate_minutes(words: int, machine_type: str) -> float | None:
    """Expected VM wall minutes for a noise pass of `words` input words on an n2-* machine type,
    or None when the type is not n2-{highcpu,standard,highmem}-N. Rough (+-50%)."""
    parts = machine_type.split("-")
    if len(parts) != 3 or parts[0] != "n2" or parts[1] not in GIB_PER_VCPU or not parts[2].isdigit():
        return None
    cpus = int(parts[2])
    mem_gb = cpus * GIB_PER_VCPU[parts[1]] * 1.0737  # GiB -> GB, as total_mem_gb() reports it
    workers = auto_workers(1, max(RENDER_GB_PER_WORKER, ASR_GB_PER_WORKER), cpus, mem_gb)
    threads = max(1, cpus // workers)
    rate = N2_WORKER_S_PER_AUDIO_S[min(threads, 2)]
    audio_s = words / WORDS_PER_AUDIO_S
    return SETUP_MIN + audio_s * rate / workers / 60.0


def worker_env(threads: int) -> None:
    """Environment the spawned workers inherit. Thread caps must be set before numpy/torch/onnx
    load in the worker (numpy loads when the worker re-imports this module, before the pool
    initializer runs), and HF offline mode stops 40+ workers from re-checking the Hub for every
    voice (the parent has already downloaded every file they need)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["HF_HUB_OFFLINE"] = "1"


def batches(items: list, size: int):
    size = size if size > 0 else max(1, len(items))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_pool(fn, init, initargs, jobs: list[dict], workers: int, label: str):
    worker_env(initargs[0])
    ctx = mp.get_context("spawn")  # heavy libs load only in workers; spawn is safe on macOS and Linux
    t0 = time.perf_counter()
    n = 0
    with ctx.Pool(workers, initializer=init, initargs=initargs) as pool:
        for res in pool.imap(fn, jobs, chunksize=1):
            n += 1
            if n == 1 or n % 10 == 0 or n == len(jobs):
                print(f"[{label}] {n}/{len(jobs)} calls, {time.perf_counter() - t0:.0f}s", flush=True)
            yield res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", required=True, help="sherpa-onnx Nemotron 160ms int8 model dir")
    ap.add_argument("--work-dir", default=None, help="rendered audio (default: <output>.work/)")
    ap.add_argument("--stats", default=None, help="timing/WER summary JSON (default: <output>.stats.json)")
    ap.add_argument("--threads", type=int, default=1, help="torch/onnx threads per worker")
    ap.add_argument("--render-workers", type=int, default=0, help="0 = auto from CPUs and memory")
    ap.add_argument("--asr-workers", type=int, default=0, help="0 = auto from CPUs and memory")
    ap.add_argument("--noise-prob", type=float, default=0.5, help="share of turns that get background noise")
    ap.add_argument("--snr-min", type=float, default=15.0)
    ap.add_argument("--snr-max", type=float, default=30.0)
    ap.add_argument("--loss-prob", type=float, default=0.0, help="per-20ms-frame packet loss probability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N calls (0 = all)")
    ap.add_argument("--phase", choices=["all", "render", "asr"], default="all")
    ap.add_argument("--batch-calls", type=int, default=DEFAULT_BATCH_CALLS,
                    help="calls per render+ASR round; rows are saved after each round (0 = all in one round)")
    a = ap.parse_args(argv)

    missing = [f for f in ("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx")
               if not (Path(a.model_dir) / f).exists()]
    if missing and a.phase != "render":
        print(f"model dir {a.model_dir} is missing {missing}", file=sys.stderr)
        return 2

    rows, errors = nl.read_input(a.input)
    for e in errors:
        print(f"skipped input {e}", file=sys.stderr)
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        print("no valid input rows", file=sys.stderr)
        return 2

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(a.work_dir or f"{out}.work")
    work.mkdir(parents=True, exist_ok=True)
    done = nl.clean_partial_output(str(out), {r.call_id: len(r.turns) for r in rows})
    todo = [r for r in rows if r.call_id not in done]
    print(f"{len(rows)} calls ({len(errors)} bad lines skipped), {len(done)} already done, {len(todo)} to do", flush=True)

    jobs = [
        {"row": r.model_dump(), "work_dir": str(work), "seed": a.seed, "noise_prob": a.noise_prob,
         "snr": (a.snr_min, a.snr_max), "loss_prob": a.loss_prob}
        for r in todo
    ]
    stats: dict = {"calls_total": len(rows), "calls_done_before": len(done), "threads": a.threads,
                   "cpus": os.cpu_count(), "mem_gb": round(total_mem_gb(), 1)}

    if jobs and a.phase in ("all", "render"):
        from huggingface_hub import hf_hub_download

        espeak_data_path()  # fail fast here rather than in every worker

        # Fetch the weights once in the parent so workers don't race on the HF cache
        # (workers then run with HF_HUB_OFFLINE=1, see worker_env).
        for f in ["config.json", "kokoro-v1_0.pth"] + [f"voices/{v}.pt" for v in nl.ALL_VOICES]:
            hf_hub_download(repo_id=KOKORO_REPO, filename=f)

    rw, rth = plan_workers(a.render_workers, a.threads, RENDER_GB_PER_WORKER)
    aw, ath = plan_workers(a.asr_workers, a.threads, ASR_GB_PER_WORKER)
    render = {"workers": 0, "threads": rth, "wall_s": 0.0, "audio_s": 0.0, "worker_s": 0.0}
    asr = {"workers": 0, "threads": ath, "wall_s": 0.0, "audio_s": 0.0, "worker_s": 0.0, "turns": 0}
    pairs, per_call, failed = [], [], []
    n_batches = 0
    for batch in batches(jobs, a.batch_calls):
        n_batches += 1
        ready = batch
        if a.phase in ("all", "render"):
            w = min(rw, len(batch))
            render["workers"] = max(render["workers"], w)
            print(f"[render] round {n_batches}: {len(batch)} calls, {w} workers x {rth} threads", flush=True)
            t0 = time.perf_counter()
            metas = list(run_pool(render_call, init_render, (rth,), batch, w, "render"))
            render["wall_s"] += time.perf_counter() - t0
            bad = [m for m in metas if "error" in m]
            good = [m for m in metas if "error" not in m]
            for m in bad:
                print(f"FAILED {m['call_id']}: {m['error']}", file=sys.stderr, flush=True)
            failed += bad
            render["audio_s"] += sum(m["audio_s"] for m in good)
            render["worker_s"] += sum(m["render_s"] for m in good)
            ok_ids = {m["call_id"] for m in good}
            ready = [j for j in batch if j["row"]["call_id"] in ok_ids]

        if a.phase in ("all", "asr") and ready:
            w = min(aw, len(ready))
            asr["workers"] = max(asr["workers"], w)
            print(f"[asr] round {n_batches}: {len(ready)} calls, {w} workers x {ath} threads", flush=True)
            t0 = time.perf_counter()
            with open(out, "a") as f:
                for res in run_pool(asr_call, init_asr, (ath, a.model_dir), ready, w, "asr"):
                    if "error" in res:
                        print(f"FAILED {res['call_id']}: {res['error']}", file=sys.stderr, flush=True)
                        failed.append(res)
                        continue
                    f.write("\n".join(res["lines"]) + "\n")
                    f.flush()
                    pairs += res["pairs"]
                    asr["worker_s"] += res["asr_s"]
                    asr["audio_s"] += res["audio_s"]
                    asr["turns"] += len(res["lines"])
                    per_call.append({k: res[k] for k in ("call_id", "audio_s", "render_s", "asr_s",
                                                         "render_maxrss_mb", "asr_maxrss_mb")})
            asr["wall_s"] += time.perf_counter() - t0

    if jobs and a.phase in ("all", "render"):
        stats["render"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in render.items()}
    if jobs and a.phase in ("all", "asr"):
        stats["asr"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in asr.items()}
        stats["asr"]["peak_worker_rss_mb"] = max((p["asr_maxrss_mb"] for p in per_call), default=None)
        stats["render_peak_worker_rss_mb"] = max((p["render_maxrss_mb"] or 0 for p in per_call), default=None)
        stats["corpus_wer"] = nl.corpus_wer(pairs)
        stats["per_call"] = per_call
    stats["rounds"] = n_batches
    stats["failed"] = [{"call_id": x["call_id"], "error": x["error"][:300]} for x in failed]

    stats_path = Path(a.stats or f"{out}.stats.json")
    if jobs or not stats_path.exists():  # a no-op resume keeps the stats of the run that did the work
        stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_call"}, indent=2))
    if failed:
        print(f"{len(failed)} of {len(jobs)} calls FAILED (listed in {stats_path})", file=sys.stderr)
    # Non-zero only when nothing at all worked, so a few bad calls still deliver the rest.
    return 1 if jobs and len(failed) == len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
