"""Build the human-replay clip set: ~60 clips of 6-12 s from >= 40 distinct real speakers.

  python -m callers.human_replay.build_set download            # LibriSpeech test-clean + EdAcc (<= ~1 GB)
  python -m callers.human_replay.build_set build               # select + write clips + manifest
  python -m callers.human_replay.build_set validate            # check manifest (+ audio files if present)

Outputs (data/ is gitignored):
  data/human_replay/clips/<clip_id>.wav     16 kHz (default) 16-bit mono PCM WAV, level-normalized
  data/human_replay/manifest.csv            one row per clip
  callers/human_replay/manifest.csv         the same rows, committed (no audio)

Selection (select_clips): one clip per speaker first; half the clips from each source; within each
source, half female and half male where the metadata allows; EdAcc speakers chosen round-robin
across accents. A second clip per speaker is used only if there are too few speakers.

EdAcc needs pyarrow, which the shared .venv lacks: run `download` from your own venv
(python -m venv /tmp/hr && /tmp/hr/bin/pip install pyarrow numpy soundfile soxr httpx), or use
--sources librispeech. `build` and `validate` need only numpy, soundfile and soxr.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from callers.human_replay import sources as src

MIN_S, MAX_S = 6.0, 12.0
DEFAULT_N = 60
MIN_SPEAKERS = 40
MANIFEST = src.DATA_DIR / "manifest.csv"
COMMITTED_MANIFEST = Path(__file__).resolve().parent / "manifest.csv"
CLIPS_DIR = src.DATA_DIR / "clips"
FIELDS = ["clip_id", "source", "speaker", "gender", "accent", "style", "license", "duration",
          "sample_rate", "path", "source_file", "trimmed", "sha256", "text"]
REQUIRED = ["clip_id", "source", "speaker", "license", "duration", "path"]


@dataclass(frozen=True)
class Candidate:
    source: str
    speaker: str
    gender: str              # female | male | unknown
    accent: str
    path: str
    duration: float          # of the source segment
    text: str = ""
    source_file: str = ""

    @property
    def clip_duration(self) -> float:
        """What we will play: segments longer than MAX_S are cut to MAX_S (at a quiet point)."""
        return min(self.duration, MAX_S)


def candidates_from(rows: Iterable[dict[str, Any]], min_s: float = MIN_S, max_trim_from: float = 30.0) -> list[Candidate]:
    """Usable segments: 6-12 s as they are, or 12-30 s segments that get cut to <= 12 s.
    Segments that are mostly a non-speech marker (e.g. EdAcc '<LAUGH>' only) are dropped."""
    out = []
    for r in rows:
        d = float(r["duration"])
        words = [w for w in str(r.get("text", "")).split() if not w.startswith("<")]
        if d < min_s or d > max_trim_from or len(words) < 6:
            continue
        out.append(Candidate(r["source"], r["speaker"], r.get("gender") or "unknown", r.get("accent", ""),
                             r["path"], d, r.get("text", ""), r.get("source_file", "")))
    return out


def select_clips(cands: list[Candidate], n: int = DEFAULT_N, seed: int = 7,
                 source_share: Optional[dict[str, float]] = None) -> list[Candidate]:
    """Pick n clips: distinct speakers first, per-source quota, gender balance inside each source,
    accent round-robin inside each source. Deterministic for a given seed and candidate list."""
    rng = random.Random(seed)
    by_source: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for c in sorted(cands, key=lambda c: (c.source, c.speaker, c.path)):
        by_source[c.source][c.speaker].append(c)
    if not by_source:
        return []
    srcs = sorted(by_source)
    share = source_share or {s: 1.0 / len(srcs) for s in srcs}
    # quotas: proportional, capped by what a source can give, the rest goes to the others
    capacity = {s: sum(len(v) for v in by_source[s].values()) for s in srcs}
    quota = {s: min(capacity[s], int(round(n * share.get(s, 0.0)))) for s in srcs}
    while sum(quota.values()) < n and any(quota[s] < capacity[s] for s in srcs):
        s = max((s for s in srcs if quota[s] < capacity[s]), key=lambda s: capacity[s] - quota[s])
        quota[s] += 1
    while sum(quota.values()) > n:
        s = max(srcs, key=lambda s: quota[s])
        quota[s] -= 1

    picked: list[Candidate] = []
    for s in srcs:
        picked.extend(_select_within_source(by_source[s], quota[s], rng))
    return _interleave(picked, rng)


def _best_clip(clips: list[Candidate], rng: random.Random, used: set[str]) -> Optional[Candidate]:
    free = [c for c in clips if c.path not in used]
    if not free:
        return None
    # prefer segments that need no cut and sit near the middle of the range, then random
    rng.shuffle(free)
    return min(free, key=lambda c: (c.duration > MAX_S, abs(c.duration - 9.0) > 2.5))


def _select_within_source(speakers: dict[str, list[Candidate]], k: int, rng: random.Random) -> list[Candidate]:
    if k <= 0:
        return []
    genders: dict[str, list[str]] = defaultdict(list)
    for spk, clips in speakers.items():
        genders[clips[0].gender].append(spk)
    for g in genders:
        rng.shuffle(genders[g])
        # accent round-robin: order speakers so consecutive picks cycle through accents
        genders[g] = _accent_round_robin(genders[g], speakers)
    order: list[str] = []
    f, m = genders.get("female", []), genders.get("male", [])
    for i in range(max(len(f), len(m))):
        if i < len(f):
            order.append(f[i])
        if i < len(m):
            order.append(m[i])
    order += genders.get("unknown", [])        # only after every known-gender speaker
    used: set[str] = set()
    out: list[Candidate] = []
    rounds = 0
    while len(out) < k and rounds < 50:
        progressed = False
        for spk in order:
            if len(out) >= k:
                break
            c = _best_clip(speakers[spk], rng, used)
            if c:
                out.append(c)
                used.add(c.path)
                progressed = True
        rounds += 1
        if not progressed:
            break
    return out


def _accent_round_robin(spks: list[str], speakers: dict[str, list[Candidate]]) -> list[str]:
    by_acc: dict[str, list[str]] = defaultdict(list)
    for s in spks:
        by_acc[speakers[s][0].accent].append(s)
    accents = sorted(by_acc, key=lambda a: (-len(by_acc[a]), a))
    out: list[str] = []
    while any(by_acc.values()):
        for a in accents:
            if by_acc[a]:
                out.append(by_acc[a].pop(0))
    return out


def _interleave(picked: list[Candidate], rng: random.Random) -> list[Candidate]:
    """Call order: alternate source and gender, so the first k calls of a partial run are mixed."""
    buckets: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in picked:
        buckets[(c.source, c.gender)].append(c)
    keys = sorted(buckets)
    out: list[Candidate] = []
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                out.append(buckets[key].pop(0))
    return out


def selection_stats(clips: list[Candidate] | list[dict[str, Any]]) -> dict[str, Any]:
    def get(c, k):
        return c[k] if isinstance(c, dict) else getattr(c, k)
    return {
        "clips": len(clips),
        "speakers": len({get(c, "speaker") for c in clips}),
        "by_source": dict(Counter(get(c, "source") for c in clips)),
        "by_gender": dict(Counter(get(c, "gender") for c in clips)),
        "accents": len({get(c, "accent") for c in clips if get(c, "accent")}),
    }


# --------------------------------------------------------------------------------------------
# audio

def render_clip(c: Candidate, dest: Path, rate: int = 16000, max_s: float = MAX_S) -> tuple[float, bool]:
    """Load, mono, resample (soxr VHQ), cut to <= max_s at a quiet 20 ms frame, remove DC,
    normalize active speech to -20 dBFS RMS with peaks <= -1 dBFS, 30 ms fades. Returns (duration, trimmed)."""
    import numpy as np
    import soundfile as sf
    import soxr

    x, sr = sf.read(c.path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != rate:
        x = soxr.resample(x, sr, rate, quality="VHQ").astype("float32")
    trimmed = False
    if len(x) > max_s * rate:
        x = x[: _quiet_cut(x, rate, max_s)]
        trimmed = True
    x = x - float(x.mean()) if len(x) else x
    frame = int(0.02 * rate)
    if len(x) >= frame:
        frames = x[: len(x) // frame * frame].reshape(-1, frame)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        active = rms[rms > max(1e-4, rms.max() * 10 ** (-35 / 20))]
        level = float(np.sqrt((active ** 2).mean())) if len(active) else float(rms.mean())
        if level > 0:
            x = x * (10 ** (-20 / 20) / level)
        peak = float(np.abs(x).max()) if len(x) else 0.0
        if peak > 10 ** (-1 / 20):
            x = x * (10 ** (-1 / 20) / peak)
    fade = min(int(0.03 * rate), len(x) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), x, rate, subtype="PCM_16", format="WAV")
    return round(len(x) / rate, 3), trimmed


def _quiet_cut(x, rate: int, max_s: float, search_s: float = 2.0) -> int:
    """Sample index <= max_s*rate at the quietest 20 ms frame in the last search_s seconds."""
    import numpy as np

    end = int(max_s * rate)
    start = max(0, end - int(search_s * rate))
    frame = int(0.02 * rate)
    seg = x[start:end]
    n = len(seg) // frame
    if n == 0:
        return end
    e = (seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    i = int(np.argmin(e))
    return start + i * frame + frame // 2


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# manifest

def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_manifest(path: Path = MANIFEST) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_manifest(rows: list[dict[str, str]], root: Path = src.ROOT, check_audio: bool = False,
                      min_speakers: int = MIN_SPEAKERS, min_s: float = MIN_S, max_s: float = MAX_S) -> list[str]:
    """Problems found, empty if fine."""
    errs: list[str] = []
    if not rows:
        return ["manifest is empty"]
    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        return [f"missing columns {missing_cols}"]
    ids = Counter(r["clip_id"] for r in rows)
    errs += [f"duplicate clip_id {k}" for k, v in ids.items() if v > 1]
    for r in rows:
        cid = r["clip_id"]
        if r["source"] not in src.SOURCES:
            errs.append(f"{cid}: unknown source {r['source']!r}")
        elif r["license"] != src.SOURCES[r["source"]].license:
            errs.append(f"{cid}: license {r['license']!r} != {src.SOURCES[r['source']].license!r}")
        try:
            d = float(r["duration"])
            if not (min_s - 0.05 <= d <= max_s + 0.05):
                errs.append(f"{cid}: duration {d} outside {min_s}-{max_s} s")
        except ValueError:
            errs.append(f"{cid}: duration {r['duration']!r} is not a number")
        if not r["speaker"] or not r["speaker"].startswith(r["source"] + ":"):
            errs.append(f"{cid}: speaker {r['speaker']!r} must be '<source>:<id>'")
        p = Path(r["path"])
        if p.is_absolute() or ".." in p.parts:
            errs.append(f"{cid}: path must be relative to the repo root: {r['path']}")
        if check_audio:
            full = root / p
            if not full.exists():
                errs.append(f"{cid}: missing audio {r['path']}")
            elif r.get("sha256") and sha256_of(full) != r["sha256"]:
                errs.append(f"{cid}: sha256 mismatch for {r['path']}")
    speakers = {r["speaker"] for r in rows}
    if len(speakers) < min_speakers:
        errs.append(f"only {len(speakers)} distinct speakers (need >= {min_speakers})")
    return errs


# --------------------------------------------------------------------------------------------
# CLI

def gather_candidates(sources: list[str]) -> list[Candidate]:
    rows: list[dict[str, Any]] = []
    if "librispeech" in sources:
        rows += list(src.libri_segments(src.libri_dir()))
    if "edacc" in sources:
        rows += list(src.edacc_segments())
    return candidates_from(rows)


def build(sources: list[str], n: int, seed: int, rate: int, out_manifest: Path = MANIFEST,
          committed: Optional[Path] = COMMITTED_MANIFEST, log=print) -> list[dict[str, Any]]:
    cands = gather_candidates(sources)
    if not cands:
        raise SystemExit("no candidates: run `download` first")
    have = {c.source for c in cands}
    chosen = select_clips(cands, n=n, seed=seed, source_share={s: 1 / len(have) for s in have})
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CLIPS_DIR.glob("hr*.wav"):
        old.unlink()
    rows = []
    for i, c in enumerate(chosen, 1):
        cid = f"hr{i:03d}"
        dest = CLIPS_DIR / f"{cid}.wav"
        dur, trimmed = render_clip(c, dest, rate=rate)
        rows.append({
            "clip_id": cid, "source": c.source, "speaker": c.speaker, "gender": c.gender, "accent": c.accent,
            "style": src.SOURCES[c.source].style, "license": src.SOURCES[c.source].license,
            "duration": f"{dur:.3f}", "sample_rate": rate, "path": str(dest.relative_to(src.ROOT)),
            "source_file": c.source_file, "trimmed": int(trimmed), "sha256": sha256_of(dest),
            "text": c.text if not trimmed else c.text + " [cut]",
        })
    errs = validate_manifest(rows, check_audio=True)
    write_manifest(rows, out_manifest)
    if committed:
        write_manifest(rows, committed)
    log(f"wrote {len(rows)} clips to {CLIPS_DIR}, manifest {out_manifest}")
    log(f"stats: {selection_stats(rows)}")
    for e in errs:
        log(f"PROBLEM: {e}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch the source audio into data/human_replay/raw/")
    d.add_argument("--sources", default="librispeech,edacc")
    d.add_argument("--edacc-max-mb", type=float, default=550, help="byte budget for EdAcc row groups")
    d.add_argument("--keep-archive", action="store_true")
    s = sub.add_parser("survey", help="EdAcc only: list conversations and the download plan (~2 MB read)")
    s.add_argument("--edacc-max-mb", type=float, default=550)
    b = sub.add_parser("build", help="select clips, write WAVs and the manifest")
    b.add_argument("--sources", default="librispeech,edacc")
    b.add_argument("--n", type=int, default=DEFAULT_N)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--rate", type=int, default=16000, choices=[8000, 16000])
    v = sub.add_parser("validate", help="check the manifest (and the audio, if present)")
    v.add_argument("--manifest", type=Path, default=MANIFEST)
    v.add_argument("--no-audio", action="store_true", help="skip the audio file checks (e.g. on a fresh clone)")
    args = p.parse_args(argv)

    if args.cmd == "download":
        wanted = [x.strip() for x in args.sources.split(",") if x.strip()]
        if "librispeech" in wanted:
            src.download_librispeech("test-clean", keep_archive=args.keep_archive)
        if "edacc" in wanted:
            src.download_edacc(max_mb=args.edacc_max_mb)
        print(f"data/human_replay is {src.disk_usage_mb(src.DATA_DIR):.0f} MB on disk")
        return 0
    if args.cmd == "survey":
        plan = src.plan_edacc_row_groups(src.survey_edacc(), int(args.edacc_max_mb * 1e6))
        for r in plan:
            spk = {s: m for s, m in r["speakers"].items() if src.conversation_of(s) == r["conversation"]}
            print(r["conversation"], f"{r['bytes'] / 1e6:.0f} MB", [(s[-1], m["gender"], m["accent"]) for s, m in spk.items()])
        print(f"{len(plan)} conversations, {sum(r['bytes'] for r in plan) / 1e6:.0f} MB")
        return 0
    if args.cmd == "build":
        build([x.strip() for x in args.sources.split(",") if x.strip()], args.n, args.seed, args.rate)
        return 0
    if args.cmd == "validate":
        rows = read_manifest(args.manifest)
        errs = validate_manifest(rows, check_audio=not args.no_audio)
        print(selection_stats(rows))
        for e in errs:
            print("PROBLEM:", e)
        print("ok" if not errs else f"{len(errs)} problem(s)")
        return 1 if errs else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
