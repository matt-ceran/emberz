"""Frozen wav2vec2 front end: a clip's first 2-4 s of speech -> one fixed-size vector.

Choice (short version, sources below):
- Checkpoint: facebook/wav2vec2-xls-r-300m (Apache-2.0, 1.27 GB). Pascu et al. 2024 train a
  logistic regression on *frozen* wav2vec2 features and show multilingual XLS-R models
  generalise far better than wav2vec2-base (mean EER over ASV19/ITW/TIM/FoR: base 36.3%,
  xls-r-300m 16.2%, xls-r-2b 6.5%). 2B is too slow for live use on a laptop and 1B is
  4 GB to download; 300M is the largest that fits the live budget, and it is also the
  front end inside the W2V2-AASIST baseline, so the comparison is like for like.
- Layer: early/middle transformer layers. El Kheir et al. 2025 (layer-wise analysis of
  SSL models for deepfake detection) find lower layers the most discriminative and that
  truncated models (first ~10-12 layers of a large model) match the full model. Pascu
  used only the last layer and did not compare layers. So we store several layers
  (default 3, 6, 9, 12, 24) during bulk extraction, pick the layer on dev data, and at
  live time run only up to the chosen layer (truncation makes live inference cheaper).
- Pooling: mean + standard deviation over time (2 x 1024 = 2048 dims per layer). Pascu
  used the mean; std adds a cheap second-order statistic and is standard in speaker /
  anti-spoofing back ends (x-vector style statistics pooling).
- Silence: only gated speech frames are used (SpeechGate below), so leading/trailing
  silence, a known shortcut in ASVspoof data (Mueller et al. 2021, "Speech is Silver,
  Silence is Golden"), cannot be learned; live audio is gated by the same code.

Sources:
  Pascu et al., Interspeech 2024: https://arxiv.org/abs/2309.05384 (Table 3)
  El Kheir et al., NAACL Findings 2025: https://arxiv.org/abs/2502.03559
  Mueller et al. 2021: https://arxiv.org/abs/2106.12914
  Model: https://huggingface.co/facebook/wav2vec2-xls-r-300m

CLI:
  python -m voice.features extract --manifest m.csv --out-dir feats/ [--device cuda]
  python -m voice.features latency --seconds 3 --layers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from voice.manifest import read_manifest
from voice.phone import MODEL_SR, phone_degrade

DEFAULT_CKPT = "facebook/wav2vec2-xls-r-300m"
DEFAULT_LAYERS = (3, 6, 9, 12, 24)
DEFAULT_WINDOWS = (2.0, 4.0)
FEATURE_VERSION = "w2v-feat-v1"

FRAME = 320                 # 20 ms at 16 kHz
GATE_DBFS = -45.0           # frame RMS above this counts as speech
GATE_HANGOVER = 10          # keep 200 ms after speech so word endings are not clipped
MIN_SPEECH_S = 0.5          # less gated speech than this: fall back to the raw clip
MIN_CLIP_S = 0.25           # shorter clips are skipped (XLS-R's conv front end needs >= 400 samples)


# --------------------------------------------------------------------------- speech gate
class SpeechGate:
    """Stateful frame-energy gate: feed audio in any chunk sizes, get speech samples out.

    Offline (`gate_speech`) and live (voice/live.py) use this same class, so the detector
    sees identically selected audio in training and on calls.
    """

    def __init__(self, dbfs: float = GATE_DBFS, hangover: int = GATE_HANGOVER):
        self.thr = 10.0 ** (dbfs / 20.0)
        self.hangover = hangover
        self._pending = np.zeros(0, dtype=np.float32)
        self._hang = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._pending, np.asarray(x, dtype=np.float32)])
        n = len(x) // FRAME
        self._pending = x[n * FRAME:]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        frames = x[: n * FRAME].reshape(n, FRAME)
        loud = np.sqrt(np.mean(frames ** 2, axis=1)) > self.thr
        keep = np.zeros(n, dtype=bool)
        for i in range(n):
            if loud[i]:
                self._hang = self.hangover
                keep[i] = True
            elif self._hang > 0:
                self._hang -= 1
                keep[i] = True
        return frames[keep].reshape(-1)


def gate_speech(wav16: np.ndarray) -> np.ndarray:
    return SpeechGate().process(wav16)


def speech_windows(wav16: np.ndarray, windows=DEFAULT_WINDOWS) -> tuple[dict[float, np.ndarray], float]:
    """Gate, then take the first `w` seconds for each window. Returns ({w: audio}, speech_seconds)."""
    if len(wav16) < MIN_CLIP_S * MODEL_SR:
        raise ValueError(f"clip too short ({len(wav16)} samples at 16 kHz)")
    sp = gate_speech(wav16)
    if len(sp) < MIN_SPEECH_S * MODEL_SR:
        sp = wav16  # nearly silent or very quiet clip: use it as is
    return {w: sp[: int(w * MODEL_SR)] for w in windows}, len(sp) / MODEL_SR


# --------------------------------------------------------------------------- model
def pick_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class W2VFeaturizer:
    """Frozen wav2vec2, truncated after the deepest requested layer, mean+std pooled."""

    def __init__(self, ckpt: str = DEFAULT_CKPT, layers=DEFAULT_LAYERS, device: str = "auto"):
        import torch
        from transformers import Wav2Vec2Model

        self.torch = torch
        self.ckpt = ckpt
        self.layers = tuple(sorted(int(l) for l in layers))
        self.device = pick_device(device)
        model = Wav2Vec2Model.from_pretrained(ckpt)
        n_total = len(model.encoder.layers)
        if self.layers[0] < 1 or self.layers[-1] > n_total:
            raise ValueError(f"layers must be in 1..{n_total}")
        # Run only up to the deepest layer we read. Layer k's output is captured by a hook
        # *before* the encoder's final layer norm, so it is identical with or without truncation.
        model.encoder.layers = model.encoder.layers[: self.layers[-1]]
        self._captured: dict[int, object] = {}
        for k in self.layers:
            model.encoder.layers[k - 1].register_forward_hook(self._hook(k))
        self.model = model.eval().to(self.device)
        self.dim = 2 * model.config.hidden_size

    def _hook(self, k):
        def fn(_module, _inp, out):
            self._captured[k] = out[0] if isinstance(out, (tuple, list)) else out
        return fn

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).to(self.device)
        # Wav2Vec2FeatureExtractor(do_normalize=True): zero mean, unit variance per clip.
        x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        self._captured.clear()
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    self.model(x)
            else:
                self.model(x)
        out = []
        for k in self.layers:
            h = self._captured[k].float()  # (B, T, H)
            out.append(torch.cat([h.mean(dim=1), h.std(dim=1, unbiased=False)], dim=-1))
        return torch.stack(out, dim=1).cpu().numpy()  # (B, n_layers, 2H)

    def embed(self, wavs: list[np.ndarray], max_batch: int = 32) -> np.ndarray:
        """16 kHz float32 clips -> (N, n_layers, 2H) float32. Equal-length clips are batched."""
        res: list[np.ndarray | None] = [None] * len(wavs)
        by_len: dict[int, list[int]] = {}
        wavs = [w if len(w) >= 400 else np.pad(np.asarray(w, np.float32), (0, 400 - len(w))) for w in wavs]
        for i, w in enumerate(wavs):  # (clips under 25 ms are zero-padded: the conv front end needs 400 samples)
            by_len.setdefault(len(w), []).append(i)
        for n, idx in by_len.items():
            for s in range(0, len(idx), max_batch):
                chunk = idx[s: s + max_batch]
                feats = self._forward(np.stack([wavs[i] for i in chunk]).astype(np.float32))
                for j, i in enumerate(chunk):
                    res[i] = feats[j]
        return np.stack(res) if res else np.zeros((0, len(self.layers), self.dim), np.float32)


# --------------------------------------------------------------------------- batch extraction
def _load_clip(path: str, phone: bool) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if phone:
        return phone_degrade(wav, sr)
    from voice.phone import resample, to_mono_float

    return resample(to_mono_float(wav), sr, MODEL_SR)


def _as_list(batch):
    return batch


class _ClipDataset:
    """Torch-style dataset so decoding + phone path run in DataLoader workers."""

    def __init__(self, rows, phone: bool, windows):
        self.rows, self.phone, self.windows = rows, phone, windows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            wins, speech_s = speech_windows(_load_clip(r["path"], self.phone), self.windows)
            return i, wins, speech_s, ""
        except Exception as e:  # unreadable file: logged and skipped
            return i, None, 0.0, f"{type(e).__name__}: {e}"


def extract(manifest: str, out_dir: str, ckpt=DEFAULT_CKPT, layers=DEFAULT_LAYERS, windows=DEFAULT_WINDOWS,
            phone: bool = True, device: str = "auto", shard_size: int = 2000, num_workers: int = 4,
            batch_size: int = 32, root: str | None = None, limit: int | None = None) -> list[str]:
    """Manifest -> out_dir/shard_00000.npz ... Resumable: existing shards are skipped."""
    import torch
    from torch.utils.data import DataLoader

    rows = read_manifest(manifest, root=root)
    if limit:
        rows = rows[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feat = None
    written = []
    for s in range(0, len(rows), shard_size):
        shard_path = out / f"shard_{s // shard_size:05d}.npz"
        if shard_path.exists():
            written.append(str(shard_path))
            continue
        if feat is None:
            feat = W2VFeaturizer(ckpt, layers, device)
            print(f"[features] {ckpt} layers={feat.layers} on {feat.device}", file=sys.stderr)
        part = rows[s: s + shard_size]
        loader = DataLoader(_ClipDataset(part, phone, windows), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_as_list)
        keep, speech, feats = [], [], {w: [] for w in windows}
        t0 = time.time()
        for batch in loader:
            ok = [b for b in batch if b[1] is not None]
            for i, _, _, err in batch:
                if err:
                    print(f"[features] skip {part[i]['path']}: {err}", file=sys.stderr)
            if not ok:
                continue
            for w in windows:
                feats[w].append(feat.embed([b[1][w] for b in ok], max_batch=batch_size).astype(np.float16))
            keep += [b[0] for b in ok]
            speech += [b[2] for b in ok]
        meta = dict(ckpt=ckpt, layers=list(feat.layers), windows=list(windows), phone=phone,
                    version=FEATURE_VERSION, manifest=os.path.abspath(manifest))
        cols = {c: np.array([part[i].get(c, "") for i in keep]) for c in ("path", "dataset", "condition", "split", "speaker")}
        tmp = shard_path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            label=np.array([part[i]["label"] for i in keep], dtype=np.int8),
            speech_s=np.array(speech, dtype=np.float32),
            meta=np.array(json.dumps(meta)),
            **cols,
            **{f"feats_w{w:g}": (np.concatenate(feats[w]) if feats[w] else
                                  np.zeros((0, len(feat.layers), feat.dim), np.float16)) for w in windows},
        )
        os.replace(tmp, shard_path)
        written.append(str(shard_path))
        print(f"[features] {shard_path.name}: {len(keep)}/{len(part)} clips in {time.time() - t0:.1f}s", file=sys.stderr)
    return written


def load_shards(paths_or_dir, window: float, layer: int):
    """Shards -> (X float32 (N, 2H), meta columns dict). `layer` must be one of the stored layers."""
    paths = paths_or_dir
    if isinstance(paths_or_dir, (str, Path)):
        p = Path(paths_or_dir)
        paths = sorted(p.glob("shard_*.npz")) if p.is_dir() else [p]
    xs, cols = [], {c: [] for c in ("label", "path", "dataset", "condition", "split", "speaker", "speech_s")}
    for f in paths:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        li = meta["layers"].index(int(layer))
        xs.append(z[f"feats_w{window:g}"][:, li, :].astype(np.float32))
        for c in cols:
            cols[c].append(z[c])
    if not xs:
        raise FileNotFoundError(f"no shards in {paths_or_dir}")
    return np.concatenate(xs), {c: np.concatenate(v) for c, v in cols.items()}


# --------------------------------------------------------------------------- latency
def measure_latency(seconds: float = 3.0, layers=(6,), device: str = "auto", runs: int = 20, ckpt=DEFAULT_CKPT) -> dict:
    feat = W2VFeaturizer(ckpt, layers, device)
    wav = (np.random.default_rng(0).standard_normal(int(seconds * MODEL_SR)) * 0.05).astype(np.float32)
    for _ in range(3):
        feat.embed([wav])
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        feat.embed([wav])
        if feat.device == "mps":
            feat.torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return dict(device=feat.device, layers=list(feat.layers), seconds=seconds, runs=runs,
                p50_ms=round(float(np.median(ts)), 1), p95_ms=round(float(np.percentile(ts, 95)), 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--root", default=None, help="base dir for relative paths (default: the manifest's dir)")
    e.add_argument("--ckpt", default=DEFAULT_CKPT)
    e.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    e.add_argument("--windows", default=",".join(f"{w:g}" for w in DEFAULT_WINDOWS))
    e.add_argument("--no-phone", action="store_true", help="skip phone_degrade (not recommended)")
    e.add_argument("--device", default="auto")
    e.add_argument("--shard-size", type=int, default=2000)
    e.add_argument("--num-workers", type=int, default=4)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--limit", type=int, default=None)
    l = sub.add_parser("latency")
    l.add_argument("--seconds", type=float, default=3.0)
    l.add_argument("--layers", default="6")
    l.add_argument("--device", default="auto")
    l.add_argument("--runs", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        extract(a.manifest, a.out_dir, a.ckpt, [int(x) for x in a.layers.split(",")],
                [float(x) for x in a.windows.split(",")], not a.no_phone, a.device, a.shard_size,
                a.num_workers, a.batch_size, a.root, a.limit)
    else:
        print(json.dumps(measure_latency(a.seconds, [int(x) for x in a.layers.split(",")], a.device, a.runs)))


if __name__ == "__main__":
    main()
