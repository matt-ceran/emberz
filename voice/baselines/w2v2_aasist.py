"""Baseline: W2V2-AASIST (Tak et al., Odyssey 2022; MIT) scored on a manifest.

Model card: https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST
We use the repo's ONNX export (w2v2-aasist.onnx, 1.26 GB): input `wav` float32
(batch, 64600) = 4.04 s at 16 kHz, output `logits` (batch, 2), index 1 = bona fide.
Windowing follows the card: first 64,600 samples, tile-repeat if shorter, no
normalisation. It was trained on ASVspoof 2019 LA (clean 16 kHz), so on phone audio it
is out of domain; that is the point of the comparison.

Output CSV (works with `python -m voice.eval scores`):
  id,label,dataset,condition,score,bonafide_logit     score = P(AI) = softmax(logits)[0]

Needs onnxruntime (CPU/CoreML on the Mac, `onnxruntime-gpu` on the L4). The shared repo
venv does not have it; on the Mac run it with the voice venv:
  PYTHONPATH=. <scratchpad>/venvs/voice/bin/python -m voice.baselines.w2v2_aasist \
      --manifest m.csv --out aasist_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from voice.manifest import read_manifest
from voice.phone import MODEL_SR, phone_degrade, resample, to_mono_float

REPO = "SpeechAntiSpoofingBenchmarks/W2V2-AASIST"
ONNX_FILE = "w2v2-aasist.onnx"
N_SAMPLES = 64600


def pad_fixed(x: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Same as SSL_Anti-spoofing data_utils_SSL.pad: first n samples, tiled if shorter."""
    if len(x) >= n:
        return x[:n]
    reps = n // max(len(x), 1) + 1
    return np.tile(x, reps)[:n]


class W2V2AASIST:
    def __init__(self, onnx_path: str | None = None, providers=None):
        import onnxruntime as ort

        if onnx_path is None:
            from huggingface_hub import hf_hub_download

            onnx_path = hf_hub_download(REPO, ONNX_FILE)
        avail = ort.get_available_providers()
        providers = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input = self.sess.get_inputs()[0].name

    def logits(self, wavs16: list[np.ndarray]) -> np.ndarray:
        x = np.stack([pad_fixed(np.asarray(w, dtype=np.float32)) for w in wavs16])
        return self.sess.run(None, {self.input: x})[0]

    @staticmethod
    def p_ai(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e[:, 0] / e.sum(axis=1)


def load16(path: str, phone: bool) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    return phone_degrade(wav, sr) if phone else resample(to_mono_float(wav), sr, MODEL_SR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default=None, help="path to w2v2-aasist.onnx (default: download from the Hub)")
    ap.add_argument("--no-phone", action="store_true", help="score the raw clip instead of the phone path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default=None, help="only rows with this split (e.g. test)")
    a = ap.parse_args(argv)

    rows = [r for r in read_manifest(a.manifest, root=a.root) if not a.split or r["split"] == a.split][: a.limit]
    model = W2V2AASIST(a.onnx)
    t0 = time.time()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "dataset", "condition", "score", "bonafide_logit"])
        for s in range(0, len(rows), a.batch_size):
            batch, wavs = [], []
            for r in rows[s: s + a.batch_size]:
                try:
                    wavs.append(load16(r["path"], not a.no_phone))
                    batch.append(r)
                except Exception as e:
                    print(f"[aasist] skip {r['path']}: {e}", file=sys.stderr)
            if not batch:
                continue
            lg = model.logits(wavs)
            for r, l, p in zip(batch, lg, model.p_ai(lg)):
                w.writerow([r["path"], r["label"], r["dataset"], r["condition"], f"{p:.6f}", f"{l[1]:.4f}"])
    print(f"[aasist] {len(rows)} clips in {time.time() - t0:.1f}s -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
