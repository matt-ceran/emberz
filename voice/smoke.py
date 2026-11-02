"""Local end-to-end smoke run on a tiny set. THE NUMBERS ARE MEANINGLESS: a handful of
macOS `say` voices vs a couple of LibriSpeech speakers, far too few clips and speakers,
and `say` is an old concatenative/parametric TTS. It only proves the pipeline runs:
phone_degrade -> speech gate -> wav2vec2 features -> head -> eval -> live detector.

  python -m voice.smoke --work <scratch>/smoke --human-dir <scratch>/smoke/human

Human clips: put folders of clips under --human-dir as <speaker>/<clip>.(wav|flac).
macOS only (uses `say`).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SAY_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Fred", "Albert", "Rishi", "Tessa",
              "Eddy (English (US))", "Flo (English (UK))", "Reed (English (US))", "Shelley (English (UK))"]
LINES = [
    "Hello, this is the fraud department at your bank calling about a suspicious charge on your card.",
    "Before we continue, can you please confirm the six digit code we just sent to your phone?",
    "Hi grandma, it's me, I'm in a bit of trouble and I need you to help me out right away.",
    "Your appointment with doctor Patel is scheduled for Tuesday at three in the afternoon.",
    "We noticed your computer has a virus, please go to your desk so I can walk you through the fix.",
    "Congratulations, you have won a free cruise, all we need is a small processing fee today.",
    "Is your mom or dad at home right now? What street do you live on?",
    "This is a reminder that your prescription is ready for pickup at the pharmacy.",
]


def make_say_clips(out: Path) -> None:
    for v in SAY_VOICES:
        d = out / re.sub(r"[^a-z0-9]+", "_", v.lower()).strip("_")
        d.mkdir(parents=True, exist_ok=True)
        for i, line in enumerate(LINES):
            f = d / f"line{i}.wav"
            if not f.exists():
                subprocess.run(["say", "-v", v, "-o", str(f), "--file-format=WAVE", "--data-format=LEI16@22050", line],
                               check=True)


def build_manifest(work: Path, human_dir: Path) -> Path:
    from voice.data.from_dir import build
    from voice.manifest import write_manifest

    ai_dir = work / "ai_say"
    make_say_clips(ai_dir)
    ai_speakers = sorted(p.name for p in ai_dir.iterdir() if p.is_dir())
    ai = build(str(ai_dir), "ai", "macsay", "none", "parent", test_speakers=ai_speakers[1::2])
    hu = build(str(human_dir), "human", "librispeech", "none", "parent")
    # Only ~3 human speakers are available offline, so human clips split by utterance
    # (every third clip to test) plus the sherpa test_wavs speaker entirely in test.
    for i, r in enumerate(hu):
        r["split"] = "test" if (i % 3 == 0 or r["speaker"] == "sherpa_test") else "train"
    m = work / "manifest.csv"
    write_manifest(str(m), ai + hu)
    n = lambda rows, s: sum(r["split"] == s for r in rows)
    print(f"[smoke] manifest: ai train/test {n(ai, 'train')}/{n(ai, 'test')}, "
          f"human train/test {n(hu, 'train')}/{n(hu, 'test')}", file=sys.stderr)
    return m


def live_demo(head_path: str, clip_path: str) -> None:
    import soundfile as sf

    from voice.live import LiveVoiceDetector, Scorer
    from voice.phone import PHONE_SR, mulaw_encode, resample, to_mono_float

    x, sr = sf.read(clip_path, dtype="float32")
    x8 = resample(to_mono_float(x), sr, PHONE_SR)
    x8 = np.concatenate([x8, x8])  # make sure there are >4 s of speech
    codes = mulaw_encode(x8).tobytes()
    got = []
    det = LiveVoiceDetector("c_smoke", got.append, scorer=Scorer(head_path))
    t0 = time.perf_counter()
    for i in range(0, len(codes), 160):  # 20 ms Telnyx frames
        det.push_mulaw(codes[i: i + 160])
    for f in det.pending:
        f.result()
    print(f"[smoke] live detector on {Path(clip_path).name}: {[s.model_dump() for s in got]} "
          f"(fed {len(codes) / PHONE_SR:.1f} s of audio, wall {1000 * (time.perf_counter() - t0):.0f} ms, model already loaded)",
          file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--human-dir", required=True)
    ap.add_argument("--layers", default="3,6,9,12")
    ap.add_argument("--layer", default="6")
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    m = build_manifest(work, Path(a.human_dir))

    from voice import eval as veval
    from voice import features, train_head

    t0 = time.time()
    features.extract(str(m), str(work / "feats"), layers=[int(x) for x in a.layers.split(",")],
                     num_workers=0, shard_size=1000)
    print(f"[smoke] features in {time.time() - t0:.1f}s", file=sys.stderr)
    head_path = str(work / "head" / "w2v-head-smoke.joblib")
    train_head.train([str(work / "feats")], head_path, (2.0, 4.0), a.layer, (0.1,))
    veval.main(["head", "--head", head_path, "--shards", str(work / "feats"), "--split", "test", "--windows", "2,4", "--by", "window",
                "--test-set", "smoke_say_vs_libri", "--n-boot", "200", "--scores-out", str(work / "head_scores.csv"),
                "--results", str(work / "voice_results_smoke.jsonl")])
    print("\n[smoke] NOTE: numbers above are meaningless (tiny set, few speakers, old TTS).", file=sys.stderr)
    from voice.manifest import read_manifest

    rows = read_manifest(str(m))
    for label in (1, 0):  # one held-out AI clip, one held-out human clip
        r = next(r for r in rows if r["split"] == "test" and r["label"] == label)
        live_demo(head_path, r["path"])


if __name__ == "__main__":
    main()
