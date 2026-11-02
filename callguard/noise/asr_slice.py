"""Local G1-style noise pass on held-out test calls: macOS TTS -> 8 kHz phone audio -> Nemotron Speech Streaming ASR.

Writes (gitignored data/):
  data/calls_test_clean50.jsonl   the chosen held-out calls, original text
  data/calls_test_asr50.jsonl     same calls, every turn's text replaced by Nemotron's transcript
  data/noisy_turns_test50.jsonl   NoisyTurn rows (call_id, turn, speaker, text_clean, text_asr, voice, wer)
Run with the sherpa venv:  <venv>/bin/python callguard/noise/asr_slice.py --n 50 --workers 6
"""
import argparse, json, os, re, subprocess, sys, tempfile, wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

CG = Path(__file__).resolve().parents[1]
MODEL = Path("/private/tmp/claude-501/-Users-mattceran-Documents-fireworksamd/db79b7b9-1daf-471f-bbea-48fddd2891bb/scratchpad/models/nemotron160p/x/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25")
VOICES_ADULT = ["Samantha", "Alex", "Daniel", "Karen", "Moira", "Tessa", "Fred", "Victoria", "Ralph", "Kathy"]
_rec = None


def recognizer():
    global _rec
    if _rec is None:
        import sherpa_onnx
        _rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(MODEL / "tokens.txt"), encoder=str(MODEL / "encoder.int8.onnx"), decoder=str(MODEL / "decoder.int8.onnx"),
            joiner=str(MODEL / "joiner.int8.onnx"), num_threads=2, sample_rate=16000, feature_dim=80)
    return _rec


def available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = {l.split()[0] for l in out.splitlines() if "en_" in l}
    return [v for v in VOICES_ADULT if v in names] or sorted(names)[:6]


def tts_8k(text, voice, tmp):
    aiff, wav = os.path.join(tmp, "t.aiff"), os.path.join(tmp, "t.wav")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True, capture_output=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@8000", "-c", "1", aiff, wav], check=True, capture_output=True)
    with wave.open(wav) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return a


def asr(samples):
    rec = recognizer()
    s = rec.create_stream()
    step = 1280  # 160 ms at 8 kHz
    for i in range(0, len(samples), step):
        s.accept_waveform(8000, samples[i:i + step])
        while rec.is_ready(s):
            rec.decode_stream(s)
    s.accept_waveform(8000, np.zeros(int(0.8 * 8000), dtype=np.float32))
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    r = rec.get_result(s)
    return (r if isinstance(r, str) else getattr(r, "text", str(r))).strip()


def wer(ref, hyp):
    norm = lambda x: re.sub(r"[^a-z0-9' ]", " ", x.lower()).split()
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def do_call(args):
    row, voices = args
    k = sum(map(ord, row["call_id"]))
    vc, vp = voices[k % len(voices)], voices[(k // 7 + 1) % len(voices)]
    if vp == vc:
        vp = voices[(voices.index(vc) + 1) % len(voices)]
    noisy, new_turns = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for t in row["turns"]:
            v = vc if t["speaker"] == "caller" else vp
            text = re.sub(r"\*[^*]+\*", " ", t["text"]).replace("[SLUR]", "slur").strip() or t["text"]
            try:
                hyp = asr(tts_8k(text, v, tmp)) or ""
            except Exception as e:  # noqa: BLE001
                hyp = ""
            # ASR is case-insensitive but outputs punctuation/casing; keep as produced
            noisy.append({"call_id": row["call_id"], "turn": t["turn"], "speaker": t["speaker"], "text_clean": t["text"],
                          "text_asr": hyp, "voice": f"macos:{v}", "wer": round(wer(t["text"], hyp), 3)})
            nt = dict(t); nt["text"] = hyp if hyp else t["text"]
            new_turns.append(nt)
    asr_row = dict(row); asr_row["turns"] = new_turns
    return row, asr_row, noisy


def pick(rows, n):
    keep, seen = [], set()
    for r in rows:
        base = r.get("rule_flip_of") or r["call_id"]
        if base in seen:
            continue
        group = [x for x in rows if (x.get("rule_flip_of") or x["call_id"]) == base]
        if len(keep) + len(group) > n:
            break
        keep += group; seen.add(base)
    return keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (CG / "data" / "calls_test_heldout.jsonl").read_text().splitlines() if l.strip()]
    sel = pick(rows, a.n)
    voices = available_voices()
    print(f"{len(sel)} calls, {sum(len(r['turns']) for r in sel)} turns, voices={voices}", flush=True)
    clean, asr_rows, noisy = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (c, ar, nz) in enumerate(ex.map(do_call, [(r, voices) for r in sel]), 1):
            clean.append(c); asr_rows.append(ar); noisy += nz
            if i % 10 == 0:
                print(f"  {i}/{len(sel)} calls", flush=True)
    d = CG / "data"
    (d / "calls_test_clean50.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean))
    (d / "calls_test_asr50.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in asr_rows))
    (d / "noisy_turns_test50.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in noisy))
    w = [x["wer"] for x in noisy]
    print(f"DONE: {len(clean)} calls, {len(noisy)} turns, mean WER {sum(w)/len(w):.3f}, median {sorted(w)[len(w)//2]:.3f}, "
          f"empty ASR turns {sum(1 for x in noisy if not x['text_asr'])}")
    for x in sorted(noisy, key=lambda x: -x["wer"])[:5]:
        print("  worst:", x["wer"], "|", x["text_clean"][:80], "=>", x["text_asr"][:80])


if __name__ == "__main__":
    main()
