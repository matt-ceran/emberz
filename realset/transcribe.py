"""Real audio -> our phone path -> live ASR -> data/realset/calls.jsonl.

Each channel: resample to 8 kHz, G.711 mu-law (line/audio.py), 20 ms frames fed into
line.asr.CallTranscriber (Nemotron streaming, the live code) faster than real time.
Robocalls: whole recording on the caller (inbound) track, digital silence on protected.
HarperValley: agent channel = caller (inbound), customer channel = protected (outbound).

    python -m realset.transcribe --workers 6

batch v2 (2026-11-02) was run on the v2 rows only, 4 shards x 2 workers x 1 ONNX thread:
    python -m realset.transcribe --manifest data/realset/manifest_v2_todo.csv \
        --out data/realset/calls_v2.jsonl --shards 4 --shard $i --workers 2 --threads 1
then its part files were appended to calls.jsonl and `--merge` rewrote it in manifest order.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

AUDIO = Path("data/realset/raw/audio")


def _frames(path: Path) -> list[bytes]:
    from line.audio import wav_to_mulaw_frames
    return wav_to_mulaw_frames(str(path))


def _init():
    from line.asr import load_recognizer
    load_recognizer(num_threads=1)


def run_one(row: dict) -> dict:
    from line.asr import CallTranscriber
    from line.audio import FRAME_BYTES, MULAW_SILENCE

    cid = row["call_id"]
    if row["source"] == "ncsu_robocall":
        inbound = _frames(AUDIO / f"{cid}.wav")
        outbound: list[bytes] = []
    else:
        inbound = _frames(AUDIO / f"{cid}_agent.wav")
        outbound = _frames(AUDIO / f"{cid}_customer.wav")
    sil = bytes([MULAW_SILENCE]) * FRAME_BYTES
    n = max(len(inbound), len(outbound))
    tr = CallTranscriber(cid, on_turn=lambda t: None)
    t0 = time.perf_counter()
    for i in range(n):
        if not outbound:  # one-sided robocall: the protected track would be digital silence, which
            tr.feed_round([("inbound", inbound[i], None)])  # yields no tokens; skip decoding it (2x faster)
            continue
        tr.feed_round([("inbound", inbound[i] if i < len(inbound) else sil, None),
                       ("outbound", outbound[i] if i < len(outbound) else sil, None)])
    tr.finish()
    return {
        "call_id": cid, "source": row["source"], "license": row["license"], "label": row["label"],
        "scam_type": row["scam_type"], "batch": row.get("batch", "v1"), "split": row.get("split", ""), "audio_s": round(n * 0.02, 2), "asr_wall_s": round(time.perf_counter() - t0, 2),
        "turns": [t.model_dump() for t in tr.turns],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="realset/manifest.csv")
    ap.add_argument("--out", default="data/realset/calls.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="only merge part files into --out")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    if a.merge:
        merge(rows, Path(a.out)); return
    out = Path(a.out)
    done = set()
    for p in [out, *out.parent.glob(out.stem + ".part*.jsonl")]:
        if p.exists():
            for l in open(p):
                try:
                    done.add(json.loads(l)["call_id"])
                except ValueError:
                    pass
    todo = [r for r in rows if r["call_id"] not in done]
    import random
    random.Random(1).shuffle(todo)  # interleave scam and benign so partial runs stay balanced
    if a.shards > 1:  # several processes (one model copy each), each on every k-th call
        todo = todo[a.shard :: a.shards]
        out = out.parent / f"{out.stem}.part{a.shard}.jsonl"
    print(f"{len(rows)} calls, {len(todo)} to do", flush=True)
    # One process, one shared recognizer (the live server shares it across calls too), one
    # thread per concurrent call: sherpa/onnxruntime release the GIL while decoding.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from line.asr import load_recognizer
    load_recognizer(num_threads=a.threads)
    with ThreadPoolExecutor(a.workers) as pool, open(out, "a") as f:
        futs = [pool.submit(run_one, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i}/{len(todo)}] {rec['call_id']} {rec['audio_s']}s audio, {len(rec['turns'])} turns, {rec['asr_wall_s']}s", flush=True)
    merge(rows, Path(a.out))


def merge(rows, out: Path):
    """Merge calls.jsonl + calls.part*.jsonl into calls.jsonl in manifest order.

    label/scam_type/batch/split are taken from the manifest (the turns are never touched), so a
    row always carries the manifest's current split; call_ids not in the manifest are dropped."""
    recs = {}
    for p in [out, *sorted(out.parent.glob(out.stem + ".part*.jsonl"))]:
        if p.exists():
            for l in open(p):
                try:
                    recs[json.loads(l)["call_id"]] = l
                except ValueError:
                    pass
    with open(out, "w") as f:
        for r in rows:
            if r["call_id"] in recs:
                rec = json.loads(recs[r["call_id"]])
                meta = {k: r[k] for k in ("label", "scam_type", "batch", "split") if k in r}
                keys = list(rec)
                rec.update(meta)
                order = keys[:keys.index("scam_type") + 1] + [k for k in ("batch", "split") if k in meta]
                rec = {**{k: rec[k] for k in order}, **{k: v for k, v in rec.items() if k not in order}}
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
