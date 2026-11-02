"""V1 on the GCP L4 VM in one command: data -> manifests -> features -> head -> eval -> results.

What the G1 launcher needs to know:
- VM: g2-standard-8 (1x L4), >= 150 GB disk (ASVspoof 5 subset ~23 GB extracted,
  In-the-Wild 8.2 GB zip + ~9 GB unzipped, HF cache 2.5 GB, features ~2 GB).
- Packages: pip install -r voice/requirements-gcp.txt (torch with CUDA comes from the
  image or the PyTorch index); `apt-get install unzip` (and espeak-ng if Kokoro renders run).
- Run from the repo root:  python -m voice.pipeline --data /data --out /data/voice_out
  Each step skips work already done, so a restarted spot VM resumes.
- Copy back to the Mac only --out minus feats/ (heads, results, eval tables, score CSVs:
  a few MB). Optionally feats/ too (~1-2 GB) to retrain heads on the Mac.

Steps (each can be run alone with --steps):
  download   ASVspoof 5 parts (default T_aa, D_aa, E_aa) + In-the-Wild, in parallel
  manifests  data/manifests/{asv5,itw}.csv (+ kokoro.csv if --kokoro-calls)
  features   wav2vec2 XLS-R 300M features, layers 3,6,9,12,24, windows 2 s and 4 s, phone path on
  head       train on asv5 train + itw train speakers (+ kokoro train voices); layer/C by asv5 dev EER
  eval       per test set, by condition and window -> eval/*.json + results/voice_results.jsonl
  aasist     W2V2-AASIST baseline on the same test clips (needs onnxruntime-gpu), --aasist-limit per set
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STEPS = ["download", "manifests", "features", "head", "eval", "aasist"]


def log(msg):
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="big disk for audio")
    ap.add_argument("--out", required=True, help="small outputs (+ feats/)")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--asv-parts", default="T_aa,D_aa,E_aa")
    ap.add_argument("--skip-itw", action="store_true")
    ap.add_argument("--itw-test-frac", type=float, default=0.5)
    ap.add_argument("--kokoro-calls", default=None, help="DatasetRow jsonl to render with Kokoro (optional)")
    ap.add_argument("--kokoro-max-calls", type=int, default=300)
    ap.add_argument("--layers", default="3,6,9,12,24")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--aasist-limit", type=int, default=2000)
    a = ap.parse_args(argv)

    steps = a.steps.split(",")
    data, out = Path(a.data), Path(a.out)
    man = data / "manifests"
    sets = {"asv5": man / "asv5.csv"}
    if not a.skip_itw:
        sets["itw"] = man / "itw.csv"
    if a.kokoro_calls:
        sets["kokoro"] = man / "kokoro.csv"

    if "download" in steps:
        from voice.data import asvspoof5, in_the_wild

        log("download start")
        with ThreadPoolExecutor(2) as ex:
            jobs = [ex.submit(asvspoof5.download, str(data / "asv5"), a.asv_parts.split(","))]
            if not a.skip_itw:
                jobs.append(ex.submit(in_the_wild.download, str(data / "itw")))
            for j in jobs:
                j.result()
        log("download done")

    if "manifests" in steps:
        from voice.data import asvspoof5, in_the_wild

        asvspoof5.build_manifest(str(data / "asv5"), str(sets["asv5"]))
        if "itw" in sets:
            in_the_wild.build_manifest(str(data / "itw"), str(sets["itw"]), a.itw_test_frac)
        if a.kokoro_calls:
            from voice.data.kokoro_render import render

            render(a.kokoro_calls, str(data / "kokoro"), str(sets["kokoro"]), a.kokoro_max_calls)
        log("manifests done")

    if "features" in steps:
        from voice.features import extract

        for name, m in sets.items():
            log(f"features {name}")
            extract(str(m), str(out / "feats" / name), layers=[int(x) for x in a.layers.split(",")],
                    num_workers=a.num_workers, batch_size=a.batch_size, device="auto")
        log("features done")

    head = out / "heads" / "w2v-head-v1.joblib"
    if "head" in steps:
        from voice.train_head import train

        train([str(out / "feats" / n) for n in sets], str(head))
        log(f"head saved {head}")

    if "eval" in steps:
        from voice import eval as veval

        res = out / "results" / "voice_results.jsonl"
        (out / "eval").mkdir(parents=True, exist_ok=True)
        for name in sets:
            for w in ("2", "4"):  # the live detector's two checkpoints, reported separately
                veval.main(["head", "--head", str(head), "--shards", str(out / "feats" / name), "--split", "test",
                            "--windows", w, "--by", "condition", "--test-set", f"{name}_test_phone_{w}s",
                            "--results", str(res), "--json-out", str(out / "eval" / f"{name}_{w}s.json"),
                            "--scores-out", str(out / "eval" / f"{name}_{w}s_scores.csv")])
        log(f"eval done -> {res}")

    if "aasist" in steps:
        from voice import eval as veval
        from voice.baselines import w2v2_aasist
        from voice.manifest import read_manifest, write_manifest

        res = out / "results" / "voice_results.jsonl"
        for name, m in sets.items():
            rows = [r for r in read_manifest(str(m)) if r["split"] == "test"]
            random.Random(0).shuffle(rows)
            sub = man / f"{name}_aasist.csv"
            write_manifest(str(sub), rows[: a.aasist_limit])
            sc = out / "eval" / f"{name}_aasist_scores.csv"
            w2v2_aasist.main(["--manifest", str(sub), "--out", str(sc), "--batch-size", "16"])
            veval.main(["scores", "--scores", str(sc), "--model", "w2v2-aasist", "--test-set", f"{name}_test_phone_4s",
                        "--results", str(res), "--json-out", str(out / "eval" / f"{name}_aasist.json")])
        log("aasist done")


if __name__ == "__main__":
    main()
