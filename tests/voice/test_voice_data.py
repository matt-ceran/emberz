"""Manifest helpers and the data-prep parsers (ASVspoof 5, In-the-Wild, V2 calls table)."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf

from voice.data import asvspoof5, in_the_wild
from voice.data.calls_table import build as build_calls
from voice.manifest import parse_label, read_manifest, speaker_split, write_manifest

# Real lines from ASVspoof5.{train,eval.track_1}.tsv (record 14498691)
TRAIN_LINES = """T_4850 T_0000000000 F - - - AC3 A05 spoof -
T_0858 T_0000000001 M - - - AC3 A07 spoof -
T_1234 T_0000000002 M - - - - bonafide bonafide -
"""
EVAL_LINES = """E_1607 E_0009538969 M C05 2 E_0009486171 AC1 A26 spoof -
E_3614 E_0009249178 F C11 4 E_0006756001 AC1 A27 spoof -
E_2192 E_0004993854 F - 0 - - bonafide bonafide -
E_2193 E_0004993855 F C08 3 E_0004993854 - bonafide bonafide -
"""


def test_labels_and_manifest_roundtrip(tmp_path):
    assert [parse_label(v) for v in ("ai", "spoof", "1", "human", "bona-fide", "bonafide", 0)] == [1, 1, 1, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        parse_label("maybe")
    rows = [{"path": "a.wav", "label": 1, "dataset": "x", "condition": "none", "split": "train", "speaker": "s1"}]
    write_manifest(str(tmp_path / "m.csv"), rows)
    back = read_manifest(str(tmp_path / "m.csv"))
    assert back[0]["path"] == str(tmp_path / "a.wav") and back[0]["label"] == 1


def test_speaker_split_is_stable_and_roughly_balanced():
    spk = [f"spk{i}" for i in range(400)]
    s = [speaker_split(x) for x in spk]
    assert s == [speaker_split(x) for x in spk]
    assert 150 < s.count("test") < 250
    assert all(speaker_split(x, test_frac=1.0) == "test" for x in spk)


def test_asv5_parse_and_manifest(tmp_path):
    proto = tmp_path / "protocols"
    proto.mkdir()
    (proto / "ASVspoof5.train.tsv").write_text(TRAIN_LINES)
    (proto / "ASVspoof5.eval.track_1.tsv").write_text(EVAL_LINES)
    rows = asvspoof5.parse_protocol(proto / "ASVspoof5.eval.track_1.tsv")
    assert [asvspoof5.condition_of(r) for r in rows] == ["C05", "C11", "none", "C08"]
    assert [r["key"] for r in rows] == ["spoof", "spoof", "bonafide", "bonafide"]
    # only files on disk make it into the manifest
    for folder, name in [("flac_T", "T_0000000000"), ("flac_T", "T_0000000002"), ("flac_E_eval", "E_0009249178"),
                         ("flac_E_eval", "E_0004993855"), ("flac_E_eval", "E_0009538969")]:
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / f"{name}.flac").write_bytes(b"")
    m = asvspoof5.build_manifest(str(tmp_path), str(tmp_path / "m.csv"))
    got = {(r["split"], r["condition"], r["label"], r["speaker"]) for r in m}
    assert got == {("train", "none", 1, "T_4850"), ("train", "none", 0, "T_1234"),
                   ("test", "C11", 1, "E_3614"), ("test", "C08", 0, "E_2193")}  # C05 is not a default condition


def test_asv5_stream_extract_checks_md5(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        data = b"fLaC-fake"
        info = tarfile.TarInfo("flac_T/T_0000000000.flac")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    src = tmp_path / "src"
    src.mkdir()
    (src / "flac_T_aa.tar").write_bytes(buf.getvalue())
    monkeypatch.setattr(asvspoof5, "RECORD", f"file://{src}/" + "{name}")
    dest = tmp_path / "dest"
    dest.mkdir()
    asvspoof5.stream_extract("flac_T_aa.tar", hashlib.md5(buf.getvalue()).hexdigest(), dest)
    assert (dest / "flac_T" / "T_0000000000.flac").read_bytes() == b"fLaC-fake"
    assert (dest / ".flac_T_aa.tar.done").exists()
    dest2 = tmp_path / "dest2"
    dest2.mkdir()
    sleeps = []
    monkeypatch.setattr(asvspoof5.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="md5"):
        asvspoof5.stream_extract("flac_T_aa.tar", "0" * 32, dest2, retries=2)
    assert len(sleeps) == 1  # one retry, then the error surfaces
    assert not (dest2 / ".flac_T_aa.tar.done").exists()


def test_itw_meta_and_manifest(tmp_path):
    root = tmp_path / "release_in_the_wild"
    root.mkdir()
    (root / "meta.csv").write_text("file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Alec Guinness,bona-fide\n"
                                   "2.wav,Ayn Rand,bona-fide\n")
    m = in_the_wild.build_manifest(str(tmp_path), str(tmp_path / "m.csv"), test_frac=0.5)
    assert [r["label"] for r in m] == [1, 0, 0]
    assert m[0]["split"] == m[1]["split"]  # same speaker, same side
    assert m[0]["path"].endswith("release_in_the_wild/0.wav")


def test_calls_table_without_head(tmp_path):
    rec = tmp_path / "rec"
    (rec / "c_a").mkdir(parents=True)
    (rec / "c_b").mkdir(parents=True)
    (rec / "labels.csv").write_text("call_id,label,condition\nc_a,ai,signalwire\nc_b,human,human\n")
    turns = [
        {"call_id": "c_a", "turn": 1, "speaker": "protected", "text": "hello", "t_start_ms": 0, "t_end_ms": 900},
        {"call_id": "c_a", "turn": 2, "speaker": "caller", "text": "Hi, I'm an AI assistant from your bank",
         "t_start_ms": 2500, "t_end_ms": 5000, "gap_ms": 1600},
    ]
    (rec / "c_a" / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns) + "\n")
    ev = {"type": "voice", "call_id": "c_a", "at_ms": 9000,
          "data": {"call_id": "c_a", "at_ms": 9000, "ai_score": 0.77, "model": "telnyx"}}
    (rec / "c_a" / "events.jsonl").write_text(json.dumps(ev) + "\n")
    sf.write(rec / "c_b" / "caller.wav", np.zeros(8000, np.float32), 8000)
    build_calls(str(rec), None, str(tmp_path / "m.csv"), str(tmp_path / "t.csv"))
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0] == "call_id,label,condition,audio,timing,telnyx"
    a = lines[1].split(",")
    assert a[:3] == ["c_a", "1", "signalwire"] and float(a[4]) >= 0.95 and float(a[5]) == 0.77
    assert lines[2].split(",")[3:] == ["", "", ""]
    m = read_manifest(str(tmp_path / "m.csv"))
    assert [r["label"] for r in m] == [1, 0] and m[0]["dataset"] == "v2_calls"
