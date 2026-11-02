"""G1 noise pass: G.711 mu-law, WER, voice choice, row building and validation, resume."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gcloud" / "noise_pass")]

import noise_lib as nl  # noqa: E402
from contracts.models import DatasetRow, NoisyTurn  # noqa: E402

SAMPLE = ROOT / "gcloud" / "noise_pass" / "sample_input.jsonl"


def sample_rows():
    return [DatasetRow.model_validate_json(l) for l in SAMPLE.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ mu-law
def test_ulaw_known_values():
    # Reference values of the classic Sun g711.c / audioop.lin2ulaw.
    assert nl.lin2ulaw(np.array([0]))[0] == 0xFF
    assert nl.lin2ulaw(np.array([32767]))[0] == 0x80
    assert nl.lin2ulaw(np.array([-32768]))[0] == 0x00
    assert nl.ulaw2lin(np.array([0xFF]))[0] == 0
    assert nl.ulaw2lin(np.array([0x80]))[0] == 32124
    assert nl.ulaw2lin(np.array([0x00]))[0] == -32124


def test_ulaw_roundtrip_error_is_small_and_relative():
    x = np.linspace(-32000, 32000, 4001).astype(np.int16)
    y = nl.ulaw2lin(nl.lin2ulaw(x)).astype(np.int32)
    err = np.abs(y - x.astype(np.int32))
    # mu-law error grows with amplitude: <= ~1/16 of the value plus the smallest step
    assert np.all(err <= np.abs(x.astype(np.int32)) / 16 + 16)
    assert np.all(np.sign(y[np.abs(x) > 100]) == np.sign(x[np.abs(x) > 100]))


def test_ulaw_encode_decode_is_identity_on_all_codes():
    codes = np.arange(256, dtype=np.uint8)
    back = nl.lin2ulaw(nl.ulaw2lin(codes))
    # encode(decode(c)) == c for every code except 0x7F ("negative zero"), which decodes to 0 -> 0xFF
    assert set(codes[back != codes].tolist()) == {0x7F}


# ------------------------------------------------------------------ WER
@pytest.mark.parametrize(
    "ref,hyp,expected",
    [
        ("read me the code", "read me the code", 0.0),
        ("Read me the code.", "read me the code", 0.0),
        ("read me the code", "read the code", 0.25),
        ("read me the code", "read me the cold now", 0.5),
        ("we open at nine a.m.", "We open at nine AM", 0.0),
        ("don't hang up", "dont hang up", 1 / 3),
        ("it's twenty-five", "it's twenty five", 0.0),
        ("", "anything", None),
        ("hello", "", 1.0),
    ],
)
def test_wer(ref, hyp, expected):
    got = nl.wer(ref, hyp)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_corpus_wer_weights_by_reference_length():
    pairs = [("a b c d", "a b c d"), ("x y", "x z")]
    assert nl.corpus_wer(pairs) == pytest.approx(1 / 6)
    assert nl.corpus_wer([]) is None


# ------------------------------------------------------------------ voices
def test_voices_differ_and_are_stable():
    for profile in ("kid", "elder", "adult"):
        for i in range(50):
            v = nl.pick_voices(f"c{i}", profile)
            assert v["caller"] != v["protected"]
            assert v["protected"] in nl.PROTECTED_VOICES[profile]
            assert v["caller"] in nl.CALLER_VOICES
            assert v == nl.pick_voices(f"c{i}", profile)
    assert {nl.pick_voices(f"c{i}", "adult")["caller"] for i in range(200)} == set(nl.CALLER_VOICES)


def test_voice_ids_are_kokoro_american_or_british():
    for v in nl.ALL_VOICES:
        assert v[:2] in ("af", "am", "bf", "bm") and nl.lang_code(v) in "ab"


def test_kid_speaks_faster_than_elder():
    kid = [nl.speaker_speed(f"c{i}", "protected", "kid") for i in range(30)]
    elder = [nl.speaker_speed(f"c{i}", "protected", "elder") for i in range(30)]
    assert min(kid) > max(elder)


# ------------------------------------------------------------------ audio helpers
def test_set_level_and_noise_snr():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 400 * np.pi, 8000)).astype(np.float32)
    y = nl.set_level(x, -26.0)
    assert 20 * np.log10(np.sqrt(np.mean(y**2))) == pytest.approx(-26.0, abs=0.01)
    z = nl.add_noise(y, 10.0, rng, exponent=1.0)
    noise = z - y
    snr = 10 * np.log10(np.mean(y**2) / np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=0.1)


def test_silence_is_left_alone():
    rng = np.random.default_rng(0)
    z = np.zeros(800, dtype=np.float32)
    assert np.array_equal(nl.add_noise(nl.set_level(z, -26), 10, rng), z)


def test_packet_loss_zeroes_whole_frames():
    rng = np.random.default_rng(1)
    x = np.ones(160 * 100, dtype=np.float32)
    y = nl.packet_loss(x, 0.2, rng)
    frames = y.reshape(100, 160)
    assert all(f.min() == f.max() for f in frames)  # each frame fully kept or fully lost
    assert 5 < int((frames[:, 0] == 0).sum()) < 40
    assert np.array_equal(nl.packet_loss(x, 0.0, rng), x)


# ------------------------------------------------------------------ rows
def test_sample_input_is_valid_and_realistic():
    rows = sample_rows()
    assert len(rows) == 3
    assert {r.profile for r in rows} == {"elder", "kid", "adult"}
    for r in rows:
        assert [t.turn for t in r.turns] == list(range(1, len(r.turns) + 1))
        assert {t.speaker for t in r.turns} == {"caller", "protected"}
        assert all(3 <= len(t.text.split()) <= 30 for t in r.turns)
        if r.ask_turn is not None:
            assert r.turns[r.ask_turn - 1].label.stage == "ask"


def test_build_noisy_rows_one_valid_row_per_turn():
    row = sample_rows()[0]
    voices = nl.pick_voices(row.call_id, row.profile)
    asr = {t.turn: t.text.lower().rstrip(".?") for t in row.turns}
    asr[3] = "we flagged a charge"  # a lossy transcript
    del asr[8]  # a turn the ASR returned nothing for
    out = nl.build_noisy_rows(row, asr, voices)
    assert [o.turn for o in out] == [t.turn for t in row.turns]
    for o, t in zip(out, row.turns):
        assert isinstance(o, NoisyTurn)
        NoisyTurn.model_validate_json(o.model_dump_json())  # round-trips through the contract
        assert o.call_id == row.call_id and o.speaker == t.speaker and o.text_clean == t.text
        assert o.voice == voices[t.speaker]
    assert out[0].wer == 0.0
    assert out[2].wer > 0.5
    assert out[7].text_asr == "" and out[7].wer == 1.0


def test_read_input_skips_bad_lines(tmp_path):
    good = SAMPLE.read_text().splitlines()
    bad_schema = json.loads(good[0])
    bad_schema["profile"] = "teenager"
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join([good[0], "{not json", json.dumps(bad_schema), good[1], good[0], "", good[2]]) + "\n")
    rows, errors = nl.read_input(str(p))
    assert [r.call_id for r in rows] == ["g1_sample_001", "g1_sample_002", "g1_sample_003"]
    assert len(errors) == 3 and "duplicate" in errors[2]


def test_clean_partial_output_keeps_only_complete_calls(tmp_path):
    rows = sample_rows()
    lines = []
    for r in rows[:2]:
        v = nl.pick_voices(r.call_id, r.profile)
        lines += [x.model_dump_json() for x in nl.build_noisy_rows(r, {}, v)]
    # call 2 was cut off mid-write: drop its last row and truncate the one before
    n1 = len(rows[0].turns)
    lines = lines[: n1 + len(rows[1].turns) - 1]
    lines[-1] = lines[-1][:20]
    out = tmp_path / "out.jsonl"
    out.write_text("\n".join(lines))
    done = nl.clean_partial_output(str(out), {r.call_id: len(r.turns) for r in rows})
    assert done == {rows[0].call_id}
    kept = out.read_text().splitlines()
    assert len(kept) == n1 and all(json.loads(l)["call_id"] == rows[0].call_id for l in kept)
    assert nl.clean_partial_output(str(tmp_path / "missing.jsonl"), {}) == set()


def test_safe_name():
    assert nl.safe_name("s_000123") == "s_000123"
    assert nl.safe_name("../x y/z") == ".._x_y_z"
