import csv
import json

import pytest

from callers import label_sheet as ls
from contracts.models import ConsoleEvent, Turn
from line.recorder import CallRecorder

TURNS = [
    ("caller", "Hello, this is the fraud team at your bank."),
    ("protected", "Oh, hello."),
    ("caller", "I just sent a code. Can you read me the six digits?"),
    ("protected", "It says 4 1 2, uh, 9 9 0."),
]


@pytest.fixture
def recording(tmp_path):
    rec = CallRecorder("c_lab1", tmp_path, t0=0.0, clock=lambda: 0.0,
                       numbers={"caller": "+15550001111"},
                       context={"profile": "elder", "rules": ["Never share codes"], "caller_known": False},
                       source={"kind": "vapi", "agent": "SOURCE_AGENT_SECRET"})
    for i, (spk, text) in enumerate(TURNS, 1):
        rec.add_turn(Turn(call_id="c_lab1", turn=i, speaker=spk, text=text, t_start_ms=i * 2000, t_end_ms=i * 2000 + 1500))
    rec.add_event(ConsoleEvent(type="risk", call_id="c_lab1", at_ms=6000,
                               data={"turn": 3, "response": {"risk": "alert", "coach_line": "MODEL_OUTPUT_SECRET"}}))
    rec.add_event(ConsoleEvent(type="coach", call_id="c_lab1", at_ms=6100, data={"text": "MODEL_OUTPUT_SECRET", "turn": 3}))
    rec.close()
    return tmp_path


def fill_csv(path, rows):
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    for row, (risk, cat, stage, ask) in zip(data, rows):
        row.update(risk=risk, category=cat, stage=stage, ask=ask)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ls.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(data)


GOOD = [("watch", "scam.impersonation_bank_gov", "hook", ""), ("none", "", "", ""),
        ("alert", "scam.impersonation_bank_gov", "ask", "x"), ("urgent", "disclosure.credentials_otp", "ask", "")]


def test_sheet_is_blind(recording):
    csv_path, md_path = ls.make_sheet(recording / "c_lab1")
    text = csv_path.read_text() + md_path.read_text()
    assert "MODEL_OUTPUT_SECRET" not in text and "SOURCE_AGENT_SECRET" not in text
    assert "Never share codes" in text and "elder" in text       # guardian inputs are shown
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["text"] for r in rows] == [t for _, t in TURNS]
    assert all(r["risk"] == "" for r in rows)


def test_import_and_validate_roundtrip(recording):
    csv_path, _ = ls.make_sheet(recording / "c_lab1")
    fill_csv(csv_path, GOOD)
    assert ls.main(["--recordings", str(recording), "import", "c_lab1", "--csv", str(csv_path), "--labeler", "matt"]) == 0
    out = recording / "labels" / "c_lab1.json"
    labels = ls.CallLabels.model_validate_json(out.read_text())
    assert labels.ask_turn == 3 and labels.profile == "elder" and labels.turns[1].label.stage == "hook"
    assert ls.validate_labels(out, recording) == []
    assert ls.main(["--recordings", str(recording), "validate"]) == 0


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d["turns"][0]["label"].update(category="scam.made_up"), "does not match"),
    (lambda d: d.update(ask_turn=2), "stage != ask"),
    (lambda d: d.update(ask_turn=9), "not a turn"),
    (lambda d: d["turns"][2].update(text="edited text"), "differ from turns.jsonl"),
    (lambda d: d["turns"][0]["label"].update(category="none"), "needs a category"),
    (lambda d: d.update(extra_field=1), "does not match"),
    (lambda d: d["turns"].pop(), "differ from turns.jsonl"),
])
def test_validator_rejects(recording, mutate, expect):
    csv_path, _ = ls.make_sheet(recording / "c_lab1")
    fill_csv(csv_path, GOOD)
    ls.main(["--recordings", str(recording), "import", "c_lab1", "--csv", str(csv_path), "--labeler", "matt"])
    out = recording / "labels" / "c_lab1.json"
    data = json.loads(out.read_text())
    mutate(data)
    out.write_text(json.dumps(data))
    problems = ls.validate_labels(out, recording)
    assert problems and any(expect in p for p in problems), problems


def test_import_rejects_unlabeled_and_double_ask(recording):
    csv_path, _ = ls.make_sheet(recording / "c_lab1")
    fill_csv(csv_path, [("", "", "", "")] + GOOD[1:])
    with pytest.raises(ValueError, match="risk is empty"):
        ls.labels_from_csv("c_lab1", csv_path, "matt", {})
    fill_csv(csv_path, [GOOD[0][:3] + ("x",)] + GOOD[1:])
    with pytest.raises(ValueError, match="more than one ask"):
        ls.labels_from_csv("c_lab1", csv_path, "matt", {})


def test_match_sources(recording, tmp_path):
    meta = json.loads((recording / "c_lab1" / "meta.json").read_text())
    from datetime import datetime, timedelta
    t = datetime.fromisoformat(meta["started_at_utc"])
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [
        {"platform": "vapi", "agent": "far_before", "started_utc": (t - timedelta(minutes=30)).isoformat()},
        {"platform": "roleplay", "card": "rp07", "started_utc": (t - timedelta(seconds=20)).isoformat()},
    ]))
    matched = ls.match_sources(recording, ledger)
    assert matched["c_lab1"]["card"] == "rp07"
    assert json.loads((recording / "c_lab1" / "source.json").read_text())["card"] == "rp07"


# ---------------------------------------------------------------- adversarial review additions
def test_sheets_go_to_a_handoff_folder_without_events_or_source(recording):
    (recording / "c_lab1" / "source.json").write_text('{"agent": "SOURCE_AGENT_SECRET"}')
    assert ls.main(["--recordings", str(recording), "sheet", "--all"]) == 0
    handoff = recording / ls.SHEETS_DIR
    assert sorted(p.name for p in handoff.iterdir()) == ["c_lab1.csv", "c_lab1.md"]
    for p in handoff.iterdir():
        text = p.read_text()
        assert "MODEL_OUTPUT_SECRET" not in text and "SOURCE_AGENT_SECRET" not in text
    # the handoff folder itself is not mistaken for a recording
    assert ls.main(["--recordings", str(recording), "sheet", "--all"]) == 0


def test_import_survives_excel_roundtrip(recording):
    """Excel adds a BOM and may 'fix' quotes/spaces: text must still come from turns.jsonl."""
    csv_path, _ = ls.make_sheet(recording / "c_lab1")
    fill_csv(csv_path, GOOD)
    raw = csv_path.read_text(encoding="utf-8").replace("Oh, hello.", "Oh,  hello ")
    csv_path.write_text("﻿" + raw, encoding="utf-8")
    assert ls.main(["--recordings", str(recording), "import", "c_lab1", "--csv", str(csv_path), "--labeler", "matt"]) == 0
    labels = ls.CallLabels.model_validate_json((recording / "labels" / "c_lab1.json").read_text())
    assert labels.turns[1].text == "Oh, hello."
    assert ls.validate_labels(recording / "labels" / "c_lab1.json", recording) == []


def test_import_reports_missing_and_bad_rows(recording, capsys):
    csv_path, _ = ls.make_sheet(recording / "c_lab1")
    fill_csv(csv_path, GOOD)
    lines = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join(lines[:-1] + ["total,,,,,,,"]) + "\n")    # last turn deleted, junk row added
    assert ls.main(["--recordings", str(recording), "import", "c_lab1", "--csv", str(csv_path), "--labeler", "matt"]) == 1
    out = capsys.readouterr().out
    assert "turns missing from the CSV: [4]" in out and "not a turn number" in out
    assert not (recording / "labels" / "c_lab1.json").exists()
