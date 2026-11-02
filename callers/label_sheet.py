"""Blind labeling for V2 real calls (Numan records, Matt labels).

  python -m callers.label_sheet sheet --all                     # every recordings/<call_id>/ with turns.jsonl
  python -m callers.label_sheet sheet c_abc c_def               # specific calls
      -> recordings/to_label/<call_id>.csv + .md: hand Matt this folder only (no audio, events or source)
  python -m callers.label_sheet import c_abc --csv recordings/to_label/c_abc.csv --labeler matt
  python -m callers.label_sheet validate                        # every recordings/labels/*.json
  python -m callers.label_sheet validate --require-committed    # also: each file is committed, unchanged
  python -m callers.label_sheet match-sources                   # recordings <-> data/callers/calls.jsonl by time

The sheet is blind: it is built from turns.jsonl (the transcript) plus the guardian *inputs*
in meta.json (profile, rules, caller_known). It never reads events.jsonl (model outputs, alerts,
coach lines), meta.json "source" (which agent or card placed the call) or anything else.

Labels file, recordings/labels/<call_id>.json (committed before any model scores it, PLAN.md V2):
  {"call_id", "labeler", "labeled_at", "profile", "rules", "caller_known",
   "turns": [contracts.models.LabeledTurn, ...], "ask_turn": int|null, "notes": ""}
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contracts.models import Category, LabeledTurn, Profile, Risk, Stage, Turn

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"
SHEETS_DIR = "to_label"           # recordings/to_label/: the only folder handed to the labeler
CONTEXT_KEYS = ("profile", "rules", "caller_known")   # guardian inputs: fine to show the labeler
SHEET_COLUMNS = ["turn", "t_start_s", "speaker", "text", "risk", "category", "stage", "ask"]


class CallLabels(BaseModel):
    """One labeled real call. Built only from contracts.models types (proposed for contracts/)."""

    model_config = ConfigDict(extra="forbid")
    call_id: str
    labeler: str
    labeled_at: str = Field(description="YYYY-MM-DD")
    profile: Optional[Profile] = None
    rules: list[str] = Field(default_factory=list)
    caller_known: Optional[bool] = None
    turns: list[LabeledTurn]
    ask_turn: Optional[int] = None
    notes: str = ""


# ---------------------------------------------------------------- reading recordings
def read_turns(call_dir: Path) -> list[Turn]:
    path = call_dir / "turns.jsonl"
    return [Turn.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_context(call_dir: Path) -> dict:
    path = call_dir / "meta.json"
    if not path.exists():
        return {}
    ctx = json.loads(path.read_text(encoding="utf-8")).get("context", {}) or {}
    return {k: ctx[k] for k in CONTEXT_KEYS if k in ctx}


# ---------------------------------------------------------------- sheet
def make_sheet(call_dir: Path, out_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """Write <call_id>.csv (fill in) and <call_id>.md (context + allowed values) into
    out_dir (default recordings/to_label/), away from the call folder's events.jsonl and
    source.json, so the folder handed to the labeler holds nothing but blind sheets."""
    turns = read_turns(call_dir)
    ctx = read_context(call_dir)
    call_id = call_dir.name
    out_dir = out_dir or call_dir.parent / SHEETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{call_id}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLUMNS)
        for t in turns:
            w.writerow([t.turn, f"{t.t_start_ms / 1000:.1f}", t.speaker, t.text, "", "", "", ""])
    md = [
        f"# Label sheet: {call_id}",
        "",
        "Blind sheet: transcript only. Label each turn from what was said, before any model has scored this call.",
        "",
        f"- Protected person profile: {ctx.get('profile', 'unknown')}",
        f"- Caller known (on the allowed list): {ctx.get('caller_known', 'unknown')}",
        "- Caretaker's rules:" + ("" if ctx.get("rules") else " (none recorded)"),
        *[f"  - {r}" for r in ctx.get("rules", [])],
        "",
        f"Fill `{call_id}.csv`, one row per turn (only the risk/category/stage/ask columns; the text is not re-read):",
        f"- `risk`: {' | '.join(get_args(Risk))} (required)",
        f"- `category`: {' | '.join(get_args(Category))} (blank = none)",
        f"- `stage`: {' | '.join(get_args(Stage))} (blank allowed only when risk is none; it becomes hook)",
        "- `ask`: put `x` on the single turn where the caller makes the ask (money, code, address, meeting). Leave empty if there is none.",
        "",
        "Then: `python -m callers.label_sheet import " + call_id + " --csv recordings/" + SHEETS_DIR + "/" + call_id + ".csv --labeler <you>`",
        "",
        "| turn | t (s) | speaker | text |",
        "|---:|---:|---|---|",
        *[f"| {t.turn} | {t.t_start_ms / 1000:.1f} | {t.speaker} | {' '.join(t.text.replace('|', '/').split())} |" for t in turns],
        "",
    ]
    md_path = out_dir / f"{call_id}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return csv_path, md_path


# ---------------------------------------------------------------- import filled CSV
def labels_from_csv(call_id: str, csv_path: Path, labeler: str, context: dict,
                    recorded: Optional[list[Turn]] = None) -> CallLabels:
    """recorded: the call's turns.jsonl. When given, speaker and text come from it, not from the
    CSV, so a spreadsheet round trip (Excel BOM, smart quotes, trimmed spaces) can't alter them."""
    turns, ask_turns, errors = [], [], []
    by_turn = {t.turn: t for t in recorded} if recorded is not None else None
    with open(csv_path, newline="", encoding="utf-8-sig") as f:   # -sig: Excel writes a BOM
        rows = list(csv.DictReader(f))
    if rows and "turn" not in rows[0]:
        raise ValueError(f"{csv_path.name}: no 'turn' column (columns: {list(rows[0])})")
    seen = set()
    for row in rows:
        n = (row.get("turn") or "").strip()
        if not n.isdigit():
            errors.append(f"row with turn {n!r}: not a turn number")
            continue
        seen.add(int(n))
        speaker, text = (row.get("speaker") or "").strip(), row.get("text") or ""
        if by_turn is not None:
            if int(n) not in by_turn:
                errors.append(f"turn {n}: not in turns.jsonl")
                continue
            speaker, text = by_turn[int(n)].speaker, by_turn[int(n)].text
        risk = (row.get("risk") or "").strip().lower()
        category = (row.get("category") or "").strip() or "none"
        stage = (row.get("stage") or "").strip().lower()
        if not risk:
            errors.append(f"turn {n}: risk is empty")
            continue
        if not stage and risk == "none":
            stage = "hook"
        try:
            turns.append(LabeledTurn(turn=int(n), speaker=speaker, text=text,
                                     label={"risk": risk, "category": category, "stage": stage}))
        except ValidationError as e:
            errors.append(f"turn {n}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
        if (row.get("ask") or "").strip():
            ask_turns.append(int(n))
    if by_turn is not None and set(by_turn) - seen:
        errors.append(f"turns missing from the CSV: {sorted(set(by_turn) - seen)}")
    if len(ask_turns) > 1:
        errors.append(f"more than one ask turn marked: {ask_turns}")
    if errors:
        raise ValueError("; ".join(errors))
    return CallLabels(call_id=call_id, labeler=labeler, labeled_at=date.today().isoformat(),
                      profile=context.get("profile"), rules=context.get("rules", []),
                      caller_known=context.get("caller_known"), turns=turns,
                      ask_turn=ask_turns[0] if ask_turns else None)


# ---------------------------------------------------------------- validate
def validate_labels(path: Path, recordings: Path = RECORDINGS) -> list[str]:
    """Return a list of problems (empty = valid)."""
    try:
        labels = CallLabels.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as e:
        return [f"{path.name}: does not match the labels shape: {e}"]
    errs = []
    if path.stem != labels.call_id:
        errs.append(f"file name {path.stem} != call_id {labels.call_id}")
    nums = [t.turn for t in labels.turns]
    if not nums:
        errs.append("no turns")
    if len(nums) != len(set(nums)):
        errs.append("duplicate turn numbers")
    if nums != sorted(nums):
        errs.append("turns are not in order")
    if labels.ask_turn is not None:
        if labels.ask_turn not in nums:
            errs.append(f"ask_turn {labels.ask_turn} is not a turn in this call")
        elif next(t for t in labels.turns if t.turn == labels.ask_turn).label.stage != "ask":
            errs.append(f"ask_turn {labels.ask_turn} has stage != ask")
    for t in labels.turns:
        if t.label.risk != "none" and t.label.category == "none":
            errs.append(f"turn {t.turn}: risk {t.label.risk} needs a category")
    call_dir = recordings / labels.call_id
    if (call_dir / "turns.jsonl").exists():
        rec = {t.turn: t for t in read_turns(call_dir)}
        if set(rec) != set(nums):
            errs.append(f"turn numbers differ from turns.jsonl (labels {len(nums)}, recorded {len(rec)})")
        for t in labels.turns:
            r = rec.get(t.turn)
            if r and (r.speaker != t.speaker or r.text != t.text):
                errs.append(f"turn {t.turn}: speaker/text differ from turns.jsonl")
    return [f"{path.name}: {e}" for e in errs]


def match_sources(recordings: Path, ledger: Path, before_s: int = 180, after_s: int = 60) -> dict[str, dict]:
    """Pair each recording with the ledger row (data/callers/calls.jsonl) that started closest
    before it: a ledger row is written when a call is placed/dialed, the recording starts when the
    line answers. Writes recordings/<call_id>/source.json (never read by the label sheet)."""
    from datetime import datetime

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.exists() else []
    recs = []
    for d in sorted(recordings.iterdir()) if recordings.exists() else []:
        meta = d / "meta.json"
        if d.is_dir() and meta.exists():
            recs.append((d, datetime.fromisoformat(json.loads(meta.read_text())["started_at_utc"])))
    pairs = []
    for ri, (d, t_rec) in enumerate(recs):
        for li, row in enumerate(rows):
            dt = (t_rec - datetime.fromisoformat(row["started_utc"])).total_seconds()
            if -after_s <= dt <= before_s:
                pairs.append((abs(dt), ri, li))
    matched: dict[str, dict] = {}
    used_r, used_l = set(), set()
    for _, ri, li in sorted(pairs):                 # greedy, closest first, one-to-one
        if ri in used_r or li in used_l:
            continue
        used_r.add(ri)
        used_l.add(li)
        d = recs[ri][0]
        (d / "source.json").write_text(json.dumps(rows[li], indent=2), encoding="utf-8")
        matched[d.name] = rows[li]
    return matched


def git_committed_problems(path: Path) -> list[str]:
    """Read-only git: the file must be tracked and identical to HEAD."""
    def git(*a: str) -> int:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True).returncode

    try:
        rel = str(path.resolve().relative_to(ROOT))
    except ValueError:
        return [f"{path.name}: outside the repo, can't check it is committed"]
    if git("ls-files", "--error-unmatch", rel) != 0:
        return [f"{path.name}: not committed"]
    if git("diff", "--quiet", "HEAD", "--", rel) != 0:
        return [f"{path.name}: has uncommitted changes"]
    return []


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recordings", type=Path, default=RECORDINGS)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("call_ids", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--out", type=Path, default=None, help="default: <recordings>/to_label")
    i = sub.add_parser("import")
    i.add_argument("call_id")
    i.add_argument("--csv", type=Path, required=True)
    i.add_argument("--labeler", required=True)
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="*", type=Path)
    v.add_argument("--require-committed", action="store_true")
    m = sub.add_parser("match-sources")
    m.add_argument("--ledger", type=Path, default=ROOT / "data" / "callers" / "calls.jsonl")
    args = p.parse_args(argv)
    rec: Path = args.recordings

    if args.cmd == "match-sources":
        matched = match_sources(rec, args.ledger)
        for call_id, row in matched.items():
            print(f"{call_id}: {row.get('platform')} {row.get('agent') or row.get('card') or row.get('caller')}")
        print(f"matched {len(matched)} recording(s)")
        return 0

    if args.cmd == "sheet":
        dirs = ([d for d in sorted(rec.iterdir()) if (d / "turns.jsonl").exists()] if args.all and rec.exists()
                else [rec / c for c in args.call_ids])
        if not dirs:
            print(f"no recordings with turns.jsonl under {rec}")
            return 1
        for d in dirs:
            if not (d / "turns.jsonl").exists():
                print(f"{d}: no turns.jsonl")
                return 1
            csv_path, md_path = make_sheet(d, args.out)
            print(f"wrote {csv_path} and {md_path}")
        return 0

    if args.cmd == "import":
        call_dir = rec / args.call_id
        recorded = read_turns(call_dir) if (call_dir / "turns.jsonl").exists() else None
        try:
            labels = labels_from_csv(args.call_id, args.csv, args.labeler, read_context(call_dir), recorded)
        except ValueError as e:
            print(f"not imported: {e}")
            return 1
        out = rec / "labels" / f"{args.call_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(labels.model_dump_json(indent=2) + "\n", encoding="utf-8")
        problems = validate_labels(out, rec)
        print(f"wrote {out}" + ("" if not problems else "\n" + "\n".join(problems)))
        return 1 if problems else 0

    files = args.files or sorted((rec / "labels").glob("*.json"))
    problems: list[str] = []
    for f in files:
        problems += validate_labels(f, rec)
        if args.require_committed:
            problems += git_committed_problems(f)
    print("\n".join(problems) if problems else f"ok: {len(files)} label file(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
