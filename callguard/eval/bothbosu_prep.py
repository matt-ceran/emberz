"""BothBosu/Scammer-Conversation (Apache-2.0, synthetic, out-of-distribution) -> our test-row format.

  python3 -I callguard/eval/bothbosu_prep.py <conv.csv> [--n-per-class 100]
Only call-level labels exist, so scam calls get one gold 'alert' on the last turn (counts for recall_any)
and benign calls are all 'none' (counts for false_alarm_rate). catch_before_ask does not apply (no ask turn).
"""
import argparse, csv, hashlib, json, random, re, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "bothbosu_test.jsonl"
NONE = dict(risk="none", category="none", stage="none", source="none", evidence="", coach_line="", caretaker_note="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n-per-class", type=int, default=100)
    a = ap.parse_args()
    csv.field_size_limit(10**8)
    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
    seen, by = set(), {"0": [], "1": []}
    for r in rows:
        conv = re.sub(r"\s+", " ", r["conversation"]).strip()
        key = re.sub(r"good (morning|afternoon|day|evening)", "", conv.lower())
        if key in seen:
            continue
        seen.add(key)
        parts = re.split(r"(Person [AB]):", conv)
        turns = []
        for i in range(1, len(parts) - 1, 2):
            text = parts[i + 1].strip()
            if text:
                turns.append({"turn": len(turns) + 1, "speaker": "caller" if parts[i] == "Person A" else "protected",
                              "text": text, "label": dict(NONE)})
        if len(turns) >= 3:
            by[r["label"]].append(turns)
    rng = random.Random(5)
    out = []
    for lab, convs in by.items():
        rng.shuffle(convs)
        for turns in convs[: a.n_per_class]:
            scam = lab == "1"
            if scam:
                turns[-1]["label"] = dict(NONE, risk="alert", category="scam.impersonation_bank_gov", source="caller",
                                          evidence=turns[-1]["text"])
            cid = "bb_" + hashlib.sha1(json.dumps(turns).encode()).hexdigest()[:10]
            out.append(dict(call_id=cid, split="bothbosu", generator="BothBosu", seed_id="bothbosu_scam" if scam else "bothbosu_benign",
                            family="scam" if scam else "benign", heldout_type=False, profile="adult", rules=[],
                            caller_known=False, ai_voice=None, rule_flip_of=None, flip_kind=None,
                            benign_twin=not scam, ask_turn=None, turns=turns))
    rng.shuffle(out)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"unique convs: scam={len(by['1'])} benign={len(by['0'])}; wrote {len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
