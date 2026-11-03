"""Seeded selection of the real-audio test set (realset). Writes realset/manifest.csv.

Scam: NCSU Robocall Audio Dataset (public domain), English, >= 25 words, stratified by a
keyword scam type read from the Whisper transcript in metadata.csv. Calls whose type is
"other" (no fraud keyword) are skipped so every scam row is a recognizable fraud pitch.
Benign: HarperValleyBank (CC BY 4.0), seeded sample stratified by task_type.

batch v1 (111 calls) is the original pick and is reproduced unchanged. batch v2 appends more
calls: NCSU calls not in v1 (same filters, seed SEED+1, QUOTA_V2) and HarperValley calls from a
second seeded metadata sample (raw/hv/sids_v2.txt -> raw/hv/metadata_v2, disjoint from v1).
split: seeded 50/50 dev/test over ALL calls, stratified by (label, scam_type). Tune on dev,
report on test.

    python -m realset.select_calls --raw data/realset/raw
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 20261102
NCSU_RAW = "https://raw.githubusercontent.com/wspr-ncsu/robocall-audio-dataset/main/"
HV_RAW = "https://raw.githubusercontent.com/cricketclub/gridspace-stanford-harper-valley/master/data/"

# order matters: first match wins
SCAM_TYPES = [
    ("gov_ssa_irs", r"social security|\bssa\b|\birs\b|internal revenue|treasury|federal|legal action|arrest|warrant\b|lawsuit|police|sheriff|department of|customs|border protection|officer|legal department|legal enforcement|fraudulent activities|ftc|registration office"),
    ("amazon_apple_order", r"amazon|apple|iphone|order (has been |was )?placed|pre-?authori[sz]ed order|your order|purchase"),
    ("delivery_parcel", r"fedex|ups|parcel|package|delivery"),
    ("auto_warranty", r"warrant(y|ies)|vehicle|car'?s|auto(motive)? (service|protection)"),
    ("bank_credit_card", r"credit card|interest rate|visa|mastercard|bank account|card member|debt|loan|credit score|chase|wells fargo|fargo|discover|cash app|rate reduction|bank of america"),
    ("tech_support_utility", r"microsoft|windows|computer|virus|tech(nical)? support|electric|utility|disconnect|internet service|delinquent|disjoin|disconnection|spectrum|direct ?tv|xfinity|comcast|50% (off|discount)|subscription|renewal|geek squad|norton|mcafee|paypal"),
    ("prize_travel_health", r"congratulations|winner|won|free|prize|gift|vacation|cruise|hotel|rewards|medicare|insurance|brace|health"),
]
# Hand review of the picked transcripts (2026-11-02): the keyword bucket mis-files these. Selection
# still uses the keyword bucket (so the pick is unchanged); only the reported scam_type is corrected.
TYPE_FIX = {
    "ncsu_500579": "tech_support_utility",  # BGE power disconnection ("disconnection department")
    "ncsu_520019": "tech_support_utility",  # electricity disconnection
    "ncsu_567525": "tech_support_utility",  # PSE&G power disconnection
    "ncsu_1631035650.66108_in_AKA_Management_Robocall__Computer-Hijack": "tech_support_utility",  # "Microsoft Help Center"
    "ncsu_974387": "gov_ssa_irs",  # "suspended your social insurance number"
    "ncsu_274460": "gov_ssa_irs",  # "COVID-19 relief payment", "state assigned phone number"
}
QUOTA_V2 = {"gov_ssa_irs": 28, "amazon_apple_order": 25, "auto_warranty": 13, "bank_credit_card": 24,
            "tech_support_utility": 22, "prize_travel_health": 8, "delivery_parcel": 0}  # 120
# Hand review of the v2 picks (2026-11-02), same rule as TYPE_FIX: only obvious keyword mis-files.
TYPE_FIX_V2 = {
    "ncsu_557776": "tech_support_utility",  # TECO Energy "disconnection department"
    "ncsu_520018": "tech_support_utility",  # Delmarva Power disconnection
    "ncsu_567528": "tech_support_utility",  # electricity disconnection
    "ncsu_409294": "tech_support_utility",  # PSE&G disconnection
    "ncsu_s6diZGlyX3ZwMmRpcjp0b21jYXQ0MTU2OjE2Mjc3NTUxOTUyNDJQ6XiC5l.gen_Telcentris_Robocall__Utility-30MinDisconnect": "tech_support_utility",
    "ncsu_240204": "tech_support_utility",  # electric bill rate reduction
    "ncsu_668376": "amazon_apple_order",  # "Marni from Amazon", unauthorized charge
    "ncsu_62996": "amazon_apple_order",  # "Apple Support Advisor" (matched "toll free")
    "ncsu_57750": "amazon_apple_order",  # same Apple Support campaign
    "ncsu_1204362": "bank_credit_card",  # Discover debt settlement
    "ncsu_303960": "bank_credit_card",  # Visa/MasterCard 0% interest
    "ncsu_274461": "gov_ssa_irs",  # COVID-19 relief payment, "state assigned phone number" (as v1 274460)
    "ncsu_974385": "gov_ssa_irs",  # "suspended your social insurance number" (as v1 974387)
}
QUOTA = {"gov_ssa_irs": 14, "amazon_apple_order": 11, "auto_warranty": 6, "bank_credit_card": 10,
         "tech_support_utility": 10, "prize_travel_health": 7, "delivery_parcel": 4}  # 62


def scam_type(text: str) -> str:
    t = text.lower()
    for name, pat in SCAM_TYPES:
        if re.search(pat, t):
            return name
    return "other"


def pick_ncsu(meta: Path, quota: dict | None = None, seed: int = SEED, exclude: set | None = None,
              fix: dict | None = None) -> list[dict]:
    quota = QUOTA if quota is None else quota
    fix = TYPE_FIX if fix is None else fix
    exclude = exclude or set()
    rows = [r for r in csv.DictReader(open(meta)) if r["language"] == "en" and len(r["transcript"].split()) >= 25]
    seen, buckets = set(), defaultdict(list)
    for r in rows:
        key = re.sub(r"\W+", " ", r["transcript"].lower())[:120]  # near-duplicate campaigns
        if r["file_name"] in seen or key in seen:
            continue
        seen.update({r["file_name"], key})
        if _ncsu_id(r) in exclude:
            continue
        buckets[scam_type(r["transcript"])].append(r)
    rng = random.Random(seed)
    out = []
    for st, n in quota.items():
        cand = sorted(buckets[st], key=lambda r: r["file_name"])
        rng.shuffle(cand)
        # one per carrier first (diversity), then fill
        chosen, carriers = [], set()
        for r in cand:
            if n == 0:
                break
            if r["case_details"] not in carriers:
                chosen.append(r); carriers.add(r["case_details"])
            if len(chosen) == n:
                break
        for r in cand:
            if len(chosen) == n:
                break
            if r not in chosen:
                chosen.append(r)
        for r in chosen:
            fid = Path(r["file_name"]).stem.replace("_normalized", "")
            out.append({"call_id": f"ncsu_{fid}", "source": "ncsu_robocall", "license": "public domain (NCSU Robocall Audio Dataset)",
                        "label": "scam", "scam_type": fix.get(f"ncsu_{fid}", st), "url": NCSU_RAW + r["file_name"], "url2": "",
                        "carrier_case": r["case_details"]})
    return out


def _ncsu_id(r: dict) -> str:
    return "ncsu_" + Path(r["file_name"]).stem.replace("_normalized", "")


def pick_hv(meta_dir: Path, n: int = 50, seed: int = SEED, exclude: set | None = None) -> list[dict]:
    exclude = exclude or set()
    by_task = defaultdict(list)
    for p in sorted(meta_dir.glob("*.json")):
        m = json.load(open(p))
        task = (m.get("tasks") or [{}])[0].get("task_type", "unknown")
        dur = (m["end_time_ms"] - m["start_time_ms"]) / 1000
        if dur < 25 or f"hv_{m['sid']}" in exclude:
            continue
        by_task[task].append(m["sid"])
    rng = random.Random(seed)
    for v in by_task.values():
        rng.shuffle(v)
    out, tasks = [], sorted(by_task)
    while len(out) < n and any(by_task.values()):
        for t in tasks:
            if by_task[t] and len(out) < n:
                sid = by_task[t].pop()
                out.append({"call_id": f"hv_{sid}", "source": "harpervalley_bank", "license": "CC BY 4.0 (Gridspace-Stanford HarperValleyBank)",
                            "label": "benign", "scam_type": f"benign.bank.{t.replace(' ', '_')}",
                            "url": HV_RAW + f"audio/agent/{sid}.wav", "url2": HV_RAW + f"audio/caller/{sid}.wav", "carrier_case": ""})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/realset/raw")
    ap.add_argument("--out", default="realset/manifest.csv")
    ap.add_argument("--drop", default="realset/dropped_v2.txt", help="v2 call_ids dropped (not transcribed in the time box)")
    a = ap.parse_args()
    raw = Path(a.raw)
    v1 = pick_ncsu(raw / "ncsu" / "metadata.csv") + pick_hv(raw / "hv" / "metadata")
    ids = {r["call_id"] for r in v1}
    v2 = pick_ncsu(raw / "ncsu" / "metadata.csv", QUOTA_V2, SEED + 1, ids, TYPE_FIX_V2)
    v2 += pick_hv(raw / "hv" / "metadata_v2", 70, SEED + 1, ids | {r["call_id"] for r in v2})
    drop = set(Path(a.drop).read_text().split()) if a.drop and Path(a.drop).exists() else set()
    v2 = [r for r in v2 if r["call_id"] not in drop]
    assert not ids & {r["call_id"] for r in v2}
    for r in v1:
        r["batch"] = "v1"
    for r in v2:
        r["batch"] = "v2"
    rows = v1 + v2
    assign_split(rows)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter((r["batch"], r["label"]) for r in rows), Counter((r["split"], r["label"]) for r in rows))
    print(Counter((r["label"], r["scam_type"]) for r in rows))


def assign_split(rows: list[dict]) -> None:
    """Seeded 50/50 dev/test, stratified by (label, scam_type); odd strata alternate the extra."""
    rng = random.Random(SEED + 7)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["label"], r["scam_type"])].append(r)
    k = 0
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r["call_id"])
        rng.shuffle(group)
        for r in group:
            r["split"] = "dev" if k % 2 == 0 else "test"
            k += 1


if __name__ == "__main__":
    main()
