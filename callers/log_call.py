"""Log a manual test call just before dialing, so its recording can be matched to its source.

  python -m callers.log_call roleplay rp07
  python -m callers.log_call human p03 --script bank_fraud --phone android --place street

Appends to data/callers/calls.jsonl (gitignored), the same ledger the Vapi/Retell scripts write.
`python -m callers.label_sheet match-sources` later pairs ledger rows with recordings by start time.
Use pseudonyms (p01, p02 ...) for people, never names or numbers.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from callers.common import append_ledger


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("platform", choices=["roleplay", "human"])
    p.add_argument("tag", help="card id (rp01..rp40) or caller pseudonym (p01..)")
    p.add_argument("--script", default=None, help="human calls: which agent script was read (e.g. bank_fraud)")
    p.add_argument("--phone", default=None, help="e.g. iphone, android, landline, speaker")
    p.add_argument("--place", default=None, help="e.g. quiet, street, car")
    args = p.parse_args(argv)
    row = {"platform": args.platform, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.platform == "roleplay":
        row["card"] = args.tag
        row["kind"] = "roleplay"
    else:
        row.update(caller=args.tag, kind="human", script=args.script, phone=args.phone, place=args.place)
    append_ledger(row)
    print(f"logged {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
