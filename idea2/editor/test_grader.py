"""Checks grader.py reproduces the screen's grades on every dev/select/test/unknown row (run on the Mac)."""
import json
from pathlib import Path

from grader import CONDITIONS, grade

SPL = Path(__file__).resolve().parent.parent / "screen" / "splits"
bad = n = 0
for name in ("dev", "select", "test", "unknown"):
    for line in (SPL / f"{name}.jsonl").open():
        r = json.loads(line)
        for c in CONDITIONS:
            g = grade(r, r["base_replies"][c])
            b = r["base_qwen3_8b"]
            n += 1
            if g["correct"] != b[f"correct_{c}"] or g["says_wrong"] != b[f"says_wrong_{c}"]:
                bad += 1
            assert r["prompts"][c], (r["id"], c)
print(f"{n - bad}/{n} grades reproduced")
assert bad == 0
