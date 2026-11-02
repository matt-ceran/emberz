"""Charts + results table from results/results.jsonl (latest row per model/test_set/metric).

  callguard/.venv/bin/python callguard/eval/charts.py
Writes results/chart_catch_vs_false_alarms.png and results/RESULTS.md.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
NICE = {"b0-keywords": "Keywords (B0)", "b4-kimi-k3": "Kimi K3 zero-shot (B4)",
        "b3-qwen3p6-35b-untuned": "Qwen3.6-35B untuned (B3)", "b1-rune": "Fireworks Rune zero-shot (B1)",
        "b5-deepseek-v4p1-flash": "DeepSeek V4.1 Flash zero-shot (B5)",
        "guardian-v1": "Guardian v1 (Qwen3.6-35B LoRA)", "guardian-v2": "Guardian v2 (Qwen3.6-35B LoRA, deployed)",
        "guardian-v3": "Guardian v3 (Qwen3.6-35B LoRA + contrast pairs)",
        "guardian-nemo-v1": "Guardian nemo-v1 (Nemotron 3.5 Super LoRA)"}
INK, MUTED, GRID, SURF, BLUE, GRAY = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6", "#9a9994"


def latest():
    rows = [json.loads(l) for l in (RES / "results.jsonl").read_text().splitlines() if l.strip()]
    best = {}
    for r in rows:
        best[(r["model"], r["test_set"], r["metric"])] = r
    out = {}
    for (m, ts, k), r in best.items():
        out.setdefault((m, ts), {})[k] = r
    return out


def nice(m):
    return NICE.get(m, m.replace("guardian-", "Emberz guardian ").replace("-", " "))


def main(test_set="heldout"):
    data = {m: v for (m, ts), v in latest().items() if ts == test_set}
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    pts = []
    for m, v in data.items():
        if "catch_before_ask" in v and "false_alarm_rate" in v and v["catch_before_ask"]["value"] is not None:
            pts.append((m, v["false_alarm_rate"], v["catch_before_ask"]))
    pts.sort(key=lambda t: -t[2]["value"])
    if not pts:
        plt.close(fig)
        fig = None
    ylo = 0 if not pts else max(0, 5 * int((min(y["ci95"][0] for _, _, y in pts) * 100 - 6) // 5))
    for i, (m, x, y) in enumerate(pts if fig else []):
        hero = m.startswith("guardian")
        c = BLUE if hero else GRAY
        xv, yv = x["value"] * 100, y["value"] * 100
        ax.errorbar(xv, yv, xerr=[[xv - 100 * x["ci95"][0]], [100 * x["ci95"][1] - xv]],
                    yerr=[[yv - 100 * y["ci95"][0]], [100 * y["ci95"][1] - yv]],
                    fmt="o", ms=9 if hero else 7, color=c, ecolor=c, elinewidth=1, capsize=0, alpha=0.95,
                    markeredgecolor=SURF, markeredgewidth=2, zorder=3 if hero else 2)
        ly = 98 - i * (98 - ylo - 4) / max(1, len(pts) - 1) if len(pts) > 1 else 90
        ax.annotate(f"{nice(m)}\n{yv:.0f}% caught · {xv:.0f}% false alarms",
                    xy=(xv, yv), xytext=(64, ly), textcoords="data", va="center", ha="left",
                    fontsize=7.5, color=INK if hero else MUTED, fontweight="bold" if hero else "normal",
                    arrowprops=dict(arrowstyle="-", color=BLUE if hero else "#c9c8c3", lw=0.8, shrinkA=2, shrinkB=5, relpos=(0, 0.5)))
    if fig is None:
        return write_table(data, test_set)
    n_r, n_b = pts[0][2]["n"], pts[0][1]["n"]
    ax.text(14, ylo + 1, f"n = {n_r} risky calls with an ask, {n_b} harmless calls", fontsize=7, color=MUTED)
    ax.set_xlabel("False alarms: % of harmless calls that triggered an alert  (lower is better)", fontsize=8, color=MUTED)
    ax.set_ylabel("Caught before the ask: % of risky calls  (higher is better)", fontsize=8, color=MUTED)
    ax.set_title(f"Catch risky calls early without crying wolf  ({ {'heldout': 'all 257 held-out calls', 'heldout120': '120-call held-out subset (same as Kimi)', 'dev': '143 dev calls, never trained on'}.get(test_set, test_set) }, 95% CIs)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlim(-2, 102); ax.set_ylim(ylo, 101)
    fig.subplots_adjust(right=0.98)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.annotate("better", xy=(1, 99), xytext=(2, ylo + 12), fontsize=7.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    fig.tight_layout()
    p = RES / f"chart_catch_vs_false_alarms_{test_set}.png"
    fig.savefig(p, facecolor=SURF)
    print(p)
    write_table(data, test_set)


def write_table(data, test_set):
    cols = ["catch_before_ask", "catch_at_or_before_ask", "recall_any", "false_alarm_rate", "rule_flip_accuracy", "unseen_type_recall",
            "lead_time_turns_median", "category_macro_f1_at_first_alert", "evidence_exact_match", "latency_ms_p50", "latency_ms_p95"]
    lines = [f"# Results ({test_set})", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, v in sorted(data.items(), key=lambda kv: (not kv[0].startswith("guardian"), kv[0])):
        cells = []
        for c in cols:
            r = v.get(c)
            if not r or r["value"] is None:
                cells.append("-")
            elif "ci95" in r:
                cells.append(f"{r['value']:.2f} [{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}] n={r['n']}")
            else:
                cells.append(f"{r['value']} (n={r['n']})")
        lines.append(f"| {nice(m)} | " + " | ".join(cells) + " |")
    lines += ["", "Latency for the guardian and the untuned Qwen (B3) is from the Fireworks Training API sampler under shared-pool "
              "contention: **not a production latency**. The serverless baselines' latency is from the normal inference API. "
              "The production latency claim waits on a dedicated deployment of the guardian.",
              "Unseen scam type (gold courier pickup) has n=2 held-out calls: too small to claim anything.",
              "Same-family caveat: the held-out set was written by Nemotron 3 Ultra, so the fine-tuned Nemotron 3.5 Super (nemo-v1) "
              "may get a style advantage on it; compare it on BothBosu too.",
              "BothBosu: synthetic out-of-distribution conversations with call-level labels only; catch-before-ask does not apply."]
    (RES / f"RESULTS_{test_set}.md").write_text("\n".join(lines) + "\n")


def fa_breakdown(test_set="heldout", test_file="data/calls_test_heldout.jsonl"):
    """Per benign-scenario false alarms for each guardian, plus the total with/without conflicted allowed-contact twins."""
    import re as _re
    root = RES.parent
    rows = {json.loads(l)["call_id"]: json.loads(l) for l in (root / test_file).read_text().splitlines() if l.strip()}
    R = ["none", "watch", "alert", "urgent"]
    risky = lambda r: not r.get("benign_twin") and any(R.index(t["label"]["risk"]) >= 2 for t in r["turns"])
    code_rule = _re.compile(r"(code|password|pin)", _re.I)
    conflicted = {cid for cid, r in rows.items() if r.get("flip_kind") == "allowed_contact"
                  and any(code_rule.search(x) for x in r["rules"] if not x.startswith("Allowed"))}
    benign = {cid: r for cid, r in rows.items() if not risky(r)}
    seeds = sorted({r["seed_id"] + (" (twin)" if r.get("flip_kind") else "") for r in benign.values()})
    models = [m for m in ("guardian-v1", "guardian-v2", "guardian-v3", "guardian-nemo-v1")
              if (RES / f"preds_{m}__{test_set}.jsonl").exists()]
    if not models:
        return
    preds = {m: {json.loads(l)["call_id"]: json.loads(l) for l in (RES / f"preds_{m}__{test_set}.jsonl").read_text().splitlines() if l.strip()}
             for m in models}
    lines = ["", f"## False alarms by harmless scenario ({test_set})", "",
             "| Harmless scenario | n | " + " | ".join(nice(m) for m in models) + " |", "|---|---|" + "---|" * len(models)]
    for sd in seeds:
        ids = [cid for cid, r in benign.items() if r["seed_id"] + (" (twin)" if r.get("flip_kind") else "") == sd]
        cells = []
        for m in models:
            k = sum(1 for cid in ids if cid in preds[m] and preds[m][cid]["first_alert"])
            cells.append(str(k))
        if any(c != "0" for c in cells):
            lines.append(f"| {sd} | {len(ids)} | " + " | ".join(cells) + " |")
    tot_cells, ex_cells = [], []
    for m in models:
        fa = [cid for cid in benign if cid in preds[m] and preds[m][cid]["first_alert"]]
        tot_cells.append(f"{len(fa)}/{len(benign)}")
        ex_cells.append(f"{len([c for c in fa if c not in conflicted])}/{len(benign) - len(conflicted)}")
    lines.append("| **All harmless calls (primary)** | | " + " | ".join(tot_cells) + " |")
    lines.append(f"| Excluding {len(conflicted)} conflicted twins* | | " + " | ".join(ex_cells) + " |")
    lines += ["", "*Allowed-contact twins whose own rules also say never to share codes/passwords/PINs; the twin generator labeled "
              "them harmless even though the transcript breaks that rule, so an alert there is defensible. Fixed for v3's new data."]
    with (RES / f"RESULTS_{test_set}.md").open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    for ts in ("heldout", "heldout120", "bothbosu", "dev"):
        main(ts)
    fa_breakdown("heldout")
    fa_breakdown("dev", "data/dev_calls.jsonl") if (RES / "RESULTS_dev.md").exists() else None
