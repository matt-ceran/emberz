"""Render the results figures from results/guardian_snapshot.json (+ optional results files).

    .venv/bin/python results/figures/build_snapshot.py   # tables -> json
    .venv/bin/python results/figures/make_figures.py     # json -> PNG (2x) + SVG

Optional inputs: results/voice_roc.json (AI-voice ROC), results/figures/line_latency.json (stage timings).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures"
SNAP = json.loads((ROOT / "results" / "guardian_snapshot.json").read_text())

# Dark chart surface, validated categorical slots (dataviz palette, dark mode).
SURF, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8a8984", "#383835"
OURS, FRONT, OTHER, NV = "#3987e5", "#d95926", "#8a8984", "#199e70"
COLOR = {"ours": OURS, "ours_nvidia": NV, "frontier": FRONT, "baseline": OTHER, "rules": OTHER}
MARKER = {"ours": "o", "ours_nvidia": "^", "frontier": "D", "baseline": "s", "rules": "s"}
SRC = "Source: Matt's TRAINING_RUN.md, refreshed 14:26 PST Nov 2 · whiskers = Wilson 95% CI"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 12,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "svg.fonttype": "none",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png", "+ .svg")


def short(label):
    return (label.replace(", Fireworks zero-shot", " (zero-shot)")
                 .replace("Nemotron 3.5 Super fine-tuned (ours, NVIDIA)", "Nemotron 3.5 FT")
                 .replace(" (ours)", ""))


# label offsets (points) per system id so direct labels don't collide
OFFS = {
    "subset_120": {"B0": (10, 8), "B1": (10, 12), "B3": (10, -16), "B4": (0, -26), "B5": (12, -10),
                    "v1": (-14, 0), "v2": (0, 24), "nemo": (-10, -30)},
    "all_257": {"B0": (10, 8), "B1": (10, 10), "B3": (10, -16), "B5": (12, -4),
                 "v1": (-12, 12), "v2": (0, 24), "nemo": (-14, -14)},
}


def headline(key):
    t = SNAP[key]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.14)
    for s in t["systems"]:
        c, fa = s["catch_before_ask"], s["false_alarms_benign"]
        x, y = fa["pct"], c["pct"]
        xl, xh = [100 * v for v in fa["wilson95"]]
        yl, yh = [100 * v for v in c["wilson95"]]
        col, ours = COLOR[s["kind"]], s["kind"].startswith("ours")
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="none",
                    ecolor=col, elinewidth=2 if ours else 1.2, alpha=0.9 if ours else 0.55,
                    capsize=0, zorder=2)
        ax.scatter([x], [y], s=150 if ours else 80, marker=MARKER[s["kind"]], color=col,
                   edgecolor=SURF, linewidth=2, zorder=3)
        dx, dy = OFFS[key].get(s["id"], (8, 6))
        name = short(s["label"])
        if s["kind"] == "frontier":
            name += " (frontier)"
        ax.annotate(f"{name}\n{y:.1f}% catch · {x:.1f}% FA", (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10.5 if ours else 9.5,
                    color=INK if ours or s["kind"] == "frontier" else INK2,
                    fontweight="bold" if ours else "normal",
                    ha="right" if dx < 0 else ("center" if dx == 0 else "left"), va="center", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    ax.set_xlim(0, 75)
    ax.set_ylim(30, 100)
    ax.set_xlabel(f"False alarms on benign calls (%)  ·  n = {t['n_benign']} benign")
    ax.set_ylabel(f"Catch before the ask (%)  ·  n = {t['n_risky']} risky")
    ax.annotate("better: more caught,\nfewer false alarms", xy=(1.5, 72), xytext=(9, 63), color=INK2,
                fontsize=10.5, ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    sub = ("Same 120 held-out calls for every model (Kimi K3 scored on these only)"
           if key == "subset_120" else "All 257 held-out calls (Kimi K3 not run on the full set)")
    fig.text(0.09, 0.94, "Fine-tuned guardian: fewer false alarms, catch rate on par",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(0.09, 0.895, sub, fontsize=11.5, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=OURS, markersize=10, label="Guardian (ours)"),
               plt.Line2D([], [], marker="^", ls="", color=NV, markersize=10, label="Ours on NVIDIA Nemotron"),
               plt.Line2D([], [], marker="D", ls="", color=FRONT, markersize=8, label="Frontier baseline"),
               plt.Line2D([], [], marker="s", ls="", color=OTHER, markersize=8, label="Other baseline")]
    ax.legend(handles=handles, loc="lower right", frameon=False, labelcolor=INK2, fontsize=10)
    fig.text(0.09, 0.03, SRC, fontsize=9, color=MUTED)
    save(fig, f"headline_{key}")


def hbars(ax, labels, vals, kinds, fmt, xmax, err=None):
    y = list(range(len(labels)))[::-1]
    cols = [COLOR[k] for k in kinds]
    ax.barh(y, vals, color=cols, height=0.62, edgecolor=SURF, linewidth=2, zorder=2)
    if err is not None:
        lo = [v - e[0] for v, e in zip(vals, err)]
        hi = [e[1] - v for v, e in zip(vals, err)]
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, zorder=3)
    ends = vals if err is None else [max(v, e[1]) for v, e in zip(vals, err)]
    for yy, v, k, s in zip(y, ends, kinds, fmt):
        ax.text(v + xmax * 0.015, yy, s, va="center",
                color=INK if k.startswith("ours") else INK2, fontsize=11,
                fontweight="bold" if k.startswith("ours") else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for t, k in zip(ax.get_yticklabels(), kinds):
        t.set_color(INK if k in ("ours", "ours_nvidia", "frontier") else INK2)
        if k.startswith("ours"):
            t.set_fontweight("bold")
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def rule_flips():
    t = SNAP["all_257"]
    sy = sorted(t["systems"], key=lambda s: -s["rule_flip_pairs"]["count"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.80, bottom=0.15)
    vals = [s["rule_flip_pairs"]["count"] for s in sy]
    hbars(ax, [short(s["label"]) + (" (frontier)" if s["kind"] == "frontier" else "") for s in sy], vals,
          [s["kind"] for s in sy], [f"{v}/22" for v in vals], 24.5)
    ax.set_xlabel("Rule-flip pairs judged correctly (out of 22)")
    fig.text(0.03, 0.93, "Rule flips: same call, one fact changed, the right answer flips",
             fontsize=16, fontweight="bold")
    fig.text(0.03, 0.87, "All 257 held-out calls · 22 minimal pairs · Kimi K3 scored 0/10 on the 120-call subset",
             fontsize=11, color=INK2)
    fig.text(0.03, 0.03, "Source: Matt's TRAINING_RUN.md, refreshed 14:26 PST Nov 2", fontsize=9, color=MUTED)
    save(fig, "rule_flips")


def latency():
    sy = list(SNAP["all_257"]["systems"])
    kimi = next(s for s in SNAP["subset_120"]["systems"] if s["id"] == "B4")
    sy.append(dict(kimi, label=kimi["label"] + "*"))
    sy.sort(key=lambda s: s["latency_p50_s"])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.29, right=0.95, top=0.80, bottom=0.20)
    vals = [s["latency_p50_s"] for s in sy]
    labels = [short(s["label"]) + (" (frontier)" if s["kind"] == "frontier" else "") for s in sy]
    y = list(range(len(sy)))[::-1]
    for yy, s, v in zip(y, sy, vals):
        sampler = s["latency_source"] == "training_api_sampler"
        ax.barh(yy, v, color=COLOR[s["kind"]], height=0.62, edgecolor=SURF, linewidth=2,
                hatch="//" if sampler else None, alpha=0.85 if sampler else 1.0, zorder=2)
        ax.text(v + 0.1, yy, f"{v:.2f} s" + ("  (sampler)" if sampler else ""), va="center",
                color=INK if s["kind"].startswith("ours") else INK2, fontsize=11, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"),
                fontweight="bold" if s["kind"].startswith("ours") else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(1.5, color=INK2, lw=1, ls="--", zorder=1)
    ax.text(1.53, y[0] + 0.45, "1.5 s coach budget", color=INK2, fontsize=9.5, va="bottom")
    ax.set_xlim(0, 10.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Latency p50 per turn (s)")
    fig.text(0.03, 0.93, "Latency p50 per turn", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.87, "All 257 held-out calls · *Kimi K3 from the 120-call subset", fontsize=11, color=INK2)
    fig.text(0.03, 0.075, "Hatched = Fireworks Training API sampler (queueing), not a deployment. "
             "Production guardian latency needs the dedicated deployment.", fontsize=10, color=INK2)
    fig.text(0.03, 0.03, "Source: Matt's TRAINING_RUN.md, refreshed 14:26 PST Nov 2", fontsize=9, color=MUTED)
    save(fig, "latency_p50")


def bothbosu():
    t = SNAP["bothbosu"]
    sy = t["systems"][::-1]
    sy = sorted(t["systems"], key=lambda s: -s["scam_recall"]["pct"])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.24, right=0.95, top=0.76, bottom=0.2)
    vals = [s["scam_recall"]["pct"] for s in sy]
    err = [[100 * v for v in s["scam_recall"]["wilson95"]] for s in sy]
    hbars(ax, [short(s["label"]) for s in sy], vals, [s["kind"] for s in sy],
          [f"{v:.0f}%   ·   FA {s['false_alarms_benign']['count']}/100" for v, s in zip(vals, sy)], 125, err)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Scam recall (%)  ·  n = {t['n_scam']} scam calls")
    fig.text(0.03, 0.92, "Out of distribution (BothBosu): 98% recall, 0 false alarms",
             fontsize=16, fontweight="bold")
    fig.text(0.03, 0.855, f"Calls written by a different model family · call-level labels · "
             f"false alarms are 0/{t['n_benign']} benign for every system", fontsize=11, color=INK2)
    fig.text(0.03, 0.03, SRC, fontsize=9, color=MUTED)
    save(fig, "bothbosu")


def voice_roc():
    p = ROOT / "results" / "voice_roc.json"
    if not p.exists():
        print("skip voice ROC: results/voice_roc.json not present")
        return False
    d = json.loads(p.read_text())
    curves = d.get("curves") or [d]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.14)
    pal = [OURS, FRONT, "#199e70"]
    for i, c in enumerate(curves[:3]):
        lab = c.get("label") or c.get("name") or f"curve {i + 1}"
        auc = c.get("auc")
        ax.plot(c["fpr"], c["tpr"], color=pal[i], lw=2, label=f"{lab}" + (f" (AUC {auc:.3f})" if auc is not None else ""))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set_xlabel("False positive rate (human called AI)")
    ax.set_ylabel("True positive rate (AI voice caught)")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right")
    fig.text(0.12, 0.92, "AI-voice detector ROC", fontsize=16, fontweight="bold")
    n = d.get("n") or d.get("n_total")
    fig.text(0.12, 0.875, f"{d.get('dataset', '')}" + (f" · n = {n}" if n else ""), fontsize=11, color=INK2)
    fig.text(0.12, 0.03, "Source: results/voice_roc.json", fontsize=9, color=MUTED)
    save(fig, "voice_roc")
    return True


def line_budget():
    """Stage timings vs the 1.5 s coach budget. Reads results/line_latency.json:
    {"stages": [{"name", "ms", "source", "estimate": bool}], "note": str}"""
    p = FIG_IN = OUT / "line_latency.json"
    if not p.exists():
        print("skip line budget: results/line_latency.json not present")
        return False
    d = json.loads(p.read_text())
    st = d["stages"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.70, bottom=0.30)
    left, cols = 0.0, [OURS, FRONT, "#199e70"]
    for i, s in enumerate(st):
        ms = s.get("ms")
        if ms is None:
            continue
        est = s.get("estimate")
        ax.barh(0, ms, left=left, color=cols[i % 3], height=0.5, edgecolor=SURF, linewidth=2,
                hatch="//" if est else None, alpha=0.8 if est else 1.0, zorder=2)
        ax.text(left + ms / 2, 0.42, f"{s['name']}\n{ms:.0f} ms" + (" (est.)" if est else ""),
                ha="center", va="bottom", fontsize=10.5, color=INK)
        left += ms
    ax.axvline(1500, color=INK, lw=1.5, ls="--", zorder=3)
    ax.text(1510, -0.42, "1.5 s target", color=INK, fontsize=10.5, va="center")
    ax.text(left, -0.42, f"total {left:.0f} ms ", ha="right", color=INK2, fontsize=10.5, va="center")
    ax.set_ylim(-0.6, 1.0)
    ax.set_yticks([])
    ax.set_xlim(0, max(1700, left * 1.08))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ms from the end of the caller's line to the coach tip")
    fig.text(0.05, 0.9, "Live-line latency budget", fontsize=16, fontweight="bold")
    fig.text(0.05, 0.82, d.get("subtitle", ""), fontsize=11, color=INK2)
    fig.text(0.05, 0.07, d.get("note", ""), fontsize=9.5, color=INK2)
    fig.text(0.05, 0.02, "Hatched = estimate or non-production measurement; sources in results/figures/line_latency.json",
             fontsize=9, color=MUTED)
    save(fig, "line_budget")
    return True


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    headline("subset_120")
    headline("all_257")
    rule_flips()
    latency()
    bothbosu()
    voice_roc()
    line_budget()
