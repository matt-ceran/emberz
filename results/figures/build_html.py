"""Build docs/results.html (self-contained: PNGs embedded as base64) from results/guardian_snapshot.json.

    .venv/bin/python results/figures/build_html.py
"""
import base64, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
S = json.loads((ROOT / "results" / "guardian_snapshot.json").read_text())


def img(name, alt):
    p = FIG / f"{name}.png"
    if not p.exists():
        return None
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{html.escape(alt)}">'


def sys(table, id_):
    return next(s for s in S[table]["systems"] if s["id"] == id_)


def ci(r):
    lo, hi = r["wilson95"]
    return f"{100 * lo:.0f}-{100 * hi:.0f}%"


v1s, kimi, ds = sys("subset_120", "v1"), sys("subset_120", "B4"), sys("subset_120", "B5")
v1a, v2a, b3a, nemo_a = sys("all_257", "v1"), sys("all_257", "v2"), sys("all_257", "B3"), sys("all_257", "nemo")
bb = {s["id"]: s for s in S["bothbosu"]["systems"]}

headline = (f"On the same 120 held-out calls, our fine-tuned guardian raised a false alarm on "
            f"<b>{v1s['false_alarms_benign']['pct']}%</b> of benign calls, vs "
            f"<b>{kimi['false_alarms_benign']['pct']}%</b> for Kimi K3 and "
            f"<b>{ds['false_alarms_benign']['pct']}%</b> for DeepSeek V4.1 Flash, while catching "
            f"<b>{v1s['catch_before_ask']['pct']}%</b> of risky calls before the ask "
            f"(Kimi {kimi['catch_before_ask']['pct']}%, DeepSeek {ds['catch_before_ask']['pct']}%).")

tiles = [
    (f"{b3a['false_alarms_benign']['pct']:.0f}% &rarr; {v1a['false_alarms_benign']['pct']:.1f}%",
     f"False alarms, untuned base vs fine-tuned v1 (all 257, n={S['all_257']['n_benign']} benign)"),
    (f"{v1a['rule_flip_pairs']['count']}/22",
     f"Rule-flip pairs right (untuned base {b3a['rule_flip_pairs']['count']}/22, DeepSeek "
     f"{sys('all_257', 'B5')['rule_flip_pairs']['count']}/22)"),
    (f"{bb['v2']['scam_recall']['pct']:.0f}%",
     f"Out-of-distribution scam recall (BothBosu, n=100), {bb['v2']['false_alarms_benign']['count']}/100 false alarms"),
    (f"{nemo_a['false_alarms_benign']['pct']}%",
     f"False alarms, our fine-tune on NVIDIA Nemotron 3.5 Super (all 257)"),
]


def table(rows, head):
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def main_rows(key):
    out = []
    for s in S[key]["systems"]:
        c, f = s["catch_before_ask"], s["false_alarms_benign"]
        name = html.escape(s["label"])
        if s["kind"].startswith("ours"):
            name = f"<b>{name}</b>"
        lat = f"{s['latency_p50_s']:.2f} s" + (" (sampler)" if s["latency_source"] == "training_api_sampler" else "")
        out.append([name, f"{c['pct']}% <span class=ci>[{ci(c)}]</span>",
                    f"{f['pct']}% <span class=ci>[{ci(f)}]</span>",
                    f"{s['rule_flip_pairs']['count']}/{s['rule_flip_pairs']['n']}", f"{s['category_f1']:.2f}", lat])
    return out


casc = [[html.escape(s["label"]),
         f"{s['all_257']['catch_before_ask']['pct']}%",
         f"{s['all_257']['false_alarms_benign']['count']}/79 ({s['all_257']['false_alarms_benign']['pct']}%)",
         f"{s['subset_120']['catch_before_ask']['pct']}%",
         f"{s['subset_120']['false_alarms_benign']['count']}/36"] for s in S["cascade"]["systems"]]
real = [[html.escape(s["label"]), f"{s['clean']['catch_before_ask_pct']}% / {s['clean']['false_alarms']['count']}/16",
         f"{s['asr']['catch_before_ask_pct']}% / {s['asr']['false_alarms']['count']}/16",
         f"{s['asr_risky_flagged_at_all_pct']}%"] for s in S["real_speech_49"]["systems"]]

H = ["System", "Catch before ask [95% CI]", "False alarms, benign [95% CI]", "Rule flips", "Cat. F1", "Latency p50"]

figs = []
for name, cap in [
    ("headline_all_257", f"All 257 held-out calls: n = {S['all_257']['n_risky']} risky, {S['all_257']['n_benign']} benign. Kimi K3 was scored on the 120-call subset only."),
    ("rule_flips", "Rule-flip pairs: the same call with one fact changed (e.g. a family code word), so the right answer flips. 22 pairs; small n."),
    ("bothbosu", "BothBosu: calls written by a different model family, call-level labels. n = 100 scam, 100 benign. Only rules, v1 and v2 were run."),
    ("latency_p50", "Latency p50 per turn. Hatched bars come from the Fireworks Training API sampler (queueing), not a dedicated deployment; production latency is not measured yet."),
    ("line_budget", "Live-line latency budget vs the 1.5 s coach target. Hatched = estimate or non-production number; see results/figures/line_latency.json."),
    ("voice_roc", "AI-voice detector ROC (results/voice_roc.json)."),
]:
    t = img(name, cap)
    if t:
        figs.append(f'<figure>{t}<figcaption>{html.escape(cap)}</figcaption></figure>')

head_img = img("headline_subset_120", "Catch-before-ask vs false-alarm rate, 120-call common subset")

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Emberz Guardian Results</title>
<style>
:root {{ --bg:#1a1a19; --card:#232322; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8a8984; --line:#383835; --ours:#3987e5; --nv:#199e70; --front:#d95926; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 16px 64px; }}
.kicker {{ color:var(--ours); font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:13px; }}
h1 {{ font-size:34px; line-height:1.2; margin:6px 0 10px; }}
.claim {{ font-size:21px; line-height:1.45; color:var(--ink2); max-width:1040px; margin:0 0 18px; }}
.claim b {{ color:var(--ink); }}
.tiles {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin:18px 0 22px; }}
.tile {{ background:var(--card); border-radius:10px; padding:14px 16px; border-top:3px solid var(--ours); }}
.tile:last-child {{ border-top-color:var(--nv); }}
.tile .v {{ font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; }}
.tile .l {{ color:var(--ink2); font-size:13.5px; line-height:1.35; }}
figure {{ margin:0 0 28px; }}
figure img {{ width:100%; height:auto; display:block; border-radius:10px; }}
figcaption {{ color:var(--muted); font-size:13.5px; margin-top:6px; }}
h2 {{ font-size:22px; margin:34px 0 10px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; margin-bottom:8px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--ink2); font-weight:600; }}
.ci {{ color:var(--muted); font-size:12.5px; }}
.wrap {{ overflow-x:auto; }}
ul {{ color:var(--ink2); padding-left:20px; }}
li {{ margin:4px 0; }}
.src {{ color:var(--muted); font-size:13px; }}
@media (max-width:760px) {{ .tiles {{ grid-template-columns:1fr 1fr; }} h1 {{ font-size:26px; }} .claim {{ font-size:17px; }} }}
</style></head>
<body><main>
<div class="kicker">Emberz &middot; scam guardian for a phone line</div>
<h1>Fine-tuned on Fireworks: fewer false alarms than frontier models, catch rate on par</h1>
<p class="claim">{headline}</p>
<div class="tiles">{''.join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles)}</div>
<figure>{head_img}<figcaption>Same 120 held-out calls for every model: n = {S['subset_120']['n_risky']} risky, {S['subset_120']['n_benign']} benign. Whiskers are Wilson 95% intervals; with 36 benign calls they overlap, so read the gap as a direction, not a settled margin.</figcaption></figure>
{''.join(figs)}
<h2>Cascade: Rune as a 0.15 s gate, our guardian confirms</h2>
<div class="wrap">{table(casc, ["Cascade", "All 257: catch", "All 257: false alarms", "Same 120: catch", "Same 120: false alarms"])}</div>
<p class="src">{html.escape(S['cascade']['note'])}.</p>
<h2>Real speech: 49 held-out calls through the phone-path ASR</h2>
<div class="wrap">{table(real, ["System", "Clean text: catch / false alarms", "Nemotron ASR text: catch / false alarms", "ASR: risky flagged at all"])}</div>
<p class="src">{html.escape(S['real_speech_49']['note'])}. 16 benign calls: differences in false alarms here are noise.</p>
<h2>Full tables</h2>
<h3>Same 120 held-out calls</h3><div class="wrap">{table(main_rows('subset_120'), H)}</div>
<h3>All 257 held-out calls</h3><div class="wrap">{table(main_rows('all_257'), H)}</div>
<h2>How to read this</h2>
<ul>{''.join(f'<li>{html.escape(n)}</li>' for n in S['notes'])}
<li>Catch before ask = share of risky calls flagged at alert level before the labeled ask turn. Counts are round(pct &times; n); intervals are Wilson 95%.</li></ul>
<p class="src">{html.escape(S['source'])} Figures: results/figures/make_figures.py. Data: results/guardian_snapshot.json.</p>
</main></body></html>
"""
out = ROOT / "docs" / "results.html"
out.write_text(page)
print("wrote", out, f"{len(page) // 1024} KB")
