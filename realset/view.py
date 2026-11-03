"""Self-contained HTML view of the realset results: chart, per-type table, and every call's
ASR transcript with each model's per-turn verdict. -> results/realset_view.html

    python -m realset.view
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from realset.metrics import NAMES, ORDER

MODELS = ORDER  # every model in data/realset/summary.json, ours first
RISK_ORDER = {"none": 0, "watch": 1, "alert": 2, "urgent": 3}


def main(path="results/realset_view.html"):
    calls = [json.loads(l) for l in open("data/realset/calls.jsonl") if l.strip()]
    summary = json.load(open("data/realset/summary.json"))
    models = [m for m in MODELS if m in summary]
    verdicts = {m: {} for m in models}
    for m in models:
        for l in open(f"data/realset/scores/{m}.jsonl"):
            r = json.loads(l)
            v = r.get("verdict") or {}
            verdicts[m][(r["call_id"], r["turn"])] = v.get("risk", "error") if v else "error"
    data = []
    for c in calls:
        row = {"id": c["call_id"], "label": c["label"], "type": c["scam_type"], "src": c["source"], "audio_s": c.get("audio_s"),
               "split": c.get("split", ""), "batch": c.get("batch", ""),
               "turns": [{"n": t["turn"], "sp": t["speaker"], "text": t["text"],
                          "v": {m: verdicts[m].get((c["call_id"], t["turn"])) for m in models} if t["speaker"] == "caller" else {}}
                         for t in c["turns"]]}
        row["flag"] = {m: any(RISK_ORDER.get(t["v"].get(m) or "none", 0) >= 2 for t in row["turns"]) for m in models}
        data.append(row)
    chart = Path("results/realset_chart.svg").read_text() if Path("results/realset_chart.svg").exists() else ""
    page = TEMPLATE.replace("__CHART__", chart).replace("__DATA__", json.dumps({"calls": data, "models": models, "names": {m: NAMES.get(m, m) for m in models}, "summary": summary}).replace("</", "<\\/"))
    Path(path).write_text(page)
    print(path, len(data), "calls")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realset Guardian Results</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--card:#ffffff;--scam:#eb6834;--benign:#2a78d6;--alert:#e34948;--watch:#c98500;--none:#9a998f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33332f;--card:#222220;--none:#77766e}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33332f;--card:#222220;--none:#77766e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}.chart{background:#fcfcfb;border:1px solid var(--line);border-radius:8px;overflow-x:auto}.chart svg{max-width:100%;height:auto;display:block}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}th{color:var(--ink2);font-weight:600}
.tbl{overflow-x:auto}.filters{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
select,input{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:6px 0;padding:8px 12px}summary{cursor:pointer;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{font-size:12px;padding:1px 8px;border-radius:10px;border:1px solid var(--line);color:var(--ink2)}.tag.scam{border-color:var(--scam)}.tag.benign{border-color:var(--benign)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:3px;vertical-align:middle}
.turn{display:grid;grid-template-columns:72px 1fr;gap:8px;padding:4px 0;border-top:1px dashed var(--line)}.who{color:var(--ink2);font-size:12px}
.v{font-size:11px;color:var(--ink2);margin-top:2px}.ok{color:var(--ink2)}.bad{font-weight:600}
</style></head><body><main>
<h1>Real-audio test set: fine-tuned guardian vs frontier baselines</h1>
<p class="sub" id="sub"></p>
<div class="chart">__CHART__</div>
<h2>Results</h2><div class="tbl"><table id="res"></table></div>
<h2>Recall by scam type</h2><div class="tbl"><table id="types"></table></div>
<h2>Calls</h2>
<div class="filters"><select id="fl"><option value="">All labels</option><option>scam</option><option>benign</option></select>
<select id="fs"><option value="">All splits</option><option>dev</option><option>test</option></select>
<select id="fo"><option value="">All outcomes</option><option value="wrong">Any model wrong</option><option value="ours">Ours wrong</option></select>
<input id="q" placeholder="Search transcript"></div>
<div id="calls"></div>
<p class="sub" style="margin-top:24px;font-size:12px">Sources: robocall audio from the FTC's Project Point of No Entry, packaged by S. Prasad and B. Reaves, "Robocall Audio from the FTC's Project Point of No Entry", NCSU TR-2023-1 (data in the public domain). HarperValleyBank, Gridspace-Stanford (CC BY 4.0, github.com/cricketclub/gridspace-stanford-harper-valley); transcripts shown here are our own ASR of that audio. No audio is included.</p>
</main>
<script>
const D=__DATA__;const C={none:'var(--none)',watch:'var(--watch)',alert:'var(--alert)',urgent:'var(--alert)',error:'var(--ink2)'};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=x=>Math.round(x*100)+'%';const S=D.summary;const N=m=>D.names[m]||m;const first=S[D.models[0]]||{};
document.getElementById('sub').textContent=`${D.calls.length} real calls (${first.scam_n} NCSU robocalls, ${first.benign_n} HarperValley bank calls) through 8 kHz mu-law and our live Nemotron ASR. Flag = any caller turn scored alert or urgent. Write-up: realset.md.`;
let h='<tr><th>Model</th><th>Scam recall</th><th>95% CI</th><th>False alarms</th><th>95% CI</th><th>Cost</th><th>p50 latency</th></tr>';
for(const m of D.models){const v=S[m];h+=`<tr${m.startsWith('emberz')?' style="font-weight:600"':''}><td>${N(m)}</td><td>${v.scam_k}/${v.scam_n} = <b>${pct(v.recall)}</b></td><td>${pct(v.recall_ci[0])}-${pct(v.recall_ci[1])}</td><td>${v.benign_k}/${v.benign_n} = <b>${pct(v.far)}</b></td><td>${pct(v.far_ci[0])}-${pct(v.far_ci[1])}</td><td>$${v.cost_usd.toFixed(2)}</td><td>${v.latency_p50_ms??'-'} ms</td></tr>`}
document.getElementById('res').innerHTML=h;
const types=[...new Set(Object.values(S).flatMap(v=>Object.keys(v.recall_by_type)))].sort();
h='<tr><th>Model</th>'+types.map(t=>`<th>${t}</th>`).join('')+'</tr>';for(const m of D.models)h+=`<tr${m.startsWith('emberz')?' style="font-weight:600"':''}><td>${N(m)}</td>`+types.map(t=>`<td>${S[m].recall_by_type[t]??'-'}</td>`).join('')+'</tr>';
document.getElementById('types').innerHTML=h;
function render(){const fs=document.getElementById('fs').value,fl=document.getElementById('fl').value,fo=document.getElementById('fo').value,q=document.getElementById('q').value.toLowerCase();let out='';
for(const c of D.calls){if(fl&&c.label!==fl)continue;if(fs&&c.split!==fs)continue;const wrong=D.models.some(m=>c.flag[m]!==(c.label==='scam'));if(fo==='wrong'&&!wrong)continue;
if(fo==='ours'&&!D.models.some(m=>m.startsWith('emberz')&&c.flag[m]!==(c.label==='scam')))continue;
if(q&&!c.turns.some(t=>t.text.toLowerCase().includes(q)))continue;
const badges=D.models.map(m=>{const ok=c.flag[m]===(c.label==='scam');return `<span class="tag ${ok?'ok':'bad'}">${esc(N(m))}: ${c.flag[m]?'flagged':'quiet'}${ok?'':' (wrong)'}</span>`}).join(' ');
let t='';for(const x of c.turns){const v=Object.entries(x.v).map(([m,r])=>`<span><span class="dot" style="background:${C[r]||C.none}"></span>${esc(N(m))} ${r??'-'}</span>`).join(' &nbsp; ');
t+=`<div class="turn"><div class="who">[${x.n}] ${x.sp}</div><div>${esc(x.text)}${v?`<div class="v">${v}</div>`:''}</div></div>`}
out+=`<details><summary><b>${esc(c.id)}</b><span class="tag ${c.label}">${c.label}</span><span class="tag">${esc(c.type)}</span><span class="tag">${esc(c.split)}</span>${badges}</summary>${t}</details>`}
document.getElementById('calls').innerHTML=out||'<p class="sub">No calls match.</p>'}
['fs','fl','fo','q'].forEach(i=>document.getElementById(i).addEventListener('input',render));render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
