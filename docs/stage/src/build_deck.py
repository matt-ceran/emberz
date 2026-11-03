import json, re, sys, html, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
order = json.loads((src/"project/deck.json").read_text())["order"]
fonts = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400..800&family=DM+Sans:wght@400..700&display=swap">'
base = """@page{size:1920px 1080px;margin:0}*{box-sizing:border-box}html,body{margin:0;padding:0;background:#14100D}
h1,h2,h3,p,ul,table{margin:0}section{width:1920px;height:1080px;position:relative;overflow:hidden;break-after:page;page-break-after:always}
section:last-of-type{break-after:auto}aside{display:none}img{display:block}"""
slides, notes = [], []
for i, sid in enumerate(order, 1):
    s = (src/f"project/slides/{sid}.html").read_text()
    n = len(order)
    num = "" if sid in ("cover","close") else f'<p style="position:absolute;right:48px;bottom:40px;font-family:DM Sans,Arial,sans-serif;font-size:20px;opacity:.55">{i} / {n}</p>'
    slides.append(s.replace("<aside>", num + "<aside>", 1))
    m = re.search(r"<aside>(.*?)</aside>", s, re.S)
    t = re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S)
    title = re.sub(r"<[^>]+>", "", t.group(1)) if t else sid
    notes.append(f'<div class="n"><p class="k">Slide {i} · {html.escape(sid)}</p><h2>{title}</h2><p>{m.group(1) if m else ""}</p></div>')
(out/"emberz_stage_deck.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Emberz — Stage Deck</title>{fonts}<style>{base}</style></head><body>{"".join(slides)}</body></html>')
nstyle = """@page{size:A4;margin:18mm}body{font-family:'DM Sans',Arial,sans-serif;color:#1C1612;font-size:13pt;line-height:1.5}
.n{break-inside:avoid;padding:0 0 14pt;margin:0 0 14pt;border-bottom:1px solid #E6DCCF}.k{color:#B8430F;font-weight:700;font-size:10pt;letter-spacing:1px;text-transform:uppercase;margin:0}
h1{font-family:'Bricolage Grotesque',Arial,sans-serif;font-size:24pt;margin:0 0 16pt}h2{font-family:'Bricolage Grotesque',Arial,sans-serif;font-size:15pt;margin:2pt 0 6pt}"""
(out/"emberz_speaker_notes.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Emberz — Speaker Notes</title>{fonts}<style>{nstyle}</style></head><body><h1>Emberz — speaker notes</h1>{"".join(notes)}</body></html>')
