"""Render ``report.md`` into a self-contained, print-ready ``report.html``.

Why this exists
---------------
The markdown report is the source of truth (it is generated from ``results/``
by ``src/report.py``).  Markdown alone looks like a README, so this module wraps
the same content in a deliberate typographic system: A4 print geometry, a cover
block, an auto-built table of contents, section rules, styled data tables and
boxed figures with captions.

The output is one file with every image embedded as a base64 data URI, so it can
be e-mailed, opened offline, or printed to PDF from any browser without the
``figures/`` folder.

Usage
-----
    python3 -m src.report_html            # report.html (self-contained)
    python3 -m src.report_html --link     # reference figures/ instead of embedding
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "report.md"
OUT = ROOT / "report.html"
FIG = ROOT / "figures"

# --------------------------------------------------------------------- CSS
CSS = r"""
:root{
  --navy:#0e1b2c; --ink:#182230; --slate:#55667c; --muted:#8a99ad;
  --rule:#d8dfe8; --paper:#ffffff; --off:#f4f7fb;
  --accent:#c0392b; --blue:#1f6feb; --green:#1e8e5a; --amber:#b87a08;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:#e9edf3; color:var(--ink);
  font-family:"Charter","Bitstream Charter","Iowan Old Style",Georgia,serif;
  font-size:10.6pt; line-height:1.62;
}
.page{
  max-width:190mm; margin:0 auto; background:var(--paper);
  padding:24mm 20mm 20mm; box-shadow:0 1px 3px rgba(16,32,56,.10);
}
/* ---------------------------------------------------------------- cover */
.cover{
  background:var(--navy); color:#fff; margin:-24mm -20mm 12mm; padding:26mm 20mm 16mm;
}
.cover .eyebrow{
  font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.6pt; letter-spacing:.16em;
  text-transform:uppercase; color:#9fb2c9; margin:0 0 10mm;
}
.cover h1{
  font-size:30pt; line-height:1.12; margin:0 0 6mm; font-weight:600; letter-spacing:-.01em;
  max-width:150mm;
}
.cover .sub{font-size:13pt; color:#c9d5e4; margin:0 0 3mm; max-width:150mm}
.cover .rule{width:46mm; height:2.6pt; background:var(--accent); margin:9mm 0}
.cover .meta{
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6mm 9mm;
  margin-top:6mm; border-top:.6pt solid rgba(255,255,255,.18); padding-top:7mm;
}
.cover .meta div span{
  display:block; font-family:"Helvetica Neue",Arial,sans-serif; font-size:7.4pt;
  letter-spacing:.13em; text-transform:uppercase; color:#8fa3bb; margin-bottom:1.6mm;
}
.cover .meta div b{font-size:11.4pt; font-weight:600; color:#f2f6fb}
.cover .note code{background:rgba(255,255,255,.10); color:#e8eef7}
.cover .note{
  font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.2pt; color:#9fb2c9;
  margin-top:9mm; line-height:1.5;
}
/* ------------------------------------------------------- running heads */
h1,h2,h3,h4{font-family:"Helvetica Neue",Arial,sans-serif; color:var(--navy)}
h2{
  font-size:16.5pt; margin:0 0 3mm; padding:0 0 2.4mm; border-bottom:1.4pt solid var(--navy);
  letter-spacing:-.005em; break-after:avoid;
}
h3{font-size:12.4pt; margin:7mm 0 2mm; break-after:avoid}
h4{font-size:10.8pt; margin:5mm 0 1.5mm; color:var(--slate); text-transform:uppercase;
   letter-spacing:.07em; break-after:avoid}
section{break-before:page}
section.first{break-before:auto}
p{margin:0 0 3.2mm; text-align:justify; hyphens:auto}
a{color:var(--blue); text-decoration:none}
strong{color:#0b1626}
em{color:var(--slate)}
hr{border:0; border-top:.6pt solid var(--rule); margin:7mm 0}
/* ------------------------------------------------------------- contents */
.toc{background:var(--off); border:.6pt solid var(--rule); padding:7mm 8mm; margin:0 0 9mm}
.toc h2{border:0; font-size:10pt; letter-spacing:.14em; text-transform:uppercase;
        color:var(--slate); padding:0; margin:0 0 4mm}
.toc ol{margin:0; padding-left:6mm; columns:2; column-gap:12mm}
.toc li{margin:0 0 1.8mm; font-size:9.6pt; line-height:1.35; break-inside:avoid}
.toc li a{color:var(--ink)}
.toc li.sub{list-style:none; margin-left:-3mm; font-size:8.8pt; color:var(--slate)}
/* --------------------------------------------------------------- tables */
table{
  width:100%; border-collapse:collapse; margin:3mm 0 5mm;
  font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.5pt; line-height:1.35;
  break-inside:avoid;
}
thead th{
  background:var(--navy); color:#fff; text-align:left; font-weight:600;
  padding:2mm 2.4mm; letter-spacing:.02em;
}
tbody td{padding:1.7mm 2.4mm; border-bottom:.6pt solid var(--rule); vertical-align:top}
tbody tr:nth-child(even){background:var(--off)}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums}
tbody td:first-child{color:var(--slate)}
table caption{caption-side:top; text-align:left; font-size:8.4pt; color:var(--slate);
             font-style:italic; padding:0 0 1.6mm}
/* -------------------------------------------------------------- figures */
figure{margin:5mm 0 6mm; break-inside:avoid}
figure img{width:100%; height:auto; display:block; border:.6pt solid var(--rule)}
figcaption{
  font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.5pt; color:var(--slate);
  margin-top:2mm; line-height:1.45; border-left:2pt solid var(--rule); padding-left:3mm;
}
/* ------------------------------------------------------------ call-outs */
blockquote{
  margin:4mm 0 5mm; padding:3.4mm 5mm; background:var(--off);
  border-left:2.6pt solid var(--accent); color:var(--ink); break-inside:avoid;
}
blockquote p{margin:0 0 2mm; text-align:left}
blockquote p:last-child{margin:0}
pre{
  background:#0e1b2c; color:#e6edf6; padding:4mm 5mm; overflow-x:auto;
  font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  font-size:8.4pt; line-height:1.5; margin:4mm 0 5mm; break-inside:avoid;
}
code{
  font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  font-size:9pt; background:var(--off); padding:.3mm 1.2mm; border-radius:2px;
  color:#123;
}
pre code{background:none; padding:0; color:inherit; font-size:inherit}
ul,ol{margin:0 0 3.4mm; padding-left:6mm}
li{margin:0 0 1.4mm}
li>p{margin-bottom:1.2mm}
/* numeric gutters in prose tables of the appendix */
.small{font-size:9pt}
/* ---------------------------------------------------------------- print */
@page{
  size:A4; margin:16mm 15mm 16mm;
  @bottom-left{
    content:"Underwater Tunnel Route Optimization · GEBCO_2026 · UTM 32N";
    font-family:"Helvetica Neue",Arial,sans-serif; font-size:7pt; color:#8a99ad;
  }
  @bottom-right{
    content:counter(page) " / " counter(pages);
    font-family:"Helvetica Neue",Arial,sans-serif; font-size:7pt; color:#55667c;
  }
}
@media print{
  body{background:#fff; font-size:9.6pt}
  .page{max-width:none; margin:0; padding:0; box-shadow:none}
  .cover{margin:0 0 8mm; padding:22mm 0 12mm}
  .toc{break-after:page}
  h2{break-before:page}
  section.first h2{break-before:auto}
  section{break-before:auto}
  figure,table,pre,blockquote{break-inside:avoid}
  a{color:var(--ink)}
}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="page">
{cover}
{toc}
{body}
</div>
</body>
</html>
"""

COVER_META = [
    ("dataset", "GEBCO_2026 · 15 arc-sec"),
    ("grid", "115 × 113 @ 450 m · UTM 32N"),
    ("formulations", "Dijkstra (snapped cells) · B-spline + SLSQP (local)"),
    ("figures", "See report figures"),
]


# ------------------------------------------------------------------ helpers
def _slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return text.strip("-")[:60]


def _embed(path: Path) -> str:
    """Inline a PNG as a data URI so the HTML is standalone."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _figure_html(md_src: str, embed: bool) -> str:
    """Turn stand-alone ```![alt](path)``` paragraphs into numbered <figure> blocks."""
    counter = {"n": 0}

    def repl(m: re.Match) -> str:
        counter["n"] += 1
        alt, src = m.group("alt"), m.group("src")
        p = (ROOT / src).resolve()
        if embed and p.exists():
            uri = _embed(p)
        else:
            uri = src
        cap = alt.strip()
        if cap:
            fid = re.search(r"fig(\d+)", Path(src).name)
            label = f"Fig. {int(fid.group(1))}" if fid else f"Figure {counter['n']}"
            cap = f"{label} — {cap[0].upper()}{cap[1:]}"
        cap_html = f"<figcaption>{html.escape(cap)}</figcaption>" if cap else ""
        return (f'\n<figure><img src="{uri}" alt="{html.escape(cap)}">'
                f"{cap_html}</figure>\n")
    return re.sub(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)", repl, md_src)


def _split_front_matter(md: str):
    """Pull the H1 / H3 / provenance note out of the markdown for the cover."""
    lines = md.splitlines()
    drop: set[int] = set()
    title = "Report"
    sub = ""
    note = ""
    sub_i = None
    for i, l in enumerate(lines):
        if l.startswith("# "):
            title = l[2:].strip()
            drop.add(i)
        elif l.startswith("### ") and sub_i is None:   # only the deck-style subtitle
            sub = l[4:].strip()
            sub_i = i
            drop.add(i)
    # the italic provenance block immediately after the subtitle (may wrap)
    if sub_i is not None:
        j = sub_i + 1
        while j < len(lines) and not lines[j].strip():
            drop.add(j)
            j += 1
        parts = []
        while j < len(lines) and lines[j].strip():
            parts.append(lines[j].strip().strip("*").strip())
            drop.add(j)
            j += 1
        note = " ".join(parts) if parts and parts[0] and not parts[0].startswith("**") else ""
        if not note:
            drop.difference_update(range(sub_i + 1, j))
    body = "\n".join(l for i, l in enumerate(lines) if i not in drop)
    return title, sub, note, body.strip("\n")


def _toc(md_body: str) -> str:
    items = []
    for m in re.finditer(r"^(##|###)\s+(.+)$", md_body, re.M):
        level, text = m.group(1), m.group(2).strip()
        cls = "" if level == "##" else ' class="sub"'
        items.append(f'<li{cls}><a href="#{_slug(text)}">{html.escape(text)}</a></li>')
    return ('<nav class="toc"><h2>Contents</h2><ol>'
            + "".join(items) + "</ol></nav>")


def _anchor_headings(body: str) -> str:
    def repl(m: re.Match) -> str:
        lvl, attrs, text = m.group(1), m.group(2) or "", m.group(3)
        return f'<h{lvl} id="{_slug(text)}"{attrs}>{text}</h{lvl}>'
    return re.sub(r"<h([23])([^>]*)>(.*?)</h\1>", repl, body)


def _numeric_cells(body: str) -> str:
    """Right-align columns that are mostly numbers (tables only)."""
    def fix_table(m: re.Match) -> str:
        tbl = m.group(0)
        cells = re.findall(r"<tr>(.*?)</tr>", tbl, re.S)
        if not cells:
            return tbl
        ncols = max(len(re.findall(r"<t[hd][^>]*>", row)) for row in cells)
        num = [True] * ncols
        for row in cells:
            tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
            for j, td in enumerate(tds):
                txt = re.sub(r"<[^>]+>", "", td).strip()
                if txt and not re.fullmatch(r"[-−+±×<>~ ]?[\d.,eE%/\s\-–+±×^]*", txt):
                    num[j] = False

        def mark_header(row: str) -> str:
            out, j = [], 0
            for cell in re.findall(r"<th[^>]*>.*?</th>", row):
                out.append(cell.replace("<th>", '<th class="num">', 1) if num[j] else cell)
                j += 1
            return "".join(out)

        def mark_row(row: str) -> str:
            parts = re.split(r"(<td[^>]*>.*?</td>)", row, flags=re.S)
            j = 0
            for k, part in enumerate(parts):
                if part.startswith("<td") and j < ncols:
                    if num[j]:
                        parts[k] = part.replace("<td>", '<td class="num">', 1)
                    j += 1
            return "".join(parts)

        tbl = re.sub(r"<tr>.*?</tr>",
                     lambda r: mark_row(r.group(0)) if "<td" in r.group(0) else mark_header(r.group(0)),
                     tbl, flags=re.S)
        return tbl
    return re.sub(r"<table>.*?</table>", fix_table, body, flags=re.S)


def build(embed: bool = True, out: Path = OUT) -> Path:
    raw = MD.read_text(encoding="utf-8")
    title, sub, note, body_md = _split_front_matter(raw)
    body_md = _figure_html(body_md, embed)
    body = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    body = _anchor_headings(body)
    body = _numeric_cells(body)
    # split into <section> at each h2 so print breaks land in the right place
    chunks = re.split(r"(?=<h2 )", body)
    chunks = ["<section>" + c + "</section>" if c.strip().startswith("<h2") else c
              for c in chunks]
    body = "".join(chunks).replace("<section><h2", '<section class="first"><h2', 1)

    note_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(note))
    stamp = _dt.datetime.now().strftime("%d %B %Y")
    meta = "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in COVER_META)
    cover = f"""<header class="cover">
  <p class="eyebrow">Course project · route optimisation · final report</p>
  <h1>{html.escape(title)}</h1>
  <p class="sub">{html.escape(sub.lstrip('#'))}</p>
  <div class="rule"></div>
  <div class="meta">{meta}</div>
  <p class="note">{note_html}<br>
  Rendered {stamp}. Self-contained: every figure is embedded in this file.</p>
</header>"""

    html_out = TEMPLATE.format(
        title=html.escape(title), css=CSS, cover=cover, toc=_toc(body_md), body=body)
    out.write_text(html_out, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true",
                    help="reference figures/ rather than embedding them (smaller file)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = build(embed=not a.link, out=Path(a.out))
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB, "
          f"{'self-contained' if not a.link else 'linked figures'})")


if __name__ == "__main__":
    main()
