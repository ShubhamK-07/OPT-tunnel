"""Design system for the presentation deck.

Everything visual lives here: palette, typography, the slide chrome (kicker,
title, hairline, footer with page number and data source), cards, stat tiles,
figure frames with captions, and styled tables.  `deck.py` only supplies
content, so the deck stays consistent by construction.

Typography notes
----------------
* Body face is Calibri (present on Windows/Office and substituted sensibly by
  Carlito in LibreOffice/Google Slides).
* Type scale: 25 / 13 / 12 / 10.5 / 9 pt.  Nothing smaller than 8.5 pt except
  the footer.
* Numbers that matter are set as stat tiles (34 pt), never inline in a
  paragraph.
"""

from __future__ import annotations

from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x0E, 0x1B, 0x2C)
INK = RGBColor(0x18, 0x22, 0x30)
SLATE = RGBColor(0x55, 0x66, 0x7C)
MUTED = RGBColor(0x8A, 0x99, 0xAD)
RULE = RGBColor(0xD8, 0xDF, 0xE8)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
OFFWHITE = RGBColor(0xF4, 0xF7, 0xFB)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)      # "hard" / emphasis
BLUE = RGBColor(0x1F, 0x6F, 0xEB)        # "easy"
GREEN = RGBColor(0x1E, 0x8E, 0x5A)       # pass / verified
AMBER = RGBColor(0xB8, 0x7A, 0x08)

BODY = "Calibri"
FIG = 11.55          # content width for a 13.333 in slide with 0.62 in margins
ML = 0.62            # left margin
MR = 0.62
INK_W = 13.333 - ML - MR


# ---------------------------------------------------------------- helpers
def _wrap(text: str, size_pt: float, width_in: float, cpi: float = 1.85) -> list[str]:
    """Rough greedy wrap.  cpi = characters per inch per point-of-font-size."""
    cap = max(8, int(width_in * cpi * 100.0 / size_pt))
    lines, cur = [], ""
    for word in str(text).split():
        trial = (cur + " " + word).strip()
        if len(trial) <= cap:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def est_height_in(text: str, size_pt: float, width_in: float, leading: float = 1.24,
                  extra_pt: float = 0.0) -> float:
    n = len(_wrap(text, size_pt, width_in))
    return (n * size_pt * leading + extra_pt) / 72.0


class Deck:
    def __init__(self, total: int = 12, running: str = "",
                 source: str = "GEBCO_2026 · UTM 32N · config.yaml"):
        from pptx import Presentation
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank = self.prs.slide_layouts[6]
        self.total = total
        self.running = running
        self.source = source
        self.warnings: list[str] = []

    # -- primitives ----------------------------------------------------
    def rect(self, s, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE,
             adj=None):
        sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        if fill is None:
            sh.fill.background()
        else:
            sh.fill.solid()
            sh.fill.fore_color.rgb = fill
        if line is None:
            sh.line.fill.background()
        else:
            sh.line.color.rgb = line
            sh.line.width = Pt(lw)
        if adj is not None:
            try:
                sh.adjustments[0] = adj
            except Exception:
                pass
        sh.shadow.inherit = False
        if sh.has_text_frame:
            sh.text_frame.text = ""
        return sh

    def hline(self, s, x, y, w, color=RULE, lw=0.75):
        return self.rect(s, x, y, w, 0.001, fill=color, lw=0)

    def text(self, s, x, y, w, h, text, size=12, bold=False, color=INK, align=PP_ALIGN.LEFT,
             italic=False, spacing=1.0, caps=False, anchor=MSO_ANCHOR.TOP, font=BODY,
             space_after=2):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = 0
        tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        lines = text if isinstance(text, (list, tuple)) else str(text).split("\n")
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = spacing
            p.space_after = Pt(space_after)
            r = p.add_run()
            r.text = line.upper() if caps else line
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = color
            r.font.name = font
        # fit check (only single-paragraph blocks are measured strictly)
        need = est_height_in(" ".join(str(t) for t in lines), size,
                            w) * (1.0 if len(lines) == 1 else 1.0)
        if need > h * 1.18 and size >= 10:
            self.warnings.append(f"text may overflow ({need:.2f}in > {h:.2f}in): "
                                 f"{str(lines[0])[:52]!r}")
        return tb

    def bullets(self, s, x, y, w, h, items, size=12.5, color=INK, gap=7, marker="—",
                marker_color=ACCENT, sub_size=None):
        """items: list of str or (str, level).  Level 1 renders as a quiet sub-line."""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        total = 0.0
        for i, it in enumerate(items):
            level = 0
            if isinstance(it, tuple):
                it, level = it
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.line_spacing = 1.12
            p.space_after = Pt(gap if level == 0 else max(2, gap - 3))
            p.space_before = Pt(0)
            if level == 0:
                m = p.add_run()
                m.text = marker + "  "
                m.font.size = Pt(size)
                m.font.bold = True
                m.font.color.rgb = marker_color
                m.font.name = BODY
            r = p.add_run()
            r.text = it
            r.font.size = Pt(size if level == 0 else (sub_size or size - 1.5))
            r.font.color.rgb = color if level == 0 else SLATE
            r.font.name = BODY
            total += est_height_in(it, size if level == 0 else (sub_size or size - 1.5),
                                   w - 0.18) + (gap if level == 0 else gap - 3) / 72.0
        if total > h * 1.15:
            self.warnings.append(f"bullets overflow ({total:.2f}in > {h:.2f}in) "
                                 f"at {str(items[0])[:40]!r}")
        return tb

    def figure(self, s, path, x, y, w, caption=None, frame=True, cap_w=None, h_pad=0.0):
        path = Path(path)
        if not path.exists():
            self.warnings.append(f"missing figure {path.name}")
            return None
        pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
        ph = pic.height / 914400
        if frame:
            self.rect(s, x - 0.05, y - 0.05, w + 0.10, ph + 0.10, fill=None, line=RULE, lw=0.75)
        if caption:
            self.text(s, x, y + ph + 0.10 + h_pad, cap_w or w, 0.42, caption, size=9,
                      color=SLATE, italic=True, spacing=1.1)
        return pic

    def stat(self, s, x, y, w, h, value, label, accent=INK, vsize=30, note=None,
             fill=OFFWHITE, line=RULE, align=PP_ALIGN.LEFT):
        self.rect(s, x, y, w, h, fill=fill, line=line)
        self.text(s, x + 0.16, y + 0.10, w - 0.30, 0.12, label, size=8.5, color=SLATE,
                  caps=True, bold=True, spacing=1.0)
        self.text(s, x + 0.16, y + 0.30, w - 0.30, 0.55, value, size=vsize, bold=True,
                  color=accent, anchor=MSO_ANCHOR.MIDDLE)
        if note:
            self.text(s, x + 0.16, y + h - 0.30, w - 0.30, 0.24, note, size=8.5, color=MUTED)

    def card(self, s, x, y, w, h, title=None, fill=PAPER, line=RULE, title_color=SLATE,
             kicker=None):
        self.rect(s, x, y, w, h, fill=fill, line=line)
        yy = y + 0.16
        if kicker:
            self.text(s, x + 0.20, yy, w - 0.40, 0.18, kicker, size=8.5, bold=True,
                      color=AMBER, caps=True)
            yy += 0.24
        if title:
            self.text(s, x + 0.20, yy, w - 0.40, 0.26, title, size=12.5, bold=True,
                      color=title_color)
            yy += 0.34
        return yy

    def table(self, s, x, y, w, rows, col_w=None, size=10, row_h=0.30,
              header_fill=NAVY, align=None, zebra=True, first_col_bold=False,
              highlight_rows=()):
        n_r, n_c = len(rows), len(rows[0])
        shp = s.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * n_r))
        tbl = shp.table
        if col_w:
            for j, cw in enumerate(col_w):
                tbl.columns[j].width = Inches(cw)
        for i in range(n_r):
            tbl.rows[i].height = Inches(row_h)
            for j in range(n_c):
                c = tbl.cell(i, j)
                c.text = str(rows[i][j])
                c.margin_left = Inches(0.07)
                c.margin_right = Inches(0.07)
                c.margin_top = Inches(0.02)
                c.margin_bottom = Inches(0.02)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                p = c.text_frame.paragraphs[0]
                p.line_spacing = 1.0
                if align and j < len(align):
                    p.alignment = align[j]
                r = p.runs[0] if p.runs else p.add_run()
                r.font.size = Pt(size)
                r.font.name = BODY
                if i == 0:
                    r.font.bold = True
                    r.font.color.rgb = PAPER
                    c.fill.solid()
                    c.fill.fore_color.rgb = header_fill
                else:
                    r.font.bold = bool(first_col_bold and j == 0) or (i in highlight_rows)
                    r.font.color.rgb = INK
                    c.fill.solid()
                    c.fill.fore_color.rgb = (OFFWHITE if (zebra and i % 2 == 0) else PAPER)
        return tbl

    # -- slide chrome --------------------------------------------------
    def title_slide(self):
        return self.prs.slides.add_slide(self.blank)

    def slide(self, page, kicker, title, subtitle=None, subsize=12.5):
        s = self.prs.slides.add_slide(self.blank)
        self.rect(s, 0, 0, 13.333, 0.075, fill=NAVY)          # top rule
        y = 0.42
        self.text(s, ML, y, 6.0, 0.18, kicker, size=8.5, bold=True, color=ACCENT, caps=True)
        y += 0.24
        self.text(s, ML, y, 11.2, 0.44, title, size=25, bold=True, color=NAVY, spacing=0.98)
        y += 0.52
        if subtitle:
            self.text(s, ML, y, 11.6, 0.30, subtitle, size=subsize, color=SLATE, spacing=1.08)
            y += 0.30 + 0.10 * (len(subtitle) > 95)
        self.hline(s, ML, 1.62, INK_W, RULE, 0.9)
        self.footer(s, page)
        return s

    def footer(self, s, page, source=None):
        self.hline(s, ML, 7.03, INK_W, RULE, 0.75)
        left = self.running or ""
        if source or self.source:
            left = (left + "   ·   " if left else "") + (source or self.source)
        self.text(s, ML, 7.12, 9.6, 0.2, left, size=8, color=MUTED)
        self.text(s, 13.333 - MR - 1.6, 7.12, 1.6, 0.2,
                  f"{page:02d} / {self.total}", size=8.5, color=SLATE, bold=True,
                  align=PP_ALIGN.RIGHT)

    def note(self, s, text):
        s.notes_slide.notes_text_frame.text = text

    def save(self, path):
        self.prs.save(path)
        return len(self.prs.slides._sldIdLst)
