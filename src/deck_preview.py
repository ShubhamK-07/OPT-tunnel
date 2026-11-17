"""Render deck.pptx to PNG previews without PowerPoint.

There is no LibreOffice in this environment, so this module draws the deck from
the shape tree: rectangles, text frames, tables and pictures, at 150 px/inch.
It is a layout check (geometry, wrapping, overlaps), not a pixel-exact render —
Calibri is substituted by DejaVu Sans, which is slightly wider, so anything that
fits here fits in PowerPoint.

Run:  python -m src.deck_preview            -> deck_preview/slide_XX.png
      python -m src.deck_preview --sheet    -> deck_preview/sheet_XX.png (2x2)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deck_preview"
DPI = 150
FONTS = {
    ("sans", False, False): "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ("sans", True, False): "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ("sans", False, True): "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ("sans", True, True): "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ("mono", False, False): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ("mono", True, False): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
}
_cache: dict = {}


def font(family: str, size_pt: float, bold: bool, italic: bool):
    key = (family, round(size_pt * DPI / 72), bold, italic)
    if key not in _cache:
        path = FONTS.get((family, bold, italic)) or FONTS[("sans", bold, italic)]
        _cache[key] = ImageFont.truetype(path, max(6, key[1]))
    return _cache[key]


def emu2px(v: float) -> float:
    return float(v) / 914400.0 * DPI


def _color(fill):
    try:
        if fill.type is not None and fill.fore_color.rgb is not None:
            rgb = fill.fore_color.rgb
            return (rgb[0], rgb[1], rgb[2], 255)
    except Exception:
        pass
    return None


def draw_rect(d: ImageDraw.ImageDraw, sh, box):
    x0, y0, x1, y1 = box
    fill = _color(sh.fill) if sh.fill.type is not None else None
    line = None
    try:
        if sh.line.fill.type is not None and sh.line.color and sh.line.color.rgb is not None:
            rgb = sh.line.color.rgb
            line = (rgb[0], rgb[1], rgb[2], 255)
    except Exception:
        line = None
    w = max(1, int(round(emu2px(sh.line.width) if sh.line.width else 1)))
    if fill or line:
        d.rectangle([x0, y0, x1, y1], fill=fill, outline=line, width=w)


def _runs(p):
    out = []
    for r in p.runs:
        size = r.font.size.pt if r.font.size else 12
        col = (24, 34, 48, 255)
        try:
            if r.font.color and r.font.color.rgb is not None:
                c = r.font.color.rgb
                col = (c[0], c[1], c[2], 255)
        except Exception:
            pass
        fam = "mono" if (r.font.name or "").lower().startswith("consol") else "sans"
        out.append((r.text, size, bool(r.font.bold), bool(r.font.italic), col, fam))
    return out


def draw_text(d: ImageDraw.ImageDraw, sh, box):
    x0, y0, x1, y1 = box
    tf = sh.text_frame
    width = x1 - x0
    cy = float(y0)
    for p in tf.paragraphs:
        runs = _runs(p)
        if not runs:
            cy += 8
            continue
        # group consecutive runs into words with their font
        words = []
        for text, size, bold, italic, col, fam in runs:
            for i, w in enumerate(text.split(" ")):
                if w == "" and i > 0:
                    continue
                words.append((w, size, bold, italic, col, fam))
        size0 = max(r[1] for r in runs)
        line_h = size0 * DPI / 72 * 1.22
        cx = float(x0)
        cur: list = []
        lines: list[list] = []
        space_w = font("sans", size0, False, False).getlength(" ") 
        for w in words:
            f = font(w[5], w[1], w[2], w[3])
            wpx = f.getlength(w[0])
            need = sum(font(c[5], c[1], c[2], c[3]).getlength(c[0]) + space_w for c in cur)
            if cur and need + wpx > width:
                lines.append(cur)
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(cur)
        align = p.alignment
        for ln in lines:
            lw = sum(font(c[5], c[1], c[2], c[3]).getlength(c[0]) + space_w for c in ln) - space_w
            if align == PP_ALIGN.CENTER:
                sx = x0 + (width - lw) / 2
            elif align == PP_ALIGN.RIGHT:
                sx = x1 - lw
            else:
                sx = cx
            for c in ln:
                f = font(c[5], c[1], c[2], c[3])
                asc = f.getbbox("Hg")[3]
                d.text((sx, cy + (line_h - asc) / 2 - 1), c[0], font=f, fill=c[4])
                sx += f.getlength(c[0]) + space_w
            cy += line_h
        cy += (p.space_after.pt * DPI / 72) if p.space_after else 0


def draw_table(d: ImageDraw.ImageDraw, sh, box):
    tbl = sh.table
    x0, y0 = box[0], box[1]
    col_x = [x0]
    for c in tbl.columns:
        col_x.append(col_x[-1] + emu2px(c.width))
    y = float(y0)
    for i, row in enumerate(tbl.rows):
        h = emu2px(row.height)
        for j, cell in enumerate(row.cells):
            cx0, cx1 = col_x[j], col_x[j + 1]
            fill = None
            try:
                if cell.fill.type is not None:
                    rgb = cell.fill.fore_color.rgb
                    fill = (rgb[0], rgb[1], rgb[2], 255)
            except Exception:
                pass
            d.rectangle([cx0, y, cx1, y + h], fill=fill, outline=(216, 223, 232, 255), width=1)
            for p in cell.text_frame.paragraphs:
                runs = _runs(p)
                if not runs:
                    continue
                text = "".join(r[0] for r in runs)
                size, bold, italic, col, fam = runs[0][1], runs[0][2], runs[0][3], runs[0][4], runs[0][5]
                f = font(fam, size, bold, italic)
                tw = f.getlength(text)
                pad = 0.07 * DPI
                align = p.alignment
                if align == PP_ALIGN.RIGHT:
                    tx = cx1 - pad - tw
                elif align == PP_ALIGN.CENTER:
                    tx = cx0 + (cx1 - cx0 - tw) / 2
                else:
                    tx = cx0 + pad
                d.text((tx, y + (h - size * DPI / 72) / 2 - 1), text, font=f, fill=col)
        y += h


def render(prs, idx, slide) -> Image.Image:
    W = int(emu2px(prs.slide_width))
    H = int(emu2px(prs.slide_height))
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img, "RGBA")
    for sh in slide.shapes:
        box = (emu2px(sh.left), emu2px(sh.top),
               emu2px(sh.left + sh.width), emu2px(sh.top + sh.height))
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            try:
                pic = Image.open(sh.image.blob and __import__("io").BytesIO(sh.image.blob))
                pic = pic.convert("RGB").resize((max(1, int(box[2] - box[0])),
                                                 max(1, int(box[3] - box[1]))), Image.LANCZOS)
                img.paste(pic, (int(box[0]), int(box[1])))
            except Exception as exc:
                d.rectangle(box, outline=(200, 200, 200), width=2)
                d.text((box[0] + 6, box[1] + 6), f"image error: {exc}", font=font("sans", 9, False, False),
                       fill=(200, 60, 60))
            continue
        if sh.has_table:
            draw_table(d, sh, box)
            continue
        draw_rect(d, sh, box)
        if sh.has_text_frame and sh.text_frame.text.strip():
            draw_text(d, sh, box)
    return img


def main():
    sheet = "--sheet" in sys.argv
    prs = Presentation(ROOT / "deck.pptx")
    OUT.mkdir(exist_ok=True)
    imgs = []
    for i, slide in enumerate(prs.slides, 1):
        img = render(prs, i, slide)
        p = OUT / f"slide_{i:02d}.png"
        img.save(p)
        imgs.append(img)
        print("wrote", p)
    if sheet:
        pad = 18
        w, h = imgs[0].size
        for k in range(0, len(imgs), 4):
            grp = imgs[k:k + 4]
            sh = Image.new("RGB", (2 * w + 3 * pad, 2 * h + 3 * pad), (235, 238, 242))
            for j, im in enumerate(grp):
                x = pad + (j % 2) * (w + pad)
                y = pad + (j // 2) * (h + pad)
                sh.paste(im, (x, y))
            p = OUT / f"sheet_{k // 4 + 1}.png"
            sh.save(p)
            print("wrote", p)


if __name__ == "__main__":
    main()
