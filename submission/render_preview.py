"""
Minimal PPTX -> PNG renderer for visual QA.

Not a general converter — it handles exactly what build_deck.js emits (solid-fill
rects, rounded rects, ellipses, lines, and styled text runs). It exists because
LibreOffice is unavailable here, and structural checks cannot show whether a deck
actually *looks* right.

Uses the real Calibri / Cambria from C:\\Windows\\Fonts, so text metrics are close
to what PowerPoint will produce.

    python render_preview.py deck.pptx out_dir
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
EMU = 914400.0
SCALE = 110                      # px per inch
SW, SH = 13.333, 7.5

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = {
    ("Calibri", False): "calibri.ttf", ("Calibri", True): "calibrib.ttf",
    ("Cambria", False): "cambria.ttc", ("Cambria", True): "cambriab.ttf",
    ("Arial", False): "arial.ttf",     ("Arial", True): "arialbd.ttf",
}
_cache: dict = {}


def font(name: str, size_pt: float, bold: bool):
    key = (name, round(size_pt, 1), bold)
    if key in _cache:
        return _cache[key]
    fn = FONT_FILES.get((name, bold)) or FONT_FILES[("Calibri", bold)]
    px = max(6, int(round(size_pt * SCALE / 72.0)))
    path = FONT_DIR / fn
    try:
        f = ImageFont.truetype(str(path), px)
    except Exception:
        try:
            f = ImageFont.truetype(str(FONT_DIR / "calibri.ttf"), px)
        except Exception:
            f = ImageFont.load_default()
    _cache[key] = f
    return f


def rgb(v: str | None, fallback=(0, 0, 0)):
    if not v or len(v) != 6:
        return fallback
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def px(inches: float) -> int:
    return int(round(inches * SCALE))


def shape_geo(sp):
    xf = sp.find(".//a:xfrm", NS)
    if xf is None:
        return None
    off, ext = xf.find("a:off", NS), xf.find("a:ext", NS)
    if off is None or ext is None:
        return None
    return {
        "x": int(off.get("x")) / EMU, "y": int(off.get("y")) / EMU,
        "w": int(ext.get("cx")) / EMU, "h": int(ext.get("cy")) / EMU,
        "flipH": xf.get("flipH") == "1", "flipV": xf.get("flipV") == "1",
    }


def wrap(draw, text, fnt, max_px):
    """Greedy wrap, honouring explicit newlines."""
    out = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        words, line = para.split(" "), ""
        for w in words:
            trial = (line + " " + w).strip()
            if draw.textlength(trial, font=fnt) <= max_px or not line:
                line = trial
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def render(pptx: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(pptx)
    slides = sorted([n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                    key=lambda s: int(re.search(r"\d+", s.split("/")[-1]).group()))
    made = []

    for idx, name in enumerate(slides, 1):
        root = ET.fromstring(z.read(name))

        bg = (255, 255, 255)
        bgel = root.find(".//p:bg//a:srgbClr", NS)
        if bgel is not None:
            bg = rgb(bgel.get("val"), bg)

        img = Image.new("RGB", (px(SW), px(SH)), bg)
        d = ImageDraw.Draw(img)

        for sp in root.iter():
            if sp.tag.split("}")[-1] != "sp":
                continue
            g = shape_geo(sp)
            if g is None:
                continue
            x0, y0 = px(g["x"]), px(g["y"])
            x1, y1 = px(g["x"] + g["w"]), px(g["y"] + g["h"])

            prst = sp.find(".//a:prstGeom", NS)
            kind = prst.get("prst") if prst is not None else None

            spPr = sp.find(".//p:spPr", NS)
            fill = None
            if spPr is not None:
                fc = spPr.find("./a:solidFill/a:srgbClr", NS)
                if fc is not None:
                    fill = rgb(fc.get("val"))
            lnc = spPr.find("./a:ln/a:solidFill/a:srgbClr", NS) if spPr is not None else None
            lncol = rgb(lnc.get("val")) if lnc is not None else None
            lnw = spPr.find("./a:ln", NS) if spPr is not None else None
            lnwidth = max(1, int(round(int(lnw.get("w", "12700")) / 12700 * SCALE / 72)))if lnw is not None else 1

            if kind == "line":
                ax, ay = (x1, y0) if g["flipH"] else (x0, y0)
                bx, by = (x0, y1) if g["flipH"] else (x1, y1)
                if g["flipV"]:
                    ay, by = by, ay
                d.line([(ax, ay), (bx, by)], fill=lncol or (120, 120, 120), width=lnwidth)
                continue
            if fill:
                if kind == "ellipse":
                    d.ellipse([x0, y0, x1, y1], fill=fill)
                elif kind == "roundRect":
                    r = min(18, (y1 - y0) // 4, (x1 - x0) // 4)
                    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill)
                else:
                    d.rectangle([x0, y0, x1, y1], fill=fill)

            # ---- text ----
            tx = sp.find(".//p:txBody", NS)
            if tx is None:
                continue
            bodyPr = tx.find("a:bodyPr", NS)
            valign = bodyPr.get("anchor") if bodyPr is not None else None

            # Runs inside ONE <a:p> flow inline in PowerPoint — concatenate them.
            # Only a new <a:p> starts a new line. (pptxgenjs breakLine:true emits
            # a new paragraph, so this matches what the deck will actually show.)
            lines = []
            for para in tx.findall("a:p", NS):
                pPr = para.find("a:pPr", NS)
                algn = pPr.get("algn") if pPr is not None else None
                runs = para.findall("a:r", NS)
                if not runs:
                    lines.append(("", None, algn, 12, ("Calibri", False)))
                    continue
                text, col, sz, face, bold = "", None, None, "Calibri", False
                for r in runs:
                    t = r.find("a:t", NS)
                    if t is None or t.text is None:
                        continue
                    text += t.text
                    if sz is None:                      # style from the first run
                        rPr = r.find("a:rPr", NS)
                        sz = float(rPr.get("sz")) / 100 if (rPr is not None and rPr.get("sz")) else 14.0
                        bold = (rPr is not None and rPr.get("b") == "1")
                        lt = r.find(".//a:latin", NS)
                        if lt is not None and lt.get("typeface"):
                            face = lt.get("typeface")
                        cel = r.find(".//a:solidFill/a:srgbClr", NS)
                        col = rgb(cel.get("val")) if cel is not None else (0, 0, 0)
                if text:
                    lines.append((text, col, algn, sz or 14.0, (face, bold)))

            if not lines:
                continue

            pad = 4
            avail = (x1 - x0) - 2 * pad
            rendered = []
            for text, col, algn, sz, fb in lines:
                face, bold = (fb if isinstance(fb, tuple) else ("Calibri", False))
                fnt = font(face, sz, bold)
                for ln in wrap(d, text, fnt, avail):
                    rendered.append((ln, col or (0, 0, 0), algn, fnt, sz))

            total_h = sum(int(r[4] * SCALE / 72 * 1.22) for r in rendered)
            cy = y0 + pad
            if valign == "ctr":
                cy = y0 + max(pad, ((y1 - y0) - total_h) // 2)

            for ln, col, algn, fnt, sz in rendered:
                lh = int(sz * SCALE / 72 * 1.22)
                tw = d.textlength(ln, font=fnt)
                lx = x0 + pad
                if algn == "ctr":
                    lx = x0 + ((x1 - x0) - tw) / 2
                elif algn == "r":
                    lx = x1 - pad - tw
                d.text((lx, cy), ln, font=fnt, fill=col)
                cy += lh

        p = out_dir / f"slide-{idx:02d}.png"
        img.save(p)
        made.append(p)
        print(f"  {p.name}")
    return made


if __name__ == "__main__":
    deck = Path(sys.argv[1] if len(sys.argv) > 1 else "WardSense_electronica_India.pptx")
    outd = Path(sys.argv[2] if len(sys.argv) > 2 else "preview")
    print(f"rendering {deck.name} -> {outd}/")
    files = render(deck, outd)
    print(f"\n{len(files)} slides rendered")
