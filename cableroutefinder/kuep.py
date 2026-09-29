"""Analyse des Kabelübersichtsplans (KÜP).

Im KÜP steht jedes Kabel als waagerechte Linie mit der Kabelnummer darüber
(z. B. "S1307010"), Querschnitt/Adern darunter ("20x1x0,9") und der Länge in
der Mitte ("30"). Die Linie beginnt/endet an einem Kasten (Kabelschrank KS,
Signalkabelverteiler SK, Kabelverteiler KV ...) oder an einem Element, dessen
Name rechts daneben steht ("13W22", "13P8", "13G2003/13W29" ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .klp import normalize_cable_id
from .pdfvector import BLACK, PageVectors, Primitive
from .textocr import TextLine, find_text_lines, ocr_lines

_LENGTH_RE = re.compile(r"^\d{1,4}$")
_SECTION_RE = re.compile(r"^\d+x\d+x[\d,]+")
_NAME_RE = re.compile(r"[A-Z]")


@dataclass
class KuepBox:
    bbox: tuple[float, float, float, float]
    name: str = ""

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class KuepCable:
    cable_id: str
    from_name: str = ""
    to_name: str = ""
    length_m: float | None = None
    cross_section: str = ""
    line: tuple[float, float, float] | None = None   # (x0, x1, y) der Kabellinie im KÜP
    text_bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict:
        return {"id": self.cable_id, "from": self.from_name, "to": self.to_name,
                "length_m": self.length_m, "cross_section": self.cross_section}


@dataclass
class KuepResult:
    cables: list[KuepCable] = field(default_factory=list)
    boxes: list[KuepBox] = field(default_factory=list)
    lines: list[TextLine] = field(default_factory=list)


def _is_axis_rect(p: Primitive) -> bool:
    if len(p.polylines) != 1:
        return False
    pl = p.polylines[0]
    if len(pl) not in (4, 5):
        return False
    xs, ys = pl[:, 0], pl[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    on_edge = (np.minimum(np.abs(xs - x0), np.abs(xs - x1)) < 0.3) & (np.minimum(np.abs(ys - y0), np.abs(ys - y1)) < 0.3)
    return bool(on_edge.all())


def _is_long_axis_stroke(p: Primitive) -> bool:
    """Waagerechte/senkrechte Einzelstriche > 6,5 pt sind Kabellinien oder Teilstriche, keine Schrift."""
    if p.size < 6.5 or len(p.polylines) != 1 or len(p.polylines[0]) != 2:
        return False
    d = p.polylines[0][1] - p.polylines[0][0]
    return abs(d[0]) < 0.2 or abs(d[1]) < 0.2


def _clean(text: str) -> str:
    t = text.replace("$", "S").replace("—", "-").strip(" .:;,'\"|")
    return re.sub(r"\s+", " ", t)


def _row_text(lines: list[TextLine]) -> str:
    return normalize_name(" ".join(_clean(l.text) for l in sorted(lines, key=lambda l: l.bbox[0]) if _clean(l.text)))


_TYPE_PREFIXES = ("KS", "SK", "KV", "ESTW")


def _fix_element_id(tok: str) -> str:
    """Typische OCR-Verwechslungen in Elementkennungen ("13G4008", "13W21A") korrigieren."""
    tok = re.sub(r"(?<=\d)\.(?=\d)", "", tok)                 # "1364.004" -> "1364004"
    tok = re.sub(r"(^|/)13G6(\d{4})(?=$|/)", r"\g<1>13G\2", tok)  # "13G62003" -> "13G2003"
    tok = re.sub(r"(^|/)136(\d{4})(?=$|/)", r"\g<1>13G\2", tok)   # G als 6 gelesen
    return tok


def normalize_name(text: str) -> str:
    """"KS 1307000 (600)" -> "KS 1307000", "SK 13P 16" -> "SK 13P16"."""
    toks = [t for t in re.split(r"\s+", text.strip()) if t]
    # Klammerwerte / Nachsatzzahlen (Aderzahl des Kastens) entfernen, wenn die Kennung davor
    # schon vollständig ist ("KS 1307000 (600)"), sonst ist es ein Teil davon ("13P 16")
    while len(toks) > 1 and re.fullmatch(r"\(?\d{1,4}\)?", toks[-1]) and (
            toks[-1].startswith("(") or re.fullmatch(r"\d{5,}", toks[-2])):
        toks.pop()
    toks = [_fix_element_id(t) for t in toks]
    text = " ".join(toks)
    # Elementkennungen ohne Leerzeichen: "13W 35/13W 38" -> "13W35/13W38"
    text = re.sub(r"(13[A-Z]{1,2}) (\d)", r"\1\2", text)
    # Kabelschrank-Bezeichnung anderer Blätter: "S$1306000" / "XSS1306000" -> "KS 1306000"
    m = re.fullmatch(r"[A-Z$]*[S$](\d{7})", text.replace(" ", ""))
    if m:
        return "KS " + m.group(1)
    toks = text.split()
    if len(toks) >= 2 and toks[0].upper() in _TYPE_PREFIXES:
        return toks[0].upper() + " " + "".join(toks[1:])
    return " ".join(toks)


def _is_name(text: str) -> bool:
    t = _clean(text)
    if not t or normalize_cable_id(t) or _SECTION_RE.match(t) or _LENGTH_RE.match(t):
        return False
    if t.startswith("(") or t.lower().startswith("km"):
        return False
    return bool(_NAME_RE.search(t.upper())) and len(t) >= 2


def parse_kuep(pv: PageVectors, log=print) -> KuepResult:
    res = KuepResult()
    inside = [p for p in pv.primitives if p.color == BLACK
              and p.bbox[0] >= -1 and p.bbox[1] >= -1 and p.bbox[2] <= pv.width + 1 and p.bbox[3] <= pv.height + 1]
    # Lange waagerechte Linien (Kabellinien); Teilstriche kreuzen sie senkrecht
    long_h = []
    for p in inside:
        for pl in p.polylines:
            for a, c in zip(pl[:-1], pl[1:]):
                if abs(a[1] - c[1]) < 0.25 and abs(a[0] - c[0]) > 12:
                    long_h.append((min(a[0], c[0]), max(a[0], c[0]), a[1]))

    def is_tick(p: Primitive) -> bool:
        if not _is_long_axis_stroke(p):
            return False
        (a, c) = p.polylines[0]
        if abs(a[1] - c[1]) < 0.2:          # waagerecht: Stück einer Linie
            return True
        x, y0, y1 = a[0], min(a[1], c[1]), max(a[1], c[1])
        return any(h0 - 0.5 <= x <= h1 + 0.5 and y0 - 0.5 <= hy <= y1 + 0.5 for h0, h1, hy in long_h)

    glyphs = [p for p in inside if p.size < 9.5 and not p.filled and not is_tick(p)
              and not (p.size >= 6.5 and _is_axis_rect(p))]
    lines = [l for l in find_text_lines(glyphs, char_gap=1.3, horizontal_only=True) if len(l.prims) >= 1]
    ocr_lines(lines)
    lines = [l for l in lines if l.text and abs(l.angle) < 20]
    res.lines = lines

    # Kästen
    big = [p for p in inside if p.size >= 6.5 and not p.filled]
    rects = [p for p in big if _is_axis_rect(p)]
    page_frame = pv.width * 0.5
    boxes = [KuepBox(bbox=p.bbox) for p in rects if (p.bbox[2] - p.bbox[0]) < page_frame
             and (p.bbox[2] - p.bbox[0]) < 60]
    for b in boxes:
        x0, y0, x1, y1 = b.bbox
        cands = [l for l in lines if _is_name(l.text)
                 and y0 - 16 <= l.bbox[3] <= y0 + 0.5 and l.bbox[2] >= x0 - 25 and l.bbox[0] <= x1 + 40]
        if cands:
            nearest_bottom = max(l.bbox[3] for l in cands)
            row = [l for l in lines if abs(l.bbox[3] - nearest_bottom) < 2.5
                   and l.bbox[2] >= x0 - 40 and l.bbox[0] <= x1 + 80 and not l.text.startswith("(")]
            b.name = _row_text(row)
    # Hohe Sammelkästen ohne Namen (Fortsetzung über ①/②) erben den Namen des vorherigen
    tall = sorted([b for b in boxes if b.height > pv.height * 0.3], key=lambda b: b.bbox[0])
    last = ""
    for b in tall:
        if b.name and re.search(r"\d{3,}", b.name):
            last = b.name
        elif last:
            b.name = last
    res.boxes = boxes

    # Waagerechte Linienstücke
    hsegs = []
    for p in big:
        for pl in p.polylines:
            for a, c in zip(pl[:-1], pl[1:]):
                if abs(a[1] - c[1]) < 0.25 and abs(a[0] - c[0]) > 2.0:
                    hsegs.append((min(a[0], c[0]), max(a[0], c[0]), (a[1] + c[1]) / 2))

    def chain(seg):
        x0, x1, y = seg
        grown = True
        while grown:
            grown = False
            for s in hsegs:
                if abs(s[2] - y) < 0.3:
                    if s[0] < x0 and s[1] >= x0 - 1.2:
                        x0, grown = s[0], True
                    if s[1] > x1 and s[0] <= x1 + 1.2:
                        x1, grown = s[1], True
        return x0, x1, y

    def box_at(x, y, side):
        for b in boxes:
            bx0, by0, bx1, by1 = b.bbox
            if by0 - 0.5 <= y <= by1 + 0.5:
                edge = bx1 if side == "left" else bx0
                if abs(edge - x) < 1.5:
                    return b
        return None

    used_text: set[int] = set()

    def text_beyond(x, y, side, exclude):
        cands = []
        for l in lines:
            if id(l) in exclude or not _is_name(l.text):
                continue
            lx0, ly0, lx1, ly1 = l.bbox
            cy = (ly0 + ly1) / 2
            if abs(cy - y) > 14:
                continue
            gap = (lx0 - x) if side == "right" else (x - lx1)
            if -3 <= gap <= 80:
                cands.append((gap + abs(cy - y) * 2, l))
        if not cands:
            return ""
        cands.sort(key=lambda c: c[0])
        best = cands[0][1]
        # Mehrzeilige Bezeichnung ("ESTW-A SKL / SKL 10/16 / Blatt 37"): oberste Zeile ist der Name
        moved = True
        while moved:
            moved = False
            for m in lines:
                if m is best or not _clean(m.text):
                    continue
                h = best.bbox[3] - best.bbox[1]
                if abs(m.bbox[0] - best.bbox[0]) < 3 and 0 <= best.bbox[1] - m.bbox[3] <= 1.6 * h:
                    best, moved = m, True
                    break
        row = [m for m in lines if abs(m.bbox[1] - best.bbox[1]) < 1.5 and 0 <= m.bbox[0] - best.bbox[2] < 12]
        return normalize_name(_row_text([best] + row))

    cable_lines = [(normalize_cable_id(_clean(l.text)), l) for l in lines]
    cable_lines = [(cid, l) for cid, l in cable_lines if cid]
    for cid, l in cable_lines:
        tx0, ty0, tx1, ty1 = l.bbox
        cands = [s for s in hsegs if ty1 - 0.5 <= s[2] <= ty1 + 7 and s[0] <= tx1 and s[1] >= tx0 - 5]
        cable = KuepCable(cable_id=cid, text_bbox=l.bbox)
        if cands:
            seg = chain(min(cands, key=lambda s: s[2] - ty1))
            x0, x1, y = seg
            cable.line = seg
            lb = box_at(x0, y, "left")
            rb = box_at(x1, y, "right")
            # Unbenannte kleine Kästen sind Elementsymbole – der Name steht dahinter
            cable.from_name = lb.name if lb and lb.name else text_beyond(lb.bbox[0] if lb else x0, y, "left", {id(l)})
            cable.to_name = rb.name if rb and rb.name else text_beyond(rb.bbox[2] if rb else x1, y, "right", {id(l)})
            # Länge über der Linie, Querschnitt unter der Linie
            above = [m for m in lines if _LENGTH_RE.match(_clean(m.text)) and y - 10 <= m.bbox[3] <= y + 0.5
                     and x0 < m.bbox[0] < x1 and m.bbox[0] > tx1]
            if above:
                cable.length_m = float(_clean(min(above, key=lambda m: m.bbox[0] - tx1).text))
            below = [m for m in lines if _SECTION_RE.match(_clean(m.text).replace(" ", ""))
                     and y - 0.5 <= m.bbox[1] <= y + 10 and abs(m.bbox[0] - tx0) < 5]
            if below:
                cable.cross_section = _clean(below[0].text)
        if cable.line is None:
            continue  # Kennung ohne Kabellinie (z. B. Kastenbezeichnung) – kein Kabel dieses Blatts
        known = next((c for c in res.cables if c.cable_id == cid), None)
        if known is None:
            res.cables.append(cable)
        elif known.line is None and cable.line is not None:
            res.cables[res.cables.index(known)] = cable
    log(f"KÜP: {len(res.cables)} Kabel, {len(boxes)} Kästen erkannt")
    return res
