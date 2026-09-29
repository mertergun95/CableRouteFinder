"""Vektor-Primitive aus einer PDF-Seite lesen.

Die Pläne (KLP/KÜP) kommen aus CAD-Systemen: Texte liegen dort nicht als Schrift,
sondern als gezeichnete Linienzüge vor. Deshalb wird hier alles als Primitive mit
Stil (Farbe, Strichstärke, Füllung) und abgetasteten Polylinien bereitgestellt.
Alle Koordinaten sind PDF-Punkte im *angezeigten* (rotierten) Seitenraum, also
dieselben Koordinaten, die PyMuPDF für ``page.rect`` und Clips verwendet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pymupdf


RED = "red"
BLACK = "black"
GRAY = "gray"
OTHER = "other"


def classify_color(rgb) -> str:
    if not rgb:
        return OTHER
    r, g, b = rgb[:3]
    if r > 0.8 and g < 0.3 and b < 0.3:
        return RED
    if r < 0.2 and g < 0.2 and b < 0.2:
        return BLACK
    if abs(r - g) < 0.05 and abs(g - b) < 0.05:
        return GRAY
    return OTHER


@dataclass
class Primitive:
    """Ein Zeichenobjekt der Seite (ein ``get_drawings``-Eintrag)."""

    index: int
    color: str
    width: float
    filled: bool
    polylines: list[np.ndarray]  # jede Polylinie: (n, 2)
    bbox: tuple[float, float, float, float]
    closed: bool = False

    @property
    def size(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(x1 - x0, y1 - y0)

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def segments(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for pl in self.polylines:
            for i in range(len(pl) - 1):
                yield pl[i], pl[i + 1]

    def length(self) -> float:
        return float(sum(np.linalg.norm(np.diff(pl, axis=0), axis=1).sum() for pl in self.polylines))


@dataclass
class PageVectors:
    width: float
    height: float
    primitives: list[Primitive] = field(default_factory=list)

    def select(self, color: str | None = None, width: float | None = None, filled: bool | None = None,
               tol: float = 0.03) -> list[Primitive]:
        out = []
        for p in self.primitives:
            if color is not None and p.color != color:
                continue
            if width is not None and abs(p.width - width) > tol:
                continue
            if filled is not None and p.filled != filled:
                continue
            out.append(p)
        return out

    def stroke_widths(self, color: str) -> dict[float, int]:
        counts: dict[float, int] = {}
        for p in self.primitives:
            if p.color == color and not p.filled:
                w = round(p.width, 2)
                counts[w] = counts.get(w, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _bezier(p0, p1, p2, p3, n: int = 6) -> list[tuple[float, float]]:
    t = np.linspace(0, 1, n)[1:]
    pts = []
    for tt in t:
        a = (1 - tt) ** 3
        b = 3 * (1 - tt) ** 2 * tt
        c = 3 * (1 - tt) * tt ** 2
        d = tt ** 3
        pts.append((a * p0.x + b * p1.x + c * p2.x + d * p3.x, a * p0.y + b * p1.y + c * p2.y + d * p3.y))
    return pts


def _items_to_polylines(items) -> list[np.ndarray]:
    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    def flush():
        nonlocal current
        if len(current) >= 2:
            polylines.append(current)
        current = []

    for it in items:
        kind = it[0]
        if kind == "l":
            a, b = it[1], it[2]
            if current and abs(current[-1][0] - a.x) < 1e-3 and abs(current[-1][1] - a.y) < 1e-3:
                current.append((b.x, b.y))
            else:
                flush()
                current = [(a.x, a.y), (b.x, b.y)]
        elif kind == "c":
            a = it[1]
            if not (current and abs(current[-1][0] - a.x) < 1e-3 and abs(current[-1][1] - a.y) < 1e-3):
                flush()
                current = [(a.x, a.y)]
            current.extend(_bezier(*it[1:5]))
        elif kind == "re":
            r = it[1]
            flush()
            polylines.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)])
        elif kind == "qu":
            q = it[1]
            flush()
            polylines.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y),
                              (q.ul.x, q.ul.y)])
    flush()
    return [np.asarray(pl, dtype=float) for pl in polylines]


def load_page_vectors(pdf_path: str, page_index: int = 0) -> PageVectors:
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    pv = PageVectors(width=page.rect.width, height=page.rect.height)
    # get_drawings liefert unrotierte Koordinaten; bei /Rotate in den Anzeigeraum drehen
    m = page.rotation_matrix
    rot = None if page.rotation == 0 else np.array([[m.a, m.b], [m.c, m.d]])
    shift = np.array([m.e, m.f])
    for i, d in enumerate(page.get_drawings()):
        stroke = d.get("color")
        fill = d.get("fill")
        filled = d.get("type") in ("f", "fs") and fill is not None
        color = classify_color(stroke if stroke else fill)
        polylines = _items_to_polylines(d["items"])
        if not polylines:
            continue
        if rot is not None:
            polylines = [pl @ rot + shift for pl in polylines]
            allp = np.vstack(polylines)
            r = pymupdf.Rect(*allp.min(axis=0), *allp.max(axis=0))
        else:
            r = d["rect"]
        pv.primitives.append(Primitive(
            index=i,
            color=color,
            width=float(d.get("width") or 0.0),
            filled=filled,
            polylines=polylines,
            bbox=(r.x0, r.y0, r.x1, r.y1),
            closed=bool(d.get("closePath")),
        ))
    doc.close()
    return pv
