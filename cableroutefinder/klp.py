"""Analyse des Kabellageplans (KLP).

Im KLP sind
* die Kabeltrassen rote Strichlinien (Kabelkanäle als rote Doppel-Strichlinie),
* die Kabelnummern rote Textstapel ("S1307010" ...), die über eine rote
  Hinweislinie mit einem Punkt auf der Trasse verbunden sind.
Ein Textstapel sagt also: "An diesem Trassenpunkt liegen diese Kabel".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .pdfvector import PageVectors, Primitive, RED
from .textocr import TextLine, cluster_boxes, find_text_lines, ocr_lines

CABLE_RE = re.compile(r"S\d{7}")
_CABLE_LOOSE_RE = re.compile(r"^[S5$]?(\d{7})$")


@dataclass
class KlpStyle:
    """Strichstärken (PDF-Punkte) der einzelnen Planebenen. ``None`` = automatisch erkennen."""

    trasse_width: float | None = None     # rote Strichlinie der Trasse
    kanal_width: float | None = None      # rote Doppel-Strichlinie (Kabelkanal)
    label_width: float | None = None      # rote Kabelnummern + Hinweislinien
    glyph_max_size: float = 6.5           # größere Objekte sind keine Schriftzeichen


@dataclass
class CableLabel:
    """Ein Kabelnummern-Stapel mit Bezugspunkt auf der Trasse."""

    cables: list[str]
    bbox: tuple[float, float, float, float]
    anchor: np.ndarray | None             # Endpunkt der Hinweislinie (auf der Trasse)
    leader: list[np.ndarray] = field(default_factory=list)
    lines: list[TextLine] = field(default_factory=list)


def normalize_cable_id(text: str, allow_missing_prefix: bool = False) -> str | None:
    """"S1307010", "$1307010" -> "S1307010". Ohne Präfix nur, wenn ausdrücklich erlaubt
    (in KLP-Kabelstapeln stehen ausschließlich Kabelnummern)."""
    t = text.strip().replace(" ", "").upper()
    if CABLE_RE.fullmatch(t):
        return t
    m = _CABLE_LOOSE_RE.match(t)
    if m and (allow_missing_prefix or t[:1] in "S5$"):
        return "S" + m.group(1)
    return None


def _is_straight(p: Primitive) -> bool:
    return len(p.polylines) == 1 and len(p.polylines[0]) == 2


def _dash_like(p: Primitive, lo: float = 2.5, hi: float = 14.0) -> bool:
    return _is_straight(p) and lo <= p.length() <= hi


def detect_style(pv: PageVectors) -> KlpStyle:
    """Ermittelt die Strichstärken aus der Statistik der roten Objekte."""
    by_width: dict[float, list[Primitive]] = {}
    for p in pv.primitives:
        if p.color == RED and not p.filled and p.width > 0:
            by_width.setdefault(round(p.width, 2), []).append(p)
    style = KlpStyle()
    if not by_width:
        return style
    # Trasse: Stärke mit den meisten strichartigen Einzellinien, bevorzugt die dickste
    dash_counts = {w: sum(1 for p in ps if _dash_like(p, 4.0, 12.0)) for w, ps in by_width.items()}
    ranked = sorted(dash_counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    style.trasse_width = ranked[0][0]
    # Beschriftung: Stärke mit den meisten kleinen Objekten (Schrift) UND langen Hinweislinien
    def label_score(ps):
        small = sum(1 for p in ps if p.size < 6.5)
        long_ = sum(1 for p in ps if p.size > 20 and len(p.polylines) == 1 and len(p.polylines[0]) <= 6)
        return small * (1 if long_ > 3 else 0.01)
    style.label_width = max(by_width, key=lambda w: label_score(by_width[w]))
    # Kabelkanal: nächstbeste strichartige Stärke (optional)
    for w, cnt in ranked[1:]:
        if w not in (style.label_width,) and cnt > 30:
            style.kanal_width = w
            break
    return style


def _outline_dash_centerline(p: Primitive) -> Primitive | None:
    """Als Umriss gezeichnete Striche (Strichstärke 0) auf ihre Mittellinie reduzieren."""
    if len(p.polylines) != 1:
        return None
    pts = p.polylines[0]
    if len(pts) < 4:
        return None
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c)
    axis = vt[0]
    proj = (pts - c) @ axis
    length = proj.max() - proj.min()
    thickness = np.ptp((pts - c) @ vt[1])
    if not (3.0 <= length <= 14.0 and thickness <= 2.5 and length > 2.5 * thickness):
        return None
    a, b = c + axis * proj.min(), c + axis * proj.max()
    return Primitive(index=p.index, color=p.color, width=thickness, filled=False,
                     polylines=[np.array([a, b])], bbox=p.bbox)


def _collinear_neighbors(prims: list[Primitive], max_gap: float = 6.0, max_angle_deg: float = 8.0,
                         min_run: int = 2) -> list[Primitive]:
    """Behält nur Striche, die in einer Reihe fluchtender Striche liegen (Strichlinie statt Schrift).
    ``min_run``: Mindestanzahl Striche je Reihe."""
    from scipy.spatial import cKDTree
    if not prims:
        return []
    ends, owner = [], []
    for i, p in enumerate(prims):
        a, b = p.polylines[0][0], p.polylines[0][-1]
        ends += [a, b]
        owner += [i, i]
    tree = cKDTree(np.array(ends))
    cos_tol = np.cos(np.radians(max_angle_deg))
    pairs: list[tuple[int, int]] = []
    for i, p in enumerate(prims):
        a, b = p.polylines[0][0], p.polylines[0][-1]
        d = b - a
        d = d / (np.linalg.norm(d) + 1e-9)
        for e in (a, b):
            for j in tree.query_ball_point(e, max_gap):
                k = owner[j]
                if k == i:
                    continue
                q = prims[k].polylines[0]
                d2 = q[-1] - q[0]
                d2 = d2 / (np.linalg.norm(d2) + 1e-9)
                if abs(float(d @ d2)) < cos_tol:
                    continue
                # seitlicher Versatz klein?
                w = ends[j] - a
                off = abs(d[0] * w[1] - d[1] * w[0])
                if off < 0.8:
                    pairs.append((i, k))
                    break
    # Reihen (Zusammenhangskomponenten) mit genügend Strichen behalten
    parent = list(range(len(prims)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    sizes: dict[int, int] = {}
    for a, b in pairs:
        for n in (a, b):
            sizes[find(n)] = 0
    members = {n for ab in pairs for n in ab}
    for n in members:
        sizes[find(n)] += 1
    return [prims[i] for i in sorted(members) if sizes[find(i)] >= min_run]


def trasse_primitives(pv: PageVectors, style: KlpStyle, labels: list["CableLabel"] | None = None,
                      exclude: set[int] | None = None) -> tuple[list[Primitive], list[Primitive]]:
    """Liefert (Trassenstriche, Kanalstriche).

    Kanalstriche (Doppel-Strichlinien, z. B. Querungen unter den Gleisen) können dieselbe
    Strichstärke wie die Beschriftung haben. Deshalb werden sie über ihre Form erkannt
    (fluchtende Striche in einer Reihe), nicht nur über die Strichstärke; Schrift der
    Kabelnummern-Stapel (``labels``/``exclude``) wird vorher ausgenommen."""
    exclude = exclude or set()
    boxes = [(l.bbox[0] - 2, l.bbox[1] - 2, l.bbox[2] + 2, l.bbox[3] + 2) for l in (labels or [])]

    def in_label(p: Primitive) -> bool:
        x0, y0, x1, y1 = p.bbox
        return any(x0 >= b[0] and y0 >= b[1] and x1 <= b[2] and y1 <= b[3] for b in boxes)

    trasse: list[Primitive] = []
    if style.trasse_width is not None:
        trasse = [p for p in pv.primitives if p.color == RED and not p.filled
                  and abs(p.width - style.trasse_width) < 0.03]
    # Als Umriss oder gefüllte Fläche gezeichnete Striche (Strichstärke 0) auf ihre Mittellinie
    # reduzieren; nur Reihen fluchtender Striche gelten als Trasse (nicht Symbolteile).
    outline = []
    for p in pv.primitives:
        if p.color == RED and p.width < 0.05 and p.index not in exclude:
            c = _outline_dash_centerline(p)
            if c is not None:
                outline.append(c)
    trasse += _collinear_neighbors(outline, min_run=3)
    cand = [p for p in pv.primitives if p.color == RED and not p.filled and p.width > 0
            and (style.trasse_width is None or abs(p.width - style.trasse_width) >= 0.03)
            and p.index not in exclude and _dash_like(p, 1.5, 12.0) and not in_label(p)]
    kanal = _collinear_neighbors(cand, min_run=3)
    return trasse, kanal


def _chain_leaders(prims: list[Primitive], tol: float = 0.6) -> list[np.ndarray]:
    """Verbindet Hinweislinien-Stücke mit gemeinsamen Endpunkten zu Linienzügen."""
    pls = [pl for p in prims for pl in p.polylines if len(pl) >= 2]
    used = [False] * len(pls)
    chains = []
    for i in range(len(pls)):
        if used[i]:
            continue
        used[i] = True
        chain = pls[i].copy()
        grown = True
        while grown:
            grown = False
            for j in range(len(pls)):
                if used[j]:
                    continue
                q = pls[j]
                if np.linalg.norm(chain[-1] - q[0]) < tol:
                    chain = np.vstack([chain, q[1:]])
                elif np.linalg.norm(chain[-1] - q[-1]) < tol:
                    chain = np.vstack([chain, q[::-1][1:]])
                elif np.linalg.norm(chain[0] - q[-1]) < tol:
                    chain = np.vstack([q[:-1], chain])
                elif np.linalg.norm(chain[0] - q[0]) < tol:
                    chain = np.vstack([q[::-1][:-1], chain])
                else:
                    continue
                used[j] = True
                grown = True
        chains.append(chain)
    return chains


def _dist_point_bbox(p, bbox) -> float:
    x0, y0, x1, y1 = bbox
    dx = max(x0 - p[0], 0, p[0] - x1)
    dy = max(y0 - p[1], 0, p[1] - y1)
    return float(np.hypot(dx, dy))


def _dist_polyline_bbox(pl: np.ndarray, bbox, samples: int = 8) -> float:
    best = np.inf
    for i in range(len(pl) - 1):
        for t in np.linspace(0, 1, samples):
            best = min(best, _dist_point_bbox(pl[i] + (pl[i + 1] - pl[i]) * t, bbox))
    return best


def extract_cable_labels(pv: PageVectors, style: KlpStyle, log=print) -> list[CableLabel]:
    lw = style.label_width
    if lw is None:
        return []
    prims = [p for p in pv.primitives if p.color == RED and abs(p.width - lw) < 0.03]
    glyphs = [p for p in prims if p.size < style.glyph_max_size]
    leader_prims = [p for p in prims if p.size >= style.glyph_max_size and not p.filled]
    leaders = _chain_leaders(leader_prims)

    # Zeilen lesen, dann Zeilen zu Stapeln (Spalten mit ~3,5 pt Zeilenabstand) gruppieren
    lines = find_text_lines(glyphs, char_gap=1.2)
    lines = [l for l in lines if len(l.prims) >= 4]
    ocr_lines(lines, whitelist="S0123456789")
    lines = [l for l in lines if normalize_cable_id(l.text, allow_missing_prefix=True)]
    for l in lines:
        l.text = normalize_cable_id(l.text, allow_missing_prefix=True)
    stacks = cluster_boxes([l.bbox for l in lines], expand_x=1.5, expand_y=2.2)
    labels: list[CableLabel] = []
    for st in stacks:
        members = sorted((lines[i] for i in st), key=lambda l: (l.bbox[1], l.bbox[0]))
        cables = list(dict.fromkeys(l.text for l in members))
        bbox = (min(l.bbox[0] for l in members), min(l.bbox[1] for l in members),
                max(l.bbox[2] for l in members), max(l.bbox[3] for l in members))
        labels.append(CableLabel(cables=cables, bbox=bbox, anchor=None, lines=members))

    # Hinweislinie je Stapel: die nächstgelegene Linie; ihr vom Text entferntes Ende ist der Anker.
    for lab in labels:
        best = None
        for ch in leaders:
            d = _dist_polyline_bbox(ch, lab.bbox)
            if d < 4.0 and (best is None or d < best[0]):
                best = (d, ch)
        if best is None:
            continue
        ch = best[1]
        d0 = _dist_point_bbox(ch[0], lab.bbox)
        d1 = _dist_point_bbox(ch[-1], lab.bbox)
        lab.anchor = ch[0] if d0 > d1 else ch[-1]
        lab.leader = [ch]
    n_anchor = sum(1 for l in labels if l.anchor is not None)
    log(f"KLP: {len(labels)} Kabelnummern-Stapel, {sum(len(l.cables) for l in labels)} Einträge, "
        f"{n_anchor} mit Hinweislinie")
    return labels


# ---------------------------------------------------------------------------
# Elementbeschriftungen (Weichen, Signale, Gleisfreimelder, Kabelschränke ...)
# ---------------------------------------------------------------------------

@dataclass
class PlanElement:
    text: str
    tokens: frozenset[str]
    center: np.ndarray
    bbox: tuple[float, float, float, float]
    angle: float = 0.0


_PREFIX_RE = re.compile(r"^(KS|SK|KV|WW|W|G|P|L|Z)?(.*)$")


def normalize_element_token(tok: str) -> str:
    """OCR-Varianten einer Elementkennung vereinheitlichen ("Ww25D" -> "W25D", "63045" -> "G3045")."""
    t = re.sub(r"[^A-Z0-9,]", "", tok.upper())
    if not t:
        return ""
    if re.fullmatch(r"6\d{4}", t):          # G als 6 gelesen
        t = "G" + t[1:]
    m = _PREFIX_RE.match(t)
    prefix, rest = m.group(1) or "", m.group(2)
    if prefix == "WW":
        prefix = "W"
    # Im Ziffernteil: S->5, O->0, I->1 (ein Kennbuchstabe am Ende bleibt erhalten)
    tail = ""
    if len(rest) >= 2 and rest[-1] in "ABCDXY" and rest[:-1][-1:].isdigit():
        rest, tail = rest[:-1], rest[-1]
    rest = rest.replace("S", "5").replace("O", "0").replace("I", "1")
    return prefix + rest + tail


def element_tokens(text: str) -> frozenset[str]:
    parts = re.split(r"[/\s]+", text.strip())
    return frozenset(t for t in (normalize_element_token(p) for p in parts) if len(t) >= 2)


def kuep_name_tokens(name: str) -> list[frozenset[str]]:
    """Mögliche KLP-Schreibweisen eines KÜP-Namens, beste zuerst.

    "13W22/13G4004" -> {W22, G4004}; "KV 13W22" -> {W22}; "13L3045Y" -> {3045Y};
    "KS 1307000" -> {KS1307000}; "SK 13P8" -> {P8}; "SK 1307110" -> {SK1307110}.
    """
    n = name.strip().upper()
    if not n:
        return []
    variants: list[frozenset[str]] = []
    m = re.fullmatch(r"(KS|SK|KV)\s*(\d{5,})", n)
    if m:
        variants.append(frozenset({m.group(1) + m.group(2)}))
        return variants
    m = re.fullmatch(r"(KS|SK|KV)\s*(.+)", n)
    if m:
        n = m.group(2)
    toks = []
    for part in n.split("/"):
        part = part.strip()
        part = re.sub(r"^13(?=[A-Z])", "", part)       # Stellrechnerbereich weglassen
        if part.startswith("L") and len(part) > 2:     # Gleisfreimeldeabschnitt: "L3045Y" -> "3045Y"
            part = part[1:]
        t = normalize_element_token(part)
        if t:
            toks.append(t)
    if toks:
        variants.append(frozenset(toks))
    return variants


def extract_elements(pv: PageVectors, style: KlpStyle, exclude: set[int] | None = None,
                     log=print) -> list[PlanElement]:
    """Liest alle roten Beschriftungen außer den Kabelnummern-Stapeln.
    ``exclude``: Primitiv-Indizes, die sicher keine Schrift sind (z. B. Kanal-Striche)."""
    exclude = exclude or set()
    groups: dict[float, list[Primitive]] = {}
    for p in pv.primitives:
        if p.color != RED or p.filled or p.size >= 14 or p.width <= 0 or p.index in exclude:
            continue
        groups.setdefault(round(p.width, 2), []).append(p)
    lines: list[TextLine] = []
    for w, prims in groups.items():
        # Kleine, eng gesetzte Schrift (W22/G4004) und große Schrift (KS 1307000) können dieselbe
        # Strichstärke haben – beide Zeichenabstände versuchen, die Zuordnung sortiert später aus.
        passes = [(0.9, 9.0)] if w < 0.45 else [(0.9, 9.0), (2.2, 14.0)]
        for gap, max_size in passes:
            sel = [p for p in prims if p.size < max_size]
            lines += [l for l in find_text_lines(sel, char_gap=gap, min_glyphs=2)]
    ocr_lines(lines)
    lines = [l for l in lines if l.text and l.conf >= 30]

    elements: list[PlanElement] = []
    used: set[int] = set()
    # "KS" + "1307000" stehen oft als getrennte Zeilen hintereinander
    for i, l in enumerate(lines):
        t = l.text.strip().upper()
        if t in ("KS", "SK", "KV"):
            best = None
            for j, m in enumerate(lines):
                if j == i or not re.fullmatch(r"\d{6,8}", m.text.strip()):
                    continue
                d = float(np.linalg.norm(m.center - l.center))
                if d < 6 * max(l.height, 1.0) and (best is None or d < best[0]):
                    best = (d, j)
            if best is not None:
                m = lines[best[1]]
                used |= {i, best[1]}
                text = t + m.text.strip()
                bbox = (min(l.bbox[0], m.bbox[0]), min(l.bbox[1], m.bbox[1]),
                        max(l.bbox[2], m.bbox[2]), max(l.bbox[3], m.bbox[3]))
                elements.append(PlanElement(text, frozenset({text}), (l.center + m.center) / 2, bbox, l.angle))
    for i, l in enumerate(lines):
        if i in used:
            continue
        text = l.text.strip()
        compact = text.replace(" ", "").upper()
        m = re.fullmatch(r"(KS|SK|KV)(\d{6,8})", compact)
        if m:
            elements.append(PlanElement(compact, frozenset({compact}), l.center, l.bbox, l.angle))
            continue
        toks = element_tokens(text)
        # Nur plausible Kennungen: Buchstabe+Ziffer oder Ziffern+Kennbuchstabe (3045Y)
        toks = frozenset(t for t in toks if re.fullmatch(r"[A-Z]{1,2}\d{1,5}[A-D]?|\d{3,5}[XY]", t))
        if toks:
            elements.append(PlanElement(text, toks, l.center, l.bbox, l.angle))
    log(f"KLP: {len(elements)} Elementbeschriftungen erkannt")
    return elements


def find_element(elements: list[PlanElement], name: str) -> PlanElement | None:
    """Sucht die KLP-Beschriftung zu einem KÜP-Namen."""
    best = None
    for want in kuep_name_tokens(name):
        for e in elements:
            common = len(want & e.tokens)
            if common == 0:
                continue
            score = common / len(want) - 0.05 * len(e.tokens - want)
            if common == len(want) or common >= 2:
                if best is None or score > best[0]:
                    best = (score, e)
        if best is not None:
            return best[1]
    return None
