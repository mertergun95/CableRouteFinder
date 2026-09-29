"""Texterkennung für als Linien gezeichnete CAD-Schrift.

Ablauf:
1. Schriftzeichen-Primitive werden zu Textzeilen gruppiert (``find_text_lines``),
   die Leserichtung jeder Zeile wird aus der Hauptachse der Zeichenpunkte geschätzt.
2. Jede Zeile wird waagerecht gedreht, isoliert (ohne Plangrafik) gerastert und
   mit vielen anderen Zeilen untereinander auf ein Sammelbild gesetzt.
3. Ein Tesseract-Aufruf je Sammelbild; die Wörter werden den Zeilen zugeordnet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract

from .pdfvector import Primitive


def _configure_tesseract() -> None:
    """Tesseract auch finden, wenn es unter Windows nicht im PATH steht (Standard beim Installer)."""
    import os
    import shutil
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.exists(env):
        pytesseract.pytesseract.tesseract_cmd = env
        return
    if shutil.which("tesseract"):
        return
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(local, "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(local, "Tesseract-OCR", "tesseract.exe"),
        r"C:\Tesseract-OCR\tesseract.exe",
        r"C:\ProgramData\chocolatey\bin\tesseract.exe",
        os.path.join(os.path.expanduser("~"), "scoop", "shims", "tesseract.exe"),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            return


_configure_tesseract()


@dataclass
class TextLine:
    prims: list[Primitive]
    bbox: tuple[float, float, float, float]
    angle: float = 0.0          # Leserichtung in Grad, mathematisch (y nach unten), 0 = waagerecht
    text: str = ""
    conf: float = -1.0
    words: list[str] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    @property
    def height(self) -> float:
        """Schrifthöhe senkrecht zur Leserichtung."""
        pts = np.vstack([pl for p in self.prims for pl in p.polylines])
        a = math.radians(self.angle)
        n = np.array([-math.sin(a), math.cos(a)])
        return float(np.ptp(pts @ n))


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_boxes(boxes: list[tuple[float, float, float, float]], expand_x: float, expand_y: float) -> list[list[int]]:
    """Gruppiert Rahmen, deren um ``expand`` vergrößerte Flächen sich überschneiden."""
    n = len(boxes)
    if n == 0:
        return []
    uf = _UnionFind(n)
    cell = max(expand_x, expand_y) * 4 + 2.0
    grid: dict[tuple[int, int], list[int]] = {}
    ex = [(b[0] - expand_x, b[1] - expand_y, b[2] + expand_x, b[3] + expand_y) for b in boxes]
    for i, (x0, y0, x1, y1) in enumerate(ex):
        for gx in range(int(x0 // cell), int(x1 // cell) + 1):
            for gy in range(int(y0 // cell), int(y1 // cell) + 1):
                grid.setdefault((gx, gy), []).append(i)
    for members in grid.values():
        for a_i in range(len(members)):
            a = members[a_i]
            ax0, ay0, ax1, ay1 = ex[a]
            for b in members[a_i + 1:]:
                bx0, by0, bx1, by1 = ex[b]
                if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                    uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def union_bbox(prims: list[Primitive]) -> tuple[float, float, float, float]:
    return (min(p.bbox[0] for p in prims), min(p.bbox[1] for p in prims),
            max(p.bbox[2] for p in prims), max(p.bbox[3] for p in prims))


def estimate_angle(prims: list[Primitive]) -> float:
    """Leserichtung aus der Hauptachse der Zeichen; Ergebnis in (-90, 90]."""
    centers = np.array([p.center for p in prims])
    pts = centers if len(prims) >= 3 else np.vstack([pl for p in prims for pl in p.polylines])
    if len(pts) < 2:
        return 0.0
    c = pts - pts.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    if len(s) < 2 or s[0] < 1e-6 or (len(s) > 1 and s[1] / s[0] > 0.6):
        return 0.0
    dx, dy = vt[0]
    ang = math.degrees(math.atan2(dy, dx))
    # Lesbare CAD-Schrift läuft von links nach rechts bzw. von unten nach oben
    if ang <= -90:
        ang += 180
    elif ang > 90:
        ang -= 180
    if abs(ang - 90) < 1e-6:
        ang = -90.0
    # Nahezu achsparallel -> exakt
    for ref in (0.0, -90.0, 90.0):
        if abs(ang - ref) < 1.5:
            ang = ref
    return -90.0 if ang == 90.0 else ang


def find_text_lines(glyphs: list[Primitive], char_gap: float = 1.2, min_glyphs: int = 1,
                    horizontal_only: bool = False, merge_gap_factor: float = 0.55) -> list[TextLine]:
    """Gruppiert Schriftzeichen zu Zeilen (beliebige Richtung).

    ``horizontal_only``: alle Zeilen gelten als waagerecht (z. B. KÜP).
    Waagerechte Zeilenstücke auf gleicher Grundlinie werden anschließend verbunden,
    wenn ihr Abstand kleiner als ``merge_gap_factor`` × Schrifthöhe ist (größere Schriften).
    """
    clusters = cluster_boxes([g.bbox for g in glyphs], char_gap, char_gap)
    lines = []
    for cl in clusters:
        if len(cl) < min_glyphs:
            continue
        prims = [glyphs[i] for i in cl]
        bbox = union_bbox(prims)
        angle = 0.0 if horizontal_only else estimate_angle(prims)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if angle != 0.0 and abs(angle) < 15 and w >= 2 * h:
            angle = 0.0
        lines.append(TextLine(prims=prims, bbox=bbox, angle=angle))
    return _merge_horizontal(lines, merge_gap_factor)


def _merge_horizontal(lines: list[TextLine], factor: float) -> list[TextLine]:
    horiz = sorted([l for l in lines if l.angle == 0.0], key=lambda l: l.bbox[0])
    rest = [l for l in lines if l.angle != 0.0]
    merged: list[TextLine] = []
    for ln in horiz:
        target = None
        for m in merged:
            mx0, my0, mx1, my1 = m.bbox
            lx0, ly0, lx1, ly1 = ln.bbox
            hmax = max(my1 - my0, ly1 - ly0)
            overlap = min(my1, ly1) - max(my0, ly0)
            if overlap < 0.6 * min(my1 - my0, ly1 - ly0):
                continue
            if abs((my1 - my0) - (ly1 - ly0)) > 0.2 * hmax or len(m.prims) < 2 or len(ln.prims) < 2:
                continue
            gap = lx0 - mx1
            if -0.5 <= gap <= factor * hmax:
                target = m
                break
        if target is None:
            merged.append(ln)
        else:
            target.prims = target.prims + ln.prims
            target.bbox = union_bbox(target.prims)
    return merged + rest


def _render_line(line: TextLine, scale: float, margin: float = 1.5) -> np.ndarray:
    a = math.radians(-line.angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = line.center
    polys = []
    for p in line.prims:
        for pl in p.polylines:
            polys.append((p, (pl - c) @ rot.T))
    allp = np.vstack([q for _, q in polys])
    mn = allp.min(axis=0) - margin
    mx = allp.max(axis=0) + margin
    w = int(math.ceil((mx[0] - mn[0]) * scale))
    h = int(math.ceil((mx[1] - mn[1]) * scale))
    img = np.full((max(h, 1), max(w, 1)), 255, np.uint8)
    for p, q in polys:
        pts = np.round((q - mn) * scale * 8).astype(np.int32)
        if p.filled and len(q) >= 3:
            cv2.fillPoly(img, [pts], 0, lineType=cv2.LINE_AA, shift=3)
        else:
            th = max(1, int(round(max(p.width, 0.35) * scale)))
            cv2.polylines(img, [pts], False, 0, th, lineType=cv2.LINE_AA, shift=3)
    return img


def ocr_lines(lines: list[TextLine], whitelist: str | None = None, target_height_px: int = 44,
              per_sheet: int = 40, gap_px: int = 40, retry_below_conf: float = 60.0) -> None:
    """Liest alle Zeilen (Ergebnis in ``line.text``/``line.conf``).

    Erst gesammelt (wenige Tesseract-Aufrufe), dann werden unsichere Zeilen einzeln
    als Textzeile (psm 7) nachgelesen und das bessere Ergebnis behalten."""
    _ocr_lines_batched(lines, whitelist, target_height_px, per_sheet, gap_px)
    cfg = "--psm 7 --oem 1"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    for ln in lines:
        if ln.conf >= retry_below_conf or len(ln.prims) < 2:
            continue
        img = _render_line(ln, min(max(target_height_px / max(ln.height, 0.5), 3.0), 30.0))
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
        words = [((t or "").strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if (t or "").strip()]
        if not words:
            continue
        conf = float(np.mean([c for _, c in words]))
        if conf > ln.conf:
            ln.words = [w for w, _ in words]
            ln.text = " ".join(ln.words)
            ln.conf = conf


def _ocr_lines_batched(lines: list[TextLine], whitelist: str | None, target_height_px: int,
                       per_sheet: int, gap_px: int) -> None:
    cfg = "--psm 6 --oem 1 -c preserve_interword_spaces=1"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    todo = [l for l in lines if l.prims]
    for start in range(0, len(todo), per_sheet):
        batch = todo[start:start + per_sheet]
        imgs = []
        for ln in batch:
            h = max(ln.height, 0.5)
            scale = min(max(target_height_px / h, 3.0), 30.0)
            img = _render_line(ln, scale)
            if img.shape[1] > 6000:
                img = cv2.resize(img, (6000, max(1, int(img.shape[0] * 6000 / img.shape[1]))))
            imgs.append(img)
        width = max(i.shape[1] for i in imgs) + 2 * gap_px
        rows = []
        y = gap_px
        for img in imgs:
            rows.append((y, y + img.shape[0]))
            y += img.shape[0] + gap_px
        sheet = np.full((y, width), 255, np.uint8)
        for img, (y0, y1) in zip(imgs, rows):
            sheet[y0:y1, gap_px:gap_px + img.shape[1]] = img
        data = pytesseract.image_to_data(sheet, config=cfg, output_type=pytesseract.Output.DICT)
        collected: list[list[tuple[int, str, float]]] = [[] for _ in batch]
        for i, txt in enumerate(data["text"]):
            txt = (txt or "").strip()
            if not txt:
                continue
            cy = data["top"][i] + data["height"][i] / 2
            for k, (y0, y1) in enumerate(rows):
                if y0 - gap_px / 2 <= cy <= y1 + gap_px / 2:
                    collected[k].append((data["left"][i], txt, float(data["conf"][i])))
                    break
        for ln, words in zip(batch, collected):
            words.sort()
            ln.words = [w for _, w, _ in words]
            ln.text = " ".join(ln.words)
            ln.conf = float(np.mean([c for _, _, c in words])) if words else -1.0
