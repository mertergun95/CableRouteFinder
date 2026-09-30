"""Ausgaben: markierte PDF, CablePlan-JSON, Bericht (JSON/CSV)."""
from __future__ import annotations

import colorsys
import csv
import json
import os

import numpy as np
import pymupdf

from . import __version__
from .pipeline import AnalysisResult

# CablePlan rendert die Grundseite mit 160 dpi (MainWindow.LoadPdfFirstPageToBitmap)
CABLEPLAN_DPI = 160


def _color(i: int, n: int) -> tuple[float, float, float]:
    h = (i * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(h, 0.85, 0.85)


def write_annotated_pdf(res: AnalysisResult, out_path: str, show_network: bool = True) -> None:
    """Zeichnet alle gefundenen Kabelwege in die KLP-PDF. Jedes Kabel ist eine eigene
    PDF-Ebene (Optional Content), damit es im PDF-Viewer einzeln ein-/ausgeblendet werden kann."""
    doc = pymupdf.open(res.klp_path)
    page = doc[res.page_index]
    # Zeichnen in Anzeigekoordinaten auch bei gedrehten Seiten
    derot = page.derotation_matrix

    def P(pt):
        return pymupdf.Point(float(pt[0]), float(pt[1])) * derot

    if show_network:
        oc = doc.add_ocg("Trassennetz (erkannt)", on=False)
        sh = page.new_shape()
        for u, v, d in res.graph.edges(data=True):
            sh.draw_polyline([P(p) for p in d["pts"]])
        sh.finish(color=(0.1, 0.6, 1.0), width=0.8, closePath=False, stroke_opacity=0.6, oc=oc)
        for lab in res.labels:
            if lab.anchor is not None:
                sh.draw_circle(P(lab.anchor), 1.6)
        sh.finish(color=(0.1, 0.4, 1.0), fill=(0.1, 0.6, 1.0), width=0.3, oc=oc)
        sh.commit()

    routed = [c for c in res.cables if c.route.polylines]
    for i, c in enumerate(routed):
        col = _color(i, len(routed))
        oc = doc.add_ocg(f"Kabel {c.cable_id}", on=True)
        sh = page.new_shape()
        for pl in c.route.polylines:
            sh.draw_polyline([P(p) for p in pl])
        dashes = None if c.confidence == "hoch" else "[4 2] 0"
        sh.finish(color=col, width=2.2, closePath=False, stroke_opacity=0.75, dashes=dashes, oc=oc,
                  lineCap=1, lineJoin=1)
        main = c.route.polylines[0]
        sh.draw_circle(P(main[0]), 2.2)
        sh.finish(color=col, fill=col, oc=oc)
        sh.draw_circle(P(main[-1]), 2.2)
        sh.finish(color=col, fill=(1, 1, 1), width=1.0, oc=oc)
        sh.commit()
        # Beschriftung am Start
        label = c.cable_id
        page.insert_text(P(main[0] + np.array([3.0, -3.0])), label, fontsize=5, color=col, oc=oc,
                         rotate=page.rotation)
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()


def cableplan_json(res: AnalysisResult, existing: dict | None = None, overwrite: bool = False) -> dict:
    """Erzeugt/ergänzt die CablePlan-Plandatei (``data/<Planname>.json``).

    Koordinaten sind Pixel des unrotierten 160-dpi-Grundbilds, das CablePlan aus der PDF rendert.
    Bereits vorhandene (z. B. von Hand gezeichnete) Kabel bleiben erhalten, außer ``overwrite``."""
    w_px = int(round(res.page_width / 72.0 * CABLEPLAN_DPI))
    h_px = int(round(res.page_height / 72.0 * CABLEPLAN_DPI))
    k = w_px / res.page_width
    data = dict(existing or {})
    data.setdefault("BaseWidth", w_px)
    data.setdefault("BaseHeight", h_px)
    data.setdefault("RotationDeg", 0)
    data.setdefault("PdfPageWidthPoints", res.page_width)
    data.setdefault("PdfPageHeightPoints", res.page_height)
    data.setdefault("Transform", {"ScaleX": 1.0, "ScaleY": 1.0, "OffsetX": 0.0, "OffsetY": 0.0,
                                  "Matrix3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]})
    for key in ("Sperrpauses", "Muffenpunkte", "BraunstrichKorrekturen", "PdfTexts"):
        data.setdefault(key, [])
    data.setdefault("StampedPdfHash", "")
    cables = list(data.get("Cables") or [])
    by_id = {c.get("Id", "").upper(): i for i, c in enumerate(cables)}
    for c in res.cables:
        if not c.route.polylines:
            continue
        pts = c.route.polylines[0]
        entry = {
            "Id": c.cable_id,
            "Points": [{"X": round(float(x) * k, 2), "Y": round(float(y) * k, 2)} for x, y in pts],
            "LabelPoint": {"X": round(float(pts[-1][0]) * k, 2), "Y": round(float(pts[-1][1]) * k, 2)},
        }
        idx = by_id.get(c.cable_id.upper())
        if idx is None:
            by_id[c.cable_id.upper()] = len(cables)
            cables.append(entry)
        elif overwrite:
            cables[idx] = entry
    data["Cables"] = cables
    return data


def write_cableplan_json(res: AnalysisResult, out_path: str, merge_with: str | None = None,
                         overwrite: bool = False) -> None:
    existing = None
    src = merge_with
    if src and os.path.exists(src):
        with open(src, encoding="utf-8-sig") as f:
            existing = json.load(f)
    data = cableplan_json(res, existing, overwrite)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_report(res: AnalysisResult, json_path: str, csv_path: str) -> None:
    rows = [c.to_dict() for c in res.cables]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "klp": os.path.basename(res.klp_path),
            "klp_path": os.path.abspath(res.klp_path),
            "tool_version": __version__,
            "tool_path": os.path.dirname(os.path.abspath(__file__)),
            "scale": res.scale,
            "labels": [{"cables": l.cables, "anchor": None if l.anchor is None else l.anchor.round(2).tolist(),
                        "bbox": [round(v, 2) for v in l.bbox]} for l in res.labels],
            "cables": rows,
        }, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Kabel", "Von", "Nach", "Länge KÜP [m]", "Länge Weg KLP [m]", "Beschriftungen im KLP",
                    "Start gefunden", "Ziel gefunden", "Sicherheit", "Hinweise"])
        for r in rows:
            w.writerow([r["id"], r["from"], r["to"], r["kuep_length_m"] or "",
                        str(r["route_length_m"] or "").replace(".", ","), r["labels_on_klp"],
                        "ja" if r["start_found"] else "nein", "ja" if r["end_found"] else "nein",
                        r["confidence"], " | ".join(r["warnings"])])
