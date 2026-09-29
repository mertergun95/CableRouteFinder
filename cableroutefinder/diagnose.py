"""Diagnose: Was auf einer KLP-Seite erkannt wird – für die Fehlersuche bei neuen Plänen.

Aufruf: ``python -m cableroutefinder.diagnose KLP.pdf [-o ausgabe]``
Schreibt eine Textzusammenfassung (``*_diagnose.txt``) und ein Bild der erkannten
Kabelnummern-Stapel/Trassen (``*_diagnose.png``).
"""
from __future__ import annotations

import argparse
import collections
import os
import re

import networkx as nx
import pymupdf

from .klp import detect_style, extract_cable_labels, trasse_primitives
from .pdfvector import RED, load_page_vectors
from .trassegraph import build_trasse_graph


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cableroutefinder.diagnose")
    ap.add_argument("klp")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("-o", "--out", default="diagnose")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.klp))[0]
    lines: list[str] = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    doc = pymupdf.open(args.klp)
    page = doc[args.page]
    log(f"Datei: {args.klp}  Seiten: {doc.page_count}  Seite: {args.page}")
    log(f"Seitengröße: {page.rect.width:.0f} x {page.rect.height:.0f} pt  Rotation: {page.rotation}")
    words = page.get_text("words")
    cable_words = [w[4] for w in words if re.fullmatch(r"S?\d{7}", w[4])]
    log(f"Echter PDF-Text: {len(words)} Wörter, davon Kabelnummern: {len(cable_words)} {cable_words[:10]}")

    pv = load_page_vectors(args.klp, args.page)
    colors = collections.Counter(p.color for p in pv.primitives)
    log(f"Zeichenobjekte: {len(pv.primitives)}  Farben: {dict(colors)}")
    stats = collections.defaultdict(lambda: [0, 0, 0])
    for p in pv.primitives:
        key = (p.color, round(p.width, 2), p.filled)
        stats[key][0] += 1
        if p.size < 6.5:
            stats[key][1] += 1
        if p.size > 20:
            stats[key][2] += 1
    log("Stil (Farbe, Strichstärke, gefüllt): Anzahl / klein<6.5pt / groß>20pt")
    for key, (n, small, big) in sorted(stats.items(), key=lambda kv: -kv[1][0])[:25]:
        log(f"  {key}: {n} / {small} / {big}")

    style = detect_style(pv)
    log(f"Erkannte Ebenen: {style}")
    labels = extract_cable_labels(pv, style, log=log)
    label_prims = {p.index for lab in labels for ln in lab.lines for p in ln.prims}
    trasse, kanal = trasse_primitives(pv, style, labels=labels, exclude=label_prims)
    G = build_trasse_graph(trasse, kanal, pv.width, pv.height)
    log(f"Trasse: {len(trasse)} Striche, Kanal: {len(kanal)}; Graph {G.number_of_nodes()} Knoten / "
        f"{G.number_of_edges()} Kanten / {nx.number_connected_components(G)} Netzteile")
    for lab in labels[:40]:
        log(f"  Stapel bei {[round(v) for v in lab.bbox]}: {lab.cables}  Anker: {None if lab.anchor is None else lab.anchor.round(0).tolist()}")

    # Bild: erkannte Trassen blau, Stapel grün, Anker rot
    sh = page.new_shape()
    derot = page.derotation_matrix
    for u, v, d in G.edges(data=True):
        sh.draw_polyline([pymupdf.Point(*p) * derot for p in d["pts"]])
    sh.finish(color=(0, 0.4, 1), width=1.2, closePath=False)
    for lab in labels:
        sh.draw_rect(pymupdf.Rect(lab.bbox) * derot)
        sh.finish(color=(0, 0.7, 0), width=1.0)
        if lab.anchor is not None:
            sh.draw_circle(pymupdf.Point(*lab.anchor) * derot, 2.5)
            sh.finish(color=(1, 0, 0), fill=(1, 0, 0))
    sh.commit()
    png = os.path.join(args.out, base + "_diagnose.png")
    page.get_pixmap(dpi=60).save(png)
    txt = os.path.join(args.out, base + "_diagnose.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nGeschrieben: {txt}\n             {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
