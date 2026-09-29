"""Kommandozeile: ``python -m cableroutefinder KLP.pdf --kuep KUEP.pdf -o ausgabe/``"""
from __future__ import annotations

import argparse
import os
import sys

from .export import write_annotated_pdf, write_cableplan_json, write_report
from .pipeline import analyze


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cableroutefinder",
        description="Zeichnet die Kabelwege aus dem Kabelübersichtsplan (KÜP) automatisch in den Kabellageplan (KLP).")
    ap.add_argument("klp", help="Kabellageplan (PDF, Vektor)")
    ap.add_argument("--kuep", help="Kabelübersichtsplan (PDF, Vektor) – liefert Kabelliste und Start/Ziel")
    ap.add_argument("-o", "--out", default="ausgabe", help="Ausgabeordner (Standard: ./ausgabe)")
    ap.add_argument("--page", type=int, default=0, help="Seite im KLP (0 = erste)")
    ap.add_argument("--scale", type=float, default=500.0, help="Maßstab des KLP, z. B. 500 für 1:500")
    ap.add_argument("--cables", help="nur diese Kabel (Komma-getrennt), z. B. S1307010,S1307505")
    ap.add_argument("--cableplan-json", help="vorhandene CablePlan-Plandatei (data/<Plan>.json) ergänzen")
    ap.add_argument("--overwrite", action="store_true",
                    help="in der CablePlan-Datei vorhandene Kabel gleichen Namens überschreiben")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.klp))[0]
    only = [c.strip() for c in args.cables.split(",")] if args.cables else None
    res = analyze(args.klp, args.kuep, page_index=args.page, scale=args.scale, only_cables=only)

    pdf_out = os.path.join(args.out, f"{base}_kabelwege.pdf")
    write_annotated_pdf(res, pdf_out)
    json_out = os.path.join(args.out, f"{base}.json")
    write_cableplan_json(res, json_out, merge_with=args.cableplan_json, overwrite=args.overwrite)
    write_report(res, os.path.join(args.out, f"{base}_bericht.json"), os.path.join(args.out, f"{base}_bericht.csv"))

    print()
    print(f"{'Kabel':<10} {'Von':<16} {'Nach':<18} {'KÜP m':>6} {'Weg m':>7}  Sicherheit  Hinweise")
    for c in res.cables:
        d = c.to_dict()
        print(f"{d['id']:<10} {d['from'][:16]:<16} {d['to'][:18]:<18} {str(d['kuep_length_m'] or ''):>6} "
              f"{str(d['route_length_m'] or ''):>7}  {d['confidence']:<10}  {'; '.join(d['warnings'])}")
    print()
    print(f"Markierte PDF:   {pdf_out}")
    print(f"CablePlan-JSON:  {json_out}")
    print(f"Bericht:         {os.path.join(args.out, base + '_bericht.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
