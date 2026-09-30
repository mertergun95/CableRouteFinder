"""Gesamtablauf: KLP (+ optional KÜP) lesen und für jedes Kabel den Weg bestimmen."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .klp import (CableLabel, KlpStyle, PlanElement, detect_style, extract_cable_labels, extract_elements,
                  find_element, trasse_primitives)
from .kuep import KuepCable, KuepResult, parse_kuep
from .pdfvector import Frame, PageVectors, load_page_vectors, rotate_vectors
from .textocr import detect_reading_rotation
from .router import CableRoute, choose_candidate, route_terminals, snap_candidates, snap_points
from .trassegraph import GraphParams, build_trasse_graph

PT_TO_MM = 25.4 / 72.0


@dataclass
class Endpoint:
    name: str
    element: PlanElement | None = None
    candidates: list[tuple[int, float]] = field(default_factory=list)
    node: int | None = None

    @property
    def found(self) -> bool:
        return self.node is not None


@dataclass
class CableResult:
    cable_id: str
    kuep: KuepCable | None
    route: CableRoute
    label_count: int = 0
    start: Endpoint | None = None
    end: Endpoint | None = None
    confidence: str = "keine"
    length_m: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.cable_id,
            "from": self.kuep.from_name if self.kuep else "",
            "to": self.kuep.to_name if self.kuep else "",
            "kuep_length_m": self.kuep.length_m if self.kuep else None,
            "route_length_m": round(self.length_m, 1) if self.length_m is not None else None,
            "labels_on_klp": self.label_count,
            "start_found": bool(self.start and self.start.found),
            "end_found": bool(self.end and self.end.found),
            "confidence": self.confidence,
            "warnings": self.route.warnings,
            "polylines_pt": [pl.round(2).tolist() for pl in self.route.polylines],
        }


@dataclass
class AnalysisResult:
    klp_path: str
    page_index: int
    page_width: float
    page_height: float
    scale: float
    graph: nx.Graph
    labels: list[CableLabel]
    elements: list[PlanElement]
    kuep: KuepResult | None
    cables: list[CableResult] = field(default_factory=list)


def _log_default(msg: str) -> None:
    print(msg, flush=True)


def analyze(klp_path: str, kuep_path: str | None = None, page_index: int = 0, scale: float = 500.0,
            only_cables: list[str] | None = None, style: KlpStyle | None = None,
            graph_params: GraphParams | None = None, endpoint_snap_pt: float = 40.0,
            log=_log_default) -> AnalysisResult:
    log(f"Lese Kabellageplan: {klp_path}")
    pv_display: PageVectors = load_page_vectors(klp_path, page_index)
    # Gedreht gespeicherte Pläne in Leserichtung auswerten, Ergebnisse am Ende zurückdrehen
    pv, frame = rotate_vectors(pv_display, detect_reading_rotation(pv_display, log=log))
    style = style or detect_style(pv)
    log(f"KLP-Ebenen: Trasse={style.trasse_width} Kanal={style.kanal_width} Beschriftung={style.label_width}")

    # Erst die Kabelnummern lesen: ihre Schrift darf weder als Kanal-Strich noch als Elementname gelten.
    labels = extract_cable_labels(pv, style, log=log)
    label_prims = {p.index for lab in labels for ln in lab.lines for p in ln.prims}
    trasse, kanal = trasse_primitives(pv, style, labels=labels, exclude=label_prims)
    G = build_trasse_graph(trasse, kanal, pv.width, pv.height, graph_params)
    log(f"Trassennetz: {G.number_of_nodes()} Knoten, {G.number_of_edges()} Kanten, "
        f"{nx.number_connected_components(G)} Netzteile")
    elements = extract_elements(pv, style, exclude=label_prims | {p.index for p in trasse + kanal}, log=log)

    kuep = None
    if kuep_path:
        log(f"Lese Kabelübersichtsplan: {kuep_path}")
        kuep_pv = load_page_vectors(kuep_path, 0)
        kuep_pv, _ = rotate_vectors(kuep_pv, detect_reading_rotation(kuep_pv, colors=("black",), log=log))
        kuep = parse_kuep(kuep_pv, log=log)

    # Kabelliste: aus dem KÜP, sonst alle im KLP beschrifteten Kabel
    if kuep is not None:
        cable_ids = [c.cable_id for c in kuep.cables]
    else:
        cable_ids = sorted({cid for lab in labels for cid in lab.cables})
    if only_cables:
        wanted = {c.upper() for c in only_cables}
        cable_ids = [c for c in cable_ids if c in wanted] + sorted(wanted - set(cable_ids))
    kuep_by_id = {c.cable_id: c for c in kuep.cables} if kuep else {}

    # Anker aller Beschriftungen einmalig in den Graphen einfügen
    anchored = [lab for lab in labels if lab.anchor is not None]
    nodes = snap_points(G, [lab.anchor for lab in anchored], max_dist=10.0)
    label_nodes = {id(lab): n for lab, n in zip(anchored, nodes)}

    endpoint_cache: dict[str, Endpoint] = {}

    def endpoint(name: str) -> Endpoint | None:
        if not name:
            return None
        if name in endpoint_cache:
            return endpoint_cache[name]
        ep = Endpoint(name=name, element=find_element(elements, name))
        if ep.element is not None:
            ep.candidates = snap_candidates(G, ep.element.center, max_dist=endpoint_snap_pt)
            if not ep.candidates:
                # Beschriftung steht etwas weiter weg (z. B. KS-Name über dem Kasten)
                ep.candidates = snap_candidates(G, ep.element.center, max_dist=2 * endpoint_snap_pt)
        endpoint_cache[name] = ep
        return ep

    result = AnalysisResult(klp_path, page_index, pv_display.width, pv_display.height, scale, G, labels,
                            elements, kuep)
    for cid in cable_ids:
        kc = kuep_by_id.get(cid)
        cable_labels = [lab for lab in labels if cid in lab.cables]
        term = [label_nodes.get(id(lab)) for lab in cable_labels]
        start = endpoint(kc.from_name) if kc else None
        end = endpoint(kc.to_name) if kc else None
        anchors = [n for n in term if n is not None]
        start = _resolved(G, start, anchors)
        end = _resolved(G, end, anchors + ([start.node] if start and start.found else []))
        ends = [ep.node for ep in (start, end) if ep is not None and ep.found]
        route = route_terminals(G, cid, [n for n in term if n is not None] + ends)
        if start is not None and start.found and route.polylines:
            _orient(route, G.nodes[start.node]["pos"])
        elif end is not None and end.found and route.polylines:
            _orient(route, G.nodes[end.node]["pos"], reverse=True)

        res = CableResult(cid, kc, route, label_count=len(cable_labels), start=start, end=end)
        n_anchor = sum(1 for n in term if n is not None)
        if route.polylines:
            res.length_m = route.length * PT_TO_MM * scale / 1000.0
            both_ends = len(ends) == 2
            if n_anchor >= 1 and both_ends and len(route.polylines) == 1:
                res.confidence = "hoch"
            elif n_anchor >= 2 or both_ends:
                res.confidence = "mittel"
            else:
                res.confidence = "niedrig"
        if not cable_labels:
            route.warnings.append("Kabelnummer im KLP nicht gefunden")
        if kc and kc.length_m and res.length_m is not None:
            diff = res.length_m - kc.length_m
            if abs(diff) > max(20.0, 0.3 * kc.length_m):
                route.warnings.append(f"Weglänge weicht vom KÜP ab ({res.length_m:.0f} m statt {kc.length_m:.0f} m)")
        for ep, what in ((start, "Start"), (end, "Ziel")):
            if ep is not None and not ep.found:
                route.warnings.append(f"{what} '{ep.name}' im KLP nicht gefunden")
        result.cables.append(res)

    ok = sum(1 for c in result.cables if c.route.polylines)
    log(f"Kabelwege bestimmt: {ok} von {len(result.cables)} Kabeln")
    _to_display(result, frame)
    return result


def _to_display(res: AnalysisResult, frame: Frame) -> None:
    """Alle Koordinaten aus dem Arbeitsraum in den Anzeigeraum der Seite umrechnen."""
    if frame.k == 0:
        return
    for n in res.graph.nodes:
        res.graph.nodes[n]["pos"] = frame.point_to_display(res.graph.nodes[n]["pos"])
    for _, _, d in res.graph.edges(data=True):
        d["pts"] = frame.to_display(d["pts"])
    for lab in res.labels:
        lab.bbox = frame.bbox_to_display(lab.bbox)
        if lab.anchor is not None:
            lab.anchor = frame.point_to_display(lab.anchor)
        lab.leader = [frame.to_display(pl) for pl in lab.leader]
    for e in res.elements:
        e.center = frame.point_to_display(e.center)
        e.bbox = frame.bbox_to_display(e.bbox)
    for c in res.cables:
        r = c.route
        r.polylines = [frame.to_display(pl) for pl in r.polylines]
        r.terminals = [frame.point_to_display(p) for p in r.terminals]
        r.unsnapped = [frame.point_to_display(p) for p in r.unsnapped]


def _resolved(G, ep: Endpoint | None, anchors: list[int]) -> Endpoint | None:
    """Anschlusspunkt je Kabel wählen (dasselbe Element kann je Kabel anders erreicht werden)."""
    if ep is None:
        return None
    node = choose_candidate(G, ep.candidates, anchors)
    return Endpoint(ep.name, ep.element, ep.candidates, node)


def _orient(route: CableRoute, start_pos: np.ndarray, reverse: bool = False) -> None:
    """Hauptlinienzug so drehen, dass er am Startpunkt beginnt."""
    pl = route.polylines[0]
    d0 = np.linalg.norm(pl[0] - start_pos)
    d1 = np.linalg.norm(pl[-1] - start_pos)
    first_is_start = d0 <= d1
    if reverse:
        first_is_start = not first_is_start
    if not first_is_start:
        route.polylines[0] = pl[::-1].copy()
