"""Kabelwege im Trassengraphen bestimmen.

Für jedes Kabel sind die Trassenpunkte bekannt, an denen es beschriftet ist
(Anker der Kabelnummern-Stapel), ggf. ergänzt um Start-/Zielpunkte. Gesucht ist
der kürzeste zusammenhängende Weg durch das Trassennetz, der alle diese Punkte
berührt (Steinerbaum-Näherung). Ist der Baum ein einfacher Pfad, ergibt das genau
einen Linienzug vom einen zum anderen Ende.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import steiner_tree

from .trassegraph import GraphLocator, polyline_length, project_on_polyline, split_edge_at


@dataclass
class CableRoute:
    cable_id: str
    polylines: list[np.ndarray] = field(default_factory=list)  # erster Eintrag = Hauptweg
    length: float = 0.0                                       # PDF-Punkte
    terminals: list[np.ndarray] = field(default_factory=list)
    unsnapped: list[np.ndarray] = field(default_factory=list)  # Punkte ohne Trasse in der Nähe
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.polylines)


def snap_points(G: nx.Graph, points: list[np.ndarray], max_dist: float = 10.0) -> list[int | None]:
    """Fügt die Punkte als Knoten in den Graphen ein (Kante wird geteilt)."""
    result: list[int | None] = []
    next_id = (max(G.nodes) + 1) if G.number_of_nodes() else 0
    for q in points:
        q = np.asarray(q, float)
        best = None
        # Kandidatenkanten über ein frisches Index-Sample (Graph ändert sich beim Teilen)
        loc = GraphLocator(G, step=2.0)
        d, idx = loc.nearest(q, k=8, max_dist=max_dist + 2.0)
        for dd, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            if not np.isfinite(dd) or ii >= len(loc.refs):
                continue
            u, v = loc.refs[ii]
            if not G.has_edge(u, v):
                continue
            dist, _, c = project_on_polyline(G.edges[u, v]["pts"], q)
            if dist <= max_dist and (best is None or dist < best[0]):
                best = (dist, u, v)
        if best is None:
            result.append(None)
            continue
        _, u, v = best
        n = split_edge_at(G, u, v, q, next_id)
        if n == next_id:
            next_id += 1
        result.append(n)
    return result


def snap_candidates(G: nx.Graph, q, max_dist: float, max_candidates: int = 4) -> list[tuple[int, float]]:
    """Mehrere mögliche Anschlusspunkte (je Trassenkante der nächste) für einen Elementstandort.
    Liefert [(Knoten, Abstand)], nach Abstand sortiert."""
    q = np.asarray(q, float)
    loc = GraphLocator(G, step=2.0)
    d, idx = loc.nearest(q, k=200, max_dist=max_dist + 2.0)
    per_edge: dict[tuple, float] = {}
    for dd, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
        if not np.isfinite(dd) or ii >= len(loc.refs):
            continue
        e = tuple(sorted(loc.refs[ii]))
        if e not in per_edge or dd < per_edge[e]:
            per_edge[e] = dd
    edges = sorted(per_edge.items(), key=lambda kv: kv[1])[:max_candidates]
    out = []
    next_id = max(G.nodes) + 1
    for (u, v), _ in edges:
        if not G.has_edge(u, v):
            continue
        dist, _, c = project_on_polyline(G.edges[u, v]["pts"], q)
        if dist > max_dist:
            continue
        n = split_edge_at(G, u, v, c, next_id)
        if n == next_id:
            next_id += 1
        out.append((n, float(dist)))
    return sorted(out, key=lambda t: t[1])


def choose_candidate(G: nx.Graph, candidates: list[tuple[int, float]], anchors: list[int],
                     snap_weight: float = 3.0) -> int | None:
    """Wählt den Anschlusspunkt, der am günstigsten zu den bekannten Trassenpunkten liegt."""
    if not candidates:
        return None
    if not anchors:
        return candidates[0][0]
    dist = nx.multi_source_dijkstra_path_length(G, set(anchors), weight="length")
    best = None
    for n, sd in candidates:
        cost = dist.get(n, np.inf) + snap_weight * sd
        if best is None or cost < best[0]:
            best = (cost, n)
    return best[1] if np.isfinite(best[0]) else candidates[0][0]


def _edge_polyline_from(G: nx.Graph, a, b) -> np.ndarray:
    pl = G.edges[a, b]["pts"]
    if np.linalg.norm(pl[0] - G.nodes[a]["pos"]) > np.linalg.norm(pl[-1] - G.nodes[a]["pos"]):
        pl = pl[::-1]
    return pl


def path_to_polyline(G: nx.Graph, path: list) -> np.ndarray:
    parts = [G.nodes[path[0]]["pos"][None, :]]
    for a, b in zip(path, path[1:]):
        parts.append(_edge_polyline_from(G, a, b)[1:])
    return np.vstack(parts)


def route_terminals(G: nx.Graph, cable_id: str, terminal_nodes: list[int], min_branch: float = 25.0) -> CableRoute:
    route = CableRoute(cable_id=cable_id)
    nodes = list(dict.fromkeys(n for n in terminal_nodes if n is not None))
    route.terminals = [G.nodes[n]["pos"] for n in nodes]
    if len(nodes) < 2:
        if len(nodes) == 1:
            route.warnings.append("nur ein Trassenpunkt bekannt – kein Weg bestimmbar")
        return route

    # Terminals in unterschiedlichen Netzteilen können nicht verbunden werden
    comps: dict[frozenset, list] = {}
    for n in nodes:
        comps.setdefault(frozenset(nx.node_connected_component(G, n)), []).append(n)
    if len(comps) > 1:
        route.warnings.append(f"Trassenpunkte liegen in {len(comps)} getrennten Netzteilen")

    for comp, comp_nodes in comps.items():
        if len(comp_nodes) < 2:
            continue
        if len(comp_nodes) == 2:
            path = nx.shortest_path(G, comp_nodes[0], comp_nodes[1], weight="length")
            T = nx.Graph()
            nx.add_path(T, path)
        else:
            T = steiner_tree(G.subgraph(comp), comp_nodes, weight="length", method="mehlhorn")
        route.polylines.extend(_tree_to_polylines(G, T, comp_nodes))

    route.polylines.sort(key=polyline_length, reverse=True)
    # Kurze Stichäste (z. B. Anker und Element am selben Kasten) nicht als Verzweigung werten
    if route.polylines:
        route.polylines = route.polylines[:1] + [p for p in route.polylines[1:] if polyline_length(p) >= min_branch]
    route.length = float(sum(polyline_length(p) for p in route.polylines))
    if len(route.polylines) > 1:
        route.warnings.append("Weg verzweigt sich – mehrere Linienzüge")
    return route


def _tree_to_polylines(G: nx.Graph, T: nx.Graph, terminals: list) -> list[np.ndarray]:
    """Zerlegt den Baum in Linienzüge: längster Pfad zuerst, dann die Äste."""
    T = T.copy()
    out = []
    while T.number_of_edges() > 0:
        leaves = [n for n in T.nodes if T.degree(n) == 1] or list(T.nodes)
        # längster Pfad zwischen zwei Blättern (Baum: zweimal BFS)
        def farthest(src):
            dist = nx.single_source_dijkstra_path_length(T, src, weight=lambda a, b, d: G.edges[a, b]["length"])
            return max(dist.items(), key=lambda kv: kv[1])[0]
        a = farthest(leaves[0])
        b = farthest(a)
        path = nx.shortest_path(T, a, b, weight=lambda u, v, d: G.edges[u, v]["length"])
        if len(path) < 2:
            break
        out.append(path_to_polyline(G, path))
        T.remove_edges_from(zip(path, path[1:]))
        T.remove_nodes_from([n for n in list(T.nodes) if T.degree(n) == 0])
    return out
