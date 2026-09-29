"""Trassennetz des Kabellageplans als Graph.

Die Kabeltrassen sind im KLP als rote Strichlinien gezeichnet. Jeder Strich wird in
Laufrichtung verlängert (schließt die Strichlücken, ohne quer zu verbreitern),
gerastert und skelettiert. Aus dem Skelett entsteht ein Graph: Knoten an Enden und
Abzweigen, Kanten mit Polylinie und Länge in PDF-Punkten.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .pdfvector import Primitive


@dataclass
class GraphParams:
    raster_scale: float = 2.0      # Pixel je PDF-Punkt
    dash_extend: float = 2.2       # Verlängerung je Strichende (Punkte), schließt Strichlücken
    trasse_line_width: float = 1.5   # Rasterbreite einer einfachen Strichlinie (Punkte)
    kanal_line_width: float = 5.0    # Rasterbreite für Doppellinien (verschmilzt beide Linien)
    bridge_distance: float = 7.0   # offene Enden werden bis zu diesem Abstand angebunden
    simplify_tol: float = 0.4


_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def rasterize_trasse(groups: list[tuple[list[Primitive], float]], width: float, height: float,
                     p: GraphParams) -> np.ndarray:
    """``groups``: Liste aus (Primitive, Linienbreite in Punkten). Doppellinien (Kabelkanal)
    werden mit einer Breite gerastert, die beide Linien zu einem Band verschmilzt."""
    s = p.raster_scale
    img = np.zeros((int(np.ceil(height * s)) + 2, int(np.ceil(width * s)) + 2), np.uint8)
    for prims, line_width in groups:
        _draw_dashes(img, prims, max(1, int(round(line_width * s))), p)
    return img


def _draw_dashes(img: np.ndarray, prims: list[Primitive], line_px: int, p: GraphParams) -> None:
    s = p.raster_scale
    for prim in prims:
        for pl in prim.polylines:
            pl = pl.copy()
            if len(pl) >= 2 and p.dash_extend > 0:
                d0 = pl[0] - pl[1]
                d1 = pl[-1] - pl[-2]
                n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
                if n0 > 1e-6:
                    pl[0] = pl[0] + d0 / n0 * p.dash_extend
                if n1 > 1e-6:
                    pl[-1] = pl[-1] + d1 / n1 * p.dash_extend
            pts = np.round(pl * s * 8).astype(np.int32)
            cv2.polylines(img, [pts], False, 1, line_px, lineType=cv2.LINE_8, shift=3)


def _rdp(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        v = pts - a
        d = np.abs(ab[0] * v[:, 1] - ab[1] * v[:, 0]) / n
    i = int(np.argmax(d))
    if d[i] > tol:
        left = _rdp(pts[: i + 1], tol)
        right = _rdp(pts[i:], tol)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


def polyline_length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def skeleton_to_graph(skel: np.ndarray, scale: float, simplify_tol: float) -> nx.Graph:
    ys, xs = np.nonzero(skel)
    on = set(zip(ys.tolist(), xs.tolist()))

    def neighbors(y, x):
        return [(y + dy, x + dx) for dy, dx in _NEIGH if (y + dy, x + dx) in on]

    deg = {px: len(neighbors(*px)) for px in on}
    node_px = {px for px, d in deg.items() if d != 2}

    # Benachbarte Knotenpixel (Kreuzungsklumpen) zu einem Knoten zusammenfassen
    node_id: dict[tuple[int, int], int] = {}
    G = nx.Graph()
    nid = 0
    for px in node_px:
        if px in node_id:
            continue
        stack, members = [px], []
        node_id[px] = nid
        while stack:
            q = stack.pop()
            members.append(q)
            for nb in neighbors(*q):
                if nb in node_px and nb not in node_id:
                    node_id[nb] = nid
                    stack.append(nb)
        arr = np.array(members, float)
        G.add_node(nid, pos=np.array([arr[:, 1].mean(), arr[:, 0].mean()]) / scale)
        nid += 1

    visited_edges: set[frozenset] = set()
    for start in node_px:
        for nb in neighbors(*start):
            if nb in node_px:
                a, b = node_id[start], node_id[nb]
                if a != b and not G.has_edge(a, b):
                    pts = np.array([G.nodes[a]["pos"], G.nodes[b]["pos"]])
                    G.add_edge(a, b, pts=pts, length=polyline_length(pts))
                continue
            key = frozenset((start, nb))
            if key in visited_edges:
                continue
            path = [start, nb]
            visited_edges.add(key)
            prev, cur = start, nb
            while cur not in node_px:
                nxt = [q for q in neighbors(*cur) if q != prev and q not in path[-3:]]
                if not nxt:
                    break
                # Bei Mehrdeutigkeit den orthogonalen Nachbarn bevorzugen
                nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
                prev, cur = cur, nxt[0]
                visited_edges.add(frozenset((prev, cur)))
                path.append(cur)
            if cur not in node_px:
                continue
            a, b = node_id[start], node_id[cur]
            arr = np.array([(x, y) for y, x in path], float) / scale
            arr[0] = G.nodes[a]["pos"]
            arr[-1] = G.nodes[b]["pos"]
            arr = _rdp(arr, simplify_tol)
            length = polyline_length(arr)
            if a == b and length < 3:
                continue
            if G.has_edge(a, b):
                if G.edges[a, b]["length"] <= length:
                    continue
            G.add_edge(a, b, pts=arr, length=length)

    # Reine Kreise (Skelettringe ohne Knoten) werden ignoriert – sie sind keine Trassen.
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return G


def _edge_sample_index(G: nx.Graph, step: float = 1.0):
    pts, refs = [], []
    for u, v, data in G.edges(data=True):
        pl = data["pts"]
        for i in range(len(pl) - 1):
            a, b = pl[i], pl[i + 1]
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            for k in range(n + 1):
                pts.append(a + (b - a) * k / n)
                refs.append((u, v))
    return np.array(pts), refs


def bridge_gaps(G: nx.Graph, max_dist: float) -> int:
    """Offene Trassenenden an die nächste fremde Trasse anbinden (Stoßstellen, Symbole)."""
    if G.number_of_edges() == 0:
        return 0
    pts, refs = _edge_sample_index(G)
    tree = cKDTree(pts)
    added = 0
    ends = [n for n in G.nodes if G.degree(n) == 1]
    next_id = max(G.nodes) + 1
    for n in ends:
        if G.degree(n) != 1:
            continue
        p = G.nodes[n]["pos"]
        own_edges = {tuple(sorted(e)) for e in G.edges(n)}
        # Richtung des Endes (vom Nachbarn weg)
        (nb,) = list(G.neighbors(n))
        pl = G.edges[n, nb]["pts"]
        inner = pl[1] if np.allclose(pl[0], p) else pl[-2]
        direction = p - inner
        dn = np.linalg.norm(direction)
        direction = direction / dn if dn > 1e-9 else direction
        best = None
        for d, idx in zip(*tree.query(p, k=40, distance_upper_bound=max_dist)):
            if not np.isfinite(d) or idx >= len(refs):
                continue
            e = tuple(sorted(refs[idx]))
            if e in own_edges or n in e:
                continue
            q = pts[idx]
            v = q - p
            vn = np.linalg.norm(v)
            # Nur Anschlüsse nach vorn (bzw. sehr nahe) zulassen
            if vn > 1.0 and np.dot(v / vn, direction) < -0.2:
                continue
            score = vn - (np.dot(v / vn, direction) if vn > 1e-9 else 1.0)
            if best is None or score < best[0]:
                best = (score, q, e)
        if best is None:
            continue
        _, q, (u, v) = best
        if not G.has_edge(u, v):
            continue
        new = split_edge_at(G, u, v, q, next_id)
        if new == next_id:
            next_id += 1
        if new != n and not G.has_edge(n, new):
            seg = np.array([p, G.nodes[new]["pos"]])
            G.add_edge(n, new, pts=seg, length=polyline_length(seg), bridge=True)
            added += 1
    return added


def project_on_polyline(pl: np.ndarray, q: np.ndarray):
    """Nächster Punkt auf Polylinie: (Abstand, Segmentindex, Punkt)."""
    best = (np.inf, 0, pl[0])
    for i in range(len(pl) - 1):
        a, b = pl[i], pl[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip((q - a) @ ab / L2, 0, 1))
        c = a + ab * t
        d = float(np.linalg.norm(q - c))
        if d < best[0]:
            best = (d, i, c)
    return best


def split_edge_at(G: nx.Graph, u, v, q: np.ndarray, new_id, snap: float = 0.5):
    """Teilt Kante (u,v) am Punkt q. Liefert den Knoten an q (ggf. vorhandenen Endknoten)."""
    data = G.edges[u, v]
    pl = data["pts"]
    # Polylinie beginnt bei u?
    if not np.allclose(pl[0], G.nodes[u]["pos"], atol=1e-6):
        pl = pl[::-1]
    d, i, c = project_on_polyline(pl, q)
    if np.linalg.norm(c - G.nodes[u]["pos"]) < snap:
        return u
    if np.linalg.norm(c - G.nodes[v]["pos"]) < snap:
        return v
    first = np.vstack([pl[: i + 1], c])
    second = np.vstack([c, pl[i + 1:]])
    extra = {k: val for k, val in data.items() if k not in ("pts", "length")}
    G.remove_edge(u, v)
    G.add_node(new_id, pos=c)
    G.add_edge(u, new_id, pts=first, length=polyline_length(first), **extra)
    G.add_edge(new_id, v, pts=second, length=polyline_length(second), **extra)
    return new_id


def build_trasse_graph(trasse: list[Primitive], kanal: list[Primitive], width: float, height: float,
                       params: GraphParams | None = None) -> nx.Graph:
    p = params or GraphParams()
    img = rasterize_trasse([(trasse, p.trasse_line_width), (kanal, p.kanal_line_width)], width, height, p)
    skel = skeletonize(img > 0)
    G = skeleton_to_graph(skel, p.raster_scale, p.simplify_tol)
    G = nx.convert_node_labels_to_integers(G)
    bridge_gaps(G, p.bridge_distance)
    _prune_spurs(G, max_len=2.5)
    return G


def _prune_spurs(G: nx.Graph, max_len: float) -> None:
    """Kurze Skelettstummel (Artefakte an Ecken) entfernen."""
    changed = True
    while changed:
        changed = False
        for n in [n for n in G.nodes if G.degree(n) == 1]:
            if n not in G or G.degree(n) != 1:
                continue
            (nb,) = list(G.neighbors(n))
            if G.degree(nb) >= 3 and G.edges[n, nb]["length"] < max_len:
                G.remove_node(n)
                changed = True


class GraphLocator:
    """Schnelle Suche des nächsten Graphpunkts."""

    def __init__(self, G: nx.Graph, step: float = 1.0):
        self.G = G
        self.pts, self.refs = _edge_sample_index(G, step)
        self.tree = cKDTree(self.pts)

    def nearest(self, q, k: int = 1, max_dist: float = np.inf):
        d, idx = self.tree.query(q, k=k, distance_upper_bound=max_dist)
        return d, idx
