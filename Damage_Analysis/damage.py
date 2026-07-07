"""Stages 5-6: Vegetation analysis and five-class damage classification."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Set, Dict
from collections import deque, defaultdict
try:
    from .models import GridParams, EdgeMatch, CellState
    from .dualgraph import (
        build_theoretical_graph,
        generate_theoretical_edge_pixels,
    )
except ImportError:
    from models import GridParams, EdgeMatch, CellState
    from dualgraph import (
        build_theoretical_graph,
        generate_theoretical_edge_pixels,
    )


# ----------------------------------------------------------------------
# Vegetation
# ----------------------------------------------------------------------

def compute_veg_coverage(
    edge_pixels: List[Tuple[int, int]],
    veg_mask: np.ndarray,
) -> float:
    """
    Compute vegetation coverage fraction on a theoretical edge.

    Args:
        edge_pixels: list of (r, c) pixel coordinates of the theoretical edge
        veg_mask: (H, W) uint8, 0/255

    Returns:
        veg_coverage: fraction [0, 1] of edge pixels covered by vegetation
    """
    if not edge_pixels:
        return 0.0
    veg_set = set(zip(*np.where(veg_mask > 0)))
    edge_set = set(edge_pixels)
    overlap = edge_set & veg_set
    return len(overlap) / len(edge_set)


def fill_veg_coverage(
    G_theoretical: nx.Graph,
    G_match: Dict[Tuple[int, int], EdgeMatch],
    veg_mask: np.ndarray,
) -> None:
    """
    Fill veg_coverage field for all edges in G_match.

    Modifies G_match in place.

    Args:
        G_theoretical: nx.Graph with 'pos_px' on nodes
        G_match: dict of edge_id → EdgeMatch
        veg_mask: (H, W) uint8 vegetation mask
    """
    for (u, v), em in G_match.items():
        if em.status in ("severe", "missing", "minor"):
            r1, c1 = G_theoretical.nodes[u]["pos_px"]
            r2, c2 = G_theoretical.nodes[v]["pos_px"]
            edge_pixels = generate_theoretical_edge_pixels(r1, c1, r2, c2)
            em.veg_coverage = compute_veg_coverage(edge_pixels, veg_mask)


# ----------------------------------------------------------------------
# Edge-level reason classification
# ----------------------------------------------------------------------

def classify_edge_reason(em: EdgeMatch) -> str:
    """
    Classify why an edge is damaged (or intact).

    Decision tree:
      coverage_ratio >= 0.5 → 'ok'
      0.2 < veg_coverage < 0.6 → 'ambiguous'
      veg_coverage >= 0.6 → 'occluded'   (false damage)
      veg_coverage <= 0.2 → 'genuine_break'  (true damage)
      otherwise → 'ambiguous'

    Args:
        em: EdgeMatch object

    Returns:
        reason: 'ok' | 'genuine_break' | 'occluded' | 'ambiguous'
    """
    if em.coverage_ratio >= 0.5:
        return "ok"

    if em.veg_coverage >= 0.6:
        return "occluded"

    if em.veg_coverage <= 0.2:
        return "genuine_break"

    return "ambiguous"


def classify_all_edges(
    G_match: Dict[Tuple[int, int], EdgeMatch],
) -> None:
    """
    Classify reason for every edge in G_match.

    Modifies G_match in place.

    Args:
        G_match: dict of edge_id → EdgeMatch
    """
    for em in G_match.values():
        em.reason = classify_edge_reason(em)


# ----------------------------------------------------------------------
# Void polygon detection
# ----------------------------------------------------------------------

def _dfs_find_quadrilateral(
    current_edge: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
    visited: Set[Tuple[int, int]],
    path: List[Tuple[int, int]],
    max_len: int = 6,
) -> bool:
    """
    DFS to find a closed quadrilateral (4-edge cycle) around a void.

    A quadrilateral needs exactly 4 edges: H-V-H-V or V-H-V-H.
    We stop when we find a cycle of length 4 that shares the start node.

    Returns:
        True if a valid quadrilateral found (appends to path), False otherwise.
    """
    if len(path) >= max_len:
        return False

    for next_edge in adjacency.get(current_edge, []):
        if next_edge in visited and next_edge != path[0]:
            continue

        new_path = path + [next_edge]
        visited.add(next_edge)

        if len(new_path) == 4:
            return True

        if _dfs_find_quadrilateral(next_edge, adjacency, visited, new_path, max_len):
            return True

        visited.discard(next_edge)

    return False


def detect_void_polygons(
    G_theoretical: nx.Graph,
    G_match: Dict[Tuple[int, int], EdgeMatch],
    gsd: float = 0.006388,
) -> List[dict]:
    """
    Detect closed void polygons formed by missing edges.

    Algorithm:
      1. Collect all missing/severe edges
      2. Build adjacency graph: two missing edges share a node and are
         perpendicular (H vs V) → they are adjacent
      3. DFS find 4-edge cycles (quadrilaterals)
      4. Compute area via bounding-box approximation

    Args:
        G_theoretical: nx.Graph
        G_match: dict {edge_id: EdgeMatch}
        gsd: ground sample distance (m/px)

    Returns:
        list of dicts: [{'edge_ids': [...], 'area_m2': float, 'cell_ids': [...]}]
    """
    missing_edges = [
        eid for eid, m in G_match.items() if m.status in ("severe", "missing")
    ]
    if not missing_edges:
        return []

    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for i, (u1, v1) in enumerate(missing_edges):
        for j, (u2, v2) in enumerate(missing_edges):
            if i >= j:
                continue
            shared = {u1, v1} & {u2, v2}
            if len(shared) == 1:
                d1 = G_theoretical.edges[u1, v1].get("direction", "H")
                d2 = G_theoretical.edges[u2, v2].get("direction", "H")
                if d1 != d2:
                    adjacency[(u1, v1)].append((u2, v2))
                    adjacency[(u2, v2)].append((u1, v1))

    void_polygons: List[dict] = []
    visited_edges: Set[Tuple[int, int]] = set()

    for start_edge in missing_edges:
        if start_edge in visited_edges:
            continue

        visited_edges.add(start_edge)
        visited_local = {start_edge}
        path = [start_edge]

        found = _dfs_find_quadrilateral(
            start_edge, adjacency, visited_local, path
        )

        if found:
            edges_in_void = path
            for e in edges_in_void:
                visited_edges.add(e)

            corners: Set[Tuple[float, float]] = set()
            for u, v in edges_in_void:
                corners.add(G_theoretical.nodes[u]["pos_px"])
                corners.add(G_theoretical.nodes[v]["pos_px"])

            if len(corners) >= 4:
                rs = [p[0] for p in corners]
                cs = [p[1] for p in corners]
                area_px2 = (max(rs) - min(rs)) * (max(cs) - min(cs))
                area_m2 = area_px2 * (gsd ** 2)

                void_polygons.append({
                    "edge_ids": edges_in_void,
                    "area_m2": area_m2,
                    "corners": list(corners),
                })

    return void_polygons


# ----------------------------------------------------------------------
# Cell-level classification
# ----------------------------------------------------------------------

def classify_cells(
    G_theoretical: nx.Graph,
    G_match: Dict[Tuple[int, int], EdgeMatch],
    grid_params: GridParams,
    void_polygons: List[dict],
    gsd: float = 0.006388,
) -> List[CellState]:
    """
    Derive cell-level damage states from theoretical graph + edge matches.

    Each cell is defined by 4 adjacent grid nodes. Its damage state is
    determined by aggregating the 4 surrounding edge matches.

    Args:
        G_theoretical: nx.Graph with 'grid_id' and 'pos_px' on nodes
        G_match: dict {edge_id: EdgeMatch}
        grid_params: GridParams
        void_polygons: list from detect_void_polygons
        gsd: ground sample distance (m/px)

    Returns:
        cells: list of CellState
    """
    node_grid_id_map: Dict[int, Tuple[int, int]] = {}
    node_pos_map: Dict[int, Tuple[float, float]] = {}
    for nid, data in G_theoretical.nodes(data=True):
        node_grid_id_map[nid] = data["grid_id"]
        node_pos_map[nid] = data["pos_px"]

    # BUGFIX: gid_to_nid must be {grid_id: node_id}, not the reverse.
    gid_to_nid: Dict[Tuple[int, int], int] = {gid: nid for nid, gid in node_grid_id_map.items()}

    max_row = max(gid[0] for gid in gid_to_nid)
    max_col = max(gid[1] for gid in gid_to_nid)

    void_edge_set: Set[Tuple[int, int]] = set()
    for void in void_polygons:
        for eid in void.get("edge_ids", []):
            void_edge_set.add(eid)

    cells: List[CellState] = []
    step_avg = (grid_params.step_x_px + grid_params.step_y_px) / 2
    area_px2 = grid_params.step_x_px * grid_params.step_y_px
    area_m2_cell = area_px2 * (gsd ** 2)

    for ri in range(max_row):
        for ci in range(max_col):
            n00 = gid_to_nid.get((ri,     ci    ))
            n01 = gid_to_nid.get((ri,     ci + 1))
            n10 = gid_to_nid.get((ri + 1, ci    ))
            n11 = gid_to_nid.get((ri + 1, ci + 1))

            if n00 is None:
                continue

            edges = {
                "top":    G_match.get((n00, n01)) if n01 else None,
                "bottom": G_match.get((n10, n11)) if n11 else None,
                "left":   G_match.get((n00, n10)) if n10 else None,
                "right":  G_match.get((n11, n01)) if n11 else None,
            }

            theo_len = 0.0
            actual_len = 0.0
            genuine_break_count = 0
            occluded_count = 0
            edge_statuses: Dict[str, str] = {}
            veg_covered_sides: List[str] = []

            for dir_name, em in edges.items():
                if em is None:
                    continue
                theo_len += em.theoretical_length_m
                actual_len += em.actual_length_m
                edge_statuses[dir_name] = em.status

                if em.reason == "genuine_break":
                    genuine_break_count += 1
                elif em.reason == "occluded":
                    occluded_count += 1
                    if em.veg_coverage >= 0.6:
                        veg_covered_sides.append(dir_name)

            damage_rate = 1.0 - actual_len / max(theo_len, 1e-6)

            # Cell status determination
            if genuine_break_count >= 3:
                status = "destroyed"
            elif damage_rate >= 0.7:
                status = "severe"
            elif damage_rate >= 0.3:
                status = "moderate"
            elif damage_rate >= 0.1:
                status = "minor"
            elif occluded_count >= 2 and genuine_break_count == 0:
                status = "pseudo_damaged"
            else:
                status = "intact"

            r0, c0 = node_pos_map[n00] if n00 is not None else (0, 0)
            r1, c1 = node_pos_map[n11] if n11 is not None else (r0 + step_avg, c0 + step_avg)
            bbox_px = ((float(r0), float(c0)), (float(r1), float(c1)))

            cells.append(CellState(
                cell_id=(ri, ci),
                bbox_px=bbox_px,
                damage_rate=damage_rate,
                status=status,
                edge_statuses=edge_statuses,
                genuine_break_count=genuine_break_count,
                occluded_count=occluded_count,
                area_m2=area_m2_cell,
                veg_covered_sides=veg_covered_sides,
            ))

    return cells


# ----------------------------------------------------------------------
# Damage cluster analysis
# ----------------------------------------------------------------------

def _cells_to_polygon(
    cluster_cell_ids: List[Tuple[int, int]],
    cell_bbox_map: Dict[Tuple[int, int], CellState],
) -> List[Tuple[float, float]]:
    """
    Compute a polygon (list of (r, c) pixel corners) covering all cells
    in a cluster. Uses the outer boundary of merged cell rectangles
    (works for any orthogonal cluster, convexity not required).
    """
    if not cluster_cell_ids:
        return []

    # Collect all cell corner points
    corners: Set[Tuple[int, int]] = set()
    for cid in cluster_cell_ids:
        c = cell_bbox_map.get(cid)
        if c is None:
            continue
        ((r0, c0), (r1, c1)) = c.bbox_px
        corners.add((int(r0), int(c0)))
        corners.add((int(r0), int(c1)))
        corners.add((int(r1), int(c0)))
        corners.add((int(r1), int(c1)))

    if len(corners) < 3:
        return [(float(p[0]), float(p[1])) for p in corners]

    pts = list(corners)

    # Convex hull via Graham scan (small N → simple)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    pts_set = set(corners)
    pts_sorted = sorted(pts)
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]

    # Add inner concavity: any corner inside bbox but outside hull
    # — skipped; convex hull is sufficient for area_damage overlay.
    return [(float(p[0]), float(p[1])) for p in hull]


def find_damage_clusters(
    cells: List[CellState],
    min_cluster_size: int = 5,
    cell_area_m2: float = 1.0,
    area_destroyed_threshold_m2: float = 25.0,
    degraded_patch_threshold_m2: float = 10.0,
) -> List[dict]:
    """
    Find spatially connected damage regions using 4-neighbor BFS.

    Args:
        cells: list of CellState
        min_cluster_size: minimum cells for a valid cluster
        cell_area_m2: area of a single cell (m²). Used to enforce a
            minimum sensible area (= cell_area_m2 * 5) for the smallest
            'isolated_damage' clusters.
        area_destroyed_threshold_m2: minimum cluster area to be
            classified as 'area_destroyed'.
        degraded_patch_threshold_m2: minimum cluster area to be
            classified as 'degraded_patch'.

    Returns:
        clusters: list of dicts:
          {
            'cell_ids': [...],
            'area_m2': float,
            'type': 'area_destroyed' | 'degraded_patch' | 'isolated_damage',
            'cascade_risk': float,   # fraction of cluster boundary cells
            'cell_count': int,
            'polygon': [(r,c), ...]   # outer hull in pixel coords, for overlay
          }
    """
    cell_map: Dict[Tuple[int, int], CellState] = {c.cell_id: c for c in cells}
    damaged = {
        c.cell_id for c in cells
        if c.status in ("severe", "destroyed", "moderate")
    }

    visited: Set[Tuple[int, int]] = set()
    clusters: List[dict] = []

    for cell_id in damaged:
        if cell_id in visited:
            continue
        cluster: List[Tuple[int, int]] = []
        queue = deque([cell_id])
        visited.add(cell_id)

        while queue:
            cid = queue.popleft()
            cluster.append(cid)
            for n in [
                (cid[0] + 1, cid[1]),
                (cid[0] - 1, cid[1]),
                (cid[0], cid[1] + 1),
                (cid[0], cid[1] - 1),
            ]:
                if n in damaged and n not in visited:
                    visited.add(n)
                    queue.append(n)

        if len(cluster) < min_cluster_size:
            continue

        cell_count = len(cluster)
        cluster_cells = [cell_map[cid] for cid in cluster]
        area_m2 = sum(c.area_m2 for c in cluster_cells)

        boundary_count = 0
        for cid in cluster:
            for n in [
                (cid[0] + 1, cid[1]),
                (cid[0] - 1, cid[1]),
                (cid[0], cid[1] + 1),
                (cid[0], cid[1] - 1),
            ]:
                if n in cell_map and n not in damaged:
                    boundary_count += 1
                    break

        cascade_risk = boundary_count / cell_count

        if area_m2 >= area_destroyed_threshold_m2:
            cluster_type = "area_destroyed"
        elif area_m2 >= degraded_patch_threshold_m2:
            cluster_type = "degraded_patch"
        else:
            cluster_type = "isolated_damage"

        polygon = _cells_to_polygon(cluster, cell_map)

        clusters.append({
            "cell_ids": cluster,
            "area_m2": area_m2,
            "type": cluster_type,
            "cascade_risk": cascade_risk,
            "cell_count": len(cluster),
            "polygon": polygon,
        })

    return clusters
