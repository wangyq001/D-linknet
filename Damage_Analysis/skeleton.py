"""Stage 2: Skeletonization and physical graph construction."""

import numpy as np
import networkx as nx
from collections import deque
from typing import List, Tuple, Set, Dict
from skimage.morphology import skeletonize


def count_neighbors(skel: np.ndarray, r: int, c: int) -> int:
    """8-neighbor non-zero pixel count for skeleton."""
    h, w = skel.shape
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] > 0:
                count += 1
    return count


def get_neighbor_directions(
    skel: np.ndarray, r: int, c: int
) -> List[str]:
    """Return list of direction names for non-zero 8-neighbors."""
    dirs = []
    h, w = skel.shape
    direction_map = {
        (-1, -1): "NW", (-1, 0): "N", (-1, 1): "NE",
        (0, -1): "W",                (0, 1): "E",
        (1, -1): "SW",  (1, 0): "S",  (1, 1): "SE",
    }
    for dr, dc in direction_map:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] > 0:
            dirs.append(direction_map[(dr, dc)])
    return dirs


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize binary mask using Lee's algorithm.

    Args:
        mask: (H, W) uint8, values 0 or 255

    Returns:
        skeleton: (H, W) uint8, values 0 or 1
    """
    binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary, method="lee").astype(np.uint8)
    return skeleton


def _dfs_spur(
    r: int, c: int,
    skel: np.ndarray,
    visited: np.ndarray,
    min_length: int,
) -> List[Tuple[int, int]]:
    """Collect spur branch pixels from endpoint (r, c) via DFS."""
    h, w = skel.shape
    stack = [(r, c)]
    result = []
    visited_local: Set[Tuple[int, int]] = set()

    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited_local:
            continue
        visited_local.add((cr, cc))
        result.append((cr, cc))
        if len(result) > min_length:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] > 0:
                if (nr, nc) not in visited_local:
                    degree = count_neighbors(skel, nr, nc)
                    if degree <= 2:
                        stack.append((nr, nc))

    return result


def remove_spurs(skeleton: np.ndarray, min_spur_length: int = 10) -> np.ndarray:
    """
    Remove short spurs (branches) from skeleton.

    From each endpoint, DFS collects the branch. If the branch length is below
    min_spur_length, those pixels are removed.

    Args:
        skeleton: (H, W) uint8, values 0 or 1
        min_spur_length: minimum spur length threshold (pixels)

    Returns:
        cleaned: (H, W) uint8, values 0 or 1
    """
    skel = skeleton.copy()
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=np.uint8)

    endpoints = [
        (r, c) for r in range(h) for c in range(w)
        if skel[r, c] > 0 and count_neighbors(skel, r, c) == 1
    ]

    for ep_r, ep_c in endpoints:
        if visited[ep_r, ep_c]:
            continue
        branch = _dfs_spur(ep_r, ep_c, skel, visited, min_spur_length)
        if len(branch) < min_spur_length:
            for br, bc in branch:
                skel[br, bc] = 0

    return skel


def detect_nodes(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Detect junction and endpoint nodes in skeleton.

    Node types:
      - endpoint: degree = 1 (line terminates here)
      - cross point: degree >= 3 with both horizontal and vertical neighbors
      - branch point: degree >= 3 without both H+V neighbors

    Args:
        skeleton: (H, W) uint8, values 0 or 1

    Returns:
        cross_points: list of (r, c) cross junctions
        end_points: list of (r, c) endpoints
        branch_points: list of (r, c) branch points
    """
    cross_pts, end_pts, branch_pts = [], [], []

    for r in range(skeleton.shape[0]):
        for c in range(skeleton.shape[1]):
            if skeleton[r, c] == 0:
                continue
            degree = count_neighbors(skeleton, r, c)
            if degree == 1:
                end_pts.append((r, c))
            elif degree >= 3:
                dirs = get_neighbor_directions(skeleton, r, c)
                has_h = any(d in dirs for d in ["E", "W"])
                has_v = any(d in dirs for d in ["N", "S"])
                if has_h and has_v:
                    cross_pts.append((r, c))
                else:
                    branch_pts.append((r, c))

    return cross_pts, end_pts, branch_pts


def trace_edges(
    skeleton: np.ndarray,
    cross_pts: List[Tuple[int, int]],
    end_pts: List[Tuple[int, int]],
    branch_pts: List[Tuple[int, int]],
) -> nx.Graph:
    """
    Trace skeleton into edge-pixels graph.

    Strategy: For every skeleton pixel, walk in all 4 cardinal directions
    until reaching another skeleton node (cross/branch/end). Each completed
    walk becomes an edge in the graph.

    Args:
        skeleton: (H, W) uint8
        cross_pts, end_pts, branch_pts: node coordinates

    Returns:
        G: nx.Graph
            Node attributes: pos_px=(r, c), type='cross'|'branch'|'end'
            Edge attributes: pixels=[(r,c),...], length_px=int, direction='H'|'V'|'diagonal'
    """
    G = nx.Graph()
    h, w = skeleton.shape

    all_nodes = cross_pts + branch_pts + end_pts
    node_idx: Dict[Tuple[int, int], int] = {pt: i for i, pt in enumerate(all_nodes)}

    for i, (r, c) in enumerate(all_nodes):
        if (r, c) in cross_pts:
            t = "cross"
        elif (r, c) in branch_pts:
            t = "branch"
        else:
            t = "end"
        G.add_node(i, pos_px=(r, c), type=t)

    # Edge between two nodes is recorded when we walk along the skeleton
    # from one to the other in some direction.
    traced_edges: Set[Tuple[int, int]] = set()

    def walk_and_record(start_idx: int, start_r: int, start_c: int,
                        init_dr: int, init_dc: int) -> None:
        """Walk from (start_r,start_c) in direction (init_dr,init_dc) until
        reaching another node, and record an edge for each step transition."""
        prev_r, prev_c = start_r, start_c
        cur_r, cur_c = start_r + init_dr, start_c + init_dc
        path: List[Tuple[int, int]] = [(start_r, start_c)]

        while 0 <= cur_r < h and 0 <= cur_c < w and skeleton[cur_r, cur_c] > 0:
            path.append((cur_r, cur_c))

            if (cur_r, cur_c) in node_idx and (cur_r, cur_c) != (start_r, start_c):
                end_idx = node_idx[(cur_r, cur_c)]
                edge_key = tuple(sorted([start_idx, end_idx]))
                if edge_key not in traced_edges:
                    length_px = len(path)
                    dr_total = cur_r - start_r
                    dc_total = cur_c - start_c
                    if abs(dc_total) >= abs(dr_total) * 2:
                        direction = "H"
                    elif abs(dr_total) >= abs(dc_total) * 2:
                        direction = "V"
                    else:
                        direction = "diagonal"

                    G.add_edge(
                        start_idx, end_idx,
                        pixels=path[:],
                        length_px=length_px,
                        direction=direction,
                    )
                    traced_edges.add(edge_key)
                return

            # Continue walking: prefer continuing same direction (collinear),
            # else branch to a different neighbor.
            next_r = cur_r + init_dr
            next_c = cur_c + init_dc
            if (0 <= next_r < h and 0 <= next_c < w
                    and skeleton[next_r, next_c] > 0
                    and (next_r, next_c) != (prev_r, prev_c)):
                prev_r, prev_c = cur_r, cur_c
                cur_r, cur_c = next_r, next_c
                continue

            # Try any other neighbor (skip prev)
            advanced = False
            for ndr, ndc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nnr, nnc = cur_r + ndr, cur_c + ndc
                if (nnr, nnc) == (prev_r, prev_c):
                    continue
                if (nnr, nnc) == (start_r, start_c):
                    continue
                if 0 <= nnr < h and 0 <= nnc < w and skeleton[nnr, nnc] > 0:
                    prev_r, prev_c = cur_r, cur_c
                    cur_r, cur_c = nnr, nnc
                    advanced = True
                    break

            if not advanced:
                return

    for start_idx, (r, c) in enumerate(all_nodes):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc] > 0:
                walk_and_record(start_idx, r, c, dr, dc)

    return G


def build_physical_graph(
    grass_mask: np.ndarray, gsd: float = 0.006388
) -> Tuple[nx.Graph, np.ndarray]:
    """
    End-to-end build G_actual from binary grass mask.

    Pipeline: skeletonize -> remove spurs -> detect nodes -> trace edges

    Args:
        grass_mask: (H, W) uint8, values 0 or 255
        gsd: ground sample distance (m/px)

    Returns:
        G_actual: nx.Graph with meter lengths on edges
        skeleton: (H, W) uint8, values 0 or 1
    """
    skel = skeletonize_mask(grass_mask)
    skel = remove_spurs(skel)
    cross_pts, end_pts, branch_pts = detect_nodes(skel)
    G = trace_edges(skel, cross_pts, end_pts, branch_pts)

    for u, v, data in G.edges(data=True):
        data["length_m"] = data["length_px"] * gsd

    return G, skel
