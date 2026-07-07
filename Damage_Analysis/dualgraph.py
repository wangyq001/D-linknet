"""Stage 4: Theoretical graph construction and corridor-based edge matching."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict
try:
    from .models import GridParams, EdgeMatch
    from .skeleton import skeletonize_mask
except ImportError:
    from models import GridParams, EdgeMatch
    from skeleton import skeletonize_mask


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def build_pixel_set(skeleton: np.ndarray) -> Set[Tuple[int, int]]:
    """Build a set of (r, c) coordinates from skeleton image."""
    return set(zip(*np.where(skeleton > 0)))


def generate_theoretical_edge_pixels(
    r1: float, c1: float, r2: float, c2: float
) -> List[Tuple[int, int]]:
    """
    Generate pixel coordinates along the straight line between two points.

    Uses Bresenham's algorithm adapted for sub-pixel coordinates.

    Args:
        r1, c1, r2, c2: start and end coordinates (can be float)

    Returns:
        pixels: list of (r, c) as integers
    """
    r1i, c1i = int(round(r1)), int(round(c1))
    r2i, c2i = int(round(r2)), int(round(c2))

    dr = abs(r2i - r1i)
    dc = abs(c2i - c1i)

    pixels = []
    if dr >= dc:
        step = 1 if r2i > r1i else -1
        err = dr // 2
        r, c = r1i, c1i
        for _ in range(dr + 1):
            pixels.append((r, c))
            err -= dc
            if err < 0:
                c += 1 if c2i > c1i else -1
                err += dr
            r += step
    else:
        step = 1 if c2i > c1i else -1
        err = dc // 2
        r, c = r1i, c1i
        for _ in range(dc + 1):
            pixels.append((r, c))
            err -= dr
            if err < 0:
                r += 1 if r2i > r1i else -1
                err += dc
            c += step

    return pixels


def find_actual_edge_in_corridor(
    theoretical_pixels: List[Tuple[int, int]],
    skeleton_pixel_set: Set[Tuple[int, int]],
    corridor_px: int,
) -> Tuple[float, float]:
    """
    Find how much of a theoretical edge is covered by actual skeleton.

    The "corridor" is the band of width corridor_px on each side of the
    theoretical edge. We count skeleton pixels inside this corridor.

    Args:
        theoretical_pixels: pixel coordinates of the theoretical edge
        skeleton_pixel_set: set of (r, c) skeleton pixels
        corridor_px: half-width of the corridor (pixels)

    Returns:
        coverage_ratio: fraction [0,1] of theoretical edge covered
        actual_length_px: matched skeleton pixel count
    """
    if not theoretical_pixels:
        return 0.0, 0.0

    corridor: Set[Tuple[int, int]] = set()
    for (r, c) in theoretical_pixels:
        for dr in range(-corridor_px, corridor_px + 1):
            nr = r + dr
            for dc in range(-corridor_px, corridor_px + 1):
                nc = c + dc
                corridor.add((nr, nc))

    covered = corridor & skeleton_pixel_set
    # coverage_ratio = skeleton pixels along the theoretical line / total theoretical pixels
    # Each theoretical pixel contributes 1 unit to the denominator;
    # the (2*corridor_px+1)^2 term in the denominator was a bug.
    coverage_ratio = len(covered) / max(len(theoretical_pixels), 1)

    # actual_length_px: number of skeleton pixels matched along this edge
    actual_length_px = len(covered)
    return float(coverage_ratio), float(actual_length_px)


# ----------------------------------------------------------------------
# Theoretical graph
# ----------------------------------------------------------------------

def build_theoretical_graph(
    grid_params: GridParams,
    bbox_px: Tuple[Tuple[int, int], Tuple[int, int]],
    margin: int = 1,
) -> nx.Graph:
    """
    Build a regular grid graph in the rotated coordinate system.

    Nodes are placed at positions (phase + i*step, phase + j*step).
    Edges connect adjacent nodes horizontally and vertically.

    Args:
        grid_params: GridParams from estimate_grid_params
        bbox_px: ((r_min, c_min), (r_max, c_max)) in rotated space
        margin: extra rows/cols beyond bbox

    Returns:
        G: nx.Graph
            node integer ID → attributes:
              'grid_id': (row_idx, col_idx)
              'pos_px': (r', c') in rotated space
            edge (u, v) → attributes:
              'length_m': step length (meters)
              'direction': 'H' or 'V'
    """
    (r_min, c_min), (r_max, c_max) = bbox_px
    step_x = grid_params.step_x_px
    step_y = grid_params.step_y_px
    phase_x = grid_params.phase_x_px
    phase_y = grid_params.phase_y_px

    c_start = (c_min // step_x) * step_x + phase_x - margin * step_x
    c_end   = c_max + margin * step_x
    r_start = (r_min // step_y) * step_y + phase_y - margin * step_y
    r_end   = r_max + margin * step_y

    node_grid_ids: Dict[Tuple[int, int], int] = {}
    G = nx.Graph()
    node_counter = 0

    c_vals = []
    r_cur = r_start
    while r_cur <= r_end:
        c_vals.append(r_cur)
        r_cur += step_y

    r_vals = []
    c_cur = c_start
    while c_cur <= c_end:
        r_vals.append(c_cur)
        c_cur += step_x

    row_idx = 0
    for r in c_vals:
        for col_idx, c in enumerate(r_vals):
            grid_id = (row_idx, col_idx)
            G.add_node(node_counter, grid_id=grid_id, pos_px=(r, c))
            node_grid_ids[grid_id] = node_counter
            node_counter += 1
        row_idx += 1

    step_m = step_x * grid_params.gsd_m
    for (row_idx, col_idx), nid in node_grid_ids.items():
        right_id = node_grid_ids.get((row_idx, col_idx + 1))
        if right_id is not None:
            G.add_edge(nid, right_id, length_m=step_m, direction="H")

        down_id = node_grid_ids.get((row_idx + 1, col_idx))
        if down_id is not None:
            G.add_edge(nid, down_id, length_m=step_m, direction="V")

    return G


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------

def build_and_match(
    G_actual: nx.Graph,
    grid_params: GridParams,
    bbox_px: Tuple[Tuple[int, int], Tuple[int, int]],
    grass_mask: np.ndarray,
    corridor_ratio: float = 0.25,
    gsd: float = 0.006388,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], EdgeMatch], np.ndarray]:
    """
    End-to-end: build G_theoretical and match against physical skeleton.

    Pipeline:
      1. Build G_theoretical from grid_params
      2. Build skeleton + pixel lookup set
      3. For each theoretical edge, run corridor matching → EdgeMatch

    Args:
        G_actual: nx.Graph from build_physical_graph (used for parameter validation)
        grid_params: GridParams from estimate_grid_params
        bbox_px: image bounds ((0,0), (H,W))
        grass_mask: (H, W) uint8 binary mask
        corridor_ratio: corridor half-width as fraction of avg step
        gsd: ground sample distance (m/px)

    Returns:
        G_theoretical: nx.Graph
        G_match: dict {edge_id_tuple: EdgeMatch}
        skeleton: (H, W) uint8
    """
    G_theoretical = build_theoretical_graph(grid_params, bbox_px)

    skel = skeletonize_mask(grass_mask)
    skel_pixels = build_pixel_set(skel)

    step_avg = (grid_params.step_x_px + grid_params.step_y_px) / 2
    corridor_px = max(int(step_avg * corridor_ratio), 10)

    G_match: Dict[Tuple[int, int], EdgeMatch] = {}

    for u, v, tdata in G_theoretical.edges(data=True):
        r1, c1 = G_theoretical.nodes[u]["pos_px"]
        r2, c2 = G_theoretical.nodes[v]["pos_px"]
        theo_pixels = generate_theoretical_edge_pixels(r1, c1, r2, c2)

        coverage_ratio, actual_len_px = find_actual_edge_in_corridor(
            theo_pixels, skel_pixels, corridor_px
        )

        theo_len_px = len(theo_pixels)
        theo_len_m = theo_len_px * gsd
        actual_len_m = actual_len_px * gsd
        missing_len_m = max(theo_len_m - actual_len_m, 0.0)

        if coverage_ratio >= 0.5:
            status = "intact"
        elif coverage_ratio >= 0.2:
            status = "minor"
        elif coverage_ratio >= 0.05:
            status = "severe"
        else:
            status = "missing"

        G_match[(u, v)] = EdgeMatch(
            edge_id=(u, v),
            coverage_ratio=coverage_ratio,
            actual_length_m=actual_len_m,
            theoretical_length_m=theo_len_m,
            missing_length_m=missing_len_m,
            status=status,
            veg_coverage=0.0,
            reason="unknown",
            direction=tdata.get("direction", "H"),
        )

    return G_theoretical, G_match, skel
