"""Stage 3: Grid parameter estimation (rotation, step, phase)."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from scipy.signal import find_peaks
try:
    from .models import GridParams
except ImportError:
    from models import GridParams

confidence_type = str


def rotate_points(
    points: List[Tuple[float, float]],
    angle_rad: float,
    center: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Rotate (r, c) coordinates around center by angle_rad (CCW positive).

    Args:
        points: list of (r, c) coordinates
        angle_rad: rotation angle in radians, CCW positive (OpenCV convention)
        center: (r_center, c_center)

    Returns:
        rotated: list of (r', c')
    """
    cr, cc = center
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    result = []
    for r, c in points:
        dr = r - cr
        dc = c - cc
        nr = dr * cos_a - dc * sin_a + cr
        nc = dr * sin_a + dc * cos_a + cc
        result.append((nr, nc))
    return result


def estimate_dominant_directions(
    G: nx.Graph,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate the two dominant edge directions using a direction histogram.

    Uses edge direction angles (0 to 180 degrees, since H and V are
    directionless) and finds the two peaks which should differ by ~90 deg.

    Args:
        G: nx.Graph with edges having 'direction' attr: 'H'|'V'|'diagonal',
           and nodes having 'pos_px' attr: (r, c)

    Returns:
        theta_h: dominant horizontal angle (radians)
        theta_v: dominant vertical angle (radians)
        H: direction entropy (None if not enough edges)
    """
    directions: List[float] = []
    weights: List[float] = []
    for u, v, data in G.edges(data=True):
        u_pos = G.nodes[u]["pos_px"]
        v_pos = G.nodes[v]["pos_px"]
        dy = v_pos[0] - u_pos[0]
        dx = v_pos[1] - u_pos[1]
        if dx == 0 and dy == 0:
            continue
        angle = np.arctan2(dy, dx) % np.pi
        length = data.get("length_px", 1)
        directions.append(angle)
        weights.append(length)

    if len(directions) < 5:
        return None, None, None

    weights_arr = np.array(weights, dtype=float)

    # Build weighted histogram
    bins = np.linspace(0, np.pi, 91)
    hist, edges = np.histogram(directions, bins=bins, weights=weights_arr)

    # Find two highest peaks with simple scanning
    max_val = hist.max()
    if max_val == 0:
        return None, None, None

    # Scan for local maxima above threshold
    peaks = []
    for i in range(len(hist)):
        if hist[i] >= max_val * 0.3:
            left = hist[i - 1] if i > 0 else 0
            right = hist[i + 1] if i < len(hist) - 1 else 0
            if hist[i] >= left and hist[i] >= right:
                peaks.append(i)
            elif i == 0 and hist[i] >= max_val * 0.5:
                peaks.append(i)
            elif i == len(hist) - 1 and hist[i] >= max_val * 0.5:
                peaks.append(i)

    if len(peaks) < 2:
        return None, None, None

    # Take top 2 by count
    peaks = sorted(peaks, key=lambda p: hist[p], reverse=True)[:2]
    peaks = sorted(peaks)

    bin_width = edges[1] - edges[0]
    peak1_angle = edges[peaks[0]] + bin_width / 2
    peak2_angle = edges[peaks[1]] + bin_width / 2

    if abs(peak2_angle - peak1_angle) < np.pi / 6:
        return peak1_angle, peak1_angle + np.pi / 2, None

    p = hist / (hist.sum() + 1e-10)
    H = -np.sum(p * np.log2(p + 1e-10))

    return peak1_angle, peak2_angle, H


def estimate_step(
    lengths_px: List[float], expected_px: float = 313
) -> Tuple[float, confidence_type]:
    """
    Estimate grid step from edge length histogram.

    Finds the main peak (most common edge length), which corresponds to
    the grid step. Short edges from breaks and long edges from double-counting
    are excluded by percentile filtering.

    If the histogram peak is not within +/- 50% of expected_px (or there are
    insufficient edges), returns expected_px as a robust fallback.

    Args:
        lengths_px: list of edge lengths (pixels)
        expected_px: expected step (pixels), used as fallback

    Returns:
        step_px: estimated step (pixels)
        confidence: 'high'|'medium'|'low'
    """
    if len(lengths_px) < 3:
        return float(expected_px), "low"

    arr = np.array(lengths_px, dtype=float)
    q1, q99 = np.percentile(arr, [1, 99])
    filtered = arr[(arr >= q1) & (arr <= q99)]

    if len(filtered) < 3:
        return float(expected_px), "low"

    # Find peak in the histogram
    hist, edges = np.histogram(filtered, bins=80)
    peaks, _ = find_peaks(hist, height=np.max(hist) * 0.3, distance=10)
    if len(peaks) == 0:
        return float(expected_px), "low"

    main_peak = peaks[np.argmax(hist[peaks])]
    step_px = float((edges[main_peak] + edges[main_peak + 1]) / 2)

    # If histogram peak is far from expected, fall back to expected
    rel_diff = abs(step_px - expected_px) / expected_px
    if rel_diff > 0.5:
        # Histogram is unreliable (e.g., due to fragmented trace)
        return float(expected_px), "medium"

    if rel_diff > 0.3:
        confidence = "low"
    elif rel_diff > 0.1:
        confidence = "medium"
    else:
        confidence = "high"

    return step_px, confidence


def estimate_phase(
    rotated_nodes: List[Tuple[float, float]], step_px: float
) -> Tuple[float, float]:
    """
    Estimate grid phase from rotated node coordinates.

    The phase is the offset of grid lines from the image origin. It is
    estimated as the mode of node coordinates modulo step_px.

    Args:
        rotated_nodes: list of (r', c') coordinates in aligned space
        step_px: grid step (pixels)

    Returns:
        phase_r_px, phase_c_px: row and column phase offsets (pixels)
    """
    rs = np.array([p[0] for p in rotated_nodes])
    cs = np.array([p[1] for p in rotated_nodes])

    def _mode_mod(values: np.ndarray, step: float) -> float:
        bins = np.arange(0, step + 1)
        hist, edges = np.histogram(values % step, bins=bins)
        return float(edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2) % step

    phase_r = _mode_mod(rs, step_px)
    phase_c = _mode_mod(cs, step_px)
    return float(phase_r), float(phase_c)


def estimate_grid_params(
    G: nx.Graph,
    H: int,
    W: int,
    gsd: float = 0.006388,
    expected_step_m: float = 2.0,
    expected_step_px: float = 313,
) -> Optional[GridParams]:
    """
    End-to-end grid parameter estimation.

    Pipeline:
      1. Direction histogram → dominant angles θ_h, θ_v
      2. Rotation angle = θ_h (align horizontal grid lines to 0°)
      3. Rotate all node coordinates by -θ_h around image center
      4. Classify edges as H or V in rotated space → estimate step_x/step_y
      5. Phase estimation via mod-step histogram mode

    Args:
        G: nx.Graph from build_physical_graph
        H, W: image dimensions (pixels)
        gsd: ground sample distance (m/px)
        expected_step_m: expected grid step (meters)
        expected_step_px: expected grid step (pixels) = expected_step_m / gsd

    Returns:
        GridParams or None if estimation fails
    """
    theta_h, theta_v, H_entropy = estimate_dominant_directions(G)
    if theta_h is None:
        return None

    theta_main = theta_h
    center = (H / 2, W / 2)

    all_pos = [G.nodes[n]["pos_px"] for n in G.nodes]
    rotated_nodes = rotate_points(all_pos, -theta_main, center)

    all_pos_rotated_map = {n: rotated_nodes[i] for i, n in enumerate(G.nodes)}

    h_lengths, v_lengths = [], []
    for u, v, data in G.edges(data=True):
        rp_u = all_pos_rotated_map[u]
        rp_v = all_pos_rotated_map[v]
        dr = abs(rp_v[0] - rp_u[0])
        dc = abs(rp_v[1] - rp_u[1])
        if dc >= dr * 2:
            h_lengths.append(dc)
        elif dr >= dc * 2:
            v_lengths.append(dr)

    step_x_px, cx = estimate_step(h_lengths, expected_step_px)
    step_y_px, cy = estimate_step(v_lengths, expected_step_px)

    step_avg = (step_x_px + step_y_px) / 2
    phase_y_px, phase_x_px = estimate_phase(rotated_nodes, step_avg)

    if H_entropy is not None and H_entropy > 2.5:
        confidence = "low"
    else:
        confidence = "high" if (cx == "high" and cy == "high") else "medium"

    return GridParams(
        theta_main=theta_main,
        step_x_px=step_x_px,
        step_y_px=step_y_px,
        phase_x_px=phase_x_px,
        phase_y_px=phase_y_px,
        gsd_m=gsd,
        direction_entropy=H_entropy if H_entropy is not None else 999.0,
        confidence=confidence,
    )
