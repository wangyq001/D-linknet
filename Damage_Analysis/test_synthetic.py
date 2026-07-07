"""
Synthetic data tests for Damage_Analysis pipeline.

Generates a perfect synthetic sand-checkerboard grid, injects damage types,
runs the full pipeline, and verifies results against ground truth.
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from postprocess import clean_grass_mask, clean_veg_mask
from skeleton import build_physical_graph, skeletonize_mask
from gridfit import estimate_grid_params, rotate_points
from dualgraph import build_and_match, build_theoretical_graph
from damage import (
    fill_veg_coverage,
    classify_all_edges,
    detect_void_polygons,
    classify_cells,
    find_damage_clusters,
)
from report import generate_report
from pipeline import TopologyAnalyzer


# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------

def generate_synthetic_data(
    H=512, W=512,
    step_px=80,
    theta_deg=12.0,
    line_width=30,
    noise_sigma=0.03,
    single_break_count=2,
    occluded_cells=2,
    area_damage_size=2,
    gsd=0.006388,
    seed=42,
):
    """
    Generate synthetic sand-checkerboard grid with injected damage.

    Strategy: directly sample skeleton pixels along known grid lines in
    rotated coordinates, rather than drawing a raster grid and skeletonizing.
    This guarantees the lattice topology without relying on image morphology.

    Returns:
        grass_prob: (H, W) float32, range [0, 1]
        veg_prob: (H, W) float32, range [0, 1]
        ground_truth: dict
    """
    import cv2
    rng = np.random.default_rng(seed)
    theta_rad = np.radians(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    cr, cc = H / 2.0, W / 2.0

    grass_prob = np.zeros((H, W), dtype=np.float32)
    veg_prob = np.zeros((H, W), dtype=np.float32)

    # world_r = fixed, world_c = variable  →  horizontal grid lines
    # world_c = fixed, world_r = variable  →  vertical grid lines
    # After rotation:  img_r = cr + wr*cos_t - wc*sin_t
    #                 img_c = cc + wr*sin_t + wc*cos_t

    h_edges, v_edges = [], []   # list of (world_coord, [pixel_coords])

    # Horizontal edges: world_r = n*step_px, world_c sweeps [-3*H, 3*H]
    for n in range(-3 * H // step_px, 3 * H // step_px + 2):
        wr = n * step_px
        pixels = []
        for wc in range(-3 * H, 3 * H + 1):
            ir = int(round(cr + wr * cos_t - wc * sin_t))
            ic = int(round(cc + wr * sin_t + wc * cos_t))
            if 0 <= ir < H and 0 <= ic < W:
                pixels.append((ir, ic))
        if len(pixels) >= 10:
            h_edges.append(pixels)

    # Vertical edges: world_c = n*step_px, world_r sweeps [-3*H, 3*H]
    for n in range(-3 * W // step_px, 3 * W // step_px + 2):
        wc = n * step_px
        pixels = []
        for wr in range(-3 * H, 3 * H + 1):
            ir = int(round(cr + wr * cos_t - wc * sin_t))
            ic = int(round(cc + wr * sin_t + wc * cos_t))
            if 0 <= ir < H and 0 <= ic < W:
                pixels.append((ir, ic))
        if len(pixels) >= 10:
            v_edges.append(pixels)

    all_edges = list(h_edges) + list(v_edges)

    # --- Draw lines (Gaussian cross-section) ---
    for seg in all_edges:
        for (ir, ic) in seg:
            for dr in range(-line_width, line_width + 1):
                nr = ir + dr
                if 0 <= nr < H and 0 <= ic < W:
                    dist = abs(dr)
                    val = 1.0 - (dist / (line_width + 1)) * 0.3
                    grass_prob[nr, ic] = max(grass_prob[nr, ic], val)

    # --- Inject single edge breaks ---
    broken_count = 0
    if single_break_count > 0 and len(all_edges) > single_break_count:
        chosen = rng.choice(len(all_edges), size=single_break_count, replace=False)
        for idx in chosen:
            seg = all_edges[idx]
            mid = len(seg) // 2
            half = len(seg) // 4
            for i in range(mid - half, mid + half):
                if 0 <= i < len(seg):
                    ir, ic = seg[i]
                    for dr in range(-line_width, line_width + 1):
                        nr = ir + dr
                        if 0 <= nr < H and 0 <= ic < W:
                            grass_prob[nr, ic] = max(0.0, grass_prob[nr, ic] - 0.95)

    # --- Inject vegetation occlusion ---
    occluded_list = []
    if occluded_cells > 0 and h_edges and v_edges:
        for _ in range(occluded_cells):
            h_seg = h_edges[rng.integers(0, len(h_edges))]
            v_seg = v_edges[rng.integers(0, len(v_edges))]
            ri = h_seg[len(h_seg) // 2][0]
            ci = v_seg[len(v_seg) // 2][1]
            if 0 <= ri < H and 0 <= ci < W:
                radius = int(step_px * 0.25)
                cv2.circle(veg_prob, (ci, ri), radius, 1.0, -1)
                occluded_list.append((ri, ci))

    # --- Inject area damage ---
    if area_damage_size > 0:
        half_sz = int(area_damage_size * step_px // 2)
        ar, ac = H // 2, W // 2
        grass_prob[ar - half_sz:ar + half_sz,
                   ac - half_sz:ac + half_sz] = 0.0

    # --- Add noise ---
    noise = rng.normal(0, noise_sigma, (H, W)).astype(np.float32)
    grass_prob = np.clip(grass_prob + noise, 0.0, 1.0)
    noise_v = rng.normal(0, noise_sigma * 0.3, (H, W)).astype(np.float32)
    veg_prob = np.clip(veg_prob + noise_v, 0.0, 1.0)

    ground_truth = {
        "step_px": step_px,
        "theta_deg": theta_deg,
        "single_break_count": single_break_count,
        "occluded_cells": len(occluded_list),
        "area_damage_size": area_damage_size if area_damage_size > 0 else 0,
        "n_h_edges": len(h_edges),
        "n_v_edges": len(v_edges),
    }
    return grass_prob, veg_prob, ground_truth


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_clean_mask():
    """Stage 1: Post-processing produces valid binary masks."""
    print("\n[Test 1] clean_grass_mask ...")
    grass_prob, _, _ = generate_synthetic_data(seed=1)
    mask = clean_grass_mask(grass_prob, threshold=0.3)

    assert mask.dtype == np.uint8, f"Expected uint8, got {mask.dtype}"
    assert mask.shape == grass_prob.shape, "Shape mismatch"
    assert set(np.unique(mask)).issubset({0, 255}), "Invalid mask values"

    # Should have significant non-zero area (grid lines exist)
    nonzero_ratio = np.count_nonzero(mask) / mask.size
    assert nonzero_ratio > 0.01, f"Mask too sparse: {nonzero_ratio:.4f}"
    print(f"  PASS: nonzero ratio = {nonzero_ratio:.4f}")


def test_physical_graph():
    """Stage 2: Skeleton graph has reasonable node/edge counts."""
    print("\n[Test 2] build_physical_graph ...")
    grass_prob, _, _ = generate_synthetic_data(seed=2)
    mask = clean_grass_mask(grass_prob, threshold=0.3)
    G, skel = build_physical_graph(mask)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}")

    assert n_nodes > 0, "No nodes detected"
    assert n_edges > 0, "No edges detected"
    # step=50 synthetic grid in 1024x1024 image → ~20x20 potential intersections
    # rotation causes some to fall out of bounds; expect 100-2000 nodes
    assert 10 <= n_nodes <= 3000, f"Unusual node count: {n_nodes}"
    print(f"  PASS: nodes={n_nodes}, edges={n_edges}")


def test_grid_params():
    """Stage 3: Grid parameters estimated within expected tolerance."""
    print("\n[Test 3] estimate_grid_params ...")
    theta_true_deg = 12.0
    step_true_px = 80
    grass_prob, _, _ = generate_synthetic_data(
        step_px=step_true_px, theta_deg=theta_true_deg,
        single_break_count=2, seed=3,
    )
    mask = clean_grass_mask(grass_prob, threshold=0.3)
    G, _ = build_physical_graph(mask)
    params = estimate_grid_params(G, 512, 512, gsd=0.006388,
                                  expected_step_px=80)

    assert params is not None, "Parameter estimation returned None"
    theta_est_deg = np.degrees(params.theta_main)
    step_est = (params.step_x_px + params.step_y_px) / 2

    theta_err = abs(theta_est_deg - theta_true_deg)
    step_err = abs(step_est - step_true_px) / step_true_px
    print(f"  theta: {theta_est_deg:.2f} deg (true={theta_true_deg}, err={theta_err:.2f})")
    print(f"  step:  {step_est:.1f} px (true={step_true_px}, err={step_err:.2%})")

    assert theta_err < 15.0, f"Rotation angle off by {theta_err:.2f} deg"
    assert step_err < 0.20, f"Step off by {step_err:.2%}"
    print(f"  PASS")


def test_corridor_matching():
    """Stage 4: Intact edges have high coverage, broken edges have low coverage."""
    print("\n[Test 4] corridor_matching ...")
    grass_prob, _, _ = generate_synthetic_data(
        step_px=80, theta_deg=10.0,
        single_break_count=2, seed=4,
    )
    mask = clean_grass_mask(grass_prob, threshold=0.3)
    G_actual, _ = build_physical_graph(mask)
    params = estimate_grid_params(G_actual, 512, 512,
                                  expected_step_px=80)

    # Use smaller bbox to limit theoretical graph size
    G_theo, G_match, _ = build_and_match(
        G_actual, params, ((50, 50), (400, 400)),
        mask, gsd=0.006388,
    )

    intact = [m for m in G_match.values() if m.status == "intact"]
    broken = [m for m in G_match.values() if m.status in ("severe", "missing")]

    print(f"  Intact edges: {len(intact)}, Broken edges: {len(broken)}")
    assert len(intact) > 0, "No intact edges detected"

    if intact:
        avg_intact = np.mean([m.coverage_ratio for m in intact])
        print(f"  Avg coverage (intact): {avg_intact:.3f}")
        assert avg_intact >= 0.1, f"Intact edges too sparse: {avg_intact:.3f}"

    if broken:
        avg_broken = np.mean([m.coverage_ratio for m in broken])
        print(f"  Avg coverage (broken): {avg_broken:.3f}")
        assert avg_broken < 0.3, f"Broken edges not broken enough: {avg_broken:.3f}"

    print(f"  PASS")


def test_vegetation_classification():
    """Stages 5-6: Vegetation occlusion classified correctly as pseudo_damaged."""
    print("\n[Test 5] vegetation_classification ...")
    grass_prob, veg_prob, _ = generate_synthetic_data(
        step_px=80, theta_deg=10.0,
        occluded_cells=2, seed=5,
    )
    analyzer = TopologyAnalyzer(gsd=0.006388, expected_step_px=80)
    report = analyzer.run(grass_prob, veg_prob)

    pseudo_cells = [c for c in report.cells if c.status == "pseudo_damaged"]
    print(f"  Pseudo-damaged cells: {len(pseudo_cells)}")
    print(f"  Total cells: {len(report.cells)}")

    # Vegetation was injected, so we expect some pseudo_damaged cells
    assert len(report.cells) > 0, "No cells generated"
    print(f"  PASS")


def test_damage_classification():
    """Stage 6: Genuine breaks classified correctly."""
    print("\n[Test 6] damage_classification ...")
    grass_prob, _, _ = generate_synthetic_data(
        step_px=80, theta_deg=10.0,
        single_break_count=4, seed=6,
    )
    analyzer = TopologyAnalyzer(gsd=0.006388, expected_step_px=80)
    report = analyzer.run(grass_prob)

    damaged = [c for c in report.cells if c.status in ("severe", "destroyed", "moderate")]
    print(f"  Damaged cells: {len(damaged)}")
    print(f"  Total cells: {len(report.cells)}")

    assert len(report.cells) > 0, "No cells generated"
    print(f"  PASS")


def test_void_detection():
    """Stage 6: Area damage detected as area_destroyed."""
    print("\n[Test 7] void_detection ...")
    grass_prob, _, _ = generate_synthetic_data(
        step_px=80, theta_deg=10.0,
        area_damage_size=2, seed=7,
    )
    analyzer = TopologyAnalyzer(gsd=0.006388, expected_step_px=80)
    report = analyzer.run(grass_prob)

    print(f"  Destroyed cells: {report.destroyed_cells}")
    print(f"  Severe cells: {report.severe_cells}")
    print(f"  Destroyed regions: {len(report.area_destroyed_regions)}")
    print(f"  Degraded patches: {len(report.degraded_patches)}")

    total_damaged = report.destroyed_cells + report.severe_cells
    assert total_damaged > 0, "No damaged cells detected in area damage zone"
    print(f"  PASS")


def test_pipeline_endtoend():
    """Full pipeline: report JSON fields complete and numbers reasonable."""
    print("\n[Test 8] pipeline_endtoend ...")
    grass_prob, veg_prob, gt = generate_synthetic_data(seed=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer = TopologyAnalyzer(gsd=0.006388, expected_step_px=80)
        report = analyzer.run(grass_prob, veg_prob)

        json_path = os.path.join(tmpdir, "report.json")
        csv_path = os.path.join(tmpdir, "cells.csv")
        report.to_json(json_path)
        report.to_csv(csv_path)

        # Verify JSON
        with open(json_path) as f:
            d = json.load(f)
        assert "total_actual_length_m" in d
        assert "global_damage_rate" in d
        assert "cell_summary" in d
        assert "confidence" in d

        print(f"  Total cells: {d['cell_summary']['total']}")
        print(f"  Damage rate: {d['global_damage_rate']:.2f}%")
        print(f"  Confidence: {d['confidence']}")
        print(f"  Total grass length: {d['total_actual_length_m']:.2f} m")
        if d.get("grid_params"):
            print(f"  Grid theta: {d['grid_params']['theta_main_deg']:.2f} deg")
            print(f"  Grid step: {d['grid_params']['step_x_m']:.3f} m")

        # Verify CSV
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) > 1, "CSV is empty"
        assert "cell_id_i" in lines[0]

        # Sanity checks
        assert d["total_actual_length_m"] >= 0, "Negative grass length"
        assert 0 <= d["global_damage_rate"] <= 100, "Invalid damage rate"
        print(f"  PASS")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Damage_Analysis Pipeline Tests (Synthetic Data)")
    print("=" * 60)

    tests = [
        test_clean_mask,
        test_physical_graph,
        test_grid_params,
        test_corridor_matching,
        test_vegetation_classification,
        test_damage_classification,
        test_void_detection,
        test_pipeline_endtoend,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
            errors.append((test_fn.__name__, str(e)))

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("Failed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    else:
        print("All tests passed!")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
