"""
Real data test script: run damage analysis on the actual test region
(`Damage_Analysis/图论测试/`).

Expected output: comparison against manual ground truth.
"""
import sys
import os
import json
import rasterio
import numpy as np
from collections import Counter

# Ensure module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocess import clean_grass_mask, clean_veg_mask
from pipeline import TopologyAnalyzer


# ROI bounds (meters) parsed from 测试roi.shp
# Polygon corners from the shp file (UTM 48N):
ROI_X_MIN, ROI_X_MAX = 308953.77, 308973.81
ROI_Y_MIN, ROI_Y_MAX = 4230747.62, 4230766.85

# From raster transform: origin x=308949.52 (col=0), y=4230769.45 (row=0), dx=dy=0.01m
ORIGIN_X = 308949.52
ORIGIN_Y = 4230769.45
GSD = 0.01  # m/px
EXPECTED_STEP_PX = 100  # 1m grid


def roi_to_window():
    """Return (r0, c0, h, w) of ROI in full image coords."""
    c_min = int((ROI_X_MIN - ORIGIN_X) / GSD)
    c_max = int((ROI_X_MAX - ORIGIN_X) / GSD)
    r_min = int((ORIGIN_Y - ROI_Y_MAX) / GSD)
    r_max = int((ORIGIN_Y - ROI_Y_MIN) / GSD)
    return r_min, c_min, r_max - r_min, c_max - c_min


def slice_into_tiles(r0, c0, full_h, full_w, tile=1024, overlap=100):
    """Yield (r0, c0, h, w) tile coords within ROI, with overlap."""
    tiles = []
    r = r0
    while r < r0 + full_h:
        h = min(tile, r0 + full_h - r)
        c = c0
        while c < c0 + full_w:
            w = min(tile, c0 + full_w - c)
            tiles.append((r, c, h, w))
            c += tile - overlap
        r += tile - overlap
    return tiles


def stitch(tile_results):
    """Deduplicate cell_id across tiles. Returns aggregated counts."""
    seen = {}
    for res in tile_results:
        for cid, status in res["cells"]:
            seen[cid] = status
    counts = Counter(seen.values())
    return counts, len(seen)


def run_tile(grass_path, veg_path, r0, c0, h, w):
    """Run analyzer on one tile; returns dict of metrics + intermediates."""
    with rasterio.open(grass_path) as src:
        win = rasterio.windows.Window(c0, r0, w, h)
        grass = src.read(1, window=win).astype(np.float32) / 255.0
    with rasterio.open(veg_path) as src:
        win = rasterio.windows.Window(c0, r0, w, h)
        veg = src.read(1, window=win).astype(np.float32) / 255.0

    analyzer = TopologyAnalyzer(gsd=GSD, expected_step_px=EXPECTED_STEP_PX)
    report = analyzer.run(grass, veg)
    grass_mask = clean_grass_mask(grass, threshold=0.3)

    cells = [(c.cell_id, c.status) for c in report.cells]
    return {
        "tile_shape": (h, w),
        "tile_total_m": report.total_actual_length_m,
        "tile_damage_rate_pct": report.global_damage_rate,
        "tile_confidence": report.confidence,
        "grid_theta_deg": np.degrees(report.grid_params.theta_main) if report.grid_params else None,
        "grid_step_px": (report.grid_params.step_x_px + report.grid_params.step_y_px) / 2
            if report.grid_params else None,
        "cells": cells,
        # For visualization: keep live objects
        "_report": report,
        "_grass_mask": grass_mask,
        "_grass": grass,
        "_veg": veg,
    }


def main():
    grass_path = "图论测试/测试草线掩码.tif"
    veg_path = "图论测试/测试植被掩码.tif"

    r0, c0, full_h, full_w = roi_to_window()
    print(f"ROI pixel bbox: rows [{r0}, {r0 + full_h}]  cols [{c0}, {c0 + full_w}]")
    print(f"ROI physical:    x [{ROI_X_MIN}, {ROI_X_MAX}]  y [{ROI_Y_MIN}, {ROI_Y_MAX}]")
    print(f"ROI size:        {full_h} x {full_w}  px  =  {full_h * GSD:.2f} x {full_w * GSD:.2f}  m")
    print()

    # For a single-tile run, pick ROI center.
    TILE = 768
    cr = r0 + (full_h - TILE) // 2
    cc = c0 + (full_w - TILE) // 2
    print(f"Running single tile ({TILE}x{TILE}) at r={cr}, c={cc} ...")

    import time
    t0 = time.time()
    res = run_tile(grass_path, veg_path, cr, cc, TILE, TILE)
    print(f"Tile run done in {time.time() - t0:.1f}s")

    print()
    print("=" * 60)
    print(f"ANALYSIS RESULTS (1 tile of {TILE}x{TILE} px = ~{TILE * GSD:.2f}m x {TILE * GSD:.2f}m)")
    print("=" * 60)
    print(f"Total grass length:     {res['tile_total_m']:.2f} m")
    print(f"Global damage rate:     {res['tile_damage_rate_pct']:.2f}%")
    print(f"Confidence:             {res['tile_confidence']}")
    if res["grid_theta_deg"] is not None:
        print(f"Grid rotation theta:    {res['grid_theta_deg']:.2f} deg")
        print(f"Grid step (avg):        {res['grid_step_px']:.1f} px ({res['grid_step_px'] * GSD:.3f} m)")
    print()

    counts, total = stitch([res])
    print(f"Total cells in tile:    {total}")
    print(f"Cell status breakdown:")
    for status, n in counts.most_common():
        pct = n / total * 100 if total else 0
        print(f"  {status:<18s}: {n:3d}  ({pct:.1f}%)")

    # Ground truth comparison
    print()
    print("=" * 60)
    print("GROUND TRUTH COMPARISON")
    print("=" * 60)
    print("Expected (manual count):")
    print("  Total cells:         ~160")
    print("  Damaged cells:       ~133  (83%)")
    print(f"Got:")
    print(f"  Total cells:         {total}")
    damaged = sum(n for s, n in counts.items() if s not in ("intact", "pseudo_damaged"))
    intact = sum(n for s, n in counts.items() if s in ("intact",))
    damaged_pct = damaged / total * 100 if total else 0
    print(f"  Damaged cells:       {damaged}  ({damaged_pct:.1f}%)")
    print(f"  Intact cells:        {intact}")

    # ------------------------------------------------------------------
    # Visualization overlays
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("GENERATING VISUALIZATION OVERLAYS")
    print("=" * 60)

    rgb_path = "图论测试/测试遥感图.tif"
    with rasterio.open(rgb_path) as src:
        win = rasterio.windows.Window(cc, cr, TILE, TILE)
        # 4-channel GeoTIFF → take first 3 as RGB
        rgb_full = src.read([1, 2, 3], window=win)
    rgb = np.transpose(rgb_full, (1, 2, 0)).astype(np.uint8)
    print(f"  RGB tile: {rgb.shape}, dtype={rgb.dtype}")

    analyzer = TopologyAnalyzer(gsd=GSD, expected_step_px=EXPECTED_STEP_PX)
    report_obj, cells_png, area_png = analyzer.run_with_visuals(
        rgb_image=rgb,
        grass_prob=res["_grass"],
        veg_prob=res["_veg"],
        output_dir="图论测试/可视化输出",
    )
    print(f"  Cells overlay:    {cells_png}")
    print(f"  Area overlay:     {area_png}")


if __name__ == "__main__":
    main()
