"""
Full-ROI test runner: tiles the ROI into overlapping chunks, runs the
analyzer on each, deduplicates cells by grid_id, then renders the two
overlay PNGs on the *entire* ROI image at native resolution.
"""
import sys
import os
import json
import time
import rasterio
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

# Ensure module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocess import clean_grass_mask
from pipeline import TopologyAnalyzer
from damage import find_damage_clusters
from visualize import cells_overlay, area_damage_overlay, save_png


# ROI bounds (meters) parsed from 测试roi.shp
ROI_X_MIN, ROI_X_MAX = 308953.77, 308973.81
ROI_Y_MIN, ROI_Y_MAX = 4230747.62, 4230766.85

# From raster transform: origin x=308949.52 (col=0), y=4230769.45 (row=0), dx=dy=0.01m
ORIGIN_X = 308949.52
ORIGIN_Y = 4230769.45
GSD = 0.01  # m/px
EXPECTED_STEP_PX = 100  # 1m grid


def roi_window():
    """Pixel bbox (r0, c0, h, w) of ROI in full image."""
    c0 = int((ROI_X_MIN - ORIGIN_X) / GSD)
    c1 = int((ROI_X_MAX - ORIGIN_X) / GSD)
    r0 = int((ORIGIN_Y - ROI_Y_MAX) / GSD)
    r1 = int((ORIGIN_Y - ROI_Y_MIN) / GSD)
    return r0, c0, r1 - r0, c1 - c0


def make_tiles(r0, c0, full_h, full_w, tile, overlap):
    """(r0, c0, h, w) tile coords with `overlap` pixel padding."""
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


def rebase_cell(cell, dr, dc, grid_origin):
    """Translate cell's pixel bbox by (dr, dc). Update grid_id as needed."""
    ((r0, c0), (r1, c1)) = cell.bbox_px
    new_bbox = ((r0 + dr, c0 + dc), (r1 + dr, c1 + dc))
    return type(cell)(
        cell_id=cell.cell_id,
        bbox_px=new_bbox,
        damage_rate=cell.damage_rate,
        status=cell.status,
        edge_statuses=cell.edge_statuses,
        genuine_break_count=cell.genuine_break_count,
        occluded_count=cell.occluded_count,
        area_m2=cell.area_m2,
        veg_covered_sides=cell.veg_covered_sides,
    )


def run_tile(analyzer, grass_path, rgb_path, r0, c0, h, w):
    """Run analyzer on a single tile. Returns list of (rebased) cells + clusters."""
    with rasterio.open(grass_path) as src:
        win = rasterio.windows.Window(c0, r0, w, h)
        grass = src.read(1, window=win).astype(np.float32) / 255.0
    with rasterio.open(rgb_path) as src:
        win = rasterio.windows.Window(c0, r0, w, h)
        rgb_full = src.read([1, 2, 3], window=win)

    rgb = np.transpose(rgb_full, (1, 2, 0)).astype(np.uint8)

    t0 = time.time()
    report = analyzer.run(grass)
    elapsed = time.time() - t0

    # Rebase cell pixel bboxes by (r0, c0)
    cells_rebased = [rebase_cell(c, r0, c0, None) for c in report.cells]
    return cells_rebased, rgb, elapsed, report


def main():
    out_dir = "图论测试/可视化输出"
    os.makedirs(out_dir, exist_ok=True)
    grass_path = "图论测试/测试草线掩码.tif"
    rgb_path = "图论测试/测试遥感图.tif"

    r0, c0, full_h, full_w = roi_window()
    print(f"ROI bbox: rows [{r0},{r0+full_h}] cols [{c0},{c0+full_w}]  = {full_h}x{full_w} px  =  {full_h*GSD:.2f}x{full_w*GSD:.2f} m")

    TILE = 1000
    OVERLAP = 100
    tiles = make_tiles(r0, c0, full_h, full_w, TILE, OVERLAP)
    print(f"Tiles: {len(tiles)}, each {TILE}x{TILE}, overlap {OVERLAP}px")

    analyzer = TopologyAnalyzer(gsd=GSD, expected_step_px=EXPECTED_STEP_PX)

    all_cells: List = []
    tile_rgbs: Dict[Tuple[int, int], np.ndarray] = {}
    tile_elapsed = []

    for i, (tr, tc, th, tw) in enumerate(tiles):
        cells, rgb, elapsed, report = run_tile(analyzer, grass_path, rgb_path, tr, tc, th, tw)
        tile_rgbs[(tr, tc, th, tw)] = rgb
        all_cells.extend(cells)
        tile_elapsed.append(elapsed)
        print(f"  tile {i+1}/{len(tiles)} at r={tr},c={tc}  size={th}x{tw}  "
              f"cells={len(cells)}  done in {elapsed:.1f}s")

    print(f"\nTotal raw cells (with overlap duplicates): {len(all_cells)}")

    # Deduplicate cells by approximate physical position: round each cell's
    # bbox centre to nearest grid step (= 1m = 100 px ≈ 1m grid cell), and
    # keep the first occurrence.
    def approx_position(cell):
        ((r0_, c0_), (r1_, c1_)) = cell.bbox_px
        cr = (r0_ + r1_) / 2 / EXPECTED_STEP_PX
        cc = (c0_ + c1_) / 2 / EXPECTED_STEP_PX
        return (int(round(cr)), int(round(cc)))

    seen_pos: Dict[Tuple[int, int], object] = {}
    for c in all_cells:
        pos = approx_position(c)
        if pos not in seen_pos:
            seen_pos[pos] = c

    unique_cells = list(seen_pos.values())
    print(f"Unique cells after position dedupe: {len(unique_cells)}")

    counts = Counter(c.status for c in unique_cells)
    print("\nFinal cell status breakdown:")
    for status, n in counts.most_common():
        print(f"  {status:<14s}: {n:3d}  ({n/len(unique_cells)*100:.1f}%)")

    # Find clusters across the WHOLE ROI
    cell_area_m2 = (EXPECTED_STEP_PX * GSD) ** 2
    clusters = find_damage_clusters(
        unique_cells,
        cell_area_m2=cell_area_m2,
    )
    print(f"\nTotal clusters: {len(clusters)}")
    for c in clusters[:10]:
        print(f"  type={c['type']}, cells={c['cell_count']}, "
              f"area={c['area_m2']:.2f}m², cascade_risk={c['cascade_risk']:.2f}, "
              f"polygon_pts={len(c.get('polygon') or [])}")

    # Build ground-truth comparison
    intact_or_pseudo = sum(n for s, n in counts.items() if s in ("intact", "pseudo_damaged"))
    damaged = sum(n for n in counts.values() if n) - intact_or_pseudo
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPARISON")
    print("=" * 60)
    print("Expected (manual count):")
    print("  Total cells:         ~160")
    print("  Damaged cells:       ~133  (83%)")
    print(f"Got:")
    print(f"  Total cells:         {len(unique_cells)}")
    print(f"  Damaged cells:       {damaged}  ({damaged/len(unique_cells)*100:.1f}%)")
    print(f"  Intact:              {sum(n for s,n in counts.items() if s=='intact')}")

    # --- Stich full-ROI RGB (no overlap; just compose by row range) ---
    print("\nAssembling full-ROI RGB ...")
    full_rgb = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    for (tr, tc, th, tw), rgb in tile_rgbs.items():
        rr = tr - r0
        cc = tc - c0
        full_rgb[rr:rr+th, cc:cc+tw] = rgb
    print(f"Full RGB: {full_rgb.shape}")

    grass_mask_full = None
    with rasterio.open(grass_path) as src:
        win = rasterio.windows.Window(c0, r0, full_w, full_h)
        grass_full = src.read(1, window=win).astype(np.float32) / 255.0
        grass_mask_full = clean_grass_mask(grass_full, threshold=0.3)

    print("\nRendering cells_overlay ...")
    cells_arr = cells_overlay(
        rgb_image=full_rgb,
        grass_mask=grass_mask_full,
        cells=unique_cells,
        title=f"Cells overlay (full ROI): n={len(unique_cells)}  "
              f"theta={np.degrees(analyzer.run.__defaults__[0]) if False else '1.00'}d  step=100px",
    )
    cells_png = os.path.join(out_dir, "cells_overlay.png")
    save_png(cells_arr, cells_png)
    print(f"  → {cells_png}  ({os.path.getsize(cells_png)/1024:.1f} KB)")

    print("Rendering area_damage_overlay ...")
    area_clusters = [c for c in clusters if c.get("type") in ("area_destroyed", "degraded_patch")]
    area_arr = area_damage_overlay(
        rgb_image=full_rgb,
        clusters=area_clusters,
        title=f"Area damage overlay: {len(area_clusters)} polygon(s)",
    )
    area_png = os.path.join(out_dir, "area_damage_overlay.png")
    save_png(area_arr, area_png)
    print(f"  → {area_png}  ({os.path.getsize(area_png)/1024:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
