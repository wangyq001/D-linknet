"""End-to-end CLI pipeline for sand checkerboard damage analysis."""

import argparse
import os
import numpy as np
from typing import Optional, Tuple, List
try:
    from .postprocess import clean_grass_mask, clean_veg_mask
    from .skeleton import build_physical_graph
    from .gridfit import estimate_grid_params
    from .dualgraph import build_and_match
    from .damage import (
        fill_veg_coverage,
        classify_all_edges,
        detect_void_polygons,
        classify_cells,
        find_damage_clusters,
    )
    from .report import generate_report
    from .visualize import (
        cells_overlay,
        area_damage_overlay,
        save_png,
    )
except ImportError:
    from postprocess import clean_grass_mask, clean_veg_mask
    from skeleton import build_physical_graph
    from gridfit import estimate_grid_params
    from dualgraph import build_and_match
    from damage import (
        fill_veg_coverage,
        classify_all_edges,
        detect_void_polygons,
        classify_cells,
        find_damage_clusters,
    )
    from report import generate_report
    from visualize import (
        cells_overlay,
        area_damage_overlay,
        save_png,
    )


class TopologyAnalyzer:
    """
    End-to-end sand checkerboard damage analyzer.

    Usage:
        analyzer = TopologyAnalyzer(gsd=0.006388, grid_size_m=2.0)
        report = analyzer.run(grass_prob, veg_prob)
        report.to_json("report.json")
    """

    def __init__(
        self,
        gsd: float = 0.006388,
        grid_size_m: float = 2.0,
        expected_step_px: Optional[float] = None,
    ):
        self.gsd = gsd
        self.grid_size_m = grid_size_m
        self.expected_step_px = (
            expected_step_px
            if expected_step_px is not None
            else grid_size_m / gsd
        )

    def run(
        self,
        grass_prob: np.ndarray,
        veg_prob: Optional[np.ndarray] = None,
    ):
        """
        Run full analysis pipeline.

        Args:
            grass_prob: (H, W) float32, range [0, 1]
            veg_prob: (H, W) float32, range [0, 1], optional

        Returns:
            AnalysisReport
        """
        grass_mask = clean_grass_mask(grass_prob)
        veg_mask = clean_veg_mask(veg_prob) if veg_prob is not None else None

        G_actual, skeleton = build_physical_graph(grass_mask, self.gsd)

        H, W = grass_prob.shape
        params = estimate_grid_params(
            G_actual, H, W,
            gsd=self.gsd,
            expected_step_m=self.grid_size_m,
            expected_step_px=self.expected_step_px,
        )

        if params is None:
            confidence = "low"
            G_theo = None
            G_match = {}
            void_polygons = []
            cells = []
        else:
            confidence = params.confidence
            # Limit bbox to avoid huge theoretical graphs.
            # Use step-based margin (in rotated space) but cap at image dims.
            step_avg = (params.step_x_px + params.step_y_px) / 2
            margin = min(max(2 * int(step_avg), 50), min(H, W) // 2)
            r_min = max(0, int(min(G_actual.nodes[n]['pos_px'][0] for n in G_actual.nodes) - margin))
            c_min = max(0, int(min(G_actual.nodes[n]['pos_px'][1] for n in G_actual.nodes) - margin))
            r_max = min(H, int(max(G_actual.nodes[n]['pos_px'][0] for n in G_actual.nodes) + margin))
            c_max = min(W, int(max(G_actual.nodes[n]['pos_px'][1] for n in G_actual.nodes) + margin))
            bbox = ((r_min, c_min), (r_max, c_max))

            G_theo, G_match, _ = build_and_match(
                G_actual,
                params,
                bbox,
                grass_mask,
                gsd=self.gsd,
            )

            if veg_mask is not None:
                fill_veg_coverage(G_theo, G_match, veg_mask)

            classify_all_edges(G_match)

            void_polygons = detect_void_polygons(G_theo, G_match, self.gsd)
            cells = classify_cells(G_theo, G_match, params, void_polygons, self.gsd)

        clusters = find_damage_clusters(
            cells,
            cell_area_m2=self.grid_size_m * self.grid_size_m,
            area_destroyed_threshold_m2=4 * self.grid_size_m * self.grid_size_m,
            degraded_patch_threshold_m2=self.grid_size_m * self.grid_size_m,
        ) if cells else []

        report = generate_report(
            G_match=G_match,
            cells=cells,
            clusters=clusters,
            grid_params=params,
            confidence=confidence,
            gsd=self.gsd,
        )
        return report

    def _run_intermediates(
        self,
        grass_prob: np.ndarray,
        veg_prob: Optional[np.ndarray] = None,
    ) -> dict:
        """Internal helper: return report + grass_mask + clusters for visuals."""
        grass_mask = clean_grass_mask(grass_prob)
        veg_mask = clean_veg_mask(veg_prob) if veg_prob is not None else None

        G_actual, skeleton = build_physical_graph(grass_mask, self.gsd)

        H, W = grass_prob.shape
        params = estimate_grid_params(
            G_actual, H, W,
            gsd=self.gsd,
            expected_step_m=self.grid_size_m,
            expected_step_px=self.expected_step_px,
        )

        if params is None:
            confidence = "low"
            G_theo = None
            G_match = {}
            void_polygons = []
            cells = []
        else:
            confidence = params.confidence
            step_avg = (params.step_x_px + params.step_y_px) / 2
            margin = min(max(2 * int(step_avg), 50), min(H, W) // 2)
            r_min = max(0, int(min(G_actual.nodes[n]['pos_px'][0] for n in G_actual.nodes) - margin))
            c_min = max(0, int(min(G_actual.nodes[n]['pos_px'][1] for n in G_actual.nodes) - margin))
            r_max = min(H, int(max(G_actual.nodes[n]['pos_px'][0] for n in G_actual.nodes) + margin))
            c_max = min(W, int(max(G_actual.nodes[n]['pos_px'][1] for n in G_actual.nodes) + margin))
            bbox = ((r_min, c_min), (r_max, c_max))

            G_theo, G_match, _ = build_and_match(
                G_actual,
                params,
                bbox,
                grass_mask,
                gsd=self.gsd,
            )

            if veg_mask is not None:
                fill_veg_coverage(G_theo, G_match, veg_mask)

            classify_all_edges(G_match)
            void_polygons = detect_void_polygons(G_theo, G_match, self.gsd)
            cells = classify_cells(G_theo, G_match, params, void_polygons, self.gsd)

        clusters = find_damage_clusters(
            cells,
            cell_area_m2=self.grid_size_m * self.grid_size_m,
            area_destroyed_threshold_m2=4 * self.grid_size_m * self.grid_size_m,
            degraded_patch_threshold_m2=self.grid_size_m * self.grid_size_m,
        ) if cells else []

        report = generate_report(
            G_match=G_match,
            cells=cells,
            clusters=clusters,
            grid_params=params,
            confidence=confidence,
            gsd=self.gsd,
        )
        return {
            "report": report,
            "grass_mask": grass_mask,
            "clusters": clusters,
        }

    def run_with_visuals(
        self,
        rgb_image: np.ndarray,
        grass_prob: np.ndarray,
        veg_prob: Optional[np.ndarray] = None,
        output_dir: str = "./results",
    ) -> Tuple:
        """
        Run analysis + write overlay PNGs.

        Args:
            rgb_image: (H, W, 3) uint8 original remote-sensing image.
            grass_prob: (H, W) float32 grass line probability / mask.
            veg_prob: optional (H, W) float32 vegetation mask.
            output_dir: directory to save PNGs.

        Returns:
            (report, cells_png_path, area_png_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        mid = self._run_intermediates(grass_prob, veg_prob)
        report = mid["report"]
        grass_mask = mid["grass_mask"]
        clusters = mid["clusters"]

        if report.grid_params:
            cells_title = (
                f"Cells overlay: n={len(report.cells)} "
                f"theta={np.degrees(report.grid_params.theta_main):.2f}d "
                f"step={report.grid_params.step_x_px:.0f}px"
            )
        else:
            cells_title = f"Cells overlay: n={len(report.cells)}"
        area_title = f"Area-destroyed clusters: {sum(1 for c in clusters if c.get('type') == 'area_destroyed')} polygons"

        cells_arr = cells_overlay(
            rgb_image=rgb_image,
            grass_mask=grass_mask,
            cells=report.cells,
            title=cells_title,
        )
        area_arr = area_damage_overlay(
            rgb_image=rgb_image,
            clusters=[c for c in clusters if c.get("type") == "area_destroyed"],
            title=area_title,
        )

        cells_png = os.path.join(output_dir, "cells_overlay.png")
        area_png = os.path.join(output_dir, "area_damage_overlay.png")
        save_png(cells_arr, cells_png)
        save_png(area_arr, area_png)

        return report, cells_png, area_png

    def run_from_files(
        self,
        grass_prob_path: str,
        veg_prob_path: Optional[str] = None,
    ):
        """
        Run from .npy files saved by the inference pipeline.

        Args:
            grass_prob_path: path to grass_prob.npy
            veg_prob_path: optional path to veg_prob.npy

        Returns:
            AnalysisReport
        """
        grass_prob = np.load(grass_prob_path)
        veg_prob = np.load(veg_prob_path) if veg_prob_path else None
        return self.run(grass_prob, veg_prob)


def main():
    parser = argparse.ArgumentParser(
        description="Sand checkerboard graph-theoretic damage analysis"
    )
    parser.add_argument(
        "--grass-prob", required=True,
        help="Path to grass_prob.npy (H, W) float32"
    )
    parser.add_argument(
        "--veg-prob",
        help="Path to veg_prob.npy (H, W) float32, optional"
    )
    parser.add_argument(
        "--gsd", type=float, default=0.006388,
        help="Ground sample distance (m/px), default: 0.006388"
    )
    parser.add_argument(
        "--grid-size", type=float, default=2.0,
        help="Sand checkerboard cell size (m), default: 2.0"
    )
    parser.add_argument(
        "--output-dir", default="./results",
        help="Output directory, default: ./results"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analyzer = TopologyAnalyzer(gsd=args.gsd, grid_size_m=args.grid_size)
    report = analyzer.run_from_files(args.grass_prob, args.veg_prob)

    report.to_json(os.path.join(args.output_dir, "report.json"))
    report.to_csv(os.path.join(args.output_dir, "cells.csv"))

    print(
        f"Analysis complete:\n"
        f"  Grass line length: {report.total_actual_length_m:.2f} m\n"
        f"  Damage rate:       {report.global_damage_rate * 100:.1f} %\n"
        f"  Total cells:       {len(report.cells)}\n"
        f"  Destroyed cells:  {report.destroyed_cells}\n"
        f"  Confidence:        {report.confidence}\n"
        f"  Results saved to:  {args.output_dir}/"
    )


if __name__ == "__main__":
    main()
