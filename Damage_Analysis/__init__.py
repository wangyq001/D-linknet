"""Damage Analysis: Graph-theoretic sand checkerboard damage detection."""

from .models import GridParams, EdgeMatch, CellState, AnalysisReport
from .postprocess import clean_grass_mask, clean_veg_mask
from .skeleton import (
    build_physical_graph,
    skeletonize_mask,
    remove_spurs,
    detect_nodes,
    trace_edges,
)
from .gridfit import (
    estimate_grid_params,
    rotate_points,
    estimate_dominant_directions,
)
from .dualgraph import (
    build_and_match,
    build_theoretical_graph,
    generate_theoretical_edge_pixels,
)
from .damage import (
    compute_veg_coverage,
    fill_veg_coverage,
    classify_edge_reason,
    detect_void_polygons,
    classify_cells,
    find_damage_clusters,
)
from .report import generate_report

__all__ = [
    "GridParams",
    "EdgeMatch",
    "CellState",
    "AnalysisReport",
    "clean_grass_mask",
    "clean_veg_mask",
    "build_physical_graph",
    "skeletonize_mask",
    "remove_spurs",
    "detect_nodes",
    "trace_edges",
    "estimate_grid_params",
    "rotate_points",
    "estimate_dominant_directions",
    "build_and_match",
    "build_theoretical_graph",
    "generate_theoretical_edge_pixels",
    "compute_veg_coverage",
    "fill_veg_coverage",
    "classify_edge_reason",
    "detect_void_polygons",
    "classify_cells",
    "find_damage_clusters",
    "generate_report",
]
