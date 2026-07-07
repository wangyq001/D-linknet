"""Stage 7: Cluster analysis and report generation."""

from typing import List, Optional
from datetime import datetime
try:
    from .models import GridParams, EdgeMatch, CellState, AnalysisReport
except ImportError:
    from models import GridParams, EdgeMatch, CellState, AnalysisReport


def generate_report(
    G_match: dict,
    cells: List[CellState],
    clusters: List[dict],
    grid_params: Optional[GridParams],
    confidence: str,
    gsd: float = 0.006388,
) -> AnalysisReport:
    """
    Generate a complete AnalysisReport from all analysis results.

    Args:
        G_match: dict of edge_id → EdgeMatch
        cells: list of CellState
        clusters: list of cluster dicts from find_damage_clusters
        grid_params: GridParams or None
        confidence: 'high'|'medium'|'low'
        gsd: ground sample distance (m/px)

    Returns:
        AnalysisReport dataclass
    """
    intact_n    = sum(1 for c in cells if c.status == "intact")
    minor_n     = sum(1 for c in cells if c.status == "minor")
    mod_n       = sum(1 for c in cells if c.status == "moderate")
    severe_n    = sum(1 for c in cells if c.status == "severe")
    destroy_n   = sum(1 for c in cells if c.status == "destroyed")
    pseudo_n    = sum(1 for c in cells if c.status == "pseudo_damaged")
    ambiguous_n = sum(1 for c in cells if c.status == "ambiguous")

    total_actual = sum(m.actual_length_m for m in G_match.values())
    total_theo   = sum(m.theoretical_length_m for m in G_match.values())
    global_rate  = 1.0 - total_actual / max(total_theo, 1e-6)

    genuine_len = sum(
        m.missing_length_m for m in G_match.values()
        if m.reason == "genuine_break"
    )
    occluded_len = sum(
        m.missing_length_m for m in G_match.values()
        if m.reason == "occluded"
    )

    destroyed_regions = [c for c in clusters if c["type"] == "area_destroyed"]
    degraded_patches  = [c for c in clusters if c["type"] == "degraded_patch"]
    isolated          = [c for c in clusters if c["type"] == "isolated_damage"]

    return AnalysisReport(
        total_actual_length_m=total_actual,
        total_theoretical_length_m=total_theo,
        global_damage_rate=global_rate,
        intact_cells=intact_n,
        minor_cells=minor_n,
        moderate_cells=mod_n,
        severe_cells=severe_n,
        destroyed_cells=destroy_n,
        pseudo_damaged_cells=pseudo_n,
        ambiguous_cells=ambiguous_n,
        cells=cells,
        area_destroyed_regions=destroyed_regions,
        degraded_patches=degraded_patches,
        isolated_damages=isolated,
        genuine_break_total_length_m=genuine_len,
        occluded_total_length_m=occluded_len,
        grid_params=grid_params,
        confidence=confidence,
        timestamp=datetime.now().isoformat(),
        gsd_m=gsd,
    )
