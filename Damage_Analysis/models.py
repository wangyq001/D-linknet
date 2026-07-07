"""Core data structures for damage analysis."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class GridParams:
    """Estimated grid parameters from the physical graph."""

    theta_main: float          # rotation angle (radians), CCW positive
    step_x_px: float           # column direction step (pixels)
    step_y_px: float           # row direction step (pixels)
    phase_x_px: float          # column direction phase offset (pixels)
    phase_y_px: float          # row direction phase offset (pixels)
    gsd_m: float              # ground sample distance (meters/pixel)
    direction_entropy: float   # direction histogram entropy (higher = less regular)
    confidence: str            # 'high' | 'medium' | 'low'

    def to_dict(self) -> dict:
        return {
            "theta_main_deg": float(np.degrees(self.theta_main)),
            "step_x_m": float(self.step_x_px * self.gsd_m),
            "step_y_m": float(self.step_y_px * self.gsd_m),
            "phase_x_px": float(self.phase_x_px),
            "phase_y_px": float(self.phase_y_px),
            "direction_entropy": float(self.direction_entropy),
            "confidence": self.confidence,
        }


@dataclass
class EdgeMatch:
    """Result of matching a theoretical edge against physical skeleton."""

    edge_id: Tuple[int, int]   # (u, v) node IDs in G_theoretical
    coverage_ratio: float      # fraction [0,1] of theoretical edge covered by skeleton
    actual_length_m: float    # matched skeleton length (meters)
    theoretical_length_m: float  # theoretical edge length (meters)
    missing_length_m: float   # uncovered length (meters)
    status: str               # 'intact' | 'minor' | 'severe' | 'missing'
    veg_coverage: float      # fraction [0,1] of edge covered by vegetation mask
    reason: str              # 'ok' | 'genuine_break' | 'occluded' | 'ambiguous' | 'unknown'
    direction: str = 'H'     # 'H' | 'V' | 'diagonal'


@dataclass
class CellState:
    """Damage state of a single sand checkerboard cell."""

    cell_id: Tuple[int, int]   # (row_idx, col_idx) grid coordinates
    bbox_px: Tuple[Tuple[float, float], Tuple[float, float]]  # ((r0,c0),(r1,c1))
    damage_rate: float        # fraction [0,1] of edge length missing
    status: str               # 'intact' | 'minor' | 'moderate' | 'severe' | 'destroyed' | 'pseudo_damaged' | 'ambiguous'
    edge_statuses: Dict[str, str]  # {'top':..., 'bottom':..., 'left':..., 'right':...}
    genuine_break_count: int  # number of edges marked genuine_break
    occluded_count: int       # number of edges marked occluded
    area_m2: float           # cell area in square meters
    veg_covered_sides: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete damage analysis report."""

    total_actual_length_m: float
    total_theoretical_length_m: float
    global_damage_rate: float
    intact_cells: int
    minor_cells: int
    moderate_cells: int
    severe_cells: int
    destroyed_cells: int
    pseudo_damaged_cells: int
    ambiguous_cells: int
    cells: List[CellState]
    area_destroyed_regions: List[dict]
    degraded_patches: List[dict]
    isolated_damages: List[dict]
    genuine_break_total_length_m: float
    occluded_total_length_m: float
    grid_params: Optional[GridParams]
    confidence: str
    timestamp: str
    gsd_m: float

    def to_dict(self) -> dict:
        total = (
            self.intact_cells + self.minor_cells + self.moderate_cells
            + self.severe_cells + self.destroyed_cells + self.pseudo_damaged_cells
        )
        return {
            "total_actual_length_m": round(self.total_actual_length_m, 3),
            "total_theoretical_length_m": round(self.total_theoretical_length_m, 3),
            "global_damage_rate": round(self.global_damage_rate * 100, 2),
            "cell_summary": {
                "total": total,
                "intact": self.intact_cells,
                "minor": self.minor_cells,
                "moderate": self.moderate_cells,
                "severe": self.severe_cells,
                "destroyed": self.destroyed_cells,
                "pseudo_damaged": self.pseudo_damaged_cells,
                "ambiguous": self.ambiguous_cells,
            },
            "damage_length_m": {
                "genuine_break": round(self.genuine_break_total_length_m, 3),
                "occluded": round(self.occluded_total_length_m, 3),
            },
            "area_clusters": {
                "destroyed_areas": len(self.area_destroyed_regions),
                "degraded_patches": len(self.degraded_patches),
                "isolated": len(self.isolated_damages),
            },
            "grid_params": self.grid_params.to_dict() if self.grid_params else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    def to_csv(self, path: str):
        """Export cell list as CSV."""
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "cell_id_i", "cell_id_j", "damage_rate",
                "status", "genuine_break_count", "occluded_count", "area_m2",
            ])
            writer.writeheader()
            for c in self.cells:
                writer.writerow({
                    "cell_id_i": c.cell_id[0],
                    "cell_id_j": c.cell_id[1],
                    "damage_rate": round(c.damage_rate, 4),
                    "status": c.status,
                    "genuine_break_count": c.genuine_break_count,
                    "occluded_count": c.occluded_count,
                    "area_m2": round(c.area_m2, 4),
                })

    def to_json(self, path: str):
        """Export report as JSON."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
