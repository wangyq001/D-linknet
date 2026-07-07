"""Stage 1: Post-processing of probability maps."""

import numpy as np
import cv2


def clean_grass_mask(
    grass_prob: np.ndarray,
    threshold: float = 0.5,
    close_radius: int = 3,
    min_area: int = 20,
) -> np.ndarray:
    """
    Clean sand-line binary mask from probability map.

    Pipeline:
      1. Threshold binarization
      2. Morphological closing (disk) to bridge small gaps
      3. Morphological opening (disk) to remove isolated noise blobs
      4. Connected-component area filter

    Args:
        grass_prob: (H, W) float32 probability map, range [0, 1]
        threshold: binarization threshold, default 0.5
        close_radius: disk radius for closing (pixels), default 3 (~2cm at GSD=0.006)
        min_area: minimum connected-component area (pixels), default 20

    Returns:
        mask: (H, W) uint8, values 0 or 255
    """
    binary = (grass_prob > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    filtered = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered


def clean_veg_mask(
    veg_prob: np.ndarray,
    threshold: float = 0.5,
    close_radius: int = 3,
) -> np.ndarray:
    """
    Clean vegetation binary mask from probability map.

    Pipeline:
      1. Threshold binarization
      2. Morphological closing (disk) to fill small interior holes

    Args:
        veg_prob: (H, W) float32 probability map, range [0, 1]
        threshold: binarization threshold, default 0.5
        close_radius: disk radius for closing (pixels), default 3

    Returns:
        mask: (H, W) uint8, values 0 or 255
    """
    binary = (veg_prob > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed
