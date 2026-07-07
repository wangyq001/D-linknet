"""
Visualization overlays for the damage analysis report.

Produces two PNGs on top of the original remote-sensing RGB image:

  1. cells_overlay.png
     - RGB image + grass skeleton + per-cell index labels
       - RED text: genuine breakage (any edge has status in
         {moderate, severe, destroyed, ambiguous} with genuine_break>0)
       - YELLOW text: pseudo_damage (vegetation occlusion only)
       - GREEN text: intact / minor, no genuine breakage

  2. area_damage_overlay.png
     - RGB image + red semi-transparent polygons for area_destroyed
       clusters (cells that span large damaged regions).
"""
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .models import CellState
except ImportError:
    from models import CellState


# Color palette
COLOR_GRASS = (0, 255, 0)          # green skeleton line
COLOR_GRASS_BG = (0, 0, 0)         # black bg for text contrast
COLOR_TEXT_RED = (220, 30, 30)     # genuine breakage
COLOR_TEXT_YELLOW = (255, 210, 0)  # pseudo-damaged
COLOR_TEXT_GREEN = (40, 200, 40)   # intact / minor
COLOR_TEXT_BLACK = (0, 0, 0)       # outline of text
COLOR_AREA_DAMAGE = (255, 30, 30)   # area-destroyed fill, alpha-blended


def _classify_label_color(cell: CellState) -> Tuple[Tuple[int, int, int], str]:
    """Decide color and label text for a cell."""
    if cell.status == "pseudo_damaged":
        text = f"{cell.cell_id[0]},{cell.cell_id[1]}"
        return COLOR_TEXT_YELLOW, text

    if cell.status in ("intact", "minor"):
        text = f"{cell.cell_id[0]},{cell.cell_id[1]}"
        return COLOR_TEXT_GREEN, text

    if cell.status == "ambiguous":
        text = f"{cell.cell_id[0]},{cell.cell_id[1]}"
        return COLOR_TEXT_YELLOW, text

    # moderate / severe / destroyed → check whether this is genuine or
    # could be vegetation
    if cell.genuine_break_count >= 1 and cell.status in (
        "moderate", "severe", "destroyed"
    ):
        text = f"{cell.cell_id[0]},{cell.cell_id[1]}"
        return COLOR_TEXT_RED, text

    # Other damaged but no genuine break (e.g., all-missing edges
    # surrounded by vegetation) → fall back to yellow
    text = f"{cell.cell_id[0]},{cell.cell_id[1]}"
    return COLOR_TEXT_YELLOW, text


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try common CJK / Latin fonts; fall back to default bitmap."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _pil_font_size_for_image(shape: Tuple[int, int]) -> int:
    """Pick a font size scaled to image size."""
    h, w = shape
    side = min(h, w)
    if side <= 256:
        return max(8, side // 24)
    if side <= 512:
        return max(10, side // 22)
    if side <= 1024:
        return max(12, side // 26)
    return max(14, side // 32)


def _draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    fill: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
    outline: Tuple[int, int, int] = COLOR_TEXT_BLACK,
    outline_width: int = 2,
):
    """Draw text with a small black outline for legibility on any bg."""
    x, y = xy
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)


def cells_overlay(
    rgb_image: np.ndarray,
    grass_mask: np.ndarray,
    cells: List[CellState],
    title: str = "Cell indices by damage type",
) -> np.ndarray:
    """
    Render `cells_overlay.png` overlay.

    Args:
        rgb_image: (H, W, 3) uint8. Original remote-sensing RGB.
        grass_mask: (H, W) binary. Detected grass-line mask, 0/1 or 0/255.
        cells: list of CellState.
        title: figure title for header bar.

    Returns:
        overlay: (H', W', 3) uint8 RGB image (with header) suitable for
                 saving as PNG.
    """
    H, W = rgb_image.shape[:2]
    header_h = max(40, H // 24)
    pad = 8

    base = Image.fromarray(rgb_image).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")

    # 1) Draw grass mask in translucent green
    if grass_mask is not None and grass_mask.any():
        grass_u8 = (grass_mask > 0).astype(np.uint8) * 255
        grass_img = Image.fromarray(grass_u8).convert("L")
        green_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        green_arr = np.array(green_layer)
        idx = np.array(grass_img) > 0
        green_arr[idx] = [0, 255, 0, 130]  # green at ~50% alpha
        green_overlay = Image.fromarray(green_arr, "RGBA")
        base = base.convert("RGBA")
        base.alpha_composite(green_overlay)
        base = base.convert("RGB")

    draw = ImageDraw.Draw(base, "RGBA")

    # 2) Draw cell index labels at cell center
    font_size = _pil_font_size_for_image((H, W))
    font = _load_font(font_size)

    for cell in cells:
        ((r0, c0), (r1, c1)) = cell.bbox_px
        cr = (r0 + r1) / 2
        cc = (c0 + c1) / 2

        color, text = _classify_label_color(cell)
        # Centre text by anchor offset
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        else:
            tw, th = draw.textsize(text, font=font)
        x = int(cc - tw / 2)
        y = int(cr - th / 2)
        _draw_text_with_outline(draw, (x, y), text, fill=color, font=font)

    # 3) Compose with header
    final_h = H + header_h
    final = Image.new("RGB", (W, final_h), (255, 255, 255))
    final.paste(base, (0, header_h))

    header_draw = ImageDraw.Draw(final)
    title_font = _load_font(max(14, header_h // 2))
    header_draw.rectangle([(0, 0), (W, header_h)], fill=(20, 20, 20))
    header_draw.text((pad, 4), title, font=title_font, fill=(255, 255, 255))

    # Legend
    legend_x = pad + 360
    legend_items = [
        ("red=genuine break", COLOR_TEXT_RED),
        ("yellow=occluded", COLOR_TEXT_YELLOW),
        ("green=intact/minor", COLOR_TEXT_GREEN),
    ]
    cur_x = legend_x
    for label, color in legend_items:
        if hasattr(title_font, "getbbox"):
            tw_l = title_font.getbbox(label)[2]
        else:
            tw_l = header_draw.textlength(label, font=title_font)
        header_draw.text((cur_x, 4), label, font=title_font, fill=color)
        cur_x += tw_l + 24

    return np.array(final)


def area_damage_overlay(
    rgb_image: np.ndarray,
    clusters: List[dict],
    title: str = "Area damage (red polygon = regional destruction)",
) -> np.ndarray:
    """
    Render `area_damage_overlay.png` overlay.

    Args:
        rgb_image: (H, W, 3) uint8. Original remote-sensing RGB.
        clusters: list from find_damage_clusters (with 'polygon' field,
                  in pixel (r, c) coordinates).
        title: header text.

    Returns:
        overlay: (H', W', 3) uint8 RGB image (with header).
    """
    H, W = rgb_image.shape[:2]
    header_h = max(40, H // 24)
    pad = 8

    base = Image.fromarray(rgb_image).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")

    legend_entries = []

    for cluster in clusters:
        poly: List[Tuple[float, float]] = cluster.get("polygon", [])
        if len(poly) < 3:
            continue
        # Convert (r, c) → (x, y) for PIL: PIL expects (x, y) = (c, r).
        xy = [(c, r) for (r, c) in poly]

        ctype = cluster.get("type", "isolated_damage")
        if ctype == "area_destroyed":
            fill = (255, 30, 30, 80)
            outline = (200, 0, 0, 230)
            label = "area_destroyed"
            color = COLOR_AREA_DAMAGE
        elif ctype == "degraded_patch":
            fill = (255, 165, 0, 60)
            outline = (200, 130, 0, 220)
            label = "degraded_patch"
            color = (255, 165, 0)
        else:
            fill = (255, 230, 0, 50)
            outline = (200, 180, 0, 200)
            label = "isolated_damage"
            color = (255, 230, 0)

        draw.polygon(xy, fill=fill, outline=outline)

        # Tiny label at first corner
        if poly:
            r0, c0 = poly[0]
            x0, y0 = int(c0), int(r0)
            text = f"{ctype[:5]}"
            small_font = _load_font(max(10, _pil_font_size_for_image((H, W)) - 4))
            _draw_text_with_outline(draw, (x0, y0), text, fill=color, font=small_font)
            legend_entries.append((label, color, cluster["cell_count"]))

    final_h = H + header_h
    final = Image.new("RGB", (W, final_h), (255, 255, 255))
    final.paste(base, (0, header_h))

    header_draw = ImageDraw.Draw(final)
    title_font = _load_font(max(14, header_h // 2))
    header_draw.rectangle([(0, 0), (W, header_h)], fill=(20, 20, 20))
    header_draw.text((pad, 4), title, font=title_font, fill=(255, 255, 255))

    cur_x = pad + 480
    for label, color, count in legend_entries:
        item = f"{label} x {count}"
        if hasattr(title_font, "getbbox"):
            tw_l = title_font.getbbox(item)[2]
        else:
            tw_l = header_draw.textlength(item, font=title_font)
        header_draw.text((cur_x, 4), item, font=title_font, fill=color)
        cur_x += tw_l + 24

    return np.array(final)


def save_png(arr_rgb: np.ndarray, path: str) -> None:
    """Save numpy RGB uint8 array as PNG."""
    Image.fromarray(arr_rgb).save(path, format="PNG", optimize=True)
