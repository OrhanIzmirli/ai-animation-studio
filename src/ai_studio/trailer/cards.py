"""Still cards for a trailer assembly: labelled grey placeholders for shots
that have no clip yet, and the title card.

Cards are editing elements, drawn here with Pillow; no model is involved.
Pillow is imported lazily (it is a development dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_studio.media import Resolution, image_dimensions
from ai_studio.quality.frames import decode_image
from ai_studio.trailer.shotlist import Shot, TitleCard

PLACEHOLDER_GREY = 128
_LIGHT, _DARK = (238, 238, 238), (44, 44, 44)


def _pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is needed to draw trailer cards: pip install pillow") from exc
    return Image, ImageDraw, ImageFont


def _font(image_font: Any, size: int) -> Any:
    try:
        return image_font.load_default(size=size)
    except TypeError:  # Pillow < 10.1: only the small bitmap font
        return image_font.load_default()


def _wrap(draw: Any, text: str, font: Any, max_width: float) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(candidate, font=font) <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines or [""]


def _draw_block(
    draw: Any, blocks: list[tuple[str, Any, tuple[int, int, int]]], resolution: Resolution, gap: int,
) -> None:
    """Draw text blocks centred, one under the other, wrapped to the card."""
    max_width = resolution.width * 0.84
    rows = [(line, font, colour) for text, font, colour in blocks for line in _wrap(draw, text, font, max_width)]
    heights = [draw.textbbox((0, 0), "Ag", font=font)[3] for _, font, _ in rows]
    y = (resolution.height - (sum(heights) + gap * (len(rows) - 1))) / 2
    for (line, font, colour), height in zip(rows, heights):
        x = (resolution.width - draw.textlength(line, font=font)) / 2
        draw.text((x, y), line, font=font, fill=colour)
        y += height + gap


def placeholder_card(shot: Shot, number: int, total: int, resolution: Resolution) -> np.ndarray:
    """A grey card that says which shot is missing. (H, W, 3) uint8."""
    image_module, image_draw, image_font = _pillow()
    image = image_module.new("RGB", (resolution.width, resolution.height), (PLACEHOLDER_GREY,) * 3)
    draw = image_draw.Draw(image)
    unit = max(8, resolution.height // 22)
    margin = unit // 2
    draw.rectangle(
        [margin, margin, resolution.width - 1 - margin, resolution.height - 1 - margin],
        outline=_DARK, width=max(1, unit // 8),
    )
    big, medium, small = (_font(image_font, size) for size in (unit * 2, int(unit * 1.4), unit))
    blocks = [
        ("PLACEHOLDER", big, _LIGHT),
        (f"SHOT {number} / {total}   {shot.id}   {shot.duration_seconds:g} s", medium, _DARK),
        (shot.name, medium, _LIGHT),
        (f"camera: {shot.camera}", small, _DARK),
    ]
    if shot.needs_character_keyframe:
        blocks.append(("needs a character keyframe", small, _DARK))
    blocks.append((f"status: {shot.status} - no clip chosen yet", small, _DARK))
    _draw_block(draw, blocks, resolution, gap=unit // 2)
    return np.asarray(image, dtype=np.uint8)


def title_card(card: TitleCard, resolution: Resolution, image_path: Path | None = None) -> np.ndarray:
    """The title card: the finished image if the shot names one (it must have
    the trailer's resolution), else white text on black. (H, W, 3) uint8."""
    if image_path is not None:
        if not image_path.is_file():
            raise FileNotFoundError(f"title card image not found: {image_path}")
        size = image_dimensions(image_path)
        if size is not None and size != resolution:
            raise ValueError(
                f"title card image is {size.width}x{size.height}, the trailer is "
                f"{resolution.width}x{resolution.height}"
            )
        return decode_image(image_path, (resolution.width, resolution.height), gray=False)
    image_module, image_draw, image_font = _pillow()
    image = image_module.new("RGB", (resolution.width, resolution.height), (0, 0, 0))
    draw = image_draw.Draw(image)
    unit = max(8, resolution.height // 22)
    blocks = [(card.text, _font(image_font, unit * 3), (255, 255, 255))]
    if card.subtitle:
        blocks.append((card.subtitle, _font(image_font, int(unit * 1.2)), (200, 200, 200)))
    _draw_block(draw, blocks, resolution, gap=unit)
    return np.asarray(image, dtype=np.uint8)
