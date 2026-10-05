"""Region metrics: how a named part of the frame is DRAWN, compared with the
rest of the frame.

Why: a video model can render one part of the picture in a different way from
the rest - typically a soft, smoothly shaded "3D" face on a body drawn with
ink lines and flat tones. Whole-frame numbers hide that. Here a benchmark
definition names analysis regions (e.g. head), each region and the rest of
the frame are measured separately, and the region / rest ratios are reported.

Regions are fixed rectangles [x0, y0, x1, y1] (x1, y1 exclusive) in pixels of
a declared frame size; they are scaled to the size the video is analysed at.
They do not follow motion: they suit shots where the part stays in place.

Measured on luma (0-255), on every frame, at REGION_ANALYSIS_WIDTH (larger
than the 320 px of the motion metrics, because line work disappears at that
size), then averaged over the frames:

  edge_density           share of pixels whose gradient magnitude is at least
                         EDGE_ABOVE: contours, hatching, halftone dots.
  smooth_gradient_share  share of pixels whose gradient magnitude is between
                         FLAT_BELOW and EDGE_ABOVE: neither flat nor an edge,
                         i.e. soft shading. Flat cel tones and hard lines
                         both score low; airbrushed or rendered surfaces high.
  ink_darkness           mean darkness, 0-1 (1 = black), of the darkest
                         INK_SHARE of the pixels: how black the darkest
                         marks are. Inked areas approach 1.
  tone_count             number of luma levels, after quantisation to steps of
                         TONE_STEP, that each hold at least TONE_MIN_SHARE of
                         the pixels. Cel shading has few; a gradient has many.

Gradient magnitude is the length of the central-difference gradient, in luma
units per pixel.

ratios[region][measure] = region value / rest value (None when the rest value
is 0). For a soft face on an inked body: edge_density and ink_darkness ratios
below 1, smooth_gradient_share and tone_count ratios above 1.

These are MEASUREMENTS without thresholds. They do not know what a face is:
a region that is simply a different material (sky, skin, a wall) differs too.
Texture added on purpose (grain, halftone) counts as edges or soft shading.
Compare runs of the same shot with the same regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REGION_ANALYSIS_WIDTH = 640
FLAT_BELOW = 1.0  # gradient magnitude below this: flat (also hides most encoder noise)
EDGE_ABOVE = 20.0  # gradient magnitude from this on: an edge
INK_SHARE = 0.10
TONE_STEP = 8
TONE_MIN_SHARE = 0.01
MIN_REGION_SIDE = 8  # pixels at the analysis size
REST = "rest"
MEASURES = ("edge_density", "smooth_gradient_share", "ink_darkness", "tone_count")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class AnalysisRegions:
    """Named rectangles in pixels of a frame of `width` x `height`."""

    width: int
    height: int
    boxes: dict[str, tuple[int, int, int, int]]

    def __post_init__(self) -> None:
        if not _is_int(self.width) or not _is_int(self.height) or self.width <= 0 or self.height <= 0:
            raise ValueError("the regions' frame size must be two positive whole numbers [width, height]")
        if not isinstance(self.boxes, dict) or not self.boxes:
            raise ValueError("at least one analysis region is needed")
        for name, box in self.boxes.items():
            if not isinstance(name, str) or not name.replace("_", "").isalnum():
                raise ValueError(f"region name must be letters, digits and underscores, got {name!r}")
            if name == REST:
                raise ValueError(f"{REST!r} is the name of everything outside the regions; choose another name")
            if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_is_int(v) for v in box):
                raise ValueError(f"region {name}: must be [x0, y0, x1, y1] in whole pixels")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
                raise ValueError(
                    f"region {name}: [x0, y0, x1, y1] = {list(box)} must lie inside the "
                    f"{self.width}x{self.height} frame with x0 < x1 and y0 < y1"
                )
        object.__setattr__(self, "boxes", {name: tuple(box) for name, box in self.boxes.items()})

    @classmethod
    def from_definition(cls, regions: Any, size: Any) -> AnalysisRegions:
        """From a definition's analysis_regions mapping and analysis_regions_size [width, height]."""
        if not isinstance(regions, dict):
            raise ValueError("analysis_regions must be a mapping: name -> [x0, y0, x1, y1]")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(
                "analysis_regions_size [width, height] is required with analysis_regions: "
                "the frame size the pixel coordinates refer to"
            )
        return cls(width=size[0], height=size[1], boxes=dict(regions))

    def scaled(self, width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
        """The boxes in pixels of a `width` x `height` frame."""
        fx, fy = width / self.width, height / self.height
        scaled = {}
        for name, (x0, y0, x1, y1) in self.boxes.items():
            box = (round(x0 * fx), round(y0 * fy), min(width, round(x1 * fx)), min(height, round(y1 * fy)))
            if box[2] - box[0] < MIN_REGION_SIDE or box[3] - box[1] < MIN_REGION_SIDE:
                raise ValueError(
                    f"region {name} is only {box[2] - box[0]}x{box[3] - box[1]} px at the analysis size "
                    f"{width}x{height}; it needs at least {MIN_REGION_SIDE}x{MIN_REGION_SIDE}"
                )
            scaled[name] = box
        return scaled

    def to_dict(self) -> dict[str, Any]:
        return {"size": [self.width, self.height], "boxes": {name: list(box) for name, box in self.boxes.items()}}


def gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    """Length of the central-difference luma gradient of one (H, W) frame."""
    dy, dx = np.gradient(frame.astype(np.float32))
    return np.hypot(dx, dy)


def area_measures(luma: np.ndarray, gradient: np.ndarray) -> dict[str, float]:
    """The four measures for one area of one frame, given its pixels' luma and
    gradient magnitude as flat arrays."""
    count = luma.size
    darkest = np.partition(luma, max(0, int(np.ceil(count * INK_SHARE)) - 1))[: max(1, int(np.ceil(count * INK_SHARE)))]
    levels = np.bincount((luma // TONE_STEP).astype(np.int64), minlength=256 // TONE_STEP)
    return {
        "edge_density": float((gradient >= EDGE_ABOVE).mean()),
        "smooth_gradient_share": float(((gradient >= FLAT_BELOW) & (gradient < EDGE_ABOVE)).mean()),
        "ink_darkness": float((255.0 - darkest.mean()) / 255.0),
        "tone_count": float((levels >= TONE_MIN_SHARE * count).sum()),
    }


def region_metrics(frames: np.ndarray, regions: AnalysisRegions) -> dict[str, Any]:
    """Measure every region and the rest of the frame on a gray stack
    (N, H, W), 0-255, and report region / rest ratios. JSON-safe dict:

        {"declared": {"size": [w, h], "boxes": {...}},
         "analysis_size": {"width", "height"},
         "regions": {name: {"box": [...], "pixels": n, <measures>}},
         "rest": {"pixels": n, <measures>},
         "ratios": {name: {<measure>: region / rest or None}}}
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError(f"frames must be a non-empty gray stack (N, H, W), got shape {frames.shape}")
    height, width = frames.shape[1:]
    boxes = regions.scaled(width, height)
    masks = {}
    rest = np.ones((height, width), dtype=bool)
    for name, (x0, y0, x1, y1) in boxes.items():
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks[name] = mask
        rest &= ~mask
    if not rest.any():
        raise ValueError("the regions cover the whole frame; nothing is left to compare them with")
    masks[REST] = rest

    totals = {name: dict.fromkeys(MEASURES, 0.0) for name in masks}
    for frame in frames:
        luma = frame.astype(np.float32)
        gradient = gradient_magnitude(luma)
        for name, mask in masks.items():
            for measure, value in area_measures(luma[mask], gradient[mask]).items():
                totals[name][measure] += value
    means = {
        name: {measure: round(total / len(frames), 6) for measure, total in values.items()}
        for name, values in totals.items()
    }
    return {
        "declared": regions.to_dict(),
        "analysis_size": {"width": int(width), "height": int(height)},
        "regions": {
            name: {"box": list(boxes[name]), "pixels": int(masks[name].sum()), **means[name]} for name in boxes
        },
        "rest": {"pixels": int(rest.sum()), **means[REST]},
        "ratios": {
            name: {
                measure: None if means[REST][measure] == 0 else round(means[name][measure] / means[REST][measure], 6)
                for measure in MEASURES
            }
            for name in boxes
        },
    }
