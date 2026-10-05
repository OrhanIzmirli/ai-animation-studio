"""Two simple measures for comparing a styled video with its raw source.

They answer two questions raised by the first profiles:

    luminance_shift    did the style make the picture darker or lighter?
    accent_retention   did small saturated accents keep their colour?

Both compare frames of the same shot pixel for pixel, so they only make
sense between a raw clip and a styled version of it with the same timing
(the styled frame i is compared with the raw drawing it shows).
"""

from __future__ import annotations

import numpy as np

from ai_studio.post.noise import luma
from ai_studio.post.operations import ACCENT_HUE_BINS, saturation_value_hue

ACCENT_SATURATION = 0.6  # raw pixel must be at least this saturated (HSV, 0-1) ...
ACCENT_VALUE = 0.5  # ... and this bright to count as vivid
ACCENT_MAX_FRAME_SHARE = 0.02  # vivid pixels of its hue may cover at most this share of the frame ("small")
RETAINED_HUE_TOLERANCE = 30.0 / 360.0  # styled hue within 30 degrees of the raw hue
RETAINED_SATURATION = 0.6  # styled saturation at least this fraction of the raw saturation


def luminance_shift(raw: np.ndarray, styled: np.ndarray) -> float:
    """Mean luma of `styled` minus mean luma of `raw` (0-255 scale; negative =
    darker). Arrays (N, H, W, 3) or (H, W, 3), same shape."""
    raw, styled = np.asarray(raw), np.asarray(styled)
    if raw.shape != styled.shape:
        raise ValueError("raw and styled must have the same shape")
    return float(luma(styled).mean() - luma(raw).mean())


def accent_mask(raw_frame: np.ndarray) -> np.ndarray:
    """Accent pixels of one raw frame (H, W) bool: vivid (saturation >=
    ACCENT_SATURATION, value >= ACCENT_VALUE) and of a hue whose vivid pixels
    cover at most ACCENT_MAX_FRAME_SHARE of the frame. A large saturated area
    (a red sky) is not an accent; a small sign in another hue is."""
    saturation, value, hue = saturation_value_hue(raw_frame)
    vivid = (saturation >= ACCENT_SATURATION) & (value >= ACCENT_VALUE)
    total = int(vivid.sum())
    if total == 0:
        return vivid
    bins = np.minimum((hue * ACCENT_HUE_BINS).astype(int), ACCENT_HUE_BINS - 1)
    counts = np.bincount(bins[vivid], minlength=ACCENT_HUE_BINS)
    rare = counts <= ACCENT_MAX_FRAME_SHARE * vivid.size
    return vivid & rare[bins]


def accent_retention(raw: np.ndarray, styled: np.ndarray) -> float | None:
    """Share of the raw clip's accent pixels (see accent_mask) that are still
    an accent of the same colour in the styled clip: hue within 30 degrees of
    the raw hue and saturation at least 60 % of the raw saturation.

    1.0 = every accent kept its colour, 0.0 = all lost. None if the raw clip
    has no accent pixels. Brightness is not part of it: an accent under an
    ink line is darker but still counts if hue and saturation survive."""
    raw, styled = np.asarray(raw), np.asarray(styled)
    if raw.shape != styled.shape:
        raise ValueError("raw and styled must have the same shape")
    if raw.ndim == 3:
        raw, styled = raw[None], styled[None]
    accents = kept = 0
    for raw_frame, styled_frame in zip(raw, styled):
        mask = accent_mask(raw_frame)
        if not mask.any():
            continue
        raw_s, _, raw_h = saturation_value_hue(raw_frame[mask])
        styled_s, _, styled_h = saturation_value_hue(styled_frame[mask])
        distance = np.abs(raw_h - styled_h)
        distance = np.minimum(distance, 1.0 - distance)  # hue is circular
        accents += int(mask.sum())
        kept += int(((distance <= RETAINED_HUE_TOLERANCE) & (styled_s >= RETAINED_SATURATION * raw_s)).sum())
    return None if accents == 0 else kept / accents
