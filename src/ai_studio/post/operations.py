"""Style-layer operations: pure functions on stacks of DRAWINGS.

Each operation takes `frames`, a uint8 array (M, H, W, 3) holding one picture
per drawing, and returns a new array of the same shape. It never modifies its
input. The same input and parameters always give the same output.

`ids` names the drawings (default 0..M-1; the pipeline passes the source
frame index of each drawing). Any random field of an operation is derived from
(seed, drawing id): it changes when a new drawing starts and never inside a
hold, because a held drawing is one picture shown several times.

No operation applies per-pixel independent noise to lines or shapes. Line
weight and wobble use smooth, spatially coherent fields (ai_studio.post.noise).
Only grain is fine-grained, and its strength follows the luminance.

Timing (which frames are drawings and which are holds) is in
ai_studio.post.timing; the operations here are applied to the drawings and
the pipeline then lays them out along the hold pattern.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_studio.post.noise import (
    SALT_GRAIN,
    SALT_HALFTONE,
    SALT_PALETTE,
    SALT_PRESSURE,
    SALT_WOBBLE_X,
    SALT_WOBBLE_Y,
    box_blur,
    fine_noise,
    luma,
    max_filter,
    rng_for,
    shift_edge,
    smooth_noise,
)


def _check(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be uint8 with shape (M, H, W, 3), got {frames.dtype} {frames.shape}")
    return frames


def _ids(frames: np.ndarray, ids: Sequence[int] | None) -> list[int]:
    ids = list(range(len(frames))) if ids is None else [int(i) for i in ids]
    if len(ids) != len(frames):
        raise ValueError("ids must have one entry per drawing")
    return ids


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


# --- palette ---

ACCENT_HUE_BINS = 24
ACCENT_MIN_SHARE = 0.0002  # a hue must cover at least this share of the image to count as an accent


def saturation_value_hue(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HSV saturation (0-1), value (0-1) and hue (0-1) of (..., 3) RGB pixels."""
    rgb = np.asarray(pixels, dtype=np.float32)
    high, low = rgb.max(axis=-1), rgb.min(axis=-1)
    chroma = high - low
    saturation = np.where(high > 0, chroma / np.maximum(high, 1e-6), 0.0)
    safe = np.maximum(chroma, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(high == r, ((g - b) / safe) % 6.0, np.where(high == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    return saturation, high / 255.0, np.where(chroma > 0, hue / 6.0, 0.0)


def _kmeans(sample: np.ndarray, k: int, seed: int, iterations: int) -> np.ndarray:
    rng = rng_for(seed, SALT_PALETTE)
    centres = sample[rng.choice(len(sample), size=min(k, len(sample)), replace=False)].copy()
    for _ in range(iterations):
        nearest = _nearest(sample, centres)
        for index in range(len(centres)):
            members = sample[nearest == index]
            if len(members):
                centres[index] = members.mean(axis=0)
    return centres


def accent_colours(
    source_image: np.ndarray, base: np.ndarray, slots: int, saturation: float = 0.55, value: float = 0.35,
) -> np.ndarray:
    """Up to `slots` saturated colours of the image that the palette `base`
    represents worst: (n, 3) float32, n <= slots.

    Pixels with HSV saturation >= `saturation` and value >= `value` are
    grouped by hue (ACCENT_HUE_BINS bins). Each hue that covers at least
    ACCENT_MIN_SHARE of the image offers its mean colour as a candidate. The
    candidates farthest from every colour already in the palette are taken
    first, so slots go to small accents that plain k-means would merge into a
    neighbouring colour, not to large saturated areas it already covers."""
    pixels = np.asarray(source_image).reshape(-1, 3).astype(np.float32)
    sat, val, hue = saturation_value_hue(pixels)
    vivid = (sat >= saturation) & (val >= value)
    bins = np.minimum((hue * ACCENT_HUE_BINS).astype(int), ACCENT_HUE_BINS - 1)
    minimum = max(16, int(ACCENT_MIN_SHARE * len(pixels)))
    candidates = [
        pixels[vivid & (bins == index)].mean(axis=0)
        for index in range(ACCENT_HUE_BINS) if int((vivid & (bins == index)).sum()) >= minimum
    ]
    palette, chosen = [colour for colour in base], []
    while candidates and len(chosen) < slots:
        distances = [min(float(((c - p) ** 2).sum()) for p in palette) for c in candidates]
        best = int(np.argmax(distances))
        if distances[best] == 0.0:
            break
        colour = candidates.pop(best)
        chosen.append(colour)
        palette.append(colour)
    return np.array(chosen, dtype=np.float32).reshape(-1, 3)


def learn_palette(
    source_image: np.ndarray, k: int, seed: int = 0, iterations: int = 25,
    accent_slots: int = 0, accent_saturation: float = 0.55, accent_value: float = 0.35,
) -> np.ndarray:
    """The palette of learn_palette_with_accents, without the accent flags."""
    return learn_palette_with_accents(
        source_image, k, seed, iterations, accent_slots, accent_saturation, accent_value,
    )[0]


def learn_palette_with_accents(
    source_image: np.ndarray, k: int, seed: int = 0, iterations: int = 25,
    accent_slots: int = 0, accent_saturation: float = 0.55, accent_value: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """A palette of at most k colours, (k, 3) float32, learned from one image
    and sorted from dark to light, plus a bool array marking which of its
    colours are reserved accents. Imitates: a production's fixed set of
    paints / inks, chosen once for a shot.

    accent_slots = 0: plain k-means with a fixed seed. Small areas lose
    their colour, because k-means spends its colours where the pixels are.

    accent_slots = n: n of the k colours are reserved for saturated accents
    of the image (see accent_colours); the other k - n come from k-means. A
    painter keeps a pot of the accent colour even if it is used in one spot."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError("k must be an int >= 2")
    if not isinstance(accent_slots, int) or isinstance(accent_slots, bool) or not 0 <= accent_slots < k:
        raise ValueError("accent_slots must be an int with 0 <= accent_slots < k")
    pixels = np.asarray(source_image).reshape(-1, 3).astype(np.float32)
    step = max(1, len(pixels) // 40000)
    sample = pixels[::step]
    centres = _kmeans(sample, k - accent_slots, seed, iterations)
    flags = np.zeros(len(centres), dtype=bool)
    if accent_slots:
        accents = accent_colours(source_image, centres, accent_slots, accent_saturation, accent_value)
        if len(accents) < accent_slots:  # fewer accents than slots: give the slots back to k-means
            centres = _kmeans(sample, k - len(accents), seed, iterations)
        flags = np.r_[np.zeros(len(centres), dtype=bool), np.ones(len(accents), dtype=bool)]
        centres = np.concatenate([centres, accents]) if len(accents) else centres
    order = np.argsort(centres @ np.array([0.299, 0.587, 0.114], dtype=np.float32), kind="stable")
    return centres[order], flags[order]


def _nearest(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    # argmin |p - c|^2 = argmin (|c|^2 - 2 p.c)
    return np.argmin((palette * palette).sum(axis=1)[None, :] - 2.0 * (pixels @ palette.T), axis=1)


def palette_limit(
    frames: np.ndarray, k: int, source_image: np.ndarray, seed: int = 0, strength: float = 1.0,
    accent_slots: int = 0, accent_saturation: float = 0.55, accent_value: float = 0.35,
    accent_pixel_saturation: float = 0.0,
) -> np.ndarray:
    """Limit every drawing to ONE k-colour palette learned from `source_image`
    (the shot's start frame).

    strength 1.0 replaces each pixel by its nearest palette colour; lower
    values blend toward the original (a softer, less posterised look).
    accent_slots reserves that many of the k colours for small saturated
    accents of the start frame (see learn_palette); 0 = plain k-means.
    accent_pixel_saturation: only pixels at least this saturated (HSV, 0-1)
    may take a reserved accent colour; duller pixels choose among the other
    colours. 0 = any pixel may. Without it a pale neutral (a white shoe) can
    be nearest to a pale accent and take on its hue.

    Imitates: painting with a limited, fixed set of colours, so that colour
    cannot drift from frame to frame. Limits: smooth gradients become bands;
    colours that are not in the start frame are pulled to the nearest one
    that is."""
    frames = _check(frames)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not 0.0 <= accent_pixel_saturation <= 1.0:
        raise ValueError("accent_pixel_saturation must be in [0, 1]")
    palette, is_accent = learn_palette_with_accents(
        source_image, k, seed, accent_slots=accent_slots, accent_saturation=accent_saturation,
        accent_value=accent_value,
    )
    guard = accent_pixel_saturation > 0 and is_accent.any() and not is_accent.all()
    base = palette[~is_accent]
    out = np.empty_like(frames)
    for index, frame in enumerate(frames):
        pixels = frame.reshape(-1, 3).astype(np.float32)
        quantized = palette[_nearest(pixels, palette)]
        if guard:
            dull = saturation_value_hue(pixels)[0] < accent_pixel_saturation
            quantized[dull] = base[_nearest(pixels[dull], base)]
        out[index] = _to_uint8(quantized * strength + pixels * (1.0 - strength)).reshape(frame.shape)
    return out


# --- ink ---

def ink_mask(
    frame: np.ndarray,
    *,
    weight_min: float,
    weight_max: float,
    threshold: float = 28.0,
    shadow_bias: float = 0.6,
    contrast_bias: float = 0.4,
    taper: float = 0.6,
    pressure_variation: float = 0.25,
    noise_scale: float = 96.0,
    seed: int = 0,
    drawing_id: int = 0,
) -> np.ndarray:
    """The ink layer of one drawing: (H, W) float32, 0 = paper, 1 = full ink.

    Lines are found as edges of the luminance. Their weight (thickness, in
    pixels, between weight_min and weight_max) depends on the picture:

      shadow_bias         thicker where the surroundings are dark, thinner in
                          highlights
      contrast_bias       thicker where the local contrast is strong
      taper               thinner toward the ends of a line (where the edge
                          has little support around it); 0 = no taper
      pressure_variation  slow variation of the weight across the picture,
                          from a smooth noise field of the drawing: the
                          uneven pressure of a hand (0 = none)
      noise_scale         size in pixels of that variation
    """
    if not 0 < weight_min <= weight_max:
        raise ValueError("weights must satisfy 0 < weight_min <= weight_max")
    brightness = box_blur(luma(frame), 1)
    padded = np.pad(brightness, 1, mode="edge")
    gx = (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
          - padded[:-2, :-2] - 2 * padded[1:-1, :-2] - padded[2:, :-2])
    gy = (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
          - padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:])
    magnitude = np.sqrt(gx * gx + gy * gy) / 4.0
    edge = np.clip((magnitude - threshold) / max(threshold, 1e-6), 0.0, 1.0)

    # Content-dependent weight, 0..1 before scaling to pixels.
    shadow = 1.0 - box_blur(brightness, 6) / 255.0
    contrast = np.clip(box_blur(magnitude, 4) / (4.0 * max(threshold, 1e-6)), 0.0, 1.0)
    weight = np.clip(shadow_bias * shadow + contrast_bias * contrast, 0.0, 1.0)
    if pressure_variation:
        weight = weight * (1.0 + pressure_variation * smooth_noise(
            edge.shape, noise_scale, seed, SALT_PRESSURE, drawing_id))
    if taper:
        radius = 4
        # A straight line fills about 2 / (2r+1) of the window around a point
        # on it; near a line end that support drops, and so does the weight.
        support = np.clip(box_blur(edge, radius) * (2 * radius + 1) / 2.0, 0.0, 1.0)
        weight = weight * (1.0 - taper * (1.0 - support))
    width = weight_min + (weight_max - weight_min) * np.clip(weight, 0.0, 1.0)

    # Variable-width dilation: blend between the dilations at the two whole
    # radii that bracket the local half-width.
    half = np.clip((width - 1.0) / 2.0, 0.0, None)
    top = int(np.ceil(half.max())) if half.size else 0
    levels = [edge] + [max_filter(edge, radius) for radius in range(1, top + 1)]
    lower = np.floor(half).astype(int)
    fraction = half - lower
    stack = np.stack(levels)
    rows, cols = np.indices(edge.shape)
    low = stack[np.minimum(lower, top), rows, cols]
    high = stack[np.minimum(lower + 1, top), rows, cols]
    return (low * (1.0 - fraction) + high * fraction).astype(np.float32)


def line_boil(mask: np.ndarray, px: float, seed: int = 0, drawing_id: int = 0, noise_scale: float = 96.0) -> np.ndarray:
    """Wobble of one drawing's ink layer: the lines are displaced by a smooth
    field of at most about `px` pixels. px = 0 returns the mask unchanged.

    The field belongs to the drawing (seed, drawing_id), so the lines move
    only when a new drawing starts, never inside a hold. Imitates: no two
    hand-drawn tracings of a line are identical, so lines "boil" slightly
    from drawing to drawing. Sub-pixel displacement uses a first-order
    approximation, accurate for px up to about 1.5."""
    if px < 0:
        raise ValueError("px must be >= 0")
    if px == 0:
        return mask
    dx = px * smooth_noise(mask.shape, noise_scale, seed, SALT_WOBBLE_X, drawing_id)
    dy = px * smooth_noise(mask.shape, noise_scale, seed, SALT_WOBBLE_Y, drawing_id)
    padded = np.pad(mask, 1, mode="edge")
    slope_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    slope_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    return np.clip(mask - dx * slope_x - dy * slope_y, 0.0, 1.0).astype(np.float32)


def ink_lines(
    frames: np.ndarray,
    strength: float,
    thickness: float | tuple[float, float],
    *,
    threshold: float = 28.0,
    shadow_bias: float = 0.6,
    contrast_bias: float = 0.4,
    taper: float = 0.6,
    pressure_variation: float = 0.25,
    boil_px: float = 0.0,
    noise_scale: float = 96.0,
    preserve_luminance: bool = False,
    seed: int = 0,
    ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Draw ink contours over each drawing.

    strength   how dark the ink is (0 = invisible, 1 = black)
    thickness  line weight in pixels: (min, max), or one number for both.
               Within that range the weight follows the picture; see ink_mask.
    boil_px    line wobble between drawings, OFF (0) by default; see line_boil.
    preserve_luminance
               True: after inking, the whole drawing is brightened by one
               gain so that its mean luminance is what it was before. Ink
               then adds contours without making the picture darker (a
               painter mixes the fills a little lighter because the ink will
               sit on top). False: ink only darkens.

    Imitates: inking by hand, where the line is heavier on the shadow side
    and at strong edges, lighter in the light, and tapers at its ends.
    Limits: lines come from luminance edges, so an edge between two colours
    of equal brightness gets no line, and texture (windows, hatching) gets
    lines as well. The taper is estimated from how much edge surrounds a
    point, not by tracing strokes."""
    frames = _check(frames)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    weight_min, weight_max = (thickness, thickness) if isinstance(thickness, (int, float)) else thickness
    out = np.empty_like(frames)
    for index, (frame, drawing_id) in enumerate(zip(frames, _ids(frames, ids))):
        mask = ink_mask(
            frame, weight_min=float(weight_min), weight_max=float(weight_max), threshold=threshold,
            shadow_bias=shadow_bias, contrast_bias=contrast_bias, taper=taper,
            pressure_variation=pressure_variation, noise_scale=noise_scale, seed=seed, drawing_id=drawing_id,
        )
        mask = line_boil(mask, boil_px, seed, drawing_id, noise_scale)
        inked = frame.astype(np.float32) * (1.0 - strength * mask)[..., None]
        if preserve_luminance:
            inked = _restore_mean_luminance(inked, float(luma(frame).mean()))
        out[index] = _to_uint8(inked)
    return out


def _restore_mean_luminance(image: np.ndarray, target: float) -> np.ndarray:
    """Scale a float image by one gain so that its mean luma is `target`.
    A few passes, because clipping at white takes a little back each time."""
    for _ in range(4):
        current = float(luma(np.clip(image, 0.0, 255.0)).mean())
        if current <= 0.0:
            break
        image = image * (target / current)
    return image


# --- halftone ---

def halftone(
    frames: np.ndarray,
    dot_size: float,
    threshold: float,
    *,
    strength: float = 0.5,
    variation: float = 0.15,
    preserve_tone: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Dot screen in the shadows only.

    dot_size   distance between dots in pixels (the screen is at 45 degrees)
    threshold  luminance (0-255) below which dots appear; the darker the
               area, the larger the dot
    strength   how much a dot darkens the picture
    variation  slow variation of the dot size across the picture (ink gain of
               the "plate"); one field for the whole clip, fixed to the screen
    preserve_tone
               True: the paper between the dots is lightened by exactly what
               the dots take away, so each area keeps its average tone, as a
               real halftone screen does (it renders a tone as dark dots on
               lighter ground). False: dots only darken.

    Imitates: halftone printing of comics, where shadows are built from dots.
    The screen itself is regular because a printing screen is; it does not
    move with the picture. Limits: only darkens; highlights get no dots."""
    frames = _check(frames)
    if dot_size < 2:
        raise ValueError("dot_size must be >= 2")
    if not 0 < threshold <= 255:
        raise ValueError("threshold must be in (0, 255]")
    height, width = frames.shape[1:3]
    y, x = np.indices((height, width), dtype=np.float32)
    u, v = (x + y) / np.sqrt(2.0), (x - y) / np.sqrt(2.0)
    cu, cv = (u / dot_size) % 1.0 - 0.5, (v / dot_size) % 1.0 - 0.5
    distance = np.sqrt(cu * cu + cv * cv)  # 0 at a dot centre, ~0.71 at a cell corner
    gain = 1.0 + variation * smooth_noise((height, width), 160.0, seed, SALT_HALFTONE) if variation else 1.0
    out = np.empty_like(frames)
    for index, frame in enumerate(frames):
        darkness = np.clip((threshold - luma(frame)) / threshold, 0.0, 1.0)
        radius = 0.6 * np.sqrt(darkness) * gain
        dots = np.clip((radius - distance) * dot_size + 0.5, 0.0, 1.0) * (darkness > 0)
        factor = 1.0 - strength * dots
        if preserve_tone:
            # What the dots take away from the shadow around each point
            # (measured over about one screen cell, counting shadow pixels
            # only) is given back to the shadow between the dots. Pixels
            # outside the shadow are not touched.
            shadow = (darkness > 0).astype(np.float32)
            radius_px = int(np.ceil(dot_size))
            taken = box_blur(strength * dots, radius_px) / np.maximum(box_blur(shadow, radius_px), 1e-3)
            factor = np.where(shadow > 0, factor / np.maximum(1.0 - taken, 0.05), 1.0)
        out[index] = _to_uint8(frame.astype(np.float32) * factor[..., None])
    return out


# --- grain ---

def grain(
    frames: np.ndarray,
    amount: float,
    seed: int = 0,
    *,
    shadow_gain: float = 1.0,
    ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Paper / film grain: fine monochrome noise whose strength follows the
    luminance.

    amount       standard deviation in the brightest areas (0-255 scale)
    shadow_gain  how much stronger the grain is in black than in white:
                 strength = amount * (1 + shadow_gain * darkness)

    The grain field is deterministic per drawing id. Applied to drawings, it
    changes with each new drawing and stays put during a hold (the grain of
    the sheet the drawing is on). Limits: plain Gaussian noise, no clumping
    or colour."""
    frames = _check(frames)
    if amount < 0 or shadow_gain < 0:
        raise ValueError("amount and shadow_gain must be >= 0")
    out = np.empty_like(frames)
    for index, (frame, drawing_id) in enumerate(zip(frames, _ids(frames, ids))):
        darkness = 1.0 - luma(frame) / 255.0
        noise = fine_noise(frame.shape[:2], seed, SALT_GRAIN, drawing_id) * (amount * (1.0 + shadow_gain * darkness))
        out[index] = _to_uint8(frame.astype(np.float32) + noise[..., None])
    return out


# --- misregistration ---

def misregistration(frames: np.ndarray, px: int) -> np.ndarray:
    """Shift the red channel `px` pixels to the right and the blue channel
    `px` pixels down; green stays. Imitates: colour plates printed slightly
    out of register, which leaves thin colour fringes at edges. The offset is
    constant, like one print run. Limits: whole pixels only."""
    frames = _check(frames)
    if not isinstance(px, int) or isinstance(px, bool) or px < 0:
        raise ValueError("px must be an int >= 0")
    if px == 0:
        return frames.copy()
    out = frames.copy()
    for index, frame in enumerate(frames):
        out[index, ..., 0] = shift_edge(frame[..., 0], px, 0)
        out[index, ..., 2] = shift_edge(frame[..., 2], 0, px)
    return out


# --- bloom ---

def bloom(frames: np.ndarray, threshold: float, radius: int, strength: float) -> np.ndarray:
    """Soft glow around bright areas.

    threshold  luminance (0-255) above which light "spills"
    radius     blur radius in pixels
    strength   0 = none, 1 = strong

    Imitates: the halation of bright artwork photographed on film, as in
    cel animation shot on an animation stand. Limits: only brightens; the
    blur is computed at quarter resolution."""
    frames = _check(frames)
    if not 0 <= threshold < 255 or radius < 1 or not 0 <= strength <= 1:
        raise ValueError("need 0 <= threshold < 255, radius >= 1, 0 <= strength <= 1")
    out = np.empty_like(frames)
    quarter = max(1, radius // 4)
    for index, frame in enumerate(frames):
        image = frame.astype(np.float32)
        bright = np.clip((luma(frame) - threshold) / (255.0 - threshold), 0.0, 1.0)[..., None] * image
        small = bright[::4, ::4]
        for _ in range(3):  # three box blurs approximate a Gaussian
            small = np.stack([box_blur(small[..., c], quarter) for c in range(3)], axis=-1)
        glow = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)[: image.shape[0], : image.shape[1]]
        glow = np.stack([box_blur(glow[..., c], 2) for c in range(3)], axis=-1)
        out[index] = _to_uint8(255.0 - (255.0 - image) * (255.0 - strength * glow) / 255.0)  # screen
    return out
