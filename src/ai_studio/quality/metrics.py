"""First automatic video metrics: pure functions on frame arrays.

Model-agnostic: they look only at decoded frames. Input is a grayscale stack
(N, H, W) or an RGB stack (N, H, W, 3), uint8 or float on a 0-255 scale; RGB
is reduced to luma first. A "transition" i is the pair frame i -> frame i+1.

These are MEASUREMENTS. There are no thresholds and no pass/fail here; what
counts as too much or too little is to be calibrated from benchmark data and
human ratings (configs/quality.yaml stays without thresholds).

Every number is on the 0-255 luma scale unless stated. Definitions are
versioned: change METRICS_VERSION whenever a definition, a default or the
frame decoding changes, so old and new numbers are never mixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

METRICS_VERSION = "2"  # 2: + hold_pattern_periodicity (all version-1 metrics unchanged)

DEFAULT_PULSE_PERIOD = 4
HELD_FRAME_THRESHOLD = 0.3  # mean |diff| below this = the frame was held
# Phase means below this are treated as this value in the pulse ratio, so a
# phase with no change at all gives a large finite ratio instead of infinity.
PULSE_FLOOR = 0.01
PSNR_IDENTICAL_DB = 100.0  # reported when the two images are exactly equal
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)  # BT.601


def to_gray(frames: np.ndarray) -> np.ndarray:
    """(N, H, W) float64 luma, 0-255, from a gray or RGB stack."""
    frames = np.asarray(frames)
    if frames.ndim == 4 and frames.shape[-1] == 3:
        return frames.astype(np.float64) @ np.array(_LUMA_WEIGHTS)
    if frames.ndim != 3:
        raise ValueError(f"frames must have shape (N, H, W) or (N, H, W, 3), got {frames.shape}")
    return frames.astype(np.float64)


def transition_diffs(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference of each consecutive frame pair: N-1 values,
    0-255. Element i belongs to the transition frame i -> i+1."""
    gray = to_gray(frames)
    if len(gray) < 2:
        return np.zeros(0)
    return np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))


def motion_mean(frames: np.ndarray) -> float:
    """Average change between consecutive frames (0-255).

    Detects: frozen or almost static videos (near 0). It cannot tell wanted
    motion from unwanted change (flicker, noise): both raise it."""
    diffs = transition_diffs(frames)
    return float(diffs.mean()) if len(diffs) else 0.0


def motion_max(frames: np.ndarray) -> float:
    """Largest change between two consecutive frames (0-255).

    Detects: a sudden jump (cut, glitch frame, flash) when it is far above
    motion_mean."""
    diffs = transition_diffs(frames)
    return float(diffs.max()) if len(diffs) else 0.0


def total_change(frames: np.ndarray) -> float:
    """Mean absolute difference between the last and the first frame (0-255).

    Detects: a clip that ends where it started (near 0) versus one that really
    went somewhere. A loop or a move that returns also scores low."""
    gray = to_gray(frames)
    return float(np.abs(gray[-1] - gray[0]).mean()) if len(gray) else 0.0


def start_frame_psnr(frames: np.ndarray, reference: np.ndarray) -> float:
    """PSNR in dB between frame 0 and a reference image of the same size
    (the start frame as the model received it). Higher = closer;
    PSNR_IDENTICAL_DB when exactly equal.

    Detects: a backend that ignores, shifts, recolours or distorts the start
    frame. It is lowered by encoding loss too, so compare runs encoded the
    same way."""
    first = to_gray(np.asarray(frames)[:1])[0]
    reference = to_gray(np.asarray(reference)[None])[0]
    if first.shape != reference.shape:
        raise ValueError(f"reference has shape {reference.shape}, frame 0 has {first.shape}")
    mse = float(((first - reference) ** 2).mean())
    if mse == 0:
        return PSNR_IDENTICAL_DB
    return min(PSNR_IDENTICAL_DB, 10.0 * float(np.log10(255.0**2 / mse)))


def pulse_from_diffs(diffs: np.ndarray, period: int = DEFAULT_PULSE_PERIOD) -> dict[str, Any]:
    """temporal_pulse on already computed transition diffs."""
    if not isinstance(period, int) or period < 2:
        raise ValueError("period must be an int >= 2")
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) < period:  # not every phase has a transition
        return {"period": period, "phase_means": None, "ratio": None}
    phase_means = [float(diffs[phase::period].mean()) for phase in range(period)]
    highest, lowest = max(phase_means), min(phase_means)
    ratio = 1.0 if highest < PULSE_FLOOR else highest / max(lowest, PULSE_FLOOR)
    return {"period": period, "phase_means": phase_means, "ratio": float(ratio)}


def temporal_pulse(frames: np.ndarray, period: int = DEFAULT_PULSE_PERIOD) -> dict[str, Any]:
    """Periodic change with a fixed period P (default 4 frames).

    Transitions are grouped by phase i % P and averaged per phase:
        phase_means  P values (0-255); phase 0 = transitions 0->1, P->P+1, ...
        ratio        max phase mean / min phase mean; 1.0 = no pulse.
                     Phase means below PULSE_FLOOR count as PULSE_FLOOR.
    Both are None when the video has fewer than P transitions.

    Detects: flicker at regular chunk boundaries, e.g. a video VAE that
    decodes 4 frames per latent frame and leaves a small jump at each seam.
    Intentional animation "on twos/fours" (held drawings) also has a period
    and raises the ratio: read it together with held_frame_ratio."""
    return pulse_from_diffs(transition_diffs(frames), period)


def luminance_per_frame(frames: np.ndarray) -> np.ndarray:
    """Mean luma of each frame (0-255)."""
    gray = to_gray(frames)
    return gray.mean(axis=(1, 2)) if len(gray) else np.zeros(0)


def luminance_flicker(frames: np.ndarray) -> float:
    """Standard deviation of the per-frame mean luminance after removing its
    straight-line trend (0-255).

    Detects: brightness pumping / global flicker. A steady fade or a slow
    exposure drift is a linear trend and does not count. A non-linear but
    intended brightness change (a flash, a light switching on) does."""
    luminance = luminance_per_frame(frames)
    if len(luminance) < 3:
        return 0.0
    index = np.arange(len(luminance), dtype=np.float64)
    slope, intercept = np.polyfit(index, luminance, 1)
    return float(np.std(luminance - (slope * index + intercept)))


def held_frame_ratio(frames: np.ndarray, threshold: float = HELD_FRAME_THRESHOLD) -> float:
    """Fraction of transitions whose mean |diff| is below `threshold` (0-1).

    NOT a defect by itself: stepped / limited animation holds drawings on
    purpose. It is reported so that an evaluator can tell deliberate holds
    (regular, with clear motion between them) from a frozen video (ratio
    near 1 and motion_mean near 0)."""
    diffs = transition_diffs(frames)
    return float((diffs < threshold).mean()) if len(diffs) else 0.0


def hold_pattern_periodicity(held: np.ndarray, max_lag: int | None = None) -> float | None:
    """How mechanical a hold pattern is: the highest autocorrelation of the
    held / new-drawing sequence at any lag >= 1.

    held   one boolean per transition: True = the frame repeats the previous
           one (a hold), False = a new drawing. For a video: the transitions
           whose difference is below HELD_FRAME_THRESHOLD.

    Detects: mechanical timing, e.g. a filter that holds every second frame
    regardless of content. Hand-timed animation varies its holds with the
    action.

    1.0 = perfectly periodic (fixed twos or threes). Lower = less regular.
    None when the sequence has no variation (no holds at all, or nothing but
    holds) or is too short: there is no pattern to measure.

    Each lag is normalised by its own overlap, so a strictly periodic
    sequence scores exactly 1.0. Lags up to half the sequence are examined.
    """
    held = np.asarray(held, dtype=np.float64)
    count = len(held)
    if count < 8:
        return None
    centred = held - held.mean()
    variance = float((centred * centred).mean())
    if variance == 0.0:
        return None
    max_lag = count // 2 if max_lag is None else min(max_lag, count - 1)
    best = max(
        float((centred[:-lag] * centred[lag:]).mean()) / variance for lag in range(1, max_lag + 1)
    )
    return float(min(1.0, max(-1.0, best)))


def compute_metrics(
    frames: np.ndarray, reference: np.ndarray | None = None, *, pulse_period: int = DEFAULT_PULSE_PERIOD,
) -> dict[str, Any]:
    """Every metric of this version for one video, as a flat JSON-safe dict.
    `reference` is the start frame at the analysis size, or None."""
    gray = to_gray(frames)
    if len(gray) == 0:
        raise ValueError("no frames to measure")
    pulse = temporal_pulse(gray, pulse_period)
    luminance = luminance_per_frame(gray)

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(float(value), 6)

    return {
        "frame_count": int(len(gray)),
        "motion_mean": rounded(motion_mean(gray)),
        "motion_max": rounded(motion_max(gray)),
        "total_change": rounded(total_change(gray)),
        "start_frame_psnr": None if reference is None else rounded(start_frame_psnr(gray, reference)),
        "temporal_pulse_period": pulse["period"],
        "temporal_pulse_ratio": rounded(pulse["ratio"]),
        "temporal_pulse_phase_means": (
            None if pulse["phase_means"] is None else [rounded(v) for v in pulse["phase_means"]]
        ),
        "luminance_flicker": rounded(luminance_flicker(gray)),
        "luminance_min": rounded(luminance.min()),
        "luminance_max": rounded(luminance.max()),
        "held_frame_ratio": rounded(held_frame_ratio(gray)),
        "hold_pattern_periodicity": rounded(hold_pattern_periodicity(transition_diffs(gray) < HELD_FRAME_THRESHOLD)),
    }
