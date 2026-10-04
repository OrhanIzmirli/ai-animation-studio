"""Animation timing: which frames get a new drawing and which hold the last.

A hold pattern is described by `drawings`: an int array with one entry per
output frame, the index of the SOURCE frame shown there. A frame is "held"
when it shows the same source frame as the frame before it. The frame count
and frame rate never change.

    on ones    0 1 2 3 4 5 ...   every frame a new drawing
    on twos    0 0 2 2 4 4 ...
    on threes  0 0 0 3 3 3 ...

Two ways to choose the pattern:

    fixed_holds     every drawing held for n frames. Mechanical: the pattern
                    is perfectly periodic.
    animator_holds  the way an animator times a shot: the hold of each
                    drawing depends on how much the picture moves there
                    (threes when nearly still, twos for moderate motion, ones
                    for fast motion), with hysteresis so that it does not
                    flip back and forth, and with the habit of breaking a
                    long even rhythm now and then. Never periodic over a clip
                    that contains holds.
"""

from __future__ import annotations

import numpy as np

from ai_studio.post.noise import SALT_TIMING, rng_for
from ai_studio.quality.metrics import hold_pattern_periodicity  # noqa: F401  (the metric for these patterns)

# Hold length per motion class.
HOLD_STILL, HOLD_MODERATE, HOLD_FAST = 3, 2, 1


def fixed_holds(frame_count: int, n: int) -> np.ndarray:
    """Every drawing held for n frames (the last one possibly shorter)."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be an int >= 1")
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")
    return (np.arange(frame_count) // n) * n


def hold_frames(frames: np.ndarray, n: int) -> np.ndarray:
    """Animate "on twos" (n=2), "on threes" (n=3), ...: every n-th frame is
    kept and shown n times. Same frame count, same frame rate."""
    frames = np.asarray(frames)
    return frames[fixed_holds(len(frames), n)]


def apply_holds(frames: np.ndarray, drawings: np.ndarray) -> np.ndarray:
    """Frames re-timed by a hold pattern (see module docstring)."""
    frames, drawings = np.asarray(frames), np.asarray(drawings)
    if len(drawings) != len(frames):
        raise ValueError("a hold pattern must have one entry per frame")
    return frames[drawings]


def hold_lengths(drawings: np.ndarray) -> list[int]:
    """Length of each hold, in order: [0,0,2,2,2,5] -> [2, 3, 1]."""
    drawings = np.asarray(drawings)
    if len(drawings) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, drawings[1:] != drawings[:-1]])
    return np.diff(np.r_[starts, len(drawings)]).astype(int).tolist()


def animator_holds(
    motion: np.ndarray,
    *,
    still_below: float,
    moderate_below: float,
    hysteresis: float = 0.2,
    lookahead: int = 3,
    rhythm_break: tuple[int, int] = (3, 6),
    seed: int = 0,
) -> np.ndarray:
    """Choose the hold of every drawing from the measured motion.

    motion          mean |difference| of each consecutive frame pair
                    (frame_count - 1 values, 0-255 luma scale, as measured by
                    ai_studio.quality.metrics.transition_diffs)
    still_below     motion below this -> held on threes
    moderate_below  motion below this -> on twos; at or above -> on ones
    hysteresis      fraction: to leave the current class the motion must pass
                    the threshold by this margin, so the timing does not flip
                    when the motion hovers around a threshold
    lookahead       how many transitions ahead are averaged when a new
                    drawing is timed
    rhythm_break    (min, max): after that many equal holds in a row (the
                    count is drawn again each time, seeded) one hold is made
                    a frame longer or shorter. An animator does not keep a
                    perfectly even rhythm, and this keeps the pattern from
                    being periodic. Not applied on ones (there is no hold).
    seed            the artist profile's seed

    Returns `drawings` (frame_count entries). Deterministic.
    """
    motion = np.asarray(motion, dtype=np.float64)
    if not 0 < still_below < moderate_below:
        raise ValueError("thresholds must satisfy 0 < still_below < moderate_below")
    if not 0 <= hysteresis < 1:
        raise ValueError("hysteresis must be in [0, 1)")
    if lookahead < 1:
        raise ValueError("lookahead must be >= 1")
    low, high = rhythm_break
    if not 2 <= low <= high:
        raise ValueError("rhythm_break must be (min, max) with 2 <= min <= max")
    frame_count = len(motion) + 1
    rng = rng_for(seed, SALT_TIMING)

    def classify(value: float, current: int | None) -> int:
        # Thresholds move away from the current class by the hysteresis margin.
        still, moderate = still_below, moderate_below
        if current == HOLD_STILL:
            still *= 1 + hysteresis
        elif current == HOLD_MODERATE:
            still *= 1 - hysteresis
            moderate *= 1 + hysteresis
        elif current == HOLD_FAST:
            moderate *= 1 - hysteresis
        if value < still:
            return HOLD_STILL
        return HOLD_MODERATE if value < moderate else HOLD_FAST

    drawings = np.empty(frame_count, dtype=np.int64)
    position, current, run, limit = 0, None, 0, int(rng.integers(low, high + 1))
    while position < frame_count:
        window = motion[position:position + lookahead]
        value = float(window.mean()) if len(window) else (float(motion[-1]) if len(motion) else 0.0)
        hold = classify(value, current)
        if hold == current:
            run += 1
        else:
            current, run = hold, 1
        if hold != HOLD_FAST and run > limit:
            # Break the even rhythm once: lean toward the neighbouring timing
            # that the motion is closer to.
            if hold == HOLD_STILL:
                hold = HOLD_MODERATE
            else:
                hold = HOLD_STILL if value < (still_below + moderate_below) / 2 else HOLD_FAST
            run, limit = 0, int(rng.integers(low, high + 1))
        hold = min(hold, frame_count - position)
        drawings[position:position + hold] = position
        position += hold
    return drawings
