"""Artist profiles: a named, versioned set of habits applied to every shot.

Principle: hand-made imperfection = consistent habits + content-dependent
variation, never uniform randomness. A profile fixes the habits (line weight
range, how the weight follows light and contrast, taper, wobble, how grain
answers to light, timing preferences, palette size) and one seed, the
artist's "hand". What varies from shot to shot comes from the pictures
themselves. The same profile on every shot of a trailer therefore gives one
consistent signature.

A profile is a YAML file (configs/style_recipes/<name>_v<version>.yaml):

    name: comic_finish          # a technique, never a studio or a title
    version: 1
    description: ...
    seed: 1701
    habits:                     # every habit is optional; absent = not applied
      timing: {...}
      palette: {...}
      ink: {...}
      halftone: {...}
      bloom: {...}
      grain: {...}
      misregistration: {...}

Habits are always applied in the order of HABIT_ORDER. Parameters that a
profile does not give take the defaults below; unknown habits or parameters
are rejected. A profile is immutable once videos were made with it: change a
habit = a new version = a new file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ai_studio.post import operations, timing
from ai_studio.quality.metrics import hold_pattern_periodicity, transition_diffs

# Version of the operations' definitions. 2: + accent slots, luminance / tone
# preservation. 3: + accent_pixel_saturation. Earlier profiles give the same output.
POST_VERSION = "3"
HABIT_ORDER = ("timing", "palette", "ink", "halftone", "bloom", "grain", "misregistration")
MOTION_ANALYSIS_WIDTH = 320  # motion for the timing is measured like the quality metrics

# habit -> parameter -> default (REQUIRED = the profile must give it)
REQUIRED = object()
HABIT_PARAMETERS: dict[str, dict[str, Any]] = {
    "timing": {
        "mode": "animator",  # "animator" (from measured motion) or "fixed"
        "n": 2,  # fixed mode: hold every drawing n frames
        "still_below": 1.0, "moderate_below": 3.0, "hysteresis": 0.2, "lookahead": 3,
        "rhythm_break": [3, 6],
    },
    "palette": {
        "k": REQUIRED, "strength": 1.0, "accent_slots": 0, "accent_saturation": 0.55, "accent_value": 0.35,
        "accent_pixel_saturation": 0.0,
    },
    "ink": {
        "strength": REQUIRED, "thickness": REQUIRED, "threshold": 28.0, "shadow_bias": 0.6,
        "contrast_bias": 0.4, "taper": 0.6, "pressure_variation": 0.25, "boil_px": 0.0, "noise_scale": 96.0,
        "preserve_luminance": False,
    },
    "halftone": {
        "dot_size": REQUIRED, "threshold": REQUIRED, "strength": 0.5, "variation": 0.15, "preserve_tone": False,
    },
    "bloom": {"threshold": REQUIRED, "radius": REQUIRED, "strength": REQUIRED},
    "grain": {"amount": REQUIRED, "shadow_gain": 1.0},
    "misregistration": {"px": REQUIRED},
}


@dataclass(frozen=True)
class ArtistProfile:
    name: str
    version: int
    description: str
    seed: int
    habits: dict[str, dict[str, Any]]  # resolved: every parameter present

    @property
    def profile_id(self) -> str:
        return f"{self.name}_v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "description": self.description,
            "seed": self.seed, "habits": self.habits,
        }


def parse_profile(data: Any) -> ArtistProfile:
    if not isinstance(data, dict):
        raise ValueError("a profile must be a mapping")
    unknown = set(data) - {"name", "version", "description", "seed", "habits"}
    if unknown:
        raise ValueError(f"unknown profile keys {sorted(unknown)}")
    name, version, seed = data.get("name"), data.get("version"), data.get("seed")
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ValueError("name must be letters, digits and underscores")
    for label, value in (("version", version), ("seed", seed)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be an int >= 0")
    if version < 1:
        raise ValueError("version must be >= 1")
    habits = data.get("habits")
    if not isinstance(habits, dict) or not habits:
        raise ValueError("habits must be a non-empty mapping")
    unknown = set(habits) - set(HABIT_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown habits {sorted(unknown)}; known: {list(HABIT_ORDER)}")
    resolved: dict[str, dict[str, Any]] = {}
    for habit in HABIT_ORDER:
        if habit not in habits:
            continue
        given = habits[habit] or {}
        if not isinstance(given, dict):
            raise ValueError(f"habit {habit} must be a mapping")
        allowed = HABIT_PARAMETERS[habit]
        unknown = set(given) - set(allowed)
        if unknown:
            raise ValueError(f"habit {habit}: unknown parameters {sorted(unknown)}; known: {sorted(allowed)}")
        missing = sorted(key for key, default in allowed.items() if default is REQUIRED and key not in given)
        if missing:
            raise ValueError(f"habit {habit}: missing parameters {missing}")
        resolved[habit] = {key: given.get(key, default) for key, default in allowed.items()}
    if "timing" in resolved and resolved["timing"]["mode"] not in ("animator", "fixed"):
        raise ValueError("timing.mode must be 'animator' or 'fixed'")
    return ArtistProfile(name, version, str(data.get("description") or ""), seed, resolved)


def load_profile(path: str | Path) -> ArtistProfile:
    """Read and validate a profile file. Its name must be <name>_v<version>.yaml."""
    path = Path(path)
    profile = parse_profile(yaml.safe_load(path.read_text(encoding="utf-8")))
    if path.stem != profile.profile_id:
        raise ValueError(f"{path.name}: the file must be named {profile.profile_id}.yaml")
    return profile


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_of(frames: np.ndarray) -> np.ndarray:
    """Transition differences (N-1 values, 0-255 luma) measured on frames
    reduced to about MOTION_ANALYSIS_WIDTH pixels wide by block averaging."""
    frames = np.asarray(frames)
    factor = max(1, frames.shape[2] // MOTION_ANALYSIS_WIDTH)
    height, width = (frames.shape[1] // factor) * factor, (frames.shape[2] // factor) * factor
    gray = (frames[:, :height, :width].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    small = gray.reshape(len(frames), height // factor, factor, width // factor, factor).mean(axis=(2, 4))
    return transition_diffs(small)


def choose_drawings(frames: np.ndarray, profile: ArtistProfile) -> np.ndarray:
    """The hold pattern the profile's timing habit gives for these frames."""
    habit = profile.habits.get("timing")
    if habit is None:
        return np.arange(len(frames))
    if habit["mode"] == "fixed":
        return timing.fixed_holds(len(frames), habit["n"])
    return timing.animator_holds(
        motion_of(frames), still_below=habit["still_below"], moderate_below=habit["moderate_below"],
        hysteresis=habit["hysteresis"], lookahead=habit["lookahead"],
        rhythm_break=tuple(habit["rhythm_break"]), seed=profile.seed,
    )


def apply_profile(
    frames: np.ndarray, profile: ArtistProfile, source_image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a profile to a clip.

    frames        (N, H, W, 3) uint8, the raw video
    source_image  (H, W, 3) uint8, the shot's start frame (palette source)

    Returns the styled frames (same shape) and timing information:
    {"drawings", "hold_lengths", "drawing_count", "held_frame_ratio",
     "hold_pattern_periodicity"}.

    The timing is decided first, on the raw frames. Every other habit is then
    applied once per DRAWING and the result is laid out along the hold
    pattern, so the frames of a hold are identical: nothing (ink, wobble,
    grain) changes inside a hold."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be uint8 with shape (N, H, W, 3)")
    drawings = choose_drawings(frames, profile)
    ids = np.unique(drawings)
    stack, id_list, seed = frames[ids], ids.tolist(), profile.seed
    for habit in HABIT_ORDER:
        p = profile.habits.get(habit)
        if p is None or habit == "timing":
            continue
        if habit == "palette":
            stack = operations.palette_limit(
                stack, p["k"], source_image, seed=seed, strength=p["strength"], accent_slots=p["accent_slots"],
                accent_saturation=p["accent_saturation"], accent_value=p["accent_value"],
                accent_pixel_saturation=p["accent_pixel_saturation"],
            )
        elif habit == "ink":
            thickness = p["thickness"]
            stack = operations.ink_lines(
                stack, p["strength"], tuple(thickness) if isinstance(thickness, list) else thickness,
                threshold=p["threshold"], shadow_bias=p["shadow_bias"], contrast_bias=p["contrast_bias"],
                taper=p["taper"], pressure_variation=p["pressure_variation"], boil_px=p["boil_px"],
                noise_scale=p["noise_scale"], preserve_luminance=p["preserve_luminance"], seed=seed, ids=id_list,
            )
        elif habit == "halftone":
            stack = operations.halftone(
                stack, p["dot_size"], p["threshold"], strength=p["strength"], variation=p["variation"],
                preserve_tone=p["preserve_tone"], seed=seed,
            )
        elif habit == "bloom":
            stack = operations.bloom(stack, p["threshold"], p["radius"], p["strength"])
        elif habit == "grain":
            stack = operations.grain(stack, p["amount"], seed, shadow_gain=p["shadow_gain"], ids=id_list)
        elif habit == "misregistration":
            stack = operations.misregistration(stack, p["px"])
    styled = stack[np.searchsorted(ids, drawings)]
    held = drawings[1:] == drawings[:-1]
    return styled, {
        "drawings": drawings.tolist(),
        "hold_lengths": timing.hold_lengths(drawings),
        "drawing_count": int(len(ids)),
        "held_frame_ratio": round(float(held.mean()), 6) if len(held) else 0.0,
        "hold_pattern_periodicity": (
            None if (value := hold_pattern_periodicity(held)) is None else round(value, 6)
        ),
    }
