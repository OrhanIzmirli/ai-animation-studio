"""Fold a request's shot controls into the prompt text.

For backends whose model has no separate camera / motion input (Wan 2.2
TI2V-5B takes one text prompt and one start frame), the only way the shot
controls reach the model is as words. fold_shot_controls appends them to the
prompt as labelled sentences, in a fixed order, so the text sent is a pure
function of the request and a run can be regenerated from its record.

    <prompt>
    Camera: <camera_motion>.
    Subject motion: <subject_motion, joined with "; ">.
    Visual direction: <visual_direction, joined with ", ">.

A camera_motion that names a preset (CAMERA_PRESETS, case-insensitive) is
replaced by the preset's wording; any other text is passed through as given.
A mapping is written as "key: value" pairs. A control that is None is left
out.
"""

from __future__ import annotations

import json
from typing import Any

from ai_studio.generation.request import ShotControl, VideoGenerationRequest

# Plain camera words, said several ways: the model only hears the text, and
# one short phrase is easy for it to outweigh.
CAMERA_PRESETS = {
    "static": (
        "static shot, fixed locked-off camera on a tripod, the camera does not move; "
        "no pan, no tilt, no zoom, no push-in, no dolly, no handheld shake; "
        "the background and the architecture stay in place and keep their size"
    ),
}

LABELS = (
    ("camera_motion", "Camera", "; "),
    ("subject_motion", "Subject motion", "; "),
    ("visual_direction", "Visual direction", ", "),
)


def describe(value: ShotControl, separator: str) -> str:
    """One shot control as text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return separator.join(item.strip() for item in value)
    if isinstance(value, dict):
        return separator.join(
            f"{key}: {item if isinstance(item, str) else json.dumps(item)}" for key, item in value.items()
        )
    raise ValueError(f"not a shot control: {value!r}")


def fold_shot_controls(request: VideoGenerationRequest) -> str:
    """The request's prompt with its shot controls appended."""
    lines = [request.prompt.strip()]
    for name, label, separator in LABELS:
        value: Any = getattr(request, name)
        if value is None:
            continue
        text = describe(value, separator)
        if name == "camera_motion" and isinstance(value, str):
            text = CAMERA_PRESETS.get(text.lower(), text)
        lines.append(f"{label}: {text.rstrip('.')}.")
    return "\n".join(lines)
