"""VideoGenerationRequest: model-agnostic input for creating a new video.

No source video. Unlike BenchmarkRequest, this carries no benchmark metadata
(the runner receives the benchmark_id separately), so the same request type
can later be used by the production pipeline.

Shot controls (camera_motion, subject_motion, visual_direction) are generic
free-form descriptions: a string, a list of strings, or a JSON-like mapping.
Backends decide how to use them (fold into the prompt, map to their own
controls, or ignore them with a warning). Model-specific settings go into
backend_parameters.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ai_studio.media import Resolution
from ai_studio.generation.capabilities import Capability

ShotControl = str | list[str] | dict[str, Any] | None


def _check_json(name: str, value: Any) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be JSON-serializable: {e}") from None


def check_shot_control(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} must not be an empty string")
    elif isinstance(value, list):
        if not all(isinstance(v, str) and v.strip() for v in value):
            raise ValueError(f"{name} list must contain non-empty strings")
    elif isinstance(value, dict):
        _check_json(name, value)
    else:
        raise ValueError(f"{name} must be a string, a list of strings, a mapping or None")


@dataclass(frozen=True, kw_only=True)
class VideoGenerationRequest:
    capability: ClassVar[Capability] = Capability.VIDEO_GENERATION

    prompt: str
    duration_seconds: float
    output_resolution: Resolution
    seed: int
    fps: float | None = None  # None: backend default (the backend reports what it used)
    negative_prompt: str | None = None
    reference_images: list[str] = field(default_factory=list)
    camera_motion: ShotControl = None
    subject_motion: ShotControl = None
    visual_direction: ShotControl = None
    backend_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.negative_prompt is not None and not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be a string or None")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be a positive number")
        if self.fps is not None and (
            not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool) or self.fps <= 0
        ):
            raise ValueError("fps must be a positive number or None")
        if not isinstance(self.output_resolution, Resolution):
            raise ValueError("output_resolution must be a Resolution")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an int")
        if not isinstance(self.reference_images, list) or not all(
            isinstance(p, str) and p for p in self.reference_images
        ):
            raise ValueError("reference_images must be a list of non-empty paths")
        for name in ("camera_motion", "subject_motion", "visual_direction"):
            check_shot_control(name, getattr(self, name))
        if not isinstance(self.backend_parameters, dict):
            raise ValueError("backend_parameters must be a dict")
        _check_json("backend_parameters", self.backend_parameters)

    # --- task-request protocol used by BenchmarkRunner ---

    @property
    def input_paths(self) -> list[str]:
        return list(self.reference_images)

    def with_resolved_paths(self, resolve: Callable[[str], str]) -> VideoGenerationRequest:
        return dataclasses.replace(self, reference_images=[resolve(p) for p in self.reference_images])

    def task_parameters(self) -> dict[str, Any]:
        """Task-level request fields not covered by the generic record columns."""
        return {
            "duration_seconds": self.duration_seconds,
            "requested_fps": self.fps,
            "camera_motion": self.camera_motion,
            "subject_motion": self.subject_motion,
            "visual_direction": self.visual_direction,
        }
