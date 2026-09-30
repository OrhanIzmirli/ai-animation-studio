"""GenerationResult: what any backend returns after one generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class StartFrameConditioning:
    """How a backend actually conditioned frame 0 on the request's start_frame.
    Required in the result whenever the request has a start_frame."""

    # Every transformation applied before the model saw the image, in order,
    # e.g. ["resize 1672x941 -> 1280x720 (lanczos)", "center-crop"]; [] if none.
    preprocessing: list[str]
    effective_width: int  # dimensions actually passed to the model
    effective_height: int
    derived_file: str | None = None  # preprocessed copy, if one was written (inside output_dir)

    def __post_init__(self) -> None:
        if not isinstance(self.preprocessing, list) or not all(
            isinstance(step, str) and step.strip() for step in self.preprocessing
        ):
            raise ValueError("preprocessing must be a list of non-empty strings")
        for name in ("effective_width", "effective_height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.derived_file is not None and (not isinstance(self.derived_file, str) or not self.derived_file):
            raise ValueError("derived_file must be a non-empty path or None")


@dataclass(frozen=True, kw_only=True)
class GenerationResult:
    output_video: str
    backend_name: str
    backend_version: str
    model_name: str
    model_version: str
    # The effective parameters the backend actually used (requested values
    # plus any defaults it applied). Backend-specific, never interpreted by the core.
    parameters: dict[str, Any] = field(default_factory=dict)
    # Backend-reported time; the runner records its own wall-clock measurement.
    generation_time_seconds: float | None = None
    # Backend-reported extras, e.g. library versions. Merged over the collected environment.
    environment: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Generic values for querying, reported only when the backend knows them.
    fps: float | None = None
    frame_count: int | None = None
    inference_steps: int | None = None
    guidance: float | None = None

    # Set if and only if the request had a start_frame.
    start_frame_conditioning: StartFrameConditioning | None = None

    def __post_init__(self) -> None:
        for name in ("output_video", "backend_name", "backend_version", "model_name", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("parameters", "environment"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a dict")
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be JSON-serializable: {e}") from None
        if not isinstance(self.warnings, list) or not all(isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be a list of strings")
        if self.generation_time_seconds is not None and self.generation_time_seconds < 0:
            raise ValueError("generation_time_seconds must be >= 0")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.start_frame_conditioning is not None and not isinstance(
            self.start_frame_conditioning, StartFrameConditioning
        ):
            raise ValueError("start_frame_conditioning must be a StartFrameConditioning")
