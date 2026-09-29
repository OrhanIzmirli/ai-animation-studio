"""GenerationResult: what any backend returns after one generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


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
