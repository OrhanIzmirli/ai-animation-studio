"""BenchmarkRequest: the backend-agnostic input to one benchmark run.

Anything model-specific (sampler, scheduler, strength, LoRAs, ...) goes into
backend_parameters, which the core never interprets.

Paths may be absolute or relative to the project root (ai_studio.paths).

ARCHITECTURE NOTE: BenchmarkRequest is a BENCHMARK-LAYER object. It mixes the
edit itself (source, references, prompt, resolution, parameters) with
experiment metadata (benchmark_id, fixed seed). The future production pipeline
will use a model-agnostic EditRequest for the edit alone; BenchmarkRequest
should then wrap an EditRequest plus experiment metadata, and backends should
accept EditRequest. Not implemented yet - see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ai_studio.benchmarking.schema import Resolution, check_file_name_safe
from ai_studio.generation.capabilities import Capability


@dataclass(frozen=True, kw_only=True)
class BenchmarkRequest:
    """Video EDITING request (source video required). For generation, see
    ai_studio.generation.request.VideoGenerationRequest."""

    capability: ClassVar[Capability] = Capability.VIDEO_EDITING

    benchmark_id: str
    source_video: str
    prompt: str
    output_resolution: Resolution
    seed: int
    reference_images: list[str] = field(default_factory=list)
    negative_prompt: str | None = None
    backend_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_file_name_safe("benchmark_id", self.benchmark_id)
        if not isinstance(self.source_video, str) or not self.source_video:
            raise ValueError("source_video must be a non-empty path")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.negative_prompt is not None and not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be a string or None")
        if not isinstance(self.output_resolution, Resolution):
            raise ValueError("output_resolution must be a Resolution")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an int")
        if not isinstance(self.reference_images, list) or not all(
            isinstance(p, str) and p for p in self.reference_images
        ):
            raise ValueError("reference_images must be a list of non-empty paths")
        if not isinstance(self.backend_parameters, dict):
            raise ValueError("backend_parameters must be a dict")
        try:
            json.dumps(self.backend_parameters)
        except (TypeError, ValueError) as e:
            raise ValueError(f"backend_parameters must be JSON-serializable: {e}") from None

    # --- task-request protocol used by BenchmarkRunner ---

    @property
    def input_paths(self) -> list[str]:
        return [self.source_video, *self.reference_images]

    def with_resolved_paths(self, resolve: Callable[[str], str]) -> BenchmarkRequest:
        return dataclasses.replace(
            self,
            source_video=resolve(self.source_video),
            reference_images=[resolve(p) for p in self.reference_images],
        )

    def task_parameters(self) -> dict[str, Any]:
        """Editing has no task-level fields beyond the generic record columns."""
        return {}
