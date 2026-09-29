"""Benchmark result schema.

One BenchmarkResult describes exactly one generation run. Model parameters
(model, model_version, seed, inference_steps, guidance) have no defaults: the
caller must always pass them explicitly, even when the value is None
(e.g. a backend that has no guidance scale). This keeps every run reproducible.

The generic fields (seed, inference_steps, guidance, gpu, ...) exist for
convenient querying across backends. Full, backend-specific detail goes into
two free-form dicts so the core schema never hardcodes any one model:

- model_parameters: everything needed to re-run the backend, e.g. sampler,
  scheduler, steps, cfg/guidance, denoise strength, model-specific options.
- environment: where the run happened, e.g. os, python_version, cuda_version,
  torch_version, gpu_driver, gpu_model, vram_gb, git_commit, library versions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Resolution:
    width: int
    height: int


@dataclass(frozen=True, kw_only=True)
class BenchmarkResult:
    # Identity
    benchmark_id: str
    run_id: str

    # Model parameters (required, never defaulted)
    model: str
    model_version: str
    workflow_version: str
    seed: int
    inference_steps: int | None
    guidance: float | None
    model_parameters: dict[str, Any] = field(default_factory=dict)

    # Inputs
    prompt: str
    negative_prompt: str | None = None
    source_video: str | None = None
    reference_assets: list[str] = field(default_factory=list)

    # Output format
    resolution: Resolution
    fps: float
    frame_count: int

    # Hardware and cost
    gpu: str | None = None
    vram_gb: float | None = None
    generation_time_seconds: float | None = None
    estimated_cost_usd: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    # Evaluation
    quality_metrics: dict[str, float] = field(default_factory=dict)
    human_rating: float | None = None
    notes: str = ""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "run_id", "model", "model_version", "workflow_version"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an int")
        if self.resolution.width <= 0 or self.resolution.height <= 0:
            raise ValueError("resolution must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        for name in ("benchmark_id", "run_id"):
            value = getattr(self, name)
            if any(c in value for c in '/\\:*?"<>|') or value in (".", ".."):
                raise ValueError(f"{name} must be usable as a file name: {value!r}")
        for name in ("model_parameters", "environment"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        data = dict(data)
        data["resolution"] = Resolution(**data["resolution"])
        return cls(**data)
