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

Schema history:
  1  initial
  2  + model_parameters, environment
  3  + backend, backend_version, input_checksums; fps/frame_count may be None
  4  + status, failure_stage, error_type, error_message (failed runs are
       recorded too); + validation/prepare/verification/total timing.
       Older files load as status="success": v1-v3 only recorded successes.
  5  + output_video: project-relative POSIX path of the generated video.
       Required for successful v5+ records; may be None for failed runs.
       Migration: none needed. v1-v4 files load with output_video=None and
       keep their original schema_version, so the requirement does not
       apply to them; they are never rewritten (results are immutable).
       human_rating is DEPRECATED (kept so v1-v4 files still load): new
       ratings go through HumanEvaluationStore, and the runner never sets it.
  6  + task_type ("video_generation" | "video_editing"), required from v6;
       + task_parameters: task-level request fields with no generic column
       (generation: duration_seconds, requested_fps, camera_motion,
       subject_motion, visual_direction). Why: generation runs have no
       source video and carry shot controls that must be recorded to be
       reproducible. Migration: none. v1-v5 files load with task_type=None
       (unknown: v1-v2 were logged by hand; v3-v5 runner records were all
       editing) and task_parameters={}.
  7  + start_frame: reproducibility block for image-to-video runs
       {path, sha256, original_width, original_height, preprocessing,
        effective_width, effective_height, derived_file}; None when the run
       had no start frame. Why: frame 0 conditioning must be pinned to an
       exact file and the backend's preprocessing must be visible for a fair
       image-to-video comparison. Migration: none; v1-v6 load with None.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_studio.media import Resolution  # re-exported: Phase 0-1A import path

SCHEMA_VERSION = 7

STATUSES = ("success", "failed")
TASK_TYPES = ("video_generation", "video_editing")  # = Capability values
START_FRAME_KEYS = (
    "path", "sha256", "original_width", "original_height",
    "preprocessing", "effective_width", "effective_height", "derived_file",
)
FAILURE_STAGES = ("validation", "prepare", "generation", "verification", "logging")

_UNSAFE_FILE_NAME_CHARS = '/\\:*?"<>|'


def check_file_name_safe(name: str, value: str) -> None:
    """Ids are used as file/directory names; reject anything that could escape."""
    if (
        not isinstance(value, str)
        or not value
        or any(c in value for c in _UNSAFE_FILE_NAME_CHARS)
        or value in (".", "..")
    ):
        raise ValueError(f"{name} must be usable as a file name: {value!r}")


@dataclass(frozen=True, kw_only=True)
class BenchmarkResult:
    # Identity
    benchmark_id: str
    run_id: str

    # Task (None only in pre-v6 records)
    task_type: str | None = None
    task_parameters: dict[str, Any] = field(default_factory=dict)

    # Outcome
    status: str = "success"
    failure_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    # Backend that ran the model (an upstream adapter or our own fork)
    backend: str | None = None
    backend_version: str | None = None

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
    # path -> sha256; proves runs being compared used identical input files
    input_checksums: dict[str, str] = field(default_factory=dict)
    # Image-to-video only; see START_FRAME_KEYS. Not listed in reference_assets.
    start_frame: dict[str, Any] | None = None

    # Output. Project-relative POSIX path when inside the project root.
    output_video: str | None = None
    resolution: Resolution
    fps: float | None = None  # None when unknown
    frame_count: int | None = None  # None when unknown

    # Hardware and cost
    gpu: str | None = None
    vram_gb: float | None = None
    # Timing breakdown. prepare = cold start (model load); generation = inference.
    validation_time_seconds: float | None = None
    prepare_time_seconds: float | None = None
    generation_time_seconds: float | None = None
    verification_time_seconds: float | None = None
    total_time_seconds: float | None = None
    estimated_cost_usd: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    # Evaluation
    quality_metrics: dict[str, float] = field(default_factory=dict)
    # DEPRECATED since v5 - never written by the runner. Human ratings live in
    # benchmarks/evaluations/ via HumanEvaluationStore. Kept only so v1-v4
    # records still load.
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
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        for name in ("benchmark_id", "run_id"):
            check_file_name_safe(name, getattr(self, name))
        if self.task_type is not None and self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}")
        if self.schema_version >= 6 and self.task_type is None:
            raise ValueError("task_type is required")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if self.status == "failed":
            if self.failure_stage not in FAILURE_STAGES:
                raise ValueError(f"failed run needs failure_stage in {FAILURE_STAGES}")
            if not self.error_type:
                raise ValueError("failed run needs error_type")
        elif any((self.failure_stage, self.error_type, self.error_message)):
            raise ValueError("successful run must not have failure fields")
        elif self.schema_version >= 5 and not self.output_video:
            raise ValueError("successful run must have output_video")
        if self.output_video is not None:
            if not isinstance(self.output_video, str) or not self.output_video.strip():
                raise ValueError("output_video must be a non-empty string or None")
            if "\\" in self.output_video:
                raise ValueError(f"output_video must use forward slashes: {self.output_video!r}")
        for name in (
            "validation_time_seconds", "prepare_time_seconds", "generation_time_seconds",
            "verification_time_seconds", "total_time_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.start_frame is not None:
            if not isinstance(self.start_frame, dict) or set(self.start_frame) != set(START_FRAME_KEYS):
                raise ValueError(f"start_frame must be a dict with keys {START_FRAME_KEYS}")
            if not self.start_frame["path"]:
                raise ValueError("start_frame needs a path")
            if self.status == "success" and (
                not self.start_frame["sha256"] or self.start_frame["effective_width"] is None
            ):
                raise ValueError("successful start-frame run must record sha256 and effective dimensions")
        for name in ("model_parameters", "environment", "input_checksums", "task_parameters"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        data = dict(data)
        data["resolution"] = Resolution(**data["resolution"])
        return cls(**data)
