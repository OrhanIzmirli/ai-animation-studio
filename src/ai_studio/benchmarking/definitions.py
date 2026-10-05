"""Benchmark definitions loaded from YAML (benchmarks/definitions/*.yaml).

A definition fixes WHAT is tested and declares its task_type explicitly:
  video_editing     source video + references + instruction -> BenchmarkRequest
  video_generation  prompt + references + shot controls     -> VideoGenerationRequest
  image_generation  prompt + references + descriptions      -> ImageGenerationRequest
  image_editing     source image + instruction (+ mask)     -> ImageEditingRequest
Per-run choices (seed, output resolution, fps, backend parameters) are
supplied when it is turned into a request. A pinned fps and the
duration_tolerance_seconds cannot be overridden per run.

Lifecycle: every definition has a status, "draft" (the default) or "ready".
Only ready definitions can be turned into requests or run. Changing
draft -> ready is an explicit human edit of the YAML file; readiness is never
inferred from whether input files exist.

Media paths inside a definition are relative to the PROJECT ROOT (see
ai_studio.paths), e.g. "benchmarks/inputs/clip.mp4" — never to the cwd.
They are kept in that portable form in the request; BenchmarkRunner resolves
them to absolute paths before calling a backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ai_studio.benchmarking.request import BenchmarkRequest
from ai_studio.benchmarking.schema import TASK_TYPES, Resolution, check_file_name_safe
from ai_studio.generation.capabilities import Capability
from ai_studio.generation.image import ImageEditingRequest, ImageGenerationRequest
from ai_studio.generation.request import (
    ShotControl,
    VideoGenerationRequest,
    check_duration_tolerance,
    check_shot_control,
)
from ai_studio.paths import definitions_dir, resolve_path


DEFINITION_STATUSES = ("draft", "ready")


class DraftBenchmarkError(RuntimeError):
    """The benchmark definition is still a draft and must not be run."""


EDITING_ONLY_FIELDS = ("source_video", "target_object", "preserve_regions")
GENERATION_ONLY_FIELDS = (
    "duration_seconds", "duration_tolerance_seconds", "fps", "start_frame",
    "camera_motion", "subject_motion", "visual_direction",
)
IMAGE_GENERATION_FIELDS = ("image_count", "composition", "subject_details", "visual_direction")
IMAGE_EDITING_FIELDS = ("source_image", "mask", *IMAGE_GENERATION_FIELDS)
# task_type -> the task-specific fields a definition of that type may set
TASK_FIELDS = {
    "video_editing": EDITING_ONLY_FIELDS,
    "video_generation": GENERATION_ONLY_FIELDS,
    "image_generation": IMAGE_GENERATION_FIELDS,
    "image_editing": IMAGE_EDITING_FIELDS,
}
_REQUIRED = {
    "video_editing": ("source_video", "prompt"),
    "video_generation": ("prompt", "duration_seconds"),
    "image_generation": ("prompt",),
    "image_editing": ("source_image", "prompt"),
}


@dataclass(frozen=True, kw_only=True)
class BenchmarkDefinition:
    benchmark_id: str
    task_type: str  # one of TASK_TYPES; must be explicit
    status: str = "draft"  # "draft" | "ready"; set to ready only by a human
    reference_images: list[str] = field(default_factory=list)
    prompt: str | None = None
    negative_prompt: str | None = None
    notes: str = ""

    # video_editing only
    source_video: str | None = None
    target_object: str | None = None
    preserve_regions: list[str] = field(default_factory=list)

    # video_generation only (shot controls: string, list of strings or mapping)
    duration_seconds: float | None = None
    duration_tolerance_seconds: float | None = None  # max |measured - duration_seconds|
    fps: float | None = None
    start_frame: str | None = None  # conditions frame 0; never also a reference image
    camera_motion: ShotControl = None
    subject_motion: ShotControl = None
    visual_direction: ShotControl = None  # also image tasks

    # image tasks only (descriptions: string, list of strings or mapping)
    image_count: int | None = None  # None: 1
    composition: ShotControl = None  # what the picture must contain / its layout, e.g. required views
    subject_details: ShotControl = None  # fixed details of the subject, e.g. the outfit
    source_image: str | None = None  # image_editing: the picture that is changed
    mask: str | None = None  # image_editing: region to change

    def __post_init__(self) -> None:
        check_file_name_safe("benchmark_id", self.benchmark_id)
        if self.status not in DEFINITION_STATUSES:
            raise ValueError(f"status must be one of {DEFINITION_STATUSES}, got {self.status!r}")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {self.task_type!r}")
        for name in ("reference_images", "preserve_regions"):
            if not isinstance(getattr(self, name), list):
                raise ValueError(f"{name} must be a list")
        allowed = TASK_FIELDS[self.task_type]
        not_applicable = [
            name for names in TASK_FIELDS.values() for name in names if name not in allowed
        ]
        set_fields = [
            name for name in dict.fromkeys(not_applicable) if getattr(self, name) not in (None, [])
        ]
        if set_fields:
            raise ValueError(f"{set_fields} do not apply to task_type {self.task_type!r}")
        if self.duration_seconds is not None and (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be a positive number")
        if self.fps is not None and (
            not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool) or self.fps <= 0
        ):
            raise ValueError("fps must be a positive number")
        check_duration_tolerance(self.duration_tolerance_seconds)
        if self.start_frame is not None:
            if not isinstance(self.start_frame, str) or not self.start_frame:
                raise ValueError("start_frame must be a non-empty path")
            if self.start_frame in self.reference_images:
                raise ValueError("start_frame must not also be listed in reference_images")
        for name in ("camera_motion", "subject_motion", "visual_direction", "composition", "subject_details"):
            check_shot_control(name, getattr(self, name))
        if self.image_count is not None and (
            not isinstance(self.image_count, int) or isinstance(self.image_count, bool) or self.image_count < 1
        ):
            raise ValueError("image_count must be an int >= 1")
        for name in ("source_image", "mask"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a non-empty path")
                if value in self.reference_images:
                    raise ValueError(f"{name} must not also be listed in reference_images")

    @property
    def is_complete(self) -> bool:
        """False while the definition still has placeholders instead of real inputs."""
        return all(getattr(self, name) for name in _REQUIRED[self.task_type])

    def to_request(
        self,
        *,
        seed: int,
        output_resolution: Resolution,
        backend_parameters: dict[str, Any] | None = None,
        fps: float | None = None,
    ) -> BenchmarkRequest | VideoGenerationRequest | ImageGenerationRequest | ImageEditingRequest:
        """Video editing -> BenchmarkRequest; video generation ->
        VideoGenerationRequest; image tasks -> ImageGenerationRequest /
        ImageEditingRequest (run those with
        runner.run(request, backend, benchmark_id=definition.benchmark_id)).
        output_resolution is the size of each image for image tasks."""
        if self.status != "ready":
            raise DraftBenchmarkError(
                f"benchmark {self.benchmark_id!r} is a draft; set 'status: ready' in its "
                "definition once its inputs are final (an explicit human decision)"
            )
        if not self.is_complete:
            required = " and ".join(_REQUIRED[self.task_type])
            raise ValueError(f"benchmark {self.benchmark_id!r} is incomplete: {required} are required")
        if self.task_type == Capability.VIDEO_EDITING.value:
            if fps is not None:
                raise ValueError("fps is not an editing request field")
            return BenchmarkRequest(
                benchmark_id=self.benchmark_id,
                source_video=self.source_video,
                reference_images=list(self.reference_images),
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                output_resolution=output_resolution,
                seed=seed,
                backend_parameters=dict(backend_parameters or {}),
            )
        if self.task_type in (Capability.IMAGE_GENERATION.value, Capability.IMAGE_EDITING.value):
            if fps is not None:
                raise ValueError("fps is not an image request field")
            common = dict(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                reference_images=list(self.reference_images),
                output_resolution=output_resolution,
                seed=seed,
                image_count=self.image_count or 1,
                composition=self.composition,
                subject_details=self.subject_details,
                visual_direction=self.visual_direction,
                backend_parameters=dict(backend_parameters or {}),
            )
            if self.task_type == Capability.IMAGE_GENERATION.value:
                return ImageGenerationRequest(**common)
            return ImageEditingRequest(source_image=self.source_image, mask=self.mask, **common)
        if fps is not None and self.fps is not None and fps != self.fps:
            raise ValueError(
                f"fps {fps} conflicts with the benchmark's pinned fps {self.fps}"
            )
        return VideoGenerationRequest(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            start_frame=self.start_frame,
            reference_images=list(self.reference_images),
            duration_seconds=self.duration_seconds,
            duration_tolerance_seconds=self.duration_tolerance_seconds,
            fps=self.fps if self.fps is not None else fps,
            output_resolution=output_resolution,
            seed=seed,
            camera_motion=self.camera_motion,
            subject_motion=self.subject_motion,
            visual_direction=self.visual_direction,
            backend_parameters=dict(backend_parameters or {}),
        )


def load_definition(path: str | Path) -> BenchmarkDefinition:
    """`path` may be absolute or relative to the project root."""
    path = resolve_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    if "task_type" not in data:
        raise ValueError(f"{path}: must declare task_type ({' or '.join(TASK_TYPES)})")
    allowed = {f.name for f in fields(BenchmarkDefinition)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    # YAML "key:" with no value means null; treat null lists as empty.
    for name in ("reference_images", "preserve_regions"):
        if data.get(name) is None:
            data.pop(name, None)
    if data.get("notes") is None:
        data.pop("notes", None)
    return BenchmarkDefinition(**data)


def load_definitions(directory: str | Path | None = None) -> dict[str, BenchmarkDefinition]:
    """Defaults to <project root>/benchmarks/definitions."""
    directory = resolve_path(directory) if directory is not None else definitions_dir()
    definitions: dict[str, BenchmarkDefinition] = {}
    for path in sorted(directory.glob("*.yaml")):
        definition = load_definition(path)
        if definition.benchmark_id in definitions:
            raise ValueError(f"duplicate benchmark_id {definition.benchmark_id!r} in {path}")
        definitions[definition.benchmark_id] = definition
    return definitions
