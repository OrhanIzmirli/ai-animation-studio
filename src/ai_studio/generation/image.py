"""Still-image tasks: model-agnostic requests and the result of an image backend.

    ImageGenerationRequest   prompt + reference images            -> new image(s)
    ImageEditingRequest      source image + instruction (+ mask)  -> edited image(s)
    ImageGenerationResult    what an image backend returns for either

Like VideoGenerationRequest, the requests carry no benchmark metadata, so the
production pipeline can use them too (character sheets, per-shot keyframes).

source_image vs reference_images - NOT interchangeable:
  source_image      (editing) the picture that is changed in place.
  reference_images  identity / style / subject references the output must
                    stay consistent with.
  mask              (editing, optional) an image marking the region to change.
A file may not appear in more than one of these roles.

output_resolution is the requested size of EACH image. image_count images
are requested from one prompt (variations of it, not different prompts).

composition, subject_details and visual_direction are generic free-form
descriptions (a string, a list of strings or a JSON-like mapping), like the
shot controls of a video request: what the picture must contain and how it is
laid out, the fixed details of the subject, and the look. Backends decide how
to use them (fold into the prompt, or ignore them with a warning).
Model-specific settings go into backend_parameters.

Seeds: many image services have no controllable seed. A backend reports the
seed the model actually used in the result (seed_used), or None if it cannot
control or know it; the runner records that and warns when it differs from
the requested seed.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ai_studio.generation.capabilities import Capability
from ai_studio.generation.request import ShotControl, check_shot_control
from ai_studio.media import Resolution

DESCRIPTIONS = ("composition", "subject_details", "visual_direction")


def _check_json(name: str, value: Any) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be JSON-serializable: {e}") from None


def _check_common(request: Any) -> None:
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if request.negative_prompt is not None and not isinstance(request.negative_prompt, str):
        raise ValueError("negative_prompt must be a string or None")
    if not isinstance(request.output_resolution, Resolution):
        raise ValueError("output_resolution must be a Resolution")
    if not isinstance(request.seed, int) or isinstance(request.seed, bool):
        raise ValueError("seed must be an int")
    if not isinstance(request.reference_images, list) or not all(
        isinstance(p, str) and p for p in request.reference_images
    ):
        raise ValueError("reference_images must be a list of non-empty paths")
    if len(set(request.reference_images)) != len(request.reference_images):
        raise ValueError("reference_images must not list the same file twice")
    count = request.image_count
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("image_count must be an int >= 1")
    for name in DESCRIPTIONS:
        check_shot_control(name, getattr(request, name))
    if not isinstance(request.backend_parameters, dict):
        raise ValueError("backend_parameters must be a dict")
    _check_json("backend_parameters", request.backend_parameters)


@dataclass(frozen=True, kw_only=True)
class ImageGenerationRequest:
    capability: ClassVar[Capability] = Capability.IMAGE_GENERATION

    prompt: str
    output_resolution: Resolution  # size of each image
    seed: int
    negative_prompt: str | None = None
    reference_images: list[str] = field(default_factory=list)
    image_count: int = 1
    composition: ShotControl = None
    subject_details: ShotControl = None
    visual_direction: ShotControl = None
    backend_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_common(self)

    # --- task-request protocol used by BenchmarkRunner ---

    @property
    def input_paths(self) -> list[str]:
        return list(self.reference_images)

    def with_resolved_paths(self, resolve: Callable[[str], str]) -> ImageGenerationRequest:
        return dataclasses.replace(self, reference_images=[resolve(p) for p in self.reference_images])

    def task_parameters(self) -> dict[str, Any]:
        """Task-level request fields not covered by the generic record columns."""
        return {"image_count": self.image_count, **{name: getattr(self, name) for name in DESCRIPTIONS}}


@dataclass(frozen=True, kw_only=True)
class ImageEditingRequest:
    capability: ClassVar[Capability] = Capability.IMAGE_EDITING

    source_image: str  # the picture that is changed in place
    prompt: str  # the edit instruction
    output_resolution: Resolution
    seed: int
    negative_prompt: str | None = None
    mask: str | None = None  # region to change; needs a backend with supports_mask
    reference_images: list[str] = field(default_factory=list)
    image_count: int = 1
    composition: ShotControl = None
    subject_details: ShotControl = None
    visual_direction: ShotControl = None
    backend_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_common(self)
        if not isinstance(self.source_image, str) or not self.source_image:
            raise ValueError("source_image must be a non-empty path")
        if self.mask is not None and (not isinstance(self.mask, str) or not self.mask):
            raise ValueError("mask must be a non-empty path or None")
        if self.source_image in self.reference_images or self.mask in self.reference_images:
            raise ValueError(
                "source_image and mask must not also be listed in reference_images: "
                "they are not interchangeable"
            )
        if self.mask == self.source_image:
            raise ValueError("mask and source_image must be different files")

    # --- task-request protocol used by BenchmarkRunner ---

    @property
    def input_paths(self) -> list[str]:
        return [self.source_image, *([self.mask] if self.mask else []), *self.reference_images]

    def with_resolved_paths(self, resolve: Callable[[str], str]) -> ImageEditingRequest:
        return dataclasses.replace(
            self,
            source_image=resolve(self.source_image),
            mask=resolve(self.mask) if self.mask else None,
            reference_images=[resolve(p) for p in self.reference_images],
        )

    def task_parameters(self) -> dict[str, Any]:
        """Task-level request fields not covered by the generic record columns."""
        return {
            "source_image": self.source_image,
            "mask": self.mask,
            "image_count": self.image_count,
            **{name: getattr(self, name) for name in DESCRIPTIONS},
        }


@dataclass(frozen=True, kw_only=True)
class ImageGenerationResult:
    """What an image backend returns (for generation and for editing)."""

    output_images: list[str]  # in order; every file inside the run's output_dir
    backend_name: str
    backend_version: str
    model_name: str
    model_version: str
    # The effective parameters the backend actually used. Backend-specific,
    # never interpreted by the core.
    parameters: dict[str, Any] = field(default_factory=dict)
    # The seed the model actually used. None: the backend cannot control or
    # know it (the request's seed was then not honoured).
    seed_used: int | None = None
    # Whatever the model or service reported about how the images were made,
    # e.g. response id, model fingerprint, revised prompt, content-credential
    # or watermark flags, per-image details. Free-form JSON, never
    # interpreted by the core.
    provenance: dict[str, Any] = field(default_factory=dict)
    # Backend-reported time; the runner records its own wall-clock measurement.
    generation_time_seconds: float | None = None
    # Backend-reported extras, e.g. library versions. Merged over the collected environment.
    environment: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Generic values for querying, reported only when the backend knows them.
    inference_steps: int | None = None
    guidance: float | None = None

    @property
    def output_paths(self) -> list[str]:
        """Every file this result claims to have written (shared with video results)."""
        return list(self.output_images)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_images, list) or not self.output_images
            or not all(isinstance(p, str) and p for p in self.output_images)
        ):
            raise ValueError("output_images must be a non-empty list of paths")
        if len(set(self.output_images)) != len(self.output_images):
            raise ValueError("output_images must not list the same file twice")
        for name in ("backend_name", "backend_version", "model_name", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("parameters", "provenance", "environment"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a dict")
            _check_json(name, value)
        if self.seed_used is not None and (not isinstance(self.seed_used, int) or isinstance(self.seed_used, bool)):
            raise ValueError("seed_used must be an int or None")
        if not isinstance(self.warnings, list) or not all(isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be a list of strings")
        if self.generation_time_seconds is not None and self.generation_time_seconds < 0:
            raise ValueError("generation_time_seconds must be >= 0")
