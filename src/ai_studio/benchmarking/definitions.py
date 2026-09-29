"""Benchmark definitions loaded from YAML (benchmarks/definitions/*.yaml).

A definition fixes WHAT is tested (source, references, instruction). Per-run
choices (seed, output resolution, backend parameters) are supplied when it is
turned into a BenchmarkRequest.

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
from ai_studio.benchmarking.schema import Resolution, check_file_name_safe
from ai_studio.paths import definitions_dir, resolve_path


@dataclass(frozen=True, kw_only=True)
class BenchmarkDefinition:
    benchmark_id: str
    source_video: str | None = None
    reference_images: list[str] = field(default_factory=list)
    prompt: str | None = None
    negative_prompt: str | None = None
    target_object: str | None = None
    preserve_regions: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        check_file_name_safe("benchmark_id", self.benchmark_id)
        for name in ("reference_images", "preserve_regions"):
            if not isinstance(getattr(self, name), list):
                raise ValueError(f"{name} must be a list")

    @property
    def is_complete(self) -> bool:
        """False while the definition still has placeholders instead of real inputs."""
        return bool(self.source_video and self.prompt)

    def to_request(
        self,
        *,
        seed: int,
        output_resolution: Resolution,
        backend_parameters: dict[str, Any] | None = None,
    ) -> BenchmarkRequest:
        if not self.is_complete:
            raise ValueError(
                f"benchmark {self.benchmark_id!r} is incomplete: source_video and prompt are required"
            )
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


def load_definition(path: str | Path) -> BenchmarkDefinition:
    """`path` may be absolute or relative to the project root."""
    path = resolve_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
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
