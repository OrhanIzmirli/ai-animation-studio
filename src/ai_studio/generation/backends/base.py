"""Generic video-editing backend contract.

A backend adapts one model/workflow to this interface. Nothing outside the
adapter knows how the model works: the core only passes a BenchmarkRequest in
and gets a GenerationResult out. Backend-specific settings travel inside
request.backend_parameters.

Lifecycle, as driven by BenchmarkRunner:
    metadata()            -> identify backend, model and workflow versions
    validate_inputs(req)  -> raise ValueError if this backend cannot run req
    prepare()             -> load weights / warm up (may be expensive)
    generate(req, dir)    -> write the output video inside dir, return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_studio.generation.result import GenerationResult

if TYPE_CHECKING:
    from ai_studio.benchmarking.request import BenchmarkRequest


class BackendNotInstalledError(RuntimeError):
    """The backend's model, weights or dependencies are not installed."""


@dataclass(frozen=True, kw_only=True)
class BackendMetadata:
    backend_name: str
    backend_version: str  # version of the adapter / fork, e.g. "upstream-1.0" or "fork-0.3"
    model_name: str
    model_version: str
    workflow_version: str

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


class VideoEditingBackend(ABC):
    @abstractmethod
    def metadata(self) -> BackendMetadata:
        """Identify the backend. Must work without the model installed."""

    @abstractmethod
    def validate_inputs(self, request: BenchmarkRequest) -> None:
        """Raise ValueError if this backend cannot handle the request."""

    @abstractmethod
    def prepare(self) -> None:
        """Load the model. Called once before generate(); may be expensive."""

    @abstractmethod
    def generate(self, request: BenchmarkRequest, output_dir: Path) -> GenerationResult:
        """Run the edit. The output video must be written inside output_dir."""


class NotInstalledBackend(VideoEditingBackend):
    """Base for candidate backends that are registered but not installed yet."""

    _metadata: BackendMetadata

    def metadata(self) -> BackendMetadata:
        return self._metadata

    def validate_inputs(self, request: BenchmarkRequest) -> None:
        pass

    def prepare(self) -> None:
        raise self._not_installed()

    def generate(self, request: BenchmarkRequest, output_dir: Path) -> GenerationResult:
        raise self._not_installed()

    def _not_installed(self) -> BackendNotInstalledError:
        return BackendNotInstalledError(
            f"Backend '{self._metadata.backend_name}' ({self._metadata.model_name}) is a "
            "placeholder: its model and dependencies are not installed yet."
        )
