"""Generic backend contracts, one per capability.

A backend adapts one model/workflow. Nothing outside the adapter knows how the
model works: the runner passes a request in and gets a GenerationResult out.
Backend-specific settings travel inside request.backend_parameters.

    VideoBackend             common lifecycle, capability declaration
    ├── VideoGenerationBackend   VideoGenerationRequest -> new video
    └── VideoEditingBackend      BenchmarkRequest (source video) -> edited video

A backend supports a capability by implementing that contract. A model that
naturally does both subclasses both contracts and handles each request type
in generate(). Its `capabilities` are the union of its contracts' capabilities.
A class may narrow them explicitly, but never claim a capability whose
contract it does not implement.

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
from typing import TYPE_CHECKING, Any, ClassVar

from ai_studio.generation.capabilities import Capability
from ai_studio.generation.result import GenerationResult

if TYPE_CHECKING:
    from ai_studio.benchmarking.request import BenchmarkRequest
    from ai_studio.generation.request import VideoGenerationRequest


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


class VideoBackend(ABC):
    # Capabilities provided by this class's contract(s). Derived automatically;
    # see __init_subclass__.
    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    _contract_capability: ClassVar[Capability | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        implemented = frozenset(
            base._contract_capability
            for base in cls.__mro__
            if "_contract_capability" in vars(base) and base._contract_capability is not None
        )
        if "capabilities" in vars(cls):
            declared = frozenset(cls.capabilities)
            if not declared <= implemented:
                raise TypeError(
                    f"{cls.__name__} declares {sorted(c.value for c in declared - implemented)} "
                    "without implementing the matching backend contract"
                )
            cls.capabilities = declared
        else:
            cls.capabilities = implemented

    @classmethod
    def supports(cls, capability: Capability) -> bool:
        return capability in cls.capabilities

    @abstractmethod
    def metadata(self) -> BackendMetadata:
        """Identify the backend. Must work without the model installed."""

    @abstractmethod
    def validate_inputs(self, request: Any) -> None:
        """Raise ValueError if this backend cannot handle the request."""

    @abstractmethod
    def prepare(self) -> None:
        """Load the model. Called once before generate(); may be expensive."""

    @abstractmethod
    def generate(self, request: Any, output_dir: Path) -> GenerationResult:
        """Produce the video. It must be written inside output_dir."""


class VideoGenerationBackend(VideoBackend):
    """Text / image / references -> new video. No source video."""

    _contract_capability = Capability.VIDEO_GENERATION

    @abstractmethod
    def generate(self, request: VideoGenerationRequest, output_dir: Path) -> GenerationResult:
        ...


class VideoEditingBackend(VideoBackend):
    """Source video / references / instruction -> edited video."""

    _contract_capability = Capability.VIDEO_EDITING

    @abstractmethod
    def generate(self, request: BenchmarkRequest, output_dir: Path) -> GenerationResult:
        ...


class NotInstalledBackend(VideoBackend):
    """Base for candidate backends that are registered but not installed yet.
    Combine with the contract(s) the candidate is expected to implement."""

    _metadata: BackendMetadata

    def metadata(self) -> BackendMetadata:
        return self._metadata

    def validate_inputs(self, request: Any) -> None:
        pass

    def prepare(self) -> None:
        raise self._not_installed()

    def generate(self, request: Any, output_dir: Path) -> GenerationResult:
        raise self._not_installed()

    def _not_installed(self) -> BackendNotInstalledError:
        return BackendNotInstalledError(
            f"Backend '{self._metadata.backend_name}' ({self._metadata.model_name}) is a "
            "placeholder: its model and dependencies are not installed yet."
        )
