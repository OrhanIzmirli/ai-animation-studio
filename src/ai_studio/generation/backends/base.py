"""Generic backend contracts, one per capability.

A backend adapts one model/workflow. Nothing outside the adapter knows how the
model works: the runner passes a request in and gets a result out
(GenerationResult for video, ImageGenerationResult for images).
Backend-specific settings travel inside request.backend_parameters.

    VideoBackend             common lifecycle, capability declaration
    ├── VideoGenerationBackend   VideoGenerationRequest -> new video
    └── VideoEditingBackend      BenchmarkRequest (source video) -> edited video

Still images have the same kind of contracts beside the video ones:

    Backend
    +-- ImageGenerationBackend   ImageGenerationRequest -> new image(s)
    +-- ImageEditingBackend      ImageEditingRequest (source image) -> edited image(s)

Backend is the common root. VideoBackend is its historical name and stays as
an alias, so existing code and type hints keep working.

A backend supports a capability by implementing that contract. A model that
naturally does both subclasses both contracts and handles each request type
in generate(). Its `capabilities` are the union of its contracts' capabilities.
A class may narrow them explicitly, but never claim a capability whose
contract it does not implement.

Lifecycle, as driven by BenchmarkRunner:
    metadata()            -> identify backend, model and workflow versions
    validate_inputs(req)  -> raise ValueError if this backend cannot run req
    prepare()             -> load weights / warm up (may be expensive)
    generate(req, dir)    -> write the output (video or images) inside dir, return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ai_studio.generation.capabilities import Capability
from ai_studio.generation.image import ImageGenerationResult
from ai_studio.generation.result import GenerationResult

if TYPE_CHECKING:
    from ai_studio.benchmarking.request import BenchmarkRequest
    from ai_studio.generation.image import ImageEditingRequest, ImageGenerationRequest
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


class Backend(ABC):
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
    def generate(self, request: Any, output_dir: Path) -> GenerationResult | ImageGenerationResult:
        """Produce the output. Every file must be written inside output_dir."""


VideoBackend = Backend  # historical name of the common root


class VideoGenerationBackend(VideoBackend):
    """Text / image / references -> new video. No source video.

    Image-to-video: a request with start_frame may only run on a backend that
    sets supports_start_frame = True. Such a backend must condition frame 0 on
    that exact image and return StartFrameConditioning in its result; it must
    never quietly treat start_frame as just another reference image.
    """

    _contract_capability = Capability.VIDEO_GENERATION
    supports_start_frame: ClassVar[bool] = False

    @abstractmethod
    def generate(self, request: VideoGenerationRequest, output_dir: Path) -> GenerationResult:
        ...


class VideoEditingBackend(VideoBackend):
    """Source video / references / instruction -> edited video."""

    _contract_capability = Capability.VIDEO_EDITING

    @abstractmethod
    def generate(self, request: BenchmarkRequest, output_dir: Path) -> GenerationResult:
        ...


class ImageGenerationBackend(Backend):
    """Text / reference images -> new still image(s). No source image.

    Reference images carry identity and style. A backend declares how many it
    can really use (max_reference_images, default 0 = none); a request with
    more is refused before anything runs, it is never quietly truncated.
    The result reports the seed the model actually used (seed_used; None if
    the model has no controllable seed) and whatever provenance the model or
    service reported.
    """

    _contract_capability = Capability.IMAGE_GENERATION
    max_reference_images: ClassVar[int] = 0

    @abstractmethod
    def generate(self, request: ImageGenerationRequest, output_dir: Path) -> ImageGenerationResult:
        ...


class ImageEditingBackend(Backend):
    """Source image / reference images / instruction -> edited image(s).

    The source image is the picture that is changed in place; it is never
    just another reference. A request with a mask (the region to change) may
    only run on a backend that sets supports_mask = True.
    """

    _contract_capability = Capability.IMAGE_EDITING
    max_reference_images: ClassVar[int] = 0
    supports_mask: ClassVar[bool] = False

    @abstractmethod
    def generate(self, request: ImageEditingRequest, output_dir: Path) -> ImageGenerationResult:
        ...


class NotInstalledBackend(Backend):
    """Base for candidate backends that are registered but not installed yet.
    Combine with the contract(s) the candidate is expected to implement."""

    _metadata: BackendMetadata

    def metadata(self) -> BackendMetadata:
        return self._metadata

    def validate_inputs(self, request: Any) -> None:
        pass

    def prepare(self) -> None:
        raise self._not_installed()

    def generate(self, request: Any, output_dir: Path) -> GenerationResult | ImageGenerationResult:
        raise self._not_installed()

    def _not_installed(self) -> BackendNotInstalledError:
        return BackendNotInstalledError(
            f"Backend '{self._metadata.backend_name}' ({self._metadata.model_name}) is a "
            "placeholder: its model and dependencies are not installed yet."
        )
