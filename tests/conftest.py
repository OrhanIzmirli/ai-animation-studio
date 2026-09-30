from pathlib import Path

import pytest

from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRequest, BenchmarkRunner, Resolution
from ai_studio.generation.backends import BackendMetadata, VideoEditingBackend, VideoGenerationBackend
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import GenerationResult

FAKE_ENVIRONMENT = {
    "operating_system": "TestOS",
    "python_version": "3.11.9",
    "git_commit": "deadbeef",
    "git_dirty": False,
    "gpu_model": "FakeGPU 9000",
    "gpu_count": 1,
    "vram_gb": 48.0,
    "driver_version": "999.1",
    "reported_cuda_version": "13.4",
    "cuda_version": None,
    "pytorch_version": None,
}


class FakeBackend(VideoEditingBackend):
    """Model-free backend: copies the source video to the output dir."""

    def __init__(self, *, applied_defaults=None, overrides=None, warnings=None, output_name="out.mp4"):
        self.applied_defaults = applied_defaults or {}
        self.overrides = overrides or {}
        self.warnings = warnings or []
        self.output_name = output_name
        self.calls = []

    def metadata(self):
        return BackendMetadata(
            backend_name="fake",
            backend_version="1.0",
            model_name="fake-model",
            model_version="2.0",
            workflow_version="wf-fake-1",
        )

    def validate_inputs(self, request):
        self.calls.append("validate_inputs")
        if request.backend_parameters.get("reject"):
            raise ValueError("fake backend rejects this request")

    def prepare(self):
        self.calls.append("prepare")

    def generate(self, request, output_dir):
        self.calls.append("generate")
        output = Path(output_dir) / self.output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(Path(request.source_video).read_bytes())
        parameters = {**self.applied_defaults, **request.backend_parameters, **self.overrides}
        return GenerationResult(
            output_video=str(output),
            backend_name="fake",
            backend_version="1.0",
            model_name="fake-model",
            model_version="2.0",
            parameters=parameters,
            environment={"fake_lib_version": "0.1"},
            warnings=list(self.warnings),
            fps=24,
            frame_count=81,
            inference_steps=parameters.get("steps"),
            guidance=parameters.get("cfg"),
        )


class OutOfMemoryError(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError (no torch needed)."""


class FailingBackend(FakeBackend):
    def __init__(self, fail_in, exc, **kwargs):
        super().__init__(**kwargs)
        self.fail_in, self.exc = fail_in, exc

    def prepare(self):
        super().prepare()
        if self.fail_in == "prepare":
            raise self.exc

    def generate(self, request, output_dir):
        if self.fail_in == "generate":
            self.calls.append("generate")
            raise self.exc
        return super().generate(request, output_dir)


@pytest.fixture
def project(tmp_path):
    """A throwaway project root."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def input_files(project):
    source = project / "benchmarks" / "inputs" / "source.mp4"
    reference = project / "benchmarks" / "references" / "ref.png"
    source.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    source.write_bytes(b"fake video bytes")
    reference.write_bytes(b"fake image bytes")
    return source, reference


@pytest.fixture
def request_factory(input_files):
    source, reference = input_files

    def make(**overrides):
        fields = dict(
            benchmark_id="object_replacement_01",
            source_video=str(source),
            reference_images=[str(reference)],
            prompt="replace the mug with the reference teapot",
            negative_prompt="blurry",
            output_resolution=Resolution(width=1280, height=720),
            seed=1234,
            backend_parameters={},
        )
        fields.update(overrides)
        return BenchmarkRequest(**fields)

    return make


@pytest.fixture
def runner(project):
    return BenchmarkRunner(
        BenchmarkLogger(project / "benchmarks" / "results"),
        outputs_dir="outputs",  # relative: resolved against the project root
        environment_collector=lambda: dict(FAKE_ENVIRONMENT),
        root=project,
    )


# --- video generation fakes ---

class FakeGenerationBackend(VideoGenerationBackend):
    """Model-free generation backend: writes a small fake video file."""

    def __init__(self, *, fail_in=None, exc=None, output_name="gen.mp4"):
        self.fail_in, self.exc, self.output_name = fail_in, exc, output_name
        self.calls = []
        self.received = None

    def metadata(self):
        return BackendMetadata(
            backend_name="fake-gen", backend_version="1.0", model_name="fake-t2v",
            model_version="3.0", workflow_version="wf-gen-1",
        )

    def validate_inputs(self, request):
        self.calls.append("validate_inputs")

    def prepare(self):
        self.calls.append("prepare")
        if self.fail_in == "prepare":
            raise self.exc

    def generate(self, request, output_dir):
        self.calls.append("generate")
        self.received = request
        if self.fail_in == "generate":
            raise self.exc
        output = Path(output_dir) / self.output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"generated video bytes")
        fps = request.fps or 16
        return GenerationResult(
            output_video=str(output), backend_name="fake-gen", backend_version="1.0",
            model_name="fake-t2v", model_version="3.0",
            parameters={**request.backend_parameters, "sampler": "fake"},
            fps=fps, frame_count=round(request.duration_seconds * fps),
        )


@pytest.fixture
def generation_request_factory(input_files):
    _, reference = input_files

    def make(**overrides):
        fields = dict(
            prompt="a young man sits on a rooftop ledge at sunset, graphic novel style",
            negative_prompt="photorealistic",
            reference_images=[str(reference)],
            duration_seconds=5,
            fps=24,
            output_resolution=Resolution(width=1280, height=720),
            seed=7,
            camera_motion="slow push-in",
            subject_motion=["mostly stationary", "slight head turn"],
            visual_direction={"style": "graphic novel", "line": "strong ink contours"},
        )
        fields.update(overrides)
        return VideoGenerationRequest(**fields)

    return make
