import json
from pathlib import Path

import pytest

from ai_studio.benchmarking import SCHEMA_VERSION, BenchmarkRequest, BenchmarkRunError, Resolution
from ai_studio.generation.backends import (
    BackendMetadata,
    JoyAIVideoEditBackend,
    KiwiEditBackend,
    VideoBackend,
    VideoEditingBackend,
    VideoGenerationBackend,
    WanEditCtrlBackend,
)
from ai_studio.generation.capabilities import Capability, UnsupportedCapabilityError
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import GenerationResult
from conftest import FAKE_ENVIRONMENT, FakeBackend, FakeGenerationBackend, OutOfMemoryError, fake_video_bytes

BENCH = "comic_rooftop_01"


# --- VideoGenerationRequest ---

def test_generation_request_needs_no_source_video():
    request = VideoGenerationRequest(
        prompt="a rooftop at sunset",
        duration_seconds=5,
        output_resolution=Resolution(width=64, height=64),
        seed=1,
    )
    assert not hasattr(request, "source_video")
    assert request.input_paths == []
    assert request.fps is None and request.reference_images == []
    assert request.capability is Capability.VIDEO_GENERATION


def test_generation_request_structured_shot_controls(generation_request_factory):
    request = generation_request_factory()
    assert request.task_parameters() == {
        "duration_seconds": 5,
        "requested_fps": 24,
        "duration_tolerance_seconds": None,
        "camera_motion": "slow push-in",
        "subject_motion": ["mostly stationary", "slight head turn"],
        "visual_direction": {"style": "graphic novel", "line": "strong ink contours"},
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": " "},
        {"prompt": None},
        {"duration_seconds": 0},
        {"duration_seconds": -5},
        {"duration_seconds": "5"},
        {"duration_seconds": True},
        {"fps": 0},
        {"fps": "24"},
        {"seed": 1.0},
        {"seed": True},
        {"output_resolution": (1280, 720)},
        {"reference_images": "one.png"},
        {"negative_prompt": 3},
        {"camera_motion": ""},
        {"camera_motion": 5},
        {"subject_motion": ["ok", ""]},
        {"subject_motion": [1, 2]},
        {"visual_direction": {"x": object()}},
        {"backend_parameters": None},
        {"backend_parameters": {"x": object()}},
    ],
)
def test_generation_request_rejects_invalid(generation_request_factory, overrides):
    with pytest.raises(ValueError):
        generation_request_factory(**overrides)


def test_generation_request_has_no_model_specific_fields():
    from dataclasses import fields

    names = {f.name for f in fields(VideoGenerationRequest)}
    assert names == {
        "prompt", "negative_prompt", "start_frame", "reference_images", "duration_seconds",
        "duration_tolerance_seconds", "fps",
        "output_resolution", "seed", "backend_parameters",
        "camera_motion", "subject_motion", "visual_direction",
    }


def test_editing_request_still_requires_source_video():
    with pytest.raises(TypeError):
        BenchmarkRequest(
            benchmark_id="b", prompt="p", seed=0,
            output_resolution=Resolution(width=64, height=64),
        )
    with pytest.raises(ValueError):
        BenchmarkRequest(
            benchmark_id="b", source_video="", prompt="p", seed=0,
            output_resolution=Resolution(width=64, height=64),
        )
    assert BenchmarkRequest.capability is Capability.VIDEO_EDITING


# --- capabilities ---

def test_capability_values_are_task_types():
    assert {c.value for c in Capability} == {
        "video_generation", "video_editing", "image_generation", "image_editing",
    }


def test_contracts_declare_one_capability_each():
    assert VideoGenerationBackend.capabilities == {Capability.VIDEO_GENERATION}
    assert VideoEditingBackend.capabilities == {Capability.VIDEO_EDITING}
    assert VideoBackend.capabilities == frozenset()


def test_single_capability_backends():
    assert FakeGenerationBackend.capabilities == {Capability.VIDEO_GENERATION}
    assert FakeBackend.capabilities == {Capability.VIDEO_EDITING}
    assert FakeGenerationBackend.supports(Capability.VIDEO_GENERATION)
    assert not FakeGenerationBackend.supports(Capability.VIDEO_EDITING)


@pytest.mark.parametrize("cls", [JoyAIVideoEditBackend, KiwiEditBackend, WanEditCtrlBackend])
def test_candidate_placeholders_are_editing_only(cls):
    assert cls.capabilities == {Capability.VIDEO_EDITING}


class MultiCapabilityBackend(VideoGenerationBackend, VideoEditingBackend):
    """One model that naturally does both; dispatches on the request type."""

    def metadata(self):
        return BackendMetadata(
            backend_name="multi", backend_version="1", model_name="multi-model",
            model_version="1", workflow_version="wf-multi",
        )

    def validate_inputs(self, request):
        pass

    def prepare(self):
        pass

    def generate(self, request, output_dir):
        kind = "edit" if isinstance(request, BenchmarkRequest) else "gen"
        output = Path(output_dir) / f"{kind}.mp4"
        res = request.output_resolution
        output.write_bytes(fake_video_bytes(res.width, res.height, 24, 81))
        return GenerationResult(
            output_video=str(output), backend_name="multi", backend_version="1",
            model_name="multi-model", model_version="1",
        )


def test_multi_capability_backend_declares_both():
    assert MultiCapabilityBackend.capabilities == {
        Capability.VIDEO_GENERATION, Capability.VIDEO_EDITING,
    }


def test_multi_capability_backend_runs_both_tasks(runner, request_factory, generation_request_factory):
    backend = MultiCapabilityBackend()
    gen = runner.run(generation_request_factory(), backend, benchmark_id=BENCH, run_id="g")
    edit = runner.run(request_factory(), backend, run_id="e")
    assert gen.task_type == "video_generation" and gen.output_video.endswith("/gen.mp4")
    assert edit.task_type == "video_editing" and edit.output_video.endswith("/edit.mp4")


def test_backend_may_narrow_but_not_invent_capabilities():
    class NarrowedToGeneration(MultiCapabilityBackend):
        capabilities = frozenset({Capability.VIDEO_GENERATION})

    assert NarrowedToGeneration.capabilities == {Capability.VIDEO_GENERATION}

    with pytest.raises(TypeError, match="without implementing"):
        class ClaimsEditing(FakeGenerationBackend):
            capabilities = frozenset({Capability.VIDEO_GENERATION, Capability.VIDEO_EDITING})


# --- common runner behaviour for generation ---

def test_generation_run_records_everything(runner, generation_request_factory, input_files):
    _, reference = input_files
    backend = FakeGenerationBackend()

    record = runner.run(
        generation_request_factory(backend_parameters={"steps": 40}),
        backend, benchmark_id=BENCH, run_id="run-1", estimated_cost_usd=1.5,
    )

    data = json.loads(runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["task_type"] == "video_generation"
    assert data["start_frame"] is None  # text/reference-to-video, no start frame
    assert data["status"] == "success"
    assert data["source_video"] is None
    assert data["output_video"] == f"outputs/{BENCH}/run-1/gen.mp4"
    assert data["reference_assets"] == ["benchmarks/references/ref.png"]
    assert list(data["input_checksums"]) == ["benchmarks/references/ref.png"]
    assert data["task_parameters"] == {
        "duration_seconds": 5,
        "requested_fps": 24,
        "duration_tolerance_seconds": None,
        "camera_motion": "slow push-in",
        "subject_motion": ["mostly stationary", "slight head turn"],
        "visual_direction": {"style": "graphic novel", "line": "strong ink contours"},
        "run_purpose": "benchmark",
    }
    assert data["fps"] == 24 and data["frame_count"] == 120
    assert data["model_parameters"] == {"steps": 40, "sampler": "fake"}
    assert data["backend"] == "fake-gen" and data["workflow_version"] == "wf-gen-1"
    assert data["seed"] == 7
    assert data["estimated_cost_usd"] == 1.5
    assert data["environment"] == FAKE_ENVIRONMENT
    for stage in ("validation", "prepare", "generation", "verification", "total"):
        assert data[f"{stage}_time_seconds"] is not None
    assert backend.calls == ["validate_inputs", "prepare", "generate"]
    assert record == runner.logger.load(BENCH, "run-1")


def test_generation_backend_receives_absolute_reference_paths(runner, generation_request_factory):
    backend = FakeGenerationBackend()
    runner.run(
        generation_request_factory(reference_images=["benchmarks/references/ref.png"]),
        backend, benchmark_id=BENCH,
    )
    assert Path(backend.received.reference_images[0]).is_absolute()


def test_text_only_generation_run(runner, generation_request_factory):
    record = runner.run(generation_request_factory(reference_images=[]), FakeGenerationBackend(), benchmark_id=BENCH)
    assert record.status == "success"
    assert record.input_checksums == {} and record.reference_assets == []


def test_generation_requires_benchmark_id(runner, generation_request_factory):
    with pytest.raises(ValueError, match="benchmark_id is required"):
        runner.run(generation_request_factory(), FakeGenerationBackend())


def test_conflicting_benchmark_id_rejected(runner, request_factory):
    with pytest.raises(ValueError, match="conflicts"):
        runner.run(request_factory(), FakeBackend(), benchmark_id="something_else")


def test_editing_run_unchanged_and_tagged(runner, request_factory):
    record = runner.run(request_factory(), FakeBackend())
    assert record.task_type == "video_editing"
    assert record.task_parameters == {"run_purpose": "benchmark"}
    assert record.source_video == "benchmarks/inputs/source.mp4"


@pytest.mark.parametrize("purpose", ["benchmark", "smoke", "experiment"])
def test_run_purpose_is_recorded_for_success_and_failure(runner, generation_request_factory, purpose):
    record = runner.run(generation_request_factory(), FakeGenerationBackend(), benchmark_id=BENCH,
                        run_id="ok", purpose=purpose)
    assert record.task_parameters["run_purpose"] == purpose
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(generation_request_factory(), FakeGenerationBackend(fail_in="generate", exc=ValueError("x")),
                   benchmark_id=BENCH, run_id="bad", purpose=purpose)
    assert exc_info.value.record.task_parameters["run_purpose"] == purpose


def test_unknown_run_purpose_is_rejected_before_anything_runs(runner, generation_request_factory):
    backend = FakeGenerationBackend()
    with pytest.raises(ValueError, match="purpose must be one of"):
        runner.run(generation_request_factory(), backend, benchmark_id=BENCH, purpose="demo")
    assert backend.calls == [] and runner.logger.list_runs(BENCH) == []


@pytest.mark.parametrize("kind", ["generation_request_on_editing_backend", "editing_request_on_generation_backend"])
def test_capability_mismatch_rejected_without_record(runner, request_factory, generation_request_factory, kind):
    if kind == "generation_request_on_editing_backend":
        request, backend, kwargs = generation_request_factory(), FakeBackend(), {"benchmark_id": BENCH}
    else:
        request, backend, kwargs = request_factory(), FakeGenerationBackend(), {}
    with pytest.raises(UnsupportedCapabilityError):
        runner.run(request, backend, **kwargs)
    assert backend.calls == []
    assert not runner.outputs_dir.exists()
    assert runner.logger.list_runs(BENCH) == [] and runner.logger.list_runs("object_replacement_01") == []


# --- common failure logging for generation ---

def test_generation_oom_is_recorded(runner, generation_request_factory):
    backend = FakeGenerationBackend(fail_in="generate", exc=OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(generation_request_factory(), backend, benchmark_id=BENCH, run_id="run-1")

    saved = runner.logger.load(BENCH, "run-1")
    assert saved == exc_info.value.record
    assert saved.status == "failed" and saved.failure_stage == "generation"
    assert saved.error_type == "conftest.OutOfMemoryError"
    assert saved.task_type == "video_generation"
    assert saved.task_parameters["duration_seconds"] == 5  # request still fully recorded
    assert saved.output_video is None


def test_generation_prepare_failure_is_recorded(runner, generation_request_factory):
    backend = FakeGenerationBackend(fail_in="prepare", exc=OSError("weights missing"))
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(generation_request_factory(), backend, benchmark_id=BENCH)
    assert exc_info.value.record.failure_stage == "prepare"
    assert exc_info.value.record.task_type == "video_generation"


def test_generation_failure_never_overwrites(runner, generation_request_factory):
    runner.run(generation_request_factory(), FakeGenerationBackend(), benchmark_id=BENCH, run_id="run-1")
    before = runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8")
    backend = FakeGenerationBackend(fail_in="generate", exc=ValueError("x"))
    with pytest.raises(FileExistsError):
        runner.run(generation_request_factory(), backend, benchmark_id=BENCH, run_id="run-1")
    assert backend.calls == []
    assert runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8") == before


# --- output verification for generation ---

def test_generation_output_outside_run_dir_rejected(runner, generation_request_factory):
    backend = FakeGenerationBackend(output_name="../../escaped.mp4")
    with pytest.raises(BenchmarkRunError, match="verification") as exc_info:
        runner.run(generation_request_factory(), backend, benchmark_id=BENCH)
    assert exc_info.value.record.failure_stage == "verification"


def test_generation_missing_output_rejected(runner, generation_request_factory):
    class NoFile(FakeGenerationBackend):
        def generate(self, request, output_dir):
            result = super().generate(request, output_dir)
            Path(result.output_video).unlink()
            return result

    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(generation_request_factory(), NoFile(), benchmark_id=BENCH)
    assert exc_info.value.record.failure_stage == "verification"


def test_generation_modified_reference_rejected(runner, generation_request_factory):
    class Tamper(FakeGenerationBackend):
        def generate(self, request, output_dir):
            Path(request.reference_images[0]).write_bytes(b"tampered")
            return super().generate(request, output_dir)

    with pytest.raises(BenchmarkRunError, match="modified an input"):
        runner.run(generation_request_factory(), Tamper(), benchmark_id=BENCH)


def test_generation_missing_reference_is_validation_failure(runner, generation_request_factory):
    request = generation_request_factory(reference_images=["benchmarks/references/missing.png"])
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request, FakeGenerationBackend(), benchmark_id=BENCH)
    assert exc_info.value.record.failure_stage == "validation"
