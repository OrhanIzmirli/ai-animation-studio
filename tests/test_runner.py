import dataclasses
import json
import os
from pathlib import Path

import pytest

from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, BenchmarkRunError, BenchmarkRunner
from ai_studio.benchmarking.runner import sha256_file
from ai_studio.generation.backends import BackendNotInstalledError, KiwiEditBackend
from conftest import FAKE_ENVIRONMENT, FailingBackend, FakeBackend, OutOfMemoryError, fake_probe

BENCH = "object_replacement_01"


# --- successful runs ---

def test_runner_logs_complete_record(runner, request_factory):
    request = request_factory(backend_parameters={"steps": 30, "cfg": 5.0})

    record = runner.run(request, FakeBackend(), run_id="run-1", estimated_cost_usd=0.42)

    data = json.loads(runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert data["failure_stage"] is None and data["error_type"] is None
    assert data["backend"] == "fake" and data["backend_version"] == "1.0"
    assert data["model"] == "fake-model" and data["model_version"] == "2.0"
    assert data["workflow_version"] == "wf-fake-1"
    assert data["seed"] == 1234
    assert data["prompt"] == request.prompt
    assert data["negative_prompt"] == "blurry"
    assert data["resolution"] == {"width": 1280, "height": 720}
    assert data["fps"] == 24 and data["frame_count"] == 81
    assert data["inference_steps"] == 30 and data["guidance"] == 5.0
    assert data["estimated_cost_usd"] == 0.42
    assert data["gpu"] == "FakeGPU 9000" and data["vram_gb"] == 48.0
    assert data["environment"] == {**FAKE_ENVIRONMENT, "fake_lib_version": "0.1"}
    assert record == runner.logger.load(BENCH, "run-1")


def test_paths_recorded_project_relative_with_checksums(runner, request_factory, input_files):
    source, reference = input_files
    record = runner.run(request_factory(), FakeBackend())
    assert record.source_video == "benchmarks/inputs/source.mp4"
    assert record.reference_assets == ["benchmarks/references/ref.png"]
    assert record.input_checksums == {
        "benchmarks/inputs/source.mp4": sha256_file(source),
        "benchmarks/references/ref.png": sha256_file(reference),
    }


def test_runner_calls_lifecycle_in_order(runner, request_factory):
    backend = FakeBackend()
    runner.run(request_factory(), backend)
    assert backend.calls == ["validate_inputs", "prepare", "generate"]


def test_timing_breakdown(runner, request_factory, monkeypatch):
    # Deterministic clock: every perf_counter() call advances 1s.
    import ai_studio.benchmarking.runner as runner_module

    ticks = iter(range(1000))
    monkeypatch.setattr(runner_module.time, "perf_counter", lambda: float(next(ticks)))

    record = runner.run(request_factory(), FakeBackend())

    for name in ("validation", "prepare", "generation", "verification"):
        assert getattr(record, f"{name}_time_seconds") == 1.0, name
    assert record.total_time_seconds == 5.0
    parts = (
        record.validation_time_seconds + record.prepare_time_seconds
        + record.generation_time_seconds + record.verification_time_seconds
    )
    assert parts <= record.total_time_seconds


def test_real_clock_timings_are_consistent(runner, request_factory):
    record = runner.run(request_factory(), FakeBackend())
    stages = [
        record.validation_time_seconds, record.prepare_time_seconds,
        record.generation_time_seconds, record.verification_time_seconds,
    ]
    assert all(t is not None and t >= 0 for t in stages)
    assert sum(stages) <= record.total_time_seconds + 1e-9


def test_backend_specific_parameters_preserved(runner, request_factory):
    params = {
        "sampler": "unipc",
        "scheduler": "flow_match",
        "denoise_strength": 0.8,
        "lora": [{"name": "style_a", "weight": 0.6}],
        "some_model_only_flag": True,
    }
    backend = FakeBackend(applied_defaults={"tile_size": 512})

    record = runner.run(request_factory(backend_parameters=params), backend)

    assert record.model_parameters == {**params, "tile_size": 512}


def test_backend_parameter_changes_are_noted(runner, request_factory):
    backend = FakeBackend(overrides={"steps": 20}, warnings=["clamped steps to 20"])

    record = runner.run(request_factory(backend_parameters={"steps": 50}), backend)

    assert record.model_parameters["steps"] == 20  # effective value is recorded
    assert "requested 50, used 20" in record.notes
    assert "clamped steps to 20" in record.notes


def test_each_run_gets_its_own_record_and_output(runner, request_factory):
    a = runner.run(request_factory(), FakeBackend())
    b = runner.run(request_factory(), FakeBackend())
    assert a.run_id != b.run_id
    assert len(runner.logger.list_runs(BENCH)) == 2
    assert len(list((runner.outputs_dir / BENCH).iterdir())) == 2


# --- failed runs ---

def assert_failed(runner, exc_info, stage, error_type):
    err = exc_info.value
    assert err.stage == stage
    assert err.record is not None
    saved = runner.logger.load(BENCH, err.record.run_id)
    assert saved == err.record
    assert saved.status == "failed"
    assert saved.failure_stage == stage
    assert saved.error_type == error_type
    return saved


def test_generation_failure_is_recorded(runner, request_factory):
    backend = FailingBackend("generate", ValueError("inference exploded"))
    request = request_factory(backend_parameters={"steps": 30})

    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request, backend)

    saved = assert_failed(runner, exc_info, "generation", "ValueError")
    assert isinstance(exc_info.value.__cause__, ValueError)  # original exception chained
    assert saved.error_message == "inference exploded"
    assert saved.model_parameters == {"steps": 30}  # requested parameters kept
    assert saved.model == "fake-model" and saved.workflow_version == "wf-fake-1"
    assert saved.environment == FAKE_ENVIRONMENT
    assert saved.prepare_time_seconds is not None
    assert saved.generation_time_seconds is not None  # time until the failure
    assert saved.verification_time_seconds is None
    assert saved.total_time_seconds is not None


def test_prepare_failure_is_recorded(runner, request_factory):
    backend = FailingBackend("prepare", OSError("checkpoint file is corrupt"))

    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), backend)

    saved = assert_failed(runner, exc_info, "prepare", "OSError")
    assert backend.calls == ["validate_inputs", "prepare"]
    assert saved.prepare_time_seconds is not None
    assert saved.generation_time_seconds is None


def test_oom_like_failure_is_recorded(runner, request_factory):
    backend = FailingBackend("generate", OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))

    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), backend)

    saved = assert_failed(runner, exc_info, "generation", "conftest.OutOfMemoryError")
    assert "out of memory" in saved.error_message
    assert isinstance(exc_info.value.__cause__, OutOfMemoryError)


def test_timeout_is_recorded(runner, request_factory):
    backend = FailingBackend("generate", TimeoutError("inference exceeded 3600s"))
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), backend)
    assert_failed(runner, exc_info, "generation", "TimeoutError")


def test_missing_input_is_recorded_as_validation_failure(runner, request_factory):
    request = request_factory(reference_images=["benchmarks/references/missing.png"])
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request, FakeBackend())
    saved = assert_failed(runner, exc_info, "validation", "FileNotFoundError")
    assert saved.prepare_time_seconds is None


def test_backend_validation_failure_is_recorded(runner, request_factory):
    backend = FakeBackend()
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(backend_parameters={"reject": True}), backend)
    assert backend.calls == ["validate_inputs"]
    assert_failed(runner, exc_info, "validation", "ValueError")


def test_invalid_output_is_recorded_as_verification_failure(runner, request_factory):
    backend = FakeBackend(output_name="../../escaped.mp4")
    with pytest.raises(BenchmarkRunError, match="verification") as exc_info:
        runner.run(request_factory(), backend)
    saved = assert_failed(runner, exc_info, "verification", "RuntimeError")
    assert saved.model == "fake-model"  # the result's metadata is still captured
    assert saved.verification_time_seconds is not None


def test_wrong_result_type_is_recorded(runner, request_factory):
    class BadBackend(FakeBackend):
        def generate(self, request, output_dir):
            return {"output_video": "x.mp4"}

    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), BadBackend())
    assert_failed(runner, exc_info, "verification", "TypeError")


def test_modified_input_is_recorded(runner, request_factory):
    class TamperingBackend(FakeBackend):
        def generate(self, request, output_dir):
            result = super().generate(request, output_dir)
            Path(request.source_video).write_bytes(b"tampered")
            return result

    with pytest.raises(BenchmarkRunError, match="modified an input") as exc_info:
        runner.run(request_factory(), TamperingBackend())
    assert_failed(runner, exc_info, "verification", "RuntimeError")


def test_logging_failure_is_recorded(project, request_factory):
    # Environment that cannot be serialized: the success record fails to log;
    # the failed-run record is forced to plain JSON and does get written.
    runner = BenchmarkRunner(
        BenchmarkLogger(project / "results"), "outputs",
        environment_collector=lambda: {"handle": object()}, root=project, media_probe=fake_probe,
    )
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), FakeBackend())
    saved = assert_failed(runner, exc_info, "logging", "TypeError")
    assert saved.environment["handle"].startswith("<object object")


def test_failure_record_unsavable_still_raises(runner, request_factory, monkeypatch):
    def broken_log(record):
        raise OSError("disk full")

    monkeypatch.setattr(runner.logger, "log", broken_log)
    with pytest.raises(BenchmarkRunError, match="NOT recorded") as exc_info:
        runner.run(request_factory(), FailingBackend("generate", ValueError("boom")))
    assert exc_info.value.record is None
    assert any("disk full" in note for note in exc_info.value.__cause__.__notes__)


def test_environment_collector_failure_does_not_break_run(project, request_factory):
    def broken_collector():
        raise RuntimeError("probe crashed")

    runner = BenchmarkRunner(
        BenchmarkLogger(project / "results"), "outputs", broken_collector, root=project, media_probe=fake_probe,
    )
    record = runner.run(request_factory(), FakeBackend())
    assert record.status == "success"
    assert "probe crashed" in record.environment["collection_error"]


def test_placeholder_backend_raises_and_records_nothing(runner, request_factory):
    with pytest.raises(BackendNotInstalledError):
        runner.run(request_factory(), KiwiEditBackend())
    assert runner.logger.list_runs(BENCH) == []
    assert not runner.outputs_dir.exists()


def test_broken_metadata_still_records_failure(runner, request_factory):
    class NoMetadata(FailingBackend):
        def metadata(self):
            raise RuntimeError("metadata broken")

    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), NoMetadata("generate", ValueError("boom")))
    saved = assert_failed(runner, exc_info, "generation", "ValueError")
    assert saved.backend == "NoMetadata" and saved.model == "unknown"


# --- never overwrite ---

def test_success_never_overwrites(runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-1")
    path = runner.logger.path_for(BENCH, "run-1")
    before = path.read_text(encoding="utf-8")

    backend = FakeBackend()
    with pytest.raises(FileExistsError):
        runner.run(request_factory(prompt="different"), backend, run_id="run-1")

    assert backend.calls == []  # refused before running anything
    assert path.read_text(encoding="utf-8") == before


def test_failure_never_overwrites_existing_run(runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-1")
    path = runner.logger.path_for(BENCH, "run-1")
    before = path.read_text(encoding="utf-8")

    backend = FailingBackend("generate", ValueError("boom"))
    with pytest.raises(FileExistsError):
        runner.run(request_factory(), backend, run_id="run-1")

    assert backend.calls == []
    assert path.read_text(encoding="utf-8") == before


def test_existing_failed_run_is_not_overwritten(runner, request_factory):
    with pytest.raises(BenchmarkRunError):
        runner.run(request_factory(), FailingBackend("generate", ValueError("boom")), run_id="run-1")
    before = runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.run(request_factory(), FakeBackend(), run_id="run-1")
    assert runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8") == before


def test_failure_during_race_does_not_overwrite(runner, request_factory):
    # Another process writes the same run_id while this run is generating.
    class RacingBackend(FakeBackend):
        def generate(self, request, output_dir):
            other = BenchmarkResult(
                benchmark_id=BENCH, run_id="run-1", task_type="video_editing",
                model="other", model_version="1",
                workflow_version="w", seed=0, inference_steps=None, guidance=None,
                prompt="p", resolution=request.output_resolution,
                output_video="outputs/other.mp4",
            )
            runner.logger.log(other)
            raise ValueError("boom")

    with pytest.raises(BenchmarkRunError, match="NOT recorded"):
        runner.run(request_factory(), RacingBackend(), run_id="run-1")
    assert runner.logger.load(BENCH, "run-1").model == "other"


# --- isolation from any specific model ---

def test_runner_source_has_no_model_knowledge():
    import inspect

    import ai_studio.benchmarking.runner as runner_module

    source = inspect.getsource(runner_module).lower()
    for name in ("joyai", "kiwi", "wan", "editctrl", "vace"):
        assert name not in source


# --- output_video / deprecated human_rating ---

def test_successful_run_stores_output_video(runner, request_factory):
    record = runner.run(request_factory(), FakeBackend(), run_id="run-1")
    assert record.output_video == "outputs/object_replacement_01/run-1/out.mp4"
    data = json.loads(runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8"))
    assert data["output_video"] == "outputs/object_replacement_01/run-1/out.mp4"
    assert (runner.root / data["output_video"]).is_file()


def test_failed_run_before_output_has_null_output_video(runner, request_factory):
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), FailingBackend("generate", ValueError("boom")))
    assert exc_info.value.record.output_video is None


def test_failed_verification_keeps_offending_output_path(runner, request_factory):
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), FakeBackend(output_name="../../escaped.mp4"), run_id="run-1")
    assert exc_info.value.record.output_video == "outputs/escaped.mp4"


@pytest.mark.parametrize(
    "style",
    ["absolute", "dotted", "relative_to_root"] + (["backslashes"] if os.name == "nt" else []),
)
def test_backend_output_paths_normalize(runner, request_factory, style):
    class StyledBackend(FakeBackend):
        def generate(self, request, output_dir):
            result = super().generate(request, output_dir)
            real = Path(result.output_video)
            path = {
                "absolute": str(real),
                "dotted": str(real.parent / ".." / real.parent.name / real.name),
                "relative_to_root": real.relative_to(runner.root).as_posix(),
                "backslashes": str(real).replace("/", "\\"),
            }[style]
            return dataclasses.replace(result, output_video=path)

    record = runner.run(request_factory(), StyledBackend(), run_id="run-1")
    assert record.output_video == "outputs/object_replacement_01/run-1/out.mp4"


def test_runner_never_writes_deprecated_human_rating(runner, request_factory):
    record = runner.run(request_factory(), FakeBackend(), run_id="run-1")
    with pytest.raises(BenchmarkRunError) as exc_info:
        runner.run(request_factory(), FailingBackend("generate", ValueError("x")), run_id="run-2")

    for run_id, rec in (("run-1", record), ("run-2", exc_info.value.record)):
        assert rec.human_rating is None
        data = json.loads(runner.logger.path_for(BENCH, run_id).read_text(encoding="utf-8"))
        assert data["human_rating"] is None


def test_runner_source_never_sets_human_rating():
    import inspect

    import ai_studio.benchmarking.runner as runner_module

    assert "human_rating" not in inspect.getsource(runner_module)
