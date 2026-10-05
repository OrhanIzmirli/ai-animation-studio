import json

import pytest

from ai_studio.benchmarking import SCHEMA_VERSION, BenchmarkLogger, BenchmarkResult, Resolution, new_run_id
from ai_studio.benchmarking.logger import UnreadableRecordWarning


def make_result(**overrides):
    fields = dict(
        benchmark_id="bench-001",
        run_id="run-001",
        task_type="video_editing",
        model="dummy-model",
        model_version="0.0.1",
        workflow_version="wf-1",
        seed=42,
        inference_steps=30,
        guidance=5.0,
        prompt="a cat walking",
        resolution=Resolution(width=1280, height=720),
        output_video="outputs/bench-001/run-001/out.mp4",
        fps=24,
        frame_count=81,
    )
    fields.update(overrides)
    return BenchmarkResult(**fields)


def test_log_writes_json_with_all_fields(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    path = logger.log(make_result(quality_metrics={"sharpness": 0.8}))

    assert path == tmp_path / "bench-001" / "run-001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "benchmark_id", "run_id", "model", "model_version", "workflow_version",
        "seed", "prompt", "negative_prompt", "source_video", "reference_assets",
        "resolution", "fps", "frame_count", "inference_steps", "guidance", "gpu",
        "vram_gb", "generation_time_seconds", "estimated_cost_usd",
        "quality_metrics", "human_rating", "notes", "timestamp", "schema_version",
        "model_parameters", "environment",
        "backend", "backend_version", "input_checksums",
        "status", "failure_stage", "error_type", "error_message",
        "validation_time_seconds", "prepare_time_seconds",
        "verification_time_seconds", "total_time_seconds", "output_video",
        "task_type", "task_parameters", "start_frame", "output_media", "image_output",
    }
    assert set(data) == expected_keys
    assert data["resolution"] == {"width": 1280, "height": 720}
    assert data["quality_metrics"] == {"sharpness": 0.8}
    assert data["model_parameters"] == {}
    assert data["environment"] == {}
    assert data["schema_version"] == SCHEMA_VERSION


def test_model_parameters_and_environment_roundtrip(tmp_path):
    model_parameters = {
        "sampler": "euler",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "denoise_strength": 0.75,
        "backend_specific": {"shift": 8.0, "tiles": [2, 2]},
    }
    environment = {
        "os": "Windows 11",
        "python_version": "3.11.9",
        "cuda_version": "12.4",
        "torch_version": "2.5.1",
        "gpu_driver": "560.94",
        "gpu_model": "RTX 4090",
        "vram_gb": 24,
        "git_commit": "abc1234",
        "libraries": {"diffusers": "0.31.0"},
    }
    logger = BenchmarkLogger(tmp_path)
    path = logger.log(make_result(model_parameters=model_parameters, environment=environment))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model_parameters"] == model_parameters
    assert data["environment"] == environment
    loaded = logger.load("bench-001", "run-001")
    assert loaded.model_parameters == model_parameters
    assert loaded.environment == environment


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": "42"},
        {"seed": True},
        {"model": ""},
        {"fps": 0},
        {"frame_count": 0},
        {"run_id": "../escape"},
        {"benchmark_id": "a/b"},
        {"model_parameters": None},
        {"environment": ["not", "a", "dict"]},
    ],
)
def test_invalid_values_rejected(overrides):
    with pytest.raises(ValueError):
        make_result(**overrides)


@pytest.mark.parametrize("width,height", [(0, 720), (1280, -1), (True, 720), (12.5, 720)])
def test_invalid_resolution_rejected(width, height):
    with pytest.raises(ValueError):
        Resolution(width=width, height=height)


def test_fps_and_frame_count_may_be_unknown(tmp_path):
    result = make_result(fps=None, frame_count=None)
    BenchmarkLogger(tmp_path).log(result)
    assert BenchmarkLogger(tmp_path).load("bench-001", "run-001") == result


def test_new_run_id_is_unique():
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_list_runs_skips_a_truncated_record_and_reports_it(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    good = make_result(run_id="run-001")
    whole = logger.log(make_result(run_id="run-002")).read_text(encoding="utf-8")
    logger.path_for("bench-001", "run-002").write_text(whole[: len(whole) // 2], encoding="utf-8")  # cut off mid-write
    logger.log(good)

    with pytest.warns(UnreadableRecordWarning) as caught:
        runs = logger.list_runs("bench-001")
    assert runs == [good]
    assert len(caught) == 1 and "run-002.json" in str(caught[0].message)
    with pytest.raises(ValueError):  # asking for that one record still fails loudly
        logger.load("bench-001", "run-002")


def test_list_runs_skips_a_record_from_a_newer_schema_and_reports_it(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    good = make_result(run_id="run-001")
    logger.log(good)
    newer = dict(good.to_dict(), run_id="run-003", schema_version=SCHEMA_VERSION + 1, field_from_the_future=1)
    logger.path_for("bench-001", "run-003").write_text(json.dumps(newer), encoding="utf-8")

    with pytest.warns(UnreadableRecordWarning, match="run-003.json") as caught:
        runs = logger.list_runs("bench-001")
    assert runs == [good] and len(caught) == 1


def test_list_runs_without_bad_records_warns_about_nothing(tmp_path, recwarn):
    logger = BenchmarkLogger(tmp_path)
    logger.log(make_result())
    assert len(logger.list_runs("bench-001")) == 1
    assert not [w for w in recwarn if issubclass(w.category, UnreadableRecordWarning)]
