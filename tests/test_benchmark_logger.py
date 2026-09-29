import json

import pytest

from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, Resolution, new_run_id


def make_result(**overrides):
    fields = dict(
        benchmark_id="bench-001",
        run_id="run-001",
        model="dummy-model",
        model_version="0.0.1",
        workflow_version="wf-1",
        seed=42,
        inference_steps=30,
        guidance=5.0,
        prompt="a cat walking",
        resolution=Resolution(width=1280, height=720),
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
    }
    assert set(data) == expected_keys
    assert data["resolution"] == {"width": 1280, "height": 720}
    assert data["quality_metrics"] == {"sharpness": 0.8}
    assert data["model_parameters"] == {}
    assert data["environment"] == {}
    assert data["schema_version"] == 2


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


def test_loads_schema_v1_result_without_new_fields(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    data = make_result().to_dict()
    del data["model_parameters"], data["environment"]
    data["schema_version"] = 1
    path = logger.path_for("bench-001", "run-001")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = logger.load("bench-001", "run-001")
    assert loaded.schema_version == 1
    assert loaded.model_parameters == {}
    assert loaded.environment == {}


def test_log_never_overwrites(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    path = logger.log(make_result(notes="first"))

    with pytest.raises(FileExistsError):
        logger.log(make_result(notes="second"))

    assert json.loads(path.read_text(encoding="utf-8"))["notes"] == "first"


def test_roundtrip(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    original = make_result(reference_assets=["ref/a.png"], human_rating=4.0)
    logger.log(original)

    assert logger.load("bench-001", "run-001") == original


def test_list_runs(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    logger.log(make_result(run_id="run-001"))
    logger.log(make_result(run_id="run-002"))

    assert [r.run_id for r in logger.list_runs("bench-001")] == ["run-001", "run-002"]
    assert logger.list_runs("missing") == []


def test_model_parameters_are_required():
    with pytest.raises(TypeError):
        BenchmarkResult(
            benchmark_id="b", run_id="r", model="m", model_version="1",
            workflow_version="w", prompt="p",
            resolution=Resolution(width=1, height=1), fps=24, frame_count=1,
        )  # seed, inference_steps, guidance missing


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": "42"},
        {"seed": True},
        {"model": ""},
        {"fps": 0},
        {"frame_count": 0},
        {"resolution": Resolution(width=0, height=720)},
        {"run_id": "../escape"},
        {"benchmark_id": "a/b"},
        {"model_parameters": None},
        {"environment": ["not", "a", "dict"]},
    ],
)
def test_invalid_values_rejected(overrides):
    with pytest.raises(ValueError):
        make_result(**overrides)


def test_new_run_id_is_unique():
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100
