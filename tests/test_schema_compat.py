"""Frozen record files from every schema version must keep loading.

The fixtures in tests/fixtures/schema/ are historical: never edit them.
When the schema changes, add a new fixture for the new version.
"""

import json
import shutil
from pathlib import Path

import pytest

from ai_studio.benchmarking import SCHEMA_VERSION, BenchmarkLogger, BenchmarkResult

FIXTURES = Path(__file__).parent / "fixtures" / "schema"


ALL_FIXTURES = [
    "v1", "v2", "v3", "v4_success", "v4_failed", "v5_success", "v5_failed",
    "v6_generation_success", "v6_editing_failed",
    "v7_image_to_video_success",
]


def load_fixture(name):
    return BenchmarkResult.from_dict(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


def test_current_schema_version_has_a_fixture():
    assert SCHEMA_VERSION == 7
    assert (FIXTURES / "v7_image_to_video_success.json").is_file()


def test_v1_loads():
    record = load_fixture("v1")
    assert record.schema_version == 1
    assert record.model == "legacy-model" and record.fps == 24
    # fields introduced later get their defaults
    assert record.model_parameters == {} and record.environment == {}
    assert record.backend is None and record.input_checksums == {}
    assert record.status == "success"  # v1-v3 only ever recorded successful runs
    assert record.prepare_time_seconds is None and record.total_time_seconds is None


def test_v2_loads():
    record = load_fixture("v2")
    assert record.schema_version == 2
    assert record.model_parameters == {"sampler": "euler"}
    assert record.environment == {"os": "Linux"}
    assert record.backend is None
    assert record.status == "success"


def test_v3_loads():
    record = load_fixture("v3")
    assert record.schema_version == 3
    assert record.backend == "fake"
    assert record.fps is None and record.frame_count is None
    assert record.status == "success" and record.failure_stage is None


def test_v4_success_loads():
    record = load_fixture("v4_success")
    assert record.schema_version == 4
    assert record.status == "success"
    assert record.prepare_time_seconds == 40.0
    assert record.generation_time_seconds == 312.5


def test_v4_failed_loads():
    record = load_fixture("v4_failed")
    assert record.status == "failed"
    assert record.failure_stage == "generation"
    assert record.error_type == "torch.OutOfMemoryError"


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_version_loads_through_logger(tmp_path, name):
    data = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    target = tmp_path / data["benchmark_id"] / f"{data['run_id']}.json"
    target.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{name}.json", target)

    logger = BenchmarkLogger(tmp_path)
    assert logger.load(data["benchmark_id"], data["run_id"]).run_id == data["run_id"]
    assert len(logger.list_runs(data["benchmark_id"])) == 1


@pytest.mark.parametrize("name", ["v1", "v2", "v3", "v4_success", "v4_failed"])
def test_pre_v5_records_have_no_output_video(name):
    record = load_fixture(name)
    assert record.output_video is None  # allowed: the requirement starts at v5


def test_v5_success_loads():
    record = load_fixture("v5_success")
    assert record.schema_version == 5
    assert record.output_video == "outputs/object_replacement_01/20260930T090000Z-aaaa0006/out.mp4"
    assert record.human_rating is None


def test_v5_failed_loads_with_null_output():
    record = load_fixture("v5_failed")
    assert record.status == "failed" and record.output_video is None


def test_current_success_without_output_video_rejected():
    data = json.loads((FIXTURES / "v5_success.json").read_text(encoding="utf-8"))
    data["output_video"] = None
    with pytest.raises(ValueError, match="output_video"):
        BenchmarkResult.from_dict(data)


@pytest.mark.parametrize("bad", ["", "   ", r"outputs\run\out.mp4"])
def test_current_success_with_invalid_output_video_rejected(bad):
    data = json.loads((FIXTURES / "v5_success.json").read_text(encoding="utf-8"))
    data["output_video"] = bad
    with pytest.raises(ValueError, match="output_video"):
        BenchmarkResult.from_dict(data)


@pytest.mark.parametrize("name", ALL_FIXTURES[:7])
def test_pre_v6_records_have_no_task_type(name):
    record = load_fixture(name)
    assert record.task_type is None and record.task_parameters == {}


def test_v6_generation_loads():
    record = load_fixture("v6_generation_success")
    assert record.schema_version == 6
    assert record.task_type == "video_generation"
    assert record.source_video is None
    assert record.task_parameters["duration_seconds"] == 5
    assert record.task_parameters["subject_motion"][0] == "character sits on the rooftop ledge"


def test_v6_editing_failed_loads():
    record = load_fixture("v6_editing_failed")
    assert record.task_type == "video_editing" and record.status == "failed"


def test_current_record_without_task_type_rejected():
    data = json.loads((FIXTURES / "v6_generation_success.json").read_text(encoding="utf-8"))
    data["task_type"] = None
    with pytest.raises(ValueError, match="task_type"):
        BenchmarkResult.from_dict(data)


def test_unknown_task_type_rejected():
    data = json.loads((FIXTURES / "v6_generation_success.json").read_text(encoding="utf-8"))
    data["task_type"] = "video_repair"  # not implemented yet
    with pytest.raises(ValueError, match="task_type"):
        BenchmarkResult.from_dict(data)


@pytest.mark.parametrize("name", ALL_FIXTURES[:9])
def test_pre_v7_records_have_no_start_frame(name):
    assert load_fixture(name).start_frame is None


def test_v7_image_to_video_loads():
    record = load_fixture("v7_image_to_video_success")
    assert record.schema_version == 7
    assert record.start_frame["path"] == "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
    assert record.start_frame["original_width"] == 1672
    assert record.start_frame["effective_width"] == 1280
    assert record.start_frame["path"] not in record.reference_assets


def test_current_records_roundtrip(tmp_path):
    record = load_fixture("v7_image_to_video_success")
    logger = BenchmarkLogger(tmp_path)
    logger.log(record)
    assert logger.load(record.benchmark_id, record.run_id) == record


def test_old_record_keeps_its_version_number():
    # Loading must not silently relabel an old record as current.
    assert load_fixture("v1").schema_version == 1
