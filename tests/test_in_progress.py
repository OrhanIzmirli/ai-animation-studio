"""In-progress markers and abandoned runs: the runner's marker lifecycle,
finding stale markers, and recording an abandoned run as failed."""

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import SCHEMA_VERSION, BenchmarkResult, BenchmarkRunError
from ai_studio.benchmarking.in_progress import (
    ABANDONED_ERROR_TYPE,
    ABANDONED_STAGE,
    DEFAULT_STALE_HOURS,
    MARKER_NAME,
    find_abandoned,
    find_unmarked,
    marker_path,
    record_abandoned,
    remove_marker,
    write_marker,
)
from ai_studio.benchmarking.schema import FAILURE_STAGES
from ai_studio.generation.backends import BackendNotInstalledError
from conftest import FailingBackend, FakeBackend

BENCH = "object_replacement_01"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script():
    spec = importlib.util.spec_from_file_location("script_find_abandoned_runs", SCRIPTS / "find_abandoned_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def later(hours):
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def abandon(runner, request, run_id="run-dead"):
    """A run whose process 'dies' during generation: Ctrl-C is not recorded."""
    with pytest.raises(KeyboardInterrupt):
        runner.run(request, FailingBackend("generate", KeyboardInterrupt()), run_id=run_id)
    return runner.outputs_dir / BENCH / run_id


class MarkerSpy(FakeBackend):
    """Looks at the output directory at the moment generation starts."""

    def generate(self, request, output_dir):
        self.marker = json.loads(marker_path(output_dir).read_text(encoding="utf-8"))
        return super().generate(request, output_dir)


# --- the runner's marker lifecycle ---

def test_marker_exists_during_generation_and_is_gone_after_the_record(runner, request_factory):
    backend = MarkerSpy()
    record = runner.run(
        request_factory(backend_parameters={"steps": 30}), backend, run_id="run-1", notes="first",
    )
    output_dir = runner.outputs_dir / BENCH / "run-1"

    marker = backend.marker
    assert marker["marker_version"] == 1
    assert (marker["benchmark_id"], marker["run_id"]) == (BENCH, "run-1")
    assert marker["pid"] == os.getpid() and marker["hostname"]
    assert datetime.fromisoformat(marker["started_at"]).tzinfo is not None
    planned = BenchmarkResult.from_dict(marker["record"])  # a complete, valid record
    assert planned.status == "failed" and planned.failure_stage == ABANDONED_STAGE
    assert planned.error_type == ABANDONED_ERROR_TYPE
    assert planned.output_video is None
    for name in ("seed", "prompt", "negative_prompt", "source_video", "reference_assets",
                 "input_checksums", "resolution", "backend", "model", "model_version", "task_type"):
        assert getattr(planned, name) == getattr(record, name), name
    assert planned.model_parameters == {"steps": 30} and planned.notes == "first"

    assert record.status == "success"
    assert not marker_path(output_dir).exists()
    assert sorted(p.name for p in output_dir.iterdir()) == ["out.mp4"]


def test_marker_is_removed_when_a_failed_record_is_saved(runner, request_factory):
    with pytest.raises(BenchmarkRunError) as info:
        runner.run(request_factory(), FailingBackend("generate", RuntimeError("boom")), run_id="run-f")
    assert info.value.record.failure_stage == "generation"
    output_dir = runner.outputs_dir / BENCH / "run-f"
    assert output_dir.is_dir() and not marker_path(output_dir).exists()


def test_no_marker_and_no_folder_when_the_run_fails_before_generation(runner, request_factory):
    with pytest.raises(BenchmarkRunError):
        runner.run(request_factory(), FailingBackend("prepare", RuntimeError("no gpu")), run_id="run-p")
    assert not (runner.outputs_dir / BENCH / "run-p").exists()


def test_marker_stays_when_the_process_dies_during_generation(runner, request_factory):
    output_dir = abandon(runner, request_factory())
    assert marker_path(output_dir).is_file()
    assert not runner.logger.path_for(BENCH, "run-dead").exists()


def test_marker_stays_when_no_record_is_written_for_a_setup_problem(runner, request_factory):
    # BackendNotInstalledError is raised as-is and never recorded.
    with pytest.raises(BackendNotInstalledError):
        runner.run(request_factory(), FailingBackend("generate", BackendNotInstalledError("x")), run_id="run-s")
    assert marker_path(runner.outputs_dir / BENCH / "run-s").is_file()


def test_marker_stays_when_the_failed_record_cannot_be_saved(runner, request_factory, monkeypatch):
    def refuse(result):
        raise OSError("disk full")

    monkeypatch.setattr(runner.logger, "log", refuse)
    with pytest.raises(BenchmarkRunError) as info:
        runner.run(request_factory(), FailingBackend("generate", RuntimeError("boom")), run_id="run-x")
    assert info.value.record is None
    assert marker_path(runner.outputs_dir / BENCH / "run-x").is_file()


def test_marker_does_not_collect_the_environment(project, request_factory):
    from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRunner
    from conftest import FAKE_ENVIRONMENT, fake_probe

    calls = []

    def collector():
        calls.append(1)
        return dict(FAKE_ENVIRONMENT)

    runner = BenchmarkRunner(
        BenchmarkLogger(project / "benchmarks" / "results"), outputs_dir="outputs",
        environment_collector=collector, root=project, media_probe=fake_probe,
    )
    backend = MarkerSpy()
    runner.run(request_factory(), backend, run_id="run-1")
    assert len(calls) == 1  # once, for the real record
    assert backend.marker["record"]["environment"] == {}


def test_write_marker_refuses_other_records_and_never_overwrites(runner, request_factory, tmp_path):
    record = runner.run(request_factory(), FakeBackend(), run_id="run-1")
    with pytest.raises(ValueError, match="abandoned"):
        write_marker(tmp_path, record)
    output_dir = abandon(runner, request_factory())
    planned = BenchmarkResult.from_dict(json.loads(marker_path(output_dir).read_text(encoding="utf-8"))["record"])
    with pytest.raises(FileExistsError):
        write_marker(output_dir, planned)


def test_remove_marker_touches_only_the_marker(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    remove_marker(tmp_path)  # no marker: nothing happens
    marker_path(tmp_path).write_text("{}", encoding="utf-8")
    remove_marker(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["out.mp4"]


# --- finding abandoned runs ---

def test_fresh_marker_is_not_abandoned_but_a_stale_one_is(runner, request_factory):
    output_dir = abandon(runner, request_factory())
    assert find_abandoned(runner.outputs_dir, runner.logger) == []  # may still be running
    found = find_abandoned(runner.outputs_dir, runner.logger, now=later(DEFAULT_STALE_HOURS + 1))
    assert [(run.benchmark_id, run.run_id, run.output_dir) for run in found] == [(BENCH, "run-dead", output_dir)]
    assert found[0].problem is None and DEFAULT_STALE_HOURS < found[0].age_hours < DEFAULT_STALE_HOURS + 2
    assert len(find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)) == 1


def test_recorded_runs_are_never_listed(runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-ok")
    with pytest.raises(BenchmarkRunError):
        runner.run(request_factory(), FailingBackend("generate", RuntimeError("boom")), run_id="run-f")
    # a leftover marker next to an existing record does not make the run abandoned
    marker_path(runner.outputs_dir / BENCH / "run-f").write_text("{}", encoding="utf-8")
    assert find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0) == []


def test_find_filters_by_benchmark_and_sorts_oldest_first(runner, request_factory):
    abandon(runner, request_factory(), "run-a")
    abandon(runner, request_factory(benchmark_id="other_01"), "run-b")
    abandon(runner, request_factory(), "run-c")
    everything = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)
    assert [run.run_id for run in everything] == ["run-a", "run-b", "run-c"]
    only = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0, benchmark="other_01")
    assert [run.run_id for run in only] == ["run-b"]
    assert find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0, benchmark="missing") == []
    with pytest.raises(ValueError):
        find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=-1)


def test_missing_outputs_folder_means_nothing_found(runner, project):
    assert find_abandoned(project / "outputs", runner.logger, stale_after_hours=0) == []
    assert find_unmarked(project / "outputs", runner.logger) == []


@pytest.mark.parametrize("text, problem", [
    ("{not json", "not readable JSON"),
    ("[]", "no record"),
    ('{"marker_version": 99, "record": {}}', "unknown marker_version"),
])
def test_broken_marker_is_listed_but_cannot_be_recorded(runner, request_factory, text, problem):
    output_dir = abandon(runner, request_factory())
    marker_path(output_dir).write_text(text, encoding="utf-8")
    (run,) = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)
    assert run.marker is None and problem in run.problem
    with pytest.raises(ValueError, match="run-dead"):
        record_abandoned(run, runner.logger)
    assert not runner.logger.path_for(BENCH, "run-dead").exists()


def test_marker_of_another_run_is_not_trusted(runner, request_factory):
    output_dir = abandon(runner, request_factory())
    moved = output_dir.with_name("run-renamed")
    output_dir.rename(moved)
    (run,) = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)
    assert run.run_id == "run-renamed" and "another run" in run.problem


def test_unmarked_folders_are_reported_separately(runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-ok")
    abandon(runner, request_factory())
    old = runner.outputs_dir / BENCH / "run-before-markers"
    old.mkdir()
    (runner.outputs_dir / "trailers" / "trailer_01").mkdir(parents=True)  # not a benchmark
    assert find_unmarked(runner.outputs_dir, runner.logger) == [old]
    assert find_unmarked(runner.outputs_dir, runner.logger, benchmark="trailers") == []


# --- recording an abandoned run ---

def test_record_abandoned_writes_a_failed_record_and_deletes_nothing(runner, request_factory):
    output_dir = abandon(runner, request_factory(backend_parameters={"steps": 4}))
    partial = output_dir / "partial.mp4"
    partial.write_bytes(b"half a video")
    (run,) = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)

    record = record_abandoned(run, runner.logger)

    assert record == runner.logger.load(BENCH, "run-dead")
    assert record.status == "failed" and record.failure_stage == "abandoned"
    assert record.error_type == "AbandonedRun" and "never finished" in record.error_message
    assert record.schema_version == SCHEMA_VERSION  # no schema version change
    assert record.output_video is None and record.output_media is None
    assert record.seed == 1234 and record.model_parameters == {"steps": 4}
    assert record.input_checksums and record.source_video == "benchmarks/inputs/source.mp4"
    assert "abandoned: generation started" in record.notes and f"pid {os.getpid()}" in record.notes
    # nothing is deleted: the marker and the partial output are still there
    assert marker_path(output_dir).is_file() and partial.read_bytes() == b"half a video"
    # recorded now, so no longer reported, and never recorded twice
    assert find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0) == []
    with pytest.raises(FileExistsError):
        record_abandoned(run, runner.logger)
    # the run id stays taken
    with pytest.raises(FileExistsError):
        runner.run(request_factory(), FakeBackend(), run_id="run-dead")


def test_abandoned_is_a_failure_stage_only_for_failed_records():
    assert "abandoned" in FAILURE_STAGES
    assert FAILURE_STAGES[:5] == ("validation", "prepare", "generation", "verification", "logging")


def test_abandoned_record_lists_with_the_other_runs(runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-ok")
    abandon(runner, request_factory())
    (run,) = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)
    record_abandoned(run, runner.logger)
    assert [(r.run_id, r.status) for r in runner.logger.list_runs(BENCH)] == [
        ("run-dead", "failed"), ("run-ok", "success"),
    ]


# --- the script ---

@pytest.fixture
def script(project, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    return load_script()


def run_script(script, *argv):
    lines = []
    code = script.main(list(argv), out=lines.append)
    return code, "\n".join(lines)


def test_script_reports_nothing_on_a_clean_project(script, runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-ok")
    code, text = run_script(script)
    assert code == 0 and "no abandoned runs" in text


def test_script_ignores_a_run_that_may_still_be_running(script, runner, request_factory):
    abandon(runner, request_factory())
    code, text = run_script(script)  # default threshold: the marker is seconds old
    assert code == 0 and "ABANDONED" not in text


def test_script_lists_without_recording_by_default(script, runner, request_factory):
    abandon(runner, request_factory())
    code, text = run_script(script, "--stale-after-hours", "0")
    assert code == 1
    assert f"ABANDONED {BENCH}/run-dead" in text and f"outputs/{BENCH}/run-dead" in text
    assert "--record" in text
    assert not runner.logger.path_for(BENCH, "run-dead").exists()


def test_script_records_with_record_flag(script, runner, request_factory):
    output_dir = abandon(runner, request_factory())
    code, text = run_script(script, "--stale-after-hours", "0", "--record")
    assert code == 0
    assert f"benchmarks/results/{BENCH}/run-dead.json" in text and "failure_stage abandoned" in text
    assert runner.logger.load(BENCH, "run-dead").failure_stage == "abandoned"
    assert (output_dir / MARKER_NAME).is_file()
    code, text = run_script(script, "--stale-after-hours", "0", "--record")
    assert code == 0 and "no abandoned runs" in text


def test_script_reports_what_it_cannot_record(script, runner, request_factory):
    output_dir = abandon(runner, request_factory())
    marker_path(output_dir).write_text("{broken", encoding="utf-8")
    (runner.outputs_dir / BENCH / "run-old").mkdir()
    runner.run(request_factory(), FakeBackend(), run_id="run-ok")
    code, text = run_script(script, "--stale-after-hours", "0", "--record", "--benchmark", BENCH)
    assert code == 1
    assert "cannot be recorded" in text
    assert "neither a marker nor a record" in text and f"outputs/{BENCH}/run-old" in text
    assert not runner.logger.path_for(BENCH, "run-dead").exists()


@pytest.mark.parametrize("argv", [["--benchmark", "../x"], ["--stale-after-hours", "-1"]])
def test_script_rejects_bad_arguments(script, argv):
    with pytest.raises(SystemExit) as info:
        script.main(argv, out=lambda line: None)
    assert info.value.code == 2
