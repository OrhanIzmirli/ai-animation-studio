"""Benchmark definition lifecycle: draft (default) -> ready (explicit human action)."""

import dataclasses
from pathlib import Path

import pytest

from ai_studio.benchmarking import (
    BenchmarkLogger,
    BenchmarkRunner,
    DraftBenchmarkError,
    Resolution,
    load_definition,
    load_definitions,
)
from conftest import FAKE_ENVIRONMENT, FakeBackend, FakeGenerationBackend

REPO_DEFINITIONS = Path(__file__).resolve().parents[1] / "benchmarks" / "definitions"
RES = Resolution(width=64, height=64)

GENERATION = """\
benchmark_id: gen_01
task_type: video_generation
prompt: a rooftop at sunset
duration_seconds: 5
camera_motion: slow push-in
"""


def write(directory, text, name="def.yaml"):
    path = Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def project_runner(project):
    return BenchmarkRunner(
        BenchmarkLogger(project / "benchmarks" / "results"), "outputs",
        environment_collector=lambda: dict(FAKE_ENVIRONMENT), root=project,
    )


# --- default and parsing ---

def test_default_is_draft(tmp_path):
    assert load_definition(write(tmp_path, GENERATION)).status == "draft"


def test_explicit_statuses_parse(tmp_path):
    assert load_definition(write(tmp_path, GENERATION + "status: draft\n")).status == "draft"
    assert load_definition(write(tmp_path, GENERATION + "status: ready\n")).status == "ready"


@pytest.mark.parametrize("bad", ["Ready", "done", "true", "null"])
def test_unknown_status_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="status"):
        load_definition(write(tmp_path, GENERATION + f"status: {bad}\n"))


def test_repository_benchmark_statuses():
    definitions = load_definitions(REPO_DEFINITIONS)
    assert definitions["comic_rooftop_01"].status == "ready"  # marked ready 2026-09-30
    assert definitions["object_replacement_01"].status == "draft"  # real assets pending


# --- draft cannot run ---

def test_draft_cannot_become_request(tmp_path):
    definition = load_definition(write(tmp_path, GENERATION))
    with pytest.raises(DraftBenchmarkError, match="draft"):
        definition.to_request(seed=1, output_resolution=RES)


def test_runner_refuses_draft_generation(project_runner, project):
    definition = load_definition(write(project, GENERATION))
    backend = FakeGenerationBackend()

    with pytest.raises(DraftBenchmarkError):
        project_runner.run_definition(definition, backend, seed=1, output_resolution=RES)

    assert backend.calls == []  # nothing executed
    assert not project_runner.outputs_dir.exists()  # no output directory
    assert project_runner.logger.list_runs("gen_01") == []  # no record, not even a failed one


def test_runner_refuses_draft_editing(project_runner, input_files):
    source, _ = input_files
    text = (
        "benchmark_id: edit_01\ntask_type: video_editing\n"
        f"source_video: {source.as_posix()}\nprompt: replace the mug\n"
    )
    definition = load_definition(write(source.parent, text))
    assert definition.is_complete  # inputs exist, yet it is still a draft
    backend = FakeBackend()

    with pytest.raises(DraftBenchmarkError):
        project_runner.run_definition(definition, backend, seed=1, output_resolution=RES)
    assert backend.calls == []


def test_repository_drafts_cannot_run(project_runner):
    definition = load_definition(REPO_DEFINITIONS / "object_replacement_01.yaml")
    with pytest.raises(DraftBenchmarkError):
        project_runner.run_definition(definition, FakeBackend(), seed=1, output_resolution=RES)


def test_existing_files_do_not_make_a_benchmark_ready(project, input_files):
    # Readiness is never inferred: all inputs exist, the status is still draft.
    _, reference = input_files
    text = GENERATION + f"reference_images: [{reference.as_posix()}]\n"
    definition = load_definition(write(project, text))
    assert all(Path(p).is_file() for p in definition.reference_images)
    assert definition.status == "draft"


# --- ready can run ---

def test_ready_generation_benchmark_runs(project_runner, project):
    definition = load_definition(write(project, GENERATION + "status: ready\n"))
    record = project_runner.run_definition(
        definition, FakeGenerationBackend(), seed=3, output_resolution=RES, fps=24, run_id="run-1",
    )
    assert record.status == "success"
    assert record.benchmark_id == "gen_01" and record.task_type == "video_generation"
    assert record.task_parameters["duration_seconds"] == 5
    assert record.task_parameters["requested_fps"] == 24
    assert project_runner.logger.load("gen_01", "run-1") == record


def test_ready_editing_benchmark_runs(project_runner, input_files):
    source, reference = input_files
    text = (
        "benchmark_id: edit_01\ntask_type: video_editing\nstatus: ready\n"
        "source_video: benchmarks/inputs/source.mp4\n"
        "reference_images: [benchmarks/references/ref.png]\nprompt: replace the mug\n"
    )
    definition = load_definition(write(source.parent, text))
    record = project_runner.run_definition(
        definition, FakeBackend(), seed=3, output_resolution=RES, backend_parameters={"steps": 20},
    )
    assert record.status == "success" and record.task_type == "video_editing"
    assert record.source_video == "benchmarks/inputs/source.mp4"
    assert record.model_parameters["steps"] == 20


def test_marking_ready_is_a_deliberate_change(tmp_path):
    draft = load_definition(write(tmp_path, GENERATION))
    ready = dataclasses.replace(draft, status="ready")
    assert draft.status == "draft"  # the loaded definition is never mutated
    assert ready.to_request(seed=1, output_resolution=RES).duration_seconds == 5


# --- backward compatibility ---

def test_old_definitions_without_status_still_load_as_draft(tmp_path):
    # A Phase 1B-style definition written before the lifecycle existed.
    old = "benchmark_id: legacy\ntask_type: video_editing\nsource_video: in.mp4\nprompt: edit\n"
    definition = load_definition(write(tmp_path, old))
    assert definition.status == "draft"
    assert definition.is_complete  # content unchanged; it just needs to be marked ready


def test_direct_request_runs_are_unaffected(project_runner, request_factory):
    # run() with a hand-built request (tests, ad-hoc experiments) has no definition lifecycle.
    record = project_runner.run(request_factory(), FakeBackend())
    assert record.status == "success"
