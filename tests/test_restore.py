"""Regenerating a recorded run's lost video: kept only if byte-identical."""

import os
import stat
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRunner, Resolution, load_definition
from ai_studio.quality.metrics import METRICS_VERSION
from ai_studio.quality.store import VideoMetricsRecord, VideoMetricsStore, sha256_file
from ai_studio.restore import RestoreError, main, recorded_video_sha256, request_parameters, restore_run_video
from conftest import FAKE_ENVIRONMENT, FakeGenerationBackend, fake_probe, fake_video_bytes

BENCH = "restore_01"
RES = Resolution(width=1280, height=704)


class ReproducibleBackend(FakeGenerationBackend):
    """Deterministic: the same request always gives the same file."""

    defaults = {"steps": 4, "shift": 8.0}


class DriftingBackend(ReproducibleBackend):
    """Gives a different file than it did when the run was recorded."""

    def generate(self, request, output_dir):
        result = super().generate(request, output_dir)
        Path(result.output_video).write_bytes(fake_video_bytes(1280, 704, 24, 121))
        return result


@pytest.fixture
def recorded(project, input_files, monkeypatch):
    """A recorded successful run whose video has been lost."""
    path = project / "benchmarks" / "definitions" / f"{BENCH}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"benchmark_id: {BENCH}\ntask_type: video_generation\nstatus: ready\n"
        "prompt: a rooftop at sunset\nnegative_prompt: blurry\nduration_seconds: 5\nfps: 24\n"
        "reference_images: [benchmarks/references/ref.png]\n", encoding="utf-8",
    )
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    definition = load_definition(path)
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    runner = BenchmarkRunner(logger, "outputs", environment_collector=lambda: dict(FAKE_ENVIRONMENT),
                             root=project, media_probe=fake_probe)
    record = runner.run_definition(definition, ReproducibleBackend(), seed=3, output_resolution=RES,
                                   backend_parameters={"steps": 6}, run_id="run-1")
    video = project / record.output_video
    digest = sha256_file(video)
    VideoMetricsStore(project / "benchmarks" / "metrics", logger).save(VideoMetricsRecord(
        benchmark_id=BENCH, run_id="run-1", metrics_version=METRICS_VERSION,
        analysis_size={"width": 320, "height": 176}, video_sha256=digest, metrics={"motion_mean": 1.0},
        per_transition_diffs=[1.0], tool_versions={"numpy": "1", "ffmpeg": "8"},
    ))
    os.chmod(video, stat.S_IWRITE | stat.S_IREAD)
    video.unlink()  # the accident
    return project, definition, record, video, digest


def test_identical_regeneration_is_restored_and_read_only(recorded):
    project, definition, record, video, digest = recorded
    record_bytes = (project / "benchmarks" / "results" / BENCH / "run-1.json").read_bytes()
    backend = ReproducibleBackend()

    outcome = restore_run_video(record, definition, backend, digest, project)

    assert outcome.restored and outcome.actual_sha256 == outcome.expected_sha256 == digest
    assert video.is_file() and sha256_file(video) == digest
    assert not video.stat().st_mode & stat.S_IWUSR  # protected against the next accident
    assert backend.received.seed == 3 and backend.received.backend_parameters == {"steps": 6}
    assert backend.received.output_resolution == RES
    assert (project / "benchmarks" / "results" / BENCH / "run-1.json").read_bytes() == record_bytes
    assert sorted(p.name for p in video.parent.iterdir()) == [video.name]  # nothing else was written there


def test_different_regeneration_is_discarded(recorded):
    project, definition, record, video, digest = recorded
    outcome = restore_run_video(record, definition, DriftingBackend(), digest, project)
    assert not outcome.restored and outcome.actual_sha256 != digest
    assert "not byte-identical" in outcome.reason
    assert not video.exists()  # the run stays without a video rather than with a wrong one


def test_existing_video_is_never_overwritten(recorded):
    project, definition, record, video, digest = recorded
    video.write_bytes(b"something is already here")
    backend = ReproducibleBackend()
    with pytest.raises(RestoreError, match="already exists"):
        restore_run_video(record, definition, backend, digest, project)
    assert video.read_bytes() == b"something is already here" and backend.calls == []


def test_runs_that_cannot_be_repeated_as_recorded_are_refused(recorded, input_files):
    import dataclasses

    project, definition, record, video, digest = recorded
    _, reference = input_files

    class OtherBackend(ReproducibleBackend):
        def metadata(self):
            return dataclasses.replace(super().metadata(), backend_name="another-backend")

    with pytest.raises(RestoreError, match="was made with 'fake-gen'"):
        restore_run_video(record, definition, OtherBackend(), digest, project)
    with pytest.raises(RestoreError, match="prompt has changed"):
        restore_run_video(record, dataclasses.replace(definition, prompt="another prompt"), ReproducibleBackend(),
                          digest, project)
    with pytest.raises(RestoreError, match="did not succeed"):
        failed = dataclasses.replace(record, status="failed", failure_stage="generation", error_type="X",
                                     output_video=None, output_media=None)
        restore_run_video(failed, definition, ReproducibleBackend(), digest, project)
    with pytest.raises(RestoreError, match="does not expose its backend parameters"):
        restore_run_video(record, definition, FakeGenerationBackend(), digest, project)

    reference.write_bytes(b"a different reference image")
    backend = ReproducibleBackend()
    with pytest.raises(RestoreError, match="checksum differs"):
        restore_run_video(record, definition, backend, digest, project)
    assert backend.calls == [] and not video.exists()


def test_request_parameters_are_the_recorded_values_of_the_backends_own_parameters(recorded):
    _, _, record, _, _ = recorded
    assert record.model_parameters == {"steps": 6, "sampler": "fake"}  # requested + what the backend added
    assert request_parameters(record, ReproducibleBackend()) == {"steps": 6}


def test_command_line(recorded):
    project, _, _, video, digest = recorded
    assert recorded_video_sha256(project, BENCH, "run-1") == digest
    lines = []
    code = main(["--benchmark", BENCH, "--run-id", "run-1", "--run-id", "no-such-run"], out=lines.append,
                backend_for=lambda name: ReproducibleBackend())
    text = "\n".join(lines)
    assert code == 1  # one of the two could not be restored
    assert "run-1: RESTORED - byte-identical; restored" in text and digest in text
    assert "no-such-run: NOT RESTORED" in text
    assert sha256_file(video) == digest

    lines.clear()
    assert main(["--benchmark", BENCH, "--run-id", "run-1"], out=lines.append,
                backend_for=lambda name: ReproducibleBackend()) == 1
    assert "already exists" in lines[0]


def test_without_a_recorded_hash_nothing_is_generated(recorded):
    project, _, _, video, _ = recorded
    metrics = project / "benchmarks" / "metrics" / BENCH / "run-1" / f"metrics_v{METRICS_VERSION}.json"
    metrics.rename(metrics.with_suffix(".moved"))
    backend = ReproducibleBackend()
    lines = []
    assert main(["--benchmark", BENCH, "--run-id", "run-1"], out=lines.append, backend_for=lambda name: backend) == 1
    assert "no recorded video hash" in lines[0] and backend.calls == [] and not video.exists()
