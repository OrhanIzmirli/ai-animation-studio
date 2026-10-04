"""The runner measures the output video instead of trusting the backend."""

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_studio.benchmarking import (
    BenchmarkLogger,
    BenchmarkRunError,
    BenchmarkRunner,
    Resolution,
    load_definition,
)
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import GenerationResult
from conftest import FAKE_ENVIRONMENT, FakeGenerationBackend, fake_video_bytes

BENCH = "comic_rooftop_01"
DEFINITIONS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "definitions"


class MeasuredBackend(FakeGenerationBackend):
    """Writes a video with `written` = (width, height, fps, frame_count) and
    reports `reported` = (fps, frame_count). written=None writes a file the
    probe cannot measure."""

    def __init__(self, written, reported="same", **kwargs):
        super().__init__(**kwargs)
        self.written = written
        self.reported = written[2:] if reported == "same" else reported

    def generate(self, request, output_dir):
        self.calls.append("generate")
        output = Path(output_dir) / self.output_name
        output.write_bytes(fake_video_bytes(*self.written) if self.written else b"not measurable")
        fps, frame_count = self.reported or (None, None)
        return GenerationResult(
            output_video=str(output), backend_name="fake-gen", backend_version="1.0",
            model_name="fake-t2v", model_version="3.0", fps=fps, frame_count=frame_count,
        )


def run(runner, request, backend, run_id="run-1"):
    return runner.run(request, backend, benchmark_id=BENCH, run_id=run_id)


def assert_verification_failure(runner, request, backend, match):
    with pytest.raises(BenchmarkRunError, match=match) as exc_info:
        run(runner, request, backend)
    error = exc_info.value
    assert error.stage == "verification"
    saved = runner.logger.load(BENCH, "run-1")
    assert saved.status == "failed" and saved.failure_stage == "verification"
    assert saved.output_media is None  # only successful runs carry a measurement
    return saved


# --- measured values are recorded ---

def test_measurement_recorded_on_success(runner, generation_request_factory):
    record = run(runner, generation_request_factory(), MeasuredBackend((1280, 720, 24, 120)))
    assert record.output_media == {
        "probe": "fake-probe 1.0", "width": 1280, "height": 720, "fps": 24,
        "frame_count": 120, "duration_seconds": 5.0, "codec": "fake",
    }
    assert record.fps == 24 and record.frame_count == 120
    assert "warning" not in record.notes


def test_measured_values_fill_in_when_backend_reports_nothing(runner, generation_request_factory):
    backend = MeasuredBackend((1280, 720, 24, 120), reported=None)
    record = run(runner, generation_request_factory(), backend)
    assert record.fps == 24 and record.frame_count == 120


def test_editing_run_is_measured_too(runner, request_factory):
    from conftest import FakeBackend

    record = runner.run(request_factory(), FakeBackend(), run_id="e")
    assert record.output_media["frame_count"] == 81 and record.output_media["fps"] == 24


# --- a) probe unavailable ---

def test_no_probe_with_pinned_fps_fails(runner, generation_request_factory):
    saved = assert_verification_failure(
        runner, generation_request_factory(fps=24), MeasuredBackend(None, reported=(24, 120)),
        "cannot verify",
    )
    assert saved.error_type == "ai_studio.media_probe.ProbeUnavailableError"


def test_no_probe_with_duration_tolerance_fails(runner, generation_request_factory):
    request = generation_request_factory(fps=None, duration_tolerance_seconds=0.1)
    assert_verification_failure(runner, request, MeasuredBackend(None, reported=(16, 80)), "cannot verify")


def test_no_probe_without_pins_is_a_warning(runner, generation_request_factory):
    request = generation_request_factory(fps=None)
    record = run(runner, request, MeasuredBackend(None, reported=(16, 80)))
    assert record.status == "success"
    assert record.output_media is None
    assert record.fps == 16 and record.frame_count == 80  # backend-reported fallback
    assert "warning: output video not measured" in record.notes


# --- b) backend fps vs measured ---

def test_backend_fps_lie_fails(runner, generation_request_factory):
    backend = MeasuredBackend((1280, 720, 25, 120), reported=(24, 120))
    request = generation_request_factory(fps=None)
    assert_verification_failure(runner, request, backend, "backend reported fps 24, measured 25")


def test_backend_fps_within_tolerance_passes(runner, generation_request_factory):
    backend = MeasuredBackend((1280, 720, 24, 120), reported=(24.005, 120))
    record = run(runner, generation_request_factory(), backend)
    assert record.fps == 24  # the measured value, not the reported one


# --- c) backend frame_count vs measured ---

def test_backend_frame_count_lie_fails(runner, generation_request_factory):
    backend = MeasuredBackend((1280, 720, 24, 121), reported=(24, 120))
    assert_verification_failure(
        runner, generation_request_factory(), backend, "backend reported 120 frames, measured 121",
    )


# --- d) requested fps vs measured ---

def test_ntsc_fps_does_not_satisfy_pinned_24(runner, generation_request_factory):
    backend = MeasuredBackend((1280, 720, 24000 / 1001, 120))  # honest backend, wrong rate
    assert_verification_failure(
        runner, generation_request_factory(fps=24), backend, "requested fps 24, measured 23.976",
    )


# --- e) duration tolerance ---

def test_wan_grid_121_frames_passes_with_tolerance(runner, generation_request_factory):
    request = generation_request_factory(
        fps=24, duration_tolerance_seconds=0.1, output_resolution=Resolution(width=1280, height=704),
    )
    record = run(runner, request, MeasuredBackend((1280, 704, 24, 121)))
    assert record.status == "success"
    assert record.frame_count == 121 and record.output_media["duration_seconds"] == 5.041667
    assert record.task_parameters["duration_tolerance_seconds"] == 0.1


@pytest.mark.parametrize("frames, seconds", [(96, "4.000s"), (144, "6.000s")])
def test_wrong_duration_fails_with_measured_details(runner, generation_request_factory, frames, seconds):
    request = generation_request_factory(fps=24, duration_tolerance_seconds=0.1)
    with pytest.raises(BenchmarkRunError) as exc_info:
        run(runner, request, MeasuredBackend((1280, 720, 24, frames)))
    message = str(exc_info.value)
    assert exc_info.value.stage == "verification"
    assert seconds in message and f"{frames} frames" in message and "24 fps" in message


def test_duration_exactly_at_tolerance_passes(runner, generation_request_factory):
    # 5 s requested, 132 frames @ 24 fps = 5.5 s, tolerance 0.5
    request = generation_request_factory(fps=24, duration_tolerance_seconds=0.5)
    assert run(runner, request, MeasuredBackend((1280, 720, 24, 132))).status == "success"


def test_zero_tolerance_requires_exact_duration(runner, generation_request_factory):
    request = generation_request_factory(fps=24, duration_tolerance_seconds=0)
    assert run(runner, request, MeasuredBackend((1280, 720, 24, 120)), "ok").status == "success"
    with pytest.raises(BenchmarkRunError, match="121 frames"):
        run(runner, request, MeasuredBackend((1280, 720, 24, 121)), "off")


def test_duration_not_enforced_without_tolerance(runner, generation_request_factory):
    record = run(runner, generation_request_factory(), MeasuredBackend((1280, 720, 24, 96)))
    assert record.status == "success" and record.output_media["duration_seconds"] == 4.0


# --- f) resolution ---

def test_resolution_mismatch_is_only_a_warning(runner, generation_request_factory):
    record = run(runner, generation_request_factory(), MeasuredBackend((1280, 704, 24, 120)))
    assert record.status == "success"
    assert record.resolution == Resolution(width=1280, height=720)  # what was requested
    assert record.output_media["height"] == 704
    assert "warning: measured resolution 1280x704 differs from requested 1280x720" in record.notes


# --- broken output ---

def test_probe_error_on_broken_file_fails_verification(project, generation_request_factory):
    def broken_probe(path):
        raise RuntimeError("moov atom not found")

    runner = BenchmarkRunner(
        BenchmarkLogger(project / "results"), "outputs",
        environment_collector=lambda: dict(FAKE_ENVIRONMENT), root=project, media_probe=broken_probe,
    )
    with pytest.raises(BenchmarkRunError, match="moov atom") as exc_info:
        run(runner, generation_request_factory(fps=None), MeasuredBackend((1280, 720, 24, 120)))
    assert exc_info.value.stage == "verification"
    assert exc_info.value.record.error_type == "RuntimeError"


# --- duration tolerance on requests and definitions ---

@pytest.mark.parametrize("bad", [-0.1, "0.1", True])
def test_request_rejects_invalid_tolerance(generation_request_factory, bad):
    with pytest.raises(ValueError, match="duration_tolerance_seconds"):
        generation_request_factory(duration_tolerance_seconds=bad)


def test_comic_rooftop_pins_duration_tolerance():
    definition = load_definition(DEFINITIONS_DIR / "comic_rooftop_01.yaml")
    assert definition.duration_tolerance_seconds == 0.1
    request = definition.to_request(seed=1, output_resolution=Resolution(width=1280, height=704))
    assert isinstance(request, VideoGenerationRequest)
    assert request.duration_tolerance_seconds == 0.1 and request.fps == 24


def test_tolerance_has_no_runtime_override(runner):
    definition = load_definition(DEFINITIONS_DIR / "comic_rooftop_01.yaml")
    with pytest.raises(TypeError):
        definition.to_request(
            seed=1, output_resolution=Resolution(width=1280, height=704), duration_tolerance_seconds=1.0,
        )
    with pytest.raises(TypeError):
        runner.run_definition(
            definition, MeasuredBackend((1280, 704, 24, 121)), seed=1,
            output_resolution=Resolution(width=1280, height=704), duration_tolerance_seconds=1.0,
        )


def test_editing_definition_rejects_tolerance(tmp_path):
    path = tmp_path / "e.yaml"
    path.write_text(
        "benchmark_id: e\ntask_type: video_editing\nduration_tolerance_seconds: 0.1\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duration_tolerance_seconds"):
        load_definition(path)


def test_definition_rejects_negative_tolerance():
    definition = load_definition(DEFINITIONS_DIR / "comic_rooftop_01.yaml")
    with pytest.raises(ValueError, match="duration_tolerance_seconds"):
        dataclasses.replace(definition, duration_tolerance_seconds=-1)


# --- end to end with the real ffprobe ---

class RealVideoBackend(FakeGenerationBackend):
    """Encodes a real H.264 clip with ffmpeg: 64x48, 24 fps, 12 frames."""

    def generate(self, request, output_dir):
        self.calls.append("generate")
        output = Path(output_dir) / "real.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
                "-frames:v", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
            ],
            check=True, capture_output=True, timeout=120,
        )
        return GenerationResult(
            output_video=str(output), backend_name="real", backend_version="1.0",
            model_name="testsrc", model_version="1", fps=24, frame_count=12,
        )


def test_runner_with_real_ffprobe(project, generation_request_factory):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")
    runner = BenchmarkRunner(  # default media_probe: ffprobe_video
        BenchmarkLogger(project / "results"), "outputs",
        environment_collector=lambda: dict(FAKE_ENVIRONMENT), root=project,
    )
    request = generation_request_factory(
        duration_seconds=0.5, fps=24, duration_tolerance_seconds=0.01,
        output_resolution=Resolution(width=64, height=48),
    )
    record = run(runner, request, RealVideoBackend())
    assert record.output_media["probe"].startswith("ffprobe ")
    assert record.output_media["codec"] == "h264"
    assert record.fps == 24 and record.frame_count == 12
    assert record.output_media["duration_seconds"] == 0.5
