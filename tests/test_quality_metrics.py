"""Automatic video quality metrics: pure functions, frame decoding, the
metrics store and scripts/measure_run.py. Synthetic videos only."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ai_studio import paths
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, Resolution
from ai_studio.quality import frames as frames_module
from ai_studio.quality.frames import FrameDecodeError, analysis_size, decode_image, decode_video
from ai_studio.quality.metrics import (
    METRICS_VERSION,
    PSNR_IDENTICAL_DB,
    compute_metrics,
    held_frame_ratio,
    luminance_flicker,
    motion_max,
    motion_mean,
    pulse_from_diffs,
    start_frame_psnr,
    temporal_pulse,
    to_gray,
    total_change,
    transition_diffs,
)
from ai_studio.quality.store import VideoMetricsRecord, VideoMetricsStore, measure_video, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = "bench_01"
WIDTH, HEIGHT = 320, 176

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


# --- synthetic frame stacks (N, H, W) uint8 ---

def textured(seed=0):
    """A fixed, blocky random image with values well inside 0-255."""
    blocks = np.random.default_rng(seed).integers(60, 180, size=(HEIGHT // 16, WIDTH // 16))
    return np.kron(blocks, np.ones((16, 16))).astype(np.uint8)


def static_frames(count=24):
    return np.repeat(textured()[None], count, axis=0)


def moving_frames(count=24, step=4):
    """A horizontal gradient that scrolls `step` pixels per frame."""
    row = (np.arange(WIDTH) * 255 // WIDTH).astype(np.uint8)
    base = np.tile(row, (HEIGHT, 1))
    return np.stack([np.roll(base, i * step, axis=1) for i in range(count)])


def pulsing_frames(count=25, bump=12):
    """Slowly scrolling texture, plus a brightness bump on every 4th frame."""
    base = textured()
    stack = np.stack([np.roll(base, i, axis=1) for i in range(count)]).astype(np.int16)
    stack[::4] += bump
    return stack.astype(np.uint8)


def fading_frames(count=24):
    """Brightness rises by 2 per frame: a linear fade."""
    return np.stack([textured() // 2 + 2 * i for i in range(count)]).astype(np.uint8)


# --- pure metrics ---

def test_static_video_has_no_motion_and_only_held_frames():
    frames = static_frames()
    assert motion_mean(frames) == 0 and motion_max(frames) == 0 and total_change(frames) == 0
    assert held_frame_ratio(frames) == 1.0
    assert luminance_flicker(frames) == pytest.approx(0, abs=1e-9)
    assert temporal_pulse(frames)["ratio"] == 1.0  # nothing changes: no pulse


def test_moving_gradient_has_motion_and_no_held_frames():
    frames = moving_frames()
    assert motion_mean(frames) > 1
    assert motion_max(frames) >= motion_mean(frames)
    assert total_change(frames) > 10
    assert held_frame_ratio(frames) == 0.0
    assert temporal_pulse(frames)["ratio"] == pytest.approx(1.0, abs=0.05)  # uniform motion: no pulse


def test_transition_diffs_are_per_pair_means():
    frames = np.zeros((4, 2, 2), dtype=np.uint8)
    frames[1] = 10
    frames[2] = 10
    frames[3] = 10
    frames[3, 0, 0] = 30  # one of four pixels changes by 20
    assert transition_diffs(frames).tolist() == [10.0, 0.0, 5.0]
    assert motion_mean(frames) == 5.0 and motion_max(frames) == 10.0
    assert held_frame_ratio(frames) == pytest.approx(1 / 3)


def test_sudden_jump_shows_in_motion_max():
    frames = static_frames(20).copy()
    frames[10:] = 255 - frames[10:]  # a hard cut
    assert motion_max(frames) > 20 * motion_mean(frames) / 2
    assert motion_max(frames) > 50


def test_brightness_bump_every_fourth_frame_gives_a_clear_pulse():
    pulse = temporal_pulse(pulsing_frames())
    assert pulse["period"] == 4 and len(pulse["phase_means"]) == 4
    assert pulse["ratio"] > 2
    # the bump frame is entered in phase 3 (3 -> 4) and left in phase 0 (4 -> 5)
    quiet, loud = sorted(pulse["phase_means"])[:2], sorted(pulse["phase_means"])[2:]
    assert max(quiet) < min(loud)
    assert {pulse["phase_means"].index(v) for v in loud} == {0, 3}


def test_pulse_matches_the_hand_measured_smoke_run():
    # Phase means measured by hand on the first Wan 2.2 smoke run.
    diffs = np.tile([0.97, 0.70, 0.62, 0.42], 30)
    pulse = pulse_from_diffs(diffs)
    assert pulse["phase_means"] == pytest.approx([0.97, 0.70, 0.62, 0.42])
    assert pulse["ratio"] == pytest.approx(0.97 / 0.42)


def test_pulse_ratio_stays_finite_when_a_phase_has_no_change():
    diffs = np.tile([5.0, 0.0, 0.0, 0.0], 10)
    assert pulse_from_diffs(diffs)["ratio"] == pytest.approx(500.0)  # 5.0 / PULSE_FLOOR
    assert pulse_from_diffs(np.zeros(40))["ratio"] == 1.0


def test_pulse_other_periods_and_too_short_videos():
    diffs = np.tile([3.0, 1.0], 20)
    assert pulse_from_diffs(diffs, period=2)["ratio"] == pytest.approx(3.0)
    assert pulse_from_diffs(diffs, period=4)["ratio"] == pytest.approx(3.0)  # a multiple of the real period sees it too
    assert pulse_from_diffs(np.tile([3.0, 1.0], 21), period=3)["ratio"] == pytest.approx(1.0)  # an unrelated period does not
    assert pulse_from_diffs(np.ones(3)) == {"period": 4, "phase_means": None, "ratio": None}
    with pytest.raises(ValueError):
        pulse_from_diffs(diffs, period=1)


def test_linear_fade_is_not_flicker_but_pumping_is():
    fade = fading_frames()
    assert motion_mean(fade) > 1  # it does change...
    assert luminance_flicker(fade) < 0.1  # ...but along a straight line

    pumping = static_frames(24).astype(np.int16)
    pumping[::2] += 8  # brightness alternates frame by frame
    assert luminance_flicker(pumping.astype(np.uint8)) == pytest.approx(4.0, abs=0.1)

    both = fade.astype(np.int16)
    both[::2] += 8
    assert luminance_flicker(both.astype(np.uint8)) == pytest.approx(4.0, abs=0.2)


def test_stepped_animation_is_reported_as_holds_not_as_a_freeze():
    on_twos = np.repeat(moving_frames(12, step=8), 2, axis=0)  # each drawing held for 2 frames
    assert held_frame_ratio(on_twos) == pytest.approx(12 / 23)
    assert motion_mean(on_twos) > 1  # unlike a frozen video


def test_start_frame_psnr():
    frames = moving_frames()
    assert start_frame_psnr(frames, frames[0]) == PSNR_IDENTICAL_DB
    off_by_one = frames[0].astype(np.int16) + 1
    assert start_frame_psnr(frames, off_by_one) == pytest.approx(48.13, abs=0.01)  # MSE 1
    shifted = np.roll(frames[0], 12, axis=1)
    unrelated = textured(seed=5)
    assert PSNR_IDENTICAL_DB > start_frame_psnr(frames, off_by_one) > start_frame_psnr(frames, shifted)
    assert start_frame_psnr(frames, unrelated) < 20
    with pytest.raises(ValueError, match="shape"):
        start_frame_psnr(frames, frames[0][:10])


def test_rgb_input_is_reduced_to_luma():
    gray = moving_frames(6)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    assert to_gray(rgb) == pytest.approx(gray.astype(float))
    assert motion_mean(rgb) == pytest.approx(motion_mean(gray))
    with pytest.raises(ValueError):
        to_gray(np.zeros((4, 4)))


def test_compute_metrics_is_flat_and_json_safe():
    frames = pulsing_frames()
    metrics = compute_metrics(frames, frames[0])
    assert set(metrics) == {
        "frame_count", "motion_mean", "motion_max", "total_change", "start_frame_psnr",
        "temporal_pulse_period", "temporal_pulse_ratio", "temporal_pulse_phase_means",
        "luminance_flicker", "luminance_min", "luminance_max", "held_frame_ratio",
        "hold_pattern_periodicity",
    }
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics
    assert metrics["frame_count"] == 25 and metrics["start_frame_psnr"] == PSNR_IDENTICAL_DB
    assert metrics["luminance_min"] < metrics["luminance_max"]
    assert compute_metrics(frames)["start_frame_psnr"] is None
    assert compute_metrics(frames[:1])["temporal_pulse_ratio"] is None  # a single frame still works
    with pytest.raises(ValueError, match="no frames"):
        compute_metrics(frames[:0])


def test_metrics_have_no_thresholds_yet():
    import yaml

    config = yaml.safe_load((REPO_ROOT / "configs" / "quality.yaml").read_text(encoding="utf-8"))
    assert all(value is None for section in config.values() if isinstance(section, dict) for value in section.values())
    assert METRICS_VERSION == "3"


# --- analysis size ---

@pytest.mark.parametrize(
    "source, expected",
    [((1280, 704), (320, 176)), ((704, 1280), (320, 582)), ((1280, 720), (320, 180)),
     ((1672, 941), (320, 180)), ((320, 176), (320, 176)), ((100, 3), (320, 10)), ((4000, 2), (320, 2))],
)
def test_analysis_size_keeps_aspect_with_even_dimensions(source, expected):
    size = analysis_size(*source)
    assert size == expected and size[0] % 2 == 0 and size[1] % 2 == 0


def test_analysis_size_rejects_nonsense():
    with pytest.raises(ValueError):
        analysis_size(0, 100)


# --- decoding real files ---

def write_video(path, frames, fps=24, size=None):
    """Encode a gray (N, H, W) uint8 stack losslessly (H.264, qp 0)."""
    height, width = frames.shape[1:]
    scale = [] if size is None else ["-vf", f"scale={size[0]}:{size[1]}:flags=neighbor"]
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", *scale, "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)],
        input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True,
    )
    return path


def write_image(path, frame):
    height, width = frame.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
         "-i", "-", "-frames:v", "1", str(path)],
        input=np.ascontiguousarray(frame).tobytes(), check=True, capture_output=True,
    )
    return path


@needs_ffmpeg
def test_decode_video_returns_every_frame_at_the_analysis_size(tmp_path):
    video = write_video(tmp_path / "v.mp4", moving_frames(13), size=(1280, 704))
    gray = decode_video(video)
    assert gray.shape == (13, 176, 320) and gray.dtype == np.uint8
    rgb = decode_video(video, gray=False)
    assert rgb.shape == (13, 176, 320, 3)
    assert decode_video(video, analysis_width=160).shape == (13, 88, 160)
    # a lossless round trip at the analysis size keeps the pixels (within range-conversion rounding)
    same_size = write_video(tmp_path / "same.mp4", moving_frames(5))
    assert np.abs(decode_video(same_size).astype(int) - moving_frames(5).astype(int)).max() <= 2


@needs_ffmpeg
def test_static_video_measures_as_static(tmp_path):
    record = measure_video(write_video(tmp_path / "v.mp4", static_frames()), benchmark_id=BENCH, run_id="r")
    assert record.metrics["motion_mean"] == pytest.approx(0, abs=0.01)
    assert record.metrics["held_frame_ratio"] == pytest.approx(1.0)
    assert record.metrics["start_frame_psnr"] is None
    assert len(record.per_transition_diffs) == 23 == record.metrics["frame_count"] - 1


@needs_ffmpeg
def test_moving_video_measures_motion(tmp_path):
    record = measure_video(write_video(tmp_path / "v.mp4", moving_frames()), benchmark_id=BENCH, run_id="r")
    assert record.metrics["motion_mean"] > 1 and record.metrics["held_frame_ratio"] == 0
    assert record.metrics["temporal_pulse_ratio"] == pytest.approx(1.0, abs=0.05)


@needs_ffmpeg
def test_pulsing_video_measures_a_pulse(tmp_path):
    record = measure_video(write_video(tmp_path / "v.mp4", pulsing_frames()), benchmark_id=BENCH, run_id="r")
    assert record.metrics["temporal_pulse_ratio"] > 2
    assert record.metrics["temporal_pulse_period"] == 4


@needs_ffmpeg
def test_fading_video_has_low_luminance_flicker(tmp_path):
    record = measure_video(write_video(tmp_path / "v.mp4", fading_frames()), benchmark_id=BENCH, run_id="r")
    assert record.metrics["luminance_flicker"] < 0.3
    assert record.metrics["luminance_max"] - record.metrics["luminance_min"] > 30


@needs_ffmpeg
def test_start_frame_psnr_is_high_for_the_exact_frame_and_lower_when_shifted(tmp_path):
    frames = np.stack([np.roll(textured(), i, axis=1) for i in range(8)])
    video = write_video(tmp_path / "v.mp4", frames, size=(1280, 704))
    big = np.kron(frames[0], np.ones((4, 4))).astype(np.uint8)  # the 1280x704 start frame
    exact = write_image(tmp_path / "exact.png", big)
    shifted = write_image(tmp_path / "shifted.png", np.roll(big, 64, axis=1))

    def psnr(reference):
        return measure_video(video, benchmark_id=BENCH, run_id="r", reference_image=reference).metrics["start_frame_psnr"]

    assert psnr(exact) > 40
    assert psnr(shifted) < psnr(exact) - 15
    assert decode_image(exact, (320, 176)).shape == (176, 320)


def test_missing_ffmpeg_is_a_clear_error(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(frames_module.shutil, "which", lambda name: None)
    with pytest.raises(FrameDecodeError, match="ffmpeg not found"):
        decode_video(video)
    with pytest.raises(FrameDecodeError, match="ffmpeg not found"):
        decode_image(video, (320, 176))
    assert frames_module.ffmpeg_version() is None


@needs_ffmpeg
def test_undecodable_or_missing_files_are_reported(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(FrameDecodeError, match="could not decode"):
        decode_image(broken, (320, 176))
    with pytest.raises(FrameDecodeError, match="not found"):
        decode_video(tmp_path / "missing.mp4")


# --- store ---

def log_run(project, run_id, *, status="success", output_video="default", start_frame=None, steps=50, purpose=None):
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    if output_video == "default":
        output_video = f"outputs/{BENCH}/{run_id}/out.mp4" if status == "success" else None
    failure = {} if status == "success" else {
        "status": "failed", "failure_stage": "generation", "error_type": "MemoryError", "error_message": "oom",
    }
    logger.log(BenchmarkResult(
        benchmark_id=BENCH, run_id=run_id, task_type="video_generation", **failure,
        task_parameters={} if purpose is None else {"run_purpose": purpose},
        model="m", model_version="1", workflow_version="w", seed=1, inference_steps=steps, guidance=5.0,
        prompt="p", output_video=output_video, start_frame=start_frame,
        resolution=Resolution(width=1280, height=704),
    ))
    return logger


def sample_record(run_id="run-1", **overrides):
    fields = dict(
        benchmark_id=BENCH, run_id=run_id, metrics_version=METRICS_VERSION,
        analysis_size={"width": 320, "height": 176}, video_sha256="ab" * 32,
        metrics={"motion_mean": 0.68}, per_transition_diffs=[0.7, 0.4],
        tool_versions={"numpy": "1.26.4", "ffmpeg": "8.1"},
    )
    fields.update(overrides)
    return VideoMetricsRecord(**fields)


@pytest.fixture
def store(project):
    logger = log_run(project, "run-1")
    return VideoMetricsStore(project / "benchmarks" / "metrics", logger)


def test_store_saves_under_run_and_version_and_roundtrips(store, project):
    record = sample_record()
    path = store.save(record)
    assert path == project / "benchmarks" / "metrics" / BENCH / "run-1" / f"metrics_v{METRICS_VERSION}.json"
    assert store.exists(BENCH, "run-1") and store.load(BENCH, "run-1") == record
    assert json.loads(path.read_text(encoding="utf-8"))["video_sha256"] == "ab" * 32


def test_store_never_overwrites(store):
    store.save(sample_record())
    before = store.path_for(BENCH, "run-1").read_bytes()
    with pytest.raises(FileExistsError):
        store.save(sample_record(metrics={"motion_mean": 99.0}))
    assert store.path_for(BENCH, "run-1").read_bytes() == before


def test_a_new_metrics_version_is_a_new_file(store):
    store.save(sample_record())
    store.save(sample_record(metrics_version="1"))
    assert store.exists(BENCH, "run-1", "1") and store.exists(BENCH, "run-1", METRICS_VERSION)


def test_store_rejects_failed_unknown_and_videoless_runs(store, project):
    log_run(project, "failed-run", status="failed")
    with pytest.raises(ValueError, match="did not succeed"):
        store.save(sample_record("failed-run"))
    with pytest.raises(FileNotFoundError):
        store.save(sample_record("no-such-run"))
    assert not (project / "benchmarks" / "metrics" / BENCH / "failed-run").exists()
    with pytest.raises(ValueError, match="did not succeed"):
        store.measure_run(BENCH, "failed-run", project)


@pytest.mark.parametrize(
    "bad",
    [
        {"video_sha256": "abc"},
        {"video_sha256": "AB" * 32},
        {"analysis_size": {"width": 320}},
        {"analysis_size": {"width": 320, "height": 0}},
        {"metrics": {}},
        {"per_transition_diffs": ["0.7"]},
        {"tool_versions": {"numpy": "1"}},
        {"run_id": "../escape"},
        {"metrics_version": "1/2"},
    ],
)
def test_invalid_metrics_records_rejected(bad):
    with pytest.raises(ValueError):
        sample_record(**bad)


def test_nan_metrics_cannot_be_stored(store):
    with pytest.raises(ValueError):
        store.save(sample_record(metrics={"motion_mean": float("nan")}))
    assert not store.exists(BENCH, "run-1")


@needs_ffmpeg
def test_measure_run_uses_the_derived_start_frame_and_records_the_video_hash(project):
    frames = np.stack([np.roll(textured(), i, axis=1) for i in range(9)])
    video = write_video(project / "outputs" / BENCH / "run-1" / "out.mp4", frames, size=(1280, 704))
    derived = f"outputs/{BENCH}/run-1/start_frame_1280x704.png"
    write_image(project / derived, np.kron(frames[0], np.ones((4, 4))).astype(np.uint8))
    logger = log_run(project, "run-1", start_frame={
        "path": "benchmarks/references/frame.png", "sha256": "cd" * 32, "original_width": 1672,
        "original_height": 941, "preprocessing": [], "effective_width": 1280, "effective_height": 704,
        "derived_file": derived,
    })
    store = VideoMetricsStore(project / "benchmarks" / "metrics", logger)

    record = store.measure_run(BENCH, "run-1", project)

    assert store.load(BENCH, "run-1") == record
    assert record.metrics_version == METRICS_VERSION
    assert record.analysis_size == {"width": 320, "height": 176}
    assert record.video_sha256 == sha256_file(video)
    assert record.metrics["start_frame_psnr"] > 40
    assert record.metrics["frame_count"] == 9 and len(record.per_transition_diffs) == 8
    assert record.tool_versions["numpy"] == np.__version__ and record.tool_versions["ffmpeg"]
    assert store.video_matches(record, video)

    with pytest.raises(FileExistsError):  # measured once; never again for this version
        store.measure_run(BENCH, "run-1", project)

    # the run record itself is untouched: metrics live beside it
    assert "metrics" not in json.loads(logger.path_for(BENCH, "run-1").read_text(encoding="utf-8"))["quality_metrics"]

    write_video(video, frames[::-1].copy(), size=(1280, 704))  # a different file under the same name
    assert not store.video_matches(record, video)


@needs_ffmpeg
def test_measure_run_without_a_local_video(project):
    logger = log_run(project, "run-1")
    store = VideoMetricsStore(project / "benchmarks" / "metrics", logger)
    with pytest.raises(FileNotFoundError, match="not on this machine"):
        store.measure_run(BENCH, "run-1", project)


# --- scripts/measure_run.py ---

def load_measure_script():
    spec = importlib.util.spec_from_file_location("script_measure_run", REPO_ROOT / "scripts" / "measure_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_ffmpeg
def test_measure_run_script(project, monkeypatch):
    script = load_measure_script()
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    log_run(project, "20261001T000000Z-aaaaaaaa", steps=2, purpose="smoke")
    log_run(project, "20261002T000000Z-bbbbbbbb", status="failed")
    log_run(project, "20261003T000000Z-cccccccc", steps=50)  # its video is not on this machine
    video = write_video(project / "outputs" / BENCH / "20261001T000000Z-aaaaaaaa" / "out.mp4", pulsing_frames())

    def run(**kwargs):
        lines = []
        return script.measure_runs(BENCH, out=lines.append, **kwargs), lines

    code, lines = run(measure_all=True)
    assert code == 1  # one selected run could not be measured
    assert lines[1].split() == ["run", "purpose", "steps", "motion_mean", "total_change", "start_psnr",
                                "pulse_ratio", "flicker", "held_ratio", "note"]
    measured, failed, absent = lines[2], lines[3], lines[4]
    assert measured.split()[:3] == ["20261001T000000Z-aaaaaaaa", "smoke", "2"]
    assert "measured (no start frame)" in measured
    assert float(measured.split()[6]) > 2  # pulse ratio column
    assert failed.split()[1] == "-"  # a record written before purposes existed
    assert "skipped: failed run" in failed
    assert "skipped: video not on this machine" in absent
    metrics_root = project / "benchmarks" / "metrics" / BENCH
    assert [p.name for p in metrics_root.iterdir()] == ["20261001T000000Z-aaaaaaaa"]

    stored = (metrics_root / "20261001T000000Z-aaaaaaaa" / f"metrics_v{METRICS_VERSION}.json").read_bytes()
    code, lines = run(run_id="20261001T000000Z-aaaaaaaa")
    assert code == 0 and "already measured" in lines[2] and "WARNING" not in lines[2]
    assert (metrics_root / "20261001T000000Z-aaaaaaaa" / f"metrics_v{METRICS_VERSION}.json").read_bytes() == stored

    write_video(video, static_frames())  # the file changed after it was measured
    code, lines = run(run_id="20261001T000000Z-aaaaaaaa")
    assert "WARNING: the video file has changed" in lines[2]

    code, lines = run()  # default: the newest run only
    assert len(lines) == 3 and lines[2].startswith("20261003T000000Z-cccccccc")

    assert run(run_id="nope")[0] == 1
    monkeypatch.setattr(script, "print_table", lambda *a, **k: None)
    assert script.measure_runs("no_such_benchmark", out=lambda line: None) == 1
