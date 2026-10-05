"""Region metrics (metrics version 3): how a named region is drawn compared
with the rest of the frame. Synthetic pictures only."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, Resolution, load_definition

np = pytest.importorskip("numpy")

from ai_studio.quality.metrics import METRICS_VERSION, compute_metrics  # noqa: E402
from ai_studio.quality.regions import (  # noqa: E402
    EDGE_ABOVE,
    FLAT_BELOW,
    MEASURES,
    REGION_ANALYSIS_WIDTH,
    AnalysisRegions,
    area_measures,
    gradient_magnitude,
    region_metrics,
)
from ai_studio.quality.store import VideoMetricsRecord, VideoMetricsStore, measure_video  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
W, H = 640, 352
HEAD = (240, 40, 400, 200)  # x0, y0, x1, y1
REGIONS = AnalysisRegions(width=W, height=H, boxes={"head": HEAD})
BENCH = "bench_01"

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


# --- synthetic pictures ---

def inked(height=H, width=W):
    """Flat cel tones with black ink lines: three tone bands, a line every 16 px."""
    image = np.empty((height, width), dtype=np.float32)
    image[:, : width // 3] = 70
    image[:, width // 3: 2 * width // 3] = 150
    image[:, 2 * width // 3:] = 215
    image[:, ::16] = 8
    image[:, 1::16] = 8
    image[::24, :] = 8
    return image


def soft(height, width):
    """A smooth, rendered-looking surface: a radial gradient without any line."""
    y, x = np.mgrid[0:height, 0:width]
    radius = np.hypot(x - width / 2, y - height / 2) / np.hypot(width / 2, height / 2)
    return (215 - 150 * radius).astype(np.float32)


def picture(face="soft"):
    image = inked()
    image[H - 8:, :] = np.linspace(60, 220, W, dtype=np.float32)[None, :]  # a thin soft strip: the rest is not purely flat
    x0, y0, x1, y1 = HEAD
    image[y0:y1, x0:x1] = soft(y1 - y0, x1 - x0) if face == "soft" else inked(y1 - y0, x1 - x0)
    return image


def stack(image, count=3):
    return np.repeat(np.rint(image).astype(np.uint8)[None], count, axis=0)


# --- the measures ---

def test_flat_area_has_no_edges_no_shading_and_one_tone():
    flat = np.full((40, 40), 128, dtype=np.float32)
    values = area_measures(flat.ravel(), gradient_magnitude(flat).ravel())
    assert values["edge_density"] == 0 and values["smooth_gradient_share"] == 0
    assert values["tone_count"] == 1
    assert values["ink_darkness"] == pytest.approx((255 - 128) / 255)


def test_inked_area_has_edges_black_ink_and_few_tones():
    image = inked(96, 96)
    values = area_measures(image.ravel(), gradient_magnitude(image).ravel())
    assert values["edge_density"] > 0.2
    assert values["smooth_gradient_share"] < 0.02
    assert values["ink_darkness"] > 0.95  # the darkest tenth is the ink
    assert values["tone_count"] <= 4


def test_soft_area_is_all_shading_no_ink_and_many_tones():
    image = soft(160, 160)
    values = area_measures(image.ravel(), gradient_magnitude(image).ravel())
    assert values["edge_density"] == 0
    assert values["smooth_gradient_share"] > 0.9  # almost every pixel is gently shaded
    assert values["ink_darkness"] < 0.75 and values["tone_count"] >= 8


def test_gradient_thresholds_split_flat_shading_and_edges():
    ramp = np.tile(np.arange(64, dtype=np.float32) * 5.0, (8, 1))  # 5 luma per pixel: shading
    step = np.zeros((8, 64), dtype=np.float32)
    step[:, 32:] = 200  # one hard edge
    assert FLAT_BELOW < 5.0 < EDGE_ABOVE
    ramp_values = area_measures(ramp.ravel(), gradient_magnitude(ramp).ravel())
    step_values = area_measures(step.ravel(), gradient_magnitude(step).ravel())
    assert ramp_values["smooth_gradient_share"] == 1.0 and ramp_values["edge_density"] == 0
    assert step_values["edge_density"] == pytest.approx(2 / 64) and step_values["smooth_gradient_share"] == 0


# --- region against rest ---

def test_soft_face_on_an_inked_body_shows_up_as_numbers():
    result = region_metrics(stack(picture("soft")), REGIONS)
    ratios = result["ratios"]["head"]
    assert ratios["edge_density"] < 0.1  # far fewer lines in the face than in the rest
    assert ratios["ink_darkness"] < 0.7  # nothing in the face is as black as the ink
    assert ratios["tone_count"] > 2  # many more tones: a gradient, not cel shading
    assert ratios["smooth_gradient_share"] > 3  # much more soft shading
    head, rest = result["regions"]["head"], result["rest"]
    assert head["edge_density"] < rest["edge_density"] and head["tone_count"] > rest["tone_count"]


def test_inked_face_on_an_inked_body_gives_ratios_near_one():
    ratios = region_metrics(stack(picture("inked")), REGIONS)["ratios"]["head"]
    assert 0.6 < ratios["edge_density"] < 1.6
    assert 0.9 < ratios["ink_darkness"] < 1.1
    assert ratios["tone_count"] <= 1.2
    assert ratios["smooth_gradient_share"] < 1.5


def test_result_structure_is_json_safe_and_complete():
    result = region_metrics(stack(picture()), REGIONS)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert set(result) == {"declared", "analysis_size", "regions", "rest", "ratios"}
    assert result["declared"] == {"size": [W, H], "boxes": {"head": list(HEAD)}}
    assert result["analysis_size"] == {"width": W, "height": H}
    head = result["regions"]["head"]
    assert head["box"] == list(HEAD) and head["pixels"] == 160 * 160
    assert result["rest"]["pixels"] == W * H - 160 * 160
    for area in (head, result["rest"]):
        assert set(MEASURES) <= set(area)
    assert set(result["ratios"]["head"]) == set(MEASURES)


def test_values_are_averaged_over_the_frames():
    frames = np.concatenate([stack(picture("soft"), 1), stack(picture("inked"), 1)])
    both = region_metrics(frames, REGIONS)["regions"]["head"]["edge_density"]
    first = region_metrics(frames[:1], REGIONS)["regions"]["head"]["edge_density"]
    second = region_metrics(frames[1:], REGIONS)["regions"]["head"]["edge_density"]
    assert both == pytest.approx((first + second) / 2, abs=1e-6)


def test_several_regions_and_the_rest_excludes_all_of_them():
    regions = AnalysisRegions(width=W, height=H, boxes={"head": HEAD, "hand": (20, 250, 120, 330)})
    result = region_metrics(stack(picture()), regions)
    assert set(result["regions"]) == set(result["ratios"]) == {"head", "hand"}
    assert result["rest"]["pixels"] == W * H - 160 * 160 - 100 * 80


def test_regions_scale_with_the_analysis_size():
    declared = AnalysisRegions(width=1280, height=704, boxes={"head": (480, 80, 800, 400)})
    assert declared.scaled(640, 352) == {"head": HEAD}
    result = region_metrics(stack(picture("soft")), declared)
    assert result["regions"]["head"]["box"] == list(HEAD)
    assert result["declared"]["size"] == [1280, 704]
    with pytest.raises(ValueError, match="only 5x5 px"):
        AnalysisRegions(width=1280, height=704, boxes={"dot": (0, 0, 10, 10)}).scaled(640, 352)


def test_zero_rest_value_gives_no_ratio_instead_of_infinity():
    flat = np.full((1, 64, 64), 100, dtype=np.uint8)
    flat[0, 8:24, 8:24] = np.arange(16, dtype=np.uint8)[None, :] * 12
    result = region_metrics(flat, AnalysisRegions(width=64, height=64, boxes={"a": (8, 8, 24, 24)}))
    assert result["rest"]["smooth_gradient_share"] > 0  # the region's border is in the rest
    still = np.full((1, 64, 64), 100, dtype=np.uint8)
    result = region_metrics(still, AnalysisRegions(width=64, height=64, boxes={"a": (8, 8, 24, 24)}))
    assert result["ratios"]["a"]["edge_density"] is None and result["ratios"]["a"]["tone_count"] == 1.0


@pytest.mark.parametrize("kwargs, match", [
    (dict(width=0, height=10, boxes={"a": (0, 0, 1, 1)}), "frame size"),
    (dict(width=100, height=100, boxes={}), "at least one"),
    (dict(width=100, height=100, boxes={"rest": (0, 0, 10, 10)}), "choose another name"),
    (dict(width=100, height=100, boxes={"my head": (0, 0, 10, 10)}), "letters, digits"),
    (dict(width=100, height=100, boxes={"a": (0, 0, 10)}), "x0, y0, x1, y1"),
    (dict(width=100, height=100, boxes={"a": (0, 0, 10.5, 10)}), "whole pixels"),
    (dict(width=100, height=100, boxes={"a": (10, 0, 10, 10)}), "x0 < x1"),
    (dict(width=100, height=100, boxes={"a": (0, 0, 101, 10)}), "inside"),
    (dict(width=100, height=100, boxes={"a": (-1, 0, 10, 10)}), "inside"),
])
def test_invalid_regions_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AnalysisRegions(**kwargs)


def test_regions_covering_everything_or_bad_frames_rejected():
    everything = AnalysisRegions(width=64, height=64, boxes={"all": (0, 0, 64, 64)})
    with pytest.raises(ValueError, match="nothing is left"):
        region_metrics(np.zeros((1, 64, 64), dtype=np.uint8), everything)
    with pytest.raises(ValueError, match="gray stack"):
        region_metrics(np.zeros((1, 64, 64, 3), dtype=np.uint8), everything)


# --- version 3 and the older metrics ---

def test_version_three_leaves_the_frame_metrics_as_they_were():
    assert METRICS_VERSION == "3"
    metrics = compute_metrics(stack(picture(), 12))
    assert "region_metrics" not in metrics and not any("region" in key for key in metrics)
    assert set(metrics) >= {"motion_mean", "held_frame_ratio", "hold_pattern_periodicity"}


def test_older_metrics_files_load_without_region_metrics():
    old = {
        "benchmark_id": BENCH, "run_id": "r", "metrics_version": "2",
        "analysis_size": {"width": 320, "height": 176}, "video_sha256": "ab" * 32,
        "metrics": {"motion_mean": 1.0}, "per_transition_diffs": [1.0],
        "tool_versions": {"numpy": "2", "ffmpeg": "7"}, "variant": None, "timestamp": "2026-10-04T00:00:00+00:00",
    }
    record = VideoMetricsRecord.from_dict(old)
    assert record.region_metrics is None and record.metrics_version == "2"
    with pytest.raises(ValueError, match="region_metrics"):
        VideoMetricsRecord.from_dict({**old, "region_metrics": {"ratios": {}}})


# --- definitions ---

def write_definition(project, extra, benchmark=BENCH):
    path = project / "benchmarks" / "definitions" / f"{benchmark}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"benchmark_id: {benchmark}\ntask_type: video_generation\nstatus: ready\nprompt: p\n"
        f"duration_seconds: 1\n{extra}", encoding="utf-8",
    )
    return path


def test_definition_declares_named_regions(project):
    definition = load_definition(write_definition(
        project, "analysis_regions:\n  head: [480, 80, 800, 400]\n  hand: [40, 500, 240, 660]\n"
                 "analysis_regions_size: [1280, 704]\n",
    ))
    regions = definition.regions
    assert (regions.width, regions.height) == (1280, 704)
    assert regions.boxes == {"head": (480, 80, 800, 400), "hand": (40, 500, 240, 660)}
    # analysis only: the request a backend receives does not change
    request = definition.to_request(seed=1, output_resolution=Resolution(width=1280, height=704))
    assert not hasattr(request, "analysis_regions") and "analysis_regions" not in request.task_parameters()
    assert load_definition(write_definition(project, "", "plain")).regions is None


@pytest.mark.parametrize("extra, match", [
    ("analysis_regions:\n  head: [480, 80, 800, 400]\n", "analysis_regions_size"),
    ("analysis_regions_size: [1280, 704]\n", "only meaningful with analysis_regions"),
    ("analysis_regions: [1, 2, 3, 4]\nanalysis_regions_size: [1280, 704]\n", "mapping"),
    ("analysis_regions:\n  head: [480, 80, 2000, 400]\nanalysis_regions_size: [1280, 704]\n", "inside"),
    ("analysis_regions:\n  rest: [0, 0, 10, 10]\nanalysis_regions_size: [1280, 704]\n", "another name"),
])
def test_invalid_region_declarations_rejected_at_load(project, extra, match):
    with pytest.raises(ValueError, match=match):
        load_definition(write_definition(project, extra))


def test_repository_definitions_still_load_and_declare_no_regions_yet():
    from ai_studio.benchmarking import load_definitions

    definitions = load_definitions(REPO_ROOT / "benchmarks" / "definitions")
    assert definitions["comic_rooftop_01"].regions is None  # its definition is frozen; not changed here


# --- on real (encoded) video ---

def write_video(path, frames, fps=24):
    height, width = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(frames[..., None], 3, axis=3)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
        input=np.ascontiguousarray(rgb).tobytes(), check=True, capture_output=True,
    )
    return path


@pytest.fixture
def measured_project(project, monkeypatch):
    """A project with one run whose video shows a soft head on an inked body."""
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    write_video(project / "outputs" / BENCH / "run-1" / "out.mp4", stack(picture("soft"), 12))
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    logger.log(BenchmarkResult(
        benchmark_id=BENCH, run_id="run-1", task_type="video_generation", model="m", model_version="1",
        workflow_version="w", seed=1, inference_steps=4, guidance=1.0, prompt="p",
        resolution=Resolution(width=W, height=H), output_video=f"outputs/{BENCH}/run-1/out.mp4",
    ))
    return project, logger


@needs_ffmpeg
def test_measure_video_adds_region_metrics_at_the_larger_analysis_size(measured_project):
    project, _ = measured_project
    video = project / "outputs" / BENCH / "run-1" / "out.mp4"
    record = measure_video(video, benchmark_id=BENCH, run_id="run-1", regions=REGIONS)
    assert record.metrics_version == "3"
    assert record.analysis_size == {"width": 320, "height": 176}  # the frame metrics, as before
    regional = record.region_metrics
    assert regional["analysis_size"] == {"width": REGION_ANALYSIS_WIDTH, "height": H}
    ratios = regional["ratios"]["head"]  # the finding survives H.264 encoding
    assert ratios["edge_density"] < 0.2 and ratios["ink_darkness"] < 0.75
    assert ratios["tone_count"] > 1.5 and ratios["smooth_gradient_share"] > 2
    assert measure_video(video, benchmark_id=BENCH, run_id="run-1").region_metrics is None  # no regions, no block


@needs_ffmpeg
def test_store_writes_version_three_beside_older_files(measured_project):
    project, logger = measured_project
    store = VideoMetricsStore(project / "benchmarks" / "metrics", logger)
    old = store.path_for(BENCH, "run-1", "2")
    old.parent.mkdir(parents=True)
    old.write_text('{"an": "old version-2 file"}\n', encoding="utf-8")

    record = store.measure_run(BENCH, "run-1", project, regions=REGIONS)

    assert old.read_text(encoding="utf-8") == '{"an": "old version-2 file"}\n'  # untouched
    assert sorted(p.name for p in old.parent.iterdir()) == ["metrics_v2.json", "metrics_v3.json"]
    loaded = store.load(BENCH, "run-1")
    assert loaded == record and loaded.region_metrics["ratios"]["head"]["edge_density"] < 0.2
    with pytest.raises(FileExistsError):
        store.measure_run(BENCH, "run-1", project, regions=REGIONS)


def load_measure_script():
    spec = importlib.util.spec_from_file_location("script_measure_run_regions", SCRIPTS / "measure_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_ffmpeg
def test_measure_run_script_uses_the_definitions_regions(measured_project):
    project, logger = measured_project
    write_definition(project, f"analysis_regions:\n  head: {list(HEAD)}\nanalysis_regions_size: [{W}, {H}]\n")
    script = load_measure_script()
    lines = []
    assert script.measure_runs(BENCH, out=lines.append) == 0
    text = "\n".join(lines)
    assert "metrics v3" in lines[0]
    assert "run-1  region head vs rest:" in text
    row = next(line for line in lines if line.strip().startswith("edge_density"))
    assert "ratio 0." in row
    stored = VideoMetricsStore(project / "benchmarks" / "metrics", logger).load(BENCH, "run-1")
    assert stored.region_metrics["declared"]["boxes"] == {"head": list(HEAD)}


@needs_ffmpeg
def test_measure_run_script_without_declared_regions_stores_none(measured_project):
    project, logger = measured_project
    script = load_measure_script()
    lines = []
    assert script.measure_runs(BENCH, out=lines.append) == 0
    assert not any("region" in line for line in lines)
    assert VideoMetricsStore(project / "benchmarks" / "metrics", logger).load(BENCH, "run-1").region_metrics is None


@needs_ffmpeg
def test_trying_regions_from_the_command_line_stores_nothing(measured_project, capsys):
    project, _ = measured_project
    script = load_measure_script()
    code = script.main(["--benchmark", BENCH, "--run-id", "run-1", "--region", "head=240,40,400,200",
                        "--region-size", f"{W}x{H}", "--print-only"])
    out = capsys.readouterr().out
    assert code == 0 and "(not stored)" in out and "region head vs rest:" in out
    assert "smooth_gradient_share" in out and "tone_count" in out
    assert not (project / "benchmarks" / "metrics").exists()

    for argv in (
        ["--benchmark", BENCH, "--run-id", "run-1", "--region", "head=240,40,400,200", "--print-only"],  # no size
        ["--benchmark", BENCH, "--run-id", "run-1", "--region", "head=1,2,3", "--region-size", "640x352", "--print-only"],
        ["--benchmark", BENCH, "--run-id", "run-1", "--region", "head=240,40,400,200", "--region-size", "640x352"],
        ["--benchmark", BENCH, "--print-only"],
    ):
        with pytest.raises(SystemExit) as info:
            script.main(argv)
        assert info.value.code == 2
    capsys.readouterr()
    assert script.main(["--benchmark", BENCH, "--run-id", "run-1", "--print-only"]) == 1  # no regions anywhere
    assert "no regions" in capsys.readouterr().out


def test_parse_regions():
    script = load_measure_script()
    assert script.parse_regions([], None) is None
    regions = script.parse_regions(["head=1,2,30,40", "hand=5,6,70,80"], "1280x704")
    assert regions.boxes == {"head": (1, 2, 30, 40), "hand": (5, 6, 70, 80)} and regions.width == 1280
    for items, size in ((["head=1,2,30,40"], None), ([], "1280x704"), (["head=1,2,30,40"], "big"),
                        (["head"], "1280x704"), (["a=1,2,30,40", "a=1,2,30,40"], "1280x704")):
        with pytest.raises(ValueError):
            script.parse_regions(items, size)
