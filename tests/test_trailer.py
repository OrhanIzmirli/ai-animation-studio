"""Trailers: the shot list format, trailer_01's shot list, the cards and the
assembly. Synthetic clips only; no model, no GPU."""

import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from ai_studio import paths
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, Resolution, load_definition
from ai_studio.post.profile import load_profile
from ai_studio.protection import is_read_only
from ai_studio.quality.store import sha256_file
from ai_studio.trailer import assemble as assemble_module
from ai_studio.trailer.assemble import TrailerError, assemble_trailer, fade_alpha
from ai_studio.trailer.shotlist import load_shotlist, parse_shotlist

np = pytest.importorskip("numpy")
pytest.importorskip("PIL.Image")

from ai_studio.quality.frames import decode_video  # noqa: E402
from ai_studio.trailer.cards import PLACEHOLDER_GREY, placeholder_card, title_card  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TRAILER_01 = REPO_ROOT / "trailers" / "trailer_01" / "shotlist.yaml"
W, H, FPS = 128, 96, 24
BENCH = "bench_01"
PROFILE = {
    "name": "test_finish", "version": 1, "description": "test", "seed": 7,
    "habits": {"timing": {"mode": "fixed", "n": 2}, "grain": {"amount": 2.0}},
}

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def shot(shot_id, seconds=1, **overrides):
    fields = {
        "id": shot_id, "name": f"name of {shot_id}", "source": "generated", "duration_seconds": seconds,
        "needs_character_keyframe": False, "status": "draft", "camera": "wide, static",
        "motion": ["clouds drift"], "prompt": "a city at sunset", "negative_prompt": "blurry", "clip": None,
    }
    fields.update(overrides)
    return fields


def title_shot(shot_id="title", seconds=1, **overrides):
    fields = dict(
        source="title_card", prompt=None, negative_prompt=None, camera="none", motion=[],
        title_card={"text": "A TITLE"},
    )
    fields.update(overrides)
    return shot(shot_id, seconds, **fields)


def shotlist_data(shots, **overrides):
    data = {
        "trailer_id": "t1", "working_title": "Working Title", "status": "draft", "fps": FPS,
        "resolution": {"width": W, "height": H}, "artist_profile": "test_finish_v1",
        "total_duration_seconds": sum(s["duration_seconds"] for s in shots), "shots": shots,
    }
    data.update(overrides)
    return data


# --- trailer_01's shot list ---

def test_trailer_01_is_seven_draft_shots_and_thirty_seconds():
    shotlist = load_shotlist(TRAILER_01)
    assert shotlist.trailer_id == "trailer_01" and shotlist.status == "draft"
    assert [s.duration_seconds for s in shotlist.shots] == [4, 5, 4, 4, 5, 5, 3]
    assert shotlist.duration_seconds == 30 and shotlist.frame_count == 720
    assert [s.id for s in shotlist.shots] == [f"shot_0{n}" for n in range(1, 8)]
    assert all(s.status == "draft" for s in shotlist.shots)
    assert all(s.clip is None for s in shotlist.shots)  # nothing chosen yet: all placeholders
    assert [s.needs_character_keyframe for s in shotlist.shots] == [False, True, True, True, False, True, False]
    for s in shotlist.shots[:6]:
        assert s.source == "generated" and s.prompt and s.negative_prompt and s.camera and s.motion


def test_trailer_01_title_card_is_made_in_editing():
    title = load_shotlist(TRAILER_01).shots[-1]
    assert title.source == "title_card" and title.prompt is None and title.negative_prompt is None
    assert title.title_card.text and title.duration_seconds == 3


def test_trailer_01_shot_two_is_the_rooftop_benchmark_unchanged():
    rooftop = load_shotlist(TRAILER_01).shots[1]
    definition = load_definition(REPO_ROOT / "benchmarks" / "definitions" / "comic_rooftop_01.yaml")
    assert rooftop.benchmark == definition.benchmark_id == "comic_rooftop_01"
    assert rooftop.prompt == definition.prompt and rooftop.negative_prompt == definition.negative_prompt
    assert rooftop.duration_seconds == definition.duration_seconds == 5
    assert rooftop.camera == definition.camera_motion


def test_trailer_01_matches_the_benchmark_format_and_has_a_real_profile():
    shotlist = load_shotlist(TRAILER_01)
    assert (shotlist.fps, shotlist.resolution) == (24, Resolution(width=1280, height=704))
    profile = load_profile(REPO_ROOT / "configs" / "style_recipes" / f"{shotlist.artist_profile}.yaml")
    assert profile.profile_id == shotlist.artist_profile


def test_trailer_01_asks_for_nothing_that_evokes_an_existing_franchise():
    # What the shots ASK for (negative prompts may name things to avoid).
    shotlist = load_shotlist(TRAILER_01)
    wanted = " ".join(
        " ".join([s.name, s.camera, *s.motion, s.prompt or "", s.title_card.text if s.title_card else ""])
        for s in shotlist.shots
    ).lower() + " " + shotlist.working_title.lower()
    for word in ("spider", "web", "mask", "arachn", "superhero", "costume", "emblem", "logo", "cape"):
        assert word not in wanted, word
    # every character shot explicitly excludes masks / costumes
    for s in shotlist.shots:
        if s.needs_character_keyframe:
            assert "superhero costume" in s.negative_prompt


# --- the format ---

def test_parse_shotlist_computes_frames_and_fades():
    shotlist = parse_shotlist(shotlist_data([
        shot("a", 1.5, fade_in_seconds=0.5), shot("b", 2, clip={"run": "bench_01/run-1"}),
        shot("c", 1, clip={"path": "clips/c.mp4"}, motion="one line"), title_shot(fade_out_seconds=1),
    ]))
    assert [s.frame_count for s in shotlist.shots] == [36, 48, 24, 24]
    assert shotlist.frame_count == 132 and shotlist.duration_seconds == 5.5
    assert (shotlist.shots[0].fade_in_frames, shotlist.shots[0].fade_out_frames) == (12, 0)
    assert shotlist.shots[3].fade_out_frames == 24
    assert shotlist.shots[1].clip == {"run": "bench_01/run-1"} and shotlist.shots[2].clip == {"path": "clips/c.mp4"}
    assert shotlist.shots[2].motion == ("one line",)


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(colour="red"), "unknown keys"),
    (lambda d: d.pop("artist_profile"), "missing keys"),
    (lambda d: d.update(total_duration_seconds=99), "add up to"),
    (lambda d: d.update(status="final"), "status must be"),
    (lambda d: d.update(fps=23.976), "fps"),
    (lambda d: d.update(resolution={"width": 127, "height": 96}), "even"),
    (lambda d: d.update(artist_profile="../secret"), "file name"),
    (lambda d: d.update(shots=[]), "non-empty"),
    (lambda d: d["shots"][0].update(seed=1), "unknown keys"),
    (lambda d: d["shots"][0].pop("negative_prompt"), "missing keys"),
    (lambda d: d["shots"][0].update(id="title"), "duplicate shot ids"),
    (lambda d: d["shots"][0].update(status="done"), "status must be"),
    (lambda d: d["shots"][0].update(source="stock"), "source must be"),
    (lambda d: d["shots"][0].update(prompt=None), "prompt must be"),
    (lambda d: d["shots"][0].update(needs_character_keyframe="yes"), "true or false"),
    (lambda d: d["shots"][0].update(duration_seconds=1.01, fade_in_seconds=0), "whole number of frames"),
    (lambda d: d["shots"][0].update(fade_in_seconds=0.75, fade_out_seconds=0.5), "longer than the shot"),
    (lambda d: d["shots"][0].update(clip="clip.mp4"), "clip must be"),
    (lambda d: d["shots"][0].update(clip={"run": "only-a-run-id"}), "benchmark_id>/<run_id>"),
    (lambda d: d["shots"][0].update(clip={"run": "b/..\\x"}), "file name"),
    (lambda d: d["shots"][0].update(title_card={"text": "x"}), "only a title_card shot"),
    (lambda d: d["shots"][1].update(prompt="a title made by a model"), "made in editing"),
    (lambda d: d["shots"][1].update(clip={"path": "x.mp4"}), "no clip"),
    (lambda d: d["shots"][1].update(title_card={"text": ""}), "title_card.text"),
    (lambda d: d["shots"][1].update(title_card={"text": "x", "font": "y"}), "unknown keys"),
])
def test_parse_shotlist_rejects(change, match):
    data = shotlist_data([shot("a"), title_shot()])
    change(data)
    if "total_duration_seconds" not in match and "add up" not in match:
        data["total_duration_seconds"] = sum(s["duration_seconds"] for s in data["shots"]) if data["shots"] else 1
    with pytest.raises(ValueError, match=match):
        parse_shotlist(data)


def test_load_shotlist_checks_the_folder_name(tmp_path):
    path = tmp_path / "other" / "shotlist.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(shotlist_data([shot("a")])), encoding="utf-8")
    with pytest.raises(ValueError, match="must be named t1"):
        load_shotlist(path)


# --- fades ---

def test_fade_alpha():
    assert [fade_alpha(i, 10, 0, 0) for i in range(10)] == [1.0] * 10  # a cut
    assert [fade_alpha(i, 10, 4, 0) for i in range(5)] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [fade_alpha(i, 10, 0, 4) for i in range(5, 10)] == [1.0, 0.75, 0.5, 0.25, 0.0]
    both = [fade_alpha(i, 4, 2, 2) for i in range(4)]
    assert both == [0.0, 0.5, 0.5, 0.0]


# --- cards ---

def test_placeholder_card_is_a_labelled_grey_card():
    shotlist = load_shotlist(TRAILER_01)
    resolution = Resolution(width=640, height=352)
    cards = [placeholder_card(s, n, 7, resolution) for n, s in enumerate(shotlist.shots[:2], start=1)]
    for card in cards:
        assert card.shape == (352, 640, 3) and card.dtype == np.uint8
        grey = (card == PLACEHOLDER_GREY).all(axis=2).mean()
        assert 0.6 < grey < 0.99  # mostly grey, with a label on it
        assert (card[2, 2] == PLACEHOLDER_GREY).all()
    assert not np.array_equal(cards[0], cards[1])  # the label names the shot
    again = placeholder_card(shotlist.shots[0], 1, 7, resolution)
    assert np.array_equal(cards[0], again)  # deterministic


def test_title_card_is_white_text_on_black():
    title = load_shotlist(TRAILER_01).shots[-1].title_card
    card = title_card(title, Resolution(width=640, height=352))
    assert card.shape == (352, 640, 3) and card.dtype == np.uint8
    assert (card[0, 0] == 0).all() and card.max() == 255
    assert 0.001 < (card > 128).mean() < 0.3


# --- assembly ---

def scene(count, base):
    """A moving bright bar on a flat background; `base` tells clips apart."""
    frames = np.full((count, H, W, 3), base, dtype=np.uint8)
    for index in range(count):
        frames[index, 20:60, 10 + index: 30 + index] = 230
    return frames


def write_video(path, frames, fps=FPS):
    height, width = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
        input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True,
    )


def write_shotlist(project, shots, trailer_id="t1", **overrides):
    path = project / "trailers" / trailer_id / "shotlist.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(shotlist_data(shots, trailer_id=trailer_id, **overrides)), encoding="utf-8")
    return path


@pytest.fixture
def trailer_project(project):
    """A project with a recorded run (30 frames), a failed run, a loose clip
    (40 frames) and one profile."""
    write_video(project / "outputs" / BENCH / "run-1" / "out.mp4", scene(30, 60))
    write_video(project / "clips" / "loose.mp4", scene(40, 120))
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    common = dict(benchmark_id=BENCH, task_type="video_generation", model="m", model_version="1",
                  workflow_version="w", seed=1, inference_steps=4, guidance=1.0, prompt="p",
                  resolution=Resolution(width=W, height=H))
    logger.log(BenchmarkResult(run_id="run-1", output_video=f"outputs/{BENCH}/run-1/out.mp4", **common))
    logger.log(BenchmarkResult(run_id="failed", status="failed", failure_stage="generation",
                               error_type="MemoryError", **common))
    recipes = project / "configs" / "style_recipes"
    recipes.mkdir(parents=True)
    (recipes / "test_finish_v1.yaml").write_text(yaml.safe_dump(PROFILE), encoding="utf-8")
    other = copy.deepcopy(PROFILE)
    other["name"] = "other_finish"
    (recipes / "other_finish_v1.yaml").write_text(yaml.safe_dump(other), encoding="utf-8")
    return project


@pytest.fixture
def style_calls(monkeypatch):
    """Records every call into the style layer."""
    calls = []
    real = assemble_module.apply_profile

    def spy(frames, profile, source_image):
        calls.append((len(frames), profile))
        return real(frames, profile, source_image)

    monkeypatch.setattr(assemble_module, "apply_profile", spy)
    return calls


MIXED = [
    ("a", dict(clip={"run": f"{BENCH}/run-1"}, fade_in_seconds=0.5)),
    ("b", dict()),  # placeholder
    ("c", dict(clip={"path": "clips/loose.mp4"}, fade_out_seconds=0.25)),
]


def mixed_shots():
    return [shot(shot_id, 1, **extra) for shot_id, extra in MIXED] + [title_shot(seconds=0.5)]


@needs_ffmpeg
def test_assembly_cuts_clips_cards_and_title_into_one_video(trailer_project, style_calls):
    project = trailer_project
    source = write_shotlist(project, mixed_shots())
    clip_hashes = {p: sha256_file(p) for p in (project / "outputs" / BENCH / "run-1" / "out.mp4",
                                                 project / "clips" / "loose.mp4")}

    manifest = assemble_trailer(project, "t1", timestamp="20261005T010000Z")

    target = project / "outputs" / "trailers" / "t1" / "20261005T010000Z"
    assert sorted(p.name for p in target.iterdir()) == ["manifest.json", "trailer.mp4"]
    assert json.loads((target / "manifest.json").read_text(encoding="utf-8")) == manifest
    video = target / "trailer.mp4"
    assert is_read_only(video)
    assert manifest["output_video"] == "outputs/trailers/t1/20261005T010000Z/trailer.mp4"
    assert manifest["output_video_sha256"] == sha256_file(video)
    assert manifest["shotlist"] == "trailers/t1/shotlist.yaml" and manifest["shotlist_sha256"] == sha256_file(source)
    assert (manifest["fps"], manifest["width"], manifest["height"]) == (FPS, W, H)
    assert manifest["frame_count"] == 84 and manifest["duration_seconds"] == 3.5
    assert manifest["output_media"]["frame_count"] == 84
    assert manifest["complete"] is False and manifest["placeholder_shots"] == ["b"]

    # one artist profile for every clip, through the style layer
    assert manifest["artist_profile"] == "test_finish_v1" and manifest["profile"]["seed"] == 7
    assert manifest["profile_file"] == "configs/style_recipes/test_finish_v1.yaml"
    assert [count for count, _ in style_calls] == [24, 24]  # cut to the shot's length first
    assert style_calls[0][1] is style_calls[1][1] and style_calls[0][1].profile_id == "test_finish_v1"

    a, b, c, title = manifest["shots"]
    assert [s["kind"] for s in manifest["shots"]] == ["clip", "placeholder", "clip", "title_card"]
    assert [s["first_frame"] for s in manifest["shots"]] == [0, 24, 48, 72]
    assert [s["styled"] for s in manifest["shots"]] == [True, False, True, False]
    assert a["video"] == f"outputs/{BENCH}/run-1/out.mp4" and a["clip"] == {"run": f"{BENCH}/run-1"}
    assert (a["clip_frame_count"], a["unused_frames"], a["fade_in_frames"]) == (30, 6, 12)
    assert c["video"] == "clips/loose.mp4" and (c["clip_frame_count"], c["unused_frames"]) == (40, 16)
    assert a["video_sha256"] == clip_hashes[project / "outputs" / BENCH / "run-1" / "out.mp4"]
    assert a["timing"]["drawing_count"] == 12 and a["timing"]["held_frame_ratio"] > 0.4  # on twos
    assert "timing" not in b and title["title_card"]["text"] == "A TITLE"

    # the pictures: fade-in from black, the clips, the grey card, a fade-out, the title
    frames = decode_video(video, analysis_width=None, gray=False).astype(np.float32)
    assert frames.shape == (84, H, W, 3)
    assert frames[0].mean() < 3  # fade-in starts at black
    assert abs(frames[20, 80:, :].mean() - 60) < 6  # clip a's background
    assert abs(frames[30, 2, 2].mean() - PLACEHOLDER_GREY) < 4  # the placeholder card
    assert np.abs(frames[30] - frames[40]).mean() < 1  # ... is a still
    assert abs(frames[50, 80:, :].mean() - 120) < 6  # clip c's background
    assert frames[71].mean() < 3  # fade-out ends at black
    assert frames[75, :4].mean() < 3 and frames[75].max() > 200  # white title on black

    # the sources are untouched
    assert {p: sha256_file(p) for p in clip_hashes} == clip_hashes


@needs_ffmpeg
def test_all_placeholder_assembly_styles_nothing(trailer_project, style_calls):
    write_shotlist(trailer_project, [shot("a"), shot("b", 0.5), title_shot(seconds=0.5)])
    manifest = assemble_trailer(trailer_project, "t1", timestamp="ts")
    assert style_calls == []
    assert manifest["placeholder_shots"] == ["a", "b"] and manifest["complete"] is False
    assert manifest["artist_profile"] == "test_finish_v1"  # still named and recorded
    assert manifest["output_media"]["frame_count"] == 48


@needs_ffmpeg
def test_complete_assembly_and_profile_override(trailer_project, style_calls):
    write_shotlist(trailer_project, [shot("a", clip={"path": "clips/loose.mp4"}), title_shot(seconds=0.5)])
    manifest = assemble_trailer(trailer_project, "t1", profile_name="other_finish_v1", timestamp="ts")
    assert manifest["complete"] is True and manifest["placeholder_shots"] == []
    assert manifest["artist_profile"] == "other_finish_v1"
    assert [profile.profile_id for _, profile in style_calls] == ["other_finish_v1"]
    assert manifest["shots"][0]["palette_source"] == "first frame"


@needs_ffmpeg
def test_an_assembly_is_never_overwritten(trailer_project):
    write_shotlist(trailer_project, [shot("a", 0.5)])
    assemble_trailer(trailer_project, "t1", timestamp="ts")
    video = trailer_project / "outputs" / "trailers" / "t1" / "ts" / "trailer.mp4"
    before = sha256_file(video)
    with pytest.raises(FileExistsError):
        assemble_trailer(trailer_project, "t1", timestamp="ts")
    assert sha256_file(video) == before
    assemble_trailer(trailer_project, "t1", timestamp="ts2")  # another timestamp is another assembly
    assert sorted(p.name for p in video.parents[1].iterdir()) == ["ts", "ts2"]


@needs_ffmpeg
@pytest.mark.parametrize("clip, video, match", [
    ({"run": f"{BENCH}/missing"}, None, "no recorded run"),
    ({"run": f"{BENCH}/failed"}, None, "did not succeed"),
    ({"path": "clips/nowhere.mp4"}, None, "clip not found"),
    ({"path": "clips/big.mp4"}, dict(size=(96, 160)), "never rescaled"),
    ({"path": "clips/slow.mp4"}, dict(fps=12), "12 fps"),
    ({"path": "clips/short.mp4"}, dict(count=10), "needs 24"),
])
def test_unusable_clip_refuses_the_assembly_before_writing_anything(trailer_project, clip, video, match):
    if video is not None:
        height, width = video.get("size", (H, W))
        frames = np.full((video.get("count", 30), height, width, 3), 90, dtype=np.uint8)
        write_video(trailer_project / clip["path"], frames, fps=video.get("fps", FPS))
    write_shotlist(trailer_project, [shot("a"), shot("b", clip=clip)])
    with pytest.raises(TrailerError, match=match):
        assemble_trailer(trailer_project, "t1", timestamp="ts")
    assert not (trailer_project / "outputs" / "trailers").exists()


def test_missing_shotlist_or_profile_is_reported(trailer_project):
    with pytest.raises(FileNotFoundError):
        assemble_trailer(trailer_project, "nope", timestamp="ts")
    write_shotlist(trailer_project, [shot("a")], artist_profile="unknown_v1")
    with pytest.raises(FileNotFoundError):
        assemble_trailer(trailer_project, "t1", timestamp="ts")
    assert not (trailer_project / "outputs" / "trailers").exists()


@needs_ffmpeg
def test_title_card_image_must_match_the_trailer(trailer_project):
    def write_png(path, height, width):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-i", "-", "-frames:v", "1", str(path)],
            input=np.full((height, width, 3), 200, dtype=np.uint8).tobytes(), check=True, capture_output=True,
        )

    (trailer_project / "trailers" / "t1").mkdir(parents=True)
    write_png(trailer_project / "trailers" / "small.png", 48, 64)
    write_png(trailer_project / "trailers" / "t1" / "title.png", H, W)
    card = {"text": "T", "image": "trailers/small.png"}
    write_shotlist(trailer_project, [title_shot(seconds=0.5, title_card=card)])
    with pytest.raises(ValueError, match="title card image is 64x48"):
        assemble_trailer(trailer_project, "t1", timestamp="ts")

    card["image"] = "trailers/t1/title.png"
    write_shotlist(trailer_project, [title_shot(seconds=0.5, title_card=card)])
    manifest = assemble_trailer(trailer_project, "t1", timestamp="ts")
    entry = manifest["shots"][0]["title_card"]
    assert entry["image"] == "trailers/t1/title.png" and entry["image_sha256"]
    frames = decode_video(trailer_project / manifest["output_video"], analysis_width=None, gray=False)
    assert abs(float(frames[5].mean()) - 200) < 4


# --- the script ---

@pytest.fixture
def script(trailer_project, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(trailer_project))
    spec = importlib.util.spec_from_file_location("script_assemble_trailer", SCRIPTS / "assemble_trailer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(script, *argv):
    lines = []
    code = script.main(list(argv), out=lines.append)
    return code, "\n".join(lines)


@needs_ffmpeg
def test_script_assembles_and_reports(script, trailer_project):
    write_shotlist(trailer_project, mixed_shots())
    code, text = run_script(script, "--trailer", "t1")
    assert code == 0
    (assembly,) = (trailer_project / "outputs" / "trailers" / "t1").iterdir()
    assert f"outputs/trailers/t1/{assembly.name}/trailer.mp4" in text
    assert "artist profile: test_finish_v1" in text and "3.5 s, 84 frames" in text
    assert "INCOMPLETE: 1 placeholder shot(s): b" in text
    assert "grey card" in text and "clips/loose.mp4" in text and "A TITLE" in text


def test_script_rejects_unknown_trailer_and_profile(script, trailer_project):
    write_shotlist(trailer_project, [shot("a")])
    code, text = run_script(script, "--trailer", "missing")
    assert code == 2 and "available: t1" in text
    code, text = run_script(script, "--trailer", "../t1")
    assert code == 2
    code, text = run_script(script, "--trailer", "t1", "--profile", "nope_v9")
    assert code == 2 and "other_finish_v1, test_finish_v1" in text
    assert not (trailer_project / "outputs" / "trailers").exists()


@needs_ffmpeg
def test_script_reports_an_unusable_clip(script, trailer_project):
    write_shotlist(trailer_project, [shot("a", clip={"run": f"{BENCH}/failed"})])
    code, text = run_script(script, "--trailer", "t1")
    assert code == 1 and "TrailerError" in text and "did not succeed" in text
