"""The static review page (generator) and the import of saved ratings."""

import importlib.util
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pytest

from ai_studio import paths
from ai_studio.benchmarking import (
    COMIC_CRITERIA,
    BenchmarkLogger,
    BenchmarkResult,
    ComicAnimationEvaluationStore,
    Resolution,
)
from ai_studio.quality.store import sha256_file
from ai_studio.review import (
    PAGE_SCRIPT,
    RATINGS_FORMAT,
    ReviewError,
    collect_item,
    import_ratings,
    parse_item_spec,
    write_review_page,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
BENCH = "bench_01"
VARIANT = "post/test_finish_v1"
HASH_A, HASH_B = "a" * 64, "b" * 64

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def review_project(project, monkeypatch):
    """Two successful runs (one with a post variant), one failed run, one
    image-less record; metrics for run-1 (v1 and v2) and for the variant."""
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    common = dict(benchmark_id=BENCH, task_type="video_generation", model="m", model_version="1",
                  workflow_version="w", inference_steps=4, guidance=1.0, prompt="p", backend="fake-gen",
                  resolution=Resolution(width=128, height=96), fps=24, frame_count=48,
                  task_parameters={"run_purpose": "experiment"})
    for run_id, seed in (("run-1", 42), ("run-2", 43)):
        video = project / "outputs" / BENCH / run_id / "out.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(f"video of {run_id}".encode())
        logger.log(BenchmarkResult(run_id=run_id, seed=seed, output_video=f"outputs/{BENCH}/{run_id}/out.mp4", **common))
    logger.log(BenchmarkResult(run_id="failed", seed=1, status="failed", failure_stage="generation",
                               error_type="MemoryError", **common))
    logger.log(BenchmarkResult(run_id="gone", seed=2, output_video=f"outputs/{BENCH}/gone/out.mp4", **common))

    metrics = project / "benchmarks" / "metrics" / BENCH / "run-1"
    write_json(metrics / "metrics_v1.json", {"video_sha256": HASH_A, "metrics": {"motion_mean": 9.999}})
    write_json(metrics / "metrics_v2.json", {
        "video_sha256": HASH_A, "metrics_version": "2",
        "metrics": {"frame_count": 48, "motion_mean": 1.23456, "start_frame_psnr": None, "held_frame_ratio": 0.5},
        "region_metrics": {"ratios": {"head": {"edge_density": 0.25, "tone_count": 3.5}}},
    })
    variant = project / "outputs" / BENCH / "run-1" / "post" / "test_finish_v1"
    variant.mkdir(parents=True)
    (variant / "out.mp4").write_bytes(b"styled video")
    write_json(variant / "manifest.json", {
        "output_video": f"outputs/{BENCH}/run-1/post/test_finish_v1/out.mp4", "output_video_sha256": HASH_B,
        "fps": 24, "frame_count": 48, "recipe": "test_finish_v1",
    })
    write_json(variant / "metrics_v2.json", {"video_sha256": HASH_B, "metrics": {"motion_mean": 0.75}})
    return project


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.videos, self.inputs, self.textareas, self.ids, self.external = [], [], [], set(), []
        self.data, self._in_data = "", False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "video":
            self.videos.append(attrs)
        elif tag == "input" and "data-criterion" in attrs:
            self.inputs.append(attrs)
        elif tag == "textarea":
            self.textareas.append(attrs)
        elif tag == "script" and attrs.get("id") == "page-data":
            self._in_data = True
        for name in ("src", "href"):
            if tag != "video" and name in attrs:
                self.external.append(attrs[name])

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_data = False

    def handle_data(self, data):
        if self._in_data:
            self.data += data


# --- item specs ---

def test_parse_item_spec():
    assert parse_item_spec("run-1") == ("run-1", None)
    assert parse_item_spec("run-1:post/comic_finish_v1") == ("run-1", "post/comic_finish_v1")
    for bad in ("", "run-1:", "run-1:post", "run-1:other/x", "run-1:post/../x", "run-1:post/a/b", "a/b"):
        with pytest.raises(ValueError):
            parse_item_spec(bad)


def test_collect_item_reads_the_newest_metrics(review_project):
    logger = BenchmarkLogger(review_project / "benchmarks" / "results")
    raw = collect_item(review_project, logger, BENCH, "run-1")
    assert raw.key == "run-1" and raw.variant is None and raw.fps == 24.0
    assert raw.metrics_version == "2" and raw.metrics["motion_mean"] == 1.23456  # v2, not v1
    assert raw.video_sha256 == HASH_A and raw.info["seed"] == 42
    assert raw.region_ratios == {"head": {"edge_density": 0.25, "tone_count": 3.5}}
    styled = collect_item(review_project, logger, BENCH, f"run-1:{VARIANT}")
    assert styled.key == f"run-1:{VARIANT}" and styled.video.name == "out.mp4"
    assert styled.video_sha256 == HASH_B and styled.metrics == {"motion_mean": 0.75}
    unmeasured = collect_item(review_project, logger, BENCH, "run-2")
    assert unmeasured.metrics is None and unmeasured.metrics_version is None


@pytest.mark.parametrize("spec, match", [
    ("missing", "no recorded run"), ("failed", "no video to review"), ("gone", "not on this machine"),
    (f"run-2:{VARIANT}", "no variant"), ("run-1:post/../../x", "variant"),
])
def test_items_that_cannot_be_reviewed(review_project, spec, match):
    logger = BenchmarkLogger(review_project / "benchmarks" / "results")
    with pytest.raises(ReviewError, match=match):
        collect_item(review_project, logger, BENCH, spec)


# --- the generated page ---

@pytest.fixture
def page_path(review_project):
    return write_review_page(review_project, BENCH, ["run-1", f"run-1:{VARIANT}", "run-2"], timestamp="ts")


def test_page_is_written_under_outputs_reviews(review_project, page_path):
    assert page_path == review_project / "outputs" / "reviews" / BENCH / "ts" / "review.html"
    assert [p.name for p in page_path.parent.iterdir()] == ["review.html"]  # one file, nothing copied
    with pytest.raises(FileExistsError):
        write_review_page(review_project, BENCH, ["run-1"], timestamp="ts")


def test_page_shows_the_videos_by_relative_path(review_project, page_path):
    page = Page(page_path.read_text(encoding="utf-8"))
    assert len(page.videos) == 3
    sources = [unquote(video["src"]) for video in page.videos]
    assert sources == [
        f"../../../{BENCH}/run-1/out.mp4", f"../../../{BENCH}/run-1/post/test_finish_v1/out.mp4",
        f"../../../{BENCH}/run-2/out.mp4",
    ]
    for source in sources:  # each resolves to the real file from where the page lies
        assert (page_path.parent / source).resolve().is_file()
        assert not source.startswith("/") and ":" not in source
    for video in page.videos:
        assert "muted" in video and "controls" not in video  # the page's own controls drive all of them


def test_page_is_self_contained(page_path):
    text = page_path.read_text(encoding="utf-8")
    page = Page(text)
    assert page.external == []  # no stylesheet, script, image or link from anywhere
    for needle in ("http://", "https://", "//cdn", "@import", "fetch(", "XMLHttpRequest", "<link", "<img"):
        assert needle not in text, needle
    assert text.count("<script") == 2 and "<style>" in text
    assert text.startswith("<!DOCTYPE html>")


def test_page_has_synchronized_controls(page_path):
    page = Page(page_path.read_text(encoding="utf-8"))
    assert {"play", "back", "forward", "first", "seek", "position", "speed", "loop", "evaluator", "save",
            "message", "page-data"} <= page.ids


def test_page_data_describes_every_item(page_path):
    data = json.loads(Page(page_path.read_text(encoding="utf-8")).data)
    assert data["format"] == RATINGS_FORMAT and data["version"] == 1
    assert data["benchmark_id"] == BENCH and data["page_id"] == f"{BENCH}/ts" and data["fps"] == 24.0
    assert data["criteria"] == list(COMIC_CRITERIA) and (data["score_min"], data["score_max"]) == (1, 10)
    assert data["items"] == [
        {"key": "run-1", "run_id": "run-1", "variant": None, "video_sha256": HASH_A, "frame_count": 48},
        {"key": f"run-1:{VARIANT}", "run_id": "run-1", "variant": VARIANT, "video_sha256": HASH_B, "frame_count": 48},
        {"key": "run-2", "run_id": "run-2", "variant": None, "video_sha256": None, "frame_count": 48},
    ]


def test_page_has_a_rating_input_for_every_criterion_of_every_item(page_path):
    page = Page(page_path.read_text(encoding="utf-8"))
    assert len(page.inputs) == 3 * len(COMIC_CRITERIA)
    for index in range(3):
        mine = [i["data-criterion"] for i in page.inputs if i["data-item"] == str(index)]
        assert mine == list(COMIC_CRITERIA)
    for field in page.inputs:
        assert (field["type"], field["min"], field["max"], field["step"]) == ("number", "1", "10", "1")
    assert len(page.textareas) == 3
    text = page_path.read_text(encoding="utf-8")
    assert "higher is always better" in text
    assert "10 = no visible distracting artifacts" in text  # the criteria are explained on the page


def test_page_shows_the_metrics_table(page_path):
    text = page_path.read_text(encoding="utf-8")
    table = text.split('<table class="metrics">', 1)[1].split("</table>", 1)[0]
    rows = {m.group(1): re.findall(r"<t[dh]>(.*?)</t[dh]>", m.group(2))
            for m in re.finditer(r"<tr><th>(.*?)</th>(.*?)</tr>", table)}
    assert rows[""] == ["run-1", f"run-1:{VARIANT}", "run-2"]
    assert rows["seed"] == ["42", "42", "43"] and rows["purpose"] == ["experiment"] * 3
    assert rows["profile"] == ["-", "test_finish_v1", "-"]
    assert rows["metrics version"] == ["2", "2", "-"]
    assert rows["motion_mean"] == ["1.235", "0.750", "-"]  # stored values; missing ones are "-"
    assert rows["frame_count"] == ["48", "-", "-"] and rows["start_frame_psnr"] == ["-", "-", "-"]
    assert rows["head / rest: edge_density"] == ["0.250", "-", "-"]
    assert rows["head / rest: tone_count"] == ["3.500", "-", "-"]


def test_page_escapes_text_and_data(review_project):
    logger = BenchmarkLogger(review_project / "benchmarks" / "results")
    video = review_project / "outputs" / BENCH / "run &x 'y" / "out.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"v")
    logger.log(BenchmarkResult(
        benchmark_id=BENCH, run_id="run &x 'y", task_type="video_generation", model="</script><script>alert(1)",
        model_version="1", workflow_version="w", seed=1, inference_steps=4, guidance=1.0, prompt="p",
        backend="b</td>", resolution=Resolution(width=8, height=8), output_video=f"outputs/{BENCH}/run &x 'y/out.mp4",
    ))
    text = write_review_page(review_project, BENCH, ["run &x 'y"], timestamp="esc").read_text(encoding="utf-8")
    assert "run &amp;x &#x27;y" in text and "run &x 'y</" not in text
    assert "b&lt;/td&gt;" in text
    assert text.count("</script>") == 2  # nothing in the data can close a script element
    page = Page(text)
    assert json.loads(page.data)["items"][0]["run_id"] == "run &x 'y"
    assert unquote(page.videos[0]["src"]).endswith("run &x 'y/out.mp4") and " " not in page.videos[0]["src"]


def test_page_needs_items_of_one_benchmark_without_duplicates(review_project):
    with pytest.raises(ReviewError, match="at least one"):
        write_review_page(review_project, BENCH, [], timestamp="a")
    with pytest.raises(ReviewError, match="twice"):
        write_review_page(review_project, BENCH, ["run-1", "run-1"], timestamp="b")
    assert not (review_project / "outputs" / "reviews").exists()


def test_different_frame_rates_are_pointed_out(review_project):
    manifest = review_project / "outputs" / BENCH / "run-1" / "post" / "test_finish_v1" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["fps"] = 12
    manifest.write_text(json.dumps(data), encoding="utf-8")
    text = write_review_page(review_project, BENCH, ["run-1", f"run-1:{VARIANT}"], timestamp="fps").read_text(encoding="utf-8")
    assert "different frame rates (12, 24)" in text


# --- the page script (pure functions, run with node) ---

@needs_node
def test_page_script_is_valid_javascript(tmp_path):
    script = tmp_path / "page.js"
    script.write_text(PAGE_SCRIPT, encoding="utf-8")
    subprocess.run(["node", "--check", str(script)], check=True, capture_output=True)


def run_node(tmp_path, body):
    (tmp_path / "page.js").write_text(PAGE_SCRIPT, encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text("const page = require('./page.js');\n" + body, encoding="utf-8")
    done = subprocess.run(["node", str(harness)], check=True, capture_output=True, text=True, cwd=tmp_path)
    return json.loads(done.stdout)


PAGE_DATA = {
    "format": RATINGS_FORMAT, "version": 1, "page_id": "bench_01/ts", "benchmark_id": BENCH,
    "criteria": list(COMIC_CRITERIA), "score_min": 1, "score_max": 10,
    "items": [
        {"key": "run-1", "run_id": "run-1", "variant": None, "video_sha256": HASH_A},
        {"key": f"run-1:{VARIANT}", "run_id": "run-1", "variant": VARIANT, "video_sha256": HASH_B},
        {"key": "run-2", "run_id": "run-2", "variant": None, "video_sha256": None},
    ],
}


def collect(tmp_path, evaluator, values):
    body = (
        f"const result = page.collectRatings({json.dumps(PAGE_DATA)}, {json.dumps(evaluator)}, "
        f"{json.dumps(values)}, '2026-10-06T08:00:00.000Z');\nconsole.log(JSON.stringify(result));\n"
    )
    return run_node(tmp_path, body)


def filled(score="7", notes=""):
    return {"scores": {name: score for name in COMIC_CRITERIA}, "notes": notes}


EMPTY = {"scores": {name: "" for name in COMIC_CRITERIA}, "notes": ""}


@needs_node
def test_frame_stepping_maths(tmp_path):
    result = run_node(tmp_path, (
        "const out = {times: [0, 1, 23, 24].map(f => page.frameTime(f, 24)),"
        " back: [0, 1, 23, 24, 119].map(f => page.frameAt(page.frameTime(f, 24), 24)),"
        " edge: page.frameAt(1.0, 24), neg: page.frameTime(-3, 24)};\nconsole.log(JSON.stringify(out));\n"
    ))
    assert result["back"] == [0, 1, 23, 24, 119]  # stepping to a frame and reading it back is exact
    assert result["times"][0] == pytest.approx(0.5 / 24) and result["edge"] == 24
    assert result["neg"] == pytest.approx(0.5 / 24)


@needs_node
def test_collect_ratings_builds_the_file(tmp_path):
    result = collect(tmp_path, "reviewer-a", [filled("7", " line boil at 3 s "), EMPTY, filled("10")])
    assert result["ok"] and result["errors"] == [] and result["bad"] == []
    saved = result["file"]
    assert saved["format"] == RATINGS_FORMAT and saved["evaluator"] == "reviewer-a"
    assert saved["benchmark_id"] == BENCH and saved["page_id"] == "bench_01/ts"
    assert saved["saved_at"] == "2026-10-06T08:00:00.000Z"
    assert saved["skipped"] == [f"run-1:{VARIANT}"]
    assert saved["ratings"] == [
        {"run_id": "run-1", "variant": None, "video_sha256": HASH_A,
         "scores": {name: 7 for name in COMIC_CRITERIA}, "notes": "line boil at 3 s"},
        {"run_id": "run-2", "variant": None, "video_sha256": None,
         "scores": {name: 10 for name in COMIC_CRITERIA}, "notes": ""},
    ]


@needs_node
@pytest.mark.parametrize("bad", ["0", "11", "5.5", "abc", "-3", "1e1", " "])
def test_collect_ratings_rejects_bad_or_missing_scores(tmp_path, bad):
    values = filled("6")
    values["scores"]["face_stability"] = bad
    result = collect(tmp_path, "reviewer-a", [values, EMPTY, EMPTY])
    assert not result["ok"] and result["bad"] == [[0, "face_stability"]]
    assert "every criterion needs a whole number from 1 to 10" in result["errors"][0]
    assert result["file"]["ratings"] == []


@needs_node
def test_collect_ratings_needs_an_evaluator_and_at_least_one_rating(tmp_path):
    for name in ("", " reviewer", "a/b", "..", "re:viewer"):
        assert not collect(tmp_path, name, [filled(), EMPTY, EMPTY])["ok"], name
    result = collect(tmp_path, "reviewer-a", [EMPTY, EMPTY, EMPTY])
    assert not result["ok"] and "Nothing is rated yet." in result["errors"]
    orphan = dict(EMPTY, notes="looks good")
    result = collect(tmp_path, "reviewer-a", [filled(), orphan, EMPTY])
    assert not result["ok"] and "notes without scores" in result["errors"][0]


# --- importing ---

def ratings_file(*ratings, **overrides):
    data = {"format": RATINGS_FORMAT, "version": 1, "page_id": f"{BENCH}/ts", "benchmark_id": BENCH,
            "evaluator": "reviewer-a", "saved_at": "2026-10-06T08:00:00.000Z", "ratings": list(ratings),
            "skipped": []}
    data.update(overrides)
    return data


def rating(run_id="run-1", variant=None, score=7, **overrides):
    entry = {"run_id": run_id, "variant": variant, "video_sha256": None,
             "scores": {name: score for name in COMIC_CRITERIA}, "notes": ""}
    entry.update(overrides)
    return entry


def stores(project):
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    return (
        ComicAnimationEvaluationStore(project / "benchmarks" / "comic_evaluations", results=logger),
        ComicAnimationEvaluationStore(project / "benchmarks" / "comic_evaluations_variants" / "post" / "test_finish_v1",
                                      results=logger),
    )


def test_import_stores_raw_and_variant_ratings_apart(review_project):
    raw_hash = sha256_file(review_project / "outputs" / BENCH / "run-1" / "out.mp4")
    written = import_ratings(ratings_file(
        rating("run-1", score=6, notes="line boil at 3 s", video_sha256=raw_hash),
        rating("run-1", VARIANT, score=8), rating("run-2", score=4),
    ), review_project)

    base = review_project / "benchmarks"
    assert written == [
        base / "comic_evaluations" / BENCH / "run-1" / "reviewer-a.r001.json",
        base / "comic_evaluations_variants" / "post" / "test_finish_v1" / BENCH / "run-1" / "reviewer-a.r001.json",
        base / "comic_evaluations" / BENCH / "run-2" / "reviewer-a.r001.json",
    ]
    raw_store, variant_store = stores(review_project)
    raw = raw_store.load(BENCH, "run-1", "reviewer-a")
    assert raw.scores() == {name: 6 for name in COMIC_CRITERIA} and raw.revision == 1
    assert raw.notes.splitlines() == [
        "line boil at 3 s", f"rated file sha256: {raw_hash}", f"rated on review page {BENCH}/ts",
    ]
    styled = variant_store.load(BENCH, "run-1", "reviewer-a")
    assert styled.character_identity == 8 and f"variant: {VARIANT}" in styled.notes
    # the two never mix
    assert [e.character_identity for e in raw_store.list_evaluations(BENCH, "run-1")] == [6]
    assert [e.character_identity for e in variant_store.list_evaluations(BENCH, "run-1")] == [8]


def test_import_never_overwrites_and_revises_only_when_asked(review_project):
    import_ratings(ratings_file(rating(score=5)), review_project)
    raw_store, _ = stores(review_project)
    with pytest.raises(ReviewError, match="already rated"):
        import_ratings(ratings_file(rating(score=9), rating("run-2", score=9)), review_project)
    assert raw_store.load(BENCH, "run-1", "reviewer-a").line_stability == 5
    assert raw_store.list_evaluations(BENCH, "run-2") == []  # all or nothing

    written = import_ratings(ratings_file(rating(score=9), rating("run-2", score=3)), review_project, revise=True)
    assert [path.name for path in written] == ["reviewer-a.r002.json", "reviewer-a.r001.json"]
    assert [e.line_stability for e in raw_store.list_evaluations(BENCH, "run-1")] == [5, 9]  # history kept
    # another evaluator is independent
    import_ratings(ratings_file(rating(score=2), evaluator="reviewer-b"), review_project)
    assert raw_store.load(BENCH, "run-1", "reviewer-b").revision == 1


@pytest.mark.parametrize("data, match", [
    ({"format": "something else"}, "not a ratings file"),
    ([], "not a ratings file"),
    (ratings_file(rating(), version=2), "unknown ratings file version"),
    (ratings_file(), "no ratings"),
    (ratings_file(rating(), evaluator=""), "evaluator"),
    (ratings_file(rating(), evaluator="../x"), "evaluator"),
    (ratings_file(rating(), benchmark_id="../x"), "benchmark_id"),
    (ratings_file(rating("missing")), "no recorded run"),
    (ratings_file(rating("failed")), "did not succeed"),
    (ratings_file(rating(score=11)), "between 1 and 10"),
    (ratings_file(rating(score=5.5)), "must be an int"),
    (ratings_file(rating(score="7")), "must be an int"),
    (ratings_file(rating(scores={"character_identity": 5})), "exactly the criteria"),
    (ratings_file(rating(notes=5)), "notes must be text"),
    (ratings_file(rating(variant="post/../../x")), "variant"),
    (ratings_file(rating(variant="other/x")), "variant"),
    (ratings_file(rating(), rating()), "rated twice"),
    (ratings_file(rating(video_sha256=HASH_A)), "not the file that was rated"),
    (ratings_file("not a mapping"), "must be a mapping"),
])
def test_bad_ratings_file_imports_nothing(review_project, data, match):
    good = rating("run-2")
    if isinstance(data, dict) and isinstance(data.get("ratings"), list) and data["ratings"]:
        data["ratings"].insert(0, good) if match not in ("rated twice",) else None
    with pytest.raises(ReviewError, match=match):
        import_ratings(data, review_project)
    assert not (review_project / "benchmarks" / "comic_evaluations").exists()
    assert not (review_project / "benchmarks" / "comic_evaluations_variants").exists()


def test_import_rejects_image_runs(review_project):
    logger = BenchmarkLogger(review_project / "benchmarks" / "results")
    logger.log(BenchmarkResult(
        benchmark_id=BENCH, run_id="sheet", task_type="image_generation", model="m", model_version="1",
        workflow_version="w", seed=1, inference_steps=None, guidance=None, prompt="p",
        resolution=Resolution(width=8, height=8),
        image_output={"images": [{"path": "outputs/x.png", "sha256": "ab", "width": 8, "height": 8}],
                      "seed_used": 1, "provenance": {}},
    ))
    with pytest.raises(ReviewError, match="video_generation"):
        import_ratings(ratings_file(rating("sheet")), review_project)


def test_hash_check_is_skipped_when_the_video_is_not_here(review_project):
    (review_project / "outputs" / BENCH / "run-2" / "out.mp4").unlink()
    written = import_ratings(ratings_file(rating("run-2", video_sha256=HASH_A)), review_project)
    assert len(written) == 1


# --- the scripts ---

def run_script(script, *argv):
    lines = []
    code = script.main([str(a) for a in argv], out=lines.append)
    return code, "\n".join(lines)


def test_make_review_page_script(review_project):
    script = load_script("make_review_page")
    code, text = run_script(script, "--benchmark", BENCH, "--run", "run-1", "--run", f"run-1:{VARIANT}")
    assert code == 0 and f"review page: outputs/reviews/{BENCH}/" in text
    (folder,) = (review_project / "outputs" / "reviews" / BENCH).iterdir()
    assert len(Page((folder / "review.html").read_text(encoding="utf-8")).videos) == 2

    code, text = run_script(script, "--benchmark", BENCH, "--run", "failed")
    assert code == 1 and "no video to review" in text
    for argv in (["--benchmark", BENCH], ["--benchmark", BENCH, "--run", "run-1", "--all"]):
        with pytest.raises(SystemExit) as info:
            script.main(argv, out=lambda line: None)
        assert info.value.code == 2


def test_make_review_page_all_takes_runs_with_a_local_video(review_project, monkeypatch):
    script = load_script("make_review_page")
    stamps = iter(["all-1"])
    import ai_studio.review as review_module

    real = review_module.write_review_page
    monkeypatch.setattr(script, "write_review_page", lambda root, b, specs: real(root, b, specs, timestamp=next(stamps)))
    code, text = run_script(script, "--benchmark", BENCH, "--all")
    assert code == 0 and "items: run-1, run-2" in text  # not the failed run, not the one whose video is gone
    code, text = run_script(script, "--benchmark", "empty", "--all")
    assert code == 1 and "no successful run" in text


def test_import_ratings_script(review_project, tmp_path):
    script = load_script("import_ratings")
    saved = tmp_path / "ratings.json"
    saved.write_text(json.dumps(ratings_file(rating(score=6), skipped=["run-2"])), encoding="utf-8")
    before = saved.read_bytes()
    code, text = run_script(script, saved)
    assert code == 0 and "imported 1 rating(s) by reviewer-a" in text
    assert f"benchmarks/comic_evaluations/{BENCH}/run-1/reviewer-a.r001.json" in text
    assert "left unrated on the page: run-2" in text
    assert saved.read_bytes() == before  # the ratings file is only read

    code, text = run_script(script, saved)
    assert code == 1 and "already rated" in text and "nothing was imported" in text
    code, text = run_script(script, saved, "--revise")
    assert code == 0 and "reviewer-a.r002.json" in text
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    code, text = run_script(script, broken)
    assert code == 1 and "cannot read" in text
