"""A static review page for comparing runs by eye, and the import of the
ratings saved from it.

    make page      runs (and post variants) of ONE benchmark -> one HTML file
                   outputs/reviews/<benchmark>/<timestamp>/review.html
    rate           open the file by double-clicking it; no server, no network
    save           the page downloads a ratings JSON file
    import         import_ratings() puts each rating into the comic
                   evaluation store (append-only, like every rating)

The page is self-contained: its style, script and data are inside the one
file, and it loads nothing from the network. The videos are NOT copied or
embedded; the page points at them with relative paths, so it must stay where
it was written (inside the project's outputs/ folder).

What the page shows: the videos side by side, played, paused, sought and
stepped frame by frame together; the stored automatic metrics of each (the
newest metrics version that exists; nothing is measured here); and a form
with the ComicAnimationEvaluation criteria, 1-10, higher is better.

An item is a run's own video ("RUN_ID") or a post-processed variant of it
("RUN_ID:post/<profile_id>"). Ratings of a run's own video go to the comic
evaluation store. Ratings of a VARIANT go to a separate store,
benchmarks/comic_evaluations_variants/<variant>/, with the same layout and
rules, so a rating of a styled video can never be mistaken for a rating of
the model's own output.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ai_studio.benchmarking.comic_evaluation import (
    COMIC_CRITERIA,
    ComicAnimationEvaluation,
    ComicAnimationEvaluationStore,
)
from ai_studio.benchmarking.evaluation_store import SCORE_MAX, SCORE_MIN
from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.generation.capabilities import Capability
from ai_studio.paths import (
    comic_evaluations_dir,
    metrics_dir,
    outputs_dir,
    resolve_path,
    results_dir,
    to_project_relative,
)
from ai_studio.quality.store import sha256_file

RATINGS_FORMAT = "ai_studio.comic_ratings"
RATINGS_VERSION = 1
PAGE_NAME = "review.html"
VARIANT_EVALUATIONS_DIR = "benchmarks/comic_evaluations_variants"
DEFAULT_FPS = 24.0
# Metrics shown in the table, in this order (others stay in the metrics files).
TABLE_METRICS = (
    "frame_count", "motion_mean", "motion_max", "total_change", "start_frame_psnr", "temporal_pulse_ratio",
    "luminance_flicker", "held_frame_ratio", "hold_pattern_periodicity",
)
_METRICS_FILE = re.compile(r"^metrics_v(\d+)\.json$")


class ReviewError(RuntimeError):
    """The page cannot be made, or the ratings cannot be imported, as asked."""


@dataclass(frozen=True)
class ReviewItem:
    benchmark_id: str
    run_id: str
    variant: str | None  # None: the run's own video
    video: Path  # absolute
    fps: float
    frame_count: int | None
    video_sha256: str | None  # of the measured / recorded file, when known
    info: dict[str, Any] = field(default_factory=dict)  # seed, steps, backend, ...
    metrics: dict[str, Any] | None = None  # the stored metrics dict, or None
    metrics_version: str | None = None
    region_ratios: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        return self.run_id if self.variant is None else f"{self.run_id}:{self.variant}"


def parse_item_spec(spec: str) -> tuple[str, str | None]:
    """"RUN_ID" -> (run_id, None); "RUN_ID:post/profile" -> (run_id, "post/profile")."""
    run_id, separator, variant = spec.partition(":")
    check_file_name_safe("run_id", run_id)
    if not separator:
        return run_id, None
    check_variant(variant)
    return run_id, variant


def check_variant(variant: Any) -> None:
    if not isinstance(variant, str) or not variant or "\\" in variant:
        raise ValueError(f"variant must look like post/<profile_id>, got {variant!r}")
    parts = variant.split("/")
    if len(parts) != 2 or parts[0] != "post":
        raise ValueError(f"variant must look like post/<profile_id>, got {variant!r}")
    for part in parts:
        check_file_name_safe("variant", part)


def _newest_metrics(folder: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(file content, version) of the highest metrics_v<N>.json in a folder."""
    if not folder.is_dir():
        return None, None
    found = [(int(m.group(1)), path) for path in folder.iterdir() if (m := _METRICS_FILE.match(path.name))]
    if not found:
        return None, None
    version, path = max(found)
    return json.loads(path.read_text(encoding="utf-8")), str(version)


def collect_item(root: Path, logger: BenchmarkLogger, benchmark_id: str, spec: str) -> ReviewItem:
    """Everything the page needs about one run or variant. ReviewError if it
    cannot be reviewed (unknown, failed, not a video, or its file is missing)."""
    try:
        run_id, variant = parse_item_spec(spec)
    except ValueError as exc:
        raise ReviewError(str(exc)) from None
    try:
        record = logger.load(benchmark_id, run_id)
    except FileNotFoundError:
        raise ReviewError(f"no recorded run {benchmark_id}/{run_id}") from None
    if record.status != "success" or not record.output_video:
        raise ReviewError(f"run {run_id} has no video to review (status {record.status})")
    info = {
        "backend": record.backend, "model": record.model, "seed": record.seed,
        "steps": record.inference_steps, "purpose": record.task_parameters.get("run_purpose"),
    }
    if variant is None:
        video = resolve_path(record.output_video, root)
        stored, version = _newest_metrics(metrics_dir(root) / benchmark_id / run_id)
        fps, frame_count = record.fps, record.frame_count
        digest = (stored or {}).get("video_sha256")
    else:
        folder = outputs_dir(root) / benchmark_id / run_id / Path(*variant.split("/"))
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            raise ReviewError(f"run {run_id} has no variant {variant} on this machine")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        video = resolve_path(manifest["output_video"], root)
        stored, version = _newest_metrics(folder)
        fps, frame_count = manifest.get("fps"), manifest.get("frame_count")
        digest = manifest.get("output_video_sha256")
        info["profile"] = manifest.get("recipe")
    if not video.is_file():
        raise ReviewError(f"video not on this machine: {to_project_relative(video, root)}")
    return ReviewItem(
        benchmark_id=benchmark_id, run_id=run_id, variant=variant, video=video,
        fps=float(fps) if fps else DEFAULT_FPS, frame_count=frame_count, video_sha256=digest, info=info,
        metrics=(stored or {}).get("metrics"), metrics_version=version,
        region_ratios=((stored or {}).get("region_metrics") or {}).get("ratios"),
    )


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metrics_table(items: list[ReviewItem]) -> str:
    def row(label: str, values: list[Any], header: bool = False) -> str:
        tag = "th" if header else "td"
        cells = "".join(f"<{tag}>{html.escape(_format(v))}</{tag}>" for v in values)
        return f"<tr><th>{html.escape(label)}</th>{cells}</tr>"

    rows = [row("", [item.key for item in items], header=True)]
    for name in ("backend", "seed", "steps", "purpose", "profile"):
        values = [item.info.get(name) for item in items]
        if any(value is not None for value in values):
            rows.append(row(name, values))
    rows.append(row("metrics version", [item.metrics_version for item in items]))
    for name in TABLE_METRICS:
        rows.append(row(name, [(item.metrics or {}).get(name) for item in items]))
    regions = sorted({name for item in items for name in (item.region_ratios or {})})
    for region in regions:
        measures = sorted({m for item in items for m in ((item.region_ratios or {}).get(region) or {})})
        for measure in measures:
            rows.append(row(
                f"{region} / rest: {measure}",
                [((item.region_ratios or {}).get(region) or {}).get(measure) for item in items],
            ))
    return "<table class=\"metrics\">\n" + "\n".join(rows) + "\n</table>"


def build_page(items: list[ReviewItem], page_dir: Path, *, page_id: str, created: str) -> str:
    """The complete HTML document for these items, to be written into page_dir."""
    if not items:
        raise ReviewError("nothing to review: give at least one run")
    benchmarks = {item.benchmark_id for item in items}
    if len(benchmarks) != 1:
        raise ReviewError("a review page shows runs of one benchmark")
    keys = [item.key for item in items]
    if len(set(keys)) != len(keys):
        raise ReviewError("the same run or variant was given twice")
    (benchmark_id,) = benchmarks
    fps_values = sorted({item.fps for item in items})
    data = {
        "format": RATINGS_FORMAT,
        "version": RATINGS_VERSION,
        "page_id": page_id,
        "benchmark_id": benchmark_id,
        "created": created,
        "fps": items[0].fps,
        "criteria": list(COMIC_CRITERIA),
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "items": [
            {"key": item.key, "run_id": item.run_id, "variant": item.variant,
             "video_sha256": item.video_sha256, "frame_count": item.frame_count}
            for item in items
        ],
    }
    videos, forms = [], []
    for index, item in enumerate(items):
        source = quote(Path(os.path.relpath(item.video, page_dir)).as_posix())
        label = html.escape(item.key)
        videos.append(
            f'<figure class="clip"><figcaption>{label}</figcaption>'
            f'<video data-index="{index}" src="{source}" muted playsinline preload="auto"></video></figure>'
        )
        inputs = "".join(
            f'<label title="{html.escape(description)}"><span>{html.escape(name)}</span>'
            f'<input type="number" min="{SCORE_MIN}" max="{SCORE_MAX}" step="1" '
            f'data-item="{index}" data-criterion="{html.escape(name)}"></label>'
            for name, description in COMIC_CRITERIA.items()
        )
        forms.append(
            f'<fieldset class="rating"><legend>{label}</legend>{inputs}'
            f'<label class="notes"><span>notes</span><textarea data-item="{index}" data-notes="1" rows="2">'
            f'</textarea></label></fieldset>'
        )
    criteria = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(description)}</td></tr>"
        for name, description in COMIC_CRITERIA.items()
    )
    note = ""
    if len(fps_values) > 1:
        note = (f'<p class="warn">The videos have different frame rates ({", ".join(f"{v:g}" for v in fps_values)}); '
                f"frame stepping uses {items[0].fps:g} fps.</p>")
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = f"Review: {benchmark_id}"
    return _TEMPLATE.format(
        title=html.escape(title), benchmark=html.escape(benchmark_id), created=html.escape(created),
        count=len(items), videos="\n".join(videos), forms="\n".join(forms), note=note,
        metrics=_metrics_table(items), criteria=criteria, data=embedded, style=_STYLE, script=PAGE_SCRIPT,
        score_min=SCORE_MIN, score_max=SCORE_MAX,
    )


def write_review_page(
    root: str | Path, benchmark_id: str, specs: list[str], *, timestamp: str | None = None,
) -> Path:
    """Make outputs/reviews/<benchmark>/<timestamp>/review.html. Never overwrites."""
    root = Path(root).resolve()
    check_file_name_safe("benchmark_id", benchmark_id)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    check_file_name_safe("timestamp", timestamp)
    logger = BenchmarkLogger(results_dir(root))
    items = [collect_item(root, logger, benchmark_id, spec) for spec in specs]
    page_dir = outputs_dir(root) / "reviews" / benchmark_id / timestamp
    if page_dir.exists():
        raise FileExistsError(f"review page already exists: {page_dir}")
    page = build_page(
        items, page_dir, page_id=f"{benchmark_id}/{timestamp}", created=datetime.now(timezone.utc).isoformat(),
    )
    page_dir.mkdir(parents=True, exist_ok=False)
    path = page_dir / PAGE_NAME
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(page)
    return path


# --- importing saved ratings ---

def variant_evaluations_dir(root: str | Path, variant: str) -> Path:
    check_variant(variant)
    return resolve_path(f"{VARIANT_EVALUATIONS_DIR}/{variant}", root)


def _store_for(root: Path, logger: BenchmarkLogger, variant: str | None) -> ComicAnimationEvaluationStore:
    folder = comic_evaluations_dir(root) if variant is None else variant_evaluations_dir(root, variant)
    return ComicAnimationEvaluationStore(folder, results=logger)


def _rated_video(root: Path, logger: BenchmarkLogger, benchmark_id: str, run_id: str, variant: str | None) -> Path:
    record = logger.load(benchmark_id, run_id)
    if variant is None:
        return resolve_path(record.output_video, root)
    return outputs_dir(root) / benchmark_id / run_id / Path(*variant.split("/")) / "out.mp4"


def import_ratings(data: Any, root: str | Path, *, revise: bool = False) -> list[Path]:
    """Store the ratings of a file saved from a review page. Returns the
    files written.

    Everything is checked first; if anything is wrong, ReviewError is raised
    and NOTHING is written. A first rating is saved as revision 1. If the
    evaluator already rated a run, the import is refused unless revise=True,
    which stores an explicit new revision (earlier ones are kept)."""
    root = Path(root).resolve()
    if not isinstance(data, dict) or data.get("format") != RATINGS_FORMAT:
        raise ReviewError(f"not a ratings file saved from a review page (format must be {RATINGS_FORMAT!r})")
    if data.get("version") != RATINGS_VERSION:
        raise ReviewError(f"unknown ratings file version {data.get('version')!r}")
    ratings = data.get("ratings")
    if not isinstance(ratings, list) or not ratings:
        raise ReviewError("the file contains no ratings")
    benchmark_id, evaluator = data.get("benchmark_id"), data.get("evaluator")
    logger = BenchmarkLogger(results_dir(root))
    planned: list[tuple[ComicAnimationEvaluationStore, ComicAnimationEvaluation, bool]] = []
    seen = set()
    for index, rating in enumerate(ratings):
        where = f"rating {index + 1}"
        if not isinstance(rating, dict):
            raise ReviewError(f"{where}: must be a mapping")
        run_id, variant, scores = rating.get("run_id"), rating.get("variant"), rating.get("scores")
        where = f"rating {index + 1} ({run_id}{':' + variant if isinstance(variant, str) else ''})"
        try:
            if variant is not None:
                check_variant(variant)
            if not isinstance(scores, dict) or set(scores) != set(COMIC_CRITERIA):
                raise ValueError(f"scores must have exactly the criteria {sorted(COMIC_CRITERIA)}")
            notes = rating.get("notes") or ""
            if not isinstance(notes, str):
                raise ValueError("notes must be text")
            lines = [notes.strip()]
            if variant is not None:
                lines.append(f"variant: {variant}")
            digest = rating.get("video_sha256")
            if digest:
                lines.append(f"rated file sha256: {digest}")
            lines.append(f"rated on review page {data.get('page_id')}")
            evaluation = ComicAnimationEvaluation(
                benchmark_id=benchmark_id, run_id=run_id, evaluator=evaluator,
                notes="\n".join(line for line in lines if line), **scores,
            )
        except (ValueError, TypeError) as exc:
            raise ReviewError(f"{where}: {exc}") from None
        if (run_id, variant) in seen:
            raise ReviewError(f"{where}: rated twice in this file")
        seen.add((run_id, variant))
        try:
            record = logger.load(benchmark_id, run_id)
        except FileNotFoundError:
            raise ReviewError(f"{where}: no recorded run {benchmark_id}/{run_id}") from None
        if record.status != "success":
            raise ReviewError(f"{where}: the run did not succeed; nothing to rate")
        if record.task_type != Capability.VIDEO_GENERATION.value:
            raise ReviewError(f"{where}: comic ratings apply to video_generation runs, this is {record.task_type!r}")
        video = _rated_video(root, logger, benchmark_id, run_id, variant)
        if digest and video.is_file() and sha256_file(video) != digest:
            raise ReviewError(
                f"{where}: the video on this machine is not the file that was rated (sha256 differs)"
            )
        store = _store_for(root, logger, variant)
        try:
            store.load(benchmark_id, run_id, evaluator)
            exists = True
        except FileNotFoundError:
            exists = False
        if exists and not revise:
            raise ReviewError(
                f"{where}: {evaluator} already rated this; import with revise to store a new revision"
            )
        planned.append((store, evaluation, exists))
    return [
        store.save_revision(evaluation) if exists else store.save(evaluation)
        for store, evaluation, exists in planned
    ]


# --- the page itself ---

_STYLE = """
:root { color-scheme: dark; }
body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #16171a; color: #e6e6e6; }
header, section { padding: 10px 16px; }
h1 { font-size: 18px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 8px; }
.sub { color: #9aa0a6; }
.warn { color: #ffb74d; }
.clips { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 8px; }
.clip { margin: 0; background: #000; }
.clip figcaption { padding: 3px 6px; font: 12px ui-monospace, monospace; background: #24262b; }
.clip video { width: 100%; display: block; cursor: pointer; }
.bar { position: sticky; top: 0; z-index: 2; background: #1e2024; border-bottom: 1px solid #33363c;
       display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 16px; }
.bar input[type=range] { flex: 1 1 240px; }
button, select, input, textarea { font: inherit; color: inherit; background: #2a2d33; border: 1px solid #454952;
       border-radius: 4px; padding: 4px 8px; }
button { cursor: pointer; } button:hover { background: #353941; }
#position { font: 12px ui-monospace, monospace; min-width: 15em; }
table { border-collapse: collapse; } th, td { border: 1px solid #33363c; padding: 3px 8px; text-align: right; }
th:first-child, .criteria td { text-align: left; } table.metrics { font: 12px ui-monospace, monospace; }
.ratings { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; }
.rating { border: 1px solid #33363c; border-radius: 4px; }
.rating legend { font: 12px ui-monospace, monospace; }
.rating label { display: flex; justify-content: space-between; gap: 8px; margin: 2px 0; }
.rating input { width: 4.5em; } .rating .notes { display: block; } .rating textarea { width: 100%; box-sizing: border-box; }
.rating input.bad { border-color: #ef5350; }
#message { min-height: 1.4em; } #message.error { color: #ef5350; } #message.ok { color: #81c784; }
details { margin-top: 8px; }
"""

# Pure functions first (they are tested with node), then the page wiring.
PAGE_SCRIPT = r"""
'use strict';

// Time (seconds) that shows frame `frame`: the middle of the frame, so that
// rounding never lands on the neighbour.
function frameTime(frame, fps) {
  return (Math.max(0, frame) + 0.5) / fps;
}

function frameAt(time, fps) {
  return Math.max(0, Math.floor(time * fps + 1e-6));
}

function safeName(text) {
  return typeof text === 'string' && text.length > 0 && text === text.trim() &&
    !/[\/\\:*?"<>|]/.test(text) && text !== '.' && text !== '..';
}

// values: one entry per item, {scores: {criterion: string}, notes: string}.
// Returns {ok, errors: [...], bad: [[itemIndex, criterion], ...], file} where
// file is the ratings document (items left completely empty are skipped).
function collectRatings(page, evaluator, values, now) {
  var errors = [], bad = [], ratings = [], skipped = [];
  if (!safeName(evaluator)) {
    errors.push('Enter your evaluator name (it becomes a file name: no spaces at the ends, none of / \\ : * ? " < > |).');
  }
  page.items.forEach(function (item, index) {
    var given = values[index] || {scores: {}, notes: ''};
    var filled = page.criteria.filter(function (name) {
      return String(given.scores[name] == null ? '' : given.scores[name]).trim() !== '';
    });
    if (filled.length === 0) {
      skipped.push(item.key);
      if ((given.notes || '').trim() !== '') {
        errors.push(item.key + ': notes without scores; fill in the scores or clear the notes.');
      }
      return;
    }
    var scores = {}, complete = true;
    page.criteria.forEach(function (name) {
      var raw = String(given.scores[name] == null ? '' : given.scores[name]).trim();
      var number = Number(raw);
      if (raw === '' || !/^\d+$/.test(raw) || number < page.score_min || number > page.score_max) {
        complete = false;
        bad.push([index, name]);
      } else {
        scores[name] = number;
      }
    });
    if (!complete) {
      errors.push(item.key + ': every criterion needs a whole number from ' + page.score_min + ' to ' +
        page.score_max + ' (or leave the whole item empty to skip it).');
      return;
    }
    ratings.push({run_id: item.run_id, variant: item.variant, video_sha256: item.video_sha256,
                  scores: scores, notes: (given.notes || '').trim()});
  });
  if (errors.length === 0 && ratings.length === 0) {
    errors.push('Nothing is rated yet.');
  }
  return {
    ok: errors.length === 0, errors: errors, bad: bad,
    file: {format: page.format, version: page.version, page_id: page.page_id,
           benchmark_id: page.benchmark_id, evaluator: evaluator, saved_at: now,
           ratings: ratings, skipped: skipped}
  };
}

if (typeof module !== 'undefined') {
  module.exports = {frameTime: frameTime, frameAt: frameAt, safeName: safeName, collectRatings: collectRatings};
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var page = JSON.parse(document.getElementById('page-data').textContent);
    var videos = Array.prototype.slice.call(document.querySelectorAll('video'));
    var master = videos[0];
    var fps = page.fps;
    var slider = document.getElementById('seek');
    var position = document.getElementById('position');
    var playButton = document.getElementById('play');
    var loop = document.getElementById('loop');
    var speed = document.getElementById('speed');
    var message = document.getElementById('message');
    var storageKey = 'ai_studio.review.' + page.page_id;

    function shortest() {
      var durations = videos.map(function (v) { return v.duration; }).filter(isFinite);
      return durations.length ? Math.min.apply(null, durations) : 0;
    }
    function lastFrame() { return Math.max(0, Math.round(shortest() * fps) - 1); }
    function seekAll(time) { videos.forEach(function (v) { v.currentTime = time; }); }
    function pauseAll() { videos.forEach(function (v) { v.pause(); }); playButton.textContent = 'Play'; }
    function playAll() {
      if (frameAt(master.currentTime, fps) >= lastFrame()) { seekAll(frameTime(0, fps)); }
      else { seekAll(master.currentTime); }
      videos.forEach(function (v) { v.playbackRate = Number(speed.value); v.play(); });
      playButton.textContent = 'Pause';
    }
    function toggle() { if (master.paused) { playAll(); } else { pauseAll(); } }
    function step(delta) {
      pauseAll();
      var frame = Math.min(lastFrame(), Math.max(0, frameAt(master.currentTime, fps) + delta));
      seekAll(frameTime(frame, fps));
    }
    function show() {
      var frame = frameAt(master.currentTime, fps);
      slider.max = String(lastFrame());
      slider.value = String(Math.min(frame, lastFrame()));
      position.textContent = 'frame ' + frame + ' / ' + lastFrame() + '   ' + master.currentTime.toFixed(3) + ' s';
    }
    function tick() {
      if (!master.paused) {
        // keep the others on the first video's clock
        videos.slice(1).forEach(function (v) {
          if (Math.abs(v.currentTime - master.currentTime) > 0.5 / fps) { v.currentTime = master.currentTime; }
        });
        if (master.currentTime >= shortest() - 0.5 / fps) {
          if (loop.checked) { seekAll(frameTime(0, fps)); videos.forEach(function (v) { v.play(); }); }
          else { pauseAll(); }
        }
      }
      show();
      window.requestAnimationFrame(tick);
    }

    playButton.addEventListener('click', toggle);
    document.getElementById('back').addEventListener('click', function () { step(-1); });
    document.getElementById('forward').addEventListener('click', function () { step(1); });
    document.getElementById('first').addEventListener('click', function () { pauseAll(); seekAll(frameTime(0, fps)); });
    slider.addEventListener('input', function () { pauseAll(); seekAll(frameTime(Number(slider.value), fps)); });
    speed.addEventListener('change', function () {
      videos.forEach(function (v) { v.playbackRate = Number(speed.value); });
    });
    videos.forEach(function (v) {
      v.addEventListener('click', toggle);
      v.addEventListener('error', function () {
        message.className = 'error';
        message.textContent = 'A video could not be loaded. The page must stay where it was written, next to the project outputs.';
      });
    });
    document.addEventListener('keydown', function (event) {
      var tag = event.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') { return; }
      if (event.key === ' ') { event.preventDefault(); toggle(); }
      else if (event.key === 'ArrowLeft' || event.key === ',') { event.preventDefault(); step(-1); }
      else if (event.key === 'ArrowRight' || event.key === '.') { event.preventDefault(); step(1); }
      else if (event.key === 'Home') { event.preventDefault(); pauseAll(); seekAll(frameTime(0, fps)); }
    });
    window.requestAnimationFrame(tick);

    // --- ratings ---
    var evaluator = document.getElementById('evaluator');
    function fields() { return Array.prototype.slice.call(document.querySelectorAll('[data-item]')); }
    function readValues() {
      var values = page.items.map(function () { return {scores: {}, notes: ''}; });
      fields().forEach(function (element) {
        var entry = values[Number(element.dataset.item)];
        if (element.dataset.notes) { entry.notes = element.value; }
        else { entry.scores[element.dataset.criterion] = element.value; }
      });
      return values;
    }
    function remember() {
      try {
        window.localStorage.setItem(storageKey, JSON.stringify({evaluator: evaluator.value, values: readValues()}));
      } catch (error) { /* storage is optional */ }
    }
    function restore() {
      try {
        var saved = JSON.parse(window.localStorage.getItem(storageKey) || 'null');
        if (!saved) { return; }
        evaluator.value = saved.evaluator || '';
        fields().forEach(function (element) {
          var entry = saved.values[Number(element.dataset.item)] || {scores: {}, notes: ''};
          element.value = element.dataset.notes ? (entry.notes || '') : (entry.scores[element.dataset.criterion] || '');
        });
      } catch (error) { /* storage is optional */ }
    }
    restore();
    document.addEventListener('input', remember);

    document.getElementById('save').addEventListener('click', function () {
      var result = collectRatings(page, evaluator.value, readValues(), new Date().toISOString());
      fields().forEach(function (element) { element.classList.remove('bad'); });
      result.bad.forEach(function (pair) {
        var element = document.querySelector('[data-item="' + pair[0] + '"][data-criterion="' + pair[1] + '"]');
        if (element) { element.classList.add('bad'); }
      });
      if (!result.ok) {
        message.className = 'error';
        message.textContent = result.errors.join(' ');
        return;
      }
      var name = 'ratings_' + page.benchmark_id + '_' + evaluator.value + '_' +
        result.file.saved_at.replace(/[-:]/g, '').replace(/\..*$/, 'Z') + '.json';
      var blob = new Blob([JSON.stringify(result.file, null, 2) + '\n'], {type: 'application/json'});
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      message.className = 'ok';
      message.textContent = 'Saved ' + result.file.ratings.length + ' rating(s) as ' + name +
        ' (in your downloads folder). Import it with: python scripts/import_ratings.py <that file>';
    });
  });
}
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<header>
<h1>{title}</h1>
<div class="sub">{count} video(s) of benchmark {benchmark} - page made {created} - nothing on this page is sent anywhere</div>
{note}
</header>
<div class="bar">
<button id="first" title="Home">|&lt;</button>
<button id="back" title="Left arrow or ,">&lt; frame</button>
<button id="play" title="Space">Play</button>
<button id="forward" title="Right arrow or .">frame &gt;</button>
<input id="seek" type="range" min="0" max="0" step="1" value="0">
<span id="position">frame 0</span>
<label>speed <select id="speed"><option value="1">1x</option><option value="0.5">0.5x</option><option value="0.25">0.25x</option></select></label>
<label><input id="loop" type="checkbox" checked> loop</label>
</div>
<section class="clips">
{videos}
</section>
<section>
<h2>Automatic metrics (stored values; nothing is measured here)</h2>
{metrics}
</section>
<section>
<h2>Rating - ComicAnimationEvaluation, whole numbers {score_min}-{score_max}, higher is always better</h2>
<p><label>evaluator <input id="evaluator" type="text" placeholder="your name"></label>
<button id="save">Save ratings file</button> <span id="message"></span></p>
<div class="ratings">
{forms}
</div>
<p class="sub">Leave an item completely empty to skip it. The saved file is imported with
<code>python scripts/import_ratings.py FILE</code>; ratings are never overwritten.</p>
<details open><summary>What the criteria mean</summary>
<table class="criteria">
{criteria}
</table>
</details>
</section>
<script type="application/json" id="page-data">{data}</script>
<script>{script}</script>
</body>
</html>
"""
