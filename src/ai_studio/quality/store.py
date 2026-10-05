"""Measure a benchmark run's video and store the result.

Run records are immutable, so automatic metrics live next to them, like the
human evaluations do:

    benchmarks/metrics/<benchmark_id>/<run_id>/metrics_v<metrics_version>.json

One file per run and metrics version, created exclusively and never
overwritten. A new METRICS_VERSION gives a new file beside the old one.

A metrics record describes either the run's own video (variant None) or a
derived, post-processed variant of it (variant e.g. "post/comic_finish_v1").
The files above are only ever the raw run's. A variant's metrics are written
next to the variant itself (save_variant), so the two can never be confused.

Only successful runs with an output_video can be measured. The record keeps
the SHA-256 of the measured file, so a later measurement of a different file
under the same run id is detectable (VideoMetricsStore.video_matches).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.benchmarking.schema import BenchmarkResult, check_file_name_safe
from ai_studio.paths import resolve_path
from ai_studio.quality.frames import (
    DEFAULT_ANALYSIS_WIDTH,
    decode_image,
    decode_video,
    ffmpeg_version,
)
from ai_studio.quality.metrics import METRICS_VERSION, compute_metrics, transition_diffs
from ai_studio.quality.regions import REGION_ANALYSIS_WIDTH, AnalysisRegions, region_metrics

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, kw_only=True)
class VideoMetricsRecord:
    benchmark_id: str
    run_id: str
    metrics_version: str
    analysis_size: dict[str, int]  # {"width", "height"} the frames were scaled to
    video_sha256: str  # of the measured file
    metrics: dict[str, Any]  # see ai_studio.quality.metrics.compute_metrics
    per_transition_diffs: list[float]  # mean |diff| of frame i -> i+1, 0-255
    tool_versions: dict[str, str | None]  # {"numpy", "ffmpeg"}
    # None: the run's own output video. Otherwise the derived variant measured,
    # e.g. "post/comic_finish_v1" (absent in files written before variants existed).
    variant: str | None = None
    # Metrics v3+: how each declared analysis region is drawn compared with the
    # rest of the frame (ai_studio.quality.regions.region_metrics). None when
    # no regions were declared, and in files written before version 3.
    region_metrics: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        check_file_name_safe("benchmark_id", self.benchmark_id)
        check_file_name_safe("run_id", self.run_id)
        check_file_name_safe("metrics_version", self.metrics_version)
        if not isinstance(self.video_sha256, str) or not _SHA256_RE.match(self.video_sha256):
            raise ValueError("video_sha256 must be a lowercase hex SHA-256")
        if not isinstance(self.analysis_size, dict) or set(self.analysis_size) != {"width", "height"}:
            raise ValueError("analysis_size must be a dict with width and height")
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in self.analysis_size.values()):
            raise ValueError("analysis_size width and height must be positive ints")
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise ValueError("metrics must be a non-empty dict")
        if not isinstance(self.per_transition_diffs, list) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in self.per_transition_diffs
        ):
            raise ValueError("per_transition_diffs must be a list of numbers")
        if not isinstance(self.tool_versions, dict) or set(self.tool_versions) != {"numpy", "ffmpeg"}:
            raise ValueError("tool_versions must be a dict with numpy and ffmpeg")
        if self.variant is not None and (not isinstance(self.variant, str) or not self.variant.strip()):
            raise ValueError("variant must be a non-empty string or None")
        if self.region_metrics is not None and (
            not isinstance(self.region_metrics, dict)
            or not {"regions", "rest", "ratios"} <= set(self.region_metrics)
        ):
            raise ValueError("region_metrics must be a dict with regions, rest and ratios, or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoMetricsRecord:
        return cls(**data)


def measure_video(
    video: str | Path,
    *,
    benchmark_id: str,
    run_id: str,
    reference_image: str | Path | None = None,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    variant: str | None = None,
    regions: AnalysisRegions | None = None,
) -> VideoMetricsRecord:
    """Decode `video` and compute every metric. `reference_image` is the
    start frame as the model received it (for start_frame_psnr), or None.
    `variant` names a derived video (None: the run's own output). `regions`
    are the benchmark's analysis regions; with them the record also gets
    region_metrics (measured in a second decode at REGION_ANALYSIS_WIDTH)."""
    video = Path(video)
    frames = decode_video(video, analysis_width=analysis_width)
    height, width = frames.shape[1:3]
    reference = None if reference_image is None else decode_image(reference_image, (width, height))
    regional = None
    if regions is not None:
        regional = region_metrics(decode_video(video, analysis_width=REGION_ANALYSIS_WIDTH), regions)
    return VideoMetricsRecord(
        benchmark_id=benchmark_id,
        run_id=run_id,
        metrics_version=METRICS_VERSION,
        analysis_size={"width": int(width), "height": int(height)},
        video_sha256=sha256_file(video),
        metrics=compute_metrics(frames, reference),
        per_transition_diffs=[round(float(v), 4) for v in transition_diffs(frames)],
        tool_versions={"numpy": np.__version__, "ffmpeg": ffmpeg_version()},
        variant=variant,
        region_metrics=regional,
    )


class VideoMetricsStore:
    """Metrics for benchmark runs. `results` is the run-record logger: metrics
    are only accepted for runs that exist there, succeeded and have a video."""

    def __init__(self, metrics_dir: str | Path, results: BenchmarkLogger) -> None:
        self.metrics_dir = resolve_path(metrics_dir)
        self.results = results

    def path_for(self, benchmark_id: str, run_id: str, metrics_version: str = METRICS_VERSION) -> Path:
        for name, value in (("benchmark_id", benchmark_id), ("run_id", run_id), ("metrics_version", metrics_version)):
            check_file_name_safe(name, value)
        return self.metrics_dir / benchmark_id / run_id / f"metrics_v{metrics_version}.json"

    def exists(self, benchmark_id: str, run_id: str, metrics_version: str = METRICS_VERSION) -> bool:
        return self.path_for(benchmark_id, run_id, metrics_version).is_file()

    def measurable_run(self, benchmark_id: str, run_id: str) -> BenchmarkResult:
        """The run record, or ValueError if the run cannot be measured."""
        run = self.results.load(benchmark_id, run_id)  # FileNotFoundError if there is no such run
        if run.status != "success":
            raise ValueError(f"run {run_id} did not succeed; there is nothing to measure")
        if not run.output_video:
            raise ValueError(f"run {run_id} has no output_video recorded")
        return run

    def measure_run(
        self, benchmark_id: str, run_id: str, root: str | Path | None = None,
        *, analysis_width: int = DEFAULT_ANALYSIS_WIDTH, regions: AnalysisRegions | None = None,
    ) -> VideoMetricsRecord:
        """Measure a run's video (paths in the run record are relative to the
        project root) and save the result. Refuses if this run already has
        metrics of the current version. `regions`: the benchmark's analysis
        regions, if it declares any."""
        run = self.measurable_run(benchmark_id, run_id)
        if self.exists(benchmark_id, run_id):
            raise FileExistsError(f"run {benchmark_id}/{run_id} already has metrics v{METRICS_VERSION}")
        video = resolve_path(run.output_video, root)
        if not video.is_file():
            raise FileNotFoundError(f"video of run {run_id} is not on this machine: {run.output_video}")
        # The start frame exactly as the model received it, when the run kept one.
        derived = (run.start_frame or {}).get("derived_file")
        reference = resolve_path(derived, root) if derived else None
        if reference is not None and not reference.is_file():
            reference = None
        record = measure_video(
            video, benchmark_id=benchmark_id, run_id=run_id, reference_image=reference,
            analysis_width=analysis_width, regions=regions,
        )
        self.save(record)
        return record

    def save(self, record: VideoMetricsRecord) -> Path:
        if not isinstance(record, VideoMetricsRecord):
            raise TypeError("VideoMetricsStore stores VideoMetricsRecord only")
        if record.variant is not None:
            raise ValueError(
                f"these are metrics of the variant {record.variant!r}, not of the run's own video; "
                "use save_variant()"
            )
        self.measurable_run(record.benchmark_id, record.run_id)
        path = self.path_for(record.benchmark_id, record.run_id, record.metrics_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
        with path.open("x", encoding="utf-8") as f:  # never overwrite
            f.write(payload + "\n")
        return path

    def save_variant(self, record: VideoMetricsRecord, directory: str | Path) -> Path:
        """Write the metrics of a derived variant into the variant's own
        directory (next to its video and manifest). Never overwrites."""
        if not isinstance(record, VideoMetricsRecord) or record.variant is None:
            raise ValueError("save_variant() needs a VideoMetricsRecord with a variant name")
        self.measurable_run(record.benchmark_id, record.run_id)
        path = Path(directory) / f"metrics_v{record.metrics_version}.json"
        payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
        with path.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
        return path

    def load(self, benchmark_id: str, run_id: str, metrics_version: str = METRICS_VERSION) -> VideoMetricsRecord:
        path = self.path_for(benchmark_id, run_id, metrics_version)
        return VideoMetricsRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def video_matches(self, record: VideoMetricsRecord, video: str | Path) -> bool:
        """True if `video` is byte-identical to the file that was measured."""
        return sha256_file(video) == record.video_sha256
