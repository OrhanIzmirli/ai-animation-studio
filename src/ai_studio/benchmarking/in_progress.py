"""In-progress markers: find runs that started generating and never got a record.

The runner writes <output dir>/RUN_IN_PROGRESS.json right before generation
and removes it as soon as the run's record (success or failed) is saved. A
run that dies in between (process killed, crash, power loss, Ctrl-C) leaves
an output folder with a marker and no record: an ABANDONED run.

The marker holds everything needed to write that run's record afterwards:

    {"marker_version": 1, "benchmark_id": ..., "run_id": ...,
     "started_at": <UTC ISO time>, "pid": ..., "hostname": ...,
     "record": {<a complete status="failed" record, failure_stage "abandoned">}}

A marker is STALE when it is older than a threshold (default
DEFAULT_STALE_HOURS). Age is the only test, so that a run that is still
generating is not mistaken for an abandoned one; choose a threshold longer
than the longest run.

record_abandoned() writes the marker's record through the logger. Nothing is
deleted: the marker stays in the folder as evidence, and because the run now
has a record it is no longer reported.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.benchmarking.schema import BenchmarkResult

MARKER_NAME = "RUN_IN_PROGRESS.json"
MARKER_VERSION = 1
ABANDONED_STAGE = "abandoned"
ABANDONED_ERROR_TYPE = "AbandonedRun"
DEFAULT_STALE_HOURS = 6.0


def marker_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / MARKER_NAME


def write_marker(output_dir: str | Path, record: BenchmarkResult) -> Path:
    """Write the marker for a run that is about to generate. `record` is the
    record to save if the run is abandoned (status "failed", stage "abandoned")."""
    if record.status != "failed" or record.failure_stage != ABANDONED_STAGE:
        raise ValueError("the marker's record must be a failed record with failure_stage 'abandoned'")
    marker = {
        "marker_version": MARKER_VERSION,
        "benchmark_id": record.benchmark_id,
        "run_id": record.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "record": record.to_dict(),
    }
    path = marker_path(output_dir)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(marker, indent=2, ensure_ascii=False) + "\n")
    return path


def remove_marker(output_dir: str | Path) -> None:
    """Remove a run's marker once its record is saved (no marker: nothing to do).

    This is the one file the project itself removes under outputs/: it is the
    runner's own bookkeeping, never run data, and only this exact file name
    is touched."""
    try:
        marker_path(output_dir).unlink(missing_ok=True)
    except OSError:
        pass  # the record exists, so the run is not abandoned either way


@dataclass(frozen=True)
class AbandonedRun:
    benchmark_id: str
    run_id: str
    output_dir: Path
    started_at: datetime  # from the marker; the marker file's time if it is unreadable
    age_hours: float
    marker: dict[str, Any] | None  # None: the marker file is not valid
    problem: str | None = None  # why the marker cannot be turned into a record


def _read_marker(path: Path, benchmark_id: str, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"marker is not readable JSON ({type(exc).__name__})"
    if not isinstance(marker, dict) or not isinstance(marker.get("record"), dict):
        return None, "marker has no record"
    if marker.get("marker_version") != MARKER_VERSION:
        return None, f"unknown marker_version {marker.get('marker_version')!r}"
    record = marker["record"]
    if (marker.get("benchmark_id"), marker.get("run_id")) != (benchmark_id, run_id) or (
        record.get("benchmark_id"), record.get("run_id")
    ) != (benchmark_id, run_id):
        return None, "marker belongs to another run (folder moved or renamed?)"
    return marker, None


def _started_at(marker: dict[str, Any] | None, path: Path) -> datetime:
    try:
        started = datetime.fromisoformat(marker["started_at"])  # type: ignore[index]
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return started
    except (TypeError, KeyError, ValueError):
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def _run_dirs(outputs_dir: Path, benchmark: str | None) -> list[tuple[str, Path]]:
    """(benchmark_id, run folder) for every outputs/<benchmark>/<run_id>/."""
    if not outputs_dir.is_dir():
        return []
    benchmarks = [outputs_dir / benchmark] if benchmark else sorted(p for p in outputs_dir.iterdir())
    return [
        (folder.name, run_dir)
        for folder in benchmarks if folder.is_dir()
        for run_dir in sorted(folder.iterdir()) if run_dir.is_dir()
    ]


def find_abandoned(
    outputs_dir: str | Path,
    logger: BenchmarkLogger,
    *,
    stale_after_hours: float = DEFAULT_STALE_HOURS,
    benchmark: str | None = None,
    now: datetime | None = None,
) -> list[AbandonedRun]:
    """Output folders with a stale marker and no record, oldest first."""
    if stale_after_hours < 0:
        raise ValueError("stale_after_hours must be >= 0")
    now = now or datetime.now(timezone.utc)
    found = []
    for benchmark_id, run_dir in _run_dirs(Path(outputs_dir), benchmark):
        path = marker_path(run_dir)
        if not path.is_file() or logger.path_for(benchmark_id, run_dir.name).exists():
            continue
        marker, problem = _read_marker(path, benchmark_id, run_dir.name)
        started = _started_at(marker, path)
        age_hours = max(0.0, (now - started).total_seconds() / 3600)  # clocks and file times differ slightly
        if age_hours >= stale_after_hours:
            found.append(AbandonedRun(benchmark_id, run_dir.name, run_dir, started, age_hours, marker, problem))
    return sorted(found, key=lambda run: (run.started_at, run.benchmark_id, run.run_id))


def find_unmarked(
    outputs_dir: str | Path, logger: BenchmarkLogger, *, benchmark: str | None = None,
) -> list[Path]:
    """Run folders with neither a marker nor a record, under benchmarks that
    have recorded runs (e.g. runs abandoned before markers existed). They
    cannot be recorded automatically: nothing says what was requested."""
    return [
        run_dir
        for benchmark_id, run_dir in _run_dirs(Path(outputs_dir), benchmark)
        if (logger.results_dir / benchmark_id).is_dir()
        and not marker_path(run_dir).is_file()
        and not logger.path_for(benchmark_id, run_dir.name).exists()
    ]


def record_abandoned(
    run: AbandonedRun, logger: BenchmarkLogger, now: datetime | None = None,
) -> BenchmarkResult:
    """Write the status="failed", failure_stage="abandoned" record of an
    abandoned run. ValueError if its marker is not usable; FileExistsError if
    the run has a record after all (records are never overwritten)."""
    if run.marker is None:
        raise ValueError(f"{run.benchmark_id}/{run.run_id}: {run.problem}")
    now = now or datetime.now(timezone.utc)
    data = dict(run.marker["record"])
    note = (
        f"abandoned: generation started {run.started_at.isoformat()} "
        f"(pid {run.marker.get('pid')} on {run.marker.get('hostname')}) and no record was saved; "
        f"recorded as failed on {now.isoformat()}"
    )
    data.update(
        status="failed",
        failure_stage=ABANDONED_STAGE,
        error_type=ABANDONED_ERROR_TYPE,
        error_message=(
            "the run started generating but never finished: no record was saved "
            "(process killed, crash or power loss)"
        ),
        output_video=None,  # whatever is in the folder was never verified
        output_media=None,
        notes="\n".join(line for line in (data.get("notes"), note) if line),
    )
    record = BenchmarkResult.from_dict(data)
    logger.log(record)  # "x" mode: never overwrites
    return record
