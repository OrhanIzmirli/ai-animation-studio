"""Write benchmark results to JSON, one file per run.

Layout: <results_dir>/<benchmark_id>/<run_id>.json

Existing result files are never overwritten: writing a run_id that already
exists raises FileExistsError.

list_runs leaves out a file it cannot load (truncated, or written by a newer
schema) and reports it with an UnreadableRecordWarning; load raises.
"""

from __future__ import annotations

import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from ai_studio.benchmarking.schema import BenchmarkResult
from ai_studio.paths import resolve_path


class UnreadableRecordWarning(UserWarning):
    """A record file that list_runs could not load and left out."""


def new_run_id() -> str:
    """Sortable, collision-resistant run id, e.g. 20260930T101500Z-3f2a9c1d."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class BenchmarkLogger:
    def __init__(self, results_dir: str | Path) -> None:
        # Relative directories are relative to the project root, not the cwd.
        self.results_dir = resolve_path(results_dir)

    def path_for(self, benchmark_id: str, run_id: str) -> Path:
        return self.results_dir / benchmark_id / f"{run_id}.json"

    def log(self, result: BenchmarkResult) -> Path:
        path = self.path_for(result.benchmark_id, result.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        # "x" mode fails if the file already exists, so results are never overwritten.
        with path.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
        return path

    def load(self, benchmark_id: str, run_id: str) -> BenchmarkResult:
        path = self.path_for(benchmark_id, run_id)
        return BenchmarkResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self, benchmark_id: str) -> list[BenchmarkResult]:
        directory = self.results_dir / benchmark_id
        if not directory.is_dir():
            return []
        runs = []
        for path in sorted(directory.glob("*.json")):
            try:
                runs.append(BenchmarkResult.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # One truncated or newer-schema file must not hide the other runs.
                warnings.warn(
                    f"skipped unreadable record {path}: {type(exc).__name__}: {exc}",
                    UnreadableRecordWarning, stacklevel=2,
                )
        return runs
