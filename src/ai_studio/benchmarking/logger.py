"""Write benchmark results to JSON, one file per run.

Layout: <results_dir>/<benchmark_id>/<run_id>.json

Existing result files are never overwritten: writing a run_id that already
exists raises FileExistsError.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ai_studio.benchmarking.schema import BenchmarkResult


def new_run_id() -> str:
    """Sortable, collision-resistant run id, e.g. 20260930T101500Z-3f2a9c1d."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class BenchmarkLogger:
    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)

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
        return [
            BenchmarkResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(directory.glob("*.json"))
        ]
