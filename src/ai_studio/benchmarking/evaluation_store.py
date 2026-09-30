"""Shared pieces for manual (human) evaluations of benchmark runs.

Every evaluation type is a frozen dataclass with benchmark_id, run_id,
evaluator, integer 1-10 scores (higher is always better), notes, revision,
timestamp, and to_dict()/from_dict(). Stores are append-only:

    <evaluations_dir>/<benchmark_id>/<run_id>/<evaluator>.r001.json

Files are never overwritten. Re-rating a run is an explicit new revision
(r002, r003, ...); earlier revisions are kept. Manual evaluations are kept
apart from run records and from any future automatic Quality Gate data.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.paths import resolve_path

SCORE_MIN = 1
SCORE_MAX = 10


def validate_evaluation(evaluation: Any, criteria: dict[str, str]) -> None:
    """Common __post_init__ checks; also fills an empty timestamp."""
    check_file_name_safe("benchmark_id", evaluation.benchmark_id)
    check_file_name_safe("run_id", evaluation.run_id)
    check_file_name_safe("evaluator", evaluation.evaluator)
    if evaluation.evaluator != evaluation.evaluator.strip():
        raise ValueError("evaluator must not have leading/trailing whitespace")
    for name in criteria:
        score = getattr(evaluation, name)
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"{name} must be an int")
        if not SCORE_MIN <= score <= SCORE_MAX:
            raise ValueError(f"{name} must be between {SCORE_MIN} and {SCORE_MAX}, got {score}")
    if not isinstance(evaluation.notes, str):
        raise ValueError("notes must be a string")
    revision = evaluation.revision
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ValueError("revision must be a positive int")
    if not evaluation.timestamp:
        object.__setattr__(evaluation, "timestamp", datetime.now(timezone.utc).isoformat())


class RevisionedEvaluationStore:
    """Append-only store for one evaluation type (set `evaluation_cls`)."""

    evaluation_cls: ClassVar[type]
    # If set, and a results logger is given, only runs of this task type are accepted.
    required_task_type: ClassVar[str | None] = None

    def __init__(self, evaluations_dir: str | Path, results: BenchmarkLogger | None = None) -> None:
        """If `results` is given, evaluations are only accepted for runs that
        exist there and succeeded."""
        self.evaluations_dir = resolve_path(evaluations_dir)
        self.results = results

    def save(self, evaluation: Any) -> Path:
        """First rating of a run by this evaluator. Refuses if one already exists."""
        if self._revisions(evaluation):
            raise FileExistsError(
                f"{evaluation.evaluator} already rated {evaluation.benchmark_id}/{evaluation.run_id}; "
                "use save_revision() to re-rate explicitly"
            )
        return self._write(dataclasses.replace(evaluation, revision=1))

    def save_revision(self, evaluation: Any) -> Path:
        """Explicit re-rating: stored as the next revision; earlier ones are kept."""
        revisions = self._revisions(evaluation)
        if not revisions:
            raise FileNotFoundError("no existing evaluation to revise; use save()")
        return self._write(dataclasses.replace(evaluation, revision=max(revisions) + 1))

    def load(self, benchmark_id: str, run_id: str, evaluator: str, revision: int | None = None) -> Any:
        """Latest revision unless `revision` is given."""
        run_dir = self._run_dir(benchmark_id, run_id)
        revisions = self._revisions_in(run_dir, evaluator)
        if not revisions:
            raise FileNotFoundError(f"no evaluation by {evaluator} for {benchmark_id}/{run_id}")
        path = run_dir / self._file_name(evaluator, revision or max(revisions))
        return self.evaluation_cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_evaluations(self, benchmark_id: str, run_id: str) -> list[Any]:
        """All evaluators, all revisions."""
        run_dir = self._run_dir(benchmark_id, run_id)
        if not run_dir.is_dir():
            return []
        return [
            self.evaluation_cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(run_dir.glob("*.r*.json"))
        ]

    # --- helpers ---

    def _write(self, evaluation: Any) -> Path:
        if not isinstance(evaluation, self.evaluation_cls):
            raise TypeError(f"{type(self).__name__} stores {self.evaluation_cls.__name__} only")
        if self.results is not None:
            record = self.results.load(evaluation.benchmark_id, evaluation.run_id)  # raises if missing
            if record.status != "success":
                raise ValueError(f"run {evaluation.run_id} did not succeed; nothing to rate")
            if self.required_task_type and record.task_type != self.required_task_type:
                raise ValueError(
                    f"{self.evaluation_cls.__name__} applies to {self.required_task_type} runs; "
                    f"run {evaluation.run_id} is {record.task_type!r}"
                )
        path = self._run_dir(evaluation.benchmark_id, evaluation.run_id) / self._file_name(
            evaluation.evaluator, evaluation.revision
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False)
        with path.open("x", encoding="utf-8") as f:  # never overwrite
            f.write(payload + "\n")
        return path

    def _run_dir(self, benchmark_id: str, run_id: str) -> Path:
        check_file_name_safe("benchmark_id", benchmark_id)
        check_file_name_safe("run_id", run_id)
        return self.evaluations_dir / benchmark_id / run_id

    def _revisions(self, evaluation: Any) -> list[int]:
        return self._revisions_in(self._run_dir(evaluation.benchmark_id, evaluation.run_id), evaluation.evaluator)

    @staticmethod
    def _revisions_in(run_dir: Path, evaluator: str) -> list[int]:
        pattern = re.compile(rf"^{re.escape(evaluator)}\.r(\d+)\.json$")
        if not run_dir.is_dir():
            return []
        return [int(m.group(1)) for p in run_dir.iterdir() if (m := pattern.match(p.name))]

    @staticmethod
    def _file_name(evaluator: str, revision: int) -> str:
        return f"{evaluator}.r{revision:03d}.json"
