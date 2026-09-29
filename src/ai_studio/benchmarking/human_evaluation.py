"""Phase 1 human evaluation: schema and storage.

Every criterion is an integer from 1 to 10 and HIGHER IS ALWAYS BETTER.
No weighted/overall score is computed and no winner is derived here.

Human evaluations are stored separately from benchmark result records (and
from any future automatic Quality Gate data):

    benchmarks/evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json

Files are never overwritten. Re-rating a run is an explicit new revision
(r002, r003, ...); earlier revisions are kept.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.paths import resolve_path

HUMAN_EVALUATION_SCHEMA_VERSION = 1
SCORE_MIN = 1
SCORE_MAX = 10

# criterion -> what 10 and 1 mean. All criteria: higher = better.
CRITERIA: dict[str, str] = {
    "reference_fidelity": "10 = edited object matches the reference exactly; 1 = unrelated to the reference",
    "temporal_consistency": "10 = stable across frames, no flicker or identity drift; 1 = severe flicker/drift",
    "background_preservation": "10 = everything outside the edit is unchanged; 1 = background heavily altered",
    "geometry": "10 = correct shape, scale and perspective; 1 = badly distorted",
    "lighting_and_shadows": "10 = lighting and shadows match the scene; 1 = clearly inconsistent",
    "occlusion": "10 = correct in front/behind ordering with other objects; 1 = broken occlusion",
    "artifact_visibility": (
        "10 = no visible distracting artifacts; 1 = severe obvious artifacts. "
        "Note the direction: a HIGH score means FEW artifacts"
    ),
    "overall_realism": "10 = indistinguishable from real footage; 1 = obviously fake",
}


@dataclass(frozen=True, kw_only=True)
class HumanEvaluation:
    benchmark_id: str
    run_id: str
    evaluator: str

    reference_fidelity: int
    temporal_consistency: int
    background_preservation: int
    geometry: int
    lighting_and_shadows: int
    occlusion: int
    artifact_visibility: int
    overall_realism: int

    notes: str = ""
    revision: int = 1  # assigned by HumanEvaluationStore
    timestamp: str = ""
    schema_version: int = HUMAN_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        check_file_name_safe("benchmark_id", self.benchmark_id)
        check_file_name_safe("run_id", self.run_id)
        check_file_name_safe("evaluator", self.evaluator)
        if self.evaluator != self.evaluator.strip():
            raise ValueError("evaluator must not have leading/trailing whitespace")
        for name in CRITERIA:
            score = getattr(self, name)
            if not isinstance(score, int) or isinstance(score, bool):
                raise ValueError(f"{name} must be an int")
            if not SCORE_MIN <= score <= SCORE_MAX:
                raise ValueError(f"{name} must be between {SCORE_MIN} and {SCORE_MAX}, got {score}")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive int")
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc).isoformat())

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in CRITERIA}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanEvaluation:
        return cls(**data)


class HumanEvaluationStore:
    def __init__(self, evaluations_dir: str | Path, results: BenchmarkLogger | None = None) -> None:
        """If `results` is given, evaluations are only accepted for runs that
        exist there and succeeded."""
        self.evaluations_dir = resolve_path(evaluations_dir)
        self.results = results

    def save(self, evaluation: HumanEvaluation) -> Path:
        """First rating of a run by this evaluator. Refuses if one already exists."""
        if self._revisions(evaluation):
            raise FileExistsError(
                f"{evaluation.evaluator} already rated {evaluation.benchmark_id}/{evaluation.run_id}; "
                "use save_revision() to re-rate explicitly"
            )
        return self._write(dataclasses.replace(evaluation, revision=1))

    def save_revision(self, evaluation: HumanEvaluation) -> Path:
        """Explicit re-rating: stored as the next revision; earlier ones are kept."""
        revisions = self._revisions(evaluation)
        if not revisions:
            raise FileNotFoundError("no existing evaluation to revise; use save()")
        return self._write(dataclasses.replace(evaluation, revision=max(revisions) + 1))

    def load(self, benchmark_id: str, run_id: str, evaluator: str, revision: int | None = None) -> HumanEvaluation:
        """Latest revision unless `revision` is given."""
        run_dir = self._run_dir(benchmark_id, run_id)
        revisions = self._revisions_in(run_dir, evaluator)
        if not revisions:
            raise FileNotFoundError(f"no evaluation by {evaluator} for {benchmark_id}/{run_id}")
        path = run_dir / self._file_name(evaluator, revision or max(revisions))
        return HumanEvaluation.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_evaluations(self, benchmark_id: str, run_id: str) -> list[HumanEvaluation]:
        """All evaluators, all revisions."""
        run_dir = self._run_dir(benchmark_id, run_id)
        if not run_dir.is_dir():
            return []
        return [
            HumanEvaluation.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(run_dir.glob("*.r*.json"))
        ]

    # --- helpers ---

    def _write(self, evaluation: HumanEvaluation) -> Path:
        if self.results is not None:
            record = self.results.load(evaluation.benchmark_id, evaluation.run_id)  # raises if missing
            if record.status != "success":
                raise ValueError(f"run {evaluation.run_id} did not succeed; nothing to rate")
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

    def _revisions(self, evaluation: HumanEvaluation) -> list[int]:
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
