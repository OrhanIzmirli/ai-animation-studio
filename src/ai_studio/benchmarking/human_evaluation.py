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

from dataclasses import asdict, dataclass
from typing import Any

from ai_studio.benchmarking.evaluation_store import (
    SCORE_MAX,
    SCORE_MIN,
    RevisionedEvaluationStore,
    validate_evaluation,
)

HUMAN_EVALUATION_SCHEMA_VERSION = 1

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
        validate_evaluation(self, CRITERIA)

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in CRITERIA}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanEvaluation:
        return cls(**data)


class HumanEvaluationStore(RevisionedEvaluationStore):
    """benchmarks/evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json"""

    evaluation_cls = HumanEvaluation
