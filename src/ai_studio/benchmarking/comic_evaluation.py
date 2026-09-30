"""EXPERIMENTAL comic / anime animation evaluation for VIDEO GENERATION runs.

Separate from the Phase 1 object-editing HumanEvaluation (unchanged) and from
any future automatic Quality Gate metrics. Every criterion is an integer 1-10
and HIGHER IS ALWAYS BETTER. No weighted score, no winner.

Stored by ComicAnimationEvaluationStore, by default under
benchmarks/comic_evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json,
with the same no-overwrite / explicit-revision rules as HumanEvaluationStore.
Criteria definitions: docs/BENCHMARK_COMIC_ROOFTOP_01.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_studio.benchmarking.evaluation_store import RevisionedEvaluationStore, validate_evaluation
from ai_studio.generation.capabilities import Capability

COMIC_EVALUATION_SCHEMA_VERSION = 1

# criterion -> what 10 and 1 mean. All criteria: higher = better.
COMIC_CRITERIA: dict[str, str] = {
    "character_identity": "10 = unmistakably the reference character throughout; 1 = a different or generic character",
    "face_stability": "10 = facial structure and proportions constant, including through head turns; 1 = face morphs or melts",
    "line_stability": "10 = ink contours hold shape and weight frame to frame; 1 = lines boil, crawl or vanish",
    "style_consistency": "10 = target style uniform for the whole shot; 1 = style drifts or changes mid-shot",
    "background_stability": "10 = background geometry and layering stable, plausible parallax; 1 = architecture warps or pops",
    "motion_quality": "10 = motion deliberate and natural, stepped timing clearly intentional; 1 = jitter, sliding or missing motion",
    "camera_motion": "10 = camera move smooth with consistent perspective; 1 = drifting, shaking or broken perspective",
    "temporal_consistency": (
        "10 = no unintended flicker, lighting and colour constant; 1 = heavy flicker or pumping. "
        "Deliberate held drawings (stepped animation) are not penalised"
    ),
    "artifact_visibility": (
        "10 = no visible distracting artifacts; 1 = severe obvious artifacts. "
        "Note the direction: a HIGH score means FEW artifacts"
    ),
    "overall_visual_quality": "10 = could be a shot from a finished production; 1 = unusable",
}


@dataclass(frozen=True, kw_only=True)
class ComicAnimationEvaluation:
    benchmark_id: str
    run_id: str
    evaluator: str

    character_identity: int
    face_stability: int
    line_stability: int
    style_consistency: int
    background_stability: int
    motion_quality: int
    camera_motion: int
    temporal_consistency: int
    artifact_visibility: int
    overall_visual_quality: int

    notes: str = ""
    revision: int = 1  # assigned by the store
    timestamp: str = ""
    schema_version: int = COMIC_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_evaluation(self, COMIC_CRITERIA)

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in COMIC_CRITERIA}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComicAnimationEvaluation:
        return cls(**data)


class ComicAnimationEvaluationStore(RevisionedEvaluationStore):
    evaluation_cls = ComicAnimationEvaluation
    required_task_type = Capability.VIDEO_GENERATION.value
