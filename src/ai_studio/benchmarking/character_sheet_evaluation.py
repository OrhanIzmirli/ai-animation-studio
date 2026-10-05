"""Manual evaluation of a CHARACTER SHEET (an IMAGE GENERATION run).

Mirrors ComicAnimationEvaluation, for stills: every criterion is an integer
1-10 and HIGHER IS ALWAYS BETTER. No weighted score, no winner. Separate from
the video evaluations and from any automatic metrics.

Stored by CharacterSheetEvaluationStore, by default under
benchmarks/character_sheet_evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json,
with the same no-overwrite / explicit-revision rules as the other stores.
Criteria definitions: docs/BENCHMARK_CHARACTER_SHEET_HERO_01.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_studio.benchmarking.evaluation_store import RevisionedEvaluationStore, validate_evaluation
from ai_studio.generation.capabilities import Capability

CHARACTER_SHEET_EVALUATION_SCHEMA_VERSION = 1

# criterion -> what 10 and 1 mean. All criteria: higher = better.
CHARACTER_SHEET_CRITERIA: dict[str, str] = {
    "identity_consistency": (
        "10 = clearly the same person in every view (face, hair, build, proportions); "
        "1 = the views show different people"
    ),
    "style_match": (
        "10 = drawn in the look of the canonical frame (ink contours, cel shading, halftone, hatching); "
        "1 = a different style, e.g. photorealistic or 3D render"
    ),
    "outfit_fidelity": (
        "10 = every fixed outfit detail is present and identical in every view, nothing added; "
        "1 = the outfit is wrong, changes between views, or carries a logo, emblem or mask"
    ),
    "line_quality": (
        "10 = clean, confident, closed contours with deliberate line weight; "
        "1 = broken, muddy or smeared lines"
    ),
    "anatomy": (
        "10 = correct proportions, hands, feet and a plausible turn of the body in every view; "
        "1 = deformed or impossible anatomy"
    ),
}


@dataclass(frozen=True, kw_only=True)
class CharacterSheetEvaluation:
    benchmark_id: str
    run_id: str
    evaluator: str

    identity_consistency: int
    style_match: int
    outfit_fidelity: int
    line_quality: int
    anatomy: int

    notes: str = ""
    revision: int = 1  # assigned by the store
    timestamp: str = ""
    schema_version: int = CHARACTER_SHEET_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_evaluation(self, CHARACTER_SHEET_CRITERIA)

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in CHARACTER_SHEET_CRITERIA}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterSheetEvaluation:
        return cls(**data)


class CharacterSheetEvaluationStore(RevisionedEvaluationStore):
    evaluation_cls = CharacterSheetEvaluation
    required_task_type = Capability.IMAGE_GENERATION.value
