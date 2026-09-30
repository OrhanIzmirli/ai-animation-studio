import pytest

from ai_studio.benchmarking import (
    COMIC_CRITERIA,
    CRITERIA,
    BenchmarkRunError,
    ComicAnimationEvaluation,
    ComicAnimationEvaluationStore,
    HumanEvaluation,
    HumanEvaluationStore,
)
from ai_studio.benchmarking.evaluation_store import SCORE_MAX, SCORE_MIN
from conftest import FakeBackend, FakeGenerationBackend

BENCH = "comic_rooftop_01"

EXPECTED = {
    "character_identity", "face_stability", "line_stability", "style_consistency",
    "background_stability", "motion_quality", "camera_motion", "temporal_consistency",
    "artifact_visibility", "overall_visual_quality",
}


def make_eval(**overrides):
    fields = dict(benchmark_id=BENCH, run_id="run-1", evaluator="reviewer-a")
    fields.update({name: 6 for name in COMIC_CRITERIA})
    fields.update(overrides)
    return ComicAnimationEvaluation(**fields)


@pytest.fixture
def store(tmp_path):
    return ComicAnimationEvaluationStore(tmp_path / "comic_evaluations")


# --- schema ---

def test_criteria_are_exactly_the_comic_set():
    assert set(COMIC_CRITERIA) == EXPECTED


def test_generic_human_evaluation_schema_unchanged():
    assert set(CRITERIA) == {
        "reference_fidelity", "temporal_consistency", "background_preservation", "geometry",
        "lighting_and_shadows", "occlusion", "artifact_visibility", "overall_realism",
    }
    assert "character_identity" not in HumanEvaluation.__dataclass_fields__


def test_every_criterion_higher_is_better():
    for name, description in COMIC_CRITERIA.items():
        assert description.startswith("10 = "), name
        assert "; 1 = " in description, name
    assert "10 = no visible distracting artifacts" in COMIC_CRITERIA["artifact_visibility"]
    assert "1 = severe obvious artifacts" in COMIC_CRITERIA["artifact_visibility"]


def test_stepped_animation_not_penalised_is_documented():
    assert "stepped" in COMIC_CRITERIA["temporal_consistency"]


@pytest.mark.parametrize("criterion", sorted(EXPECTED))
@pytest.mark.parametrize("bad", [0, 11, 5.5, "5", True, None])
def test_invalid_scores_rejected(criterion, bad):
    with pytest.raises(ValueError):
        make_eval(**{criterion: bad})


@pytest.mark.parametrize("score", [SCORE_MIN, SCORE_MAX])
def test_bounds_accepted(score):
    assert make_eval(**{name: score for name in EXPECTED}).scores() == {name: score for name in EXPECTED}


def test_no_aggregate_score():
    evaluation = make_eval()
    for name in ("total", "overall", "weighted_score", "score", "winner"):
        assert not hasattr(evaluation, name)


# --- persistence and revisions ---

def test_persists_and_loads(store, tmp_path):
    path = store.save(make_eval(notes="line boil on the jacket at 3s"))
    assert path == tmp_path / "comic_evaluations" / BENCH / "run-1" / "reviewer-a.r001.json"
    loaded = store.load(BENCH, "run-1", "reviewer-a")
    assert loaded.notes == "line boil on the jacket at 3s"
    assert loaded.revision == 1 and loaded.schema_version == 1


def test_no_accidental_overwrite(store):
    path = store.save(make_eval(line_stability=3))
    before = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="save_revision"):
        store.save(make_eval(line_stability=9))
    assert path.read_text(encoding="utf-8") == before


def test_explicit_revisions_keep_history(store):
    store.save(make_eval(face_stability=4))
    second = store.save_revision(make_eval(face_stability=5))
    store.save_revision(make_eval(face_stability=7))
    assert second.name == "reviewer-a.r002.json"
    assert store.load(BENCH, "run-1", "reviewer-a").face_stability == 7
    assert store.load(BENCH, "run-1", "reviewer-a", revision=1).face_stability == 4
    assert [e.revision for e in store.list_evaluations(BENCH, "run-1")] == [1, 2, 3]


def test_revision_requires_existing(store):
    with pytest.raises(FileNotFoundError):
        store.save_revision(make_eval())


def test_store_rejects_other_evaluation_types(store):
    human = HumanEvaluation(
        benchmark_id=BENCH, run_id="run-1", evaluator="x", **{name: 5 for name in CRITERIA}
    )
    with pytest.raises(TypeError):
        store.save(human)


# --- linking to runs ---

def test_linked_store_accepts_successful_generation_run(tmp_path, runner, generation_request_factory):
    runner.run(generation_request_factory(), FakeGenerationBackend(), benchmark_id=BENCH, run_id="run-1")
    store = ComicAnimationEvaluationStore(tmp_path / "comic_evaluations", results=runner.logger)
    store.save(make_eval())
    with pytest.raises(FileNotFoundError):
        store.save(make_eval(run_id="missing"))


def test_linked_store_rejects_failed_run(tmp_path, runner, generation_request_factory):
    backend = FakeGenerationBackend(fail_in="generate", exc=ValueError("x"))
    with pytest.raises(BenchmarkRunError):
        runner.run(generation_request_factory(), backend, benchmark_id=BENCH, run_id="run-1")
    store = ComicAnimationEvaluationStore(tmp_path / "comic_evaluations", results=runner.logger)
    with pytest.raises(ValueError, match="did not succeed"):
        store.save(make_eval())


def test_linked_store_rejects_editing_run(tmp_path, runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-1")
    store = ComicAnimationEvaluationStore(tmp_path / "comic_evaluations", results=runner.logger)
    with pytest.raises(ValueError, match="video_generation"):
        store.save(make_eval(benchmark_id="object_replacement_01"))


def test_comic_and_human_evaluations_are_separate(tmp_path, runner, generation_request_factory):
    record = runner.run(generation_request_factory(), FakeGenerationBackend(), benchmark_id=BENCH, run_id="run-1")
    comic = ComicAnimationEvaluationStore(tmp_path / "comic_evaluations")
    human = HumanEvaluationStore(tmp_path / "evaluations")
    comic.save(make_eval())

    assert human.list_evaluations(BENCH, "run-1") == []
    assert runner.logger.load(BENCH, "run-1") == record  # run record untouched
    assert record.quality_metrics == {}  # reserved for automatic metrics
