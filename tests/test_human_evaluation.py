import pytest

from ai_studio.benchmarking import BenchmarkRunError, HumanEvaluationStore
from ai_studio.benchmarking.human_evaluation import CRITERIA, SCORE_MAX, SCORE_MIN, HumanEvaluation
from conftest import FailingBackend, FakeBackend

EXPECTED_CRITERIA = {
    "reference_fidelity",
    "temporal_consistency",
    "background_preservation",
    "geometry",
    "lighting_and_shadows",
    "occlusion",
    "artifact_visibility",
    "overall_realism",
}


def make_eval(**overrides):
    fields = dict(benchmark_id="object_replacement_01", run_id="run-1", evaluator="reviewer-a")
    fields.update({name: 7 for name in CRITERIA})
    fields.update(overrides)
    return HumanEvaluation(**fields)


def test_criteria_are_exactly_the_phase1_set():
    assert set(CRITERIA) == EXPECTED_CRITERIA


def test_valid_evaluation_with_notes():
    evaluation = make_eval(notes="teapot handle flickers around frame 40")
    assert evaluation.scores() == {name: 7 for name in CRITERIA}
    assert evaluation.notes.startswith("teapot")
    assert evaluation.timestamp
    assert HumanEvaluation.from_dict(evaluation.to_dict()) == evaluation


@pytest.mark.parametrize("score", [SCORE_MIN, SCORE_MAX])
def test_bounds_accepted(score):
    make_eval(**{name: score for name in CRITERIA})


@pytest.mark.parametrize("criterion", sorted(EXPECTED_CRITERIA))
@pytest.mark.parametrize("bad", [0, 11, -1, 7.5, "7", True, None])
def test_invalid_scores_rejected(criterion, bad):
    with pytest.raises(ValueError):
        make_eval(**{criterion: bad})


def test_missing_criterion_rejected():
    fields = make_eval().to_dict()
    del fields["occlusion"]
    with pytest.raises(TypeError):
        HumanEvaluation(**fields)


def test_empty_evaluator_rejected():
    with pytest.raises(ValueError):
        make_eval(evaluator=" ")


def test_artifact_visibility_higher_is_better():
    description = CRITERIA["artifact_visibility"]
    assert "10 = no visible distracting artifacts" in description
    assert "1 = severe obvious artifacts" in description


def test_every_criterion_documents_10_as_best():
    for name, description in CRITERIA.items():
        assert description.startswith("10 = "), name
        assert "; 1 = " in description, name


def test_no_aggregate_score_is_provided():
    evaluation = make_eval()
    for name in ("total", "overall", "weighted_score", "score", "winner"):
        assert not hasattr(evaluation, name)


# --- storage ---

@pytest.fixture
def store(tmp_path):
    return HumanEvaluationStore(tmp_path / "evaluations")


def test_evaluation_persists_and_links_to_run(store, tmp_path):
    path = store.save(make_eval(notes="looks good"))

    assert path == tmp_path / "evaluations" / "object_replacement_01" / "run-1" / "reviewer-a.r001.json"
    loaded = store.load("object_replacement_01", "run-1", "reviewer-a")
    assert loaded.notes == "looks good"
    assert loaded.revision == 1
    assert loaded.benchmark_id == "object_replacement_01" and loaded.run_id == "run-1"


def test_second_save_refuses_to_overwrite(store):
    path = store.save(make_eval(geometry=3))
    before = path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="save_revision"):
        store.save(make_eval(geometry=9))

    assert path.read_text(encoding="utf-8") == before


def test_explicit_revision_keeps_history(store):
    store.save(make_eval(geometry=3, notes="first look"))
    path2 = store.save_revision(make_eval(geometry=6, notes="re-rated on a better monitor"))
    store.save_revision(make_eval(geometry=8))

    assert path2.name == "reviewer-a.r002.json"
    assert store.load("object_replacement_01", "run-1", "reviewer-a").geometry == 8  # latest
    assert store.load("object_replacement_01", "run-1", "reviewer-a", revision=1).geometry == 3
    assert [e.revision for e in store.list_evaluations("object_replacement_01", "run-1")] == [1, 2, 3]


def test_revision_requires_existing_evaluation(store):
    with pytest.raises(FileNotFoundError):
        store.save_revision(make_eval())


def test_multiple_evaluators_are_independent(store):
    store.save(make_eval(evaluator="reviewer-a"))
    store.save(make_eval(evaluator="reviewer-b"))
    evaluators = sorted(e.evaluator for e in store.list_evaluations("object_replacement_01", "run-1"))
    assert evaluators == ["reviewer-a", "reviewer-b"]


def test_similar_evaluator_names_do_not_collide(store):
    store.save(make_eval(evaluator="ann"))
    store.save(make_eval(evaluator="ann.r001"))  # odd but legal name
    assert store.load("object_replacement_01", "run-1", "ann").evaluator == "ann"


def test_missing_evaluation(store):
    assert store.list_evaluations("object_replacement_01", "nope") == []
    with pytest.raises(FileNotFoundError):
        store.load("object_replacement_01", "nope", "reviewer-a")


def test_unsafe_evaluator_name_rejected():
    with pytest.raises(ValueError):
        make_eval(evaluator="../escape")


def test_store_with_results_requires_existing_successful_run(tmp_path, runner, request_factory):
    record = runner.run(request_factory(), FakeBackend(), run_id="run-1")
    store = HumanEvaluationStore(tmp_path / "evaluations", results=runner.logger)

    store.save(make_eval(run_id=record.run_id))
    with pytest.raises(FileNotFoundError):
        store.save(make_eval(run_id="run-does-not-exist"))


def test_store_refuses_rating_failed_run(tmp_path, runner, request_factory):
    with pytest.raises(BenchmarkRunError):
        runner.run(request_factory(), FailingBackend("generate", ValueError("x")), run_id="run-f")
    store = HumanEvaluationStore(tmp_path / "evaluations", results=runner.logger)
    with pytest.raises(ValueError, match="did not succeed"):
        store.save(make_eval(run_id="run-f"))


def test_evaluations_are_separate_from_result_records(tmp_path, runner, request_factory):
    runner.run(request_factory(), FakeBackend(), run_id="run-1")
    store = HumanEvaluationStore(tmp_path / "evaluations", results=runner.logger)
    store.save(make_eval())

    # The result record is untouched and list_runs sees only run records.
    runs = runner.logger.list_runs("object_replacement_01")
    assert [r.run_id for r in runs] == ["run-1"]
    assert runs[0].human_rating is None and runs[0].quality_metrics == {}
