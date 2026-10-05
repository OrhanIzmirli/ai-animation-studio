"""The draft character-sheet benchmark and its manual evaluation."""

from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import (
    CHARACTER_SHEET_CRITERIA,
    COMIC_CRITERIA,
    BenchmarkRunError,
    CharacterSheetEvaluation,
    CharacterSheetEvaluationStore,
    ComicAnimationEvaluation,
    ComicAnimationEvaluationStore,
    DraftBenchmarkError,
    Resolution,
    load_definition,
    load_definitions,
)
from ai_studio.benchmarking.evaluation_store import SCORE_MAX, SCORE_MIN
from ai_studio.generation.backends import FakeImageBackend
from ai_studio.generation.image import ImageGenerationRequest
from conftest import FakeGenerationBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINITION = REPO_ROOT / "benchmarks" / "definitions" / "character_sheet_hero_01.yaml"
BENCH = "character_sheet_hero_01"
CANONICAL = "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
CANONICAL_SHA = "3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb"
SIZE = Resolution(width=96, height=64)

EXPECTED = {"identity_consistency", "style_match", "outfit_fidelity", "line_quality", "anatomy"}


# --- the definition ---

@pytest.fixture(scope="module")
def definition():
    return load_definition(DEFINITION)


def test_definition_is_a_draft_image_generation_benchmark(definition):
    assert definition.benchmark_id == BENCH
    assert definition.task_type == "image_generation"
    assert definition.status == "draft"
    assert definition.is_complete  # nothing is missing; it waits for a human decision
    assert load_definitions(DEFINITION.parent)[BENCH] == definition


def test_draft_cannot_be_turned_into_a_request_or_run(definition, runner):
    with pytest.raises(DraftBenchmarkError):
        definition.to_request(seed=1, output_resolution=SIZE)
    with pytest.raises(DraftBenchmarkError):
        runner.run_definition(definition, FakeImageBackend(), seed=1, output_resolution=SIZE, run_id="r")
    assert not runner.logger.path_for(BENCH, "r").exists()
    assert not (runner.outputs_dir / BENCH).exists()


def test_canonical_frame_is_the_only_reference(definition):
    assert definition.reference_images == [CANONICAL]
    assert CANONICAL_SHA in definition.notes
    assert definition.start_frame is None and definition.source_image is None
    assert definition.image_count == 1  # one run = one sheet


def test_all_five_views_are_required(definition):
    assert definition.composition["views"] == [
        "front", "three_quarter_left", "three_quarter_right", "profile", "back",
    ]
    prompt = definition.prompt.lower()
    for phrase in ("front view", "three-quarter view turned to his left",
                   "three-quarter view turned to his right", "side profile", "back view", "five full-body views"):
        assert phrase in prompt, phrase
    assert "missing view" in definition.negative_prompt and "extra view" in definition.negative_prompt


def test_outfit_is_fixed_from_the_canonical_frame(definition):
    details = definition.subject_details
    assert set(details) >= {"hair", "jacket", "backpack", "trousers", "shoes", "accents", "face", "forbidden"}
    assert "curly dark hair" in details["hair"]
    assert "hooded" in details["jacket"] and "dark-red" in details["jacket"]
    assert "backpack" in details["backpack"] and "dark-red" in details["backpack"]
    assert "dark" in details["trousers"] and "sneakers" in details["shoes"]
    # the frame shows him from behind: the face is explicitly not fixed by it
    assert "NOT visible in the canonical frame" in details["face"]
    prompt = definition.prompt.lower()
    for phrase in ("curly dark hair", "black hooded jacket", "dark-red accents", "trousers", "sneakers", "backpack"):
        assert phrase in prompt, phrase


def test_style_constraints_follow_the_canonical_look(definition):
    direction = " ".join(definition.visual_direction).lower()
    for phrase in ("graphic novel", "ink contours", "cel shading", "halftone", "hatch"):
        assert phrase in direction, phrase
        assert phrase in definition.prompt.lower(), phrase
    assert "photorealistic" in definition.negative_prompt and "3D render" in definition.negative_prompt


def test_no_franchise_motifs(definition):
    # What the benchmark ASKS for must not contain them ...
    wanted = " ".join([
        definition.prompt, " ".join(definition.visual_direction),
        " ".join(str(v) for k, v in definition.subject_details.items() if k != "forbidden"),
        " ".join(str(v) for v in definition.composition.values()),
    ]).lower()
    for word in ("spider", "web", "mask", "emblem", "logo", "cape", "superhero", "costume", "armor", "weapon"):
        assert word not in wanted, word
    # ... and it says so explicitly.
    for word in ("mask", "logo", "emblem", "cape", "superhero costume", "text"):
        assert word in definition.negative_prompt, word
    forbidden = definition.subject_details["forbidden"]
    for word in ("mask", "logo", "emblem", "text", "costume"):
        assert f"no {word}" in forbidden, word
    assert "face is uncovered" in definition.prompt


def test_ready_copy_runs_on_the_fake_backend(definition, runner, project, input_files):
    import dataclasses

    frame = project / CANONICAL
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"stand-in for the canonical frame")
    ready = dataclasses.replace(definition, status="ready")
    request = ready.to_request(seed=3, output_resolution=SIZE)
    assert isinstance(request, ImageGenerationRequest) and request.reference_images == [CANONICAL]

    record = runner.run_definition(ready, FakeImageBackend(), seed=3, output_resolution=SIZE, run_id="sheet-1")
    assert record.task_type == "image_generation" and record.status == "success"
    assert record.reference_assets == [CANONICAL] and list(record.input_checksums) == [CANONICAL]
    assert record.task_parameters["composition"]["views"][0] == "front"
    assert record.task_parameters["subject_details"]["shoes"].startswith("white low-top sneakers")
    assert len(record.image_output["images"]) == 1


# --- the evaluation schema ---

def make_eval(**overrides):
    fields = dict(benchmark_id=BENCH, run_id="run-1", evaluator="reviewer-a")
    fields.update({name: 6 for name in CHARACTER_SHEET_CRITERIA})
    fields.update(overrides)
    return CharacterSheetEvaluation(**fields)


@pytest.fixture
def store(tmp_path):
    return CharacterSheetEvaluationStore(tmp_path / "character_sheet_evaluations")


def test_criteria_are_exactly_the_character_sheet_set():
    assert set(CHARACTER_SHEET_CRITERIA) == EXPECTED
    assert list(CHARACTER_SHEET_CRITERIA) == [
        "identity_consistency", "style_match", "outfit_fidelity", "line_quality", "anatomy",
    ]


def test_every_criterion_higher_is_better():
    for name, description in CHARACTER_SHEET_CRITERIA.items():
        assert description.startswith("10 = "), name
        assert "; 1 = " in description, name
    assert "every view" in CHARACTER_SHEET_CRITERIA["identity_consistency"]
    assert "canonical frame" in CHARACTER_SHEET_CRITERIA["style_match"]


def test_criteria_are_documented():
    text = (REPO_ROOT / "docs" / "BENCHMARK_CHARACTER_SHEET_HERO_01.md").read_text(encoding="utf-8")
    for name in CHARACTER_SHEET_CRITERIA:
        assert f"`{name}`" in text, name


def test_other_evaluation_schemas_unchanged():
    assert "character_identity" in COMIC_CRITERIA and len(COMIC_CRITERIA) == 10
    assert "identity_consistency" not in ComicAnimationEvaluation.__dataclass_fields__


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


@pytest.mark.parametrize("bad", [
    dict(evaluator="../x"), dict(evaluator=" a "), dict(run_id=""), dict(notes=None), dict(revision=0),
])
def test_invalid_identity_fields_rejected(bad):
    with pytest.raises(ValueError):
        make_eval(**bad)


# --- persistence and revisions ---

def test_persists_and_loads(store, tmp_path):
    path = store.save(make_eval(notes="back view lost the backpack straps"))
    assert path == tmp_path / "character_sheet_evaluations" / BENCH / "run-1" / "reviewer-a.r001.json"
    loaded = store.load(BENCH, "run-1", "reviewer-a")
    assert loaded.notes == "back view lost the backpack straps"
    assert loaded.revision == 1 and loaded.schema_version == 1 and loaded.timestamp
    assert loaded == CharacterSheetEvaluation.from_dict(loaded.to_dict())


def test_no_accidental_overwrite(store):
    path = store.save(make_eval(anatomy=3))
    before = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="save_revision"):
        store.save(make_eval(anatomy=9))
    assert path.read_text(encoding="utf-8") == before


def test_explicit_revisions_keep_history(store):
    store.save(make_eval(style_match=4))
    second = store.save_revision(make_eval(style_match=5))
    store.save_revision(make_eval(style_match=7))
    assert second.name == "reviewer-a.r002.json"
    assert store.load(BENCH, "run-1", "reviewer-a").style_match == 7
    assert store.load(BENCH, "run-1", "reviewer-a", revision=1).style_match == 4
    assert [e.revision for e in store.list_evaluations(BENCH, "run-1")] == [1, 2, 3]


def test_revision_requires_existing(store):
    with pytest.raises(FileNotFoundError):
        store.save_revision(make_eval())


def test_stores_reject_each_others_evaluations(store, tmp_path):
    comic = ComicAnimationEvaluation(
        benchmark_id=BENCH, run_id="run-1", evaluator="x", **{name: 5 for name in COMIC_CRITERIA}
    )
    with pytest.raises(TypeError):
        store.save(comic)
    with pytest.raises(TypeError):
        ComicAnimationEvaluationStore(tmp_path / "comic").save(make_eval())


def test_default_directory_is_under_benchmarks(project):
    assert paths.character_sheet_evaluations_dir(project) == (
        project / "benchmarks" / "character_sheet_evaluations"
    ).resolve()


# --- linking to runs ---

@pytest.fixture
def sheet_request(project, input_files):
    reference = project / "benchmarks" / "references" / "ref.png"
    return ImageGenerationRequest(
        prompt="character sheet", output_resolution=SIZE, seed=1, reference_images=[str(reference)],
    )


def test_linked_store_accepts_a_successful_image_generation_run(tmp_path, runner, sheet_request):
    record = runner.run(sheet_request, FakeImageBackend(), benchmark_id=BENCH, run_id="run-1")
    store = CharacterSheetEvaluationStore(tmp_path / "evals", results=runner.logger)
    store.save(make_eval())
    with pytest.raises(FileNotFoundError):
        store.save(make_eval(run_id="missing"))
    assert runner.logger.load(BENCH, "run-1") == record  # the run record is untouched
    assert record.quality_metrics == {}


def test_linked_store_rejects_a_failed_run(tmp_path, runner, sheet_request):
    class Fails(FakeImageBackend):
        def generate(self, request, output_dir):
            raise ValueError("x")

    with pytest.raises(BenchmarkRunError):
        runner.run(sheet_request, Fails(), benchmark_id=BENCH, run_id="run-1")
    store = CharacterSheetEvaluationStore(tmp_path / "evals", results=runner.logger)
    with pytest.raises(ValueError, match="did not succeed"):
        store.save(make_eval())


def test_linked_store_rejects_video_runs(tmp_path, runner, generation_request_factory):
    runner.run(generation_request_factory(), FakeGenerationBackend(), benchmark_id=BENCH, run_id="run-1")
    store = CharacterSheetEvaluationStore(tmp_path / "evals", results=runner.logger)
    with pytest.raises(ValueError, match="image_generation"):
        store.save(make_eval())


def test_comic_animation_store_rejects_image_runs(tmp_path, runner, sheet_request):
    runner.run(sheet_request, FakeImageBackend(), benchmark_id=BENCH, run_id="run-1")
    store = ComicAnimationEvaluationStore(tmp_path / "comic", results=runner.logger)
    comic = ComicAnimationEvaluation(
        benchmark_id=BENCH, run_id="run-1", evaluator="x", **{name: 5 for name in COMIC_CRITERIA}
    )
    with pytest.raises(ValueError, match="video_generation"):
        store.save(comic)
