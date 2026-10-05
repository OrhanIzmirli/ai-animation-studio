"""Still-image capabilities: requests, the result type, the backend contracts,
the fake backend, the registry, and image tasks through BenchmarkRunner and
the CLI. No model, no network."""

import dataclasses
import json
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import (
    SCHEMA_VERSION,
    BenchmarkRunError,
    DraftBenchmarkError,
    Resolution,
    load_definition,
)
from ai_studio.benchmarking.in_progress import find_abandoned, marker_path, record_abandoned
from ai_studio.benchmarking.runner import sha256_file
from ai_studio.benchmarking.schema import IMAGE_TASK_TYPES, TASK_TYPES, VIDEO_TASK_TYPES
from ai_studio.cli import run_benchmark
from ai_studio.generation.backends import (
    BACKENDS,
    IMAGE_BACKENDS,
    Backend,
    BackendMetadata,
    FakeImageBackend,
    ImageEditingBackend,
    ImageGenerationBackend,
    VideoBackend,
    all_backends,
    get_backend,
)
from ai_studio.generation.capabilities import (
    IMAGE_CAPABILITIES,
    VIDEO_CAPABILITIES,
    Capability,
    UnsupportedCapabilityError,
)
from ai_studio.generation.image import ImageEditingRequest, ImageGenerationRequest, ImageGenerationResult
from ai_studio.generation.result import GenerationResult
from ai_studio.media import image_dimensions
from ai_studio.protection import is_read_only
from conftest import FakeBackend, FakeGenerationBackend, fake_video_bytes

BENCH = "character_sheet_test"
SIZE = Resolution(width=96, height=64)


@pytest.fixture
def images(project, input_files):
    """Input pictures: a source image, a mask and three references."""
    folder = project / "benchmarks" / "references"
    files = {}
    for name in ("source", "mask", "ref_a", "ref_b", "ref_c"):
        files[name] = folder / f"{name}.png"
        files[name].write_bytes(f"picture {name}".encode())
    return files


def gen_request(images=None, **overrides):
    fields = dict(
        prompt="character sheet of a young man in dark hooded streetwear", negative_prompt="photorealistic",
        output_resolution=SIZE, seed=7,
        reference_images=[str(images["ref_a"])] if images else [],
        composition={"views": ["front", "back"]}, subject_details=["dark-red cuffs"],
        visual_direction="graphic novel",
    )
    fields.update(overrides)
    return ImageGenerationRequest(**fields)


def edit_request(images, **overrides):
    fields = dict(
        source_image=str(images["source"]), prompt="turn his head to the left",
        output_resolution=SIZE, seed=7, reference_images=[str(images["ref_a"])],
    )
    fields.update(overrides)
    return ImageEditingRequest(**fields)


# --- capabilities ---

def test_image_capabilities_exist_beside_the_video_ones():
    assert Capability.IMAGE_GENERATION.value == "image_generation"
    assert Capability.IMAGE_EDITING.value == "image_editing"
    assert IMAGE_CAPABILITIES == {Capability.IMAGE_GENERATION, Capability.IMAGE_EDITING}
    assert VIDEO_CAPABILITIES == {Capability.VIDEO_GENERATION, Capability.VIDEO_EDITING}
    assert IMAGE_CAPABILITIES | VIDEO_CAPABILITIES == set(Capability)


def test_capability_values_are_the_record_task_types():
    assert {c.value for c in Capability} == set(TASK_TYPES)
    assert {c.value for c in IMAGE_CAPABILITIES} == set(IMAGE_TASK_TYPES)
    assert {c.value for c in VIDEO_CAPABILITIES} == set(VIDEO_TASK_TYPES)


# --- requests ---

def test_generation_request_protocol(images):
    request = gen_request(images, reference_images=[str(images["ref_a"]), str(images["ref_b"])], image_count=3)
    assert request.capability is Capability.IMAGE_GENERATION
    assert request.input_paths == [str(images["ref_a"]), str(images["ref_b"])]
    assert request.task_parameters() == {
        "image_count": 3, "composition": {"views": ["front", "back"]},
        "subject_details": ["dark-red cuffs"], "visual_direction": "graphic novel",
    }
    resolved = request.with_resolved_paths(lambda p: p.upper())
    assert resolved.reference_images == [str(images["ref_a"]).upper(), str(images["ref_b"]).upper()]
    assert resolved.prompt == request.prompt and request.reference_images[0] == str(images["ref_a"])


def test_generation_request_defaults():
    request = ImageGenerationRequest(prompt="a city", output_resolution=SIZE, seed=0)
    assert request.image_count == 1 and request.reference_images == [] and request.input_paths == []
    assert request.negative_prompt is None and request.backend_parameters == {}


def test_editing_request_protocol(images):
    request = edit_request(images, mask=str(images["mask"]))
    assert request.capability is Capability.IMAGE_EDITING
    assert request.input_paths == [str(images["source"]), str(images["mask"]), str(images["ref_a"])]
    assert request.task_parameters()["source_image"] == str(images["source"])
    assert request.task_parameters()["mask"] == str(images["mask"])
    resolved = request.with_resolved_paths(lambda p: "R:" + p)
    assert resolved.source_image == "R:" + str(images["source"]) and resolved.mask == "R:" + str(images["mask"])
    assert edit_request(images).input_paths == [str(images["source"]), str(images["ref_a"])]
    assert edit_request(images).with_resolved_paths(str).mask is None


@pytest.mark.parametrize("bad", [
    dict(prompt="  "), dict(prompt=None), dict(negative_prompt=5), dict(output_resolution=(96, 64)),
    dict(seed="7"), dict(seed=True), dict(reference_images="ref.png"), dict(reference_images=[""]),
    dict(reference_images=["a.png", "a.png"]), dict(image_count=0), dict(image_count=1.0),
    dict(image_count=True), dict(composition=5), dict(subject_details=[""]), dict(visual_direction=""),
    dict(backend_parameters=[]), dict(backend_parameters={"x": object()}),
])
def test_invalid_generation_request_rejected(bad):
    with pytest.raises(ValueError):
        gen_request(**bad)


@pytest.mark.parametrize("bad", [
    dict(source_image=""), dict(source_image=None), dict(mask=""),
    dict(reference_images=["source.png"], source_image="source.png"),
    dict(reference_images=["mask.png"], mask="mask.png"),
    dict(mask="source.png", source_image="source.png"),
    dict(image_count=0), dict(prompt=""),
])
def test_invalid_editing_request_rejected(bad):
    fields = dict(source_image="source.png", prompt="p", output_resolution=SIZE, seed=1)
    fields.update(bad)
    with pytest.raises(ValueError):
        ImageEditingRequest(**fields)


# --- result ---

def make_result(**overrides):
    fields = dict(
        output_images=["out/image_01.png"], backend_name="b", backend_version="1",
        model_name="m", model_version="1",
    )
    fields.update(overrides)
    return ImageGenerationResult(**fields)


def test_result_defaults_and_output_paths():
    result = make_result(output_images=["out/a.png", "out/b.png"])
    assert result.output_paths == ["out/a.png", "out/b.png"]
    assert result.seed_used is None and result.provenance == {} and result.parameters == {}
    assert result.inference_steps is None and result.guidance is None
    # the video result answers the same question
    video = GenerationResult(output_video="out/v.mp4", backend_name="b", backend_version="1",
                             model_name="m", model_version="1")
    assert video.output_paths == ["out/v.mp4"]


@pytest.mark.parametrize("bad", [
    dict(output_images=[]), dict(output_images="out/a.png"), dict(output_images=[""]),
    dict(output_images=["a.png", "a.png"]), dict(backend_name=""), dict(model_version=None),
    dict(seed_used="7"), dict(seed_used=True), dict(provenance=[]), dict(provenance={"x": object()}),
    dict(parameters=None), dict(warnings="careful"), dict(generation_time_seconds=-1),
])
def test_invalid_result_rejected(bad):
    with pytest.raises(ValueError):
        make_result(**bad)


# --- contracts ---

def test_image_contracts_declare_one_capability_each():
    assert ImageGenerationBackend.capabilities == {Capability.IMAGE_GENERATION}
    assert ImageEditingBackend.capabilities == {Capability.IMAGE_EDITING}
    assert ImageGenerationBackend.max_reference_images == 0  # nothing is assumed
    assert ImageEditingBackend.max_reference_images == 0 and ImageEditingBackend.supports_mask is False
    assert Backend.capabilities == frozenset()
    assert VideoBackend is Backend  # the historical name keeps working


def test_image_contracts_are_abstract():
    for contract in (ImageGenerationBackend, ImageEditingBackend):
        with pytest.raises(TypeError):
            contract()
        for method in ("metadata", "validate_inputs", "prepare", "generate"):
            assert method in contract.__abstractmethods__


def test_a_backend_cannot_claim_an_image_capability_without_the_contract():
    with pytest.raises(TypeError, match="without implementing"):
        class ClaimsImages(FakeGenerationBackend):
            capabilities = frozenset({Capability.VIDEO_GENERATION, Capability.IMAGE_GENERATION})

    class GenerationOnly(FakeImageBackend):
        capabilities = frozenset({Capability.IMAGE_GENERATION})

    assert GenerationOnly.capabilities == {Capability.IMAGE_GENERATION}


def test_video_backends_have_no_image_capability():
    for backend_class in BACKENDS.values():
        assert not backend_class.capabilities & IMAGE_CAPABILITIES


# --- the fake backend and the registry ---

def test_fake_backend_implements_both_image_contracts():
    assert FakeImageBackend.capabilities == IMAGE_CAPABILITIES
    assert FakeImageBackend.supports(Capability.IMAGE_EDITING)
    assert not FakeImageBackend.supports(Capability.VIDEO_GENERATION)
    assert FakeImageBackend.max_reference_images == 8 and FakeImageBackend.supports_mask is True
    meta = FakeImageBackend().metadata()
    assert isinstance(meta, BackendMetadata) and meta.backend_name == "fake-image"


def test_registry_keeps_image_backends_apart_and_resolves_both():
    assert IMAGE_BACKENDS == {"fake-image": FakeImageBackend}
    assert "fake-image" not in BACKENDS
    assert all_backends() == {**BACKENDS, **IMAGE_BACKENDS}
    assert get_backend("fake-image") is FakeImageBackend
    assert get_backend("wan-ti2v-5b") is BACKENDS["wan-ti2v-5b"]
    for name, backend_class in IMAGE_BACKENDS.items():
        assert issubclass(backend_class, Backend)
        assert backend_class().metadata().backend_name == name
    with pytest.raises(KeyError) as info:
        get_backend("nano")
    assert "fake-image" in str(info.value) and "wan-ti2v-5b" in str(info.value)


def test_fake_backend_writes_deterministic_pngs_of_the_requested_size(images, tmp_path):
    backend = FakeImageBackend()
    request = gen_request(images, image_count=2)
    backend.validate_inputs(request)
    backend.prepare()
    first = backend.generate(request, tmp_path / "a")
    again = backend.generate(request, tmp_path / "b")
    other_seed = backend.generate(dataclasses.replace(request, seed=8), tmp_path / "c")
    other_ref = backend.generate(
        dataclasses.replace(request, reference_images=[str(images["ref_b"])]), tmp_path / "d",
    )

    assert [Path(p).name for p in first.output_images] == ["image_01.png", "image_02.png"]
    for path in first.output_images:
        assert image_dimensions(path) == SIZE
    hashes = [[sha256_file(p) for p in result.output_images] for result in (first, again, other_seed, other_ref)]
    assert hashes[0] == hashes[1]  # same request, same bytes
    assert hashes[0][0] != hashes[0][1]  # the two images of one request differ
    assert hashes[0] != hashes[2] and hashes[0] != hashes[3]  # seed and references matter
    assert first.seed_used == 7
    assert first.provenance["input_sha256"]["reference_images"] == [sha256_file(images["ref_a"])]
    assert first.provenance["task"] == "image_generation"
    assert any("test patterns" in warning for warning in first.warnings)


def test_fake_backend_edits_depend_on_the_source_and_mask(images, tmp_path):
    backend = FakeImageBackend()
    plain = backend.generate(edit_request(images), tmp_path / "a")
    masked = backend.generate(edit_request(images, mask=str(images["mask"])), tmp_path / "b")
    other = backend.generate(edit_request(images, source_image=str(images["ref_b"])), tmp_path / "c")
    hashes = [sha256_file(result.output_images[0]) for result in (plain, masked, other)]
    assert len(set(hashes)) == 3
    assert plain.provenance["task"] == "image_editing"
    assert plain.provenance["input_sha256"]["source_image"] == sha256_file(images["source"])
    assert masked.provenance["input_sha256"]["mask"] == sha256_file(images["mask"])


def test_fake_backend_rejects_what_it_cannot_do():
    backend = FakeImageBackend()
    with pytest.raises(ValueError, match="4096"):
        backend.validate_inputs(gen_request(output_resolution=Resolution(width=5000, height=64)))
    with pytest.raises(ValueError, match="no parameters"):
        backend.validate_inputs(gen_request(backend_parameters={"quality": "high"}))


# --- image tasks through the runner ---

def test_image_generation_run_is_recorded(runner, project, images):
    request = gen_request(images, reference_images=[str(images["ref_a"]), str(images["ref_b"])], image_count=2)
    record = runner.run(request, FakeImageBackend(), benchmark_id=BENCH, run_id="run-1", notes="first sheet")

    data = json.loads(runner.logger.path_for(BENCH, "run-1").read_text(encoding="utf-8"))
    assert record == runner.logger.load(BENCH, "run-1")
    assert data["schema_version"] == SCHEMA_VERSION and data["status"] == "success"
    assert data["task_type"] == "image_generation"
    assert data["backend"] == "fake-image" and data["model"] == "fake-image-pattern"
    assert data["seed"] == 7 and data["resolution"] == {"width": 96, "height": 64}
    assert data["output_video"] is None and data["output_media"] is None
    assert data["fps"] is None and data["frame_count"] is None and data["source_video"] is None
    assert data["start_frame"] is None
    assert data["reference_assets"] == ["benchmarks/references/ref_a.png", "benchmarks/references/ref_b.png"]
    assert data["input_checksums"] == {
        "benchmarks/references/ref_a.png": sha256_file(images["ref_a"]),
        "benchmarks/references/ref_b.png": sha256_file(images["ref_b"]),
    }
    assert data["task_parameters"] == {
        "image_count": 2, "composition": {"views": ["front", "back"]},
        "subject_details": ["dark-red cuffs"], "visual_direction": "graphic novel", "run_purpose": "benchmark",
    }
    output = data["image_output"]
    assert set(output) == {"images", "seed_used", "provenance"}
    assert output["seed_used"] == 7
    assert output["provenance"]["generator"].startswith("deterministic")  # model-reported, copied as is
    assert [image["path"] for image in output["images"]] == [
        f"outputs/{BENCH}/run-1/image_01.png", f"outputs/{BENCH}/run-1/image_02.png",
    ]
    for image in output["images"]:  # measured by the runner
        path = project / image["path"]
        assert image["sha256"] == sha256_file(path)
        assert (image["width"], image["height"]) == (96, 64)
        assert is_read_only(path)  # recorded run files are protected
    assert "first sheet" in data["notes"] and "warning: fake backend" in data["notes"]
    assert "requested seed" not in data["notes"]
    assert not marker_path(runner.outputs_dir / BENCH / "run-1").exists()
    assert record.generation_time_seconds is not None and record.total_time_seconds is not None


def test_image_editing_run_is_recorded(runner, images):
    request = edit_request(images, mask=str(images["mask"]))
    record = runner.run(request, FakeImageBackend(), benchmark_id=BENCH, run_id="edit-1")
    assert record.task_type == "image_editing" and record.status == "success"
    assert set(record.input_checksums) == {
        "benchmarks/references/source.png", "benchmarks/references/mask.png", "benchmarks/references/ref_a.png",
    }
    assert record.reference_assets == ["benchmarks/references/ref_a.png"]  # the source is not a reference
    assert record.task_parameters["source_image"] == str(images["source"])
    assert len(record.image_output["images"]) == 1


def test_image_request_needs_a_benchmark_id(runner, images):
    with pytest.raises(ValueError, match="benchmark_id is required"):
        runner.run(gen_request(images), FakeImageBackend())


def test_image_request_on_a_video_backend_is_refused(runner, images):
    with pytest.raises(UnsupportedCapabilityError, match="image_generation"):
        runner.run(gen_request(images), FakeGenerationBackend(), benchmark_id=BENCH, run_id="r")
    assert not (runner.outputs_dir / BENCH).exists() and not runner.logger.path_for(BENCH, "r").exists()


def test_video_request_on_an_image_backend_is_refused(runner, generation_request_factory, request_factory):
    with pytest.raises(UnsupportedCapabilityError):
        runner.run(generation_request_factory(), FakeImageBackend(), benchmark_id=BENCH)
    with pytest.raises(UnsupportedCapabilityError):
        runner.run(request_factory(), FakeImageBackend())


def test_more_references_than_the_backend_uses_is_refused_not_truncated(runner, images):
    class TwoReferences(FakeImageBackend):
        max_reference_images = 2

    three = [str(images[name]) for name in ("ref_a", "ref_b", "ref_c")]
    with pytest.raises(UnsupportedCapabilityError, match="at most 2 reference image"):
        runner.run(gen_request(images, reference_images=three), TwoReferences(), benchmark_id=BENCH, run_id="r")
    assert not (runner.outputs_dir / BENCH).exists() and not runner.logger.path_for(BENCH, "r").exists()
    record = runner.run(gen_request(images, reference_images=three[:2]), TwoReferences(), benchmark_id=BENCH)
    assert record.status == "success"


def test_mask_needs_a_backend_that_declares_it(runner, images):
    class NoMask(FakeImageBackend):
        supports_mask = False

    with pytest.raises(UnsupportedCapabilityError, match="supports_mask"):
        runner.run(edit_request(images, mask=str(images["mask"])), NoMask(), benchmark_id=BENCH, run_id="r")
    assert not runner.logger.path_for(BENCH, "r").exists()
    assert runner.run(edit_request(images), NoMask(), benchmark_id=BENCH).status == "success"


def test_reference_limit_does_not_apply_to_video_backends(runner, request_factory):
    assert not hasattr(FakeBackend, "max_reference_images")
    assert runner.run(request_factory(), FakeBackend()).status == "success"


class Misbehaving(FakeImageBackend):
    """A fake that can break one promise of the contract."""

    def __init__(self, mode):
        self.mode = mode

    def generate(self, request, output_dir):
        result = super().generate(request, output_dir)
        if self.mode == "video_result":
            video = Path(output_dir) / "out.mp4"
            video.write_bytes(fake_video_bytes())
            return GenerationResult(output_video=str(video), backend_name="x", backend_version="1",
                                    model_name="m", model_version="1")
        if self.mode == "extra_image":
            extra = Path(output_dir) / "extra.png"
            extra.write_bytes(Path(result.output_images[0]).read_bytes())
            return dataclasses.replace(result, output_images=[*result.output_images, str(extra)])
        if self.mode == "outside":
            stray = Path(output_dir).parent / "stray.png"
            stray.write_bytes(b"x")
            return dataclasses.replace(result, output_images=[str(stray)])
        if self.mode == "missing":
            return dataclasses.replace(result, output_images=[str(Path(output_dir) / "nowhere.png")])
        if self.mode == "touches_reference":
            Path(request.reference_images[0]).write_bytes(b"changed")
        if self.mode == "no_seed":
            return dataclasses.replace(result, seed_used=None)
        if self.mode == "other_seed":
            return dataclasses.replace(result, seed_used=request.seed + 1)
        if self.mode == "wrong_size":
            Path(result.output_images[0]).write_bytes(b"\xff\xd8 not really a jpeg")
        return result


@pytest.mark.parametrize("mode, error, match", [
    ("video_result", "TypeError", "expected ImageGenerationResult"),
    ("extra_image", "RuntimeError", "asked for 1 image"),
    ("outside", "RuntimeError", "must be a file inside"),
    ("missing", "RuntimeError", "must be a file inside"),
    ("touches_reference", "RuntimeError", "modified an input file"),
])
def test_broken_image_result_fails_verification(runner, images, mode, error, match):
    with pytest.raises(BenchmarkRunError, match=match) as info:
        runner.run(gen_request(images), Misbehaving(mode), benchmark_id=BENCH, run_id="run-x")
    record = info.value.record
    assert info.value.stage == "verification" and record.failure_stage == "verification"
    assert record.error_type == error and record.task_type == "image_generation"
    assert record.image_output is None and record.output_video is None
    assert runner.logger.load(BENCH, "run-x") == record


def test_video_backend_returning_an_image_result_fails_verification(runner, generation_request_factory, tmp_path):
    class Confused(FakeGenerationBackend):
        def generate(self, request, output_dir):
            super().generate(request, output_dir)
            return make_result(output_images=[str(Path(output_dir) / self.output_name)])

    with pytest.raises(BenchmarkRunError, match="expected GenerationResult") as info:
        runner.run(generation_request_factory(), Confused(), benchmark_id=BENCH)
    record = info.value.record
    assert record.failure_stage == "verification" and record.task_type == "video_generation"
    assert record.backend == "fake-gen" and record.image_output is None and record.output_video is None


@pytest.mark.parametrize("mode, text", [
    ("no_seed", "did not use the requested seed 7 (no controllable seed)"),
    ("other_seed", "did not use the requested seed 7 (seed 8)"),
])
def test_unhonoured_seed_is_recorded_with_a_warning(runner, images, mode, text):
    record = runner.run(gen_request(images), Misbehaving(mode), benchmark_id=BENCH)
    assert record.status == "success" and record.seed == 7  # what was asked for
    assert record.image_output["seed_used"] == (None if mode == "no_seed" else 8)  # what happened
    assert f"warning: backend {text}" in record.notes


def test_unreadable_or_different_size_is_a_warning(runner, images):
    record = runner.run(gen_request(images), Misbehaving("wrong_size"), benchmark_id=BENCH, run_id="u")
    (image,) = record.image_output["images"]
    assert image["width"] is None and image["height"] is None and image["sha256"]
    assert "not measured (not a PNG or JPEG header)" in record.notes

    class Smaller(FakeImageBackend):
        def generate(self, request, output_dir):
            return super().generate(dataclasses.replace(request, output_resolution=Resolution(width=48, height=32)),
                                    output_dir)

    record = runner.run(gen_request(images), Smaller(), benchmark_id=BENCH, run_id="s")
    assert (record.image_output["images"][0]["width"], record.image_output["images"][0]["height"]) == (48, 32)
    assert record.resolution == SIZE  # the request
    assert "measured size 48x32" in record.notes and "differs from requested 96x64" in record.notes


def test_failed_image_generation_is_recorded_without_image_output(runner, images):
    class Fails(FakeImageBackend):
        def generate(self, request, output_dir):
            raise RuntimeError("service refused the prompt")

    with pytest.raises(BenchmarkRunError) as info:
        runner.run(gen_request(images), Fails(), benchmark_id=BENCH, run_id="run-f")
    record = info.value.record
    assert record.status == "failed" and record.failure_stage == "generation"
    assert record.task_type == "image_generation" and record.image_output is None
    assert record.backend == "fake-image" and record.task_parameters["image_count"] == 1
    assert not marker_path(runner.outputs_dir / BENCH / "run-f").exists()


def test_missing_reference_fails_validation(runner, images):
    with pytest.raises(BenchmarkRunError) as info:
        runner.run(gen_request(images, reference_images=["benchmarks/references/none.png"]),
                   FakeImageBackend(), benchmark_id=BENCH)
    assert info.value.stage == "validation" and isinstance(info.value.__cause__, FileNotFoundError)


def test_abandoned_image_run_can_be_recorded(runner, images):
    class Dies(FakeImageBackend):
        def generate(self, request, output_dir):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.run(gen_request(images), Dies(), benchmark_id=BENCH, run_id="run-dead")
    (run,) = find_abandoned(runner.outputs_dir, runner.logger, stale_after_hours=0)
    record = record_abandoned(run, runner.logger)
    assert record.task_type == "image_generation" and record.failure_stage == "abandoned"
    assert record.image_output is None and record.schema_version == SCHEMA_VERSION


def test_runner_source_names_no_image_model():
    import inspect

    import ai_studio.benchmarking.runner as runner_module

    source = inspect.getsource(runner_module).lower()
    for name in ("gemini", "banana", "openai", "gpt", "qwen", "flux", "fake"):
        assert name not in source, name


# --- definitions ---

SHEET = """\
benchmark_id: sheet_01
task_type: image_generation
status: {status}
reference_images:
  - benchmarks/references/ref_a.png
prompt: character sheet of a young man in dark hooded streetwear
negative_prompt: photorealistic
composition:
  layout: one sheet
  views: [front, back]
subject_details:
  - dark-red cuffs
visual_direction: graphic novel
"""


def write_definition(project, text, name="sheet_01"):
    path = project / "benchmarks" / "definitions" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_image_generation_definition_becomes_a_request(project):
    definition = load_definition(write_definition(project, SHEET.format(status="ready") + "image_count: 2\n"))
    assert definition.task_type == "image_generation" and definition.is_complete
    request = definition.to_request(seed=3, output_resolution=SIZE, backend_parameters={"quality": "high"})
    assert isinstance(request, ImageGenerationRequest)
    assert request.reference_images == ["benchmarks/references/ref_a.png"]  # still project-relative
    assert request.image_count == 2 and request.seed == 3 and request.output_resolution == SIZE
    assert request.composition == {"layout": "one sheet", "views": ["front", "back"]}
    assert request.subject_details == ["dark-red cuffs"] and request.visual_direction == "graphic novel"
    assert request.backend_parameters == {"quality": "high"}
    with pytest.raises(ValueError, match="fps is not an image request field"):
        definition.to_request(seed=3, output_resolution=SIZE, fps=24)


def test_image_count_defaults_to_one(project):
    definition = load_definition(write_definition(project, SHEET.format(status="ready")))
    assert definition.image_count is None
    assert definition.to_request(seed=1, output_resolution=SIZE).image_count == 1


def test_draft_image_definition_cannot_be_run(runner, project, images):
    definition = load_definition(write_definition(project, SHEET.format(status="draft")))
    with pytest.raises(DraftBenchmarkError):
        definition.to_request(seed=1, output_resolution=SIZE)
    with pytest.raises(DraftBenchmarkError):
        runner.run_definition(definition, FakeImageBackend(), seed=1, output_resolution=SIZE, run_id="r")
    assert not runner.logger.path_for("sheet_01", "r").exists()


def test_image_definition_runs_through_run_definition(runner, project, images):
    definition = load_definition(write_definition(project, SHEET.format(status="ready")))
    record = runner.run_definition(definition, FakeImageBackend(), seed=5, output_resolution=SIZE, run_id="r1")
    assert record.benchmark_id == "sheet_01" and record.task_type == "image_generation"
    assert record.task_parameters["composition"]["views"] == ["front", "back"]
    assert record.image_output["images"][0]["path"] == "outputs/sheet_01/r1/image_01.png"


def test_image_editing_definition(project):
    text = (
        "benchmark_id: edit_01\ntask_type: image_editing\nstatus: ready\nprompt: turn the head\n"
        "source_image: benchmarks/references/source.png\nmask: benchmarks/references/mask.png\n"
    )
    definition = load_definition(write_definition(project, text, "edit_01"))
    request = definition.to_request(seed=1, output_resolution=SIZE)
    assert isinstance(request, ImageEditingRequest)
    assert request.source_image == "benchmarks/references/source.png"
    assert request.mask == "benchmarks/references/mask.png"

    incomplete = load_definition(write_definition(
        project, "benchmark_id: edit_02\ntask_type: image_editing\nstatus: ready\nprompt: p\n", "edit_02",
    ))
    assert not incomplete.is_complete
    with pytest.raises(ValueError, match="source_image and prompt"):
        incomplete.to_request(seed=1, output_resolution=SIZE)


@pytest.mark.parametrize("task_type, extra", [
    ("image_generation", "duration_seconds: 5"),
    ("image_generation", "start_frame: a.png"),
    ("image_generation", "camera_motion: pan"),
    ("image_generation", "source_video: a.mp4"),
    ("image_generation", "source_image: a.png"),
    ("image_generation", "mask: a.png"),
    ("image_editing", "fps: 24"),
    ("video_generation", "image_count: 2"),
    ("video_generation", "composition: wide"),
    ("video_generation", "source_image: a.png"),
    ("video_editing", "subject_details: [x]"),
    ("video_editing", "mask: a.png"),
])
def test_fields_of_another_task_type_are_rejected(project, task_type, extra):
    text = f"benchmark_id: x\ntask_type: {task_type}\nprompt: p\n{extra}\n"
    with pytest.raises(ValueError, match="do not apply"):
        load_definition(write_definition(project, text, "x"))


@pytest.mark.parametrize("extra", [
    "image_count: 0", "image_count: many", "composition: 5", "subject_details: ['']",
])
def test_invalid_image_fields_rejected_at_load(project, extra):
    with pytest.raises(ValueError):
        load_definition(write_definition(project, f"benchmark_id: x\ntask_type: image_generation\n{extra}\n", "x"))


def test_source_image_is_never_also_a_reference(project):
    text = (
        "benchmark_id: x\ntask_type: image_editing\nprompt: p\nsource_image: a.png\n"
        "reference_images: [a.png]\n"
    )
    with pytest.raises(ValueError, match="must not also be listed"):
        load_definition(write_definition(project, text, "x"))


def test_visual_direction_is_shared_by_video_generation_and_image_tasks(project):
    for task_type in ("video_generation", "image_generation", "image_editing"):
        text = f"benchmark_id: x\ntask_type: {task_type}\nprompt: p\nvisual_direction: [ink]\n"
        assert load_definition(write_definition(project, text, "x")).visual_direction == ["ink"]


# --- the command line ---

@pytest.fixture
def cli_project(project, images, monkeypatch):
    write_definition(project, SHEET.format(status="ready"))
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    return project


def test_cli_runs_an_image_benchmark_on_the_fake_backend(cli_project):
    lines = []
    code, written = run_benchmark(
        benchmark="sheet_01", backend="fake-image", seed=11, width=96, height=64, run_id="cli-1",
        out=lines.append,
    )
    text = "\n".join(lines)
    assert code == 0
    assert "status: success" in text
    assert "image: outputs/sheet_01/cli-1/image_01.png (96x64)" in text
    assert "metrics" not in text  # the automatic metrics are video metrics; nothing is attempted
    assert "not measured; backend reported fps" not in text
    assert written == ["benchmarks/results/sheet_01/cli-1.json", "outputs/sheet_01/cli-1"]
    record = json.loads((cli_project / written[0]).read_text(encoding="utf-8"))
    assert record["task_type"] == "image_generation" and record["seed"] == 11


def test_cli_refuses_an_image_benchmark_on_a_video_backend(cli_project):
    lines = []
    code, written = run_benchmark(
        benchmark="sheet_01", backend="kiwi-edit", seed=1, width=96, height=64, out=lines.append,
    )
    assert code == 2 and written == []
    assert "UnsupportedCapabilityError" in "\n".join(lines)
