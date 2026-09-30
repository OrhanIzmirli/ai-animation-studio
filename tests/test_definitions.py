import dataclasses
from pathlib import Path

import pytest

from ai_studio.benchmarking import DraftBenchmarkError, Resolution, load_definition, load_definitions
from ai_studio.generation.request import VideoGenerationRequest

DEFINITIONS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "definitions"


def test_repository_placeholder_definition_loads():
    definition = load_definition(DEFINITIONS_DIR / "object_replacement_01.yaml")
    assert definition.benchmark_id == "object_replacement_01"
    assert definition.source_video is None
    assert definition.reference_images == []
    assert definition.preserve_regions == []
    assert not definition.is_complete


def test_all_repository_definitions_load():
    assert "object_replacement_01" in load_definitions(DEFINITIONS_DIR)


def test_placeholder_cannot_become_request():
    definition = load_definition(DEFINITIONS_DIR / "object_replacement_01.yaml")
    with pytest.raises(DraftBenchmarkError):
        definition.to_request(seed=1, output_resolution=Resolution(width=64, height=64))
    # Even if someone marked it ready, the missing inputs are still refused.
    with pytest.raises(ValueError, match="incomplete"):
        dataclasses.replace(definition, status="ready").to_request(
            seed=1, output_resolution=Resolution(width=64, height=64)
        )


def write(tmp_path, text, name="def.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_complete_definition_to_request(tmp_path):
    path = write(tmp_path, """
benchmark_id: object_replacement_test
task_type: video_editing
status: ready
source_video: benchmarks/inputs/clip.mp4
reference_images:
  - benchmarks/references/a.png
  - benchmarks/references/b.png
prompt: replace the mug with the teapot
negative_prompt: blurry
target_object: mug
preserve_regions: [background, hands]
notes: test
""")
    definition = load_definition(path)
    request = definition.to_request(
        seed=7,
        output_resolution=Resolution(width=1280, height=720),
        backend_parameters={"strength": 0.9},
    )
    assert request.reference_images == ["benchmarks/references/a.png", "benchmarks/references/b.png"]
    assert request.backend_parameters == {"strength": 0.9}
    assert definition.preserve_regions == ["background", "hands"]


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        load_definition(write(tmp_path, "benchmark_id: x\ntask_type: video_editing\nsource_vdeo: typo.mp4\n"))


def test_missing_benchmark_id_rejected(tmp_path):
    with pytest.raises(TypeError):
        load_definition(write(tmp_path, "task_type: video_editing\nprompt: hi\n"))


def test_non_mapping_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_definition(write(tmp_path, "- a\n- b\n"))


def test_duplicate_ids_rejected(tmp_path):
    write(tmp_path, "benchmark_id: same\ntask_type: video_editing\n", "a.yaml")
    write(tmp_path, "benchmark_id: same\ntask_type: video_editing\n", "b.yaml")
    with pytest.raises(ValueError, match="duplicate"):
        load_definitions(tmp_path)


# --- task_type and structured generation fields ---

def test_task_type_must_be_declared(tmp_path):
    with pytest.raises(ValueError, match="task_type"):
        load_definition(write(tmp_path, "benchmark_id: x\nprompt: hi\n"))


def test_unknown_task_type_rejected(tmp_path):
    with pytest.raises(ValueError, match="task_type"):
        load_definition(write(tmp_path, "benchmark_id: x\ntask_type: video_repair\n"))


def test_repository_definitions_declare_task_types():
    definitions = load_definitions(DEFINITIONS_DIR)
    assert definitions["object_replacement_01"].task_type == "video_editing"
    assert definitions["comic_rooftop_01"].task_type == "video_generation"


def test_comic_rooftop_structured_fields():
    definition = load_definition(DEFINITIONS_DIR / "comic_rooftop_01.yaml")
    assert definition.duration_seconds == 5
    assert "push-in" in definition.camera_motion
    assert any("head turn" in m for m in definition.subject_motion)
    assert "halftone print texture" in definition.visual_direction
    assert "stable facial structure" in definition.visual_direction
    assert definition.source_video is None


def test_comic_rooftop_becomes_generation_request():
    # The repository file is a draft; simulate the human decision to mark it ready.
    definition = dataclasses.replace(load_definition(DEFINITIONS_DIR / "comic_rooftop_01.yaml"), status="ready")
    request = definition.to_request(seed=3, output_resolution=Resolution(width=1280, height=720), fps=24)
    assert isinstance(request, VideoGenerationRequest)
    assert request.duration_seconds == 5 and request.fps == 24
    assert request.camera_motion == definition.camera_motion
    assert request.subject_motion == definition.subject_motion
    assert request.visual_direction == definition.visual_direction
    assert request.prompt == definition.prompt


GENERATION_YAML = """
benchmark_id: gen_test
task_type: video_generation
status: ready
prompt: a rooftop at sunset
duration_seconds: 4.5
camera_motion: {type: push_in, speed: slow}
subject_motion: [stationary]
visual_direction: graphic novel
"""


def test_generation_definition_accepts_generic_structures(tmp_path):
    definition = load_definition(write(tmp_path, GENERATION_YAML))
    assert definition.camera_motion == {"type": "push_in", "speed": "slow"}
    assert definition.is_complete


@pytest.mark.parametrize(
    "extra",
    ["source_video: benchmarks/inputs/clip.mp4", "target_object: mug", "preserve_regions: [sky]"],
)
def test_editing_fields_rejected_for_generation(tmp_path, extra):
    with pytest.raises(ValueError, match="do not apply"):
        load_definition(write(tmp_path, GENERATION_YAML + extra + "\n"))


@pytest.mark.parametrize(
    "extra", ["duration_seconds: 5", "camera_motion: pan", "subject_motion: [walk]", "visual_direction: ink"],
)
def test_generation_fields_rejected_for_editing(tmp_path, extra):
    text = "benchmark_id: e\ntask_type: video_editing\n" + extra + "\n"
    with pytest.raises(ValueError, match="do not apply"):
        load_definition(write(tmp_path, text))


@pytest.mark.parametrize("bad", ["duration_seconds: 0", "duration_seconds: soon", "camera_motion: 5"])
def test_invalid_generation_fields_rejected_at_load(tmp_path, bad):
    text = "benchmark_id: g\ntask_type: video_generation\nprompt: p\n" + bad + "\n"
    with pytest.raises(ValueError):
        load_definition(write(tmp_path, text))


def test_generation_without_duration_is_incomplete(tmp_path):
    definition = load_definition(write(tmp_path, "benchmark_id: g\ntask_type: video_generation\nstatus: ready\nprompt: p\n"))
    assert not definition.is_complete
    with pytest.raises(ValueError, match="duration_seconds"):
        definition.to_request(seed=1, output_resolution=Resolution(width=64, height=64))


def test_editing_definition_rejects_fps(tmp_path):
    definition = load_definition(write(tmp_path, """
benchmark_id: e
task_type: video_editing
status: ready
source_video: benchmarks/inputs/clip.mp4
prompt: edit
"""))
    with pytest.raises(ValueError, match="fps"):
        definition.to_request(seed=1, output_resolution=Resolution(width=64, height=64), fps=24)
