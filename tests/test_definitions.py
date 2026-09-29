from pathlib import Path

import pytest

from ai_studio.benchmarking import Resolution, load_definition, load_definitions

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
    with pytest.raises(ValueError, match="incomplete"):
        definition.to_request(seed=1, output_resolution=Resolution(width=64, height=64))


def write(tmp_path, text, name="def.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_complete_definition_to_request(tmp_path):
    path = write(tmp_path, """
benchmark_id: object_replacement_test
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
        load_definition(write(tmp_path, "benchmark_id: x\nsource_vdeo: typo.mp4\n"))


def test_missing_benchmark_id_rejected(tmp_path):
    with pytest.raises(TypeError):
        load_definition(write(tmp_path, "prompt: hi\n"))


def test_non_mapping_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_definition(write(tmp_path, "- a\n- b\n"))


def test_duplicate_ids_rejected(tmp_path):
    write(tmp_path, "benchmark_id: same\n", "a.yaml")
    write(tmp_path, "benchmark_id: same\n", "b.yaml")
    with pytest.raises(ValueError, match="duplicate"):
        load_definitions(tmp_path)
