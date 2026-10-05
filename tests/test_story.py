import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.media import Resolution
from ai_studio.story import (
    DraftShotError,
    MissingKeyframeError,
    StoryValidationError,
    asset_index,
    check_story,
    compose_prompt,
    load_character_bible,
    load_shot_plan,
    load_story_bible,
    missing_keyframes,
    parse_character_bible,
    parse_shot_plan,
    parse_story_bible,
    request_from_dict,
    shot_to_request_dict,
    validate_story,
    verify_assets,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "stories" / "example_rooftop"
ROOFTOP_SHA = "3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb"
OTHER_SHA = "ab" * 32


def raw(name):
    return yaml.safe_load((EXAMPLE_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def story_data():
    return raw("story_bible.yaml")


@pytest.fixture
def characters_data():
    return raw("character_bible.yaml")


@pytest.fixture
def plan_data():
    return raw("shot_plan.yaml")


def shot_data(plan_data, shot_id):
    return next(s for scene in plan_data["scenes"] for s in scene["shots"] if s["id"] == shot_id)


def ready(plan):
    """The plan with every shot marked ready (the human decision, simulated)."""
    scenes = tuple(
        dataclasses.replace(scene, shots=tuple(dataclasses.replace(s, status="ready") for s in scene.shots))
        for scene in plan.scenes
    )
    return dataclasses.replace(plan, scenes=scenes)


# --- the repository example ---

def test_example_files_load_and_are_consistent():
    story = load_story_bible(EXAMPLE_DIR / "story_bible.yaml")
    characters = load_character_bible(EXAMPLE_DIR / "character_bible.yaml")
    plan = load_shot_plan(EXAMPLE_DIR / "shot_plan.yaml")
    assert validate_story(plan, characters, story) == []
    check_story(plan, characters, story)
    assert story.status == plan.status == "draft"
    assert [shot.id for shot in plan.shots] == ["shot_01", "shot_02"]
    assert plan.scene_of("shot_02").id == "scene_01"
    assert plan.resolution == Resolution(width=1280, height=704)
    assert characters.get("COMIC_HERO_01").role == "protagonist"


def test_example_paths_may_be_project_relative():
    assert load_story_bible("stories/example_rooftop/story_bible.yaml").story_id == "example_rooftop"


def test_example_asset_index_and_to_do_list():
    characters = load_character_bible(EXAMPLE_DIR / "character_bible.yaml")
    plan = load_shot_plan(EXAMPLE_DIR / "shot_plan.yaml")
    index = asset_index(characters, plan)
    assert list(index) == [ROOFTOP_SHA]  # only available assets are indexed
    assert index[ROOFTOP_SHA].path == "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
    assert missing_keyframes(plan) == [("shot_01", "start", "the establishing view of the city, frame 0")]


# --- loading: shape of one file ---

def test_unknown_keys_rejected(story_data, characters_data, plan_data):
    story_data["logine"] = "typo"
    with pytest.raises(ValueError, match="unknown keys"):
        parse_story_bible(story_data)
    characters_data["characters"][0]["wardrob"] = []
    with pytest.raises(ValueError, match="unknown keys"):
        parse_character_bible(characters_data)
    shot_data(plan_data, "shot_01")["camra"] = "pan"
    with pytest.raises(ValueError, match="unknown keys"):
        parse_shot_plan(plan_data)


@pytest.mark.parametrize("key", ["story_id", "title", "logline", "visual_style"])
def test_story_required_keys(story_data, key):
    del story_data[key]
    with pytest.raises(ValueError, match="missing keys"):
        parse_story_bible(story_data)


@pytest.mark.parametrize("key", ["shot_type", "camera", "motion", "duration_seconds", "action"])
def test_shot_required_keys(plan_data, key):
    del shot_data(plan_data, "shot_01")[key]
    with pytest.raises(ValueError, match="missing keys"):
        parse_shot_plan(plan_data)


def test_load_error_names_the_file(tmp_path):
    path = tmp_path / "story_bible.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="story_bible.yaml"):
        load_story_bible(path)


def test_status_defaults_to_draft_and_is_checked(story_data, plan_data):
    del story_data["status"]
    assert parse_story_bible(story_data).status == "draft"
    del shot_data(plan_data, "shot_01")["status"]
    assert parse_shot_plan(plan_data).get_shot("shot_01").status == "draft"
    story_data["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        parse_story_bible(story_data)


@pytest.mark.parametrize(
    "field, value",
    [
        ("shot_type", "dutch_tilt_zoom"),
        ("duration_seconds", 0),
        ("duration_seconds", True),
        ("duration_seconds", "soon"),
        ("camera", ""),
        ("camera", 5),
        ("motion", []),
        ("motion", [""]),
        ("characters", "COMIC_HERO_01"),
        ("characters", ["a/b"]),
        ("action", "  "),
        ("id", "../escape"),
    ],
)
def test_invalid_shot_fields_rejected(plan_data, field, value):
    shot_data(plan_data, "shot_02")[field] = value
    with pytest.raises(ValueError):
        parse_shot_plan(plan_data)


def test_camera_accepts_the_generic_control_shapes(plan_data):
    shot_data(plan_data, "shot_01")["camera"] = {"type": "push_in", "speed": "slow"}
    shot_data(plan_data, "shot_02")["camera"] = ["low angle", "slow push-in"]
    shot_data(plan_data, "shot_02")["motion"] = "static"  # a single string is one item
    plan = parse_shot_plan(plan_data)
    assert plan.get_shot("shot_01").camera == {"type": "push_in", "speed": "slow"}
    assert plan.get_shot("shot_02").camera == ["low angle", "slow push-in"]
    assert plan.get_shot("shot_02").motion == ("static",)


@pytest.mark.parametrize("fps", [0, 23.976, True, "24"])
def test_plan_fps_must_be_a_positive_whole_number(plan_data, fps):
    plan_data["fps"] = fps
    with pytest.raises(ValueError, match="fps"):
        parse_shot_plan(plan_data)


def test_duplicate_ids_rejected(characters_data, plan_data):
    characters_data["characters"].append(copy.deepcopy(characters_data["characters"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        parse_character_bible(characters_data)
    shot_data(plan_data, "shot_02")["id"] = "shot_01"
    with pytest.raises(ValueError, match="duplicate"):
        parse_shot_plan(plan_data)


def test_shot_ids_are_unique_across_scenes(plan_data):
    second = copy.deepcopy(plan_data["scenes"][0])
    second["id"] = "scene_02"
    plan_data["scenes"].append(second)
    with pytest.raises(ValueError, match="duplicate"):
        parse_shot_plan(plan_data)


def test_available_asset_needs_a_well_formed_hash(characters_data):
    asset = characters_data["characters"][0]["reference_assets"][0]
    for bad in (None, "3a3e", ROOFTOP_SHA.upper(), 12):
        asset["sha256"] = bad
        with pytest.raises(ValueError, match="64 lowercase hex"):
            parse_character_bible(characters_data)


def test_missing_asset_must_not_carry_a_hash(characters_data):
    characters_data["characters"][0]["reference_assets"][1]["sha256"] = OTHER_SHA
    with pytest.raises(ValueError, match="must be null"):
        parse_character_bible(characters_data)


def test_keyframe_is_available_with_hash_or_missing_without(plan_data):
    keyframe = shot_data(plan_data, "shot_02")["keyframes"][0]
    del keyframe["asset_sha256"]
    with pytest.raises(ValueError, match="asset_sha256"):
        parse_shot_plan(plan_data)
    keyframe["status"] = "missing"
    assert parse_shot_plan(plan_data).get_shot("shot_02").keyframes[0].asset_sha256 is None
    keyframe["asset_sha256"] = ROOFTOP_SHA
    with pytest.raises(ValueError, match="must be null"):
        parse_shot_plan(plan_data)
    keyframe["status"] = "pending"
    with pytest.raises(ValueError, match="status"):
        parse_shot_plan(plan_data)


def test_keyframe_role_checked(plan_data):
    shot_data(plan_data, "shot_02")["keyframes"][0]["role"] = "end_frame"
    with pytest.raises(ValueError, match="role"):
        parse_shot_plan(plan_data)


# --- validation: across files ---

def problems(plan_data, characters_data, story_data=None):
    return validate_story(
        parse_shot_plan(plan_data),
        parse_character_bible(characters_data),
        parse_story_bible(story_data) if story_data is not None else None,
    )


def test_scene_duration_must_equal_its_shots(plan_data, characters_data):
    shot_data(plan_data, "shot_01")["duration_seconds"] = 3
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "scene scene_01" in found[0] and "8 s" in found[0]


def test_total_duration_must_equal_the_scenes(plan_data, characters_data):
    plan_data["total_duration_seconds"] = 30
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "total_duration_seconds is 30" in found[0]


def test_shot_must_be_a_whole_number_of_frames(plan_data, characters_data):
    shot_data(plan_data, "shot_01")["duration_seconds"] = 4.01
    plan_data["scenes"][0]["duration_seconds"] = 9.01
    plan_data["total_duration_seconds"] = 9.01
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "whole number of frames" in found[0]


def test_fractional_seconds_that_are_whole_frames_pass(plan_data, characters_data):
    shot_data(plan_data, "shot_01")["duration_seconds"] = 3.5  # 84 frames
    shot_data(plan_data, "shot_02")["duration_seconds"] = 5.5
    assert problems(plan_data, characters_data) == []


def test_shot_character_must_exist_in_the_bible(plan_data, characters_data):
    shot_data(plan_data, "shot_01")["characters"] = ["VILLAIN_01"]
    found = problems(plan_data, characters_data)
    assert found == ["shot shot_01: character 'VILLAIN_01' is not in the character bible"]


def test_story_characters_must_exist_and_cover_the_shots(plan_data, characters_data, story_data):
    story_data["characters"] = ["COMIC_HERO_01", "GHOST_01"]
    assert any("GHOST_01" in p for p in problems(plan_data, characters_data, story_data))
    second = copy.deepcopy(characters_data["characters"][0])
    second.update(id="EXTRA_01", reference_assets=[])
    characters_data["characters"].append(second)
    story_data["characters"] = ["EXTRA_01"]
    found = problems(plan_data, characters_data, story_data)
    assert found == ["shot shot_02: character 'COMIC_HERO_01' is not listed in the story bible"]


def test_plan_must_belong_to_the_story(plan_data, characters_data, story_data):
    plan_data["story_id"] = "another_story"
    found = problems(plan_data, characters_data, story_data)
    assert len(found) == 1 and "another_story" in found[0]


def test_keyframe_hash_must_match_a_recorded_asset(plan_data, characters_data):
    shot_data(plan_data, "shot_02")["keyframes"][0]["asset_sha256"] = OTHER_SHA
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "matches no available asset" in found[0]


def test_keyframe_may_point_to_a_shared_plan_asset(plan_data, characters_data):
    plan_data["assets"][0].update(status="available", sha256=OTHER_SHA)
    shot_data(plan_data, "shot_01")["keyframes"][0].update(status="available", asset_sha256=OTHER_SHA)
    assert problems(plan_data, characters_data) == []
    assert missing_keyframes(parse_shot_plan(plan_data)) == []


def test_character_keyframe_must_use_that_characters_asset(plan_data, characters_data):
    plan_data["assets"][0].update(status="available", sha256=OTHER_SHA)
    shot_data(plan_data, "shot_02")["keyframes"][0]["asset_sha256"] = OTHER_SHA
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "not one of COMIC_HERO_01's reference assets" in found[0]


def test_keyframe_character_must_be_in_the_shot(plan_data, characters_data):
    shot_data(plan_data, "shot_02")["characters"] = []
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "not listed in the shot's characters" in found[0]


def test_one_start_frame_per_shot_and_no_asset_used_twice(plan_data, characters_data):
    keyframes = shot_data(plan_data, "shot_02")["keyframes"]
    keyframes.append({**keyframes[0], "id": "again"})
    found = problems(plan_data, characters_data)
    assert any("2 start_frame keyframes" in p for p in found)
    assert any("same asset as keyframe start" in p for p in found)


def test_one_hash_cannot_be_two_files(plan_data, characters_data):
    plan_data["assets"][0].update(status="available", sha256=ROOFTOP_SHA)
    found = problems(plan_data, characters_data)
    assert len(found) == 1 and "recorded for both" in found[0]


def test_check_story_raises_with_every_problem(plan_data, characters_data):
    plan_data["total_duration_seconds"] = 30
    shot_data(plan_data, "shot_01")["characters"] = ["VILLAIN_01"]
    with pytest.raises(StoryValidationError) as error:
        check_story(parse_shot_plan(plan_data), parse_character_bible(characters_data))
    assert len(error.value.problems) == 2


def test_verify_assets_compares_hashes_with_the_disk(tmp_path, characters_data, plan_data):
    content = b"not really a png"
    assets = characters_data["characters"][0]["reference_assets"]
    assets[0].update(path="refs/frame.png", sha256=hashlib.sha256(content).hexdigest())
    assets[1]["path"] = "refs/front.png"
    del assets[2:]
    plan_data["assets"] = []
    characters, plan = parse_character_bible(characters_data), parse_shot_plan(plan_data)

    found = verify_assets(characters, plan, root=tmp_path)
    assert found == ["asset rooftop_frame_v001: refs/frame.png is not on this machine"]

    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "frame.png").write_bytes(content)
    assert verify_assets(characters, plan, root=tmp_path) == []

    (tmp_path / "refs" / "frame.png").write_bytes(content + b"edited")
    (tmp_path / "refs" / "front.png").write_bytes(b"new")
    found = verify_assets(characters, plan, root=tmp_path)
    assert len(found) == 2
    assert "does not match the recorded sha256" in found[0]
    assert "marked missing, but refs/front.png exists" in found[1]


# --- shot -> VideoGenerationRequest ---

@pytest.fixture
def example():
    return (
        load_shot_plan(EXAMPLE_DIR / "shot_plan.yaml"),
        load_character_bible(EXAMPLE_DIR / "character_bible.yaml"),
        load_story_bible(EXAMPLE_DIR / "story_bible.yaml"),
    )


def test_draft_shot_is_refused(example):
    plan, characters, story = example
    with pytest.raises(DraftShotError):
        shot_to_request_dict(plan, "shot_02", characters, story, seed=1)


def test_shot_with_a_missing_keyframe_is_refused(example):
    plan, characters, story = example
    with pytest.raises(MissingKeyframeError, match="start"):
        shot_to_request_dict(ready(plan), "shot_01", characters, story, seed=1)


def test_unknown_shot_is_a_key_error(example):
    plan, characters, story = example
    with pytest.raises(KeyError):
        shot_to_request_dict(plan, "shot_99", characters, story, seed=1)


def test_ready_shot_becomes_a_valid_generation_request(example):
    plan, characters, story = example
    data = shot_to_request_dict(
        ready(plan), "shot_02", characters, story,
        seed=42, backend_parameters={"steps": 4}, duration_tolerance_seconds=0.1,
    )
    assert set(data) == {f.name for f in dataclasses.fields(VideoGenerationRequest)}
    assert json.loads(json.dumps(data)) == data  # plain data: can be written to a file
    assert data["start_frame"] == "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
    assert data["reference_images"] == []
    assert data["duration_seconds"] == 5 and data["fps"] == 24
    assert data["output_resolution"] == {"width": 1280, "height": 704}
    assert data["camera_motion"].startswith("slow cinematic push-in")
    assert data["subject_motion"][-1] == "near the end, a slight head turn toward the camera"
    assert "halftone print texture" in data["visual_direction"]
    assert data["backend_parameters"] == {"steps": 4}

    request = request_from_dict(data)
    assert isinstance(request, VideoGenerationRequest)
    assert request.seed == 42 and request.duration_tolerance_seconds == 0.1
    assert request.output_resolution == Resolution(width=1280, height=704)
    assert request.input_paths == [data["start_frame"]]


def test_composed_prompt_uses_both_bibles(example):
    plan, characters, story = example
    data = shot_to_request_dict(ready(plan), "shot_02", characters, story, seed=1)
    prompt = data["prompt"]
    assert prompt.startswith("Graphic novel style animated shot, hybrid 2D and 3D animation. Wide shot.")
    assert "A high rooftop over a dense city, late sunset" in prompt
    assert "curly dark hair" in prompt  # the character's description
    assert "turns his head slightly toward the camera." in prompt  # the action
    assert "Slow cinematic push-in from a low three-quarter angle" in prompt
    assert prompt.endswith("limited stepped animation timing.")
    assert prompt == compose_prompt(plan, plan.get_shot("shot_02"), characters, story)  # deterministic
    negative = data["negative_prompt"].split(", ")
    assert "photorealistic" in negative and "mask" in negative  # style + character "never"
    assert negative.count("logo") == 1  # listed by both, kept once


def test_shot_prompt_overrides_the_composed_one(plan_data, characters_data, story_data):
    shot = shot_data(plan_data, "shot_02")
    shot.update(prompt="A complete prompt.", negative_prompt="blurry", status="ready")
    data = shot_to_request_dict(
        parse_shot_plan(plan_data), "shot_02", parse_character_bible(characters_data),
        parse_story_bible(story_data), seed=1,
    )
    assert data["prompt"] == "A complete prompt." and data["negative_prompt"] == "blurry"


def test_conversion_without_a_story_bible(example):
    plan, characters, _ = example
    data = shot_to_request_dict(ready(plan), "shot_02", characters, seed=1)
    assert data["visual_direction"] is None
    assert data["negative_prompt"] == "mask, logo, emblem, text on the clothing, superhero costume"
    request_from_dict(data)


def test_preview_shows_a_draft_shot_without_its_missing_keyframes(example):
    plan, characters, story = example
    data = shot_to_request_dict(plan, "shot_01", characters, story, seed=1, preview=True)
    assert data["start_frame"] is None
    assert data["prompt"].startswith("Graphic novel style animated shot, hybrid 2D and 3D animation. Wide establishing shot.")
    request_from_dict(data)


def test_reference_keyframes_become_reference_images(plan_data, characters_data, story_data):
    characters_data["characters"][0]["reference_assets"][1].update(status="available", sha256=OTHER_SHA)
    shot = shot_data(plan_data, "shot_02")
    shot["status"] = "ready"
    shot["keyframes"].append({
        "id": "identity", "role": "reference", "description": "front view", "character": "COMIC_HERO_01",
        "status": "available", "asset_sha256": OTHER_SHA,
    })
    plan, characters = parse_shot_plan(plan_data), parse_character_bible(characters_data)
    assert validate_story(plan, characters) == []
    data = shot_to_request_dict(plan, "shot_02", characters, parse_story_bible(story_data), seed=1)
    assert data["reference_images"] == ["benchmarks/references/comic_hero_01/front.png"]
    assert request_from_dict(data).input_paths == [data["start_frame"], *data["reference_images"]]


def test_conversion_refuses_an_unrecorded_hash_and_unknown_characters(plan_data, characters_data):
    shot = shot_data(plan_data, "shot_02")
    shot["status"] = "ready"
    shot["keyframes"][0]["asset_sha256"] = OTHER_SHA
    characters = parse_character_bible(characters_data)
    with pytest.raises(ValueError, match="no recorded asset"):
        shot_to_request_dict(parse_shot_plan(plan_data), "shot_02", characters, seed=1)
    shot["characters"] = ["VILLAIN_01"]
    shot["keyframes"] = []
    with pytest.raises(ValueError, match="not in the character bible"):
        shot_to_request_dict(parse_shot_plan(plan_data), "shot_02", characters, seed=1)
