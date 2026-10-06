import pytest

from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.shot_prompt import CAMERA_PRESETS, fold_shot_controls
from ai_studio.media import Resolution


def request(**controls):
    return VideoGenerationRequest(
        prompt="A man sits on a bench. ", duration_seconds=5, output_resolution=Resolution(width=1280, height=704),
        seed=42, **controls,
    )


def test_no_controls_leave_the_prompt_as_it_is():
    assert fold_shot_controls(request()) == "A man sits on a bench."


def test_controls_are_appended_in_a_fixed_order():
    folded = fold_shot_controls(request(
        visual_direction=["ink contours", "cel shading"],
        subject_motion=["sits still", "turns his head."],
        camera_motion="slow push-in",
    ))
    assert folded == (
        "A man sits on a bench.\n"
        "Camera: slow push-in.\n"
        "Subject motion: sits still; turns his head.\n"
        "Visual direction: ink contours, cel shading."
    )


@pytest.mark.parametrize("name", ["static", " Static "])
def test_camera_preset_is_expanded(name):
    assert fold_shot_controls(request(camera_motion=name)).endswith(f"Camera: {CAMERA_PRESETS['static']}.")


def test_mapping_is_written_as_pairs():
    folded = fold_shot_controls(request(camera_motion={"move": "pan left", "speed": 0.5}))
    assert folded.endswith("Camera: move: pan left; speed: 0.5.")
