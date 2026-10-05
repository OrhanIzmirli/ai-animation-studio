"""Turn one planned shot into the input of a video backend.

shot_to_request_dict() returns a plain, JSON/YAML-serializable dict whose keys
are exactly the fields of ai_studio.generation.request.VideoGenerationRequest.
The only difference from the dataclass is output_resolution, which is a
{"width", "height"} mapping so that the dict can be written to a file;
request_from_dict() turns the dict into the real, validated request.

Nothing here runs a model. Per-run choices (seed, backend parameters, the
duration tolerance) are arguments, exactly as for benchmark definitions.

What maps to what:
  shot.prompt, or a prompt composed from the bibles   -> prompt
  shot.negative_prompt, or style + character "never"  -> negative_prompt
  keyframe with role start_frame (asset path)         -> start_frame
  keyframes with role reference (asset paths)         -> reference_images
  shot.camera                                         -> camera_motion
  shot.motion                                         -> subject_motion
  story.visual_style.direction                        -> visual_direction
  shot.duration_seconds, plan.fps, plan.resolution    -> duration_seconds, fps,
                                                         output_resolution

A shot is refused while it is a draft (DraftShotError) or while one of its
keyframes is still marked missing (MissingKeyframeError): generating without
the planned start frame would silently produce a different shot. preview=True
lifts both refusals, for reading the composed prompt before the shot is ready;
a preview dict leaves out the keyframes that do not exist.
"""

from __future__ import annotations

from typing import Any

from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.media import Resolution
from ai_studio.story.schema import CharacterBible, Shot, ShotPlan, StoryBible
from ai_studio.story.validation import asset_index

SHOT_TYPE_PHRASES = {
    "establishing": "Wide establishing shot",
    "wide": "Wide shot",
    "full": "Full shot",
    "medium": "Medium shot",
    "medium_close_up": "Medium close-up",
    "close_up": "Close-up",
    "extreme_close_up": "Extreme close-up",
    "over_the_shoulder": "Over-the-shoulder shot",
    "point_of_view": "Point-of-view shot",
    "two_shot": "Two shot",
    "insert": "Insert shot",
}


class DraftShotError(RuntimeError):
    """The shot is still a draft and must not be generated."""


class MissingKeyframeError(RuntimeError):
    """The shot needs a keyframe image that has not been made yet."""


def _sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if text.endswith((".", "!", "?")) else text + "."


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def compose_prompt(
    plan: ShotPlan, shot: Shot, characters: CharacterBible, story: StoryBible | None = None
) -> str:
    """The prompt for a shot that has none of its own. Deterministic: the same
    files always give the same text. Order: look, framing, place, who, what
    happens, camera, style keywords."""
    scene = plan.scene_of(shot.id)
    parts = []
    if story is not None:
        parts.append(_sentence(story.visual_style.summary))
    parts.append(_sentence(SHOT_TYPE_PHRASES[shot.shot_type]))
    place = ", ".join(text for text in (scene.location, scene.time_of_day) if text)
    if place:
        parts.append(_sentence(place[0].upper() + place[1:]))
    parts.extend(_sentence(characters.get(character_id).description) for character_id in shot.characters)
    parts.append(_sentence(shot.action))
    if isinstance(shot.camera, str):
        parts.append(_sentence(shot.camera[0].upper() + shot.camera[1:]))
    elif isinstance(shot.camera, list):
        parts.append(_sentence("Camera: " + ", ".join(shot.camera)))
    if story is not None and story.visual_style.direction:
        direction = ", ".join(story.visual_style.direction)
        parts.append(_sentence(direction[0].upper() + direction[1:]))
    return " ".join(parts)


def compose_negative_prompt(
    shot: Shot, characters: CharacterBible, story: StoryBible | None = None
) -> str | None:
    terms = list(story.visual_style.negative) if story is not None else []
    for character_id in shot.characters:
        terms.extend(characters.get(character_id).never)
    return ", ".join(_unique(terms)) or None


def shot_to_request_dict(
    plan: ShotPlan,
    shot_id: str,
    characters: CharacterBible,
    story: StoryBible | None = None,
    *,
    seed: int,
    backend_parameters: dict[str, Any] | None = None,
    duration_tolerance_seconds: float | None = None,
    preview: bool = False,
) -> dict[str, Any]:
    shot = plan.get_shot(shot_id)
    if shot.status != "ready" and not preview:
        raise DraftShotError(
            f"shot {shot.id!r} is a draft; set 'status: ready' in the shot plan once it is "
            "final (an explicit human decision)"
        )
    unknown = [character_id for character_id in shot.characters if character_id not in characters.ids]
    if unknown:
        raise ValueError(f"shot {shot.id!r}: characters {unknown} are not in the character bible")

    assets = asset_index(characters, plan)
    start_frame: str | None = None
    reference_images: list[str] = []
    for keyframe in shot.keyframes:
        if keyframe.asset_sha256 is None:
            if preview:
                continue
            raise MissingKeyframeError(
                f"shot {shot.id!r}: keyframe {keyframe.id!r} ({keyframe.description}) is marked missing"
            )
        asset = assets.get(keyframe.asset_sha256)
        if asset is None:
            raise ValueError(
                f"shot {shot.id!r}: keyframe {keyframe.id!r} points to sha256 "
                f"{keyframe.asset_sha256[:12]}…, which is no recorded asset"
            )
        if keyframe.role == "start_frame":
            if start_frame is not None:
                raise ValueError(f"shot {shot.id!r} has more than one start_frame keyframe")
            start_frame = asset.path
        else:
            reference_images.append(asset.path)

    direction = list(story.visual_style.direction) if story is not None else []
    return {
        "prompt": shot.prompt or compose_prompt(plan, shot, characters, story),
        "negative_prompt": shot.negative_prompt or compose_negative_prompt(shot, characters, story),
        "duration_seconds": shot.duration_seconds,
        "output_resolution": {"width": plan.resolution.width, "height": plan.resolution.height},
        "seed": seed,
        "fps": plan.fps,
        "duration_tolerance_seconds": duration_tolerance_seconds,
        "start_frame": start_frame,
        "reference_images": reference_images,
        "camera_motion": shot.camera,
        "subject_motion": list(shot.motion),
        "visual_direction": direction or None,
        "backend_parameters": dict(backend_parameters or {}),
    }


def request_from_dict(data: dict[str, Any]) -> VideoGenerationRequest:
    """The validated request for a dict made by shot_to_request_dict()."""
    return VideoGenerationRequest(
        **{**data, "output_resolution": Resolution(**data["output_resolution"])}
    )
