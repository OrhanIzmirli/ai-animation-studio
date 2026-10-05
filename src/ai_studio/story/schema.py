"""Story / director layer: typed planning data.

Three YAML files describe a production before anything is generated:

  story bible      what the story is and how it looks        (StoryBible)
  character bible  who appears and which reference images
                   define them, linked by SHA-256            (CharacterBible)
  shot plan        scenes -> shots: shot type, camera,
                   motion, duration, characters, keyframe
                   requirements, continuity notes            (ShotPlan)

This module only holds the value types. Reading YAML is in
ai_studio.story.loading, cross-file checks in ai_studio.story.validation and
the shot -> VideoGenerationRequest conversion in ai_studio.story.convert.
Nothing in this package generates media or calls a model.

Lifecycle follows the benchmark definitions: everything has a status, "draft"
(the default) or "ready", and draft -> ready is an explicit human edit.

Reference assets are identified by the SHA-256 of the file, never by path
alone: a path can be re-used for a different image, a hash cannot. An asset
that has been planned but not made yet is recorded with `status: missing` and
no hash, so that the plan says openly what does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_studio.media import Resolution

STATUSES = ("draft", "ready")

ASSET_STATUSES = ("available", "missing")
ASSET_KINDS = (
    "start_frame",    # a finished frame 0 of some shot
    "front", "three_quarter", "profile", "back",  # character sheet views
    "face_closeup",
    "expression",
    "style_frame",    # a finished frame in the target look
    "environment",    # a location plate without characters
    "prop",
    "other",
)

SHOT_TYPES = (
    "establishing", "wide", "full", "medium", "medium_close_up", "close_up",
    "extreme_close_up", "over_the_shoulder", "point_of_view", "two_shot", "insert",
)

# How a keyframe is used by the video model (see generation/request.py:
# start_frame and reference_images are not interchangeable).
KEYFRAME_ROLES = ("start_frame", "reference")

CHARACTER_ROLES = ("protagonist", "antagonist", "supporting", "background")

# string, list of strings or JSON-like mapping, as generation.request.ShotControl
Control = str | list[str] | dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class ReferenceAsset:
    id: str  # unique within its owner (a character, or a shot plan's shared assets)
    kind: str  # one of ASSET_KINDS
    path: str  # relative to the project root; media files themselves are git-ignored
    status: str  # "available" (sha256 set) | "missing" (sha256 None)
    sha256: str | None = None
    description: str = ""


@dataclass(frozen=True, kw_only=True)
class Character:
    id: str
    name: str
    role: str  # one of CHARACTER_ROLES
    description: str  # one prompt-ready sentence; used when a shot has no prompt of its own
    appearance: tuple[str, ...] = ()  # fixed traits (face, hair, build)
    wardrobe: tuple[str, ...] = ()
    never: tuple[str, ...] = ()  # things this character must never show (go to the negative prompt)
    reference_assets: tuple[ReferenceAsset, ...] = ()
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class CharacterBible:
    characters: tuple[Character, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(character.id for character in self.characters)

    def get(self, character_id: str) -> Character:
        for character in self.characters:
            if character.id == character_id:
                return character
        raise KeyError(f"no character {character_id!r} in the character bible")


@dataclass(frozen=True, kw_only=True)
class VisualStyle:
    summary: str  # one prompt-ready sentence
    direction: tuple[str, ...] = ()  # becomes the request's visual_direction
    negative: tuple[str, ...] = ()  # becomes part of the negative prompt
    artist_profile: str | None = None  # configs/style_recipes/<name>.yaml, applied in post


@dataclass(frozen=True, kw_only=True)
class StoryBible:
    story_id: str
    title: str
    logline: str
    visual_style: VisualStyle
    status: str = "draft"
    synopsis: str = ""
    themes: tuple[str, ...] = ()
    tone: str = ""
    setting: str = ""
    world_rules: tuple[str, ...] = ()
    characters: tuple[str, ...] = ()  # character ids; each must exist in the character bible
    originality_notes: str = ""
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class KeyframeRequirement:
    """An image a shot needs before it can be generated.

    Either it points to an existing asset by hash (status "available",
    asset_sha256 set) or it is openly marked as not made yet (status
    "missing", asset_sha256 None)."""

    id: str  # unique within the shot
    role: str  # one of KEYFRAME_ROLES
    description: str  # what the image must show
    status: str  # one of ASSET_STATUSES
    asset_sha256: str | None = None
    character: str | None = None  # the character it shows, if any


@dataclass(frozen=True, kw_only=True)
class Shot:
    id: str
    name: str
    shot_type: str  # one of SHOT_TYPES
    camera: Control  # becomes the request's camera_motion
    motion: tuple[str, ...]  # becomes the request's subject_motion
    duration_seconds: float
    action: str  # what happens, in prompt-ready prose
    status: str = "draft"
    characters: tuple[str, ...] = ()
    keyframes: tuple[KeyframeRequirement, ...] = ()
    continuity: tuple[str, ...] = ()  # what must match the neighbouring shots
    prompt: str | None = None  # full prompt; None: composed from the bibles
    negative_prompt: str | None = None  # None: composed from the bibles
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class Scene:
    id: str
    name: str
    duration_seconds: float  # must equal the sum of its shots
    shots: tuple[Shot, ...]
    location: str = ""
    time_of_day: str = ""
    summary: str = ""


@dataclass(frozen=True, kw_only=True)
class ShotPlan:
    plan_id: str
    story_id: str
    fps: int
    resolution: Resolution
    total_duration_seconds: float  # must equal the sum of the scenes
    scenes: tuple[Scene, ...]
    status: str = "draft"
    assets: tuple[ReferenceAsset, ...] = ()  # shared, non-character assets (environments, props)
    notes: str = ""

    @property
    def shots(self) -> tuple[Shot, ...]:
        """Every shot, in playing order."""
        return tuple(shot for scene in self.scenes for shot in scene.shots)

    def get_shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError(f"no shot {shot_id!r} in shot plan {self.plan_id!r}")

    def scene_of(self, shot_id: str) -> Scene:
        for scene in self.scenes:
            if any(shot.id == shot_id for shot in scene.shots):
                return scene
        raise KeyError(f"no shot {shot_id!r} in shot plan {self.plan_id!r}")
