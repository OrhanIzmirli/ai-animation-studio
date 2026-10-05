"""Read story bibles, character bibles and shot plans from YAML.

Loading checks the SHAPE of one file: required keys, types, allowed values,
well-formed hashes. Unknown keys are rejected, so a typo cannot silently drop
a field. Checks that span files or add things up (durations, characters that
must exist, keyframes that must point to an asset) are in
ai_studio.story.validation and report every problem at once.

Paths given to the load_* functions may be absolute or relative to the
project root (ai_studio.paths). The file formats are documented with examples
in docs/STORY_LAYER.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.media import Resolution
from ai_studio.paths import resolve_path
from ai_studio.story.schema import (
    ASSET_KINDS,
    ASSET_STATUSES,
    CHARACTER_ROLES,
    KEYFRAME_ROLES,
    SHOT_TYPES,
    STATUSES,
    Character,
    CharacterBible,
    Control,
    KeyframeRequirement,
    ReferenceAsset,
    Scene,
    Shot,
    ShotPlan,
    StoryBible,
    VisualStyle,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _check_keys(where: str, data: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown, missing = set(data) - required - optional, required - set(data)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(map(str, unknown))}")
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")


def _text(where: str, value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string" + (" or null" if optional else ""))
    return value.strip()


def _free_text(where: str, value: Any) -> str:
    """Optional prose (notes, synopsis): null or absent means empty."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value.strip()


def _strings(where: str, value: Any) -> tuple[str, ...]:
    """A list of non-empty strings; null or absent means empty."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{where} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _identifier(where: str, value: Any) -> str:
    text = _text(where, value)
    check_file_name_safe(where, text)
    return text


def _identifiers(where: str, value: Any) -> tuple[str, ...]:
    ids = _strings(where, value)
    for item in ids:
        check_file_name_safe(where, item)
    _no_duplicates(where, ids)
    return ids


def _choice(where: str, value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{where} must be one of {allowed}, got {value!r}")
    return value


def _number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} must be a positive number")
    return float(value)


def _no_duplicates(where: str, ids: list[str] | tuple[str, ...]) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"{where}: duplicate ids {duplicates}")


def _hash_and_status(where: str, status: Any, sha256: Any, key: str) -> tuple[str, str | None]:
    """An available item carries a SHA-256; a missing one must not."""
    _choice(f"{where}: status", status, ASSET_STATUSES)
    if status == "missing":
        if sha256 is not None:
            raise ValueError(f"{where}: status is missing, so {key} must be null")
        return status, None
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError(f"{where}: status is available, so {key} must be 64 lowercase hex characters")
    return status, sha256


def _control(where: str, value: Any) -> Control:
    """Same shapes as generation.request.ShotControl, but required."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    if isinstance(value, dict) and value:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where} must be JSON-serializable: {e}") from None
        return value
    raise ValueError(f"{where} must be a non-empty string, list of strings or mapping")


def _parse_asset(where: str, data: Any) -> ReferenceAsset:
    _check_keys(where, data, {"id", "kind", "path", "status"}, {"sha256", "description"})
    asset_id = _identifier(f"{where}.id", data["id"])
    where = f"{where} {asset_id}"
    status, sha256 = _hash_and_status(where, data["status"], data.get("sha256"), "sha256")
    return ReferenceAsset(
        id=asset_id,
        kind=_choice(f"{where}: kind", data["kind"], ASSET_KINDS),
        path=_text(f"{where}: path", data["path"]),
        status=status,
        sha256=sha256,
        description=_free_text(f"{where}: description", data.get("description")),
    )


def _parse_assets(where: str, value: Any) -> tuple[ReferenceAsset, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    assets = tuple(_parse_asset(f"{where}[{index}]", item) for index, item in enumerate(value))
    _no_duplicates(where, [asset.id for asset in assets])
    return assets


# --- character bible ---

def _parse_character(index: int, data: Any) -> Character:
    where = f"characters[{index}]"
    _check_keys(
        where, data, {"id", "name", "role", "description"},
        {"appearance", "wardrobe", "never", "reference_assets", "notes"},
    )
    character_id = _identifier(f"{where}.id", data["id"])
    where = f"character {character_id}"
    return Character(
        id=character_id,
        name=_text(f"{where}: name", data["name"]),
        role=_choice(f"{where}: role", data["role"], CHARACTER_ROLES),
        description=_text(f"{where}: description", data["description"]),
        appearance=_strings(f"{where}: appearance", data.get("appearance")),
        wardrobe=_strings(f"{where}: wardrobe", data.get("wardrobe")),
        never=_strings(f"{where}: never", data.get("never")),
        reference_assets=_parse_assets(f"{where}: reference_assets", data.get("reference_assets")),
        notes=_free_text(f"{where}: notes", data.get("notes")),
    )


def parse_character_bible(data: Any) -> CharacterBible:
    _check_keys("character bible", data, {"characters"})
    if not isinstance(data["characters"], list) or not data["characters"]:
        raise ValueError("characters must be a non-empty list")
    characters = tuple(_parse_character(index, item) for index, item in enumerate(data["characters"]))
    _no_duplicates("characters", [character.id for character in characters])
    return CharacterBible(characters=characters)


# --- story bible ---

def parse_story_bible(data: Any) -> StoryBible:
    _check_keys(
        "story bible", data, {"story_id", "title", "logline", "visual_style"},
        {"status", "synopsis", "themes", "tone", "setting", "world_rules", "characters",
         "originality_notes", "notes"},
    )
    style = data["visual_style"]
    _check_keys("visual_style", style, {"summary"}, {"direction", "negative", "artist_profile"})
    profile = _text("visual_style.artist_profile", style.get("artist_profile"), optional=True)
    if profile is not None:
        check_file_name_safe("visual_style.artist_profile", profile)
    return StoryBible(
        story_id=_identifier("story_id", data["story_id"]),
        title=_text("title", data["title"]),
        logline=_text("logline", data["logline"]),
        visual_style=VisualStyle(
            summary=_text("visual_style.summary", style["summary"]),
            direction=_strings("visual_style.direction", style.get("direction")),
            negative=_strings("visual_style.negative", style.get("negative")),
            artist_profile=profile,
        ),
        status=_choice("status", data.get("status", "draft"), STATUSES),
        synopsis=_free_text("synopsis", data.get("synopsis")),
        themes=_strings("themes", data.get("themes")),
        tone=_free_text("tone", data.get("tone")),
        setting=_free_text("setting", data.get("setting")),
        world_rules=_strings("world_rules", data.get("world_rules")),
        characters=_identifiers("characters", data.get("characters")),
        originality_notes=_free_text("originality_notes", data.get("originality_notes")),
        notes=_free_text("notes", data.get("notes")),
    )


# --- shot plan ---

def _parse_keyframe(where: str, data: Any) -> KeyframeRequirement:
    _check_keys(where, data, {"id", "role", "description", "status"}, {"asset_sha256", "character"})
    keyframe_id = _identifier(f"{where}.id", data["id"])
    where = f"{where} {keyframe_id}"
    status, sha256 = _hash_and_status(where, data["status"], data.get("asset_sha256"), "asset_sha256")
    character = _text(f"{where}: character", data.get("character"), optional=True)
    return KeyframeRequirement(
        id=keyframe_id,
        role=_choice(f"{where}: role", data["role"], KEYFRAME_ROLES),
        description=_text(f"{where}: description", data["description"]),
        status=status,
        asset_sha256=sha256,
        character=character,
    )


def _parse_shot(where: str, data: Any) -> Shot:
    _check_keys(
        where, data, {"id", "name", "shot_type", "camera", "motion", "duration_seconds", "action"},
        {"status", "characters", "keyframes", "continuity", "prompt", "negative_prompt", "notes"},
    )
    shot_id = _identifier(f"{where}.id", data["id"])
    where = f"shot {shot_id}"
    motion = data["motion"]
    if isinstance(motion, str):
        motion = [motion]
    motion = _strings(f"{where}: motion", motion)
    if not motion:
        raise ValueError(f"{where}: motion must not be empty (say 'static' if nothing moves)")
    keyframes = data.get("keyframes")
    if keyframes is None:
        keyframes = []
    if not isinstance(keyframes, list):
        raise ValueError(f"{where}: keyframes must be a list")
    keyframes = tuple(
        _parse_keyframe(f"{where}: keyframes[{index}]", item) for index, item in enumerate(keyframes)
    )
    _no_duplicates(f"{where}: keyframes", [keyframe.id for keyframe in keyframes])
    return Shot(
        id=shot_id,
        name=_text(f"{where}: name", data["name"]),
        shot_type=_choice(f"{where}: shot_type", data["shot_type"], SHOT_TYPES),
        camera=_control(f"{where}: camera", data["camera"]),
        motion=motion,
        duration_seconds=_number(f"{where}: duration_seconds", data["duration_seconds"]),
        action=_text(f"{where}: action", data["action"]),
        status=_choice(f"{where}: status", data.get("status", "draft"), STATUSES),
        characters=_identifiers(f"{where}: characters", data.get("characters")),
        keyframes=keyframes,
        continuity=_strings(f"{where}: continuity", data.get("continuity")),
        prompt=_text(f"{where}: prompt", data.get("prompt"), optional=True),
        negative_prompt=_text(f"{where}: negative_prompt", data.get("negative_prompt"), optional=True),
        notes=_free_text(f"{where}: notes", data.get("notes")),
    )


def _parse_scene(index: int, data: Any) -> Scene:
    where = f"scenes[{index}]"
    _check_keys(
        where, data, {"id", "name", "duration_seconds", "shots"}, {"location", "time_of_day", "summary"},
    )
    scene_id = _identifier(f"{where}.id", data["id"])
    where = f"scene {scene_id}"
    if not isinstance(data["shots"], list) or not data["shots"]:
        raise ValueError(f"{where}: shots must be a non-empty list")
    return Scene(
        id=scene_id,
        name=_text(f"{where}: name", data["name"]),
        duration_seconds=_number(f"{where}: duration_seconds", data["duration_seconds"]),
        shots=tuple(
            _parse_shot(f"{where}: shots[{shot_index}]", shot) for shot_index, shot in enumerate(data["shots"])
        ),
        location=_free_text(f"{where}: location", data.get("location")),
        time_of_day=_free_text(f"{where}: time_of_day", data.get("time_of_day")),
        summary=_free_text(f"{where}: summary", data.get("summary")),
    )


def parse_shot_plan(data: Any) -> ShotPlan:
    _check_keys(
        "shot plan", data, {"plan_id", "story_id", "fps", "resolution", "total_duration_seconds", "scenes"},
        {"status", "assets", "notes"},
    )
    fps = data["fps"]
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive whole number")
    _check_keys("resolution", data["resolution"], {"width", "height"})
    if not isinstance(data["scenes"], list) or not data["scenes"]:
        raise ValueError("scenes must be a non-empty list")
    scenes = tuple(_parse_scene(index, scene) for index, scene in enumerate(data["scenes"]))
    _no_duplicates("scenes", [scene.id for scene in scenes])
    _no_duplicates("shots", [shot.id for scene in scenes for shot in scene.shots])
    return ShotPlan(
        plan_id=_identifier("plan_id", data["plan_id"]),
        story_id=_identifier("story_id", data["story_id"]),
        fps=fps,
        resolution=Resolution(**data["resolution"]),
        total_duration_seconds=_number("total_duration_seconds", data["total_duration_seconds"]),
        scenes=scenes,
        status=_choice("status", data.get("status", "draft"), STATUSES),
        assets=_parse_assets("assets", data.get("assets")),
        notes=_free_text("notes", data.get("notes")),
    )


# --- files ---

def _read(path: str | Path) -> tuple[Path, Any]:
    path = resolve_path(path)
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def _load(path: str | Path, parse: Any) -> Any:
    path, data = _read(path)
    try:
        return parse(data)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{path}: {e}") from None


def load_story_bible(path: str | Path) -> StoryBible:
    return _load(path, parse_story_bible)


def load_character_bible(path: str | Path) -> CharacterBible:
    return _load(path, parse_character_bible)


def load_shot_plan(path: str | Path) -> ShotPlan:
    return _load(path, parse_shot_plan)
