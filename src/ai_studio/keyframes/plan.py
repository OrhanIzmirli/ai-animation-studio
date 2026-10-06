"""Keyframe plans: trailers/<trailer_id>/keyframes.yaml.

Which still images a trailer needs (one start frame per generated shot, title
plates, ...) and, for each, the pose and scene to add to the style lock.
Planning data: loading it generates nothing. Format: the plan file's header.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.keyframes.lock import Prompt, StyleLock, _keys, _text, compose
from ai_studio.media import Resolution


@dataclass(frozen=True)
class Keyframe:
    id: str
    shot: str
    character: str | None
    scene: str
    pose: str = ""
    solo: bool = True
    character_reference: bool = False
    negative: str = ""
    seeds: tuple[int, ...] = ()
    # Where the character goes, as fractions of the frame (x0, y0, x1, y1): the
    # reference acts only there, so it does not repaint the scene. None: everywhere.
    character_region: tuple[float, float, float, float] | None = None

    def prompt(self, lock: StyleLock) -> Prompt:
        return compose(lock, character=self.character, scene=self.scene, pose=self.pose, solo=self.solo,
                       negative=self.negative)


@dataclass(frozen=True)
class KeyframePlan:
    trailer_id: str
    style_lock: str
    resolution: Resolution
    seeds: tuple[int, ...]
    keyframes: tuple[Keyframe, ...]

    def get(self, keyframe_id: str) -> Keyframe:
        for keyframe in self.keyframes:
            if keyframe.id == keyframe_id:
                return keyframe
        raise KeyError(f"no keyframe {keyframe_id!r} in {self.trailer_id}; have {[k.id for k in self.keyframes]}")


def _seeds(where: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in value
    ):
        raise ValueError(f"{where} must be a non-empty list of ints >= 0")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} has a seed twice")
    return tuple(value)


def _region(where: str, value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ValueError(f"{where} must be [x0, y0, x1, y1], fractions of the frame")
    x0, y0, x1, y1 = (float(v) for v in value)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"{where} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return (x0, y0, x1, y1)


def parse_plan(data: Any, lock: StyleLock, shot_ids: set[str] | None = None) -> KeyframePlan:
    """shot_ids: the shots of the trailer's shot list; every keyframe must name one."""
    data = _keys("plan", data, {"trailer_id", "style_lock", "resolution", "seeds", "keyframes"})
    if data["style_lock"] != lock.lock_id:
        raise ValueError(f"plan uses lock {data['style_lock']!r}, got {lock.lock_id!r}")
    resolution = _keys("resolution", data["resolution"], {"width", "height"})
    keyframes, seen = [], set()
    for i, item in enumerate(data["keyframes"] or []):
        where = f"keyframes[{i}]"
        item = _keys(where, item, {"id", "shot", "character", "scene"},
                     {"pose", "solo", "character_reference", "character_region", "negative", "seeds"})
        keyframe_id = _text(f"{where}.id", item["id"])
        check_file_name_safe(f"{where}.id", keyframe_id)
        if keyframe_id in seen:
            raise ValueError(f"{where}: duplicate id {keyframe_id!r}")
        seen.add(keyframe_id)
        if shot_ids is not None and item["shot"] not in shot_ids:
            raise ValueError(f"{where}: shot {item['shot']!r} is not in the shot list")
        for flag in ("solo", "character_reference"):
            if flag in item and not isinstance(item[flag], bool):
                raise ValueError(f"{where}.{flag} must be true or false")
        keyframe = Keyframe(
            id=keyframe_id, shot=item["shot"], character=item["character"],
            scene=_text(f"{where}.scene", item["scene"]), pose=_text(f"{where}.pose", item.get("pose"), empty=True),
            solo=item.get("solo", True), character_reference=item.get("character_reference", False),
            negative=_text(f"{where}.negative", item.get("negative"), empty=True),
            seeds=_seeds(f"{where}.seeds", item["seeds"]) if "seeds" in item else (),
            character_region=_region(f"{where}.character_region", item.get("character_region")),
        )
        if keyframe.character_region is not None and not keyframe.character_reference:
            raise ValueError(f"{where}: character_region needs character_reference")
        keyframe.prompt(lock)  # character known, pose only with a character
        if keyframe.character_reference:
            char = lock.characters[keyframe.character] if keyframe.character else None
            if char is None or char.reference_image is None or lock.ipadapter is None:
                raise ValueError(f"{where}: character_reference needs a character with a reference image "
                                 "and an ipadapter block in the lock")
        keyframes.append(keyframe)
    return KeyframePlan(
        trailer_id=_text("trailer_id", data["trailer_id"]), style_lock=lock.lock_id,
        resolution=Resolution(width=resolution["width"], height=resolution["height"]),
        seeds=_seeds("seeds", data["seeds"]), keyframes=tuple(keyframes),
    )


def load_plan(path: str | Path, lock: StyleLock, shot_ids: set[str] | None = None) -> KeyframePlan:
    with open(path, encoding="utf-8") as f:
        plan = parse_plan(yaml.safe_load(f), lock, shot_ids)
    if plan.trailer_id != Path(path).parent.name:
        raise ValueError(f"trailer_id {plan.trailer_id!r} must equal the folder name {Path(path).parent.name!r}")
    return plan
