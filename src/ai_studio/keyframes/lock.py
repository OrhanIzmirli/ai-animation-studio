"""Style locks: the frozen settings and prompt blocks of one approved look.

A lock (configs/style_locks/<lock_id>.yaml) fixes the checkpoint, LoRAs,
sampler and the prompt text that carries the style. A keyframe only adds a
pose and a scene; compose() puts the parts together in a fixed order:

    positive  trigger, [name, [solo], body], [pose], style, scene, tail
    negative  base, [solo_negative], [character negative], keyframe extras

Empty parts are left out; parts are joined with ", ". With the lock's
design_check this reproduces the approved image's prompt exactly.

Unknown keys are rejected, so a typo cannot silently drop a setting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelFile:
    file: str
    sha256: str
    strength: float = 1.0  # LoRAs only


@dataclass(frozen=True)
class Character:
    name: str
    solo: str
    solo_negative: str
    body: str
    negative: str = ""
    reference_image: str | None = None
    reference_sha256: str | None = None


@dataclass(frozen=True)
class IPAdapter:
    file: str
    sha256: str
    clip_vision: str
    clip_vision_sha256: str
    weight: float
    weight_type: str


@dataclass(frozen=True)
class DesignCheck:
    character: str | None
    solo: bool
    scene: str
    seed: int
    width: int
    height: int
    expected_image: str  # compared pixel by pixel
    expected_sha256: str  # of expected_image, so the comparison is against the right file
    pose: str = ""


@dataclass(frozen=True)
class StyleLock:
    lock_id: str
    checkpoint: ModelFile
    loras: tuple[ModelFile, ...]
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    trigger: str
    style: str
    tail: str
    negative: str
    characters: dict[str, Character]
    ipadapter: IPAdapter | None = None
    design_check: DesignCheck | None = None


@dataclass(frozen=True)
class Prompt:
    positive: str
    negative: str


def _keys(where: str, data: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    missing, unknown = required - set(data), set(data) - required - set(optional)
    if missing or unknown:
        raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    return data


def _text(where: str, value: Any, *, empty: bool = False) -> str:
    if value is None and empty:
        return ""
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{where} must be a non-empty string")
    return " ".join(value.split())  # folded YAML: one line, single spaces


def _sha(where: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256.match(value):
        raise ValueError(f"{where} must be a lower-case SHA-256 hex digest")
    return value


def _number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    return float(value)


def parse_lock(data: Any) -> StyleLock:
    data = _keys("lock", data, {
        "lock_id", "checkpoint", "loras", "sampler", "trigger", "style", "tail", "negative", "characters",
    }, {"ipadapter", "design_check"})
    checkpoint = _keys("checkpoint", data["checkpoint"], {"file", "sha256"})
    loras = []
    for i, lora in enumerate(data["loras"] or []):
        lora = _keys(f"loras[{i}]", lora, {"file", "sha256", "strength"}, {"source"})
        loras.append(ModelFile(_text(f"loras[{i}].file", lora["file"]), _sha(f"loras[{i}].sha256", lora["sha256"]),
                               _number(f"loras[{i}].strength", lora["strength"])))
    sampler = _keys("sampler", data["sampler"], {"steps", "cfg", "sampler", "scheduler"})
    if isinstance(sampler["steps"], bool) or not isinstance(sampler["steps"], int) or sampler["steps"] <= 0:
        raise ValueError("sampler.steps must be a positive int")
    characters = {}
    for key, char in (data["characters"] or {}).items():
        char = _keys(f"characters.{key}", char, {"name", "solo", "solo_negative", "body"},
                     {"negative", "reference_image", "reference_sha256"})
        if (char.get("reference_image") is None) != (char.get("reference_sha256") is None):
            raise ValueError(f"characters.{key}: reference_image and reference_sha256 go together")
        characters[key] = Character(
            name=_text(f"characters.{key}.name", char["name"]),
            solo=_text(f"characters.{key}.solo", char["solo"]),
            solo_negative=_text(f"characters.{key}.solo_negative", char["solo_negative"]),
            body=_text(f"characters.{key}.body", char["body"]),
            negative=_text(f"characters.{key}.negative", char.get("negative"), empty=True),
            reference_image=char.get("reference_image"),
            reference_sha256=None if char.get("reference_sha256") is None
            else _sha(f"characters.{key}.reference_sha256", char["reference_sha256"]),
        )
    ipadapter = None
    if data.get("ipadapter") is not None:
        ipa = _keys("ipadapter", data["ipadapter"],
                    {"file", "sha256", "clip_vision", "clip_vision_sha256", "weight", "weight_type"})
        ipadapter = IPAdapter(ipa["file"], _sha("ipadapter.sha256", ipa["sha256"]), ipa["clip_vision"],
                              _sha("ipadapter.clip_vision_sha256", ipa["clip_vision_sha256"]),
                              _number("ipadapter.weight", ipa["weight"]), _text("ipadapter.weight_type", ipa["weight_type"]))
    check = None
    if data.get("design_check") is not None:
        c = _keys("design_check", data["design_check"],
                  {"character", "solo", "scene", "seed", "width", "height", "expected_image", "expected_sha256"}, {"pose"})
        if c["character"] is not None and c["character"] not in characters:
            raise ValueError(f"design_check.character {c['character']!r} is not a character of this lock")
        check = DesignCheck(c["character"], bool(c["solo"]), _text("design_check.scene", c["scene"]), int(c["seed"]),
                            int(c["width"]), int(c["height"]), _text("design_check.expected_image", c["expected_image"]),
                            _sha("design_check.expected_sha256", c["expected_sha256"]),
                            _text("design_check.pose", c.get("pose"), empty=True))
    return StyleLock(
        lock_id=_text("lock_id", data["lock_id"]),
        checkpoint=ModelFile(_text("checkpoint.file", checkpoint["file"]), _sha("checkpoint.sha256", checkpoint["sha256"])),
        loras=tuple(loras),
        steps=sampler["steps"], cfg=_number("sampler.cfg", sampler["cfg"]),
        sampler=_text("sampler.sampler", sampler["sampler"]), scheduler=_text("sampler.scheduler", sampler["scheduler"]),
        trigger=_text("trigger", data["trigger"]), style=_text("style", data["style"]),
        tail=_text("tail", data["tail"], empty=True), negative=_text("negative", data["negative"]),
        characters=characters, ipadapter=ipadapter, design_check=check,
    )


def load_lock(path: str | Path) -> StyleLock:
    with open(path, encoding="utf-8") as f:
        lock = parse_lock(yaml.safe_load(f))
    if lock.lock_id != Path(path).stem:
        raise ValueError(f"lock_id {lock.lock_id!r} must equal the file name {Path(path).stem!r}")
    return lock


def compose(
    lock: StyleLock, *, character: str | None, scene: str, pose: str = "", solo: bool = True, negative: str = "",
) -> Prompt:
    """The full prompt of one keyframe under this lock."""
    char = None
    if character is not None:
        if character not in lock.characters:
            raise ValueError(f"character {character!r} is not in lock {lock.lock_id!r}")
        char = lock.characters[character]
    elif pose:
        raise ValueError("a pose needs a character")
    positive = [lock.trigger]
    negatives = [lock.negative]
    if char is not None:
        positive += [char.name, char.solo if solo else "", char.body]
        negatives += [char.solo_negative if solo else "", char.negative]
    positive += [" ".join(pose.split()), lock.style, " ".join(scene.split()), lock.tail]
    negatives.append(" ".join(negative.split()))
    return Prompt(", ".join(p for p in positive if p), ", ".join(n for n in negatives if n))
