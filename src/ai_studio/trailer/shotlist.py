"""Trailer shot lists.

A shot list is a YAML file, trailers/<trailer_id>/shotlist.yaml. It is
planning data: it describes the shots of a trailer and which clip was chosen
for each. Loading it never generates anything.

    trailer_id: trailer_01            # = the folder name
    working_title: ...
    status: draft                     # draft | ready
    fps: 24
    resolution: {width: 1280, height: 704}
    total_duration_seconds: 30        # must equal the sum of the shots
    artist_profile: comic_finish_v1   # ONE profile for every shot
    character: ...                    # optional, free text
    shots:
      - id: shot_01                   # unique; shots play in file order
        name: ...
        source: generated             # generated (by a video model) | title_card (made in editing)
        duration_seconds: 4           # a whole number of frames at fps
        needs_character_keyframe: false
        status: draft                 # draft | ready
        camera: ...
        motion: [...]
        prompt: ...                   # generated shots: required; title cards: null
        negative_prompt: ...
        benchmark: ...                # optional: the benchmark this shot is
        clip: null                    # null = not chosen yet (placeholder card), or
                                      # {run: <benchmark_id>/<run_id>} a recorded run's video, or
                                      # {path: <video file>}
        fade_in_seconds: 0            # optional; 0 = cut. Fades are from / to black.
        fade_out_seconds: 0
        title_card: {text, subtitle, image}   # title_card shots only
        notes: ...                    # optional

Unknown keys are rejected, so a typo cannot silently drop a field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.media import Resolution

STATUSES = ("draft", "ready")
SOURCES = ("generated", "title_card")
_TRAILER_KEYS = {
    "trailer_id", "working_title", "status", "fps", "resolution", "total_duration_seconds",
    "artist_profile", "character", "shots",
}
_SHOT_REQUIRED = {
    "id", "name", "source", "duration_seconds", "needs_character_keyframe", "status",
    "camera", "motion", "prompt", "negative_prompt",
}
_SHOT_OPTIONAL = {"benchmark", "clip", "fade_in_seconds", "fade_out_seconds", "title_card", "notes"}
_TITLE_CARD_KEYS = {"text", "subtitle", "image"}


@dataclass(frozen=True)
class TitleCard:
    text: str
    subtitle: str | None = None
    image: str | None = None  # a finished card; replaces the rendered text


@dataclass(frozen=True)
class Shot:
    id: str
    name: str
    source: str
    duration_seconds: float
    frame_count: int
    needs_character_keyframe: bool
    status: str
    camera: str
    motion: tuple[str, ...]
    prompt: str | None
    negative_prompt: str | None
    benchmark: str | None = None
    clip: dict[str, str] | None = None  # {"run": "b/r"} or {"path": "..."}; None = placeholder
    fade_in_frames: int = 0
    fade_out_frames: int = 0
    title_card: TitleCard | None = None
    notes: str = ""


@dataclass(frozen=True)
class Shotlist:
    trailer_id: str
    working_title: str
    status: str
    fps: int
    resolution: Resolution
    artist_profile: str
    shots: tuple[Shot, ...]
    character: str | None = None

    @property
    def frame_count(self) -> int:
        return sum(shot.frame_count for shot in self.shots)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


def _text(where: str, value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string" + (" or null" if optional else ""))
    return value.strip()


def _number(where: str, value: Any, *, minimum: float, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    if value < minimum or (exclusive and value == minimum):
        raise ValueError(f"{where} must be {'>' if exclusive else '>='} {minimum:g}")
    return float(value)


def _frames(where: str, seconds: float, fps: int) -> int:
    frames = seconds * fps
    if abs(frames - round(frames)) > 1e-6:
        raise ValueError(f"{where}: {seconds:g} s is not a whole number of frames at {fps} fps")
    return int(round(frames))


def _check_keys(where: str, data: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown, missing = set(data) - required - optional, required - set(data)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")


def _parse_clip(where: str, value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) != 1 or next(iter(value)) not in ("run", "path"):
        raise ValueError(f"{where} must be null, {{run: <benchmark_id>/<run_id>}} or {{path: <video file>}}")
    kind, target = next(iter(value.items()))
    target = _text(f"{where}.{kind}", target)
    if kind == "run":
        parts = target.split("/")
        if len(parts) != 2:
            raise ValueError(f"{where}.run must be <benchmark_id>/<run_id>, got {target!r}")
        for label, part in zip(("benchmark_id", "run_id"), parts):
            check_file_name_safe(f"{where}.run {label}", part)
    return {kind: target}


def _parse_shot(index: int, data: Any, fps: int) -> Shot:
    where = f"shots[{index}]"
    _check_keys(where, data, _SHOT_REQUIRED, _SHOT_OPTIONAL)
    shot_id = _text(f"{where}.id", data["id"])
    check_file_name_safe(f"{where}.id", shot_id)
    where = f"shot {shot_id}"
    source, status = data["source"], data["status"]
    if source not in SOURCES:
        raise ValueError(f"{where}: source must be one of {SOURCES}")
    if status not in STATUSES:
        raise ValueError(f"{where}: status must be one of {STATUSES}")
    if not isinstance(data["needs_character_keyframe"], bool):
        raise ValueError(f"{where}: needs_character_keyframe must be true or false")
    motion = data["motion"]
    if isinstance(motion, str):
        motion = [motion]
    if not isinstance(motion, list) or not all(isinstance(item, str) and item.strip() for item in motion):
        raise ValueError(f"{where}: motion must be a list of strings")
    duration = _number(f"{where}: duration_seconds", data["duration_seconds"], minimum=0, exclusive=True)
    frame_count = _frames(f"{where}: duration_seconds", duration, fps)
    fades = {}
    for key in ("fade_in_seconds", "fade_out_seconds"):
        seconds = _number(f"{where}: {key}", data.get(key) or 0, minimum=0)
        fades[key] = _frames(f"{where}: {key}", seconds, fps)
    if fades["fade_in_seconds"] + fades["fade_out_seconds"] > frame_count:
        raise ValueError(f"{where}: the fades are longer than the shot")

    clip = _parse_clip(f"{where}: clip", data.get("clip"))
    title_card = None
    if source == "title_card":
        if data["prompt"] is not None or data["negative_prompt"] is not None:
            raise ValueError(f"{where}: a title card is made in editing; prompt and negative_prompt must be null")
        if data["needs_character_keyframe"] or clip is not None:
            raise ValueError(f"{where}: a title card has no clip and needs no character keyframe")
        card = data.get("title_card")
        _check_keys(f"{where}: title_card", card, {"text"}, _TITLE_CARD_KEYS - {"text"})
        title_card = TitleCard(
            text=_text(f"{where}: title_card.text", card["text"]),
            subtitle=_text(f"{where}: title_card.subtitle", card.get("subtitle"), optional=True),
            image=_text(f"{where}: title_card.image", card.get("image"), optional=True),
        )
        prompt = negative_prompt = None
    else:
        if "title_card" in data:
            raise ValueError(f"{where}: only a title_card shot has a title_card block")
        prompt = _text(f"{where}: prompt", data["prompt"])
        negative_prompt = _text(f"{where}: negative_prompt", data["negative_prompt"])
    benchmark = _text(f"{where}: benchmark", data.get("benchmark"), optional=True)
    if benchmark is not None:
        check_file_name_safe(f"{where}: benchmark", benchmark)
    return Shot(
        id=shot_id, name=_text(f"{where}: name", data["name"]), source=source, duration_seconds=duration,
        frame_count=frame_count, needs_character_keyframe=data["needs_character_keyframe"], status=status,
        camera=_text(f"{where}: camera", data["camera"]), motion=tuple(item.strip() for item in motion),
        prompt=prompt, negative_prompt=negative_prompt, benchmark=benchmark, clip=clip,
        fade_in_frames=fades["fade_in_seconds"], fade_out_frames=fades["fade_out_seconds"],
        title_card=title_card, notes=str(data.get("notes") or ""),
    )


def parse_shotlist(data: Any) -> Shotlist:
    _check_keys("shot list", data, _TRAILER_KEYS - {"character"}, {"character"})
    trailer_id = _text("trailer_id", data["trailer_id"])
    check_file_name_safe("trailer_id", trailer_id)
    if data["status"] not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    fps = data["fps"]
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive whole number")
    _check_keys("resolution", data["resolution"], {"width", "height"})
    resolution = Resolution(**data["resolution"])
    if resolution.width % 2 or resolution.height % 2:
        raise ValueError("resolution must be even in both directions (H.264, yuv420p)")
    profile = _text("artist_profile", data["artist_profile"])
    check_file_name_safe("artist_profile", profile)
    if not isinstance(data["shots"], list) or not data["shots"]:
        raise ValueError("shots must be a non-empty list")
    shots = tuple(_parse_shot(index, shot, fps) for index, shot in enumerate(data["shots"]))
    ids = [shot.id for shot in shots]
    duplicates = sorted({shot_id for shot_id in ids if ids.count(shot_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate shot ids {duplicates}")
    total = _number("total_duration_seconds", data["total_duration_seconds"], minimum=0, exclusive=True)
    actual = sum(shot.frame_count for shot in shots) / fps
    if abs(actual - total) > 1e-6:
        raise ValueError(f"the shots add up to {actual:g} s, but total_duration_seconds is {total:g}")
    return Shotlist(
        trailer_id=trailer_id, working_title=_text("working_title", data["working_title"]),
        status=data["status"], fps=fps, resolution=resolution, artist_profile=profile, shots=shots,
        character=_text("character", data.get("character"), optional=True),
    )


def load_shotlist(path: str | Path) -> Shotlist:
    """Read and validate trailers/<trailer_id>/shotlist.yaml. The folder must
    be named after the trailer_id."""
    path = Path(path)
    shotlist = parse_shotlist(yaml.safe_load(path.read_text(encoding="utf-8")))
    if path.resolve().parent.name != shotlist.trailer_id:
        raise ValueError(f"{path}: the folder must be named {shotlist.trailer_id}")
    return shotlist
