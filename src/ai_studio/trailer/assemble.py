"""Assemble a trailer from its shot list: one chosen clip per shot, one artist
profile for all of them, cuts and fades, a title card.

    outputs/trailers/<trailer_id>/<timestamp>/
        trailer.mp4      the assembly (the project's standard encoder settings)
        manifest.json    what it was made of: shot list hash, profile (every
                         parameter), per shot the clip and its hash, the hold
                         pattern, fades, tool versions. Written last.

Per shot:
  clip chosen   the clip is decoded, cut to the shot's length (its first
                frames), and the artist profile is applied through the style
                layer's public API (ai_studio.post.profile.apply_profile).
                The SAME profile, with the same seed, for every shot.
  no clip yet   a labelled grey card of the shot's length (not styled).
  title card    drawn here or taken from a finished image (not styled; never
                made by a video model).
Fades are from / to black and are applied after styling.

Nothing is generated and nothing is guessed: a clip must have exactly the
trailer's resolution and frame rate and at least the shot's length, or the
assembly is refused before anything is rendered. An assembly is never
overwritten (one per timestamp). The video is built in a temporary folder
outside the project and copied into place when it is complete.
"""

from __future__ import annotations

import json
import platform
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.media_probe import ffprobe_video
from ai_studio.paths import outputs_dir, resolve_path, results_dir, to_project_relative
from ai_studio.post.profile import POST_VERSION, ArtistProfile, apply_profile, load_profile
from ai_studio.protection import make_read_only, safe_rmtree
from ai_studio.quality.frames import decode_image, decode_video, ffmpeg_version
from ai_studio.quality.store import sha256_file
from ai_studio.trailer.cards import placeholder_card, title_card
from ai_studio.trailer.shotlist import Shot, Shotlist, load_shotlist
from ai_studio.video_encode import encode_frames

TRAILERS_DIR = "trailers"
RECIPES_DIR = "configs/style_recipes"
VIDEO_NAME = "trailer.mp4"
MANIFEST_NAME = "manifest.json"
FPS_TOLERANCE = 0.01


class TrailerError(RuntimeError):
    """The trailer cannot be assembled as described; nothing was written."""


def shotlist_path(root: str | Path, trailer_id: str) -> Path:
    return resolve_path(f"{TRAILERS_DIR}/{trailer_id}/shotlist.yaml", root)


def assembly_dir(root: str | Path, trailer_id: str, timestamp: str) -> Path:
    return outputs_dir(root) / "trailers" / trailer_id / timestamp


def new_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fade_alpha(index: int, frame_count: int, fade_in: int, fade_out: int) -> float:
    """Brightness (0-1) of frame `index` of a shot: a fade-in starts at black,
    a fade-out ends at black, both linear."""
    alpha = 1.0
    if index < fade_in:
        alpha = min(alpha, index / fade_in)
    remaining = frame_count - 1 - index
    if remaining < fade_out:
        alpha = min(alpha, remaining / fade_out)
    return alpha


def _faded(frame: np.ndarray, alpha: float) -> np.ndarray:
    return frame if alpha >= 1.0 else np.rint(frame.astype(np.float32) * alpha).astype(np.uint8)


@dataclass
class _Segment:
    """One shot on the timeline. `render` returns its (N, H, W, 3) frames."""

    shot: Shot
    kind: str  # "clip" | "placeholder" | "title_card"
    entry: dict[str, Any]  # this shot's part of the manifest
    video: Path | None = None  # clip only
    start_frame: Path | None = None  # clip only: the palette source, if the run has one


@dataclass
class _Timeline(Sequence):
    """All frames of the trailer, rendered one shot at a time while the
    encoder reads them, so only one shot is in memory."""

    shotlist: Shotlist
    profile: ArtistProfile
    segments: list[_Segment]
    cards: dict[str, np.ndarray]
    _cache: tuple[int, np.ndarray] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return self.shotlist.frame_count

    def __getitem__(self, index: Any) -> np.ndarray:
        if index != 0:
            raise IndexError("the timeline is read in order")
        return self._frame(0, 0)

    def __iter__(self) -> Iterator[np.ndarray]:
        for number, segment in enumerate(self.segments):
            for index in range(segment.shot.frame_count):
                yield self._frame(number, index)
            self._cache = None

    def _frame(self, number: int, index: int) -> np.ndarray:
        shot = self.segments[number].shot
        alpha = fade_alpha(index, shot.frame_count, shot.fade_in_frames, shot.fade_out_frames)
        return _faded(self._frames(number)[index], alpha)

    def _frames(self, number: int) -> np.ndarray:
        if self._cache is None or self._cache[0] != number:
            self._cache = (number, self._render(self.segments[number]))
        return self._cache[1]

    def _render(self, segment: _Segment) -> np.ndarray:
        shot, resolution = segment.shot, self.shotlist.resolution
        shape = (shot.frame_count, resolution.height, resolution.width, 3)
        if segment.kind != "clip":
            return np.broadcast_to(self.cards[shot.id], shape)
        frames = decode_video(segment.video, analysis_width=None, gray=False)[: shot.frame_count]
        if frames.shape != shape:
            raise TrailerError(f"shot {shot.id}: decoded clip has shape {frames.shape}, expected {shape}")
        source_image = (
            decode_image(segment.start_frame, (resolution.width, resolution.height), gray=False)
            if segment.start_frame is not None else frames[0]
        )
        styled, timing = apply_profile(frames, self.profile, source_image)
        segment.entry["timing"] = timing
        return styled


def _resolve_clip(shot: Shot, shotlist: Shotlist, logger: BenchmarkLogger, root: Path) -> _Segment:
    """Find and check a shot's chosen clip. TrailerError if it cannot be used."""
    clip = shot.clip or {}
    entry: dict[str, Any] = {"clip": dict(clip)}
    start_frame = None
    if "run" in clip:
        benchmark_id, run_id = clip["run"].split("/")
        try:
            record = logger.load(benchmark_id, run_id)
        except FileNotFoundError:
            raise TrailerError(f"shot {shot.id}: no recorded run {clip['run']}") from None
        if record.status != "success" or not record.output_video:
            raise TrailerError(f"shot {shot.id}: run {clip['run']} did not succeed; it has no video")
        video = resolve_path(record.output_video, root)
        derived = (record.start_frame or {}).get("derived_file")
        if derived and resolve_path(derived, root).is_file():
            start_frame = resolve_path(derived, root)
    else:
        video = resolve_path(clip["path"], root)
    if not video.is_file():
        raise TrailerError(f"shot {shot.id}: clip not found: {to_project_relative(video, root)}")
    info = ffprobe_video(video)
    resolution = shotlist.resolution
    if (info.width, info.height) != (resolution.width, resolution.height):
        raise TrailerError(
            f"shot {shot.id}: clip is {info.width}x{info.height}, the trailer is "
            f"{resolution.width}x{resolution.height} (clips are never rescaled)"
        )
    if abs(info.fps - shotlist.fps) > FPS_TOLERANCE:
        raise TrailerError(f"shot {shot.id}: clip has {info.fps:g} fps, the trailer has {shotlist.fps}")
    if info.frame_count < shot.frame_count:
        raise TrailerError(
            f"shot {shot.id}: clip has {info.frame_count} frames, the shot needs {shot.frame_count} "
            f"({shot.duration_seconds:g} s)"
        )
    entry.update(
        video=to_project_relative(video, root),
        video_sha256=sha256_file(video),
        clip_frame_count=info.frame_count,
        unused_frames=info.frame_count - shot.frame_count,  # cut from the end
        palette_source=to_project_relative(start_frame, root) if start_frame is not None else "first frame",
    )
    return _Segment(shot, "clip", entry, video=video, start_frame=start_frame)


def assemble_trailer(
    root: str | Path, trailer_id: str, *, profile_name: str | None = None, timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble trailers/<trailer_id>/shotlist.yaml. Returns the manifest.

    profile_name overrides the shot list's artist_profile. FileExistsError if
    this timestamp's assembly exists; FileNotFoundError / ValueError for a
    missing or invalid shot list or profile; TrailerError for an unusable clip."""
    root = Path(root).resolve()
    source = shotlist_path(root, trailer_id)
    shotlist = load_shotlist(source)
    profile_path = resolve_path(f"{RECIPES_DIR}/{profile_name or shotlist.artist_profile}.yaml", root)
    profile = load_profile(profile_path)
    timestamp = timestamp or new_timestamp()
    target = assembly_dir(root, trailer_id, timestamp)
    if target.exists():
        raise FileExistsError(f"assembly already exists: {target}")

    logger = BenchmarkLogger(results_dir(root))
    resolution, total = shotlist.resolution, len(shotlist.shots)
    segments: list[_Segment] = []
    cards: dict[str, np.ndarray] = {}
    for number, shot in enumerate(shotlist.shots, start=1):
        if shot.source == "title_card":
            image = resolve_path(shot.title_card.image, root) if shot.title_card.image else None
            cards[shot.id] = title_card(shot.title_card, resolution, image)
            entry = {"title_card": {
                "text": shot.title_card.text, "subtitle": shot.title_card.subtitle,
                "image": to_project_relative(image, root) if image else None,
                "image_sha256": sha256_file(image) if image else None,
            }}
            segments.append(_Segment(shot, "title_card", entry))
        elif shot.clip is None:
            cards[shot.id] = placeholder_card(shot, number, total, resolution)
            segments.append(_Segment(shot, "placeholder", {}))
        else:
            segments.append(_resolve_clip(shot, shotlist, logger, root))

    work = Path(tempfile.mkdtemp(prefix="ai_studio_trailer_"))  # outside the project
    try:
        started = time.perf_counter()
        built = work / VIDEO_NAME
        encoder = encode_frames(_Timeline(shotlist, profile, segments, cards), built, fps=shotlist.fps)
        seconds = time.perf_counter() - started
        info = ffprobe_video(built)
        if info.frame_count != shotlist.frame_count or (info.width, info.height) != (
            resolution.width, resolution.height,
        ):
            raise TrailerError(
                f"the encoded trailer has {info.frame_count} frames at {info.width}x{info.height}, expected "
                f"{shotlist.frame_count} at {resolution.width}x{resolution.height}"
            )

        final = target / VIDEO_NAME
        shots, position = [], 0
        for segment in segments:
            shot = segment.shot
            shots.append({
                "id": shot.id, "name": shot.name, "kind": segment.kind, "status": shot.status,
                "first_frame": position, "frame_count": shot.frame_count,
                "duration_seconds": shot.duration_seconds,
                "fade_in_frames": shot.fade_in_frames, "fade_out_frames": shot.fade_out_frames,
                "styled": segment.kind == "clip",
                **segment.entry,
            })
            position += shot.frame_count
        placeholders = [segment.shot.id for segment in segments if segment.kind == "placeholder"]
        manifest = {
            "trailer_id": shotlist.trailer_id,
            "assembly": timestamp,
            "working_title": shotlist.working_title,
            "shotlist": to_project_relative(source, root),
            "shotlist_sha256": sha256_file(source),
            "shotlist_status": shotlist.status,
            "complete": not placeholders,  # False: at least one shot is a placeholder card
            "placeholder_shots": placeholders,
            "artist_profile": profile.profile_id,  # one profile for every clip
            "profile": profile.to_dict(),  # every parameter, defaults included
            "profile_file": to_project_relative(profile_path, root),
            "profile_sha256": sha256_file(profile_path),
            "post_version": POST_VERSION,
            "fps": shotlist.fps,
            "width": resolution.width,
            "height": resolution.height,
            "frame_count": shotlist.frame_count,
            "duration_seconds": shotlist.duration_seconds,
            "shots": shots,
            "output_video": to_project_relative(final, root),
            "output_video_sha256": sha256_file(built),
            "output_media": info.to_record(),
            "encoder": encoder,
            "tool_versions": {
                "python": platform.python_version(), "numpy": np.__version__, "ffmpeg": ffmpeg_version(),
            },
            "processing_seconds": round(seconds, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Publish: create the folder exclusively, the video first, the manifest last.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)  # FileExistsError: never overwrite an assembly
        shutil.copyfile(built, final)
        with (target / MANIFEST_NAME).open("x", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        make_read_only(final)
    finally:
        safe_rmtree(work, root, ignore_errors=True)  # the temporary folder only
    return manifest
