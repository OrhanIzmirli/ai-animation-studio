"""Measure a video file instead of trusting what produced it.

The benchmark runner records the MEASURED fps and frame count of every output
video, so a backend that reports "24 fps, 120 frames" but wrote something else
cannot slip into the results.

Two different failures:
  ProbeUnavailableError  the probe tool is not installed; nothing was measured
  RuntimeError           the probe ran and the file is broken / not a video

duration_seconds is frame_count / fps, never the container duration: the
container value includes edit lists and rounding and is not what a viewer sees.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

# Keys of BenchmarkResult.output_media (schema v8). Single source of truth.
OUTPUT_MEDIA_KEYS = ("probe", "width", "height", "fps", "frame_count", "duration_seconds", "codec")

FFPROBE_TIMEOUT_SECONDS = 300


class ProbeUnavailableError(RuntimeError):
    """The probe tool is missing, so the video could not be measured at all."""


@dataclass(frozen=True)
class VideoInfo:
    probe: str  # tool and version that measured the file, e.g. "ffprobe 6.1.1"
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.probe, str) or not self.probe:
            raise ValueError("probe must be a non-empty string")
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool) or self.fps <= 0:
            raise ValueError("fps must be a positive number")
        if self.codec is not None and (not isinstance(self.codec, str) or not self.codec):
            raise ValueError("codec must be a non-empty string or None")

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    def to_record(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration_seconds, 6),
            "codec": self.codec,
        }


def _parse_rate(value: Any) -> float | None:
    """"24/1" -> 24.0, "24000/1001" -> 23.976...; None for "0/0" or garbage."""
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def _ffprobe_version(ffprobe: str) -> str:
    try:
        out = subprocess.run(
            [ffprobe, "-version"], capture_output=True, text=True, check=True, timeout=30,
        ).stdout
        words = out.split()  # "ffprobe version 6.1.1 Copyright ..."
        if len(words) >= 3 and words[1] == "version":
            return f"ffprobe {words[2]}"
    except (OSError, subprocess.SubprocessError):
        pass
    return "ffprobe unknown"


def ffprobe_video(path: str | Path) -> VideoInfo:
    """Measure the first video stream of `path` by decoding every frame."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ProbeUnavailableError("ffprobe not found on PATH; cannot measure the output video")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as e:
        raise ProbeUnavailableError(f"ffprobe could not be started: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffprobe timed out on {path}") from e
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {completed.stderr.strip()}")
    try:
        streams = json.loads(completed.stdout).get("streams") or []
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}: {e}") from e
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    stream = streams[0]

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps is None:
        raise RuntimeError(f"ffprobe reported no frame rate for {path}")
    try:
        width, height = int(stream["width"]), int(stream["height"])
        frame_count = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"ffprobe output for {path} is missing {e}") from e
    try:
        return VideoInfo(
            probe=_ffprobe_version(ffprobe), width=width, height=height, fps=fps,
            frame_count=frame_count, codec=stream.get("codec_name") or None,
        )
    except ValueError as e:
        raise RuntimeError(f"ffprobe measured an invalid video {path}: {e}") from e
