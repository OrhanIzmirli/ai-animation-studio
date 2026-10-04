"""Decode a video (or one image) into a numpy array at a fixed, small
analysis size.

Metrics are computed on downscaled frames so that they are cheap, comparable
between runs and insensitive to encoder noise at the pixel level:

    analysis width 320 (default), height from the aspect ratio, both even
    1280x704 -> 320x176

Decoding goes through an ffmpeg rawvideo pipe (area scaling), for videos and
for reference images alike, so both pass through the same scaler.

    grayscale  (N, H, W)     uint8, luma 0-255
    RGB        (N, H, W, 3)  uint8
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from ai_studio.media_probe import ffprobe_video

DEFAULT_ANALYSIS_WIDTH = 320
SCALE_FLAGS = "area"  # averaging downscale: no ringing, stable between frames
FFMPEG_TIMEOUT_SECONDS = 600


class FrameDecodeError(RuntimeError):
    """ffmpeg is missing, or it could not decode the file."""


def analysis_size(width: int, height: int, analysis_width: int = DEFAULT_ANALYSIS_WIDTH) -> tuple[int, int]:
    """(width, height) frames are scaled to: the given width, the height that
    keeps the aspect ratio, both rounded to even numbers (at least 2)."""
    if width <= 0 or height <= 0 or analysis_width <= 0:
        raise ValueError("width, height and analysis_width must be positive")

    def even(value: float) -> int:
        return max(2, 2 * round(value / 2))

    scaled_width = even(analysis_width)
    return scaled_width, even(height * scaled_width / width)


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FrameDecodeError("ffmpeg not found on PATH; it is needed to decode frames for the quality metrics")
    return ffmpeg


def ffmpeg_version() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    try:
        words = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return words[2] if len(words) >= 3 and words[1] == "version" else None


def _decode(path: Path, size: tuple[int, int], gray: bool, extra: list[str]) -> np.ndarray:
    width, height = size
    channels = 1 if gray else 3
    command = [
        _ffmpeg(), "-v", "error", "-i", str(path), *extra,
        "-vf", f"scale={width}:{height}:flags={SCALE_FLAGS}",
        "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise FrameDecodeError(f"ffmpeg timed out decoding {path}") from exc
    if completed.returncode != 0:
        raise FrameDecodeError(f"ffmpeg could not decode {path}: {completed.stderr.decode(errors='replace').strip()}")
    frame_bytes = width * height * channels
    if not completed.stdout or len(completed.stdout) % frame_bytes:
        raise FrameDecodeError(
            f"ffmpeg returned {len(completed.stdout)} bytes for {path}, not a whole number of "
            f"{width}x{height} frames"
        )
    shape = (-1, height, width) if gray else (-1, height, width, 3)
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(shape)


def decode_video(
    path: str | Path, *, analysis_width: int | None = DEFAULT_ANALYSIS_WIDTH, gray: bool = True,
) -> np.ndarray:
    """All frames of the first video stream, scaled to the analysis size
    (analysis_width=None: the video's own size, no scaling).
    (N, H, W) uint8 when gray, else (N, H, W, 3) uint8 RGB."""
    path = Path(path)
    if not path.is_file():
        raise FrameDecodeError(f"video not found: {path}")
    _ffmpeg()  # the clear "ffmpeg missing" error comes before any probing
    info = ffprobe_video(path)
    size = (
        (info.width, info.height) if analysis_width is None
        else analysis_size(info.width, info.height, analysis_width)
    )
    # passthrough: exactly the encoded frames, no duplicated or dropped ones
    return _decode(path, size, gray, ["-map", "0:v:0", "-fps_mode", "passthrough"])


def decode_image(path: str | Path, size: tuple[int, int], *, gray: bool = True) -> np.ndarray:
    """One image scaled to exactly `size` = (width, height), aspect ratio not
    preserved. (H, W) uint8 when gray, else (H, W, 3)."""
    path = Path(path)
    if not path.is_file():
        raise FrameDecodeError(f"image not found: {path}")
    return _decode(path, size, gray, ["-frames:v", "1"])[0]
