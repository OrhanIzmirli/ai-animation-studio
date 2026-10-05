"""Encode frames to the project's standard H.264 video with ffmpeg.

Shared by every producer of video (model adapters, the style layer), so that
all outputs use the same encoder settings. numpy is imported lazily: this
module must import on a machine without it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_FPS = 24
ENCODER_SETTINGS = {
    "codec": "libx264", "pix_fmt": "yuv420p", "crf": 12, "preset": "slow",
    "fps": DEFAULT_FPS, "audio": False,
}


def frame_to_uint8(frame: Any) -> Any:
    """One H x W x 3 frame as uint8 RGB. Float frames are taken as [0, 1]:
    clipped, scaled and rounded (never truncated)."""
    import numpy as np

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape (height, width, 3), got {frame.shape}")
    if frame.dtype == np.uint8:
        return frame
    if not np.issubdtype(frame.dtype, np.floating):
        raise ValueError(f"frame must be float in [0, 1] or uint8, got {frame.dtype}")
    return np.rint(np.clip(np.nan_to_num(frame.astype(np.float32)), 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_frames(frames: Sequence[Any], output: str | Path, fps: int = DEFAULT_FPS) -> dict[str, Any]:
    """Encode frames to H.264 by piping raw RGB into ffmpeg. Returns the
    encoder settings used. No audio, no frame-rate conversion."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode the output video")
    if len(frames) == 0:
        raise RuntimeError("the pipeline returned no frames")
    first = frame_to_uint8(frames[0])
    height, width = first.shape[:2]
    settings = dict(ENCODER_SETTINGS, fps=fps)
    command = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", settings["codec"], "-pix_fmt", settings["pix_fmt"],
        "-crf", str(settings["crf"]), "-preset", settings["preset"], "-r", str(fps), str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    written = 0
    try:
        # An error from the frame source (a failed download, a truncated image)
        # is not caught here: it must not look like "ffmpeg closed the pipe".
        for index, frame in enumerate(frames):
            data = first if index == 0 else frame_to_uint8(frame)
            if data.shape != first.shape:
                raise ValueError(f"frame {index} has shape {data.shape}, expected {first.shape}")
            try:
                process.stdin.write(data.tobytes())
            except OSError:
                break  # ffmpeg exited early; its exit code and stderr are reported below
            written += 1
    except BaseException:
        process.kill()
        process.communicate()
        raise
    _, stderr = process.communicate()  # closes stdin, waits
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({process.returncode}): {stderr.decode(errors='replace').strip()}")
    if written != len(frames):
        raise RuntimeError(f"only {written} of {len(frames)} frames reached the encoder; {output} is incomplete")
    return settings


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
