"""Post-process a recorded run into a derived variant.

The run record and the run's own video are never touched. A variant is a
derived artifact that lives beside them:

    outputs/<benchmark>/<run_id>/post/<profile_id>/
        out.mp4          the styled video (same encoder settings, same fps)
        manifest.json    what was applied to what: profile (every parameter),
                         profile file hash, source video hash, hold pattern,
                         tool versions
        metrics_v<N>.json   the quality metrics of the styled video, marked
                         variant "post/<profile_id>"

A variant directory is never overwritten: one profile version applied to one
run gives one variant.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.media_probe import ffprobe_video
from ai_studio.paths import resolve_path, to_project_relative
from ai_studio.post.profile import POST_VERSION, apply_profile, file_sha256, load_profile
from ai_studio.quality.frames import decode_image, decode_video, ffmpeg_version
from ai_studio.quality.store import VideoMetricsStore, measure_video, sha256_file
from ai_studio.video_encode import encode_frames

RECIPES_DIR = "configs/style_recipes"


def variant_dir(root: Path, benchmark_id: str, run_id: str, profile_id: str) -> Path:
    return resolve_path(f"outputs/{benchmark_id}/{run_id}/post/{profile_id}", root)


def postprocess_run(
    logger: BenchmarkLogger, metrics: VideoMetricsStore, root: str | Path,
    benchmark_id: str, run_id: str, profile_path: str | Path,
) -> dict[str, Any]:
    """Apply an artist profile to a run's video. Returns the manifest.

    FileExistsError if this variant already exists; ValueError if the run did
    not succeed; FileNotFoundError if its video or start frame is not here."""
    root = Path(root)
    run = metrics.measurable_run(benchmark_id, run_id)  # success + has a video
    profile = load_profile(profile_path)
    source = resolve_path(run.output_video, root)
    if not source.is_file():
        raise FileNotFoundError(f"video of run {run_id} is not on this machine: {run.output_video}")
    derived = (run.start_frame or {}).get("derived_file")
    start_frame = resolve_path(derived, root) if derived else None
    if "palette" in profile.habits and (start_frame is None or not start_frame.is_file()):
        raise FileNotFoundError(
            f"profile {profile.profile_id} learns its palette from the start frame, "
            f"but run {run_id} has no derived start frame on this machine"
        )

    target = variant_dir(root, benchmark_id, run_id, profile.profile_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)  # FileExistsError: never overwrite a variant
    try:
        started = time.perf_counter()
        info = ffprobe_video(source)
        frames = decode_video(source, analysis_width=None, gray=False)
        source_image = (
            decode_image(start_frame, (info.width, info.height), gray=False) if start_frame is not None
            else frames[0]
        )
        styled, timing_info = apply_profile(frames, profile, source_image)
        output = target / "out.mp4"
        encoder = encode_frames(styled, output, fps=int(round(info.fps)))
        seconds = time.perf_counter() - started

        variant = f"post/{profile.profile_id}"
        measured = measure_video(
            output, benchmark_id=benchmark_id, run_id=run_id, reference_image=start_frame, variant=variant,
        )
        metrics_path = metrics.save_variant(measured, target)
        manifest = {
            "benchmark_id": benchmark_id,
            "run_id": run_id,
            "variant": variant,
            "recipe": profile.profile_id,
            "profile": profile.to_dict(),  # every parameter, defaults included
            "recipe_file": to_project_relative(resolve_path(profile_path, root), root),
            "recipe_sha256": file_sha256(profile_path),
            "post_version": POST_VERSION,
            "source_video": run.output_video,
            "source_video_sha256": sha256_file(source),
            "start_frame": to_project_relative(start_frame, root) if start_frame is not None else None,
            "start_frame_sha256": sha256_file(start_frame) if start_frame is not None else None,
            "output_video": to_project_relative(output, root),
            "output_video_sha256": sha256_file(output),
            "frame_count": int(len(styled)),
            "fps": int(round(info.fps)),
            "width": info.width,
            "height": info.height,
            "encoder": encoder,
            "timing": timing_info,
            "metrics_file": to_project_relative(metrics_path, root),
            "tool_versions": {
                "python": platform.python_version(), "numpy": np.__version__, "ffmpeg": ffmpeg_version(),
            },
            "processing_seconds": round(seconds, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with (target / "manifest.json").open("x", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)  # no half-made variant is left behind
        raise
    return manifest
