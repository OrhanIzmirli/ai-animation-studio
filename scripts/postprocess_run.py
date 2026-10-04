"""Apply a style-layer artist profile ("recipe") to a recorded run.

    python scripts/postprocess_run.py --benchmark B --run-id R --recipe NAME

NAME is a profile in configs/style_recipes/ (e.g. comic_finish_v1). Writes
outputs/B/R/post/NAME/ with out.mp4, manifest.json and the quality metrics of
the styled video. The run record and the run's own video are not touched, and
an existing variant is never overwritten. CPU only. See docs/STYLE_LAYER.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.benchmarking import BenchmarkLogger  # noqa: E402
from ai_studio.paths import metrics_dir, project_root, resolve_path, results_dir  # noqa: E402
from ai_studio.post.variant import RECIPES_DIR, postprocess_run  # noqa: E402
from ai_studio.quality.frames import FrameDecodeError  # noqa: E402
from ai_studio.quality.store import VideoMetricsStore  # noqa: E402


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe", required=True, help=f"profile name in {RECIPES_DIR}/, without .yaml")
    args = parser.parse_args(argv)
    root = project_root()
    logger = BenchmarkLogger(results_dir(root))
    recipe = resolve_path(f"{RECIPES_DIR}/{args.recipe}.yaml", root)
    if "/" in args.recipe or "\\" in args.recipe or not recipe.is_file():
        known = sorted(p.stem for p in resolve_path(RECIPES_DIR, root).glob("*.yaml"))
        out(f"ERROR: no recipe {args.recipe!r}; available: {', '.join(known) or 'none'}")
        return 2
    try:
        manifest = postprocess_run(
            logger, VideoMetricsStore(metrics_dir(root), logger), root, args.benchmark, args.run_id, recipe,
        )
    except FileExistsError:
        out(f"ERROR: outputs/{args.benchmark}/{args.run_id}/post/{args.recipe} already exists; "
            "a variant is never overwritten")
        return 1
    except (FileNotFoundError, ValueError, FrameDecodeError, RuntimeError) as exc:
        out(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    timing = manifest["timing"]
    out(f"variant: {manifest['variant']} of {args.benchmark}/{args.run_id}")
    out(f"video: {manifest['output_video']}")
    out(f"drawings: {timing['drawing_count']} for {manifest['frame_count']} frames, "
        f"held_frame_ratio {timing['held_frame_ratio']:.3f}, "
        f"hold_pattern_periodicity {timing['hold_pattern_periodicity']}")
    out(f"metrics: {manifest['metrics_file']}")
    out(f"processing time: {manifest['processing_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
