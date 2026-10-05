"""Make a static HTML page for reviewing runs side by side and rating them.

    python scripts/make_review_page.py --benchmark B --run R1 --run R2
    python scripts/make_review_page.py --benchmark B --run R1 --run R1:post/comic_finish_v1
    python scripts/make_review_page.py --benchmark B --all

--run names a run's own video (RUN_ID) or a post-processed variant of it
(RUN_ID:post/<profile_id>). --all takes every successful run of the benchmark
whose video is on this machine (no variants).

Writes outputs/reviews/<B>/<timestamp>/review.html and prints its path. Open
it by double-clicking: no server and no network are needed. It shows the
videos with synchronized play / pause / frame stepping, the stored metrics,
and a rating form (ComicAnimationEvaluation criteria). "Save ratings file"
downloads a JSON file; import it with scripts/import_ratings.py.

The page points at the videos with relative paths, so leave it where it is.
Nothing is measured, copied or changed. An existing page is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.benchmarking import BenchmarkLogger  # noqa: E402
from ai_studio.paths import project_root, resolve_path, results_dir, to_project_relative  # noqa: E402
from ai_studio.review import ReviewError, write_review_page  # noqa: E402


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run", action="append", default=[], metavar="RUN_ID[:post/PROFILE]")
    parser.add_argument("--all", action="store_true", help="every successful run whose video is here")
    args = parser.parse_args(argv)
    if bool(args.run) == args.all:
        parser.error("give --run (one or more) or --all")
    root = project_root()
    specs = list(args.run)
    if args.all:
        for record in BenchmarkLogger(results_dir(root)).list_runs(args.benchmark):
            if record.status == "success" and record.output_video and resolve_path(record.output_video, root).is_file():
                specs.append(record.run_id)
        if not specs:
            out(f"ERROR: benchmark {args.benchmark!r} has no successful run with a video on this machine")
            return 1
    try:
        path = write_review_page(root, args.benchmark, specs)
    except (ReviewError, FileExistsError, ValueError) as exc:
        out(f"ERROR: {exc}")
        return 1
    out(f"review page: {to_project_relative(path, root)}")
    out(f"items: {', '.join(specs)}")
    out(f"open it by double-clicking: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
