"""Import a ratings file saved from a review page into the comic evaluation store.

    python scripts/import_ratings.py ratings_<benchmark>_<evaluator>_<time>.json
    python scripts/import_ratings.py FILE --revise

Each rating becomes a ComicAnimationEvaluation, stored append-only:
  a run's own video   benchmarks/comic_evaluations/<benchmark>/<run>/<evaluator>.r001.json
  a post variant      benchmarks/comic_evaluations_variants/post/<profile>/<benchmark>/<run>/<evaluator>.r001.json

Everything is checked before anything is written; one problem means nothing
is imported. A rating is never overwritten: if you already rated a run, the
import is refused unless you pass --revise, which stores a new revision
(r002, ...) and keeps the earlier ones. The ratings file itself is only read.

Exit code 0 if imported, 1 if not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.paths import project_root, to_project_relative  # noqa: E402
from ai_studio.review import ReviewError, import_ratings  # noqa: E402


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="ratings JSON saved from a review page")
    parser.add_argument("--revise", action="store_true",
                        help="store a new revision where you already rated a run")
    args = parser.parse_args(argv)
    try:
        data = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        out(f"ERROR: cannot read {args.file}: {exc}")
        return 1
    root = project_root()
    try:
        written = import_ratings(data, root, revise=args.revise)
    except ReviewError as exc:
        out(f"ERROR: {exc}")
        out("nothing was imported")
        return 1
    out(f"imported {len(written)} rating(s) by {data['evaluator']}:")
    for path in written:
        out(f"  {to_project_relative(path, root)}")
    skipped = data.get("skipped") or []
    if skipped:
        out(f"left unrated on the page: {', '.join(str(item) for item in skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
