"""List runs that started generating and never got a record; optionally record them as failed.

    python scripts/find_abandoned_runs.py                       (list only)
    python scripts/find_abandoned_runs.py --benchmark B
    python scripts/find_abandoned_runs.py --stale-after-hours 12
    python scripts/find_abandoned_runs.py --record

The runner leaves outputs/<benchmark>/<run_id>/RUN_IN_PROGRESS.json behind
when a run dies during generation. A folder with such a marker, older than
--stale-after-hours (default 6; use more than your longest run, a run that is
still generating also has a marker) and without a record is listed here.

--record writes a status="failed" record with failure_stage "abandoned" for
each listed run, from the data in its marker. Records are never overwritten
and nothing is deleted: the marker and whatever the run left in its folder
stay where they are.

Folders that have neither a marker nor a record (runs abandoned before
markers existed) are listed separately; they cannot be recorded automatically.

Exit code 0 if no abandoned run is left without a record, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.benchmarking import BenchmarkLogger  # noqa: E402
from ai_studio.benchmarking.in_progress import (  # noqa: E402
    DEFAULT_STALE_HOURS,
    find_abandoned,
    find_unmarked,
    record_abandoned,
)
from ai_studio.paths import outputs_dir, project_root, results_dir, to_project_relative  # noqa: E402


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default=None, help="default: every benchmark")
    parser.add_argument("--stale-after-hours", type=float, default=DEFAULT_STALE_HOURS)
    parser.add_argument("--record", action="store_true", help="write a failed record for each abandoned run")
    args = parser.parse_args(argv)
    if args.stale_after_hours < 0:
        parser.error("--stale-after-hours must be >= 0")
    if args.benchmark is not None and (
        any(c in args.benchmark for c in "/\\") or args.benchmark in (".", "..")
    ):
        parser.error("--benchmark must be a benchmark id, not a path")
    root = project_root()
    logger = BenchmarkLogger(results_dir(root))
    outputs = outputs_dir(root)

    abandoned = find_abandoned(
        outputs, logger, stale_after_hours=args.stale_after_hours, benchmark=args.benchmark,
    )
    if not abandoned:
        out(f"no abandoned runs (marker older than {args.stale_after_hours:g} h and no record)")
    unrecorded = 0
    for run in abandoned:
        folder = to_project_relative(run.output_dir, root)
        out(f"ABANDONED {run.benchmark_id}/{run.run_id}: started {run.started_at.isoformat()} "
            f"({run.age_hours:.1f} h ago), no record; folder {folder}")
        if run.problem:
            out(f"  cannot be recorded: {run.problem}")
            unrecorded += 1
        elif not args.record:
            unrecorded += 1
        else:
            try:
                record = record_abandoned(run, logger)
            except (FileExistsError, ValueError, TypeError, OSError) as exc:
                out(f"  NOT RECORDED - {type(exc).__name__}: {exc}")
                unrecorded += 1
                continue
            path = to_project_relative(logger.path_for(record.benchmark_id, record.run_id), root)
            out(f"  recorded as failed (failure_stage {record.failure_stage}): {path}")
    if abandoned and not args.record:
        out("run again with --record to write a failed record for each of them")

    unmarked = find_unmarked(outputs, logger, benchmark=args.benchmark)
    if unmarked:
        out("folders with neither a marker nor a record (not recorded automatically):")
        for folder in unmarked:
            out(f"  {to_project_relative(folder, root)}")
    return 1 if unrecorded else 0


if __name__ == "__main__":
    sys.exit(main())
