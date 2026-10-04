"""Zip one benchmark run (record + output directory) for download. Run on the
machine that produced the run.

    python3 scripts/collect_run.py --benchmark B [--run-id R]

Writes <project>/dist/run-<B>-<R>.zip. Paths inside the zip are
project-relative ("benchmarks/results/B/R.json", "outputs/B/R/out.mp4"), so
scripts/import_run.py can put them back at the same places. Without --run-id
the newest run of that benchmark is taken. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

UNSAFE_NAME_CHARS = '/\\:*?"<>|'


class CollectError(RuntimeError):
    """The run cannot be collected; the message says why."""


def default_root() -> Path:
    override = os.environ.get("AI_STUDIO_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[1]


def check_name(kind: str, value: str) -> None:
    if not value or value in (".", "..") or any(c in value for c in UNSAFE_NAME_CHARS):
        raise CollectError(f"{kind} is not a valid id: {value!r}")


def newest_run_id(root: Path, benchmark: str) -> str:
    """Run id of the record with the latest timestamp."""
    records = sorted((root / "benchmarks" / "results" / benchmark).glob("*.json"))
    if not records:
        raise CollectError(f"no runs recorded for benchmark {benchmark!r}")

    def written_at(path: Path) -> tuple[str, str]:
        try:
            stamp = json.loads(path.read_text(encoding="utf-8")).get("timestamp") or ""
        except (OSError, ValueError):
            stamp = ""
        return (str(stamp), path.stem)

    return max(records, key=written_at).stem


def collect_run(root: Path, benchmark: str, run_id: str | None = None) -> Path:
    """Write <root>/dist/run-<benchmark>-<run_id>.zip and return its path."""
    root = Path(root).resolve()
    check_name("benchmark", benchmark)
    if run_id is None:
        run_id = newest_run_id(root, benchmark)
    check_name("run id", run_id)

    record = root / "benchmarks" / "results" / benchmark / f"{run_id}.json"
    if not record.is_file():
        raise CollectError(f"no record benchmarks/results/{benchmark}/{run_id}.json")
    output_dir = root / "outputs" / benchmark / run_id
    files = [record]
    if output_dir.is_dir():  # a run that failed before generation has none
        files += sorted(path for path in output_dir.rglob("*") if path.is_file())

    archive_path = root / "dist" / f"run-{benchmark}-{run_id}.zip"
    archive_path.parent.mkdir(exist_ok=True)
    temporary = archive_path.with_suffix(".zip.part")
    # Stored, not deflated: the video is already compressed.
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive_path)
    return archive_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-id", default=None, help="default: the newest run of the benchmark")
    args = parser.parse_args(argv)
    try:
        archive_path = collect_run(default_root(), args.benchmark, args.run_id)
    except CollectError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    print(f"run zip: {archive_path}")
    print(f"size: {archive_path.stat().st_size / 1e6:.1f} MB, {len(names)} files")
    for name in names:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
