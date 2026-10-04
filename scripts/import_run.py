"""Import a run zip made by scripts/collect_run.py into this project. Run on
Windows.

    python scripts/import_run.py path/to/run-<benchmark>-<run_id>.zip

Every file is created at the same project-relative path it has in the zip
(benchmarks/results/..., outputs/...). Nothing is ever overwritten: if any
target already exists, nothing at all is written. Entries that would land
outside the project, or anywhere other than the results, metrics and outputs
folders, are rejected. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ALLOWED_PREFIXES = ("benchmarks/results/", "benchmarks/metrics/", "outputs/")


class ImportRunError(RuntimeError):
    """The zip was not imported; the message says why."""


def default_root() -> Path:
    override = os.environ.get("AI_STUDIO_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[1]


def safe_target(root: Path, name: str) -> Path:
    """Absolute target for a zip entry name, or ImportRunError if the name is
    not a plain project-relative path inside an allowed folder."""
    parts = name.split("/")
    if "\\" in name or ":" in name or any(part in ("", ".", "..") for part in parts):
        raise ImportRunError(f"unsafe path in zip: {name!r}")
    if not name.startswith(ALLOWED_PREFIXES):
        raise ImportRunError(
            f"unexpected path in zip: {name!r} (only {', '.join(ALLOWED_PREFIXES)} are imported)"
        )
    target = (root / Path(*parts)).resolve()
    if not target.is_relative_to(root):
        raise ImportRunError(f"path escapes the project root: {name!r}")
    return target


def import_run(zip_path: str | Path, root: Path) -> list[str]:
    """Extract the run zip into `root`. Returns the project-relative paths written."""
    root = Path(root).resolve()
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ImportRunError(f"cannot open {zip_path}: {exc}") from None
    with archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        if not entries:
            raise ImportRunError(f"{zip_path} contains no files")
        # Check everything before writing anything.
        targets = [(info, safe_target(root, info.filename)) for info in entries]
        if len({target for _, target in targets}) != len(targets):
            raise ImportRunError("the zip contains the same path more than once")
        existing = [info.filename for info, target in targets if target.exists()]
        if existing:
            raise ImportRunError(
                "refusing to overwrite existing files (nothing was written):\n  " + "\n  ".join(existing)
            )
        written: list[str] = []
        for info, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as source, open(target, "xb") as destination:  # exclusive create
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            except FileExistsError:
                raise ImportRunError(f"refusing to overwrite {info.filename} (it appeared during import)") from None
            written.append(info.filename)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", help="run zip made by scripts/collect_run.py")
    args = parser.parse_args(argv)
    root = default_root()
    try:
        written = import_run(args.zip, root)
    except ImportRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"imported {len(written)} files into {root}:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
