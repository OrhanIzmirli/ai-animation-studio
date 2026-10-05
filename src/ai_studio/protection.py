"""Protect recorded run data from being deleted or changed by accident.

Two safeguards:

1. Read-only run files. Once a run is verified and recorded, its output
   video (or images) and derived start frame are marked read-only
   (protect_run_files).
   On Windows a read-only file cannot be deleted or overwritten by normal
   means (os.remove and Explorer's plain delete fail or ask first).
   This guards against accidents; it is not access control: anyone can clear
   the flag.

2. No clean-up under the data folders. Code in this project that deletes
   things (temporary files, another program's working copies) must go
   through safe_unlink / safe_rmtree / safe_rmdir, which refuse any path
   inside <project root>/outputs or <project root>/benchmarks. Clean-up code
   therefore cannot remove a run's data even if it is handed a wrong path.
   If the project root cannot be determined, deletion is refused as well.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path
from typing import Any

from ai_studio.paths import project_root, resolve_path

PROTECTED_FOLDERS = ("outputs", "benchmarks")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ProtectedPathError(PermissionError):
    """Clean-up code tried to delete something inside a protected data folder."""


def is_protected(path: str | Path, root: str | Path | None = None) -> bool:
    """True if `path` is one of <root>/outputs, <root>/benchmarks or anything
    inside them (after resolving symlinks and '..')."""
    base = (Path(root) if root is not None else project_root()).resolve()
    resolved = Path(path).resolve()
    return any(
        resolved == base / folder or (base / folder) in resolved.parents for folder in PROTECTED_FOLDERS
    )


def assert_deletable(path: str | Path, root: str | Path | None = None) -> None:
    """Raise ProtectedPathError unless `path` is outside the data folders."""
    try:
        protected = is_protected(path, root)
    except RuntimeError as exc:  # no project root: we cannot know, so we refuse
        raise ProtectedPathError(f"refusing to delete {path}: the project root is unknown ({exc})") from exc
    if protected:
        raise ProtectedPathError(
            f"refusing to delete {path}: clean-up never touches {' or '.join(PROTECTED_FOLDERS)}/ "
            "of the project"
        )


def safe_unlink(path: str | Path, root: str | Path | None = None) -> None:
    assert_deletable(path, root)
    Path(path).unlink()


def safe_rmdir(path: str | Path, root: str | Path | None = None) -> None:
    """Remove an EMPTY directory outside the data folders."""
    assert_deletable(path, root)
    Path(path).rmdir()


def safe_rmtree(path: str | Path, root: str | Path | None = None, ignore_errors: bool = False) -> None:
    assert_deletable(path, root)
    shutil.rmtree(path, ignore_errors=ignore_errors)


def make_read_only(path: str | Path) -> None:
    """Clear every write permission of a file."""
    path = Path(path)
    path.chmod(stat.S_IMODE(path.stat().st_mode) & ~_WRITE_BITS)


def is_read_only(path: str | Path) -> bool:
    return not Path(path).stat().st_mode & _WRITE_BITS


def run_files(record: Any, root: str | Path | None = None) -> list[Path]:
    """The files of a recorded run that must not change: its output video or
    images and its derived start frame (those that exist)."""
    candidates = [
        getattr(record, "output_video", None),
        (getattr(record, "start_frame", None) or {}).get("derived_file"),
        *(image["path"] for image in (getattr(record, "image_output", None) or {}).get("images", [])),
    ]
    return [path for item in candidates if item and (path := resolve_path(item, root)).is_file()]


def protect_run_files(record: Any, root: str | Path | None = None) -> list[Path]:
    """Mark a recorded run's output video / images and derived start frame read-only.
    Returns the files that were protected. Best effort: a file whose
    permissions cannot be changed is skipped."""
    protected = []
    for path in run_files(record, root):
        try:
            make_read_only(path)
        except OSError:
            continue
        protected.append(path)
    return protected
