"""Project-root path resolution.

Rule: every relative path in this project (benchmark definitions, media paths,
results/outputs/evaluations directories) is relative to the PROJECT ROOT, never
to the shell's current working directory.

The project root is:
  1. $AI_STUDIO_ROOT, if set (e.g. on a cloud machine where the package is
     installed non-editable), otherwise
  2. the directory containing pyproject.toml above this source file.

Paths are recorded in benchmark files in project-relative POSIX form
("benchmarks/inputs/clip.mp4") so records are comparable across machines;
paths outside the project root are recorded as absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV_VAR = "AI_STUDIO_ROOT"


def project_root() -> Path:
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"{ROOT_ENV_VAR}={override!r} is not a directory")
        return root
    # src/ai_studio/paths.py -> project root
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            f"cannot locate the project root from {__file__}; set {ROOT_ENV_VAR}"
        )
    return root


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Absolute path for `path`; relative paths are taken relative to the project root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((Path(root) if root is not None else project_root()).resolve() / path).resolve()


def to_project_relative(path: str | Path, root: str | Path | None = None) -> str:
    """Portable form for records: POSIX path relative to the root when inside it,
    otherwise the absolute path, still with forward slashes."""
    absolute = resolve_path(path, root)
    base = (Path(root) if root is not None else project_root()).resolve()
    try:
        return absolute.relative_to(base).as_posix()
    except ValueError:
        return absolute.as_posix()


def definitions_dir(root: str | Path | None = None) -> Path:
    return resolve_path("benchmarks/definitions", root)


def results_dir(root: str | Path | None = None) -> Path:
    return resolve_path("benchmarks/results", root)


def evaluations_dir(root: str | Path | None = None) -> Path:
    return resolve_path("benchmarks/evaluations", root)


def comic_evaluations_dir(root: str | Path | None = None) -> Path:
    return resolve_path("benchmarks/comic_evaluations", root)


def metrics_dir(root: str | Path | None = None) -> Path:
    return resolve_path("benchmarks/metrics", root)


def outputs_dir(root: str | Path | None = None) -> Path:
    return resolve_path("outputs", root)
