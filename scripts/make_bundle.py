"""Pack the project into one zip to upload to the cloud machine. Run on Windows.

    python scripts/make_bundle.py [--allow-dirty]

Writes dist/ai-animation-studio-bundle.zip containing, under one top-level
folder "ai-animation-studio/":
  - every git-tracked file (working-tree content)
  - the .git directory, so runs on the other machine record the real git_commit
  - the git-ignored canonical start frame, after verifying its SHA-256

Untracked and ignored files (.venv, outputs, dist, caches, ...) are never
packed. A dirty working tree is refused unless --allow-dirty: a bundle should
be exactly one commit. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "ai-animation-studio-bundle.zip"
TOP_FOLDER = "ai-animation-studio"
START_FRAME = "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
START_FRAME_SHA256 = "3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb"


class BundleError(RuntimeError):
    """The bundle cannot be made; the message says why."""


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True,
        )
    except FileNotFoundError:
        raise BundleError("git is not installed or not on PATH") from None
    except subprocess.CalledProcessError as exc:
        raise BundleError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from None
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_bundle(
    root: Path = PROJECT_ROOT,
    *,
    allow_dirty: bool = False,
    start_frame: str = START_FRAME,
    start_frame_sha256: str = START_FRAME_SHA256,
) -> Path:
    """Write <root>/dist/ai-animation-studio-bundle.zip and return its path."""
    root = Path(root).resolve()
    if not (root / ".git").is_dir():
        raise BundleError(f"{root} is not a git repository (no .git directory)")

    status = _git(root, "status", "--porcelain").strip()
    if status and not allow_dirty:
        raise BundleError(
            "the working tree has uncommitted or untracked files:\n"
            f"{status}\n"
            "commit them first, or pass --allow-dirty (untracked files are still not packed)"
        )

    frame = root / start_frame
    if not frame.is_file():
        raise BundleError(f"start frame not found: {start_frame}")
    actual = sha256_file(frame)
    if actual != start_frame_sha256:
        raise BundleError(
            f"start frame {start_frame} has sha256 {actual}, expected {start_frame_sha256}; "
            "it is not the canonical file"
        )

    tracked = [name for name in _git(root, "ls-files", "-z").split("\0") if name]
    git_files = sorted(
        path.relative_to(root).as_posix() for path in (root / ".git").rglob("*") if path.is_file()
    )
    members = [name for name in tracked if (root / name).is_file()]  # skips deleted files
    if start_frame not in members:
        members.append(start_frame)

    bundle = root / "dist" / BUNDLE_NAME
    bundle.parent.mkdir(exist_ok=True)
    temporary = bundle.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members) + git_files:
            archive.write(root / name, f"{TOP_FOLDER}/{name}")
    temporary.replace(bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-dirty", action="store_true", help="bundle even with uncommitted changes")
    args = parser.parse_args(argv)
    try:
        bundle = make_bundle(allow_dirty=args.allow_dirty)
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(bundle) as archive:
        count = len(archive.namelist())
    print(f"bundle: {bundle}")
    print(f"size: {bundle.stat().st_size / 1e6:.1f} MB, {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
