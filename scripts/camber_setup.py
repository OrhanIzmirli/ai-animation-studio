"""One-time (idempotent) environment setup on Camber. Run with the base
python from a notebook cell:

    python3 scripts/camber_setup.py

Creates the conda env /home/jovyan/envs/ai-studio (Python 3.11), installs
requirements/camber-wan.txt and this project into it, verifies the imports
and reports disk use. Only /home/jovyan survives a node stop, so the env
lives there; conda and pip caches go to /tmp so they do not eat the 50 GB
quota. Safe to run again: finished steps are skipped or are no-ops.

Standard library only; stops at the first error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_DIR = Path("/home/jovyan")
ENVS_DIR = HOME_DIR / "envs"
ENV_DIR = ENVS_DIR / "ai-studio"
ENV_PYTHON = ENV_DIR / "bin" / "python"
HF_HOME = HOME_DIR / "hf"
REQUIREMENTS = "requirements/camber-wan.txt"
PYTHON_VERSION = "3.11"
CACHE_ENVIRONMENT = {"CONDA_PKGS_DIRS": "/tmp/conda-pkgs", "PIP_CACHE_DIR": "/tmp/pip-cache"}
QUOTA_WARNING_GB = 45.0

VERIFY_CODE = (
    "import torch, torchvision, diffusers, transformers, ai_studio, sys; "
    "print('python', sys.version.split()[0]); "
    "print('torch', torch.__version__, '| cuda build', torch.version.cuda, "
    "'| cuda available', torch.cuda.is_available()); "
    "print('torchvision', torchvision.__version__); "
    "print('diffusers', diffusers.__version__); "
    "print('transformers', transformers.__version__); "
    "print('ai_studio', ai_studio.__file__)"
)


class SetupError(RuntimeError):
    """A setup step failed; the message says which."""


def stream(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Run a command, printing its output line by line as it arrives (shown
    live in a notebook cell). Returns the exit code."""
    print("$ " + " ".join(str(part) for part in command), flush=True)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def step(title: str, command: list[str], *, env: dict[str, str], cwd: Path | None = None) -> None:
    print(f"\n=== {title} ===", flush=True)
    try:
        code = stream(command, cwd=cwd, env=env)
    except OSError as exc:
        raise SetupError(f"{title}: could not start {command[0]}: {exc}") from exc
    if code != 0:
        raise SetupError(f"{title}: command exited with code {code}")


def disk_use_gb(path: Path) -> float | None:
    """Size of a directory tree in GB (du), or None if it cannot be measured."""
    try:
        completed = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True)
        return int(completed.stdout.split()[0]) / 1024**2
    except (OSError, ValueError, IndexError):
        return None


def report_disk_use() -> None:
    print("\n=== disk use ===", flush=True)
    for path in (ENVS_DIR, HF_HOME):
        size = disk_use_gb(path) if path.exists() else None
        print(f"{path}: {'missing or not measurable' if size is None else f'{size:.1f} GB'}")
    if not HF_HOME.exists():
        print(f"WARNING: {HF_HOME} does not exist: the model weights are not on this machine")
    total = disk_use_gb(HOME_DIR)
    if total is None:
        print(f"WARNING: could not measure {HOME_DIR}")
        return
    print(f"{HOME_DIR} total: {total:.1f} GB of the 50 GB quota")
    if total > QUOTA_WARNING_GB:
        print(f"WARNING: more than {QUOTA_WARNING_GB:g} GB used; a full disk makes runs fail. Free space first.")


def setup(project_root: Path = PROJECT_ROOT) -> None:
    env = {**os.environ, **CACHE_ENVIRONMENT}
    requirements = project_root / REQUIREMENTS
    if not requirements.is_file():
        raise SetupError(f"{requirements} not found: is the project unpacked correctly?")

    if ENV_PYTHON.exists():
        print(f"conda env already exists: {ENV_DIR}")
    else:
        conda = shutil.which("conda") or "/opt/conda/bin/conda"
        if ENV_DIR.exists():
            # Left over from an interrupted create: conda refuses a non-empty prefix.
            print(f"removing incomplete env {ENV_DIR}")
            shutil.rmtree(ENV_DIR)
        ENVS_DIR.mkdir(parents=True, exist_ok=True)
        # conda-forge only: the default channels need an interactive terms-of-service step.
        step(
            f"create conda env (python {PYTHON_VERSION})",
            [conda, "create", "--yes", "--prefix", str(ENV_DIR), "--override-channels",
             "--channel", "conda-forge", f"python={PYTHON_VERSION}", "pip"],
            env=env,
        )
        if not ENV_PYTHON.exists():
            raise SetupError(f"conda finished but {ENV_PYTHON} does not exist")

    step(
        f"install {REQUIREMENTS}",
        [str(ENV_PYTHON), "-m", "pip", "install", "--requirement", str(requirements)],
        env=env, cwd=project_root,
    )
    step(
        "install the project (editable, no dependencies)",
        [str(ENV_PYTHON), "-m", "pip", "install", "--editable", ".", "--no-deps"],
        env=env, cwd=project_root,
    )
    step("verify imports", [str(ENV_PYTHON), "-c", VERIFY_CODE], env=env, cwd=project_root)
    report_disk_use()


def main() -> int:
    try:
        setup()
    except SetupError as exc:
        print(f"\nSETUP FAILED: {exc}", flush=True)
        return 1
    print("\nSETUP OK. 'cuda available False' is expected on a CPU node.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
