"""Collect a snapshot of the machine and software a run executed on.

Every probe is optional: a missing GPU, CUDA, PyTorch, nvidia-smi or git
yields None for that key instead of an error.

env_vars holds a small allowlist of process environment variables that change
how a run behaves (allocator settings, visible devices, offline mode), each
None when unset. Nothing else from the process environment is recorded.

Whose PyTorch? The process that runs a benchmark (the "orchestrator") is not
always the process that runs the model: a backend may drive an external
program with its own Python. So this module records only what it can know:
  orchestrator_pytorch_version / orchestrator_cuda_version
                         the torch importable in THIS process, if any
  pytorch_version / cuda_version
                         the torch that actually ran the model. Left None
                         here; the backend reports them in its result.
  reported_cuda_version  highest CUDA version the NVIDIA DRIVER supports
                         (from the nvidia-smi header; no PyTorch needed)
Records written before this split have the orchestrator's torch in
pytorch_version / cuda_version and no orchestrator_* keys.

git_dirty answers "does the commit hash reproduce the CODE of this run?".
Changes that are only under the data paths (run records, metrics,
evaluations, outputs) do not make it true: every recorded run adds such
files. Nothing is hidden: git_status keeps the raw `git status --porcelain`
lines, and git_untracked_data_only says whether data paths were the only
changes.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ai_studio.paths import project_root

ENVIRONMENT_KEYS = (
    "operating_system",
    "python_version",
    "git_commit",
    "git_dirty",
    "git_untracked_data_only",
    "git_status",
    "gpu_model",
    "gpu_count",
    "vram_gb",
    "driver_version",
    "reported_cuda_version",
    "cuda_version",
    "pytorch_version",
    "orchestrator_cuda_version",
    "orchestrator_pytorch_version",
    "env_vars",
)

# Paths that hold run data, not code. Changes only here do not make a run "dirty".
DATA_PATH_PREFIXES = (
    "benchmarks/results/",
    "benchmarks/metrics/",
    "benchmarks/evaluations/",
    "benchmarks/comic_evaluations/",
    "outputs/",
)
MAX_GIT_STATUS_LINES = 200

# Process environment variables worth recording. Keep it short: no paths, no secrets.
RECORDED_ENV_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")

# Header formats seen in the wild:
#   "| NVIDIA-SMI 550.54   Driver Version: 550.54   CUDA Version: 12.4 |"
#   "| NVIDIA-SMI 617.14   KMD Version: 617.14   CUDA UMD Version: 13.4 |"
_CUDA_VERSION_RE = re.compile(r"CUDA(?: UMD)? Version:\s*([0-9][0-9.]*)")


def collect_environment(repo_dir: str | Path | None = None) -> dict[str, Any]:
    env: dict[str, Any] = dict.fromkeys(ENVIRONMENT_KEYS)
    env["operating_system"] = platform.platform()
    env["python_version"] = platform.python_version()
    env.update(_git_info(Path(repo_dir) if repo_dir else _default_repo_dir()))

    smi = _nvidia_smi_info()
    torch = _torch_info()
    env["driver_version"] = smi.get("driver_version")
    env["reported_cuda_version"] = smi.get("reported_cuda_version")
    env["gpu_count"] = smi.get("gpu_count") or torch.get("gpu_count")
    # pytorch_version / cuda_version stay None: only the backend knows which
    # torch ran the model (see the module docstring).
    env["orchestrator_pytorch_version"] = torch.get("pytorch_version")
    env["orchestrator_cuda_version"] = torch.get("cuda_version")
    env["gpu_model"] = torch.get("gpu_model") or smi.get("gpu_model")
    env["vram_gb"] = torch.get("vram_gb") or smi.get("vram_gb")
    env["env_vars"] = {name: os.environ.get(name) for name in RECORDED_ENV_VARS}
    return env


def _default_repo_dir() -> Path | None:
    try:
        return project_root()
    except RuntimeError:
        return None


def _run(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip("\n") if args[:2] == ["git", "status"] else out.stdout.strip()


def _git_info(repo_dir: Path | None) -> dict[str, Any]:
    if repo_dir is None or shutil.which("git") is None:
        return {}
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    if not commit:
        return {}
    # --untracked-files=all: list every file, so a new folder that holds only
    # data is not reported as one opaque "?? folder/" line.
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_dir)
    if status is None:
        return {"git_commit": commit, "git_dirty": None, "git_untracked_data_only": None, "git_status": None}
    return {"git_commit": commit, **classify_git_status(status)}


def _porcelain_paths(line: str) -> list[str]:
    """Paths named by one `git status --porcelain` line ("XY path" or
    "XY old -> new"), without git's quoting."""
    return [part.strip().strip('"') for part in line[3:].split(" -> ")]


def classify_git_status(porcelain: str) -> dict[str, Any]:
    """git_dirty / git_untracked_data_only / git_status from porcelain output.

    git_dirty                True if anything outside DATA_PATH_PREFIXES is
                             modified, staged or untracked: the commit hash
                             alone does not reproduce the code of the run.
    git_untracked_data_only  True if there are changes and all of them are
                             under the data paths (so git_dirty is False).
    git_status               the raw lines (at most MAX_GIT_STATUS_LINES).
    """
    lines = [line for line in porcelain.splitlines() if line.strip()]
    code_changes = [
        line for line in lines
        if not all(path.startswith(DATA_PATH_PREFIXES) for path in _porcelain_paths(line))
    ]
    return {
        "git_dirty": bool(code_changes),
        "git_untracked_data_only": bool(lines) and not code_changes,
        "git_status": lines[:MAX_GIT_STATUS_LINES],
    }


def _nvidia_smi_info() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {}
    info: dict[str, Any] = {}
    query = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if query:
        info.update(parse_gpu_query(query))
    header = _run(["nvidia-smi"])
    if header:
        cuda = parse_reported_cuda_version(header)
        if cuda:
            info["reported_cuda_version"] = cuda
    return info


def parse_gpu_query(output: str) -> dict[str, Any]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,driver_version` CSV output.
    Describes GPU 0; gpu_count counts all GPUs listed."""
    rows = [line for line in output.splitlines() if line.strip()]
    try:
        name, memory_mib, driver = (part.strip() for part in rows[0].split(","))
        return {
            "gpu_model": name,
            "vram_gb": round(float(memory_mib) / 1024, 2),
            "driver_version": driver,
            "gpu_count": len(rows),
        }
    except (ValueError, IndexError):
        return {}


def parse_reported_cuda_version(output: str) -> str | None:
    match = _CUDA_VERSION_RE.search(output)
    return match.group(1) if match else None


def _torch_info() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception:  # not installed, or a broken CUDA install failing at import
        return {}
    info: dict[str, Any] = {
        "pytorch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_model"] = props.name
            info["vram_gb"] = round(props.total_memory / 1024**3, 2)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info
