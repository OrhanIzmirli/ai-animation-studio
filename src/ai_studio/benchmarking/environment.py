"""Collect a snapshot of the machine and software a run executed on.

Every probe is optional: a missing GPU, CUDA, PyTorch, nvidia-smi or git
yields None for that key instead of an error.

Two different CUDA versions are recorded on purpose:
  reported_cuda_version  highest CUDA version the NVIDIA DRIVER supports
                         (from the nvidia-smi header; no PyTorch needed)
  cuda_version           CUDA runtime PyTorch was BUILT with (torch.version.cuda)
"""

from __future__ import annotations

import importlib
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
    "gpu_model",
    "gpu_count",
    "vram_gb",
    "driver_version",
    "reported_cuda_version",
    "cuda_version",
    "pytorch_version",
)

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
    env["pytorch_version"] = torch.get("pytorch_version")
    env["cuda_version"] = torch.get("cuda_version")
    env["gpu_model"] = torch.get("gpu_model") or smi.get("gpu_model")
    env["vram_gb"] = torch.get("vram_gb") or smi.get("vram_gb")
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
    return out.stdout.strip()


def _git_info(repo_dir: Path | None) -> dict[str, Any]:
    if repo_dir is None or shutil.which("git") is None:
        return {}
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    if not commit:
        return {}
    status = _run(["git", "status", "--porcelain"], cwd=repo_dir)
    # Uncommitted changes mean the commit hash alone does not reproduce the run.
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


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
