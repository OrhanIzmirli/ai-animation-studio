"""Run one benchmark on a Camber GPU node. Run with the base python from a
notebook cell, e.g.:

    python3 scripts/camber_run.py --seed 42

Defaults: benchmark comic_rooftop_01, backend wan-ti2v-5b, 1280x704, and
what the L4 (22 GiB usable) needs: cpu_offload=true, vae_tiling=true and
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Each can be overridden
(--param vae_tiling=false; set PYTORCH_CUDA_ALLOC_CONF yourself beforehand).

    python3 scripts/camber_run.py --seed 42 --smoke

runs only 2 denoising steps and marks the run "SMOKE TEST - not for
evaluation": do this first in every GPU session, then the full run.

What it does:
  1. prints nvidia-smi; refuses to start (exit 2) when no GPU is visible
  2. runs scripts/run_benchmark.py with the env python made by
     camber_setup.py, with AI_STUDIO_ROOT, HF_HOME and HF_HUB_OFFLINE=1 set,
     showing its output live
  3. zips the run with scripts/collect_run.py and prints the zip path

The exit code is the benchmark CLI's: 0 success, 1 recorded failed run,
2 setup problem. Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_DIR = Path("/home/jovyan")
ENV_PYTHON = HOME_DIR / "envs" / "ai-studio" / "bin" / "python"
HF_HOME = HOME_DIR / "hf"
# Without offload the model does not fit; without tiling the VAE decode runs
# out of memory (measured on the L4: OOM at 18.34 GiB allocated).
DEFAULT_PARAMS = {"cpu_offload": "true", "vae_tiling": "true"}
DEFAULT_ALLOC_CONF = "expandable_segments:True"
SMOKE_PARAMS = {"num_inference_steps": "2"}
SMOKE_NOTE = "SMOKE TEST - not for evaluation"
EXIT_SETUP = 2


def stream(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Run a command, printing its output line by line as it arrives (shown
    live in a notebook cell). Returns the exit code."""
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def gpu_report() -> str | None:
    """nvidia-smi's output, or None when no GPU is visible."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        if listing.returncode != 0 or not listing.stdout.strip():
            return None
        report = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return report.stdout if report.returncode == 0 else listing.stdout


def new_run_id() -> str:
    """Same format as ai_studio.benchmarking.new_run_id."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def effective_params(args: argparse.Namespace) -> list[str]:
    """key=value backend parameters: defaults (plus the smoke-test ones) that
    the caller did not set, then the caller's own."""
    given = {param.partition("=")[0].strip() for param in args.param}
    defaults = {**DEFAULT_PARAMS, **(SMOKE_PARAMS if args.smoke else {})}
    return [f"{key}={value}" for key, value in defaults.items() if key not in given] + args.param


def effective_notes(args: argparse.Namespace) -> str:
    if not args.smoke:
        return args.notes
    return f"{SMOKE_NOTE}. {args.notes}" if args.notes else SMOKE_NOTE


def build_command(args: argparse.Namespace, run_id: str, project_root: Path) -> list[str]:
    params = effective_params(args)
    notes = effective_notes(args)
    command = [
        str(ENV_PYTHON), "-u", str(project_root / "scripts" / "run_benchmark.py"),
        "--benchmark", args.benchmark, "--backend", args.backend, "--seed", str(args.seed),
        "--width", str(args.width), "--height", str(args.height), "--run-id", run_id,
        "--purpose", "smoke" if args.smoke else args.purpose,
    ]
    for param in params:
        command += ["--param", param]
    if notes:
        command += ["--notes", notes]
    return command


def run_environment(project_root: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "AI_STUDIO_ROOT": str(project_root),
        "HF_HOME": str(HF_HOME),
        "HF_HUB_OFFLINE": "1",  # weights are already in HF_HOME; never touch the network
        "PYTHONUNBUFFERED": "1",
    }
    if not env.get("PYTORCH_CUDA_ALLOC_CONF"):  # the caller's own setting wins
        env["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOC_CONF
    return env


def collect(project_root: Path, benchmark: str, run_id: str) -> Path:
    """Zip the run with the collect_run.py next to this script."""
    spec = importlib.util.spec_from_file_location("collect_run", project_root / "scripts" / "collect_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect_run(project_root, benchmark, run_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--benchmark", default="comic_rooftop_01")
    parser.add_argument("--backend", default="wan-ti2v-5b")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="backend parameter (repeatable); cpu_offload=true and vae_tiling=true are added "
             "unless you set them yourself",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--purpose", choices=("benchmark", "smoke", "experiment"), default="benchmark",
        help="stored in the record; --smoke implies smoke",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="quick end-to-end check: 2 denoising steps, marked as not for evaluation",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    args = build_parser().parse_args(argv)

    report = gpu_report()
    if report is None:
        print("No GPU is visible on this node (nvidia-smi is missing or lists no GPU).")
        print("Stop this node and start the GPU Xsmall node, then run this again.")
        return EXIT_SETUP
    print(report, flush=True)
    if not ENV_PYTHON.exists():
        print(f"{ENV_PYTHON} not found: run scripts/camber_setup.py first.")
        return EXIT_SETUP
    if not HF_HOME.is_dir():
        print(f"{HF_HOME} not found: the model weights are not on this machine.")
        return EXIT_SETUP

    run_id = new_run_id()
    command = build_command(args, run_id, project_root)
    env = run_environment(project_root)
    print("mode: " + ("SMOKE TEST (not for evaluation)" if args.smoke else "full run"))
    print(f"backend parameters: {' '.join(effective_params(args))}")
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "HF_HUB_OFFLINE", "AI_STUDIO_ROOT"):
        print(f"{name}={env[name]}")
    print("$ " + " ".join(command), flush=True)
    code = stream(command, cwd=project_root, env=env)

    record = project_root / "benchmarks" / "results" / args.benchmark / f"{run_id}.json"
    if record.is_file():
        archive = collect(project_root, args.benchmark, run_id)
        print(f"\nrun zip (download this file): {archive}")
        print(f"size: {archive.stat().st_size / 1e6:.1f} MB")
    else:
        print("\nnothing was recorded, so there is no run zip.")
    print(f"exit code: {code}")
    print("Remember to STOP the GPU node when you are done: GPU time counts while it is on.")
    return code


if __name__ == "__main__":
    sys.exit(main())
