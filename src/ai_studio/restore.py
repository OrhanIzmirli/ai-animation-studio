"""Regenerate the lost output video of a recorded run, and keep it only if it
is the same file.

    python -m ai_studio.restore --benchmark B --run-id R [--run-id R2 ...]

A run record is immutable and names its video; the SHA-256 of that video is
in the run's metrics record. If the video is lost, the run can be generated
again with the same backend, seed and settings. The result is put back at
the recorded path ONLY if its SHA-256 equals the recorded one: then it is
provably the same file. Otherwise it is discarded and the run stays without
a video. Nothing is ever overwritten. The regeneration happens in a
temporary folder outside the project, and that folder is the only thing this
module ever deletes. A restored video is marked read-only.

Model-agnostic: it uses the backend contract. Which of the recorded
model_parameters were request parameters is asked from the backend
(`backend.defaults`, the names of its backend_parameters).
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_studio.benchmarking.definitions import BenchmarkDefinition, load_definition
from ai_studio.benchmarking.logger import BenchmarkLogger
from ai_studio.benchmarking.schema import BenchmarkResult
from ai_studio.generation.backends.base import BackendNotInstalledError, VideoBackend
from ai_studio.paths import definitions_dir, metrics_dir, project_root, resolve_path, results_dir
from ai_studio.quality.store import sha256_file


@dataclass(frozen=True)
class RestoreOutcome:
    run_id: str
    restored: bool
    expected_sha256: str
    actual_sha256: str | None
    reason: str  # one sentence: why it was (not) restored


class RestoreError(RuntimeError):
    """The run cannot be regenerated as recorded; nothing was attempted."""


def request_parameters(record: BenchmarkResult, backend: VideoBackend) -> dict[str, Any]:
    """The backend_parameters to request: the recorded values of the
    parameters this backend accepts."""
    names = getattr(backend, "defaults", None)
    if not isinstance(names, dict):
        raise RestoreError(f"{type(backend).__name__} does not expose its backend parameters (defaults)")
    return {name: record.model_parameters[name] for name in names if name in record.model_parameters}


def recorded_video_sha256(root: Path, benchmark_id: str, run_id: str) -> str | None:
    """The video hash from a metrics record of the RAW run (newest version
    first; records of post-processed variants are ignored)."""
    folder = metrics_dir(root) / benchmark_id / run_id
    for path in sorted(folder.glob("metrics_v*.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("variant") is None and record.get("video_sha256"):
            return record["video_sha256"]
    return None


def restore_run_video(
    record: BenchmarkResult,
    definition: BenchmarkDefinition,
    backend: VideoBackend,
    expected_sha256: str,
    root: str | Path,
) -> RestoreOutcome:
    """Generate the run again and restore its video if the hash matches.

    RestoreError if the run cannot be repeated exactly as recorded (it did
    not succeed, the definition or an input changed, the backend differs) or
    if the video is already there. The backend is prepared and used but not
    closed; the caller owns it."""
    root = Path(root)
    if record.status != "success" or not record.output_video:
        raise RestoreError(f"run {record.run_id} did not succeed; it has no video to restore")
    target = resolve_path(record.output_video, root)
    if target.exists():
        raise RestoreError(f"{record.output_video} already exists; nothing is overwritten")
    meta = backend.metadata()
    if meta.backend_name != record.backend:
        raise RestoreError(f"run {record.run_id} was made with {record.backend!r}, not {meta.backend_name!r}")
    if definition.benchmark_id != record.benchmark_id:
        raise RestoreError("the definition belongs to another benchmark")
    if definition.prompt != record.prompt or definition.negative_prompt != record.negative_prompt:
        raise RestoreError("the benchmark definition's prompt has changed since the run was recorded")

    request = definition.to_request(
        seed=record.seed, output_resolution=record.resolution,
        backend_parameters=request_parameters(record, backend),
    )
    resolved = request.with_resolved_paths(lambda p: str(resolve_path(p, root)))
    for path in resolved.input_paths:
        relative = Path(path).resolve().relative_to(root.resolve()).as_posix()
        if not Path(path).is_file():
            raise RestoreError(f"input file missing: {relative}")
        if record.input_checksums.get(relative) != sha256_file(path):
            raise RestoreError(f"input file {relative} is not the file the run used (checksum differs)")

    work = Path(tempfile.mkdtemp(prefix="ai_studio_restore_"))  # outside the project
    try:
        backend.validate_inputs(resolved)
        backend.prepare()
        result = backend.generate(resolved, work)
        actual = sha256_file(result.output_video)
        if result.model_version != record.model_version:
            return RestoreOutcome(record.run_id, False, expected_sha256, actual,
                                  f"model version differs: {result.model_version} vs recorded {record.model_version}")
        if actual != expected_sha256:
            return RestoreOutcome(record.run_id, False, expected_sha256, actual,
                                  "the regenerated video is not byte-identical; discarded")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(result.output_video, "rb") as source, open(target, "xb") as destination:  # exclusive create
            shutil.copyfileobj(source, destination)
        if sha256_file(target) != expected_sha256:  # the copy itself is verified too
            raise RestoreError(f"copying to {record.output_video} failed verification")
        target.chmod(stat.S_IMODE(target.stat().st_mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        return RestoreOutcome(record.run_id, True, expected_sha256, actual, "byte-identical; restored")
    finally:
        shutil.rmtree(work, ignore_errors=True)  # our own temporary folder, never anything in the project


def main(
    argv: Sequence[str] | None = None,
    out: Callable[[str], None] = print,
    backend_for: Callable[[str], VideoBackend] | None = None,
) -> int:
    """Exit code 0 if every requested run was restored, 1 otherwise."""
    parser = argparse.ArgumentParser(prog="python -m ai_studio.restore", description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-id", action="append", required=True)
    args = parser.parse_args(argv)
    if backend_for is None:
        from ai_studio.generation.backends.registry import get_backend

        def backend_for(name: str) -> VideoBackend:
            return get_backend(name)()

    root = project_root()
    logger = BenchmarkLogger(results_dir(root))
    definition = load_definition(definitions_dir(root) / f"{args.benchmark}.yaml")
    failures = 0
    for run_id in args.run_id:
        try:
            record = logger.load(args.benchmark, run_id)
            expected = recorded_video_sha256(root, args.benchmark, run_id)
            if expected is None:
                raise RestoreError("no recorded video hash (the run has no metrics record)")
            backend = backend_for(record.backend)
            try:
                outcome = restore_run_video(record, definition, backend, expected, root)
            finally:
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
        except (RestoreError, BackendNotInstalledError, FileNotFoundError, KeyError, ValueError) as exc:
            out(f"{run_id}: NOT RESTORED - {type(exc).__name__}: {exc}")
            failures += 1
            continue
        out(f"{run_id}: {'RESTORED' if outcome.restored else 'NOT RESTORED'} - {outcome.reason}")
        out(f"  recorded sha256    {outcome.expected_sha256}")
        out(f"  regenerated sha256 {outcome.actual_sha256}")
        failures += not outcome.restored
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
