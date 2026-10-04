"""Command-line launcher for one benchmark run.

    python scripts/run_benchmark.py --benchmark comic_rooftop_01 --backend wan-ti2v-5b
        --seed 42 --width 1280 --height 704 [--param key=value ...] [--notes TEXT]

Exit codes:
    0  the run succeeded and was recorded
    1  the run failed and was recorded as a failed run
    2  setup problem, nothing recorded (backend not installed, draft
       benchmark, unknown benchmark/backend, bad arguments, run id exists)

The launcher only wires things together: definition file -> request,
backend name -> class (registry), default results/outputs directories ->
BenchmarkRunner. Everything about a model stays in its adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ai_studio.benchmarking import (
    BenchmarkLogger,
    BenchmarkResult,
    BenchmarkRunError,
    BenchmarkRunner,
    DraftBenchmarkError,
    Resolution,
    load_definition,
    new_run_id,
)
from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.generation.backends import BackendNotInstalledError
from ai_studio.generation.backends.registry import get_backend
from ai_studio.generation.capabilities import UnsupportedCapabilityError
from ai_studio.media_probe import VideoInfo, ffprobe_video
from ai_studio.paths import definitions_dir, outputs_dir, project_root, results_dir, to_project_relative

EXIT_SUCCESS, EXIT_FAILED_RUN, EXIT_SETUP = 0, 1, 2


def parse_param(text: str) -> tuple[str, Any]:
    """"key=value" -> (key, value). The value is parsed as JSON when possible
    (cpu_offload=true, num_inference_steps=50), otherwise kept as a string."""
    key, separator, raw = text.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError(f"--param must look like key=value, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def parse_params(items: Sequence[str]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for item in items:
        key, value = parse_param(item)
        if key in parameters:
            raise ValueError(f"--param {key} given more than once")
        parameters[key] = value
    return parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark", description="Run one benchmark definition on one backend and record the result.",
    )
    parser.add_argument("--benchmark", required=True, help="benchmark id: benchmarks/definitions/<id>.yaml")
    parser.add_argument("--backend", required=True, help="backend name, e.g. wan-ti2v-5b")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--width", required=True, type=int, help="output width in pixels")
    parser.add_argument("--height", required=True, type=int, help="output height in pixels")
    parser.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="backend parameter; the value is parsed as JSON when possible (repeatable)",
    )
    parser.add_argument("--notes", default="", help="free text stored in the record")
    parser.add_argument("--run-id", default=None, help="default: a new timestamped id")
    return parser


def run_benchmark(
    *,
    benchmark: str,
    backend: str,
    seed: int,
    width: int,
    height: int,
    parameters: dict[str, Any] | None = None,
    notes: str = "",
    run_id: str | None = None,
    media_probe: Callable[[Path], VideoInfo] | None = None,
    out: Callable[[str], None] = print,
) -> tuple[int, list[str]]:
    """Run one benchmark. Returns (exit_code, written project-relative paths):
    the record file and, if it exists, the run's output directory. The list
    is empty when nothing was recorded."""
    try:
        check_file_name_safe("benchmark", benchmark)
        root = project_root()
        definition_path = definitions_dir(root) / f"{benchmark}.yaml"
        if not definition_path.is_file():
            raise FileNotFoundError(f"no benchmark definition {to_project_relative(definition_path, root)}")
        definition = load_definition(definition_path)
        backend_instance = get_backend(backend)()
        resolution = Resolution(width=width, height=height)
        run_id = run_id or new_run_id()
        check_file_name_safe("run_id", run_id)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        return _setup_problem(exc, out)

    logger = BenchmarkLogger(results_dir(root))
    runner = BenchmarkRunner(
        logger, outputs_dir(root), root=root, media_probe=media_probe or ffprobe_video,
    )
    out(f"benchmark: {definition.benchmark_id}   backend: {backend}   seed: {seed}   "
        f"resolution: {width}x{height}   run_id: {run_id}")
    try:
        record = runner.run_definition(
            definition, backend_instance, seed=seed, output_resolution=resolution,
            backend_parameters=dict(parameters or {}), run_id=run_id, notes=notes,
        )
    except BenchmarkRunError as exc:
        cause = exc.__cause__
        out(f"status: failed (stage: {exc.stage})")
        out(f"error: {type(cause).__name__}: {cause}")
        if exc.record is None:
            out("the failed run could NOT be recorded")
        else:
            _print_timing(exc.record, out)
        return EXIT_FAILED_RUN, _written_paths(logger, definition.benchmark_id, run_id, root, out)
    except (
        DraftBenchmarkError, BackendNotInstalledError, UnsupportedCapabilityError,
        FileExistsError, ValueError,
    ) as exc:
        return _setup_problem(exc, out)

    out("status: success")
    media = record.output_media
    if media:
        out(f"measured: {media['fps']:g} fps, {media['frame_count']} frames, "
            f"{media['duration_seconds']:.3f} s, {media['width']}x{media['height']} ({media['probe']})")
    else:
        out(f"not measured; backend reported fps={record.fps} frames={record.frame_count}")
    _print_timing(record, out)
    for line in record.notes.splitlines():
        if line.startswith("warning: "):
            out(line)
    return EXIT_SUCCESS, _written_paths(logger, definition.benchmark_id, run_id, root, out)


def _setup_problem(exc: BaseException, out: Callable[[str], None]) -> tuple[int, list[str]]:
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
    out(f"setup problem, nothing recorded: {type(exc).__name__}: {message}")
    return EXIT_SETUP, []


def _print_timing(record: BenchmarkResult, out: Callable[[str], None]) -> None:
    def seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f} s"

    out(f"total time: {seconds(record.total_time_seconds)} "
        f"(prepare {seconds(record.prepare_time_seconds)}, "
        f"generation {seconds(record.generation_time_seconds)})")


def _written_paths(
    logger: BenchmarkLogger, benchmark_id: str, run_id: str, root: Path, out: Callable[[str], None],
) -> list[str]:
    written: list[str] = []
    record_path = logger.path_for(benchmark_id, run_id)
    if record_path.is_file():
        written.append(to_project_relative(record_path, root))
        out(f"record: {written[-1]}")
        output_video = json.loads(record_path.read_text(encoding="utf-8")).get("output_video")
        if output_video:
            out(f"output video: {output_video}")
    output_dir = outputs_dir(root) / benchmark_id / run_id
    if output_dir.is_dir():
        written.append(to_project_relative(output_dir, root))
        out(f"output directory: {written[-1]}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # bad arguments: argparse exits with code 2
    try:
        parameters = parse_params(args.param)
    except ValueError as exc:
        parser.error(str(exc))
    code, _ = run_benchmark(
        benchmark=args.benchmark, backend=args.backend, seed=args.seed, width=args.width,
        height=args.height, parameters=parameters, notes=args.notes, run_id=args.run_id,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
