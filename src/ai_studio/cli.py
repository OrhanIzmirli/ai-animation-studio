"""Command-line launcher for one benchmark run.

    python scripts/run_benchmark.py --benchmark comic_rooftop_01 --backend wan-ti2v-5b
        --seed 42 --width 1280 --height 704 [--param key=value ...] [--notes TEXT]
        [--purpose benchmark|smoke|experiment] [--no-metrics]

--purpose says why the run is made (default: benchmark) and is stored in the
record as task_parameters["run_purpose"]; smoke tests and experiments can
then be told apart from runs meant for evaluation.

After a successful run the automatic quality metrics are computed and stored
(benchmarks/metrics/...). That step is not part of the run: if it fails, a
warning is printed and the exit code stays 0, because the run record is
already saved. --no-metrics skips it.

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
from ai_studio.benchmarking.runner import RUN_PURPOSES
from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.generation.backends import BackendNotInstalledError
from ai_studio.generation.backends.registry import get_backend
from ai_studio.generation.capabilities import UnsupportedCapabilityError
from ai_studio.media_probe import VideoInfo, ffprobe_video
from ai_studio.paths import (
    definitions_dir,
    metrics_dir,
    outputs_dir,
    project_root,
    results_dir,
    to_project_relative,
)
from ai_studio.quality.store import VideoMetricsStore

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
    parser.add_argument(
        "--purpose", choices=RUN_PURPOSES, default="benchmark",
        help="why this run is made; stored in the record (default: benchmark)",
    )
    parser.add_argument(
        "--no-metrics", action="store_true",
        help="do not compute the automatic quality metrics after a successful run",
    )
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
    purpose: str = "benchmark",
    metrics: bool = True,
    media_probe: Callable[[Path], VideoInfo] | None = None,
    out: Callable[[str], None] = print,
) -> tuple[int, list[str]]:
    """Run one benchmark. Returns (exit_code, written project-relative paths):
    the record file, the run's output directory if it exists and, after a
    successful run with `metrics`, the metrics file. The list is empty when
    nothing was recorded."""
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
        if purpose not in RUN_PURPOSES:
            raise ValueError(f"purpose must be one of {RUN_PURPOSES}, got {purpose!r}")
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        return _setup_problem(exc, out)

    logger = BenchmarkLogger(results_dir(root))
    runner = BenchmarkRunner(
        logger, outputs_dir(root), root=root, media_probe=media_probe or ffprobe_video,
    )
    out(f"benchmark: {definition.benchmark_id}   backend: {backend}   seed: {seed}   "
        f"resolution: {width}x{height}   purpose: {purpose}   run_id: {run_id}")
    try:
        try:
            record = runner.run_definition(
                definition, backend_instance, seed=seed, output_resolution=resolution,
                backend_parameters=dict(parameters or {}), run_id=run_id, notes=notes, purpose=purpose,
            )
        finally:
            # A backend that started something (e.g. a local server) stops it here.
            close = getattr(backend_instance, "close", None)
            if callable(close):
                close()
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
    if record.image_output:
        for image in record.image_output["images"]:
            size = f"{image['width']}x{image['height']}" if image["width"] else "size not measured"
            out(f"image: {image['path']} ({size})")
    elif media:
        out(f"measured: {media['fps']:g} fps, {media['frame_count']} frames, "
            f"{media['duration_seconds']:.3f} s, {media['width']}x{media['height']} ({media['probe']})")
    else:
        out(f"not measured; backend reported fps={record.fps} frames={record.frame_count}")
    _print_timing(record, out)
    for line in record.notes.splitlines():
        if line.startswith("warning: "):
            out(line)
    written = _written_paths(logger, definition.benchmark_id, run_id, root, out)
    if metrics and not record.image_output:  # the automatic metrics are video metrics
        written += _measure(logger, definition.benchmark_id, run_id, root, out, definition.regions)
    return EXIT_SUCCESS, written


def _measure(
    logger: BenchmarkLogger, benchmark_id: str, run_id: str, root: Path, out: Callable[[str], None],
    regions: Any = None,
) -> list[str]:
    """Compute and store the quality metrics of a finished run. Never fatal:
    the run record is already saved."""
    store = VideoMetricsStore(metrics_dir(root), logger)
    try:
        measured = store.measure_run(benchmark_id, run_id, root, regions=regions)
    except Exception as exc:  # metrics are an extra; any failure is only reported
        out(f"warning: quality metrics were not computed: {type(exc).__name__}: {exc}")
        return []
    values = measured.metrics
    psnr = values.get("start_frame_psnr")
    out(f"metrics v{measured.metrics_version}: motion_mean {values['motion_mean']:.3f}, "
        f"total_change {values['total_change']:.3f}, "
        f"start_frame_psnr {'n/a' if psnr is None else f'{psnr:.2f} dB'}, "
        f"pulse_ratio {'n/a' if values['temporal_pulse_ratio'] is None else format(values['temporal_pulse_ratio'], '.2f')}, "
        f"luminance_flicker {values['luminance_flicker']:.3f}, held_frame_ratio {values['held_frame_ratio']:.3f}")
    for name, ratios in ((measured.region_metrics or {}).get("ratios") or {}).items():
        out(f"region {name} / rest: " + ", ".join(
            f"{measure} {'n/a' if value is None else format(value, '.2f')}" for measure, value in ratios.items()
        ))
    path = to_project_relative(store.path_for(benchmark_id, run_id, measured.metrics_version), root)
    out(f"metrics: {path}")
    return [path]


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
        purpose=args.purpose, metrics=not args.no_metrics,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
