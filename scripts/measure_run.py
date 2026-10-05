"""Compute the automatic quality metrics for benchmark runs and print a table.

    python scripts/measure_run.py --benchmark B              (newest run)
    python scripts/measure_run.py --benchmark B --run-id R
    python scripts/measure_run.py --benchmark B --all

Only successful runs whose video is on this machine are measured. Results go
to benchmarks/metrics/<B>/<run_id>/metrics_v<version>.json and are never
overwritten: runs that already have metrics are shown, not measured again.
Needs ffmpeg and ffprobe. See docs/QUALITY_METRICS.md.

Region metrics: if the benchmark definition declares analysis_regions, each
region is measured against the rest of the frame and the region / rest ratios
are printed under the table. To try regions without changing a definition:

    python scripts/measure_run.py --benchmark B --run-id R
        --region head=560,120,760,330 --region-size 1280x704 --print-only

--print-only measures and prints but stores nothing (use it for trying
regions; stored metrics always use the definition's regions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.benchmarking import BenchmarkLogger, load_definition  # noqa: E402
from ai_studio.paths import definitions_dir, metrics_dir, project_root, resolve_path, results_dir  # noqa: E402
from ai_studio.quality.frames import FrameDecodeError  # noqa: E402
from ai_studio.quality.metrics import METRICS_VERSION  # noqa: E402
from ai_studio.quality.regions import MEASURES, AnalysisRegions  # noqa: E402
from ai_studio.quality.store import VideoMetricsStore, measure_video  # noqa: E402

COLUMNS = (
    ("run", "{}"), ("purpose", "{}"), ("steps", "{}"), ("motion_mean", "{:.3f}"), ("total_change", "{:.3f}"),
    ("start_psnr", "{:.2f}"), ("pulse_ratio", "{:.2f}"), ("flicker", "{:.3f}"), ("held_ratio", "{:.3f}"),
    ("note", "{}"),
)


def table_row(run, record, note: str) -> list[str]:
    metrics = record.metrics if record else {}
    values = [
        run.run_id, run.task_parameters.get("run_purpose"), run.inference_steps, metrics.get("motion_mean"), metrics.get("total_change"),
        metrics.get("start_frame_psnr"), metrics.get("temporal_pulse_ratio"),
        metrics.get("luminance_flicker"), metrics.get("held_frame_ratio"), note,
    ]
    return ["-" if value is None else fmt.format(value) for (_, fmt), value in zip(COLUMNS, values)]


def print_table(rows: list[list[str]], out=print) -> None:
    header = [name for name, _ in COLUMNS]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        out("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def parse_regions(items: list[str], size: str | None) -> AnalysisRegions | None:
    """--region NAME=x0,y0,x1,y1 (repeatable) and --region-size WxH -> regions."""
    if not items:
        if size:
            raise ValueError("--region-size is only meaningful with --region")
        return None
    if not size:
        raise ValueError("--region needs --region-size WxH (the frame size the pixels refer to)")
    try:
        width, height = (int(part) for part in size.lower().split("x"))
        boxes = {}
        for item in items:
            name, _, numbers = item.partition("=")
            if name in boxes:
                raise ValueError(f"region {name} given twice")
            boxes[name] = [int(part) for part in numbers.split(",")]
    except ValueError as exc:
        raise ValueError(f"regions must look like NAME=x0,y0,x1,y1 with --region-size WxH ({exc})") from None
    return AnalysisRegions(width=width, height=height, boxes=boxes)


def definition_regions(root: Path, benchmark: str) -> AnalysisRegions | None:
    """The analysis regions of the benchmark's definition, if it has one and declares any."""
    path = definitions_dir(root) / f"{benchmark}.yaml"
    return load_definition(path).regions if path.is_file() else None


def print_regions(run_id: str, region_metrics: dict | None, out=print) -> None:
    for name, ratios in ((region_metrics or {}).get("ratios") or {}).items():
        values = region_metrics["regions"][name]
        rest = region_metrics["rest"]
        out(f"{run_id}  region {name} vs rest:")
        for measure in MEASURES:
            ratio = ratios[measure]
            out(f"    {measure:<22} {values[measure]:>9.4f}  rest {rest[measure]:>9.4f}  "
                f"ratio {'n/a' if ratio is None else format(ratio, '.3f')}")


def try_regions(benchmark: str, run_id: str, regions: AnalysisRegions | None, out=print) -> int:
    """Measure one run with the given (or the definition's) regions and print
    the result. Nothing is stored."""
    root = project_root()
    store = VideoMetricsStore(metrics_dir(root), BenchmarkLogger(results_dir(root)))
    try:
        regions = regions or definition_regions(root, benchmark)
        if regions is None:
            out("no regions: give --region / --region-size, or declare analysis_regions in the definition")
            return 1
        run = store.measurable_run(benchmark, run_id)
        video = resolve_path(run.output_video, root)
        if not video.is_file():
            out(f"video not on this machine: {run.output_video}")
            return 1
        record = measure_video(video, benchmark_id=benchmark, run_id=run_id, regions=regions)
    except (ValueError, FileNotFoundError, FrameDecodeError) as exc:
        out(f"ERROR: {exc}")
        return 1
    size = record.region_metrics["analysis_size"]
    out(f"benchmark {benchmark}, run {run_id}: region metrics at {size['width']}x{size['height']} (not stored)")
    print_regions(run_id, record.region_metrics, out)
    return 0


def measure_runs(benchmark: str, run_id: str | None = None, measure_all: bool = False, out=print) -> int:
    """Returns 0 when every selected run has metrics afterwards, else 1."""
    root = project_root()
    logger = BenchmarkLogger(results_dir(root))
    store = VideoMetricsStore(metrics_dir(root), logger)
    regions = definition_regions(root, benchmark)
    runs = logger.list_runs(benchmark)
    if not runs:
        out(f"no runs recorded for benchmark {benchmark!r}")
        return 1
    if run_id is not None:
        runs = [run for run in runs if run.run_id == run_id]
        if not runs:
            out(f"no run {run_id!r} recorded for benchmark {benchmark!r}")
            return 1
    elif not measure_all:
        runs = [max(runs, key=lambda run: (run.timestamp, run.run_id))]

    rows, missing, regional = [], 0, []
    for run in runs:
        record, note = None, ""
        try:
            store.measurable_run(benchmark, run.run_id)
            if store.exists(benchmark, run.run_id):
                record = store.load(benchmark, run.run_id)
                video = resolve_path(run.output_video, root)
                if not video.is_file():
                    note = "already measured"
                elif store.video_matches(record, video):
                    note = "already measured"
                else:
                    note = "already measured; WARNING: the video file has changed since"
            else:
                record = store.measure_run(benchmark, run.run_id, root, regions=regions)
                note = "measured" if record.metrics["start_frame_psnr"] is not None else "measured (no start frame)"
        except ValueError as exc:
            note = f"skipped: {str(exc).split(';')[0]}" if run.status == "success" else "skipped: failed run"
        except FileNotFoundError:
            note = "skipped: video not on this machine"
            missing += 1
        except FrameDecodeError as exc:
            note = f"error: {exc}"
            missing += 1
        rows.append(table_row(run, record, note))
        regional.append((run.run_id, record.region_metrics if record else None))

    out(f"benchmark {benchmark}, metrics v{METRICS_VERSION} (0-255 luma scale; psnr in dB)")
    print_table(rows, out)
    for measured_run, region_metrics in regional:
        print_regions(measured_run, region_metrics, out)
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", default=None, help="default: the newest run")
    group.add_argument("--all", action="store_true", help="every run of the benchmark")
    parser.add_argument("--region", action="append", default=[], metavar="NAME=x0,y0,x1,y1")
    parser.add_argument("--region-size", default=None, metavar="WxH")
    parser.add_argument("--print-only", action="store_true",
                        help="measure the regions of one run and print them; store nothing")
    args = parser.parse_args(argv)
    try:
        regions = parse_regions(args.region, args.region_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.print_only:
        if not args.run_id:
            parser.error("--print-only needs --run-id")
        return try_regions(args.benchmark, args.run_id, regions)
    if regions is not None:
        parser.error("--region is for trying regions: add --print-only (stored metrics use the definition's regions)")
    return measure_runs(args.benchmark, args.run_id, args.all)


if __name__ == "__main__":
    sys.exit(main())
