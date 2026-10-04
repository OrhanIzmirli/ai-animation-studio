"""Compute the automatic quality metrics for benchmark runs and print a table.

    python scripts/measure_run.py --benchmark B              (newest run)
    python scripts/measure_run.py --benchmark B --run-id R
    python scripts/measure_run.py --benchmark B --all

Only successful runs whose video is on this machine are measured. Results go
to benchmarks/metrics/<B>/<run_id>/metrics_v<version>.json and are never
overwritten: runs that already have metrics are shown, not measured again.
Needs ffmpeg and ffprobe. See docs/QUALITY_METRICS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.benchmarking import BenchmarkLogger  # noqa: E402
from ai_studio.paths import metrics_dir, project_root, resolve_path, results_dir  # noqa: E402
from ai_studio.quality.frames import FrameDecodeError  # noqa: E402
from ai_studio.quality.metrics import METRICS_VERSION  # noqa: E402
from ai_studio.quality.store import VideoMetricsStore  # noqa: E402

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


def measure_runs(benchmark: str, run_id: str | None = None, measure_all: bool = False, out=print) -> int:
    """Returns 0 when every selected run has metrics afterwards, else 1."""
    root = project_root()
    logger = BenchmarkLogger(results_dir(root))
    store = VideoMetricsStore(metrics_dir(root), logger)
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

    rows, missing = [], 0
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
                record = store.measure_run(benchmark, run.run_id, root)
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

    out(f"benchmark {benchmark}, metrics v{METRICS_VERSION} (0-255 luma scale; psnr in dB)")
    print_table(rows, out)
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", default=None, help="default: the newest run")
    group.add_argument("--all", action="store_true", help="every run of the benchmark")
    args = parser.parse_args(argv)
    return measure_runs(args.benchmark, args.run_id, args.all)


if __name__ == "__main__":
    sys.exit(main())
