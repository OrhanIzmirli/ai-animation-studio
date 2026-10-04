# AI Animation Studio

A long-term project to build a **self-correcting AI production pipeline** for
animation and video, eventually capable of producing high-quality anime
trailers and episodes.

## Long-term architecture

```
Input / Creative Intent
  → Generation
  → Quality Evaluation
  → Error Diagnosis
  → Local Repair
  → Re-evaluation
  → Final Output
```

Instead of regenerating a whole clip when something is wrong, the pipeline
evaluates the output, diagnoses what failed and where, repairs only that part,
and evaluates again. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/ROADMAP.md](docs/ROADMAP.md).

## Current status: Phase 1C — Measured benchmark outputs

The benchmark infrastructure (1A: backend interface, runner, YAML definitions,
human evaluation schema) covers video editing and video generation (1B). In 1C
the runner measures every output video with ffprobe and enforces a
benchmark's pinned fps and duration instead of trusting what the backend
reports. The first real adapter, Wan 2.2 TI2V-5B (image-to-video), is
implemented but not yet run on a GPU
([docs/BACKEND_WAN_TI2V_5B.md](docs/BACKEND_WAN_TI2V_5B.md)); the editing
candidates are still placeholders.
See [docs/PHASE_1_BENCHMARK.md](docs/PHASE_1_BENCHMARK.md).

> No modification is considered an improvement until it beats the previous
> version on the fixed benchmark suite.

## Phase 0 — Engineering foundation

Phase 0 contains **no AI models**. It provides:

- the project layout for the future pipeline stages (`src/ai_studio/`)
- a benchmark result schema and a JSON logger (`src/ai_studio/benchmarking/`)
- placeholder quality configuration (`configs/quality.yaml`, no thresholds yet)
- unit tests

## Layout

```
src/ai_studio/
  generation/     capability contracts, requests, placeholder adapters
  segmentation/   masks / regions to preserve or repair (future)
  tracking/       tracking regions across frames (future)
  quality/        modular quality evaluators (future)
  repair/         local repair (future)
  pipeline/       orchestration of the stages (future)
  media_probe.py  measures output videos with ffprobe
  benchmarking/   schemas, logger, environment capture, runner (Phase 0-1C)
benchmarks/
  definitions/    fixed benchmark definitions (YAML)
  inputs/         original input files — never modified
  references/     reference assets (character sheets, style refs, ...)
  results/        one JSON file per run (success or failure) — never overwritten
  evaluations/    human ratings (editing), one file per evaluator and revision
  comic_evaluations/  experimental comic ratings (generation)
configs/          configuration (quality thresholds, later backends)
scripts/          command-line entry points (future)
tests/            unit tests
docs/             architecture and roadmap
outputs/          generated media (git-ignored)
```

## Setup

Requires Python 3.11. Benchmarks that pin fps or duration also need `ffprobe`
(FFmpeg) on `PATH`; the tests that use the real ffmpeg/ffprobe are skipped
when it is missing.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[dev]"
pytest
```

## Logging a benchmark run

```python
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, Resolution, new_run_id

logger = BenchmarkLogger("benchmarks/results")
logger.log(BenchmarkResult(
    benchmark_id="phase1-baseline",
    run_id=new_run_id(),
    model="some-model", model_version="1.0", workflow_version="wf-1",
    seed=42, inference_steps=30, guidance=5.0,
    model_parameters={"sampler": "euler", "scheduler": "karras", "denoise_strength": 1.0},
    environment={"os": "Windows 11", "python_version": "3.11.9", "git_commit": "abc1234"},
    prompt="a girl running through a field",
    resolution=Resolution(width=1280, height=720), fps=24, frame_count=81,
))
# -> benchmarks/results/phase1-baseline/<run_id>.json
```

Generic fields (`seed`, `guidance`, `gpu`, ...) are for querying across
backends. Backend-specific settings go in `model_parameters`; the machine and
software versions go in `environment`. Neither dict has a fixed shape, so no
single model's fields are hardcoded into the schema.

Relative paths anywhere in the project are relative to the project root, not
the current directory (override with `AI_STUDIO_ROOT`).

## Engineering rules

- Generation backends must be replaceable; quality evaluators must be modular.
- Every experiment must be reproducible; model parameters are always logged.
- Benchmark results are never overwritten. Original input files are never modified.
- Quality is more important than generation speed.
- One architectural change at a time, and benchmark it.
- No premature UI, agents or over-engineering.
