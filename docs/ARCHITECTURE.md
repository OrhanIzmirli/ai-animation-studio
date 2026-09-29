# Architecture

> Status: design only. Nothing below is implemented except benchmark logging.

## Future pipeline

```
GENERATE
   ↓
QUALITY GATE ── PASS ──→ FINAL
   ↓ FAIL
DIAGNOSE
   ↓
LOCAL REPAIR
   ↓
QUALITY GATE   (loop back: pass → FINAL, fail → DIAGNOSE)
```

The loop needs a maximum number of repair iterations (to be decided from
benchmarks); a clip that still fails after that is flagged for human review.

## Stages

| Stage | Package | Responsibility |
|---|---|---|
| Generate | `src/ai_studio/generation` | Produce a clip from intent, prompt and references. Backends are replaceable behind one interface. |
| Segment / Track | `src/ai_studio/segmentation`, `src/ai_studio/tracking` | Locate regions across frames: what to preserve, what to repair. |
| Quality Gate | `src/ai_studio/quality` | Run modular evaluators, compare metrics to `configs/quality.yaml`, return pass/fail per check. |
| Diagnose | `src/ai_studio/quality` / `src/ai_studio/repair` | Turn failed checks into located defects (which frames, which region, what kind). |
| Local repair | `src/ai_studio/repair` | Fix only the located defect; everything outside it is preserved. |
| Orchestration | `src/ai_studio/pipeline` | Wire the stages together and log every run. |
| Benchmarking | `src/ai_studio/benchmarking` | Record every run's parameters, cost and metrics as immutable JSON. |

## Requests: benchmark layer vs. production layer

`BenchmarkRequest` (`src/ai_studio/benchmarking/request.py`) is a
**benchmark-layer object**. It combines the edit itself (source video,
references, prompt, output resolution, backend parameters) with experiment
metadata (`benchmark_id`, a fixed seed).

The future production pipeline must not depend on benchmark concepts. The
intended design, **not implemented yet**:

```
EditRequest          model-agnostic description of one edit
                     (source, references, instruction, resolution, parameters)
BenchmarkRequest     benchmark_id + seed + other experiment metadata
                     wrapping an EditRequest
VideoEditingBackend  accepts EditRequest
```

The router, Quality Gate and repair loop will pass `EditRequest`s; the
benchmark runner will unwrap its `BenchmarkRequest`. Until that refactor,
backends receive `BenchmarkRequest` and must use only its edit fields.

## Paths

Every relative path is relative to the project root (`$AI_STUDIO_ROOT` or the
directory containing `pyproject.toml`), never to the current working directory.
Records store project-relative POSIX paths. See `src/ai_studio/paths.py` and
[PHASE_1_BENCHMARK.md](PHASE_1_BENCHMARK.md#paths).

## Design principles

- **Replaceable backends.** Generation and repair models sit behind small
  interfaces so a model can be swapped and the swap benchmarked.
- **Modular evaluators.** Each quality check is independent; the gate combines
  them. Adding a check must not require changing the others.
- **Reproducibility.** Every run logs model, version, workflow version, seed
  and sampling parameters. Results are append-only and never overwritten.
- **Preserve inputs.** Original input files are read-only; derived files go to
  `outputs/`.
- **Quality over speed.**
- **One change at a time**, each measured against the benchmark.
