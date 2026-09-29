# Phase 1 — Multi-Backend Video Editing Benchmark

> ## Engineering rule
>
> **No modification is considered an improvement until it beats the previous
> version on the fixed benchmark suite.**

Phase 1A (this document's current scope) builds only the infrastructure:
backend interface, request/result schemas, environment capture, runner,
YAML benchmark definitions and the human evaluation schema. No model is
installed yet.

## Why multiple backends

We are building a model-agnostic production engine, not a wrapper around one
model. Different open video-editing models are likely good at different things
(reference fidelity, temporal stability, background preservation, ...). To
eventually route each task to the best model, and let one model repair
another's failures, we first need to measure every candidate under identical
conditions.

Initial candidates (placeholders in `src/ai_studio/generation/backends/`):

| Backend name | Model |
|---|---|
| `joyai-video-edit` | JoyAI-Video-Edit |
| `kiwi-edit` | Kiwi-Edit |
| `wan2.2-vace-fun-a14b-editctrl` | Wan 2.2 VACE-Fun A14B + EditCtrl |

## Why paper / README claims are not enough

Published results use their authors' own inputs, resolutions, prompts,
cherry-picked samples and metrics, often on hardware and settings we will not
use. They are not comparable to each other and say nothing about *our* tasks.
A claim is a hypothesis to test, not a result.

## Why no winner is chosen in advance

Picking a model before measuring would bias every later decision (which
backend gets tuning effort, which failures get investigated). The winner — or
more likely, the winner *per criterion* — is decided only from benchmark runs
and human ratings, never from reputation. The code deliberately computes no
weighted score and declares no winner; that is a human decision made from
the recorded data.

## Fixed-source methodology

A benchmark (`benchmarks/definitions/<id>.yaml`) fixes *what* is tested:

- the source video (`benchmarks/inputs/`, never modified)
- the reference image(s) (`benchmarks/references/`, never modified)
- the instruction prompt and negative prompt
- the target object and regions that must be preserved

Every backend receives exactly the same definition. The runner records a
SHA-256 checksum of every input file with each run, so it is provable that
compared runs used byte-identical inputs, and it aborts if a backend modifies
an input.

Once runs exist for a definition, the definition is frozen. A changed input or
prompt is a new benchmark (`object_replacement_02`), never an edit.

First benchmark, `object_replacement_01`:
real source video + reference image + instruction → reference-guided object
replacement.

### Variables that must remain constant across compared runs

- source video and reference images (checked by checksum)
- prompt and negative prompt
- output resolution
- seed (and, for fairness, the same set of seeds per backend when several are used)
- hardware class where possible; otherwise recorded in `environment`
- evaluation criteria, scale and evaluator instructions

What legitimately differs per backend is only `backend_parameters` — each
model's own settings — and every value is recorded in `model_parameters`.

### What every run records

backend and backend version, model and model version, workflow version, seed,
prompt, negative prompt, input and reference paths plus checksums, output
resolution, fps / frame count when the backend reports them, all effective
backend parameters (requested values plus defaults the backend applied; any
value the backend changed is noted), environment (OS, Python, git commit and
dirty flag, GPU, VRAM, driver, CUDA, PyTorch, backend library versions),
a per-stage timing breakdown, estimated cost when supplied, the run
status (success or failed, with failure stage and error), and the generated
`output_video` (project-relative, forward slashes; required for successful
runs, `null` for failures that produced no usable output).

Results are written to `benchmarks/results/<benchmark_id>/<run_id>.json`;
existing files are never overwritten, and each run gets its own output
directory `outputs/<benchmark_id>/<run_id>/`.

### Failed runs are benchmark data

A backend that runs out of VRAM, fails to load, crashes during inference,
times out or produces an invalid output tells us something about that
backend. The runner therefore records failures too:

| `failure_stage` | What failed |
|---|---|
| `validation` | missing input file, or the backend's `validate_inputs()` rejected the request |
| `prepare` | model load (corrupt checkpoint, not enough memory to load, ...) |
| `generation` | inference (CUDA OOM, exception, timeout raised by the backend) |
| `verification` | output missing / outside its run directory / wrong type, or an input file was modified |
| `logging` | the success record could not be written |

A failed record has `status: failed`, `failure_stage`, `error_type` (fully
qualified, e.g. `torch.OutOfMemoryError`, so OOMs can be counted per backend),
`error_message`, every parameter that was requested, the environment and the
timings up to the point of failure.

The exception is never swallowed: `run()` raises `BenchmarkRunError`, whose
`.stage` and `.record` describe what was saved and whose `__cause__` is the
original exception. If even the failed record cannot be written, `.record` is
`None` and a note on the original exception explains why.

Not recorded, raised directly: an existing `run_id` (`FileExistsError` — a
failure never overwrites an earlier run, successful or failed), and
`BackendNotInstalledError` (a setup problem, not a property of the model;
recording it would fill the results with noise). Timeouts are enforced by the
backend — only it can stop its own inference — and are recorded like any
other generation failure.

### Timing breakdown

| Field | Measures |
|---|---|
| `validation_time_seconds` | input checks, backend validation, input checksums |
| `prepare_time_seconds` | model load / warm-up — the **cold-start cost** |
| `generation_time_seconds` | inference only — the **actual inference cost** |
| `verification_time_seconds` | output checks, input re-checksum |
| `total_time_seconds` | whole run, excluding writing the record |

`generation_time_seconds` keeps its original meaning (and name) for
compatibility with older records. A backend's self-reported time, if any, is
kept in `notes`.

## Paths

All relative paths are relative to the **project root**, never to the
directory the shell was started in. See `src/ai_studio/paths.py`.

- The project root is `$AI_STUDIO_ROOT` if set (e.g. on a cloud machine with a
  non-editable install), otherwise the directory containing `pyproject.toml`.
- Media paths in benchmark definitions are written project-relative:
  `benchmarks/inputs/clip.mp4`.
- Directories given to `BenchmarkLogger`, `BenchmarkRunner` and
  `HumanEvaluationStore` may be relative; they are resolved against the root.
- Backends always receive **absolute** paths.
- Records store paths **project-relative in POSIX form**, so a run on Windows
  and a run on a Linux cloud GPU record the same `source_video`, and their
  `input_checksums` can be compared directly.

## Human evaluation criteria

Each generated video is rated by a person on eight criteria, integers 1–10.
**For every criterion, higher is better.**

| Criterion | 10 | 1 |
|---|---|---|
| `reference_fidelity` | matches the reference exactly | unrelated to the reference |
| `temporal_consistency` | stable, no flicker or identity drift | severe flicker/drift |
| `background_preservation` | everything outside the edit unchanged | background heavily altered |
| `geometry` | correct shape, scale, perspective | badly distorted |
| `lighting_and_shadows` | matches the scene | clearly inconsistent |
| `occlusion` | correct in-front/behind ordering | broken occlusion |
| `artifact_visibility` | **no visible distracting artifacts** | **severe obvious artifacts** |
| `overall_realism` | indistinguishable from real footage | obviously fake |

Note `artifact_visibility`: despite its name, a **high score means few
artifacts**, so that all criteria point the same direction.

Free-text notes are recorded with every evaluation. No weighted total is
computed. AI-based quality evaluation is out of scope for Phase 1.

### Storage

```
benchmarks/evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json
```

- Stored by `HumanEvaluationStore`, separately from the run records in
  `benchmarks/results/` and from any future automatic Quality Gate data. A
  run record's `quality_metrics` is reserved for automatic metrics.
- `save()` stores the first rating and refuses if this evaluator already
  rated the run. Re-rating is explicit: `save_revision()` writes `r002`,
  `r003`, ...; earlier revisions are kept. `load()` returns the latest
  revision unless one is requested.
- Given the results logger, the store only accepts ratings for runs that
  exist and succeeded.
- **`BenchmarkResult.human_rating` is deprecated.** It is kept only so v1–v4
  records still load. The runner never writes it, and all new human ratings
  must go through `HumanEvaluationStore`.

## Record schema versions and migration

Result records are immutable, so old files are never rewritten. Each file
keeps the `schema_version` it was written with, and the loader fills fields
added later with their defaults.

| Version | Change | Loading older files |
|---|---|---|
| 1 | initial | — |
| 2 | + `model_parameters`, `environment` | empty dicts |
| 3 | + `backend`, `backend_version`, `input_checksums`; `fps`/`frame_count` optional | `null` / empty |
| 4 | + `status`, `failure_stage`, `error_type`, `error_message`; per-stage timings | `status: success` (v1–v3 only recorded successes), timings `null` |
| 5 | + `output_video`; `human_rating` deprecated | `output_video: null` |

v5 migration: no action is needed. From v5 on, a successful record **must**
have `output_video`; the rule is checked only for records with
`schema_version >= 5`, so v1–v4 successes without it remain valid. For older
runs, the video can still be found under `outputs/<benchmark_id>/<run_id>/`.

Frozen example records for every version live in `tests/fixtures/schema/`
and must keep loading. Never edit them; add a new fixture for a new version.

## Upstream backends vs. our forks

Long term we will improve open-source backends ourselves: preprocessing,
inference modifications, LoRAs / fine-tuning and repair strategies. An
upstream backend and our modified fork are **separate benchmarkable versions**:

- each is its own backend adapter (or the same adapter with a distinct
  `backend_version`, e.g. `upstream-1.0` vs `fork-0.3`) with its own
  `workflow_version`;
- both run on the same fixed benchmark suite with the same seeds;
- the fork replaces upstream only if it beats it on that suite — per the
  engineering rule at the top of this document;
- the upstream results stay in the record permanently as the baseline.

## Adding a backend

1. Create `src/ai_studio/generation/backends/<name>.py` with a subclass of
   `VideoEditingBackend` implementing `metadata()`, `validate_inputs()`,
   `prepare()` and `generate()`.
2. Read model-specific settings from `request.backend_parameters`; return the
   full effective parameters in `GenerationResult.parameters`.
3. Write the output inside the `output_dir` the runner provides. Input paths
   arrive absolute; never modify input files.
4. Raise exceptions freely (OOM, load errors, `TimeoutError`) — the runner
   records them. Raise `BackendNotInstalledError` when the model is not set up.

`BenchmarkRunner` does not change.
