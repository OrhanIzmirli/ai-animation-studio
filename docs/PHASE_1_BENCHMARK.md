# Phase 1 — Multi-Backend Video Benchmark

> ## Engineering rule
>
> **No modification is considered an improvement until it beats the previous
> version on the fixed benchmark suite.**

Phase 1A built the infrastructure: backend interface, request/result
schemas, environment capture, runner, YAML benchmark definitions and the
human evaluation schema. Phase 1B adds a second capability, **video
generation**, beside video editing. Phase 1C makes the runner **measure** the
output video instead of trusting the backend's reported fps / frame count. No
model is installed yet.

## Two capabilities, benchmarked separately

| `task_type` | Input | Request | Backend contract |
|---|---|---|---|
| `video_editing` | source video + references + instruction | `BenchmarkRequest` | `VideoEditingBackend` |
| `video_generation` | prompt + references + shot controls, **no source video** | `VideoGenerationRequest` | `VideoGenerationBackend` |

Every benchmark definition declares its `task_type` explicitly. The two
capabilities are ranked **separately**: which open models are best at
generation is a different question from which are best at editing, and
neither is assumed from the other. The editing candidates (JoyAI-Video-Edit,
Kiwi-Edit, Wan 2.2 VACE-Fun + EditCtrl) are declared editing-only until
verified. Generation candidates are still to be researched.

### Image-to-video: `start_frame` vs `reference_images`

- **`start_frame`** is the exact image that must condition **frame 0**.
- **`reference_images`** are additional identity, style or subject references.

The two are **not interchangeable**, and the same file may not appear in
both. A request with a `start_frame` only runs on a backend that declares
`supports_start_frame`. That backend must return `StartFrameConditioning`,
listing every preprocessing step and the effective dimensions passed to the
model, plus the path of any derived copy, which lives in the run's output
directory.

A backend that silently treats the start frame as a plain reference fails
verification. The run record's `start_frame` block keeps the path, SHA-256,
original and effective dimensions, preprocessing and derived file. The
canonical image itself is never modified; the input checksum check enforces
this.

Generation definitions may pin `fps`. A run-time `fps` that differs from the
pinned value is rejected. They may also pin `duration_tolerance_seconds`, the
largest allowed difference between the measured duration and
`duration_seconds`; it has no run-time override. Both are enforced on the
measured output, see [Output verification](#output-verification-measured-not-reported).

Generation definitions carry structured shot controls: `duration_seconds`,
`camera_motion`, `subject_motion`, `visual_direction`. Each control is a
generic string, list or mapping, never a model-specific field. They are
recorded with every run in `task_parameters`, together with `requested_fps`
and `duration_tolerance_seconds`. Run a generation definition
with:

```python
request = definition.to_request(seed=..., output_resolution=..., fps=...)
runner.run(request, backend, benchmark_id=definition.benchmark_id)
```

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

**Lifecycle.** Every definition has `status: draft` (the default) or
`status: ready`. `BenchmarkRunner.run_definition()` refuses drafts before
anything runs: no backend call, no output directory, no record.
`to_request()` also refuses them. Changing `draft` to `ready` is an explicit
human edit of the YAML file, made once the inputs are final. Readiness is
never inferred from whether files exist. Definitions written before this
field existed load as drafts.

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
resolution, the measured output video (`output_media`; `fps` / `frame_count`
are the measured values, or the backend-reported ones when nothing could be
measured), all effective
backend parameters (requested values plus defaults the backend applied; any
value the backend changed is noted), environment (OS, Python, git commit and
dirty flag, GPU, VRAM, driver, CUDA, PyTorch, backend library versions, and
`env_vars`: the process environment variables `PYTORCH_CUDA_ALLOC_CONF`,
`CUDA_VISIBLE_DEVICES` and `HF_HUB_OFFLINE`, `null` when unset),
a per-stage timing breakdown, estimated cost when supplied, the run
status (success or failed, with failure stage and error), and the generated
`output_video` (project-relative, forward slashes; required for successful
runs, `null` for failures that produced no usable output).

Results are written to `benchmarks/results/<benchmark_id>/<run_id>.json`;
existing files are never overwritten, and each run gets its own output
directory `outputs/<benchmark_id>/<run_id>/`.

### Output verification: measured, not reported

After a backend returns, the verification stage checks that the output is a
file inside the run's own output directory, that no input file was modified,
and that start-frame conditioning was reported. It then **measures** the
output video (`src/ai_studio/media_probe.py`):

```
ffprobe -v error -select_streams v:0 -count_frames
  -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames -of json
```

`frame_count` is the number of frames actually decoded, `fps` comes from
`avg_frame_rate`, and the duration is `frame_count / fps`, not the container
duration.

| Check | Result |
|---|---|
| ffprobe not installed, and the request pins `fps` or a `duration_tolerance_seconds` | **fail** ("cannot verify") |
| ffprobe not installed, nothing pinned | warning in `notes`, `output_media: null`, backend-reported `fps` / `frame_count` kept |
| backend-reported fps differs from measured by more than 0.01 | **fail** |
| backend-reported frame count differs from measured | **fail** |
| requested fps differs from measured by more than 0.01 (23.976 is not 24) | **fail** |
| `duration_seconds` and `duration_tolerance_seconds` set, and the measured duration is further away than the tolerance | **fail**; the message gives measured seconds, frames and fps |
| measured resolution differs from `output_resolution` | warning in `notes` only |
| ffprobe runs but cannot read the file | **fail** |

A failure here is a normal failed run with `failure_stage: verification`. On
success the record's `output_media` holds
`{probe, width, height, fps, frame_count, duration_seconds, codec}`, and the
`fps` / `frame_count` columns hold the measured values. Runner warnings are
written to `notes` as `warning: ...`. The checks use only generic request
fields, so the runner stays model-agnostic; tests inject a fake probe through
`BenchmarkRunner(media_probe=...)`.

ffprobe (part of FFmpeg) must be on `PATH` to run any benchmark that pins fps
or a duration tolerance.

### Failed runs are benchmark data

A backend that runs out of VRAM, fails to load, crashes during inference,
times out or produces an invalid output tells us something about that
backend. The runner therefore records failures too:

| `failure_stage` | What failed |
|---|---|
| `validation` | missing input file, or the backend's `validate_inputs()` rejected the request |
| `prepare` | model load (corrupt checkpoint, not enough memory to load, ...) |
| `generation` | inference (CUDA OOM, exception, timeout raised by the backend) |
| `verification` | output missing / outside its run directory / wrong type, an input file was modified, or the measured video contradicts the backend's report or the pinned fps / duration |
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
| `verification_time_seconds` | output checks, input re-checksum, measuring the output video |
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
| 6 | + `task_type` (required), `task_parameters` | `task_type: null` (unknown), `task_parameters: {}` |
| 7 | + `start_frame` (image-to-video reproducibility block) | `start_frame: null` |
| 8 | + `output_media` (measured output video); `fps`/`frame_count` are measured values when a measurement exists | `output_media: null`; `fps`/`frame_count` stay backend-reported |

v5 migration: no action is needed. From v5 on, a successful record **must**
have `output_video`; the rule is checked only for records with
`schema_version >= 5`, so v1–v4 successes without it remain valid. For older
runs, the video can still be found under `outputs/<benchmark_id>/<run_id>/`.

v6 migration: no action is needed. Generation runs have no source video and
carry shot controls, so a record must say which task it was and keep those
controls to be reproducible. `task_type` is required for records with
`schema_version >= 6`. Older records keep `task_type: null`. v3–v5 runner
records were all editing runs; v1–v2 were logged by hand.

v8 migration: no action is needed. `output_media` is allowed only on
successful records and must have exactly the keys in
`ai_studio.media_probe.OUTPUT_MEDIA_KEYS`. In v1–v7 records `fps` and
`frame_count` are whatever the backend reported and were never checked.

Frozen example records for every version live in `tests/fixtures/schema/`
and must keep loading. Never edit them; add a new fixture for a new version.

## Reference assets are external dependencies

Benchmark media (source videos, reference images, character sheets) is
git-ignored. It is an **external reproducibility dependency** of every run:
the repository holds the definition and the run records, and each record
holds a SHA-256 checksum of every input file. Anyone reproducing a run must
obtain byte-identical files and can verify them against those checksums.

No Git LFS or cloud storage is used yet. Shared object storage for benchmark
media will be introduced before multi-machine benchmark reproduction becomes
necessary (e.g. before the first rented-GPU run that must be compared with a
local run). Until then, never edit or replace a media file that a run has
used. Add a new file under a new name instead.

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
   the contract(s) the model **naturally** supports: `VideoGenerationBackend`,
   `VideoEditingBackend`, or both, dispatching on the request type in
   `generate()`. Implement `metadata()`, `validate_inputs()`, `prepare()` and
   `generate()`. Capabilities follow from the contracts; never claim one the
   model does not really have.
2. Read model-specific settings from `request.backend_parameters`; return the
   full effective parameters in `GenerationResult.parameters`.
3. Write the output inside the `output_dir` the runner provides. Input paths
   arrive absolute; never modify input files.
4. Raise exceptions freely (OOM, load errors, `TimeoutError`) — the runner
   records them. Raise `BackendNotInstalledError` when the model is not set up.

`BenchmarkRunner` does not change.
