# Architecture

> Status: the capability contracts and the benchmark infrastructure are
> implemented; the production pipeline (router, Quality Gate, repair) is not.

## Studio engine: capabilities

```
STUDIO ENGINE
│
├── VIDEO GENERATION     Text / Image / References          → New Video
│     contract: VideoGenerationBackend    request: VideoGenerationRequest
│
├── VIDEO EDITING        Source Video / References / Instruction → Edited Video
│     contract: VideoEditingBackend       request: BenchmarkRequest (EditRequest later)
│
└── VIDEO REPAIR         Video / Frames / Region             → Repaired Video
      (future — not implemented)
```

Each capability has its own request type and backend contract
(`src/ai_studio/generation/`). A backend declares its capabilities by the
contracts it implements. A model that naturally does both generation and
editing implements both contracts; a model that only edits implements only
the editing contract.

**Models are selected by capability and by benchmark performance.** We do not
force a model into a universal interface it does not naturally support, e.g.
faking a source video to make a text-to-video model look like an editor, or
feeding a blank prompt to an editor to make it "generate". The best
generation model and the best editing model may be different models. Each is
chosen from its own benchmark results.

What stays common to all capabilities is the infrastructure around the model.
One `BenchmarkRunner` handles every capability: input validation, checksums,
timing, environment capture, output verification, failure recording and
logging are shared. It checks that the backend declares the request's
capability, and it never looks at model-specific details.

Adding `VIDEO_REPAIR` later means adding a `Capability` member, a request type
and a `VideoRepairBackend` contract. The runner and record schema already
carry `task_type`.

## Backends

| Backend | Capability | State |
|---|---|---|
| `WanTI2V5BBackend` (`wan_ti2v.py`) | generation, image-to-video (start frame) | implemented; needs the `wan` extra and a CUDA GPU. See [BACKEND_WAN_TI2V_5B.md](BACKEND_WAN_TI2V_5B.md) |
| `WanTurboComfyUIBackend` (`wan_turbo_comfyui.py`) | generation, image-to-video (start frame) | implemented; a different model (Turbo finetune, GGUF) run through a local ComfyUI that the adapter starts and stops. Works on an 8 GB GPU. See [BACKEND_WAN_TI2V_5B_TURBO_COMFYUI.md](BACKEND_WAN_TI2V_5B_TURBO_COMFYUI.md) |
| `JoyAIVideoEditBackend` (`joyai.py`) | editing | placeholder, not installed |
| `KiwiEditBackend` (`kiwi.py`) | editing | placeholder, not installed |
| `WanEditCtrlBackend` (`wan_editctrl.py`) | editing | placeholder, not installed |

An adapter imports its heavy dependencies only when it is prepared, so the
package and the test suite work on a machine without any model installed.
An adapter may also drive an external program instead: the ComfyUI adapter
speaks HTTP to a ComfyUI installed outside the repository, whose location
comes from a git-ignored local config file (`configs/comfyui.local.yaml`).

Names are mapped to adapter classes in one place,
`src/ai_studio/generation/backends/registry.py` (`BACKENDS`, `get_backend`).
Launchers use it: the command line (`src/ai_studio/cli.py`,
`scripts/run_benchmark.py`) resolves `--backend` to a class and hands the
runner an instance. The runner never imports the registry.

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
| Generate / Edit | `src/ai_studio/generation` | Produce or edit a clip. Backends are replaceable behind one contract per capability. |
| Segment / Track | `src/ai_studio/segmentation`, `src/ai_studio/tracking` | Locate regions across frames: what to preserve, what to repair. |
| Quality Gate | `src/ai_studio/quality` | Run modular evaluators, compare metrics to `configs/quality.yaml`, return pass/fail per check. **So far:** the first automatic metrics only (motion, start-frame PSNR, temporal pulse, luminance flicker, held frames), measured on demand and stored beside the run records; no thresholds, no pass/fail. See [QUALITY_METRICS.md](QUALITY_METRICS.md). |
| Diagnose | `src/ai_studio/quality` / `src/ai_studio/repair` | Turn failed checks into located defects (which frames, which region, what kind). |
| Local repair | `src/ai_studio/repair` | Fix only the located defect; everything outside it is preserved. |
| Orchestration | `src/ai_studio/pipeline` | Wire the stages together and log every run. |
| Benchmarking | `src/ai_studio/benchmarking` | Record every run's parameters, cost and metrics as immutable JSON. |

## Requests: benchmark layer vs. production layer

**Generation** already follows the target design: `VideoGenerationRequest`
(`src/ai_studio/generation/request.py`) contains no benchmark metadata. The
runner receives `benchmark_id` separately, so the same request type can be
used by the production pipeline.

**Editing** still uses `BenchmarkRequest` (`src/ai_studio/benchmarking/request.py`),
which is a **benchmark-layer object**. It combines the edit itself (source video,
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
