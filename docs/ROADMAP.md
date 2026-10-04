# Roadmap

Each phase starts only after the previous one is reviewed and explicitly
approved. Within a phase, make one architectural change at a time and
benchmark it.

## Phase 0 — Engineering foundation *(done, commit 2a592df)*
Project layout, benchmark result schema, JSON benchmark logger, placeholder
quality config, tests, documentation. No AI models.

## Phase 1 — Maximum-quality video generation benchmark *(current: 1C)*
- **1A** *(done)* — model-agnostic benchmark infrastructure (backend
  interface, runner, YAML definitions, human evaluation schema).
- **1B** *(done)* — video generation as a second capability beside editing;
  image-to-video start frames; first comic benchmark
  ([BENCHMARK_COMIC_ROOFTOP_01.md](BENCHMARK_COMIC_ROOFTOP_01.md)).
- **1C** *(current)* — the runner measures the output video (ffprobe) and
  enforces pinned fps and duration instead of trusting backend-reported
  values. First real generation adapter: Wan 2.2 TI2V-5B, image-to-video
  ([BACKEND_WAN_TI2V_5B.md](BACKEND_WAN_TI2V_5B.md)), tested with an injected
  pipeline. Command-line launcher (`scripts/run_benchmark.py`) and a
  workflow for the free Camber L4 GPU node
  ([RUNNING_ON_CAMBER.md](RUNNING_ON_CAMBER.md)). The first two runs on the
  L4 loaded the model and denoised, then failed in the VAE decode (out of
  memory, then a diffusers 0.35.2 tiling bug; fixed by pinning 0.36.0).
  Next: the first successful run; no baseline results exist yet.

See [PHASE_1_BENCHMARK.md](PHASE_1_BENCHMARK.md).

Integrate a first generation backend behind a replaceable interface. Establish
reproducible baseline runs and log every run with full parameters. Optimise
for quality, not speed.

## Phase 2 — Segmentation and preservation
Segment and track regions across frames so that parts of a video can be
preserved unchanged while other parts are edited or repaired.

## Phase 3 — Quality Gate
Modular quality evaluators producing metrics per run. Calibrate thresholds in
`configs/quality.yaml` from benchmark data and human ratings.

## Phase 4 — Automatic diagnosis and repair
When the Quality Gate fails: identify what failed and where, repair only the
affected region/frames, and re-evaluate.

## Phase 5 — Anime consistency benchmark
Benchmark character identity, line art, colour palette and style consistency
across shots.

## Phase 6 — Anime-specific Quality Gate
Anime-specific evaluators and thresholds built on the Phase 5 benchmark.

## Phase 7 — Story / Director system
Turn creative intent into a structured shot plan that drives generation.

## Phase 8 — 60–90 second production-quality trailer
End-to-end production of a full trailer through the self-correcting pipeline.

## Phase 9 — Automation / desktop application
Automate the workflow and wrap it in a desktop application.
