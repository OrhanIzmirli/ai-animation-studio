# AI Animation Studio

An **experimental hobby project**: can a single laptop (RTX 4060, 8 GB VRAM)
make a short, watchable animated teaser with open models? The repository is a
lab notebook as much as a codebase: every run, setting and failure is
recorded, and the documents under `docs/` describe what was tried and what
came out of it.

The current result is a 25-second, 1920×1056 teaser starring **Milo**, a small
friendly ghost, in a 1980s cel-cartoon look.

## What was tried

1. **A benchmark harness for video models.** Fixed benchmark definitions
   (YAML), a runner that records every run as JSON (seed, steps, model
   versions, machine), ffprobe checks on every output, and automatic video
   metrics ([docs/PHASE_1_BENCHMARK.md](docs/PHASE_1_BENCHMARK.md),
   [docs/QUALITY_METRICS.md](docs/QUALITY_METRICS.md)).
2. **Image-to-video with Wan 2.2 TI2V-5B**, first on a cloud GPU, then
   locally through ComfyUI with the Turbo (distilled, GGUF) model
   ([docs/BACKEND_WAN_TI2V_5B_TURBO_COMFYUI.md](docs/BACKEND_WAN_TI2V_5B_TURBO_COMFYUI.md),
   [docs/experiments/E1_REPORT.md](docs/experiments/E1_REPORT.md)).
3. **Locked, reproducible keyframes with SDXL**: a style LoRA, IP-Adapter
   with a region mask for character identity, and a "style lock" file that
   pins every setting and prompt block, so that an approved image can be
   regenerated pixel for pixel
   ([configs/style_locks/](configs/style_locks/)).
4. **A deterministic style layer**: timing on twos, palette, ink lines,
   halftone and grain as plain post-processing
   ([docs/STYLE_LAYER.md](docs/STYLE_LAYER.md)).
5. **Code-driven cut-out animation**: the final approach (below).

## What was learned

**Why Wan was dropped.** On this hardware, Wan 2.2 Turbo clips looked
promising on a contact sheet but did not hold up in motion:

- the character changed from shot to shot; extra legs and hands appeared;
- colours drifted (saturation rises after the first frame) and the face was
  rendered in a different style from the rest of the picture;
- the camera drifted even when the prompt asked for a static shot. The Turbo
  model runs at CFG 1, so camera wording in the prompt is a very weak lever
  ([docs/experiments/L1_CAMERA_CONTROL.md](docs/experiments/L1_CAMERA_CONTROL.md));
- vague words become objects (a "mist" turned into a pink balloon);
- RAM, not VRAM, was the real limit: the text encoder and the model both pass
  through system memory.

The automatic metrics caught camera instability and global colour shifts,
but none of the problems that actually ruined the clips (malformed hands,
inconsistent faces). They measure change between frames, not whether a frame
is good ([docs/experiments/L0_CALIBRATION.md](docs/experiments/L0_CALIBRATION.md)).

**The switch to code-based animation.** In the final pipeline, generative
models only make **still images**; all **motion** is written in code, in the
spirit of 1980s cel animation (painted background plus a separate character
layer):

```
SDXL + style LoRA      → one character drawing (front view) + painted backgrounds
Real-ESRGAN            → 4× upscale of every plate
Depth-Anything-V2      → depth map per background (2.5D parallax camera)
ai_studio.puppet       → Milo as a puppet: body and two arm layers, a face drawn
                         every frame, cloth-like bend, squash and stretch
make_milo_teaser*.py   → acting, timing and camera in world coordinates;
                         cel shading, motion blur, bloom, film grain
Kokoro-82M + numpy     → narration, synthesized sound effects, music and ambience
```

The character never breaks, because it is always the same drawing, and the
motion does what was planned. That is where the quality jump came from. The
full write-up (in Turkish) is in [docs/MILO_PIPELINE.md](docs/MILO_PIPELINE.md).

Limits: Milo exists only as a front view (no profile, no turn-around, no lip
sync), and the music is simple synthesis.

## Layout

```
src/ai_studio/
  benchmarking/   result schema, logger, environment capture, benchmark runner
  generation/     backend contracts and adapters (Wan 2.2, ComfyUI, fake)
  keyframes/      style locks and locked SDXL keyframe generation via ComfyUI
  puppet.py       cut-out puppet rig for the code-driven animation
  post/           style layer: timing, palette, ink, halftone, grain (CPU only)
  quality/        automatic video metrics
  story/          story / character bible and shot plan (skeleton)
  trailer/        trailer shot lists and assembly
benchmarks/       definitions, run results, metrics, human ratings, provenance
configs/          configuration and style locks
trailers/         shot lists for the trailers (milo_01 to milo_03)
scripts/          command-line tools (benchmarks, keyframes, teaser, audio)
spikes/           short feasibility experiments and their reports
docs/             design notes, experiment reports, research
tests/            unit tests
```

Model weights and generated media (`outputs/`, videos, reference images) are
not in the repository.

## Setup

Requires Python 3.11. Tests that use the real ffmpeg/ffprobe are skipped
when they are not on `PATH`.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[dev]"
pytest
```

Image generation needs a local ComfyUI install (see
`configs/comfyui.example.yaml`) and the model files listed in
[docs/MILO_PIPELINE.md](docs/MILO_PIPELINE.md).

## Ground rules the project kept

- Every experiment is reproducible; model parameters are always logged.
- Results are never overwritten; inputs are never modified.
- A change counts as an improvement only when it beats the previous version
  on the same benchmark.

## License

Apache 2.0, see [LICENSE](LICENSE). Third-party models used by the pipeline
keep their own licences.
