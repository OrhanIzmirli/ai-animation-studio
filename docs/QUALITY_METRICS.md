# Automatic video quality metrics (version 1)

The first automatic, model-agnostic measurements of a generated video: the
seed of the Quality Gate. They look only at decoded frames, never at the
model that made them.

> **These are measurements, not judgements.** There are no thresholds and no
> pass/fail. `configs/quality.yaml` still has no thresholds: they will be set
> from benchmark data and human ratings, never guessed. One smoke run has been
> measured so far, which is far too little to say what a good value is.

Code: `src/ai_studio/quality/` (`frames.py` decoding, `metrics.py` pure
functions, `store.py` measuring and storing). Command:
`scripts/measure_run.py`.

## How a video is measured

1. Every frame is decoded with ffmpeg and scaled to the **analysis size**:
   width 320, height from the aspect ratio, both even (1280×704 → 320×176),
   area scaling, 8-bit grayscale (luma). The small size makes the numbers
   cheap, comparable between runs and less sensitive to pixel-level encoder
   noise.
2. For each **transition** (frame i → frame i+1) the mean absolute difference
   is computed. All 0–255.
3. The metrics below are derived from the frames and those transition values.

All values are on the **0–255 luma scale** unless a unit is given.

## Metrics

| Metric | Definition | Unit | What it detects |
|---|---|---|---|
| `motion_mean` | mean of the transition differences | 0–255 | a frozen or almost static video (near 0) |
| `motion_max` | largest transition difference | 0–255 | a sudden jump: cut, glitch frame, flash (far above `motion_mean`) |
| `total_change` | mean \|last frame − first frame\| | 0–255 | whether the clip ends somewhere other than where it started |
| `start_frame_psnr` | PSNR between frame 0 and the run's derived start frame, scaled to the analysis size | dB, higher = closer; 100 = identical | a backend that ignores, shifts, recolours or distorts the start frame |
| `temporal_pulse_ratio`, `temporal_pulse_phase_means`, `temporal_pulse_period` | transitions grouped by phase `i % P` (P = 4) and averaged per phase; ratio = highest / lowest phase mean | ratio, 1.0 = no pulse | flicker at regular chunk boundaries |
| `luminance_flicker` | standard deviation of the per-frame mean luminance after removing its straight-line trend | 0–255 | brightness pumping; a steady fade does not count |
| `held_frame_ratio` | fraction of transitions with a difference below 0.3 | 0–1 | **not a defect**: tells deliberate holds from a freeze |
| `luminance_min`, `luminance_max` | lowest / highest per-frame mean luminance | 0–255 | context for `luminance_flicker` |
| `frame_count` | frames decoded | frames | context |

The record also keeps `per_transition_diffs`, the full list of transition
differences, so that new questions can be asked of old runs without the
video.

### Limits of each metric

- **`motion_mean` / `motion_max`** cannot tell wanted motion from unwanted
  change. Flicker, noise and boiling lines raise them just like real motion.
  A low value means "little changes", not "bad".
- **`total_change`** is low for a loop or a camera move that returns, even if
  a lot happened in between.
- **`start_frame_psnr`** also drops with encoding loss, and it compares
  against the *derived* start frame (after the backend's resize and crop), so
  it does not judge that preprocessing. It is `null` when the run kept no
  derived start frame. PSNR is a pixel measure: a small shift costs a lot, a
  plausible but different face costs little.
- **`temporal_pulse_ratio`** only looks for one period (4 frames; phase 0 is
  the transitions 0→1, 4→5, ...). Wan-family VAEs decode 4 frames per latent
  frame, which is why 4 is the default; other models may need another period.
  Animation held "on twos" or "on fours" is periodic on purpose and raises
  the ratio too: read it together with `held_frame_ratio`. A phase with no
  change at all is counted as 0.01, so the ratio stays finite. It is `null`
  for videos with fewer than 4 transitions.
- **`luminance_flicker`** treats any non-linear brightness change as flicker,
  including intended ones (a flash, a light switching on, a sunset that
  accelerates). It is global: local flicker that keeps the mean constant is
  invisible to it.
- **`held_frame_ratio`** is reported, not judged. Stepped / limited animation
  holds drawings on purpose (the comic benchmark explicitly allows it). A
  frozen video has a ratio near 1 *and* `motion_mean` near 0; stepped
  animation has regular holds with clear motion between them. The threshold
  0.3 is a first choice, not calibrated.
- Everything is measured on grayscale at 320 px width: colour shifts with
  constant luma, and fine detail such as thin line art, are not seen.

## First data point: the Wan 2.2 smoke run

Run `comic_rooftop_01 / 20261004T145422Z-a264ad6e`: Wan 2.2 TI2V-5B on an L4,
**2 denoising steps** (a smoke test, not for evaluation), 1280×704, 121
frames, 24 fps.

| Metric | By hand, before this code existed | `measure_run.py`, metrics v1 |
|---|---|---|
| `motion_mean` | 0.68 | 0.689 |
| `motion_max` | – | 1.286 |
| `total_change` | 3.99 | 4.150 |
| pulse phase means (i % 4 = 0, 1, 2, 3) | 0.97 / 0.70 / 0.62 / 0.42 | 1.011 / 0.701 / 0.624 / 0.420 |
| `temporal_pulse_ratio` | 2.31 (from the values above) | 2.406 |
| luminance range | 55.0 .. 57.6 | 55.02 .. 57.55 |
| `luminance_flicker` | – | 0.250 |
| `held_frame_ratio` | – | 0.058 |
| `start_frame_psnr` | – | 41.66 dB |

The two columns agree; the small differences come from the scaler used for
the downscale. What the numbers say about this one video:

- It is **almost static**: neighbouring frames differ by about 0.7 out of
  255, and the last frame differs from the first by about 4.
- There is a **clear 4-frame pulse**: the transitions at the VAE's chunk
  boundaries (phase 0) change about 2.4 times as much as the quietest phase.
  That is periodic micro-flicker.
- Frame 0 matches the start frame closely (41.7 dB).

What they do not say: whether this is typical. It is a single 2-step run.
Whether a full 50-step run moves more, or pulses less, is exactly what the
next runs will show. No conclusion about the model should be drawn from it.

## Storage

Run records are immutable, so metrics are stored beside them, like the human
evaluations:

```
benchmarks/metrics/<benchmark_id>/<run_id>/metrics_v<metrics_version>.json
```

```
benchmark_id, run_id, metrics_version, analysis_size {width, height},
video_sha256, metrics {...}, per_transition_diffs [...],
tool_versions {numpy, ffmpeg}, timestamp
```

- **Never overwritten.** The file is created exclusively; measuring a run
  that already has metrics of the current version is refused.
- **Versioned.** `METRICS_VERSION` (now `"1"`) changes whenever a definition,
  a default or the decoding changes. A new version writes a new file next to
  the old one; numbers from different versions must not be compared.
- **Only successful runs with an output video** can be measured.
- **`video_sha256`** is the hash of the file that was measured. If the video
  under that run id is later a different file, `measure_run.py` says so.
- Numbers can differ in the last digits between ffmpeg versions (scaler,
  decoder), which is why `tool_versions` is recorded.

## Measuring runs

```
python scripts/measure_run.py --benchmark comic_rooftop_01              # newest run
python scripts/measure_run.py --benchmark comic_rooftop_01 --run-id R
python scripts/measure_run.py --benchmark comic_rooftop_01 --all
```

Needs ffmpeg and ffprobe, and the run's video on this machine (import it with
`scripts/import_run.py` first). The run's `start_frame.derived_file` is used
for `start_frame_psnr`. Output:

```
benchmark comic_rooftop_01, metrics v1 (0-255 luma scale; psnr in dB)
run                        steps  motion_mean  total_change  start_psnr  pulse_ratio  flicker  held_ratio  note
20261004T145422Z-a264ad6e  2      0.689        4.150         41.66       2.41         0.250    0.058       measured
```

Runs that already have metrics are shown with `already measured` and not
measured again; failed runs and runs whose video is not on this machine are
listed as skipped. Commit the new `benchmarks/metrics/...json` files like run
records.

## Not yet

- thresholds, pass/fail, any combined score
- anything semantic: identity, line stability, style, artifacts
- locating a defect in space (region) rather than only in time
- use by the runner: metrics are computed after the fact, on demand
