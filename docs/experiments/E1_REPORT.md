# Experiment E1: local Turbo seeds, latent scale, and the base model

Date: 2026-10-04 / 05. Benchmark `comic_rooftop_01`, 1280×704, 121 frames,
24 fps, on the development laptop (RTX 4060 Laptop 8 GB, 16 GB RAM) through
the local ComfyUI backends. All E1 runs have `run_purpose: experiment`.

> **Incomplete: E1-6 (the base model) did not finish.** Five of the six
> planned runs succeeded. The base-model run was killed from outside while it
> was running, so there is no base-model clip in this report. See
> [E1-6](#e1-6-base-model-seed-42-not-completed).

> **This report describes; it does not rank.** Nobody has rated these clips.
> Which one looks best is for a human to decide, with the review material
> listed at the end.

## Runs

`turbo_seed42_92e20c65` is the earlier seed-42 Turbo run, made before E1 and
included for comparison. Code for the five E1 runs: commit `52ef9a5`, clean
tree (`git_dirty: false`).

| label | backend | seed | settings | status | total s | prepare s | generation s | RAM before GB | RAM lowest GB | RAM peak used GB | motion_mean | motion_max | total_change | start_psnr dB | pulse_ratio | flicker | lum min | lum max | held_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `turbo_seed42_92e20c65` | `wan-ti2v-5b-turbo-comfyui` | 42 | 4 steps, CFG 1, uni_pc/simple, shift 8 | success | 277.9 | 43.3 | 233.3 | 7.79 | 0.21 | 15.50 | 1.684 | 3.138 | 27.638 | 41.29 | 1.20 | 0.470 | 52.0 | 54.9 | 0.000 |
| `turbo_seed1_c21a84d8` (E1-1) | `wan-ti2v-5b-turbo-comfyui` | 1 | 4 steps, CFG 1, uni_pc/simple, shift 8 | success | 321.0 | 47.1 | 272.6 | 7.43 | 0.31 | 15.40 | 1.313 | 2.692 | 25.727 | 41.31 | 1.31 | 0.383 | 51.4 | 54.7 | 0.000 |
| `turbo_seed2_0c960de6` (E1-2) | `wan-ti2v-5b-turbo-comfyui` | 2 | 4 steps, CFG 1, uni_pc/simple, shift 8 | success | 321.6 | 44.6 | 275.6 | 7.69 | 0.30 | 15.41 | 5.898 | 9.232 | 32.588 | 41.30 | 1.08 | 0.399 | 51.9 | 55.9 | 0.000 |
| `turbo_seed3_9917a359` (E1-3) | `wan-ti2v-5b-turbo-comfyui` | 3 | 4 steps, CFG 1, uni_pc/simple, shift 8 | success | 318.4 | 42.6 | 274.5 | 6.88 | 0.33 | 15.38 | 1.658 | 3.628 | 25.214 | 41.33 | 1.17 | 0.451 | 51.5 | 54.6 | 0.000 |
| `turbo_seed4_b3d7d934` (E1-4) | `wan-ti2v-5b-turbo-comfyui` | 4 | 4 steps, CFG 1, uni_pc/simple, shift 8 | success | 319.6 | 51.3 | 267.0 | 8.05 | 0.29 | 15.42 | 1.596 | 3.071 | 24.676 | 41.32 | 1.13 | 0.393 | 51.8 | 54.6 | 0.000 |
| `turbo_seed42_ls0.8_b0ae7318` (E1-5) | `wan-ti2v-5b-turbo-comfyui` | 42 | 4 steps, CFG 1, uni_pc/simple, shift 8, latent_scale 0.8 | success | 312.5 | 39.2 | 271.9 | 8.23 | 0.34 | 15.37 | 1.511 | 2.953 | 24.798 | 27.58 | 1.11 | 0.554 | 60.4 | 64.0 | 0.000 |
| E1-6 | `wan-ti2v-5b-comfyui` | 42 | 20 steps, CFG 5, uni_pc/simple, shift 8 | **not completed** | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |

Columns: times are the runner's wall-clock stages. "RAM before" is the
available RAM before loading, "RAM lowest" the lowest available value sampled
during generation, "RAM peak used" is total RAM (15.71 GB) minus that. The
quality columns are metrics v1 ([QUALITY_METRICS.md](../QUALITY_METRICS.md)):
0–255 luma scale, PSNR in dB against the derived start frame.

Run ids: `20261004T210526Z-92e20c65`, `20261004T213140Z-c21a84d8`,
`20261004T213707Z-0c960de6`, `20261004T214234Z-9917a359`,
`20261004T214758Z-b3d7d934`, `20261004T215325Z-b0ae7318`.

### Conditions

- A web browser was open on the laptop during the batch. Available RAM
  before loading was 6.9–8.2 GB; the runs with less than 8 GB carry the
  low-RAM warning. In every run available RAM fell to about 0.3 GB.
- The five E1 runs took 312–322 s; the earlier seed-42 run took 278 s. The
  difference is in ComfyUI's own time (248–261 s against 226 s). The cause
  was not investigated; memory pressure is a candidate, not a finding.

## What is the same in every Turbo clip

Seen on frames 0, 24, 48, 72, 96 and 120 of each clip.

- **Frame 0 matches the start frame** (41.3 dB in all default runs).
- **Colour saturation rises after the first frame.** Mean saturation (HSV,
  0–255) is about 120 on frame 0 and 135–139 on frame 24 in every default
  run, then stays near that level (up to 136–145 on frame 120). Reds and
  oranges in the sky become more intense than in the start frame.
- **The face is drawn in a different style from the rest.** Wherever the
  face becomes visible it has smooth, soft shading and rounded modelling,
  while the hair, clothing and city keep the flat inked look with hard
  contours.
- **The character does not stay "mostly still".** In every clip the head and
  upper body turn well beyond a slight head turn, and in several the arms or
  legs change pose. The prompt asks for a slight turn near the end.
- **The camera move is small.** The left pillar shifts and narrows a little
  and the framing tightens slightly. No clear push-in is visible on the
  sampled frames, except in seed 2 where the whole view changes.
- No held frames (`held_frame_ratio` 0), and little periodic pulse
  (`pulse_ratio` 1.08–1.31).

## Per run

### `turbo_seed42_92e20c65` — Turbo defaults, seed 42 (before E1)

- **Head turn:** back of the head through frame 72. Profile visible at frame
  96; at frame 120 the face is turned three-quarters toward the camera and
  the upper body has rotated with it.
- **Face vs. rest:** smooth, softly shaded face with a slight smile; hair
  and jacket remain inked and flat.
- **Colour:** saturation 120 → 138 by frame 24, 142 at frame 120.
- **Background:** skyline, sun and the pink billboard keep their positions
  and shapes on all six frames.
- **Other:** at frame 120 the arm rests on a raised knee in a new pose.

### `turbo_seed1_c21a84d8` — E1-1, Turbo defaults, seed 1

- **Head turn:** back of the head through frame 72. At frame 96 the face is
  already three-quarters visible; at frame 120 it is turned further toward
  the camera, the largest turn of the five default seeds on the sampled
  frames (head-region change against frame 0: 33.1).
- **Face vs. rest:** smooth shaded face, strong contrast to the inked hair.
- **Colour:** saturation 120 → 138, about 137–138 afterwards.
- **Background:** stable on the sampled frames.
- **Other:** at frame 120 a gloved fist is raised in front of the body; it
  is not in the start frame. Lowest `motion_mean` of the default runs (1.31).

### `turbo_seed2_0c960de6` — E1-2, Turbo defaults, seed 2

- **Head turn:** the only run where the turn happens in the middle. Profile
  toward the right at frame 72, then back to the view from behind at frames
  96 and 120.
- **Face vs. rest:** at frame 72 the profile is shaded more smoothly than the
  hair and clothes, though less so than the frontal faces of the other
  seeds.
- **Colour:** saturation 120 → 139, rising to 145 at frame 120, the highest.
- **Background:** **not stable.** From frame 24 on the skyline is
  rearranged: the tall tower moves and changes shape, new spires and a red
  neon building appear on the left, the billboard moves, and by frame 72 the
  sun sits further left in the frame. It reads as a large camera move across a
  city that is being redrawn, not as the same buildings seen from a new
  position.
- **Other:** by far the most frame-to-frame change (`motion_mean` 5.90,
  about four times the other seeds; `total_change` 32.6).

### `turbo_seed3_9917a359` — E1-3, Turbo defaults, seed 3

- **Head turn:** from behind through frame 96. At frame 120 a sharp profile
  facing right, with hair covering the eye area.
- **Face vs. rest:** the profile is smoother than the surroundings but, as a
  side view with little visible, the difference is smaller than in seeds 1,
  4 and 42.
- **Colour:** saturation 120 → 138, 140 at frame 120.
- **Background:** stable on the sampled frames.
- **Other:** the body changes most here. At frame 96 the shoulders lift and
  an arm with a bare forearm appears at the right; at frame 120 a leg is
  stretched out and a sneaker is at the right edge. The hair silhouette at
  frame 24 is rounder and smoother than in the start frame.

### `turbo_seed4_b3d7d934` — E1-4, Turbo defaults, seed 4

- **Head turn:** from behind through frame 72; head lowered at frame 96; at
  frame 120 a three-quarter view to the right with the eye clearly drawn.
- **Face vs. rest:** smooth shaded face against inked hair and jacket.
- **Colour:** saturation 120 → 135, the lowest rise of the default runs
  (134–136 afterwards).
- **Background:** buildings stable, but the **billboard changes colour**:
  pink in the start frame, yellow at frames 24 and 48, pink-red from frame
  72 on.
- **Other:** at frame 120 the legs are in a new position and a bare forearm
  with a wristband is visible.

### `turbo_seed42_ls0.8_b0ae7318` — E1-5, Turbo, seed 42, `latent_scale=0.8`

Same seed as the first run, so the content can be compared directly.

- **Motion and composition:** the same as the default seed-42 run on every
  sampled frame: same head turn (profile at 96, three-quarter face at 120),
  same poses, same stable background.
- **Colour:** clearly paler and lighter. Saturation 92 on frame 0 and
  105–108 afterwards, against 120 and 138–142 at scale 1.0. Mean luminance
  60–64 against 52–55; dark areas are lifted toward grey.
- **Start frame:** the paler look applies to frame 0 as well, so the first
  frame no longer matches the start frame: `start_frame_psnr` 27.6 dB
  against 41.3 dB. The scale is applied to the whole latent, including the
  part that encodes the start frame.
- **Face vs. rest:** as in the default run.

So at 0.8 the saturation after frame 0 (105–108) ends up below the start
frame's own level (120), not at it.

### E1-6 — base model, seed 42: not completed

`wan-ti2v-5b-comfyui`, defaults (20 steps, CFG 5), started 23:58:43. The
whole batch command was stopped by the coding assistant's host program
because the system was critically low on memory, at some point before 00:03
(the exact moment was not logged). It was an external stop of the process,
not an error raised by the run, so **no run record was written** and there
is nothing to rate. Its partial output folder (only the derived start frame)
and the copy uploaded to ComfyUI were deleted.

What this does and does not show: it is not evidence that the base model
cannot run here (the spike completed a base run in 34 minutes with an
untiled decode). It does show that a 13-minute sampling phase on this
machine, with a browser open and RAM at its limit, is exposed to this kind
of interruption. The run has not been repeated.

## Review material

In `spikes/output/E1/` (git-ignored; regenerate with `python spikes/e1_review.py`):

| File | |
|---|---|
| `contact_<label>.png` | frames 0, 24, 48, 72, 96, 120 of one run, labelled, 640×352 each, three per row |
| `face_<label>.png` | the same frames cropped to the head-and-shoulders box |
| `compare_grid.mp4` | the six clips tiled in one 24 fps video, each tile labelled with run id, seed and settings |
| `table.md`, `numbers.json` | the table above and the saturation / head-region numbers quoted here |

**Head-and-shoulders box:** left 160, top 100, right 660, bottom 420 (500×320
pixels in the 1280×704 frame). Chosen once by eye on the derived start
frame, where the head is centred near (395, 215), with margin for movement;
the same box is used for every run and frame. It is fixed, so when the
character moves the crop does not follow.

The videos themselves: `outputs/comic_rooftop_01/<run_id>/out.mp4`.

## Limits of this report

- Descriptions come from six sampled frames per clip and the numeric
  metrics, written by the coding assistant. Nobody watched the clips at
  speed; anything that happens between sampled frames (flicker, brief
  glitches, how smooth the turn is) is not described here.
- One run per setting; five seeds for the defaults. The "largest turn" and
  similar comparisons are about these frames only.
- The base model is missing, so nothing here compares Turbo with base.
