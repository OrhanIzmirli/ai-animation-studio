# Style layer: a deterministic hand-made finish

Post-processing that gives a generated clip the feel of hand-made animation:
timing in held drawings, a limited palette, ink contours, halftone, grain.
CPU only, no models, no GPU, fully reproducible.

Code: `src/ai_studio/post/` (`timing.py`, `operations.py`, `noise.py`,
`profile.py`, `variant.py`). Profiles: `configs/style_recipes/`. Command:
`scripts/postprocess_run.py`.

> **Status:** two first profiles, applied to one clip (the seed-42 Turbo run).
> The numbers below are measurements of that one clip. Nobody has judged yet
> whether the result looks good; the profiles' parameter values are first
> choices, not tuned ones.

## The principle

**Hand-made imperfection = consistent habits + content-dependent variation,
never uniform randomness.**

A filter that thickens every line by the same amount, holds every second
frame, and adds independent noise to every pixel looks like a filter. A
person does none of that. A person has habits (a preferred line weight, a
way of pressing harder in the shadows, a way of timing a slow moment) and
applies them to what is in the picture. So in this layer:

- **Habits are fixed per artist, not per shot.** They live in an *artist
  profile*: a named, versioned set of parameters plus one seed, the artist's
  "hand". The same profile on every shot of a trailer gives one consistent
  signature.
- **Variation comes from the picture.** Line weight follows light and
  contrast. The hold of a drawing follows how much the picture moves there.
  Grain follows luminance.
- **Randomness is seeded, smooth and tied to the drawing.** Where a random
  field is needed (pressure, wobble) it is low-frequency, spatially coherent
  noise, derived from (profile seed, habit, drawing). It changes when a new
  drawing starts and **never inside a hold**, because a held drawing is one
  picture shown several times. There is no per-pixel independent noise on
  lines or shapes. Only grain is fine-grained.
- **Deterministic, never regular.** The same input and profile always give
  the same video, bit for bit. But the hold pattern is not periodic and the
  line weight is not constant.

## Artist profiles

A profile is a YAML file `configs/style_recipes/<name>_v<version>.yaml`.
Profiles are named by technique, never by a studio or a title.

```yaml
name: comic_finish
version: 1
description: ...
seed: 1701                 # the artist's hand
habits:                    # each habit is optional; absent = not applied
  timing: {...}
  palette: {...}
  ink: {...}
  halftone: {...}
  bloom: {...}
  grain: {...}
  misregistration: {...}
```

Habits are always applied in this order: timing, palette, ink, halftone,
bloom, grain, misregistration. Unknown habits or parameters are rejected;
parameters a profile leaves out take documented defaults, and the manifest
of every styled video records all of them. A profile is immutable once
videos were made with it: changing a habit means a new version in a new file.

| | `comic_finish_v1` | `retro_cel_v1` |
|---|---|---|
| Idea | printed comic | cel animation photographed on film |
| Timing | by the action; rhythm broken every 3–6 holds | by the action; rhythm broken every 4–7 holds |
| Palette | 24 colours from the start frame, hard | 32 colours, blended 60 % (softer) |
| Ink | weight 1.0–3.5 px, strength 0.75, wobble 0.6 px | weight 1.0–2.0 px, strength 0.6, no wobble |
| Halftone | 6 px dots below luminance 55 | none |
| Bloom | none | slight glow above luminance 170 |
| Grain | amount 2.5, twice as strong in black | amount 3.5, 1.6× in black |
| Misregistration | 1 px | none |
| Seed | 1701 | 4207 |

## What each operation does

Every operation is a pure function on arrays of drawings (`uint8`,
`(M, H, W, 3)`): it returns a new array and never changes its input.

### Timing: `animator_holds`, `hold_frames`

- **Imitates:** animation is drawn "on twos" or "on threes": one drawing is
  photographed for two or three frames. An animator chooses this per moment:
  slow, subtle moments are held longer, fast action is drawn on ones.
- **Does:** measures the motion between consecutive frames (the same measure
  as the `motion_mean` metric) and gives each new drawing a hold from it:
  **threes** when nearly still, **twos** for moderate motion, **ones** for
  fast motion. **Hysteresis** keeps the timing from flipping when the motion
  hovers around a threshold. After a seeded number of equal holds in a row
  the rhythm is **broken once** (one hold a frame longer or shorter), as a
  person does not keep a perfectly even beat. The frame count and the 24 fps
  never change.
- `hold_frames(n)` is the mechanical version (every drawing n frames), kept
  as a building block and as `timing.mode: fixed`.
- **Limits:** motion is one number per frame for the whole picture, so a
  small fast gesture in a still scene is timed as "still". On a clip with
  even motion the result is mostly one hold length with occasional breaks,
  which is far from mechanical but still fairly regular (see results).
  Holding drawings discards the frames in between.

### `palette_limit(k, source_image)`

- **Imitates:** a production paints with a fixed, limited set of colours.
- **Does:** learns a k-colour palette once from the shot's start frame
  (k-means, fixed seed) and maps every drawing to it, so all frames share
  one palette and colour cannot drift. `strength` below 1 blends with the
  original for a softer look.
- **Limits:** gradients become bands. Colours that are rare in the start
  frame are pulled to the nearest palette colour, so small bright accents
  can lose their hue. Colours that only appear later in the shot have no
  entry at all.

### `ink_lines(strength, thickness)`

- **Imitates:** inking by hand: heavier line on the shadow side and at
  strong edges, lighter in the light, tapering at the ends of a stroke.
- **Does:** finds contours as edges of the luminance and draws them dark.
  The line weight is **not a constant**. Between the profile's minimum and
  maximum it is:
  - heavier where the surroundings are dark, thinner in highlights
    (`shadow_bias`),
  - heavier where local contrast is strong (`contrast_bias`),
  - thinner toward line ends (`taper`),
  - varied slowly across the picture by a smooth pressure field of the
    drawing (`pressure_variation`).
- **Limits:** lines come from luminance edges: two colours of equal
  brightness get no line between them, and fine texture (windows, hatching)
  is inked too. The taper is estimated from how much edge surrounds a point,
  not by tracing strokes, so a corner does not taper.

### `line_boil(px, seed)` — off by default

- **Imitates:** no two tracings of a line are identical, so hand-drawn lines
  "boil" slightly from drawing to drawing.
- **Does:** displaces the ink layer by a smooth field of at most about `px`
  pixels. The field belongs to the drawing, so lines move only when a new
  drawing starts. In a profile it is the ink habit's `boil_px` (0.6 in
  `comic_finish_v1`, 0 in `retro_cel_v1`).
- **Limits:** a first-order approximation, meant for up to about 1.5 px.

### `halftone(dot_size, threshold)`

- **Imitates:** printed comics build shadows from dots.
- **Does:** a 45° dot screen in areas darker than `threshold`; the darker,
  the larger the dot. Dot size varies slowly across the picture (ink gain).
- **Limits:** the screen is regular and fixed to the frame, because a
  printing screen is; this is the one deliberately mechanical element. It
  only darkens. In a dark scene most of the picture counts as shadow.

### `grain(amount, seed)`

- **Imitates:** paper and film grain.
- **Does:** fine monochrome noise whose strength follows luminance: stronger
  in shadows (`shadow_gain`). One grain field per drawing: it changes with
  each new drawing and stays put during a hold.
- **Limits:** plain Gaussian noise without clumping or colour.

### `misregistration(px)`

- **Imitates:** colour plates printed slightly out of register.
- **Does:** shifts red right and blue down by `px` whole pixels, leaving
  thin colour fringes at edges. Constant, like one print run.

### `bloom(threshold, radius, strength)`

- **Imitates:** halation of bright artwork photographed on film.
- **Does:** a soft glow around areas brighter than `threshold`.
- **Limits:** only brightens; computed at quarter resolution.

## Applying a profile to a run

```
python scripts/postprocess_run.py --benchmark comic_rooftop_01 --run-id <run_id> --recipe comic_finish_v1
```

Run records stay immutable. A styled video is a **derived artifact** of a
run and lives beside the run's output:

```
outputs/<benchmark>/<run_id>/post/<recipe>/
    out.mp4            same encoder settings and frame rate as the run
    manifest.json      recipe name, the whole profile with every parameter,
                       recipe file SHA-256, source video SHA-256, start frame
                       SHA-256, output SHA-256, the hold pattern, tool versions
    metrics_v2.json    quality metrics of the styled video
```

- **Never overwritten:** applying the same profile to the same run again is
  refused. Applied again from scratch it produces the same video (tested).
- **Metrics are kept apart.** The metrics file of a styled video carries
  `variant: "post/<recipe>"` and is written next to its manifest. The store
  of raw-run metrics (`benchmarks/metrics/...`) refuses variant records, so
  a styled video's numbers can never stand in for the run's.
- `outputs/` is git-ignored, so variants are local files. They travel with a
  run in `scripts/collect_run.py` zips.

### Reading the metrics of a styled video

- **`held_frame_ratio` goes up to about 0.5 on purpose.** That is animation
  on twos and threes, **not a freeze**: a frozen video has a ratio near 1
  and `motion_mean` near 0.
- **`hold_pattern_periodicity`** (new in metrics v2) is the highest
  autocorrelation of the held / new-drawing sequence: **1.0 = perfectly
  mechanical** (fixed twos), lower = less regular. It is undefined (`-`) for
  a video without holds.
- **`start_frame_psnr` drops**, because frame 0 is restyled too.
- `motion_max` rises: a drawing held for two frames carries two frames of
  motion in one step.

## Measured results

Source: `comic_rooftop_01 / 20261004T210526Z-92e20c65` (Turbo defaults,
seed 42, 1280×704, 121 frames). Processing took 44 s (`comic_finish_v1`) and
47 s (`retro_cel_v1`) on the laptop's CPU.

| | raw | + `comic_finish_v1` | + `retro_cel_v1` |
|---|---|---|---|
| drawings (of 121 frames) | 121 | 58 | 58 |
| held_frame_ratio | 0.000 | 0.525 | 0.525 |
| hold_pattern_periodicity | – | 0.885 | 0.793 |
| motion_mean | 1.684 | 1.452 | 1.559 |
| motion_max | 3.138 | 3.970 | 4.280 |
| total_change | 27.638 | 22.463 | 23.722 |
| start_frame_psnr (dB) | 41.29 | 23.00 | 26.89 |
| temporal_pulse_ratio | 1.203 | 1.126 | 1.309 |
| luminance_flicker | 0.470 | 0.345 | 0.396 |
| luminance min .. max | 52.0 .. 54.9 | 46.1 .. 48.0 | 49.4 .. 51.5 |

`hold_pattern_periodicity` measured on the encoded video equals the value
computed from the hold pattern itself (0.884712 and 0.793268), so the holds
survive encoding and are detected.

Hold patterns chosen for this clip (frames per drawing):

- `comic_finish_v1`: 2 2 2 2 2 2 **3** 2 2 2 2 2 2 **3** 2 2 2 2 **3** 2 2 2
  **3** 2 2 2 2 2 2 **3** 2 2 2 2 2 2 **3** 2 2 2 2 **3** 2 2 2 2 2 **1** 2 2
  2 2 2 **1** 2 2 2 2
- `retro_cel_v1` breaks the rhythm at different places (its own seed and
  habit), also ending with two drawings on ones.

What can be said from these numbers and from the review frames:

- **Timing follows the action only weakly on this clip.** Its motion is
  fairly even and falls in the "moderate" class almost throughout, so the
  result is twos with a three every 4–7 drawings and ones twice near the
  end, where the motion is largest. Periodicity 0.79–0.89 is below the 1.0
  of fixed twos but still high. A clip with more varied motion should vary
  more; that has not been measured.
- **Both profiles darken the picture**: mean luminance drops by about 6
  (`comic_finish_v1`: ink over a detailed city, and halftone over a scene
  that is mostly shadow) and about 3 (`retro_cel_v1`).
- **The palette flattens the face.** At frame 120 the smoothly shaded face
  noted in experiment E1 becomes a few flat tones under `comic_finish_v1`,
  closer to the inked look of the rest. Under `retro_cel_v1` (blended
  palette) the face keeps most of its soft shading.
- **Colour accents are lost under `comic_finish_v1`:** the pink billboard and
  small purple lights lose their hue, and distant buildings lose detail.
- **Luminance flicker goes down** (0.47 → 0.35 / 0.40), consistent with held
  drawings and one palette.

### Not measured

- **E1-3 (seed 3) was not processed.** Its raw video was no longer on disk
  when this was done: on 2026-10-05 at 00:36 every `out.mp4` under
  `outputs/` and `spikes/output/` disappeared, for a reason that was not
  established. The seed-42 clip was rebuilt from the spike's surviving PNG
  frames and verified byte-identical to the recorded SHA-256
  (`5237eff9…29d1f6b3`); seed 3 has no such frames and would have to be
  generated again.
- No human judgement of either profile.
- Only one clip: nothing here shows the "same signature across shots" on
  real footage. It holds by construction (same parameters, same seed) and is
  covered by tests on synthetic frames.

## Review material

In `spikes/output/style/` (git-ignored; regenerate with
`python spikes/style_review.py <run_id> ...`):

| File | |
|---|---|
| `before_after_92e20c65_<recipe>.mp4` | raw left, styled right, labelled, 24 fps |
| `contact_92e20c65_<recipe>.png` | frames 0 / 48 / 96 / 120: raw row above, styled row below |
| `face_92e20c65_<recipe>.png` | frame 120, head-and-shoulders crop (box 160,100–660,420), raw vs. styled |
| `table.md` | the metrics table |

## Known limits of the layer

- It restyles pixels; it cannot fix what the model drew (identity, anatomy,
  a background that changes).
- Lines, palette and timing are driven by luminance and whole-frame motion,
  not by an understanding of the picture: no notion of character vs.
  background, of a face, or of a stroke.
- Whole-clip in memory: about 330 MB per 121 frames at 1280×704.
- Determinism is bit-exact for the same numpy and ffmpeg versions; they are
  recorded in every manifest.
