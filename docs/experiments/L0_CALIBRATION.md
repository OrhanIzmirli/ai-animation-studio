# Level 0 — first calibration of the automatic metrics against the owner's judgement

Date: 2026-10-06. Clips: the eight recorded runs whose video is on this
machine (seven of `comic_rooftop_01`, one of `hero01_sdxl_bench_01`). The
Camber smoke run `a264ad6e` has no video here (lost on 2026-10-05) and was
not shown.

## What the owner gave

- **Numeric ratings for one clip**, through the review page:
  `hero01_sdxl_bench_01/20261006T004754Z-bc8b8338`, evaluator "3",
  stored in `benchmarks/comic_evaluations/hero01_sdxl_bench_01/.../3.r001.json`.
  Scores 1–2 out of 10 on every criterion (camera_motion 1,
  background_stability 1, character_identity 2, overall 2).
- **Spoken notes for the rooftop clips** (with screenshots),
  not entered into the page, so **not stored as ratings**. Quoted:
  "suratlar tutarsız kötü" (faces inconsistent, bad) and "hepsi korkunç"
  (all of them terrible), plus one defect per clip:

| Run | Setting | Owner's note |
|---|---|---|
| `b0ae7318` | Turbo, seed 42, latent_scale 0.8 | jacket colours wrong |
| `92e20c65` | Turbo, seed 42 | pixelation |
| `0c960de6` | Turbo, seed 2 | the camera angle wobbles |
| `b3d7d934` | Turbo, seed 4 | the hand is malformed |
| `c21a84d8`, `9917a359` (Turbo seeds 1, 3), `84f3fd82` (base) | | no specific note (covered by "all terrible") |

No numbers were made up from these notes. The comparison below therefore
asks only: does a metric single out the clip the owner singled out?

## Metrics (latest version per run)

| Run | motion_mean | motion_max | total_change | start_frame_psnr | pulse_ratio | lum_flicker | held_ratio | Owner |
|---|---|---|---|---|---|---|---|---|
| 92e20c65 | 1.68 | 3.14 | 27.6 | 41.3 | 1.20 | 0.47 | 0.00 | pixelation |
| c21a84d8 | 1.31 | 2.69 | 25.7 | 41.3 | 1.31 | 0.38 | 0.00 | – |
| **0c960de6** | **5.90** | **9.23** | 32.6 | 41.3 | 1.08 | 0.40 | 0.00 | **camera wobbles** |
| 9917a359 | 1.66 | 3.63 | 25.2 | 41.3 | 1.17 | 0.45 | 0.00 | – |
| b3d7d934 | 1.60 | 3.07 | 24.7 | 41.3 | 1.13 | 0.39 | 0.00 | hand malformed |
| **b0ae7318** | 1.51 | 2.95 | 24.8 | **27.6** | 1.11 | **0.55** | 0.00 | **jacket colours wrong** |
| 84f3fd82 | 0.22 | 0.73 | 2.9 | 41.3 | 1.83 | 0.16 | 0.91 | – |
| **bc8b8338** (bench) | **5.87** | **12.8** | **77.8** | 45.3 | 1.19 | **0.72** | 0.00 | **camera 1/10, background 1/10** |

## Which metrics agree with the owner

| Metric | Agrees? | Evidence |
|---|---|---|
| `motion_mean`, `motion_max` | **Yes, for camera / background instability** | The two clips the owner called out for camera movement (0c960de6; the bench clip, camera 1/10) are the only two with motion_mean above 2 (5.90 and 5.87, against 1.3–1.7 for the other Turbo clips). Two cases. |
| `start_frame_psnr` | **Yes, for a global colour change** | The clip with wrong jacket colours is the only one far below 41 dB (27.6). It catches a change of the whole picture from frame 0, not a wrong colour that the start frame already has. One case. |
| `luminance_flicker` | Weakly | Highest rooftop value on the wrong-colour clip (0.55) and highest overall on the bench clip (0.72); otherwise 0.38–0.47 with no note attached. Also depends on scene brightness. |
| `total_change` | Only for the bench clip | 77.8 there; on the rooftop clips it does not separate 0c960de6 (32.6) from the rest (24.7–27.6). |
| `temporal_pulse_ratio` | No signal | 1.08–1.31 on all Turbo clips; nothing noted about pulsing. |
| `held_frame_ratio` | No signal | 0 on every Turbo clip; 0.91 on the base clip, about which the owner said nothing specific. |
| (none) | **No metric** | Pixelation (92e20c65), the malformed hand (b3d7d934), inconsistent faces (all clips) and the overall verdict "all terrible": every clip has ordinary values. The metrics measure change between frames, not how good a frame is. |

## Limits

- Eight clips, one numeric rating, the rest spoken notes. No correlation
  coefficient is meaningful at this size; this is "does the metric point at
  the clip", counted by hand.
- The ratings are near the floor everywhere ("all terrible", 1–2/10), so
  there is no spread of quality for a metric to follow.
- The notes name the most visible defect per clip; a clip without a note is
  not a clip without defects.
