# Level 1 — camera control through the prompt

Date: 2026-10-06. Backend `wan-ti2v-5b-turbo-comfyui`, seed 42, 1280x704,
5 s (121 frames), laptop.

## Why the shot controls never reached the model

`camera_motion`, `subject_motion` and `visual_direction` are read from the
definition (`benchmarking/definitions.py`), carried by
`VideoGenerationRequest` (`generation/request.py`) and arrive at the backend.
Both Wan backends then dropped them on purpose and wrote a warning
("not consumed separately: only the prompt is passed to the model"): Wan 2.2
TI2V-5B has one text prompt and one start frame, no camera input.

## What was connected

- `src/ai_studio/generation/shot_prompt.py`: `fold_shot_controls` appends the
  three controls to the prompt as labelled sentences, in a fixed order.
  `camera_motion: static` is expanded to a longer wording (fixed locked-off
  camera; no pan, tilt, zoom, push-in, dolly or shake; background keeps its
  place and size).
- ComfyUI backends: new backend parameter `shot_controls`, `"ignore"`
  (default, the behaviour of every earlier run, so they still regenerate
  identically) or `"prompt"`. The text actually sent is recorded as
  `model_parameters.prompt_sent`. The diffusers backend (`wan-ti2v-5b`,
  cloud) is unchanged.
- Two new definitions, because a definition with runs is never edited:
  - `comic_rooftop_01_static_cam`: comic_rooftop_01 with `camera_motion:
    static`, and without the prompt's sentence "Slow cinematic camera
    push-in." (it would contradict the control).
  - `hero01_sdxl_bench_01_static_cam`: the request of the drifting bench run
    `hero01_sdxl_bench_01/20261006T004754Z-bc8b8338` (same keyframe, prompt,
    negative prompt, seed) plus `camera_motion: static`. This is the only
    like-for-like comparison with 5.87: that number belongs to the bench
    scene, 1.68 to the rooftop scene.

## Results

| Run | Scene | Camera request | motion_mean | motion_max | total_change | lum_flicker |
|---|---|---|---|---|---|---|
| `comic_rooftop_01/20261004T210526Z-92e20c65` (reference) | rooftop | "slow push-in" in the prompt | 1.68 | 3.14 | 27.6 | 0.47 |
| `comic_rooftop_01_static_cam/20261006T154422Z-46c8c209` | rooftop | static, folded into the prompt | **1.58** | 2.73 | 25.8 | 0.37 |
| `hero01_sdxl_bench_01/20261006T004754Z-bc8b8338` (drift) | bench | "Static camera." in the prompt | 5.87 | 12.8 | 77.8 | 0.72 |
| `hero01_sdxl_bench_01_static_cam/20261006T154902Z-2bbcb820` | bench | + static, folded into the prompt | **5.69** | 12.6 | 79.7 | 0.53 |

- Rooftop: 1.68 → 1.58 (−6 %). Within the spread of seeds alone (Turbo,
  same scene, seeds 1–4: 1.31 to 5.90). Not evidence of an effect.
- Bench: 5.87 → 5.69 (−3 %), total_change *higher* (77.8 → 79.7).
- Frames 0 / 60 / 120 of the two bench clips, side by side: nearly the same
  pictures. The frame still drifts and zooms (bench and trees shift and grow
  by frame 60); the longer camera wording changed the result very little.
- Rooftop frames 0 / 60 / 120: the view still moves in (the left wall and the
  skyline shift); the figure turns fully toward the camera and the face is
  invented, as in E1.

## Reading

Folding the camera control into the prompt does **not** stop the drift on
this model at this setting. A likely reason, not tested: the Turbo model runs
at CFG 1, so the text is never amplified against the unconditional pass;
what the start frame and the distilled motion prior do dominates. The
connection itself works and is recorded per run; it is simply too weak a
lever for Wan 2.2 TI2V-5B Turbo.

Not tried: the base model (CFG 5), where the text weighs more; a model with
a real camera input; stabilising the clip afterwards. One seed per setting.
No human rating of the two new clips.
