# Spike: SDXL base + IP-Adapter (+ Lineart T2I-Adapter) in the local ComfyUI

Date: 2026-10-05. Laptop: RTX 4060 Laptop 8 GB, 15.7 GB RAM, ComfyUI v0.38.2.
Research behind it: `docs/research/LOCAL_IMAGE_BACKEND.md`.
Script: `spikes/sdxl_ipadapter_spike.py`. Images: `spikes/output/sdxl_ipadapter/`
(git-ignored), each with a `report*.json` holding the exact settings.

Feasibility only. Nothing here is a benchmark run; nothing was written under
`outputs/` or `benchmarks/`. Judgements of the pictures are mine, from
looking at each image once; no metric was computed.

## Setup

- `sd_xl_base_1.0.safetensors`, `ip-adapter-plus_sdxl_vit-h.safetensors`,
  CLIP ViT-H image encoder, `t2i-adapter-lineart-sdxl-1.0.fp16.safetensors`;
  custom node `ComfyUI_IPAdapter_plus` (commit `a0f451a`), loads without
  errors.
- 1024x1024, 25 steps, CFG 6, dpmpp_2m / karras, seed 42 unless noted.
- Reference: a 300x300 crop of the front figure (head to waist) of
  `benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg`, with
  the handwritten note beside the hair painted over. The sheet is only read.
- Prompt: the character "sitting on a park bench, three-quarter view,
  daylight" plus the outfit.

## Results

| Image | IP-Adapter | Lineart adapter | Prompt change | Result |
|---|---|---|---|---|
| `sdxl_ipadapter_test_01.png` (Desktop only) | 0.8, linear, crop with label text | - | - | Copy of the reference: front view, no bench, invented handwriting |
| `..._test_w04.png` | 0.4, ease out, clean crop | - | - | Pose from the prompt; loosest identity; jacket open over a T-shirt, grey |
| `..._test_w06.png` | 0.6 | - | - | Pose from the prompt; face closer; jacket open, grey; a wristwatch appears |
| `..._test_w08.png` | 0.8 | - | - | Pose from the prompt; closest face and outfit (zipped hoodie, red straps); jacket grey; wristwatch |
| `..._lineart_a.png` | 0.8 | lines of w08, strength 0.8, to the last step | + "pure black jacket"; negative + "no watch, no wristwatch" | **Broken**: a mesh of white strokes and red smears, no figure |
| `..._lineart_b.png` | 0.8 | strength 0.4, first half of the steps | same | Clean. Same layout as w08, no watch, hands changed. Jacket grey with dark-red sleeves and hood: colour is worse |
| `..._prompt_only_c.png` | 0.8 | - | same | Clean. No watch. Jacket still grey, and a small chest logo appears |
| `..._lineart_d.png` | 0.8 | strength 0.6, first 70 % | + "(pure black jacket:1.4), black sleeves, black hood"; negative + "watch, wristwatch, bracelet, grey jacket, red jacket, red sleeves" | **Broken**, like a |
| `..._lineart_e_seed43.png` | 0.8 | as d, seed 43 | as d | **Broken**; the strokes follow w08's lines, so the adapter does drive the layout |

`<name>_lines.png` beside each lineart image is the line picture the adapter
received.

## What holds

- **IP-Adapter at 0.8 with "ease out"** ("prompt is more important") and a
  text-free crop gives a new pose from the prompt and a recognisably similar
  character: curly hair, face type, hooded zip jacket with drawstrings,
  dark-red backpack straps with buckles. It is the best setting found.
- The first run's copying was caused by the weight type and the label text
  in the crop, not by the weight: 0.4, 0.6 and 0.8 all follow the prompt.
- The stack fits: peak VRAM 6.8 to 6.9 GB of 8.2 GB with or without the
  lineart adapter; about 31 to 34 s per image including loading the models
  (sampling about 17 s). No crash, no out-of-memory.
- A negative prompt removes the wristwatch (b and c).

## What does not

1. **The jacket is never black.** Grey in every image, with or without
   "pure black jacket", weighted or not. Not solved.
2. **The Lineart T2I-Adapter is not usable as set up.** At strength 0.6 and
   0.8 it destroys the picture; at 0.4 over the first half of the steps the
   picture is clean, but then its effect cannot be told apart from the seed
   (b has the same layout as w08, which shares its seed). Cause not
   established. Candidates, none tested: the lines come from ComfyUI's Canny
   node because no lineart preprocessor is installed, and one-pixel Canny
   edges are not what this adapter was trained on; or a problem in how this
   ComfyUI version applies this adapter file.
3. **Identity is "similar", not "the same".** The face is the same type in
   every image, not the same face, and details drift (chest logo in c, open
   jacket at low weights).
4. Small invented details keep appearing and have to be named in the
   negative prompt one by one.

## Memory

| Run | Seconds | Peak VRAM | RAM available before | lowest |
|---|---|---|---|---|
| w04 / w06 / w08 (one session) | 50.0 / 35.3 / 31.3 | 6,823 MiB | 3.36 GB | 0.79 GB |
| lineart_a | 32.4 | 6,913 MiB | 8.21 GB | 1.12 GB |
| lineart_b | 32.4 | 6,849 MiB | 8.65 GB | 1.72 GB |
| prompt_only_c | 31.3 | 6,815 MiB | 8.76 GB | 2.01 GB |
| lineart_d | 32.3 | 6,881 MiB | 4.78 GB | 1.47 GB |
| lineart_e_seed43 | 33.5 | 6,849 MiB | 4.76 GB | 1.50 GB |

VRAM is the whole GPU as reported by `nvidia-smi`, sampled twice a second.
Available RAM falls to 1 to 2 GB even when 8 GB were free at the start:
ComfyUI keeps models cached in RAM as long as there is room. Nothing failed
because of it.

## Image-to-image (second session, same day)

From `sdxl_ipadapter_prompt_only_c.png`, same prompts, IP-Adapter 0.8:

| Image | Denoise | Result |
|---|---|---|
| `..._i2i_c_d35.png` | 0.35 | Same picture, slightly harder contrast. Jacket grey, chest logo kept |
| `..._i2i_c_d42.png` | 0.42 | Not looked at (between the other two) |
| `..._i2i_c_d50.png` | 0.50 | Same layout; jacket grey; logo kept; a wristwatch comes back |

Image-to-image holds the composition exactly, so it is a working answer to
"same frame, change one thing" as long as the change is small. It does not
repair colour or identity: what is wrong in the start picture stays wrong.
Peak VRAM 6.8 GB, about 34 s per image.

## Character LoRA (started, not finished)

- Training set: `spikes/build_hero_lora_dataset.py` writes eight 1024x1024
  crops with captions (trigger word `hero01`) to
  `spikes/output/hero_lora_dataset/`: six from the turnaround sheet (three
  full figures, three upper bodies; notes, swatches, shoe drawing and
  signature painted over), one from the rooftop frame, one from the shot_03
  keyframe. `contact_sheet.png` there shows them. w08 and prompt_only_c are
  left out on purpose (grey jacket, invented details).
- Trainer: ComfyUI's own `TrainLoraNode`, driven by
  `spikes/train_hero_lora_spike.py`. Nothing was installed.
- Smoke test 1 (10 steps, defaults: checkpoint depth 1, no offloading):
  **ComfyUI aborted** on the first step, VRAM 7.9 GB, available RAM 0.02 GB.
- Smoke test 2 (10 steps, `--checkpoint-depth 3 --offloading`): **ran**,
  4.4 s per step, peak VRAM 7.1 GB, available RAM down to 0.01 GB.
- Full run (600 steps, rank 8, learning rate 1e-4, same flags): reached step
  359 of 600 after 23.5 minutes at 4.07 s per step, then the session's
  background shell was stopped because the machine was critically low on
  memory. The node only saves at the end, so **no LoRA exists yet**. The run
  itself had not failed.
- So local training fits this laptop's GPU, at about 41 minutes for 600
  steps; the limit met was system RAM while training in the background.
  No cloud GPU was needed up to this point and none was set up.

## Character LoRA, trained in parts (third session, same day)

- Started again from step 0: the interrupted run had saved nothing, so there
  was nothing to continue from.
- Six parts of 100 steps each (not 120: a part has to finish inside the
  10-minute limit of one foreground command, and 100 steps took 6.7 to
  7.5 minutes including start-up). Each part saved its LoRA and the next
  continued from that file with a new seed; the optimizer state is not
  carried over between parts.
- Rank 8, learning rate 1e-4, AdamW, bf16, batch size 1,
  `--checkpoint-depth 3 --offloading`, the eight-image set above.
- Total 2,561 s (42.7 minutes) for 600 steps, about 4.1 s per step. Peak VRAM
  7.0 to 7.1 GB in every part. Available RAM fell to 0.01 to 0.19 GB in
  every part and came back to about 10.5 GB between parts; no part failed.
- Files: `spikes/output/hero_lora/ai_studio_hero01_r8_<N>_steps_*.safetensors`
  for N = 100 ... 600 (53 MB each), also in `<ComfyUI>/models/loras/`.

### Validation (600-step LoRA, strength 1.0, seed 42)

Same prompt as `prompt_only_c` with the trigger word `hero01` in front, same
negative prompt.

| Image | Setup | Result |
|---|---|---|
| `..._lora600_alone.png` | LoRA only, no IP-Adapter | Bench, three-quarter view, outdoors. Jacket dark charcoal, the darkest of all tries but not pure black. Red straps, no watch, no logo. The picture turns almost black-and-white: skin nearly white, heavy ink. The face is not the reference face |
| `..._lora600_ipa08.png` | LoRA + IP-Adapter 0.8 "ease out" | Face, skin tone and hair closest to the reference of all tries. Zipped hooded jacket, straps with buckles, no watch, no logo. Jacket olive-brown, not black. The pose turned frontal and tighter; the bench is barely in the frame |

Against the earlier tries:

| | Jacket colour | Watch | Chest logo | Line / layout drift |
|---|---|---|---|---|
| w08 (IP-Adapter, ease out) | grey | yes | no | none noted |
| prompt_only_c | grey | no | yes | none noted |
| i2i from c, denoise 0.35 / 0.50 | grey / grey | no / yes | yes / yes | layout held exactly |
| LoRA alone | dark charcoal | no | no | colours drain to near monochrome |
| LoRA + IP-Adapter | olive-brown | no | no | three-quarter view lost, framing tighter |

What this shows: the LoRA removes the invented details (watch, logo) and,
alone, pulls the jacket towards black; with the IP-Adapter the face is the
best so far. No single setting gives the reference face, a black jacket and
the asked-for pose together. "The same character" is not reached; it is
closer than before. Not tried: other LoRA strengths, the earlier part files
(100 to 500 steps), more steps or a higher rank, IP-Adapter weights other
than 0.8 together with the LoRA.

Each image took 31 to 33 s; peak VRAM 6.8 to 6.9 GB.

### Grid: LoRA with a weaker IP-Adapter, and earlier LoRA files

Same prompt, negative prompt and seed (42) as the validation above, LoRA
strength 1.0, IP-Adapter "ease out". One image per cell, judged by eye.

| Image | LoRA steps | IP-Adapter | Jacket colour | Watch | Chest logo | Pose / framing | Face against the reference |
|---|---|---|---|---|---|---|---|
| `..._lora600_ipa04.png` | 600 | 0.4 | dark charcoal | no | no | bench, three-quarter, head almost in profile; trousers olive | similar, hard to judge in profile |
| `..._lora600_ipa05.png` | 600 | 0.5 | dark charcoal | no | yes | bench, body three-quarter, face towards the viewer | close |
| `..._lora600_ipa06.png` | 600 | 0.6 | charcoal | no | yes | bench, nearly frontal | close |
| `..._lora600_ipa08.png` (earlier) | 600 | 0.8 | olive-brown | no | no | frontal, tight, bench almost gone | closest |
| `..._lora300_ipa04.png` | 300 | 0.4 | grey-green | no | no, but letter-like toggles at the collar | bench, three-quarter | similar |
| `..._lora400_ipa04.png` | 400 | 0.4 | dark charcoal | no | no | bench, body three-quarter, face towards the viewer; trousers dark | close |
| `..._lora500_ipa04.png` | 500 | 0.4 | charcoal | no | yes | bench, three-quarter, head almost in profile | similar |

- IP-Adapter weight with the LoRA: 0.4 keeps the pose from the prompt; from
  0.5 up the figure turns towards the camera and at 0.8 the bench is lost.
  The face gets closer to the reference as the weight goes up. 0.4 was taken
  for the second row of tries because it is the only weight with the asked
  pose and without a logo.
- LoRA file: no steady trend from 300 to 600 steps. The chest logo comes and
  goes (500 and 600 at 0.5 / 0.6 have it, 400 and 600 at 0.4 do not), which
  looks like chance between single images more than an effect of training
  length.
- The wristwatch is gone in every image with the LoRA.
- The jacket is dark charcoal at best, never pure black.

**Best of this grid: the 400-step LoRA with IP-Adapter 0.4** (dark jacket and
trousers, no watch, no logo, asked pose, face turned to the viewer and close
to the reference). The 600-step LoRA with 0.4 is second (olive trousers, face
in profile). This rests on one seed per setting; a few seeds per setting are
needed before calling either one the pipeline.

### Seeds: 400 steps against 600 steps (2026-10-06)

Both with IP-Adapter 0.4 "ease out", LoRA strength 1.0, same prompt and
negative prompt, seeds 42 to 45. Files `..._lora<N>_ipa04.png` (seed 42) and
`..._lora<N>_ipa04_seed<S>.png`. Judged by eye, one look per image.

| LoRA | Seed | Jacket colour | Watch | Logo | Pose / framing | Outfit kept | Face against the reference |
|---|---|---|---|---|---|---|---|
| 400 | 42 | dark charcoal | no | no | bench, three-quarter, face to the viewer | zipped hoodie, red straps | close |
| 400 | 43 | near black | no | no | bench, three-quarter, full body | **no**: jacket open over a grey T-shirt, red hood lining, no backpack | similar, leaner and older |
| 400 | 44 | dark brown | no | no | bench, three-quarter | zipped, one red strap; trousers brown | similar |
| 400 | 45 | dark brown-black | no | no | **frontal**, full bench | zipped hoodie, red straps; trousers brown | close |
| 600 | 42 | dark charcoal | no | no | bench, three-quarter, head almost in profile | zipped hoodie, red straps; trousers olive | similar |
| 600 | 43 | dark charcoal-brown | no | no | bench, three-quarter, full body | zipped hoodie, backpack straps, red hood lining | close |
| 600 | 44 | dark charcoal-brown | no | no | bench, three-quarter | zipped hoodie, red straps; trousers brown | similar |
| 600 | 45 | charcoal | no | no | bench, three-quarter, closer framing | zipped hoodie, red straps, an added red line across the chest; trousers olive-grey | similar, nose larger |

- **600 steps is the more consistent one.** It gives the asked three-quarter
  view and the zipped hooded jacket with backpack straps in four of four
  seeds. 400 steps does so in two of four: one seed drops the backpack and
  opens the jacket, another turns the figure frontal.
- No watch and no chest logo in any of the eight. The logo seen earlier at
  500 and 600 steps with IP-Adapter 0.5 / 0.6 did not come back at 0.4.
- Not solved by either: the jacket is dark charcoal to dark brown, never pure
  black; the trousers come out brown or olive in five of eight; the face is
  the same type in all eight but not one identical face.
- 53 to 66 s per image this time (less RAM was free at the start, 7.4 GB);
  peak VRAM 6.8 to 6.9 GB; no failure.

## Pipeline proposal (as far as this spike supports it)

| Setting | Value |
|---|---|
| Base model | `sd_xl_base_1.0.safetensors` |
| Character LoRA | `ai_studio_hero01_r8_600_steps_00501_.safetensors` (rank 8, 600 steps), strength 1.0, trigger word `hero01` at the start of the prompt |
| IP-Adapter | `ip-adapter-plus_sdxl_vit-h`, weight **0.4**, weight_type **"ease out"** ("prompt is more important"), reference = the text-free front upper-body crop of the turnaround sheet |
| Sampler | 1024x1024, 25 steps, CFG 6, dpmpp_2m / karras |
| Negative prompt | the base list plus "no watch, no wristwatch" |
| Same frame, small change (e.g. shot_04 start / end) | image-to-image from the accepted frame at denoise about 0.35 |
| Not used | Lineart T2I-Adapter (breaks the picture as set up), IP-Adapter-FaceID (licence) |

What it delivers: a recognisably consistent character across seeds, in the
asked pose, without invented accessories, in about 35 to 65 s per image on
this laptop. What it does not deliver: a pure black jacket, reliably dark
trousers, or one fixed face. It was tested on one prompt and one pose only.

## The locked pipeline on real project scenes (2026-10-06)

600-step LoRA at 1.0, IP-Adapter 0.4 "ease out", same reference crop, seed
42, 1280x704. Prompts are short versions of the scene texts in
`docs/prompts/trailer_01_keyframes.md` with `hero01` in front. One image
each, judged by eye.

| Image | Scene | What came out | Usable in the real scene? |
|---|---|---|---|
| `..._pipe_shot04_start.png` | shot_04 start: hand on the ledge, dusk, no glow | Hand resting on a concrete ledge, city below, no glow, no watch. Sleeve blue-grey, not black; no dark-red cuff; pale daylight instead of blue-violet dusk; four fingers visible | **No, not as it is.** The framing is right; colour of the sleeve and the light are wrong |
| `..._pipe_shot04_end_d35.png` | shot_04 end: image-to-image from the start, denoise 0.35, prompt asks for the grip and the red glow | The same frame, line for line. **No glow, no change in the fingers** | **No.** Nothing of the asked change happened |
| `..._pipe_shot04_end_d50.png` | same, denoise 0.50 | Same frame again, still no glow; lines and colours oversharpened | **No** |
| `..._pipe_shot03_eyes.png` | shot_03: extreme close-up of the eyes, sunset from one side | A close-up of two eyes, but hazel irises with blue around them, no curls, no warm / violet split, smoother line work | **No.** Not recognisably this character; the hand-made keyframe is far better |
| `..._pipe_shot06_start.png` | shot_06 start: from behind on the ledge, rising, night, full moon | From behind: curly dark hair, dark jacket, black backpack with dark-red piping, red hood lining. Sitting crouched, not rising; daytime city, no moon; dark sneakers | **Partly.** The character from behind is recognisable; time of day, moon and pose are not followed |

By the earlier criteria: no wristwatch and no logo in any of them; the
jacket (where visible) is blue-grey in shot_04 and near black in shot_06; the
face only appears in shot_03, where it is not the reference face.

What this adds to the picture:

- **"Keep the same frame" is half solved.** Image-to-image at 0.35 to 0.50
  does keep the frame exactly, but at that strength it cannot add something
  new: the red glow never appears. The start / end pair of shot_04 is
  therefore **not** produced. Not tried, and needing no download: redrawing
  only the palm and the cracks through a mask at high denoise (inpainting).
- The pipeline holds the character when the prompt is a figure in view (the
  bench, shot_06 from behind). On a detail shot (a hand, two eyes) the LoRA
  and the upper-body reference have little to hold on to, and the result is
  generic.
- Scene lighting from the prompt (dusk, night, moon, sunset from one side)
  was not followed in any of the four scenes.

### Weighted prompt for the jacket (separate try)

`..._pipe_bench_weighted_black.png`: the bench prompt, seed 42, with
"(solid black jacket:1.3), (pure black:1.2)" added. Against
`..._lora600_ipa04.png` (same seed, no weights) the jacket is a little
darker, still dark charcoal with grey panels, not pure black; trousers stay
olive; pose, face, no watch and no logo are unchanged. The weighting helps
slightly and does not solve it.

## Open, for a decision

- Composition locking (the shot_04 start / end problem) is not shown to work.
  Options that need a download and therefore approval: the small Canny
  ControlNet (320 MB, OpenRAIL++), which matches the Canny lines we can
  already make; or a lineart preprocessor for the adapter we have. An option
  that needs no download: image-to-image from the start frame at low denoise.
- The jacket colour and exact identity probably need either image-to-image
  from an approved image or the small character LoRA from the research note.
- Whether "recognisably similar" is good enough for keyframes is the owner's
  call after looking at w08, b and c.
