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
