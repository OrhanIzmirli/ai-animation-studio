# Local image backend for character keyframes (spike, part 1: research)

Date read: 2026-10-05. Web pages only; nothing was downloaded, installed or
run. Goal: a local, free, scriptable replacement for the manual Gemini step
that makes character keyframes, on the laptop (RTX 4060 Laptop, 8 GB VRAM,
15.7 GB RAM, ComfyUI v0.38.2 already installed with only the Wan models and
the `ComfyUI-GGUF` node; 101 GB free on C:).

Labels: **Fact** = read on the linked page today. **Derived** = my arithmetic
on those facts. **Estimate** = my judgement, no source. **UNVERIFIED** = not
found or not checked. No recommendation is final; the choice is the owner's.

This builds on `docs/research/IMAGE_BACKENDS.md` (section 4, written
2026-10-05), which left SDXL's fit, speed and identity quality
on 8 GB unverified.

## Summary

- The SDXL family has everything asked for under commercially usable
  licences: base model (OpenRAIL++-M), IP-Adapter (Apache-2.0), ControlNet
  for Canny and Lineart (OpenRAIL++ or Apache-2.0).
- **IP-Adapter-FaceID is out**: it is research-only because it depends on
  InsightFace models. The plain and "plus" IP-Adapters use CLIP only and are
  Apache-2.0.
- On 8 GB VRAM: base SDXL alone is measured to work on the same GPU class
  (about 5.6 GB peak, 16 s per 1024x1024 image). Base + IP-Adapter + a
  **small** ControlNet is Derived to be workable but tight; with the 2.5 GB
  union ControlNet the weights alone exceed 8 GB and ComfyUI has to offload.
  Nobody measured the full stack on 8 GB in the pages read.
- IP-Adapter copies the general look of a reference, not an exact identity.
  It sees the reference at 224x224, so the three-figure sheet must be cropped
  to one figure or one head first. How well it holds this face and outfit is
  exactly what the spike has to show.
- A FLUX.2 klein 4B route (Apache-2.0, native reference images, no
  IP-Adapter needed) is the alternative family; it has no confirmed
  ControlNet.

## 1. Candidate checkpoints

| # | Checkpoint | Licence | File size | Style fit | Notes |
|---|---|---|---|---|---|
| A | **SDXL base 1.0** `sd_xl_base_1.0.safetensors` | CreativeML Open RAIL++-M (tag `openrail++`) | 6.94 GB | General; style comes from the prompt and the reference | The model every SDXL IP-Adapter and ControlNet was trained against |
| B | **DreamShaper XL 1.0** (Lykon) | `openrail++` on the diffusers repo; the repo with the single-file checkpoints is tagged `other` | 6.94 GB single file; 21 GB diffusers repo | Tagged art / artistic / anime; card example is a portrait photo | Licence tag differs between its two repos |
| C | **FLUX.2 klein 4B** (not SDXL) | Apache-2.0 | 4.07 GB fp8 transformer + text encoder (5.63 GB fp8_mixed or 3.48 GB fp4_mixed) + VAE | General | Reference images are built in (up to 4); no IP-Adapter or ControlNet confirmed |

Facts and sources:

- A: licence tag and sizes from the [SDXL base 1.0 file list](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main).
  Licence text: "Licensor claims no rights in the Output You generate using
  the Model"; redistribution and hosting are permitted if the use
  restrictions of Attachment A are passed on
  ([LICENSE.md](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)).
  The fp16 U-Net alone is 5.14 GB
  ([unet folder](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet)).
- B: `openrail++`, fine-tuned from SDXL base 1.0, tags and example prompt
  from the [dreamshaper-xl-1-0 card](https://huggingface.co/Lykon/dreamshaper-xl-1-0);
  that repo has no single-file checkpoint and totals 21 GB
  ([file list](https://huggingface.co/Lykon/dreamshaper-xl-1-0/tree/main)).
  The single file `DreamShaperXL1.0Alpha2_fixedVae_half_00001_.safetensors`
  (6.94 GB) is in [Lykon/DreamShaper](https://huggingface.co/Lykon/DreamShaper/tree/main),
  whose licence tag is `other`. UNVERIFIED: which terms apply to that file.
- C: all figures are from `IMAGE_BACKENDS.md` section 4a and 4d (not
  re-opened here), except: klein supports
  "text-to-image, single image editing, and multi-reference generation"
  and runs "in about 8GB of VRAM" (Secondary:
  [comfy.org FLUX workflows](https://comfy.org/workflows/model/flux/)).
  UNVERIFIED: a ControlNet for klein, and any measured 8 GB timing.

Looked at and excluded:

- **Juggernaut XL v9**: CreativeML Open RAIL-M, "may not be deployed behind
  paid API services" without a licence from the vendor, and "purpose-built
  for photorealism" ([card](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)).
  Wrong style and a narrower licence.
- Anime-only SDXL fine-tunes (Animagine, Illustrious and their mixes): the
  brief excludes anime-only; not researched further.

UNVERIFIED: no SDXL fine-tune was found today that is both clearly
commercially licensed and specifically a painterly / graphic-novel model.
With A, the comic look has to come from the prompt, the IP-Adapter reference
and, later, a LoRA.

## 2. IP-Adapter for the SDXL family

- **Supported, Apache-2.0** ([h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)).
- SDXL files ([sdxl_models folder](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models)):

  | File | Size | Image encoder | Use |
  |---|---|---|---|
  | `ip-adapter_sdxl_vit-h.safetensors` | 698 MB | ViT-H | general reference |
  | `ip-adapter-plus_sdxl_vit-h.safetensors` | 848 MB | ViT-H | "closer to the reference image" |
  | `ip-adapter-plus-face_sdxl_vit-h.safetensors` | 848 MB | ViT-H | cropped face as the condition |
  | `ip-adapter_sdxl.safetensors` | 703 MB | ViT-bigG | general, needs the larger encoder |

- Image encoder: OpenCLIP ViT-H-14, `model.safetensors` 2.53 GB
  ([image_encoder folder](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder)).
  The `vit-h` adapters use it; only `ip-adapter_sdxl` needs ViT-bigG
  ([card](https://huggingface.co/h94/IP-Adapter)).
- ComfyUI: the usual node pack is
  [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus):
  GPL-3.0, and "in 'maintenance only' mode". Adapter files go to
  `ComfyUI/models/ipadapter`, the encoder to `ComfyUI/models/clip_vision`
  as `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`. It is a custom node,
  so installing it is a code install that needs approval. UNVERIFIED: that
  it runs unchanged on ComfyUI v0.38.2.
- **IP-Adapter-FaceID: not usable.** "IP-Adapter-FaceID models are released
  exclusively for research purposes and is not intended for commercial
  use", because "InsightFace pretrained models are available for
  non-commercial research purposes"
  ([h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)).
  This closes the "UNVERIFIED" in `IMAGE_BACKENDS.md` 4e. It would also be a
  poor technical fit: a face-recognition model trained on real faces "fails
  to represent comic character faces accurately", while CLIP-only adapters
  do better on comic characters (Secondary: search summary of
  [arXiv 2504.05838](https://arxiv.org/pdf/2504.05838); I did not open the
  paper).

What this means for `character_turnaround_hero_01.jpg` (Estimate, from how
the adapter works):

- The CLIP encoder sees a 224x224 square. The 1024x765 sheet with three
  figures, notes and swatches would be reduced to a thumbnail of "a sheet
  with three figures". The reference has to be a **crop**: one full figure
  for the outfit, or the head for the face (`plus-face` expects a cropped
  face). A crop is a derived copy in a scratch folder; the imported original
  is not touched.
- IP-Adapter transfers overall look, colours and rough features. It does not
  lock an exact face or exact clothing details the way a trained LoRA does.
  Expect "recognisably similar", not "identical".

## 3. ControlNet for the SDXL family

All three options below take Canny or Lineart input and are commercially
usable.

| Model | Licence | File | Control types |
|---|---|---|---|
| `diffusers/controlnet-canny-sdxl-1.0-small` | `openrail++` | 320 MB (fp16) / 640 MB | Canny |
| `xinsir/controlnet-union-sdxl-1.0` | Apache-2.0 | 2.51 GB | 12 types incl. Canny, Lineart, Anime Lineart, Depth, Openpose |
| `TencentARC/t2i-adapter-lineart-sdxl-1.0` (T2I-Adapter, not ControlNet) | Apache-2.0 | 158 MB (fp16) / 316 MB | Lineart |

Sources: [canny small](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/tree/main),
[union file list](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/main)
and [union card](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0),
[T2I lineart](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0/tree/main).

- Canny needs no extra preprocessor model (an edge filter). Lineart needs a
  preprocessor network from another node pack; UNVERIFIED: its licence and
  size.
- For the shot_04 start/end problem (Estimate): Canny edges of the accepted
  start frame hold the hand, ledge and framing, and the prompt adds the
  glow. The edges also resist the wanted change (fingers gripping harder),
  so control strength and its end point need tuning. Image-to-image from the
  start frame at low denoise, or inpainting only the palm and cracks, is a
  second route that needs no ControlNet at all.

## 4. Will base + IP-Adapter + ControlNet fit 8 GB VRAM / 16 GB RAM?

**Measured by others, same GPU class (Fact):** an SDXL fine-tune in ComfyUI
on an RTX 4060 Laptop 8 GB: 1024x1024, 20 steps, 15.8 s, VRAM peak about
5.6 GB; the author did not test IP-Adapter or ControlNet
([lilting.ch](https://lilting.ch/en/articles/comfyui-local-rtx4060-wai-illustrious)).
1280x704 is 0.90 megapixels against 1.05 for 1024x1024, so the base model
alone should behave the same or slightly better (Derived).

**Weights that must be on the GPU while sampling (Derived from file sizes):**

| Stack | U-Net fp16 | IP-Adapter plus | ControlNet | Sum of weights |
|---|---|---|---|---|
| base + IP-Adapter | 5.14 GB | 0.85 GB | - | 6.0 GB |
| + Canny small | 5.14 GB | 0.85 GB | 0.32 GB | 6.3 GB |
| + T2I lineart | 5.14 GB | 0.85 GB | 0.16 GB | 6.2 GB |
| + union ControlNet | 5.14 GB | 0.85 GB | 2.51 GB | 8.5 GB |

On top come activations (the measurement above implies roughly 0.5 GB beyond
the U-Net at 1024x1024; a ControlNet adds its own). The CLIP image encoder
(2.53 GB file, about half that in fp16) and the text encoders are only needed
before sampling, and the VAE only after; ComfyUI moves models on and off the
GPU by itself ("Dynamic VRAM is enabled by default", Secondary:
[eastondev.com](https://eastondev.com/blog/en/posts/ai/20260721-comfyui-low-vram-acceleration/)).

**Plainly:**

- base + IP-Adapter (what part 2 of the spike needs): **workable** on paper,
  about 6 GB of weights plus activations inside 8 GB.
- base + IP-Adapter + small Canny or T2I lineart: **tight but workable** on
  paper, about 6.3 GB of weights; little headroom, and Windows itself uses
  some VRAM.
- base + IP-Adapter + union ControlNet: **does not fit** without offloading
  (8.5 GB of weights). It will probably still run through ComfyUI's
  offloading, slower. Secondary sources budget the same stack at about
  12.9 GB without offloading and call 8 GB "challenging" (search summary of
  [synpixcloud.com](https://www.synpixcloud.com/blog/comfyui-complex-workflow-gpu-guide);
  not opened, unmeasured).
- **System RAM** is the same risk as with Wan: the 6.94 GB checkpoint, the
  2.53 GB encoder and a ControlNet all pass through 15.7 GB of RAM, with
  Windows and a browser beside them. It should be easier than the Wan runs
  (those hold a 6.27 GB text encoder plus a 5 GB model), but ComfyUI must
  not still hold the Wan models, and nothing else heavy should run.
- **No page read today measured SDXL + IP-Adapter + ControlNet on 8 GB.**
  Every verdict above is arithmetic. The spike is the measurement.

Speed (Estimate): 20 to 40 s per image for base + IP-Adapter at 20 to 30
steps, from the measured 15.8 s for the base alone; more if offloading
starts.

## 5. A small LoRA on the approved character images, later

Not done now. What it would take:

- **Tooling:** kohya-ss `sd-scripts` (a separate Python environment with
  its own PyTorch; a multi-GB install).
- **VRAM:** "8GB GPU memory is the minimum" for SDXL LoRA training, 10 GB
  recommended; for 8 GB: train the U-Net only, gradient checkpointing,
  `--cache_text_encoder_outputs` with cached latents, an 8-bit or Adafactor
  optimizer, and "lower dim (4 to 8 for 8GB GPU)"
  ([sd-scripts SDXL guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_SDXL-en.md)).
  So it is at the minimum on this laptop.
- **Data:** published examples use 10 to 13 images (Secondary: search
  summary). We have fewer real ones than "about 5": one approved turnaround
  sheet (three figures, which can be cropped into three images, at 1024x765
  for the whole sheet, so each crop is small), the shot_03 eye close-up, and
  the canonical rooftop frame from behind. The handwritten notes and the
  signature on the sheet must be cropped out or the LoRA learns them.
  Estimate: enough to test, high risk of overfitting to those exact poses.
- **Time:** Estimate, no source: one to two hours for roughly 1,000 to
  1,500 steps at batch size 1 on this GPU, plus setup. UNVERIFIED.
- **Licence:** a LoRA trained on SDXL base is a derivative under the same
  OpenRAIL++-M terms (Derived from the licence's "Derivatives of the Model").

## What part 2 would download (nothing is downloaded yet)

For candidate A, IP-Adapter only, as the spike asks:

| File | Size | Licence | Goes to |
|---|---|---|---|
| `sd_xl_base_1.0.safetensors` | 6.94 GB | OpenRAIL++-M | `ComfyUI/models/checkpoints` |
| `ip-adapter-plus_sdxl_vit-h.safetensors` | 848 MB | Apache-2.0 | `ComfyUI/models/ipadapter` |
| CLIP ViT-H image encoder `model.safetensors` | 2.53 GB | Apache-2.0 (in h94/IP-Adapter) | `ComfyUI/models/clip_vision` |
| **Total** | **about 10.3 GB** | | 101 GB free on C: |

Plus one code install: the `ComfyUI_IPAdapter_plus` custom node (GPL-3.0,
maintenance-only). Optional for a face test: `ip-adapter-plus-face_sdxl_vit-h`
(848 MB more). No ControlNet is needed for part 2.

## Open questions

1. Does IP-Adapter hold this character's face and outfit from one crop well
   enough to be useful? Unknown until the spike runs.
2. Does `ComfyUI_IPAdapter_plus` work on ComfyUI v0.38.2?
3. Which licence applies to the single-file DreamShaper XL checkpoint?
4. Is there a commercially licensed SDXL fine-tune aimed at painterly comic
   art? None found today.
5. Does FLUX.2 klein 4B have a ControlNet, and how does its native reference
   input compare with IP-Adapter for identity? Not researched beyond
   `IMAGE_BACKENDS.md`.
6. Real VRAM and RAM peaks for the full stack on this laptop.
7. Licence and size of a Lineart preprocessor model.

## Sources

- [stabilityai/stable-diffusion-xl-base-1.0, files](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main), [unet](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/unet), [LICENSE.md](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- [Lykon/dreamshaper-xl-1-0](https://huggingface.co/Lykon/dreamshaper-xl-1-0), [files](https://huggingface.co/Lykon/dreamshaper-xl-1-0/tree/main); [Lykon/DreamShaper files](https://huggingface.co/Lykon/DreamShaper/tree/main)
- [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter), [sdxl_models](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models), [image_encoder](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder)
- [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)
- [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [diffusers/controlnet-canny-sdxl-1.0-small](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/tree/main)
- [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), [files](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/main)
- [TencentARC/t2i-adapter-lineart-sdxl-1.0](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0/tree/main)
- [lilting.ch: ComfyUI on an RTX 4060 Laptop](https://lilting.ch/en/articles/comfyui-local-rtx4060-wai-illustrious) (measured by its author)
- [eastondev.com: ComfyUI low-VRAM optimisation](https://eastondev.com/blog/en/posts/ai/20260721-comfyui-low-vram-acceleration/) (secondary)
- [kohya-ss/sd-scripts: SDXL training](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_SDXL-en.md)
- Search summaries only, pages not opened: [synpixcloud.com](https://www.synpixcloud.com/blog/comfyui-complex-workflow-gpu-guide), [arXiv 2504.05838](https://arxiv.org/pdf/2504.05838), [comfy.org FLUX workflows](https://comfy.org/workflows/model/flux/)
