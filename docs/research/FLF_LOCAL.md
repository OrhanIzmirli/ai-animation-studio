# First/last-frame (FLF) and keyframe video models with open weights - local feasibility on 8 GB VRAM / 16 GB RAM

Date read: 2026-10-05

This field changes fast. Every statement below reflects what the linked page said on the date read. Model lists, licences and file listings can change without notice.

## How to read this document

- Nothing was run. No model, no ComfyUI, no download of weights. Everything is desk research from web pages.
- Target machine for the "fits?" question: laptop, NVIDIA RTX 4060 8 GB VRAM, 16 GB system RAM, Windows 11, ComfyUI.
- Pages were read through an automated fetch-and-extract tool. Quoted strings are as returned by that tool; a few may differ in whitespace or punctuation from the live page. File sizes come from Hugging Face file-listing API pages (bytes), converted here to decimal GB (1 GB = 10^9 bytes).
- "Measured" = a source reports a number it observed. "Derived" = arithmetic done in this document from file sizes; it is not a measurement. "Secondary" = blog / forum / third-party site. "Anecdotal" = individual user report.
- "UNVERIFIED:" = could not be confirmed from a page that was actually opened.
- General memory arithmetic used in "Derived" lines: a diffusion model file larger than free VRAM cannot sit fully on the GPU, so the remainder has to live in system RAM (ComfyUI offloading, GGUF loader, block swap). The text encoder also has to be loaded (RAM or VRAM) before sampling. With 16 GB RAM, Windows and ComfyUI already use part of it, so (model part not in VRAM) + (text encoder) + (VAE) + (OS) has to stay below 16 GB or the system pages to disk.

---

## 1. Wan 2.1 FLF2V 14B (720p)

### What it is, release date, links
- Official first-last-frame model of Wan 2.1. Repo news entry: "Apr 17, 2025: We introduce Wan2.1 FLF2V with its inference code and weights" ([Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)). Model card: [Wan-AI/Wan2.1-FLF2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P).
- The Hugging Face org listing shows `Wan-AI/Wan2.1-FLF2V-14B-720P` created 2025-04-17 ([HF API listing of Wan-AI models](https://huggingface.co/api/models?author=Wan-AI&sort=createdAt&direction=-1&limit=40)).

### First/last-frame support
- Native: the model is trained for first+last frame. The Wan2.1 README table lists FLF2V-14B as 720P only (480P not supported) ([Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)).
- ComfyUI native: the official tutorial uses the `WanFirstLastFrameToVideo` node with `wan2.1_flf2v_720p_14B_fp16.safetensors` or `wan2.1_flf2v_720p_14B_fp8_e4m3fn.safetensors`, `umt5_xxl_*`, `wan_2.1_vae.safetensors` and `clip_vision_h.safetensors` ([ComfyUI docs: Wan FLF2V](https://docs.comfy.org/tutorials/video/wan/wan-flf)).
- In ComfyUI source, `WanFirstLastFrameToVideo` has optional inputs `clip_vision_start_image`, `clip_vision_end_image`, `start_image`, `end_image` ([ComfyUI nodes_wan.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_wan.py)).

### Licence
- "The models in this repository are licensed under the Apache 2.0 License." ([Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)); the model card carries the same Apache 2.0 statement ([model card](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P)). ComfyUI docs also state Apache 2.0 for Wan2.1 ([ComfyUI docs: Wan FLF2V](https://docs.comfy.org/tutorials/video/wan/wan-flf)).
- No revenue threshold or territory exclusion was found on these pages.

### Sizes
- 14B parameters (model name; [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)).
- ComfyUI repackaged: fp16 32,792,693,440 bytes (32.79 GB); fp8_e4m3fn 16,397,952,536 bytes (16.40 GB) ([Comfy-Org/Wan_2.1_ComfyUI_repackaged file listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models)).
- GGUF (city96): Q3_K_S 7.93 GB, Q3_K_M 8.59 GB, Q4_K_S 10.44 GB, Q4_K_M 11.34 GB, Q5_K_M 12.75 GB, Q6_K 14.24 GB, Q8_0 18.15 GB ([city96/Wan2.1-FLF2V-14B-720P-gguf file listing](https://huggingface.co/api/models/city96/Wan2.1-FLF2V-14B-720P-gguf/tree/main)).
- Text encoder UMT5-XXL: fp16 11.37 GB, fp8 scaled 6.74 GB ([text_encoders listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders)); GGUF Q3_K_S 2.86 GB, Q4_K_M 3.66 GB, Q5_K_M 4.15 GB, Q6_K 4.67 GB, Q8_0 6.04 GB ([city96/umt5-xxl-encoder-gguf listing](https://huggingface.co/api/models/city96/umt5-xxl-encoder-gguf/tree/main)).
- VAE `wan_2.1_vae.safetensors` 0.25 GB ([vae listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae)); `clip_vision_h.safetensors` 1.26 GB ([clip_vision listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/clip_vision)).

### VRAM / RAM
- Measured: no VRAM figure for FLF2V was found on the official pages; the ComfyUI tutorial gives no measurement ([ComfyUI docs: Wan FLF2V](https://docs.comfy.org/tutorials/video/wan/wan-flf)).
- Derived: the smallest listed GGUF (Q3_K_S, 7.93 GB) is about the size of the whole 8 GB VRAM, so it cannot be fully resident together with 720p latents; part must be offloaded to RAM. Q4_K_S (10.44 GB) + UMT5 Q4_K_M (3.66 GB) + CLIP vision (1.26 GB) = 15.36 GB of files, of which at most about 6-7 GB can be on the GPU at any time. On 16 GB RAM this is at the limit and depends on the text encoder being unloaded before sampling.
- UNVERIFIED: any report of this exact model running at 720p on 8 GB VRAM + 16 GB RAM.

### Speed
- UNVERIFIED: no timing for 8 GB-class cards was found for this model.

### Resolution / fps / frames
- 720P (1280x720), 81 frames, 16 fps in the official examples ([model card](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P), [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)).
- ComfyUI docs: "small generation size may not produce good output with this model" ([ComfyUI docs: Wan FLF2V](https://docs.comfy.org/tutorials/video/wan/wan-flf)). This matters for 8 GB, where lower resolution is the usual way to fit.

### Known quality issues
- Official note: "we train our model primarily on Chinese text-video pairs. Therefore, we recommend using Chinese prompt to achieve better results." ([Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)).
- GitHub issues (user reports): "For FLF2V model, the last frame generation has abnormal scaling" ([Wan2.1 issue 342](https://github.com/Wan-Video/Wan2.1/issues/342), closed, no fix text visible); titles "Why do I get these ugly results with the wan flf2v model?" (#348), static result when first and last frame are identical (#372), "WAN2.1 FLF2V: Incorrect MASK Creation???" (#513) ([Wan2.1 issue search for FLF2V](https://github.com/Wan-Video/Wan2.1/issues?q=is%3Aissue+FLF2V)). Only #342 was opened individually.
- UNVERIFIED: behaviour on flat 2D / comic line art specifically; no primary source found.

---

## 2. Wan 2.1 Fun InP (1.3B and 14B)

### What it is, release date, links
- Alibaba PAI "Fun" fine-tunes of Wan 2.1 for image-to-video with start and end image. Model card: [alibaba-pai/Wan2.1-Fun-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP). Code: [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun).
- UNVERIFIED: release date (not shown on the pages read; the earliest related GitHub issue found is dated 2025-03-22, see below).

### First/last-frame support
- Native to the InP variant: the model card table describes multi-resolution training with support for first and last image prediction ([model card](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP)); VideoX-Fun README: "supports start-end image prediction" ([VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)).
- ComfyUI native: `WanFunInpaintToVideo` node with `wan2.1_fun_inp_1.3B_bf16.safetensors` or `Wan2.1-Fun-14B-InP.safetensors`, UMT5, `wan_2.1_vae`, `clip_vision_h` ([ComfyUI docs: Wan Fun InP](https://docs.comfy.org/tutorials/video/wan/fun-inp)). Node inputs include optional `start_image` and `end_image` ([ComfyUI nodes_wan.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_wan.py)).
- Kijai's ComfyUI-WanVideoWrapper lists "WanVideoFun" among supported models ([ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)).

### Licence
- Model card: "Apache License (Version 2.0)" ([model card](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP)).

### Sizes
- Model card storage (whole repo incl. encoders): 1.3B InP 19.0 GB, 14B InP 47.0 GB ([model card](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP)).
- ComfyUI repackaged diffusion model only: `wan2.1_fun_inp_1.3B_bf16.safetensors` 3,128,957,992 bytes (3.13 GB) ([Wan 2.1 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models)).
- ComfyUI docs: 14B version "Model size reaches 32GB+" ([ComfyUI docs: Wan Fun InP](https://docs.comfy.org/tutorials/video/wan/fun-inp)).
- UNVERIFIED: GGUF file sizes for Wan2.1-Fun-14B-InP (no listing opened).
- Text encoder / VAE / CLIP vision: same files as section 1.

### VRAM / RAM
- Measured: none found. ComfyUI docs only say the 1.3B version is "Suitable for local deployment and quick inference with lower VRAM requirements" ([ComfyUI docs: Wan Fun InP](https://docs.comfy.org/tutorials/video/wan/fun-inp)).
- Derived (1.3B): 3.13 GB model fits fully in 8 GB VRAM. RAM side: UMT5 GGUF Q4_K_M 3.66 GB or fp8 6.74 GB + CLIP vision 1.26 GB; this is within 16 GB RAM.
- Derived (14B): same order of size as section 1; same 16 GB RAM pressure.

### Speed
- UNVERIFIED: no timing on 8 GB-class cards found.

### Resolution / fps / frames
- Trained on 81 frames at 16 fps; resolutions 512, 768, 1024 ([model card](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP)). ComfyUI docs: "Supports generating videos at 512x512, 768x768, 1024x1024 and other resolutions" ([ComfyUI docs: Wan Fun InP](https://docs.comfy.org/tutorials/video/wan/fun-inp)).

### Known quality issues
- User report (GitHub issue, open, no maintainer reply visible): with Wan2.1-Fun (1.3B and 14B) "the last few frames of the generated videos will become black gradually" ([WanVideoWrapper issue 339](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/339), 2025-03-28).
- User report titles: "Endframe workflow jumping in last few frames, anyone know how to fix?" (#295, 2025-03-22) ([WanVideoWrapper issue search](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues?q=is%3Aissue+last+frame+color)); only the title was read.
- User report (Fun Control, not InP): "after adding an end frame the last few frame are washed out, like almost completely white" ([WanVideoWrapper issue 872](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/872), 2025-07-26, open).

---

## 3. Wan 2.1 VACE (1.3B / 14B) with start/end frames

### What it is, release date, links
- "May 14, 2025: We introduce Wan2.1 VACE, an all-in-one model for video creation and editing" ([Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)). Model card: [Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B).

### First/last-frame support
- Via VACE masked-video conditioning. ComfyUI docs list a "VACE First-Last Frame" workflow: "Generate smooth video transitions by defining start and end frames. Supports custom keyframe sequences." Core node `WanVaceToVideo`; constraints "length-1 is divisible by 4" and "Batch_size = length - 2" (the in-between frames are supplied as a control batch). The docs mark the detailed section "[To be updated]" ([ComfyUI docs: VACE](https://docs.comfy.org/tutorials/video/wan/vace)).
- `WanVaceToVideo` has no dedicated `end_image` input in ComfyUI source; the end frame is passed through the control video/mask inputs ([ComfyUI nodes_wan.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_wan.py)).
- Kijai wrapper lists VACE as supported ([ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)).

### Licence
- "The model is open-sourced under the Apache-2.0 license and can be used for personal or commercial purposes." ([ComfyUI docs: VACE](https://docs.comfy.org/tutorials/video/wan/vace)); model card: Apache 2.0 ([model card](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)).

### Sizes
- File listing: `wan2.1_vace_1.3B_fp16.safetensors` 4.31 GB; `wan2.1_vace_14B_fp16.safetensors` 34.68 GB ([Wan 2.1 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models)).
- Conflict: ComfyUI docs describe the same files as "~7GB" (1.3B) and "~32GB" (14B) ([ComfyUI docs: VACE](https://docs.comfy.org/tutorials/video/wan/vace)). The byte counts from the file listing are used here.
- GGUF 14B (QuantStack): Q3_K_S 7.84 GB, Q4_K_S 10.55 GB, Q4_K_M 11.64 GB, Q5_K_M 13.04 GB, Q6_K 14.52 GB, Q8_0 18.66 GB ([QuantStack/Wan2.1_14B_VACE-GGUF listing](https://huggingface.co/api/models/QuantStack/Wan2.1_14B_VACE-GGUF/tree/main)).

### VRAM / RAM
- Measured (official, Wan T2V-1.3B base, quoted on the VACE card): "only 8.19 GB VRAM" ([model card](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)). This is the upstream script figure, not ComfyUI with offloading.
- Derived (1.3B): 4.31 GB model fits in 8 GB VRAM; text encoder as in section 2; within 16 GB RAM.
- Derived (14B): GGUF Q3_K_S 7.84 GB to Q4_K_M 11.64 GB; same constraints as section 1.

### Speed
- Measured (ComfyUI docs, RTX 4090, VACE 14B): 720x1280, 81 frames about 40 minutes; 640x640, 49 frames about 7 minutes ([ComfyUI docs: VACE](https://docs.comfy.org/tutorials/video/wan/vace)).
- UNVERIFIED: timings on 8 GB-class cards.

### Resolution / fps / frames
- 1.3B: about 81 frames at 480x832; 720P "generally less stable". 14B: 480P and 720P ([model card](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B), [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1)).

### Known quality issues
- User remark in an issue: VACE considered as alternative for start/end frames but "causes more drastic image changes" (paraphrase of the reporter) ([WanVideoWrapper issue 872](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/872)). Anecdotal.
- UNVERIFIED: line-art specific reports.

---

## 4. Wan 2.2 I2V-A14B (high noise + low noise) with first-last frame

### What it is, release date, links
- Wan 2.2 released Jul 28, 2025; I2V-A14B is described as "Image-to-Video MoE model, supports 480P & 720P" ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)). It is two 14B expert models (high noise, low noise), shipped as two files ([Wan 2.2 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)).

### First/last-frame support
- Not mentioned in the official Wan2.2 README (no FLF task listed) ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)).
- ComfyUI native: the official ComfyUI tutorial has a section "Wan2.2 14B FLF2V Workflow Example" that uses the same I2V high/low-noise models with the `WanFirstLastFrameToVideo` node ([ComfyUI docs: Wan2.2](https://docs.comfy.org/tutorials/video/wan/wan2_2)). So this is a ComfyUI-level use of the I2V model, not a separately trained FLF checkpoint.
- A ComfyUI template "Wan 2.2 14B First-Last Frame to Video" exists ([comfy.org template](https://comfy.org/workflows/video_wan2_2_14B_flf2v-7016f027bcf1/)); found through search, page not opened in detail.

### Licence
- "The Wan2.2 series models are based on the Apache 2.0 open source license and support commercial use." ([ComfyUI docs: Wan2.2](https://docs.comfy.org/tutorials/video/wan/wan2_2)); repo: Apache 2.0 ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)).
- 4-step LoRAs (lightx2v Wan2.2-Lightning): "Apache 2.0 License" ([lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)).

### Sizes
- Each of the two models: fp16 28.58 GB; fp8_scaled 14.29 GB ([Wan 2.2 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)).
- GGUF, HighNoise folder (QuantStack): Q2_K 5.30 GB, Q3_K_S 6.52 GB, Q3_K_M 7.18 GB, Q4_K_S 8.75 GB, Q4_K_M 9.65 GB, Q5_K_M 10.79 GB, Q6_K 12.00 GB, Q8_0 15.41 GB ([QuantStack/Wan2.2-I2V-A14B-GGUF HighNoise listing](https://huggingface.co/api/models/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main/HighNoise)). UNVERIFIED: LowNoise folder sizes (not opened; the fp8 files of both experts are byte-identical in size).
- lightx2v 4-step LoRAs: 1.23 GB each for high and low noise ([Wan 2.2 loras listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras)).
- Text encoder and VAE: UMT5 and `wan_2.1_vae` as in section 1 ([ComfyUI docs: Wan2.2](https://docs.comfy.org/tutorials/video/wan/wan2_2)).

### VRAM / RAM
- Official script figure: T2V/I2V A14B "can run on a GPU with at least 80GB VRAM" ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)) - not the ComfyUI offloaded case.
- Measured (Secondary, RTX 4060 8 GB, but 32 GB RAM, Windows 11, single-image I2V with a merged "14B Rapid" all-in-one 22 GB checkpoint, not FLF): "4569MB of the 14B model in VRAM, 11067MB (~11GB) offloaded to main memory" ([lilting.ch article](https://lilting.ch/en/articles/wan22-comfyui-rtx4060-i2v)).
- Derived: with 16 GB RAM, about 11 GB of offloaded model weights plus a text encoder leaves almost nothing for the OS; the two-expert setup needs both models read in turn (2 x 8.75 GB at Q4_K_S = 17.5 GB of files, more than total RAM), so one expert has to be dropped and re-read from disk or paged for every clip.
- Secondary: a community workflow README lists system RAM "Minimum: 16GB, Recommended: 32GB" for Wan 2.2 GGUF Q3-Q5 with a Q3/Q4 text encoder ([Cordux/ComfyUI-Wan2.2-workflow](https://github.com/Cordux/ComfyUI-Wan2.2-workflow)).
- User bug report (12 GB card, 32 GB RAM): "Cumulative SysRAM leak builds over repeated generations ... system RAM climbs steadily (20-30GB+ reserved after 3-5 runs)" with Wan2.2 GGUF at 1280x720, 81 frames ([ComfyUI issue 11775](https://github.com/Comfy-Org/ComfyUI/issues/11775), opened 2026-01-10, closed).

### Speed
- Measured (Secondary, RTX 4060 8 GB, 32 GB RAM): 480x480, 33 frames, 4 steps, CFG 1.0, "14B Rapid" merged checkpoint with `--lowvram`: 111 seconds per clip total, 45 seconds of it diffusion; plain I2V, not FLF ([lilting.ch article](https://lilting.ch/en/articles/wan22-comfyui-rtx4060-i2v)).
- Measured (Secondary, RTX 3050 6 GB): about 4.6 minutes for 16 frames at 512x512, 4 steps, GGUF; the same README's RTX 4060 8 GB figure (about 2.5 minutes per second of video) is labelled expected, not measured ([Cordux/ComfyUI-Wan2.2-workflow](https://github.com/Cordux/ComfyUI-Wan2.2-workflow)).
- Reference point (official ComfyUI docs, RTX 4090D, Fun InP A14B, 640x640, 81 frames): fp8_scaled about 520 s; with 4-step LoRA about 79 s on second run ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- UNVERIFIED: a measured FLF run of I2V-A14B on an 8 GB card with 16 GB RAM.

### Resolution / fps / frames
- 480P and 720P ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)); 16 fps, e.g. 81 frames for 5 s ("5*16 = 80") in ComfyUI docs for the A14B family ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).

### Known quality issues
- 4-step LoRA trade-off, official ComfyUI wording: "The acceleration with LoRA is significant but the video dynamic is lost" ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- Lightning model card limitation: "When the video contains elements with extremely large motion, the generated results may include artifacts." ([lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)).
- User report (open issue): with Wan 2.2 Lightning I2V 4 steps and first+last frame, "a generation beyond 120 frames does not smoothly transition between the first and last ... Almost as if it MUST reach the last frame around 100 frames, and then move back to the first frame." ([WanVideoWrapper issue 1342](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1342), 2025-10-02).
- User report title: "Wan 2.2 I2V Looping same image Start-End step - Color contrast increase over time" (#1541, 2025-10-24, open) ([WanVideoWrapper issue search](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues?q=is%3Aissue+last+frame+color)); only the title was read.
- UNVERIFIED: reports specific to comic / flat 2D line art (search results mentioned a community node "ComfyUI-PainterI2V" aimed at slow motion with 4-step LoRAs, page not opened).

---

## 5. Wan 2.2 Fun InP (A14B and 5B)

### What it is, release date, links
- Alibaba PAI Fun fine-tunes of Wan 2.2. Model cards: [alibaba-pai/Wan2.2-Fun-A14B-InP](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP), [alibaba-pai/Wan2.2-Fun-5B-InP](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP). Code: [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun).
- UNVERIFIED: release dates (not on the pages read).

### First/last-frame support
- Native to the InP variants: both model cards describe multi-resolution training with support for first and last image prediction ([A14B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP), [5B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP)).
- ComfyUI native, A14B: `WanFunInpaintToVideo` with `wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors` + `wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors`, optionally the `wan2.2_i2v_lightx2v_4steps_lora_v1_*` LoRAs ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- ComfyUI native, 5B: an official template "Wan 2.2 5B Fun Inpaint" exists, described as video generation "from start and end frames" ([comfy.org template: Wan 2.2 5B Fun Inpaint](https://comfy.org/workflows/video_wan2_2_5B_fun_inpaint-00d3d57a5195/)); the repackaged file is `wan2.2_fun_inpaint_5B_bf16.safetensors` ([Wan 2.2 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)). UNVERIFIED: which exact node the 5B template uses for the end image (the template page did not show node names).

### Licence
- Both cards: apache-2.0 ([A14B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP), [5B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP)). ComfyUI docs: "The model is released under the Apache 2.0 license and supports commercial use." ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).

### Sizes
- A14B: bf16 28.58 GB per expert, fp8_scaled 14.29 GB per expert ([Wan 2.2 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)). GGUF HighNoise: Q2_K 5.31 GB, Q3_K_S 6.52 GB, Q4_K_S 8.76 GB, Q4_K_M 9.66 GB, Q5_K_M 10.80 GB, Q6_K 12.01 GB, Q8_0 15.42 GB ([QuantStack/Wan2.2-Fun-A14B-InP-GGUF HighNoise listing](https://huggingface.co/api/models/QuantStack/Wan2.2-Fun-A14B-InP-GGUF/tree/main/HighNoise)).
- 5B: bf16 10.00 GB ([Wan 2.2 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)). GGUF: Q2_K 1.86 GB, Q3_K_S 2.30 GB, Q4_K_S 3.12 GB, Q4_K_M 3.44 GB, Q5_K_M 3.82 GB, Q6_K 4.22 GB, Q8_0 5.41 GB ([QuantStack/Wan2.2-Fun-5B-InP-GGUF listing](https://huggingface.co/api/models/QuantStack/Wan2.2-Fun-5B-InP-GGUF/tree/main)).
- For comparison, the base TI2V-5B GGUF Q8_0 is 5.40 GB ([QuantStack/Wan2.2-TI2V-5B-GGUF listing](https://huggingface.co/api/models/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main)), i.e. the Fun 5B InP Q8_0 is the same size class as the model the project already runs.
- Model card storage (full repos): 5B 23.0 GB; A14B 47.0 GB ([5B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP), [A14B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP)).
- VAE: the 5B family uses `wan2.2_vae.safetensors`, 1.41 GB ([vae listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae), [ComfyUI docs: Wan2.2](https://docs.comfy.org/tutorials/video/wan/wan2_2)); the A14B family uses `wan_2.1_vae.safetensors` ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).

### VRAM / RAM
- Measured (official ComfyUI docs, A14B, RTX 4090D 24 GB, 640x640, 81 frames): fp8_scaled 83% VRAM; with 4-step LoRA 89% ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- Fun 5B card: tested hardware list starts at a 12 GB RTX 3060, with memory-saving modes `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, `sequential_cpu_offload` (VideoX-Fun scripts, not ComfyUI) ([5B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP)).
- Derived (5B): Fun 5B InP Q8_0 (5.41 GB) is 0.01 GB larger than TI2V-5B Q8_0 (5.40 GB), so memory behaviour should match the TI2V-5B Q8 setup this project already runs, plus one extra encoded end image. ComfyUI docs on the base model: "The Wan2.2 5B version should fit well on 8GB vram with the ComfyUI native offloading." ([ComfyUI docs: Wan2.2](https://docs.comfy.org/tutorials/video/wan/wan2_2)).
- Derived (A14B): same as section 4 (two experts; 16 GB RAM is the bottleneck).
- UNVERIFIED: a measured Fun 5B InP run on 8 GB VRAM / 16 GB RAM.

### Speed
- Measured (official ComfyUI docs, A14B, RTX 4090D, 640x640, 81 frames): about 524 s first run / 520 s second run at fp8_scaled; about 138 s / 79 s with the 4-step LoRA ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- UNVERIFIED: any timing for Fun 5B InP; any timing on 8 GB-class cards.
- UNVERIFIED: whether the TI2V-5B "Turbo" few-step weights/LoRA used in this project are compatible with Fun 5B InP. No page read addressed this.

### Resolution / fps / frames
- 5B InP: 121 frames, 24 fps, multi-resolution 512/768/1024 ([5B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP), [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)).
- A14B InP: 81 frames, 16 fps, multi-resolution 512/768/1024 ([A14B card](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP)).

### Known quality issues
- Official ComfyUI note for the A14B workflow with the 4-step LoRA: "The acceleration with LoRA is significant but the video dynamic is lost"; and "Please make sure the corresponding Diffusion model and LoRA are matched." ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- Secondary (Wan 2.2 TI2V-5B base, RTX 4060 8 GB, plain I2V): the author reports that 5B attempts "produced shape distortion and minimal motion" compared with a 14B distilled model (paraphrase from the extraction) ([lilting.ch article](https://lilting.ch/en/articles/wan22-comfyui-rtx4060-i2v)). Anecdotal, not FLF, not Fun InP.
- Earlier Fun-family end-frame issues (fade to black, washed-out end frames) are listed in section 2; UNVERIFIED whether they apply to the 2.2 Fun InP models.

---

## 6. Wan 2.2 TI2V-5B (the model already in use): end frame?

- Official description: "natively supports both text-to-video and image-to-video tasks within a single unified framework"; 720P at 24 fps; VAE compression 4x16x16; no first-last-frame task is listed ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)).
- ComfyUI source: the 5B latent node `Wan22ImageToVideoLatent` accepts only `start_image`; among the Wan nodes only `WanFirstLastFrameToVideo` and `WanFunInpaintToVideo` expose `end_image` ([ComfyUI nodes_wan.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_wan.py)).
- The end-frame route for the 5B architecture found in primary sources is the separate Fun 5B InP checkpoint (section 5).
- UNVERIFIED: whether Kijai's WanVideoWrapper offers an end-image input that works with the plain TI2V-5B checkpoint (the wrapper README read does not discuss end images; [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)).
- Official VRAM figure for the upstream script: "at least 24GB VRAM"; ComfyUI docs: "should fit well on 8GB vram with the ComfyUI native offloading" ([Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2), [ComfyUI docs: Wan2.2](https://docs.comfy.org/tutorials/video/wan/wan2_2)).

---

## 7. Wan 2.2 VACE-Fun (A14B)

- What: "Control weights for Wan2.2 trained using the VACE scheme" ([VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)); model card [alibaba-pai/Wan2.2-VACE-Fun-A14B](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B), base model Wan2.2-T2V-A14B. UNVERIFIED: release date.
- FLF: the card lists first and last frame prediction, reference-image subject control and Canny/Depth/Pose/MLSD/trajectory control ([model card](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)).
- Licence: "Apache License (Version 2.0)" ([model card](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)).
- Sizes: repo storage 64.0 GB ([model card](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)); ComfyUI repackaged per expert: bf16 34.68 GB, fp8_scaled 17.35 GB ([Wan 2.2 repackaged listing](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)). UNVERIFIED: GGUF sizes.
- VRAM/RAM, Derived: larger than I2V-A14B per expert (17.35 GB vs 14.29 GB at fp8); two experts; harder than section 4 on 16 GB RAM. No measurement found.
- Resolution/frames: trained at 81 frames, 16 fps, 512/768/1024 ([model card](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B)).
- Quality issues: UNVERIFIED (none read).

---

## 8. Wan 2.5 / 2.6 / 2.7 / 3.x - are weights open?

- Primary check: the Wan-AI organisation listing on Hugging Face, sorted by creation date, contains Wan2.1 and Wan2.2 repos only (newest entries: `Wan2.2-Animate-2-14B-Distilled-Diffusers` and `Wan2.2-Animate-2-14B-Diffusers` 2026-08-06, `Wan2.2-Animate-2-14B` 2026-07-14, `Wan-Dancer-14B` 2026-07-10); no repo named Wan2.5, Wan2.6, Wan2.7 or Wan3 appears in the 27 entries returned ([HF API listing of Wan-AI models](https://huggingface.co/api/models?author=Wan-AI&sort=createdAt&direction=-1&limit=40)).
- Secondary (search-result snippets, pages not opened): Wan 2.5-Preview (2025-09-24), Wan 2.6 (2025-12-16) and Wan 2.7 (April 2026) were released as API products without published weights; several SEO pages claim otherwise ([howaiworks.ai](https://howaiworks.ai/blog/alibaba-wan-open-weights-stopped-at-2-2), [wan27.org on 2.6](https://www.wan27.org/blog/wan-2-6-open-source-guide), [wan27.org on 3.0](https://wan27.org/blog/wan-3-open-source)).
- UNVERIFIED: status of "Wan 3.0" (only a search-result title was seen).
- UNVERIFIED: whether `Wan2.2-Animate-2` or `Wan-Dancer-14B` have any first-last-frame capability (names only were read; they appear to be character-animation models).

---

## 9. LTX-Video 0.9.x (2B and 13B)

### What it is, release date, links
- Lightricks DiT video model. Repo: [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video); model card: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video). News entries include "May 14th, 2025: New 13B distilled model ltxv-13b-0.9.7-distilled" and "July 16th, 2025: LTX-Video now supports up to 60 seconds of video"; current configs are `ltxv-13b-0.9.8-dev`, `ltxv-13b-0.9.8-distilled`, `ltxv-2b-0.9.8-distilled` and fp8 variants ([LTX-Video GitHub](https://github.com/Lightricks/LTX-Video)).

### First/last-frame / keyframe support
- Native: "The model supports image-to-video, multi-keyframe conditioning, keyframe-based animation, video extension (both forward and backward), video-to-video transformations" ([LTX-Video GitHub](https://github.com/Lightricks/LTX-Video)). "You can now generate a video conditioned on a set of images and/or short video segments. Simply provide a list of paths to the images or video segments you want to condition on, along with their target frame numbers." ([LTX-Video model card](https://huggingface.co/Lightricks/LTX-Video)).
- ComfyUI native nodes: `LTXVAddGuide` takes an image, a `frame_idx` ("Frame index to start the conditioning at ... Negative values are counted from the end of the video.") and a strength; also `LTXVImgToVideo`, `LTXVImgToVideoInplace`, `LTXVAddLatentGuide`, `LTXVCropGuides` ([ComfyUI nodes_lt.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_lt.py)). So first, last and intermediate keyframes can be attached as guides at arbitrary frame indices.

### Licence
- Model card licence tag: "other"; 2B v0.9.6 onwards and all 13B versions point to `LTX-Video-Open-Weights-License-0.X.txt` ([LTX-Video model card](https://huggingface.co/Lightricks/LTX-Video)).
- Licence text ("LTXV Open Weights License 0.X", dated April 15, 2025): grants a "non-exclusive, worldwide, non-transferable and royalty-free limited license"; "entities with annual revenues of at least $10,000,000 (the 'Commercial Entities')" must obtain a separate paid Commercial Use Agreement; unauthorised commercial use incurs liquidated damages "equal to double the amount" owed; "Licensor claims no rights in the Output you generate using the Model"; Attachment A lists use restrictions (among them: undisclosed AI generation, impersonation/deepfakes) ([LTX-Video-Open-Weights-License-0.X.txt](https://huggingface.co/Lightricks/LTX-Video/raw/main/LTX-Video-Open-Weights-License-0.X.txt)).
- Licence trap: not Apache/MIT; a revenue threshold applies to the using entity; Attachment A restrictions apply to outputs' use. The exact wording of the "undisclosed AI generation" item was not quoted by the extraction - read the licence file directly before relying on it.

### Sizes
- `ltxv-2b-0.9.8-distilled.safetensors` 6.34 GB; `ltxv-2b-0.9.8-distilled-fp8.safetensors` 4.46 GB; `ltxv-13b-0.9.8-dev.safetensors` and `-distilled.safetensors` 28.58 GB each; fp8 versions 15.69 GB each; spatial upscaler 0.51 GB; temporal upscaler 0.52 GB ([Lightricks/LTX-Video file listing](https://huggingface.co/api/models/Lightricks/LTX-Video/tree/main)).
- UNVERIFIED: text encoder (T5-XXL) file size and GGUF sizes for the 13B model (not opened).

### VRAM / RAM
- Measured: none found on the official pages read; the card only says the distilled variants need "less VRAM" ([LTX-Video-0.9.8-13B-distilled card](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled)).
- Derived: the 2B distilled fp8 file (4.46 GB) fits fully in 8 GB VRAM; the 13B fp8 file (15.69 GB) is about the size of total system RAM, so most of it would sit in RAM with nothing left for the text encoder.
- UNVERIFIED: measured 8 GB / 16 GB results.

### Speed
- UNVERIFIED: no timing on 8 GB-class cards was read.

### Resolution / fps / frames
- "The model works on resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1 (e.g. 257)"; "works best on resolutions under 720 x 1280 and number of frames below 257" ([LTX-Video model card](https://huggingface.co/Lightricks/LTX-Video)). The 13B distilled card describes 30 FPS at 1216x704 ([LTX-Video-0.9.8-13B-distilled card](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled)).

### Known quality issues
- Model card limitations: may fail to match prompts; prompt following depends heavily on phrasing style ([LTX-Video model card](https://huggingface.co/Lightricks/LTX-Video)).
- UNVERIFIED: keyframe-specific or line-art-specific defects (no source read).

---

## 10. LTX-2 (19B), LTX-2.3 (22B), LTX-2.5 (22B)

### What it is, release date, links
- "LTX-2 is the first DiT-based audio-video foundation model"; the repo documents LTX-2 (original), LTX-2.3 (legacy) and LTX-2.5 ("LTX-2.5 is the recommended model") ([Lightricks/LTX-2 GitHub](https://github.com/Lightricks/LTX-2)).
- Hugging Face creation dates: `Lightricks/LTX-2.3` 2026-03-04; `Lightricks/LTX-2.5` 2026-07-23; `Lightricks/LTX-2.5-Diffusers` 2026-07-26 ([HF API listing of Lightricks models](https://huggingface.co/api/models?author=Lightricks&sort=createdAt&direction=-1&limit=40)).
- Conflict on LTX-2.5 release date: a secondary article gives August 11, 2026 ([magica.com](https://magica.com/news/ltx-2-5-open-weight-video-world-model)); the LTX-2.x licence is effective August 11, 2026 ([LICENSE-2_x](https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2_x)); the HF repo was created 2026-07-23 (listing above); the extraction of the model card returned "January 6, 2026", which is the date of the arXiv preprint 2601.03233 and matches the first LTX-2 open-weights release ("LTX-2 Community License Agreement" dated January 5, 2026, [scancode licence DB](https://scancode-licensedb.aboutcode.org/ltx-2-cla-2026.html)), not LTX-2.5.
- Model card: [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5).

### First/last-frame / keyframe support
- Upstream pipelines include `KeyframeInterpolationPipeline` ("Interpolate between keyframe images"), which uses the full (dev) model plus the spatial upscaler, not the distilled model ([LTX-2 README](https://raw.githubusercontent.com/Lightricks/LTX-2/main/README.md)).
- Model card capability list: "image-to-video with frame-specific conditioning" ([LTX-2.5 card](https://huggingface.co/Lightricks/LTX-2.5)). The hosted-API docs list a `last_frame_uri` parameter for image-to-video ([docs.ltx.io LTX-2.5](https://docs.ltx.io/models/ltx-2-5)) - that is the API, not proof of a local workflow.
- ComfyUI: "Official LTX-2.5 workflow templates ship in ComfyUI" with dedicated int8/convrot checkpoints ([LTX-2.5 card](https://huggingface.co/Lightricks/LTX-2.5)); ComfyUI's LTX-2 tutorial lists "Keyframe-driven generation: Interpolate between keyframe images" as a feature ([ComfyUI docs: LTX-2](https://docs.comfy.org/tutorials/video/ltx/ltx-2)); the guide nodes in section 9 (`LTXVAddGuide` with `frame_idx`) are in ComfyUI core ([ComfyUI nodes_lt.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_lt.py)).
- The Lightricks custom-node repo README (covering LTX-2.3 and legacy LTX-2.0) lists no dedicated keyframe / first-last-frame example workflow among its 2.3 workflows ([Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)).
- UNVERIFIED: a ready-made first+last-frame ComfyUI workflow for LTX-2.5 with GGUF weights, and whether the distilled 2.5 model accepts end-frame guides with good results.

### Licence
- LTX-2 (January 2026): "LTX-2 Community License Agreement", dated January 5, 2026: "Non-exclusive, worldwide, non-transferable and royalty-free limited license"; "Entities with annual revenues of at least $10,000,000 (the 'Commercial Entities') are required to obtain a paid commercial use license"; "Except as set forth herein, Licensor claims no rights in the Output you generate using LTX-2. You are accountable for input you insert into LTX-2, the Output you generate and its subsequent uses."; Attachment A prohibits use "To train, improve, or fine-tune any other machine learning model, artificial intelligence system, or competing model, except for Derivatives of LTX-2" and use "in any product, service, or application that directly competes with Licensor's commercial products or services" ([scancode licence DB copy of the LTX-2 licence](https://scancode-licensedb.aboutcode.org/ltx-2-cla-2026.html)).
- LTX-2.5: "LTX-2.x Community License Agreement", effective August 11, 2026, covering "all LTX-2.5 versions released since August 11, 2026, and all future releases of LTX-2.x under this license": same $10,000,000 threshold; "When determining whether an Entity meets any threshold under this Agreement (including revenue thresholds in Section 2.1), all subsidiaries, affiliates, and companies under common Control shall be considered collectively."; same Output clause; competing-product restriction extended to anything that "is designed to replace or substitute Licensor's offerings"; no territory limit found (licence is "worldwide") ([LICENSE-2_x](https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2_x)). HF licence tag: "ltx-2.x-community-license-agreement" ([LTX-2.5 card](https://huggingface.co/Lightricks/LTX-2.5)).
- Could not be opened: [ltx.io/model/license](https://ltx.io/model/license) (parse error), [HF LICENSE file of LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/raw/main/LICENSE) (HTTP 401), [docs.ltx.video ComfyUI page](https://docs.ltx.video/open-source-model/integration-tools/comfy-ui) (HTTP 403), [ltx.io consumer-GPU blog](https://ltx.io/blog/how-to-run-ltx-2-on-consumer-gpus) (parse error).
- Licence traps: revenue threshold counted across affiliates; Attachment A use restrictions; competing-product clause; the licence text was read through an extraction tool, so the full Attachment A must be read directly.

### Sizes
- LTX-2 (19B): dev 43.29 GB; dev-fp8 27.08 GB; dev-fp4 19.99 GB; distilled 43.29 GB; distilled-fp8 27.08 GB; distilled LoRA 7.67 GB ([Lightricks/LTX-2 file listing](https://huggingface.co/api/models/Lightricks/LTX-2/tree/main)). Gemma 3 12B text encoder (ComfyUI repack): full 24.38 GB; fp8_scaled 13.21 GB; fp4_mixed 9.45 GB ([Comfy-Org/ltx-2 text_encoders listing](https://huggingface.co/api/models/Comfy-Org/ltx-2/tree/main/split_files/text_encoders)).
- LTX-2.3 (22B): dev and distilled 46.15 GB each ([Lightricks/LTX-2.3 file listing](https://huggingface.co/api/models/Lightricks/LTX-2.3/tree/main)).
- LTX-2.5 (22B): dev and distilled transformer bf16 42.02 GB each; ComfyUI int8-convrot 21.50 GB each; distilled nvfp4 18.72 GB ([LTX-2.5 diffusion_models listing](https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/main/diffusion_models)). Text encoder "Gemma 4 12B, fine-tuned for LTX" ([LTX-2 GitHub](https://github.com/Lightricks/LTX-2)): bf16 26.26 GB; int8-convrot 15.37 GB ([LTX-2.5 text_encoders listing](https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/main/text_encoders)).
- LTX-2.5 community GGUF (distilled): Q2_K 8.83 GB, Q3_K_M 11.53 GB, Q4_K_S 13.85 GB, Q4_K_M 15.09 GB, Q8_0 23.63 GB ([realrebelai/LTX-2.5_GGUFs listing](https://huggingface.co/api/models/realrebelai/LTX-2.5_GGUFs/tree/main)); another repo: Q2_K 7.33 GB, Q3_K_S 9.45 GB, Q4_K_M 12.22 GB, Q8_0 22.66 GB ([agosh/LTX-2.5-Comfy-GGUF listing](https://huggingface.co/api/models/agosh/LTX-2.5-Comfy-GGUF/tree/main)). The two repos give different sizes for the same quant names.
- Secondary: community Gemma 4 GGUF encoders range from 5.96 GB (Q2_K) to 9.51 GB (Q5_K_M) ([ltxworkflow.com, third-party, "Not affiliated with Lightricks"](https://ltxworkflow.com/guide/ltx-2-5-vram-requirements)).

### VRAM / RAM
- Official custom-node README (LTX-2.3): "CUDA-compatible GPU with 32GB+ VRAM", "100GB+ free disk space"; low-VRAM loader nodes and `--reserve-vram 5` are mentioned ([Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)).
- Secondary (third-party site): "No official combination fits" a 16 GB card for LTX-2.5; smallest official pair is nvfp4 transformer 18.72 GB + int8 Gemma 4 15.37 GB = about 34.09 GB ([ltxworkflow.com](https://ltxworkflow.com/guide/ltx-2-5-vram-requirements)).
- Anecdotal (HF discussion, LTX-2 19B, Feb 2026): a laptop with RTX 3050 6 GB and 44 GB RAM ran 1280x736, 10 seconds in "less than 15 minutes"; the thread's summary is that LTX-2 is "very vram friendly but it's hungry with RAM" and 32 GB+ RAM is recommended ([Kijai/LTXV2_comfy discussion 32](https://huggingface.co/Kijai/LTXV2_comfy/discussions/32)).
- Derived: for LTX-2.5 the smallest files found are transformer Q2_K 7.33 GB + Gemma 4 Q2_K 5.96 GB = 13.29 GB before VAE, audio VAE and OS. With 8 GB VRAM and 16 GB RAM the text encoder alone takes over a third of RAM while encoding, and the transformer must be mostly offloaded; this is at or beyond the limit of a 16 GB machine, at the lowest quantisation level. LTX-2 19B is in the same class (fp4 transformer 19.99 GB; Gemma 3 fp4_mixed 9.45 GB).
- UNVERIFIED: any report of LTX-2 / 2.3 / 2.5 running with only 16 GB system RAM.

### Speed
- Anecdotal: the RTX 3050 6 GB / 44 GB RAM report above (10 s at 1280x736 in under 15 minutes) ([Kijai/LTXV2_comfy discussion 32](https://huggingface.co/Kijai/LTXV2_comfy/discussions/32)).
- UNVERIFIED: timings on 8 GB cards with 16 GB RAM; timings for keyframe workflows.

### Resolution / fps / frames
- Frame count must satisfy `num_frames % 8 == 1`; width and height divisible by 32 ([LTX-2.5 card](https://huggingface.co/Lightricks/LTX-2.5)). Quick start uses 121 frames; DFR defaults "1024x1536 at 24 fps" ([LTX-2 GitHub](https://github.com/Lightricks/LTX-2)).

### Known quality issues
- Model card: "prompt adherence varies by style" (paraphrased limitation) ([LTX-2.5 card](https://huggingface.co/Lightricks/LTX-2.5)).
- UNVERIFIED: keyframe-specific and line-art-specific defects; effect of Q2/Q3 quantisation on line quality.

---

## 11. FramePack / FramePack-F1 (HunyuanVideo-based)

### What it is, release date, links
- "Official implementation and desktop software for 'Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models.'"; "2025 May 03: The FramePack-F1 is released." ([lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)). Weights: [lllyasviel/FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY) ("FramePack for HY", 13B parameters) and [lllyasviel/FramePack_F1_I2V_HY_20250503](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503) ("forward-only FramePack-F1 ... trained with anti-drifting regulations").
- UNVERIFIED: the original FramePack release date (the README extraction gave only the F1 date; the first end-frame PR is dated 2025-04-20, so the release predates that).

### First/last-frame support
- Official repo: no end-frame feature in the README ([lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)).
- Community hack: PR #167 "start and end frame control", submitted 2025-04-20: "by simply replace the last predicted image by inserted end image. now we can easily control the video to what we want."; still open/unmerged; a commenter reports "for now this only works for the first 1 second (all the transformation is done on 1st second and all the other seconds are just static)"; the owner indicated such features belong in a separate repository ([FramePack PR 167](https://github.com/lllyasviel/FramePack/pull/167)).
- ComfyUI: Kijai's wrapper `FramePackSampler` has optional `end_latent` ("end Latents to use for image2video"), `end_image_embeds`, `start_embed_strength` and `embed_interpolation` inputs; the same file showed no F1-specific sampler option in the extraction ([ComfyUI-FramePackWrapper nodes.py](https://raw.githubusercontent.com/kijai/ComfyUI-FramePackWrapper/main/nodes.py), [ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper)).
- FramePack-F1 is described as forward-only (predicts future frames from history) ([F1 model card](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503)). UNVERIFIED: any end-frame support for F1.
- Secondary (search snippet, page not opened): "when the start and end frames differ too greatly, the model struggles and often produces 'slideshow-style' cuts" for a community start/end node ([TTP_Comfyui_FramePack_SE listing](https://comfy.icu/extension/TTPlanetPig__TTP_Comfyui_FramePack_SE)).

### Licence
- Code: Apache-2.0 ([lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)); wrapper code: Apache-2.0 ([ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper)).
- Weights: neither weight model card shows a licence tag or licence text ([FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY), [F1](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503)).
- Licence trap: the weights are a HunyuanVideo ("HY") derivative and the workflow uses HunyuanVideo text encoders and VAE from `Comfy-Org/HunyuanVideo_repackaged` ([ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper)). The Tencent Hunyuan Community License (section 12) excludes the EU, UK and South Korea and covers "Model Derivatives". UNVERIFIED: which licence legally governs the FramePack weights; no page read states it.

### Sizes
- 13B parameters ([FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY)). Diffusers shards 9.97 + 9.99 + 5.79 = 25.75 GB ([FramePackI2V_HY file listing](https://huggingface.co/api/models/lllyasviel/FramePackI2V_HY/tree/main)). Single files: bf16 25.75 GB; fp8_e4m3fn 16.33 GB ([Kijai/HunyuanVideo_comfy listing](https://huggingface.co/api/models/Kijai/HunyuanVideo_comfy/tree/main)).
- UNVERIFIED: GGUF sizes; sizes of the HunyuanVideo text encoders (llava-llama3, clip-l), VAE and SigLIP vision files (listings not opened).

### VRAM / RAM
- Author claim: "To generate 1-minute video (60 seconds) at 30fps (1800 frames) using 13B model, the minimal required GPU memory is 6GB." ([lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)). This is the author's statement for the official desktop software, not an independent measurement.
- UNVERIFIED: the official system-RAM requirement (not returned by the README extraction).
- Derived: the fp8 transformer is 16.33 GB, i.e. larger than total system RAM of the target machine before the LLM text encoder is counted; with 8 GB VRAM roughly 9-10 GB of it must sit in RAM at fp8. 16 GB RAM is the bottleneck, not the 6 GB VRAM figure.

### Speed
- Author statement: "On my RTX 4090 desktop it generates at a speed of 2.5 seconds/frame (unoptimized) or 1.5 seconds/frame (teacache). On my laptops like 3070ti laptop or 3060 laptop, it is about 4x to 8x slower." ([lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)).
- Derived from that statement: 4x-8x of 1.5-2.5 s/frame = about 6-20 s per frame on a laptop 3060/3070 Ti class GPU; a 5 s clip at 30 fps (150 frames) would be about 15-50 minutes. Not a measurement on an RTX 4060.

### Resolution / fps / frames
- 30 fps; length is extended progressively, up to 1800 frames in the author's example ([lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)). UNVERIFIED: resolution buckets.

### Known quality issues
- End-frame hack concentrates the transformation in the first second for longer clips (PR comment quoted above) ([FramePack PR 167](https://github.com/lllyasviel/FramePack/pull/167)). Anecdotal.
- UNVERIFIED: line-art specific behaviour.

---

## 12. HunyuanVideo / HunyuanVideo-I2V / HunyuanVideo 1.5

### What it is, release date, links
- HunyuanVideo-I2V: inference code and weights released March 6, 2025; "resolution up to 720P and video length up to 129 frames (5 seconds)" ([HunyuanVideo-I2V GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)).
- HunyuanVideo 1.5: 8.3B parameters; inference code and weights Nov 20, 2025; step-distilled model Dec 5, 2025 ([HunyuanVideo-1.5 GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [model card](https://huggingface.co/tencent/HunyuanVideo-1.5)).

### First/last-frame support
- None found in official sources: the 1.5 repo lists T2V and I2V only ([HunyuanVideo-1.5 GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)); HunyuanVideo-I2V docs mention no end frame ([HunyuanVideo-I2V GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)).
- ComfyUI source: `HunyuanVideo15ImageToVideo` and `HunyuanImageToVideo` accept only `start_image`; no end-image input ([ComfyUI nodes_hunyuan.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_hunyuan.py)).
- Community adapter for the original HunyuanVideo T2V model: "An adapter for HunyuanVideo T2V model for keyframe-based video generation" allowing "precise control over the generated video's start and end frames"; recommended 720x1280 / 544x960 and 33 to 97 frames, 50 steps (30 minimum); "The model works best on human subjects." ([dashtoon/hunyuan-video-keyframe-control-lora](https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora)). UNVERIFIED: its licence (not shown on the page), ComfyUI support, and any few-step variant.

### Licence
- "TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT" (HunyuanVideo: release date December 3, 2024; HunyuanVideo 1.5: November 21, 2025): "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"; Territory = "the worldwide territory, excluding the territory of the European Union, United Kingdom and South Korea"; licence is "non-exclusive, non-transferable and royalty-free" within the Territory; a separate licence must be requested above 100 million monthly active users; "Tencent claims no rights in Outputs You generate. You and Your users are solely responsible for Outputs and their subsequent uses."; "You must not use the Tencent Hunyuan Works or any Output or results of the Tencent Hunyuan Works to improve any other AI model (other than Tencent Hunyuan or Model Derivatives thereof)." ([HunyuanVideo LICENSE](https://huggingface.co/tencent/HunyuanVideo/raw/main/LICENSE), [HunyuanVideo-1.5 LICENSE](https://huggingface.co/tencent/HunyuanVideo-1.5/raw/main/LICENSE)).
- Licence trap: the territory exclusion. If the owner or the trailer's distribution is in the EU, UK or South Korea, the licence text says it does not apply there. This also bears on HunyuanVideo derivatives such as FramePack (section 11).
- UNVERIFIED: the licence file of the HunyuanVideo-1.5 GitHub code repo (not opened; one extraction guessed "Apache 2.0" from the presence of a LICENSE file, which contradicts the HF licence tag "tencent-hunyuan-community" for the weights).

### Sizes
- HunyuanVideo 1.5 ComfyUI repack: each fp16 model 16.65 GB; fp8_scaled 8.33 GB (480p/720p I2V cfg-distilled); 480p I2V step-distilled fp8_scaled 8.34 GB ([Comfy-Org/HunyuanVideo_1.5_repackaged listing](https://huggingface.co/api/models/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/diffusion_models)).
- Text encoders in the ComfyUI workflow: `qwen_2.5_vl_7b_fp8_scaled.safetensors` and `byt5_small_glyphxl_fp16.safetensors`; VAE `hunyuanvideo15_vae_fp16.safetensors` ([ComfyUI docs: HunyuanVideo 1.5](https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5)). UNVERIFIED: their file sizes.

### VRAM / RAM
- Official: minimum "14 GB (with model offloading enabled)" for 1.5 ([model card](https://huggingface.co/tencent/HunyuanVideo-1.5)); the repo also links WanGP as supporting "as low 6 GB of VRAM" ([HunyuanVideo-1.5 GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)); ComfyUI docs describe 24 GB consumer GPUs as the target ([ComfyUI docs: HunyuanVideo 1.5](https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5)). HunyuanVideo-I2V: "The minimum GPU memory required is 60GB for 720p" (official script) ([HunyuanVideo-I2V GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)).
- Not assessed further for 8 GB / 16 GB because no first-last-frame path was found for 1.5.

### Speed
- Official (RTX 4090, 480p I2V step-distilled, 8 or 12 steps): "videos within 75 seconds" ([model card](https://huggingface.co/tencent/HunyuanVideo-1.5)). UNVERIFIED: 8 GB-class timings.

### Resolution / fps / frames
- 1.5: 480p and 720p (1080p via super-resolution), up to 121 frames, 24 fps ([model card](https://huggingface.co/tencent/HunyuanVideo-1.5)).

### Known quality issues
- Keyframe LoRA: "works best on human subjects" ([dashtoon card](https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora)). UNVERIFIED: anything line-art specific.

---

## 13. CogVideoX (5B I2V, 1.5, Fun InP)

### What it is, release date, links
- Official models and dates: CogVideoX-2B 2024-08-06; 5B 2024-08-27; 5B-I2V 2024-09-19; CogVideoX1.5-5B and 1.5-5B-I2V 2024-11-08 ([zai-org/CogVideo GitHub](https://github.com/zai-org/CogVideo)).
- Fun variant: [alibaba-pai/CogVideoX-Fun-V1.1-5b-InP](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP), V1.1 dated 2024.09.29 on the card.

### First/last-frame support
- Official CogVideoX: no first-last-frame support mentioned in the README ([CogVideo GitHub](https://github.com/zai-org/CogVideo)).
- UNVERIFIED: start+end image support of CogVideoX-Fun InP. The VideoX-Fun README extraction confirmed start-end prediction for the Wan Fun models and listed CogVideoX-Fun as supported, but did not return an explicit start/end statement for CogVideoX-Fun ([VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)).

### Licence
- CogVideoX-2B: Apache 2.0. CogVideoX-5B family (T2V/I2V, 1.5): "CogVideoX LICENSE" ([CogVideo GitHub](https://github.com/zai-org/CogVideo)).
- "The CogVideoX License": commercial users must "register and obtain a basic commercial license" at https://open.bigmodel.cn/mla/form; then free commercial use provided "the number of service users (visits) for your commercial activities must not exceed 1 million visits per month"; governed by the laws of the People's Republic of China, disputes at Haidian District People's Court in Beijing ([CogVideoX-5b-I2V LICENSE](https://huggingface.co/zai-org/CogVideoX-5b-I2V/raw/main/LICENSE)).
- Licence trap: the CogVideoX-Fun card says "Apache License (Version 2.0)" for the project, but VideoX-Fun states the CogVideoX-5B model component remains under the CogVideoX LICENSE ([VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)). The registration requirement applies to 5B-based weights.

### Sizes
- CogVideoX-Fun V1.1: 2B 13.0 GB unpacked; 5B 20.0 GB unpacked (whole repo) ([Fun card](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP)). UNVERIFIED: individual transformer / T5 / GGUF file sizes.

### VRAM / RAM
- Official diffusers figures with optimisations: CogVideoX-5B-I2V "4GB minimum"; CogVideoX1.5-5B-I2V "from 4GB" (sequential CPU offload class settings; slow) ([CogVideo GitHub](https://github.com/zai-org/CogVideo)). UNVERIFIED: system RAM needed in that mode and ComfyUI behaviour.

### Speed
- UNVERIFIED: no 8 GB-class timing read.

### Resolution / fps / frames
- CogVideoX-5B-I2V: frames 8N+1 with N <= 6 (default 49), 8 fps. CogVideoX1.5-5B-I2V: frames 16N+1 with N <= 10 (default 81), 16 fps, Min(W,H)=768, Max up to 1360 ([CogVideo GitHub](https://github.com/zai-org/CogVideo)). CogVideoX-Fun V1.1: 1-49 frames at 8 fps, 512 to 1280 resolutions ([Fun card](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP)).

### Known quality issues
- UNVERIFIED (none read).

---

## 14. SkyReels V2 / V3

- SkyReels V2: released Apr 21, 2025; variants Diffusion Forcing (DF) 1.3B-540P / 14B-540P / 14B-720P, T2V 14B, I2V 1.3B-540P / 14B ([SkyReels-V2 GitHub](https://github.com/SkyworkAI/SkyReels-V2)).
- FLF: "When controlling the start and end frames, you need to pass the `--image ${image}` parameter to control the generation of the start frame and the `--end_image ${end_image}` parameter" ([SkyReels-V2 GitHub](https://github.com/SkyworkAI/SkyReels-V2)). UNVERIFIED: which checkpoints this applies to (the extraction did not say; the DF models are the likely context). Kijai's WanVideoWrapper lists SkyReels as supported ([ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)); UNVERIFIED whether end-frame control is exposed there.
- Licence: HF tag "skywork-license" ([SkyReels-V2-DF-1.3B-540P card](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P)); repo terms: "The Skywork model supports commercial use. If you plan to use the Skywork model or its derivatives for commercial purposes, you must abide by terms and conditions within [Skywork Community License]"; not to be deployed "for internet services without appropriate security reviews" ([SkyReels-V2 LICENSE.txt](https://github.com/SkyworkAI/SkyReels-V2/blob/main/LICENSE.txt)). UNVERIFIED: the full Skywork Community License PDF (not opened).
- VRAM (authors' measurement): "Generating a 540P video using the 1.3B model requires approximately 14.7GB peak VRAM"; 14B about 43.4-51.2 GB ([SkyReels-V2 GitHub](https://github.com/SkyworkAI/SkyReels-V2)). UNVERIFIED: file sizes, GGUF availability, 8 GB results.
- Resolution/frames: 540P = 544x960, 97 frames; 720P = 720x1280, 121 frames; 24 fps ([SkyReels-V2 GitHub](https://github.com/SkyworkAI/SkyReels-V2)).
- SkyReels V3: released Jan 29, 2026; models: Reference-to-Video 14B-720P, Video Extension 14B-720P, Talking Avatar 19B-720P; no first-last-frame task listed; `--low_vram` for GPUs under 24 GB ([SkyReels-V3 GitHub](https://github.com/SkyworkAI/SkyReels-V3)). UNVERIFIED: V3 licence terms.

---

## 15. Other models checked (no first-last-frame support found)

- MAGI-1 (Sand AI): released Apr 21, 2025; 24B and 4.5B, plus "MAGI-1.1-24B" on June 17, 2026; tasks "t2v", "i2v", "v2v" only; 4.5B quantised works on "GPUs with at least 12GB of memory"; Apache 2.0 ([MAGI-1 GitHub](https://github.com/SandAI-org/MAGI-1)). No FLF/keyframe mode documented; below the 12 GB minimum.
- Mochi 1 (Genmo): 10B parameters, text-to-video only, 480p, Apache 2.0, about 60 GB VRAM single GPU or under 20 GB in ComfyUI ([genmoai/mochi](https://github.com/genmoai/mochi)). No image or end-frame conditioning documented.
- Helios (PKU-YuanGroup): 14B, released March 4, 2026, T2V/I2V/V2V, Apache 2.0, about 6 GB VRAM with group offloading, no ComfyUI support mentioned, no end-frame mode mentioned ([Helios GitHub](https://github.com/PKU-YuanGroup/Helios)).
- LongCat-Video (Meituan): 13.6B, released October 25, 2025, T2V / I2V / video continuation, MIT licence, no end-frame mode mentioned ([LongCat-Video GitHub](https://github.com/meituan-longcat/LongCat-Video)).
- UNVERIFIED: whether other 2026 open-weight releases (not found by the searches run) add first-last-frame conditioning.

---

## 16. Two-keyframe interpolation tools

### ToonCrafter (generative cartoon interpolation)
- What: "a video diffusion model that takes in two still images as conditioning images and text prompt describing dynamics, and generates interpolation videos from them" ([Doubiiu/ToonCrafter model card](https://huggingface.co/Doubiiu/ToonCrafter)). Code and weights released May 29, 2024 ([ToonCrafter GitHub](https://github.com/Doubiiu/ToonCrafter)).
- FLF: native (two keyframes are the input). ComfyUI community nodes: ComfyUI-DynamiCrafterWrapper, ComfyUI-ToonCrafter ([ToonCrafter GitHub](https://github.com/Doubiiu/ToonCrafter)).
- Licence: Apache-2.0 tag; "Feel free to use it under the Apache-2.0 license. Note that we don't have any official commercial product for ToonCrafter currently." ([model card](https://huggingface.co/Doubiiu/ToonCrafter)); repo disclaimer: "This is an open-source research exploration, instead of commercial products." ([ToonCrafter GitHub](https://github.com/Doubiiu/ToonCrafter)). UNVERIFIED: licences of upstream components it builds on (not checked).
- Limits: 512x320, up to 16 frames, about 2 seconds at 8 fps ([model card](https://huggingface.co/Doubiiu/ToonCrafter)). Far below the project's 1280x704 / 24 fps / 121 frames.
- VRAM: about 24 GB on A100 at 50 DDIM steps; the README relays community reports of 10-12 GB and an fp16 pruned wrapper model at 12 GB ([ToonCrafter GitHub](https://github.com/Doubiiu/ToonCrafter)). UNVERIFIED: 8 GB operation and file sizes.
- Quality notes from the authors: "The autoencoding part of the model is lossy, resulting in slight flickering artifacts"; "the success rate is not guaranteed" ([model card](https://huggingface.co/Doubiiu/ToonCrafter), [ToonCrafter GitHub](https://github.com/Doubiiu/ToonCrafter)).

### Other
- "Generative Inbetweening: Adapting Image-to-Video Diffusion Models for Keyframe Interpolation" (arXiv 2408.15239) appeared in search results ([arXiv 2408.15239](https://arxiv.org/pdf/2408.15239)); page not opened. UNVERIFIED: weights, licence, requirements.
- RIFE and FILM are non-generative frame interpolators (they blend/flow between two nearby frames, they do not synthesise new motion). UNVERIFIED: no page was opened for them in this research.

---

## Cross-cutting facts on quantisation, few-step LoRAs and block swap

- Kijai wrapper: "GGUF models can now be loaded in the main model loader as well."; block swap offloads transformer blocks to system RAM; example from the README: "you use 1GB LoRA unmerged and swap 20 blocks on 14B model ... single block grows by 25MB, 20 blocks grow by 500MB" ([ComfyUI-WanVideoWrapper readme](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/readme.md)). Block swap moves the load to system RAM, which is the scarce resource on a 16 GB machine.
- Wan 2.2 Lightning (lightx2v) LoRAs: 4 steps, I2V-A14B version released Aug 7, 2025, Apache 2.0 ([lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)); ComfyUI docs' caution "the video dynamic is lost" ([ComfyUI docs: Wan2.2 Fun InP](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp)).
- Anecdotal 8 GB preset for Wan 2.2 TI2V-5B GGUF from a forum thread (RTX 5060 Ti 8 GB, Nov 2025): 672x384, 33 frames at 24 fps, 12-16 steps; "memory scales with width x height x frames x batch" ([HF forum thread](https://discuss.huggingface.co/t/a-model-for-converting-photos-to-videos-with-8gb-of-vram/170034)).
- UNVERIFIED: any source measuring loss of line quality on 2D/comic art from few-step LoRAs or low-bit GGUF in first-last-frame use. No such primary or secondary source was found.

---

## Comparison table

Sizes are diffusion-model files only unless noted. "Fits?" entries marked D are Derived from file sizes, not measured.

| Model | FLF support | Licence / commercial | Smallest practical files | Fits 8 GB VRAM / 16 GB RAM? | Reported speed | Main issues (sourced) |
|---|---|---|---|---|---|---|
| Wan 2.1 FLF2V 14B | Native; ComfyUI `WanFirstLastFrameToVideo` | Apache 2.0 | GGUF Q3_K_S 7.93 GB, Q4_K_S 10.44 GB; fp8 16.40 GB | D: at the limit; needs offload; 720p only by design | None for 8 GB (UNVERIFIED) | Chinese prompts recommended; small sizes "may not produce good output"; last-frame scaling issue report |
| Wan 2.1 Fun InP 1.3B | Native InP; `WanFunInpaintToVideo` | Apache 2.0 | bf16 3.13 GB | D: yes (model fits VRAM; encoder 3.7-6.7 GB) | UNVERIFIED | User reports: fade to black / jump in last frames |
| Wan 2.1 Fun InP 14B | Native InP | Apache 2.0 | "32GB+" full; GGUF UNVERIFIED | D: limit, as FLF2V 14B | UNVERIFIED | As above |
| Wan 2.1 VACE 1.3B / 14B | Via VACE masks; `WanVaceToVideo` | Apache 2.0 | 1.3B 4.31 GB; 14B GGUF Q3_K_S 7.84 GB | D: 1.3B yes; 14B at the limit | 4090: 640x640x49 about 7 min (14B) | 1.3B 720p "less stable"; more drastic changes (anecdotal) |
| Wan 2.2 I2V-A14B (2 experts) | ComfyUI `WanFirstLastFrameToVideo` on I2V weights (not an official FLF checkpoint) | Apache 2.0 | GGUF Q3_K_S 6.52 GB x2, Q4_K_S 8.75 GB x2; LoRAs 1.23 GB x2 | D: VRAM ok with offload; 16 GB RAM is the bottleneck (two experts) | Secondary: 4060 8 GB + 32 GB RAM, 480x480x33, 4 steps, 111 s (plain I2V, merged model) | 4-step LoRA: "video dynamic is lost"; ping-pong beyond about 100 frames; contrast drift reports |
| Wan 2.2 Fun InP A14B | Native InP; `WanFunInpaintToVideo` | Apache 2.0 | GGUF Q3_K_S 6.52 GB x2, Q4_K_S 8.76 GB x2 | D: as I2V-A14B | 4090D: 640x640x81 about 79 s with 4-step LoRA | Same LoRA caveat |
| Wan 2.2 Fun InP 5B | Native InP; official ComfyUI template | Apache 2.0 | GGUF Q8_0 5.41 GB, Q4_K_M 3.44 GB; VAE 1.41 GB | D: same class as the TI2V-5B Q8 already in use | UNVERIFIED | UNVERIFIED (Turbo compatibility unknown) |
| Wan 2.2 TI2V-5B | No end-frame input in ComfyUI node | Apache 2.0 | GGUF Q8_0 5.40 GB | Already runs (project context) | n/a | No FLF; use Fun 5B InP for end frame |
| Wan 2.2 VACE-Fun A14B | FLF + reference + control | Apache 2.0 | fp8 17.35 GB x2; GGUF UNVERIFIED | D: harder than I2V-A14B | UNVERIFIED | UNVERIFIED |
| Wan 2.5 / 2.6 / 2.7 / 3 | n/a | No open weights in Wan-AI HF org | n/a | n/a | n/a | API-only (Secondary) |
| LTX-Video 0.9.8 2B distilled | Native multi-keyframe; `LTXVAddGuide` frame_idx | LTXV Open Weights License; $10M revenue threshold | fp8 4.46 GB; full 6.34 GB | D: model fits VRAM; text encoder size UNVERIFIED | UNVERIFIED | UNVERIFIED |
| LTX-Video 0.9.8 13B | Native multi-keyframe | Same | fp8 15.69 GB | D: no margin on 16 GB RAM | UNVERIFIED | UNVERIFIED |
| LTX-2 / 2.3 / 2.5 | Native keyframe pipeline (dev model); ComfyUI guides | LTX-2 / LTX-2.x Community License; $10M threshold incl. affiliates; competing-product clause | 2.5 GGUF Q2_K 7.33-8.83 GB + Gemma 4 GGUF 5.96 GB+ | D: at or beyond the limit; reports stress 32 GB+ RAM | Anecdotal: 3050 6 GB + 44 GB RAM, 10 s 720p under 15 min (LTX-2) | UNVERIFIED for keyframes |
| FramePack (HY) | Community: wrapper `end_latent`; PR 167 unmerged | Code Apache 2.0; weights untagged; HunyuanVideo derivative (territory exclusion) | fp8 16.33 GB | D: 6 GB VRAM claimed by author; RAM is the bottleneck | Author: 4090 1.5-2.5 s/frame; laptops 4x-8x slower | End frame effect concentrated in first second; slideshow cuts (anecdotal) |
| HunyuanVideo 1.5 | None found (start image only) | Tencent Hunyuan Community; excludes EU, UK, South Korea | fp8 8.33 GB | Not assessed (no FLF) | 4090: 75 s (480p step-distilled) | No FLF |
| HunyuanVideo + keyframe LoRA | Community LoRA (start + end) | Base: Tencent licence; LoRA licence UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | "works best on human subjects"; 30-50 steps |
| CogVideoX 5B I2V / Fun InP | Official: none; Fun InP: UNVERIFIED | CogVideoX LICENSE: registration + 1M visits/month cap | Fun 5B repo 20 GB | Official: from 4 GB VRAM (diffusers offload); RAM UNVERIFIED | UNVERIFIED | 49 frames at 8 fps (1.0) |
| SkyReels V2 | `--image` + `--end_image` | Skywork Community License (commercial allowed under its terms; PDF not read) | UNVERIFIED | 1.3B: 14.7 GB peak VRAM (authors) | UNVERIFIED | UNVERIFIED |
| ToonCrafter | Native two-keyframe | Apache 2.0 | UNVERIFIED | 10-12 GB per community reports | About 24 s per clip on A100 (authors) | 512x320, 16 frames; flicker from lossy autoencoder |

---

## Open questions / could not verify

1. No source was found that measured any first-last-frame workflow on an 8 GB card with only 16 GB system RAM. All such "fits?" statements above are Derived.
2. Whether the TI2V-5B Turbo few-step setup used in this project works with Wan 2.2 Fun 5B InP, and which node the official 5B Fun Inpaint template uses for the end image.
3. Whether Kijai's WanVideoWrapper exposes an end image for the plain TI2V-5B checkpoint.
4. Low-noise GGUF file sizes for Wan 2.2 I2V-A14B / Fun InP A14B; GGUF sizes for Wan 2.1 Fun 14B InP and Wan 2.2 VACE-Fun.
5. Release dates of the Wan 2.1 / 2.2 Fun and VACE-Fun models.
6. Which licence governs the FramePack weights (no tag on the model cards; HunyuanVideo derivative).
7. The full text of the LTX licences' Attachment A (read only through extraction); ltx.io licence page, the HF LICENSE of LTX-2.5 and docs.ltx.video could not be opened.
8. LTX-Video 0.9.8 text-encoder size, measured VRAM/RAM, and 8 GB timings.
9. LTX-2.5 local first+last-frame workflow with GGUF, and the LTX-2.5 release date (sources conflict: 2026-07-23 repo creation vs 2026-08-11 article/licence date).
10. CogVideoX-Fun InP end-frame support; Skywork Community License full terms; SkyReels V2 checkpoint(s) that accept `--end_image`.
11. Any documented evidence about 2D / comic line-art quality under FLF, few-step LoRAs or low-bit quantisation. None was found; the GitHub issues cited concern colour/brightness and end-of-clip artefacts in general.
12. Pages that failed to open: `docs.comfy.org/tutorials/video/wan/wan2-2` (404; the working URL is `wan2_2`), `stable-diffusion-art.com/wan-2-2-first-last-frame-video/` (403), `raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/README.md` (404), plus the four LTX pages in item 7.

---

## Sources

All read on 2026-10-05.

Wan
- https://github.com/Wan-Video/Wan2.1 - Wan2.1 GitHub README
- https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P - Wan2.1-FLF2V-14B-720P model card
- https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B - Wan2.1-VACE-1.3B model card
- https://github.com/Wan-Video/Wan2.2 - Wan2.2 GitHub README
- https://huggingface.co/api/models?author=Wan-AI&sort=createdAt&direction=-1&limit=40 - Hugging Face API listing of Wan-AI models
- https://github.com/Wan-Video/Wan2.1/issues?q=is%3Aissue+FLF2V - Wan2.1 issue search "FLF2V"
- https://github.com/Wan-Video/Wan2.1/issues/342 - Issue: For FLF2V model, the last frame generation has abnormal scaling
- https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP - Wan2.1-Fun-1.3B-InP model card
- https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP - Wan2.2-Fun-5B-InP model card
- https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP - Wan2.2-Fun-A14B-InP model card
- https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B - Wan2.2-VACE-Fun-A14B model card
- https://github.com/aigc-apps/VideoX-Fun - VideoX-Fun GitHub README
- https://huggingface.co/lightx2v/Wan2.2-Lightning - Wan2.2-Lightning model card
- https://howaiworks.ai/blog/alibaba-wan-open-weights-stopped-at-2-2 - "The Newest Wan You Can Download Is Still Wan 2.2" (search snippet only)
- https://www.wan27.org/blog/wan-2-6-open-source-guide - "Is Wan 2.6 Open Source?" (search snippet only)
- https://wan27.org/blog/wan-3-open-source - "Is Wan 3.0 Open Source?" (search result title only)

ComfyUI
- https://docs.comfy.org/tutorials/video/wan/wan-flf - ComfyUI docs: Wan2.1 FLF2V
- https://docs.comfy.org/tutorials/video/wan/wan2_2 - ComfyUI docs: Wan2.2
- https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp - ComfyUI docs: Wan2.2 Fun InP
- https://docs.comfy.org/tutorials/video/wan/fun-inp - ComfyUI docs: Wan2.1 Fun InP
- https://docs.comfy.org/tutorials/video/wan/vace - ComfyUI docs: Wan VACE
- https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5 - ComfyUI docs: HunyuanVideo 1.5
- https://docs.comfy.org/tutorials/video/ltx/ltx-2 - ComfyUI docs: LTX-2
- https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_wan.py - ComfyUI source, Wan nodes
- https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_lt.py - ComfyUI source, LTX nodes
- https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_hunyuan.py - ComfyUI source, Hunyuan nodes
- https://comfy.org/workflows/video_wan2_2_5B_fun_inpaint-00d3d57a5195/ - ComfyUI template: Wan 2.2 5B Fun Inpaint
- https://comfy.org/workflows/video_wan2_2_14B_flf2v-7016f027bcf1/ - ComfyUI template: Wan 2.2 14B First-Last Frame to Video (search result only)
- https://github.com/Comfy-Org/ComfyUI/issues/11775 - Issue: cumulative system RAM leak with Wan2.2 GGUF
- https://github.com/comfyanonymous/ComfyUI/issues?q=is%3Aissue+WanFirstLastFrameToVideo - ComfyUI issue search
- https://github.com/kijai/ComfyUI-WanVideoWrapper - ComfyUI-WanVideoWrapper GitHub
- https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/readme.md - ComfyUI-WanVideoWrapper readme
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues?q=is%3Aissue+last+frame+color - WanVideoWrapper issue search
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/339 - Issue: Wan2.1-Fun fade-out / overexposure
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/872 - Issue: start/end frame with Fun Control, washed out end frames
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1342 - Issue: First and Last Frame Pingponging Wan 2.2

Hugging Face file listings (API)
- https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models
- https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders
- https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/clip_vision
- https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models
- https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae
- https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras
- https://huggingface.co/api/models/city96/Wan2.1-FLF2V-14B-720P-gguf/tree/main
- https://huggingface.co/api/models/city96/umt5-xxl-encoder-gguf/tree/main
- https://huggingface.co/api/models/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main/HighNoise
- https://huggingface.co/api/models/QuantStack/Wan2.2-Fun-A14B-InP-GGUF/tree/main/HighNoise
- https://huggingface.co/api/models/QuantStack/Wan2.2-Fun-5B-InP-GGUF/tree/main
- https://huggingface.co/api/models/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main
- https://huggingface.co/api/models/QuantStack/Wan2.1_14B_VACE-GGUF/tree/main
- https://huggingface.co/api/models/Lightricks/LTX-Video/tree/main
- https://huggingface.co/api/models/Lightricks/LTX-2/tree/main
- https://huggingface.co/api/models/Lightricks/LTX-2.3/tree/main
- https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/main
- https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/main/diffusion_models
- https://huggingface.co/api/models/Lightricks/LTX-2.5/tree/main/text_encoders
- https://huggingface.co/api/models/realrebelai/LTX-2.5_GGUFs/tree/main
- https://huggingface.co/api/models/agosh/LTX-2.5-Comfy-GGUF/tree/main
- https://huggingface.co/api/models/Comfy-Org/ltx-2/tree/main/split_files/text_encoders
- https://huggingface.co/api/models/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/diffusion_models
- https://huggingface.co/api/models/lllyasviel/FramePackI2V_HY/tree/main
- https://huggingface.co/api/models/Kijai/HunyuanVideo_comfy/tree/main
- https://huggingface.co/api/models?author=Lightricks&sort=createdAt&direction=-1&limit=40 - Hugging Face API listing of Lightricks models

LTX
- https://github.com/Lightricks/LTX-Video - LTX-Video GitHub README
- https://huggingface.co/Lightricks/LTX-Video - LTX-Video model card
- https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled - LTX-Video 0.9.8 13B distilled model card
- https://huggingface.co/Lightricks/LTX-Video/raw/main/LTX-Video-Open-Weights-License-0.X.txt - LTXV Open Weights License 0.X
- https://github.com/Lightricks/LTX-2 - LTX-2 GitHub README
- https://raw.githubusercontent.com/Lightricks/LTX-2/main/README.md - LTX-2 README (raw)
- https://github.com/Lightricks/LTX-2/blob/main/LICENSE - LTX-2 licence index page (lists the two licence agreements)
- https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2_x - LTX-2.x Community License Agreement
- https://scancode-licensedb.aboutcode.org/ltx-2-cla-2026.html - ScanCode LicenseDB copy of the LTX-2 Community License Agreement
- https://huggingface.co/Lightricks/LTX-2.5 - LTX-2.5 model card
- https://docs.ltx.io/models/ltx-2-5 - LTX docs: LTX-2.5 (hosted API)
- https://github.com/Lightricks/ComfyUI-LTXVideo - ComfyUI-LTXVideo GitHub README
- https://magica.com/news/ltx-2-5-open-weight-video-world-model - "LTX-2.5: what its open weights actually change" (Secondary)
- https://ltxworkflow.com/guide/ltx-2-5-vram-requirements - "LTX 2.5 VRAM Requirements" (Secondary, third-party)
- https://huggingface.co/Kijai/LTXV2_comfy/discussions/32 - HF discussion with user hardware reports (Anecdotal)
- https://ltx.io/model/license - LTX model licence page (could not be opened)
- https://huggingface.co/Lightricks/LTX-2.5/raw/main/LICENSE - LTX-2.5 LICENSE on HF (HTTP 401, not read)
- https://docs.ltx.video/open-source-model/integration-tools/comfy-ui - LTX docs: ComfyUI (HTTP 403, not read)
- https://ltx.io/blog/how-to-run-ltx-2-on-consumer-gpus - LTX blog (could not be opened)

FramePack / Hunyuan
- https://github.com/lllyasviel/FramePack - FramePack GitHub README
- https://github.com/lllyasviel/FramePack/pull/167 - PR: start and end frame control
- https://huggingface.co/lllyasviel/FramePackI2V_HY - FramePackI2V_HY model card
- https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503 - FramePack-F1 model card
- https://github.com/kijai/ComfyUI-FramePackWrapper - ComfyUI-FramePackWrapper GitHub
- https://raw.githubusercontent.com/kijai/ComfyUI-FramePackWrapper/main/nodes.py - ComfyUI-FramePackWrapper nodes.py
- https://comfy.icu/extension/TTPlanetPig__TTP_Comfyui_FramePack_SE - TTP_Comfyui_FramePack_SE listing (search snippet only)
- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 - HunyuanVideo-1.5 GitHub README
- https://huggingface.co/tencent/HunyuanVideo-1.5 - HunyuanVideo-1.5 model card
- https://huggingface.co/tencent/HunyuanVideo-1.5/raw/main/LICENSE - Tencent Hunyuan Community License (HunyuanVideo 1.5)
- https://huggingface.co/tencent/HunyuanVideo/raw/main/LICENSE - Tencent Hunyuan Community License (HunyuanVideo)
- https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V - HunyuanVideo-I2V GitHub README
- https://huggingface.co/dashtoon/hunyuan-video-keyframe-control-lora - HunyuanVideo keyframe control LoRA model card

Others
- https://github.com/zai-org/CogVideo - CogVideo GitHub README
- https://huggingface.co/zai-org/CogVideoX-5b-I2V/raw/main/LICENSE - The CogVideoX License
- https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP - CogVideoX-Fun V1.1 5b InP model card
- https://github.com/SkyworkAI/SkyReels-V2 - SkyReels-V2 GitHub README
- https://github.com/SkyworkAI/SkyReels-V2/blob/main/LICENSE.txt - SkyReels-V2 LICENSE.txt
- https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P - SkyReels-V2-DF-1.3B-540P model card
- https://github.com/SkyworkAI/SkyReels-V3 - SkyReels-V3 GitHub README
- https://github.com/SandAI-org/MAGI-1 - MAGI-1 GitHub README
- https://github.com/genmoai/mochi - Mochi 1 GitHub README
- https://github.com/PKU-YuanGroup/Helios - Helios GitHub README
- https://github.com/meituan-longcat/LongCat-Video - LongCat-Video GitHub README
- https://huggingface.co/Doubiiu/ToonCrafter - ToonCrafter model card
- https://github.com/Doubiiu/ToonCrafter - ToonCrafter GitHub README
- https://arxiv.org/pdf/2408.15239 - Generative Inbetweening paper (search result only)

Secondary hardware reports
- https://lilting.ch/en/articles/wan22-comfyui-rtx4060-i2v - "WAN 2.2 I2V on RTX 4060 8GB in ComfyUI: 111 seconds per clip with 14B Rapid distilled"
- https://github.com/Cordux/ComfyUI-Wan2.2-workflow - Community Wan 2.2 GGUF workflow README
- https://discuss.huggingface.co/t/a-model-for-converting-photos-to-videos-with-8gb-of-vram/170034 - HF forum thread on 8 GB VRAM
