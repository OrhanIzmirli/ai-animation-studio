# Backend: Wan 2.2 TI2V-5B Turbo (GGUF) through a local ComfyUI

Adapter: [`src/ai_studio/generation/backends/wan_comfyui.py`](../src/ai_studio/generation/backends/wan_comfyui.py),
class `WanTurboComfyUIBackend`. The same module also provides the base-model
backend `wan-ti2v-5b-comfyui`; see [The base-model backend](#the-base-model-backend-wan-ti2v-5b-comfyui). ComfyUI plumbing (config, HTTP client, server
lifecycle): [`comfyui.py`](../src/ai_studio/generation/backends/comfyui.py).
Contract: `VideoGenerationBackend`, `supports_start_frame = True`.

> **This is not the base Wan 2.2 TI2V-5B.** Turbo is quanhaol's distilled
> finetune of it (4 steps, CFG 1), here as hum-ma's Q8_0 GGUF. It has its own
> backend name and model identity so that its runs are never mixed with the
> base model's ([BACKEND_WAN_TI2V_5B.md](BACKEND_WAN_TI2V_5B.md)). **Whether
> its quality is close to the base model is an open question** that only
> benchmark runs and human ratings can answer.

> **Status:** works on the development laptop (RTX 4060 Laptop 8 GB, 16 GB
> RAM). One real run through the adapter exists, with one seed. Nobody has
> rated its output yet.

| | |
|---|---|
| `backend_name` | `wan-ti2v-5b-turbo-comfyui` |
| `backend_version` | `comfyui-0.1` |
| `model_name` | `Wan2.2-TI2V-5B-Turbo (quanhaol) GGUF Q8_0` |
| `model_version` | SHA-256 of the GGUF file |
| `workflow_version` | `i2v-turbo-tiled-v1` |

## Why it exists

It generates the full `comic_rooftop_01` shot (1280×704, 121 frames, 24 fps)
on an 8 GB laptop GPU in under five minutes, with no cloud GPU budget. The
settings come from a feasibility spike that compared four variants on this
machine: [spikes/LOCAL_COMFYUI_SPIKE.md](../spikes/LOCAL_COMFYUI_SPIKE.md).
This adapter is that spike's run (d).

## Setup on Windows

ComfyUI is a separate program with its own Python environment. It is
installed **outside** this repository; the adapter only talks HTTP to it and
adds no dependency to the project. About 26 GB of disk are needed.

1. **ComfyUI** (git clone, own venv, pinned release):

   ```
   mkdir C:\tools & cd C:\tools
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   git checkout v0.38.2
   python -m venv venv
   venv\Scripts\python -m pip install --timeout 60 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cu128
   venv\Scripts\python -m pip install --timeout 60 --retries 10 -r requirements.txt
   ```

   (`--timeout` / `--retries`: without them pip once hung for an hour on the
   torch download.)

2. **ComfyUI-GGUF** custom node:

   ```
   cd custom_nodes
   git clone https://github.com/city96/ComfyUI-GGUF.git
   cd ComfyUI-GGUF & git checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
   ..\..\venv\Scripts\python -m pip install -r requirements.txt
   ```

3. **Model files** (18.95 GB with the base GGUF, 13.55 GB without it), into
   `ComfyUI\models\`:

   | Folder | File | Source repo | Bytes |
   |---|---|---|---|
   | `diffusion_models` | `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` | `hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF` | 5,404,990,176 |
   | `text_encoders` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`, `split_files/text_encoders/` | 6,735,906,897 |
   | `vae` | `wan2.2_vae.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`, `split_files/vae/` | 1,409,400,960 |

4. **ffmpeg and ffprobe** on `PATH` (encoding, measurement, metrics).

5. **Tell the project where ComfyUI is:**

   ```
   copy configs\comfyui.example.yaml configs\comfyui.local.yaml
   ```

   and set `comfyui_dir` in `configs\comfyui.local.yaml`. That file is
   git-ignored; there are no machine paths in the source tree. Alternatively
   set `AI_STUDIO_COMFYUI_CONFIG` to a config file anywhere.

6. **Run:**

   ```
   python scripts\run_benchmark.py --benchmark comic_rooftop_01 --backend wan-ti2v-5b-turbo-comfyui --seed 42 --width 1280 --height 704
   ```

   Close other heavy programs first; see "Memory" below.

## Configuration file

`configs/comfyui.local.yaml` (see `configs/comfyui.example.yaml`):

| Key | Default | |
|---|---|---|
| `comfyui_dir` | required | the folder containing ComfyUI's `main.py` |
| `python` | `<comfyui_dir>/venv` python | the interpreter that runs ComfyUI |
| `port` | 8188 | ComfyUI is started on, or reused from, `127.0.0.1:<port>` |
| `startup_timeout_seconds` | 300 | |
| `generation_timeout_seconds` | 3600 | a longer prompt is interrupted and recorded as a failed run |
| `models.text_encoder` / `vae` | required | file names under `<comfyui_dir>/models/` |
| `models.unet` | needed by the Turbo backend | the Turbo GGUF |
| `models.unet_base` | needed by the base backend | the base-model GGUF |
| `sha256.unet` / `unet_base` / `text_encoder` / `vae` | optional | expected hashes; a file that does not match is refused |
| `extra_args` | `[]` | extra arguments for ComfyUI's `main.py` |

## What is pinned and recorded

Pinned by the example configuration (and verified before every run):

| File | SHA-256 |
|---|---|
| `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` | `b4d7d1888d5ec2c29046cc443239faad99614cd3d356c61d8192fcac5f4c98fc` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |
| `wan2.2_vae.safetensors` | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` |

Software used for the measured run: ComfyUI `v0.38.2`
(`daeb5e53681e2b10a3f0727d9ec5bc90784bee10`), ComfyUI-GGUF
`6ea2651e7df66d7585f6ffee804b20e92fb38b8a`, torch 2.11.0+cu128 in ComfyUI's
venv. The adapter does not enforce these; it **records** what it finds.

Every run record gets, in `environment`:

| Key | |
|---|---|
| `comfyui_commit`, `comfyui_gguf_commit` | git commits of the two checkouts |
| `comfyui_version` | as reported by the running server |
| `comfyui_torch_version`, `comfyui_torch_cuda_build` | torch in ComfyUI's venv, e.g. `2.11.0+cu128` / `cu128` |
| `model_sha256` | `{unet, text_encoder, vae}`, computed from the files before the run |
| `ram_total_gb`, `ram_available_before_gb`, `ram_available_min_gb` | see "Memory" |
| `comfyui_prompt_seconds`, `fetch_and_encode_seconds`, `model_hashing_seconds` | timing detail |
| `comfyui_server_reused` | whether an already running ComfyUI was used |

The record's generic `pytorch_version` / `cuda_version` also name ComfyUI's
torch (e.g. `2.11.0+cu128` / `12.8`), because that is the torch that ran the
model. The torch of the Python that only drives the run is recorded
separately as `orchestrator_pytorch_version`. The first adapter run
(`20261004T210526Z-92e20c65`) predates this: its `pytorch_version` is the
orchestrator's `2.13.0+cpu`; use its `comfyui_torch_version`.

`model_parameters` holds every effective setting: steps, shift, sampler,
scheduler, `cfg`, `spatial_tile`, `temporal_tile`, tile overlaps,
`latent_scale`, frame count, size, the three file names and the encoder
settings.

## Lifecycle

- `prepare()` checks the configuration, the install and the model files,
  hashes the three files, then **starts ComfyUI headless** on
  `127.0.0.1:<port>` (`main.py --listen 127.0.0.1 --port N --disable-auto-launch`),
  or **reuses** a ComfyUI that already answers on that port. It waits until
  the server answers and verifies through `/object_info` that every node
  class the workflow needs exists, that their inputs match, and that the
  server lists the configured model files.
- Anything missing (configuration, ComfyUI, its python, a model file, a node
  class, a mismatching file hash) raises `BackendNotInstalledError`: a setup
  problem, exit code 2, nothing recorded.
- `close()` stops the server **only if the adapter started it**. The backend
  is also a context manager, and the CLI calls `close()` after every run. A
  reused server is left running, and the run gets a warning, because the
  recorded commits and hashes then describe the configured install, which
  may not be the server that answered.
- The server's log is copied to `comfyui.log` in the run's output folder.

## Workflow

Built in ComfyUI API format and sent to `/prompt`:

```
UnetLoaderGGUF ─ ModelSamplingSD3(shift) ─┐
CLIPLoader(wan) ─ CLIPTextEncode ×2 ──────┼─ KSampler ─ [LatentMultiply] ─ VAEDecodeTiled ─ SaveImage
VAELoader ─ Wan22ImageToVideoLatent ──────┘
LoadImage (start frame) ─┘
```

| Setting | Value | Parameter |
|---|---|---|
| Steps | 4 | `num_inference_steps` |
| CFG | 1.0 | fixed |
| Sampler / scheduler | `uni_pc` / `simple` | `sampler`, `scheduler` |
| Shift (`ModelSamplingSD3`) | 8.0 | `shift` |
| Spatial tile / overlap | 512 / 64 | `spatial_tile` |
| Temporal tile / overlap | computed / 8 | **not a parameter** |
| Latent scale | 1.0 (no node) | `latent_scale` |

Other `backend_parameters` keys are rejected.

**The temporal tile is computed, never set.** With ComfyUI's default 64-frame
temporal tiles the video gets a brightness dip at every tile seam (spike run
(c): mean luminance 52.5 → 47.8 around frame 60). The adapter always uses one
temporal tile for the whole clip: the next multiple of 4 above the frame
count, plus 4 (121 frames → 128).

**`latent_scale`** is exposed because the Turbo model card suggests 0.8
against oversaturation. Values other than 1.0 insert a `LatentMultiply` node
between the sampler and the decode. The default stays 1.0; it must not be
changed without a benchmark comparison. A value other than 1.0 has not been
run on the real model yet.

**Start frame:** preprocessed by the same function as the diffusers adapter
(`prepare_start_frame`: scale to cover, centre crop; 1672×941 → 1280×720 →
1280×704), saved as a derived PNG in the run's output folder and uploaded to
ComfyUI. The original is only read.

**Prompt:** passed unchanged. Two warnings are always attached when they
apply: the negative prompt has **no effect** at CFG 1, and the shot controls
(`camera_motion`, `subject_motion`, `visual_direction`) are not consumed
separately.

**Frames:** fetched from ComfyUI as PNG over HTTP and encoded with the same
encoder as the diffusers adapter (libx264, yuv420p, CRF 12, preset slow,
24 fps). The copies ComfyUI keeps in its own `input` / `output` folders are
deleted after the run.

Request rules are the same as for the diffusers adapter: a start frame is
required, fps 24 or unset, 1280×704 or 704×1280, at most 121 frames; and the
seed must be ≥ 0.

## Memory

On a 16 GB machine **system RAM, not VRAM, is the tight resource**. While the
text encoder and the model load, available RAM fell to 0.0–0.5 GB in every
measured run.

- Before loading, the adapter reads the available RAM. Below **8 GB** it adds
  a warning to the run (it does not refuse). The threshold comes from the
  spike: a run that started with 6.6 GB available paged about 2 GB while
  loading; runs that started with 8.8 GB or more did not grow the pagefile.
- During the run it samples available RAM once per second and records the
  lowest value (`ram_available_min_gb`).

Close browsers and other large programs before a run.

## Measured

Spike, same machine, seed 42, 1280×704×121 ([full report](../spikes/LOCAL_COMFYUI_SPIKE.md)):

| | Turbo, untiled decode | Turbo, tiled, 64-frame temporal tiles | **Turbo, tiled, one temporal tile (this adapter)** | base Q8_0, 20 steps, untiled |
|---|---|---|---|---|
| Total | 16.6 min | 3.8 min | **3.6 min** | 34.0 min |
| Sampling | 63 s | 63 s | 63 s (15.7 s/step) | 787 s |
| VAE decode | 867 s | 117 s | 109 s | 1181 s |
| Peak VRAM | 7.9 GB | 7.2 GB | 7.2 GB | 7.9 GB |
| Result | fine | brightness dips at tile seams | matches the untiled result | two colour glitches |

First run through the adapter and the CLI
(`comic_rooftop_01 / 20261004T210526Z-92e20c65`, seed 42):

| | |
|---|---|
| Total | 277.9 s (4.6 min) |
| Prepare | 43.3 s (hashing the model files 12.3 s, starting ComfyUI ~31 s) |
| Generation | 233.3 s: ComfyUI prompt 225.8 s (load, 4 steps, decode, saving PNGs), fetching and encoding 7.2 s |
| Measured output | 1280×704, 24 fps, 121 frames, 5.042 s, H.264 |
| RAM | 7.8 GB available before (warning issued), lowest 0.21 GB |
| Metrics v1 | motion_mean 1.684, total_change 27.64, start_frame_psnr 41.29 dB, pulse ratio 1.20, luminance_flicker 0.470, held_frame_ratio 0.000 |
| vs. spike run (d) | **byte-identical video** (same SHA-256 `5237eff9…29d1f6b3`): all 121 frames equal |

The identical file shows that this ComfyUI setup is reproducible for a fixed
seed on this machine, and that the adapter reproduces the spike exactly.

## The base-model backend: `wan-ti2v-5b-comfyui`

`WanBaseComfyUIBackend` runs the **base** Wan 2.2 TI2V-5B as Q8_0 GGUF
(`Wan2.2-TI2V-5B-Q8_0.gguf`, `QuantStack/Wan2.2-TI2V-5B-GGUF`, SHA-256
`57bece983817ab2f957546683bb670f13be7d99022d45674840cd999a050ea8f`) with the
same code: same workflow, lifecycle, start-frame preprocessing, tiled decode
with the computed temporal tile, encoder and recorded provenance.

| | Turbo backend | Base backend |
|---|---|---|
| `backend_name` | `wan-ti2v-5b-turbo-comfyui` | `wan-ti2v-5b-comfyui` |
| `model_name` | `Wan2.2-TI2V-5B-Turbo (quanhaol) GGUF Q8_0` | `Wan2.2-TI2V-5B GGUF Q8_0` |
| `model_version` | SHA-256 of the Turbo GGUF | SHA-256 of the base GGUF |
| `workflow_version` | `i2v-turbo-tiled-v1` | `i2v-tiled-v1` |
| Config key of its GGUF | `models.unet` | `models.unet_base` |
| Steps | 4 | 20 |
| CFG | 1, fixed | 5, parameter `guidance_scale` (≥ 1) |
| Sampler / scheduler / shift | `uni_pc` / `simple` / 8 | `uni_pc` / `simple` / 8 |
| Negative prompt | no effect (warning) | used |

The base defaults are those of ComfyUI's own template for this model. It is
also distinct from the diffusers backend `wan-ti2v-5b`, which runs the
unquantised weights. Measured cost on this laptop: see
[experiments/E1_REPORT.md](experiments/E1_REPORT.md).

## Known limits

- **One seed, one real run, no human rating.** Nothing here says the output
  is good, only that it is produced reliably and what it measures.
- **Turbo vs. base quality is unknown.** The only base-model video made on
  this laptop was flawed for reasons not yet established, and the only cloud
  run of the base model is a 2-step smoke test.
- **RAM has no margin** on 16 GB (see "Memory"). A run can page and slow
  down if anything else uses memory.
- **Reproducibility is only shown on one machine.** Other GPUs, drivers or
  torch builds may give different pixels for the same seed.
- ComfyUI logs that its optimised CUDA operations need a cu130 torch build;
  cu128 is installed. Timings may improve with cu130; not tested.
- `latent_scale` ≠ 1.0, `spatial_tile` ≠ 512, other samplers and the portrait
  size are accepted but have not been run on the real model.
- The negative prompt is ignored by the model (CFG 1); `reference_images`
  are ignored with a warning.
- Lossy output (CRF 12); the PNG frames are not kept.
- A reused ComfyUI is assumed to be the configured install; the adapter
  checks that it lists the model files, not that it is the same checkout.
- The adapter is tested against a fake ComfyUI HTTP server
  (`tests/fake_comfyui.py`); the tests need no GPU, model or ComfyUI.
