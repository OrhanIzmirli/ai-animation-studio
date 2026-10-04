# Spike 6a: Wan 2.2 TI2V-5B GGUF in ComfyUI on the RTX 4060 laptop

Date: 2026-10-04. Throwaway feasibility spike, not production code.
Script: [`comfyui_wan_spike.py`](comfyui_wan_spike.py). Raw results (git-ignored):
`spikes/output/<run>/result.json`, `comfyui.log`, `samples.json`, frames, `out.mp4`.

## Answer

**Yes. This laptop can generate the `comic_rooftop_01` shot locally at full
size (1280×704, 121 frames, 24 fps), without running out of memory.**

- With the **Turbo** GGUF (4 steps) and a **spatially tiled VAE decode** it
  takes **3.6 minutes** per clip (run d).
- With ComfyUI's default (untiled) VAE decode the same clip takes 16.6
  minutes, 14.4 of them in the decode (run a).
- The **base** GGUF at 20 steps takes 34 minutes, pages heavily, and its
  video has two visible colour glitches (run b).

Everything below is measured on one seed (42). Nobody has rated the videos;
the numbers say the pipeline works and how fast, not that the result is good.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8188 MiB, driver 617.14 |
| CPU / RAM | Intel Core i5-12450H, 15.7 GB RAM |
| Disk | NVMe SSD (Kingston OM8PGP4512Q) |
| OS | Windows 11 Pro, build 26300 |
| Power | AC (confirmed before each run) |
| Free space on C: | 128.5 GB before, 100.7 GB after (27.8 GB used by ComfyUI, models, outputs) |
| Pagefile | system-managed; grew from 5632 MB to 7119 MB during the spike |

## Software

Installed outside the repository in `C:\Users\rhn23\ai-tools\ComfyUI`.

| | |
|---|---|
| ComfyUI | git clone, tag `v0.38.2`, commit `daeb5e53681e2b10a3f0727d9ec5bc90784bee10` (2026-10-02) |
| ComfyUI-GGUF (city96) | commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` (2026-01-12) |
| Python | 3.11.9, own venv (`ComfyUI\venv`) |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| Other | gguf 0.19.0, transformers 5.18.0, safetensors 0.8.0, numpy 2.4.6, av 18.1.0 |
| ffmpeg (encoding, metrics) | 8.1.2 (gyan.dev full build) |

**Why a git clone with its own venv, not the Windows portable build:** the
exact commit can be recorded and checked out again, the system Python 3.11
was already there, and no 7-Zip extraction step is needed. The cost is one
manual torch install.

## Model files

All four SHA-256 values were computed locally and match the values published
on Hugging Face.

| File (in `ComfyUI\models\...`) | Repo @ revision | Bytes | SHA-256 |
|---|---|---|---|
| `diffusion_models\Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` | `hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF` @ `9873ba17` | 5,404,990,176 | `b4d7d1888d5ec2c29046cc443239faad99614cd3d356c61d8192fcac5f4c98fc` |
| `diffusion_models\Wan2.2-TI2V-5B-Q8_0.gguf` | `QuantStack/Wan2.2-TI2V-5B-GGUF` @ `57437632` | 5,400,179,040 | `57bece983817ab2f957546683bb670f13be7d99022d45674840cd999a050ea8f` |
| `vae\wan2.2_vae.safetensors` | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` @ `ee6f4a40`, `split_files/vae/` | 1,409,400,960 | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` |
| `text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors` | same repo, `split_files/text_encoders/` | 6,735,906,897 | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` |

Total 18.95 GB. The VAE and text encoder are the official ComfyUI repackaged
files. Both GGUF repos are Apache-2.0 per their cards.

## How the runs were made

- ComfyUI started headless (`main.py --listen 127.0.0.1 --port 8199`) and
  driven only over HTTP: `GET /system_stats`, `GET /object_info/<node>`,
  `POST /upload/image`, `POST /prompt`, `GET /history/<id>`, `GET /view`.
- The workflow is API-format JSON. Before queueing, every node class, every
  input name and every choice value is checked against `/object_info` of the
  running server. Nodes used and where they come from:
  `UnetLoaderGGUF` (ComfyUI-GGUF); `CLIPLoader` (type `wan`), `VAELoader`,
  `LoadImage`, `CLIPTextEncode`, `KSampler`, `VAEDecode` / `VAEDecodeTiled`,
  `SaveImage`, `SaveLatent` (core `nodes`); `ModelSamplingSD3`
  (`comfy_extras.nodes_model_advanced`); `Wan22ImageToVideoLatent`
  (`comfy_extras.nodes_wan`).
- Sampler `uni_pc`, scheduler `simple`, `ModelSamplingSD3` shift 8. These are
  the values in ComfyUI's own template `video_wan2_2_5B_ti2v.json` shipped
  with this install (which also uses 20 steps, CFG 5, 121 frames, plain
  `VAEDecode`). The Turbo card allows `uni_pc` and `simple`.
- **Inputs:** the canonical start frame, preprocessed by the Wan adapter's
  own function (`resize 1672x941 -> 1280x720 (lanczos, scale to cover)`,
  `center-crop 1280x720 -> 1280x704 (8 px top, 8 px bottom)`) into a temp
  directory, original untouched; prompt and negative prompt verbatim from
  `benchmarks/definitions/comic_rooftop_01.yaml`; 121 frames; seed 42.
- **Output:** 121 PNG frames via `SaveImage`, fetched over `/view`, then
  encoded with the Wan adapter's `encode_frames` (libx264, yuv420p, CRF 12,
  preset slow, 24 fps, no audio).
- **Timing split:** two prompts are queued. A = everything up to the sampler
  plus `SaveLatent`; B = the same graph plus decode and `SaveImage`, so A's
  nodes come from ComfyUI's cache and B is decode + saving only. Sampling
  time is taken from the sampler's progress lines in the server log; decode
  vs. PNG saving from the first PNG's modification time.
- **Memory:** sampled once per second: `nvidia-smi` memory.used (whole GPU),
  system RAM and swap (psutil), and the ComfyUI process tree.

## Results

All four runs: 1280×704, 121 frames, seed 42, status success, no
out-of-memory error, 121 frames returned. No retry at 960×544 was needed.

| | (a) Turbo | (b) base | (c) Turbo, tiled | (d) Turbo, tiled spatial-only |
|---|---|---|---|---|
| GGUF | Turbo Q8_0 | base Q8_0 | Turbo Q8_0 | Turbo Q8_0 |
| Steps / CFG | 4 / 1 | 20 / 5 | 4 / 1 | 4 / 1 |
| VAE decode node | `VAEDecode` | `VAEDecode` | `VAEDecodeTiled` 512 / 64, temporal 64 / 8 | `VAEDecodeTiled` 512 / 64, temporal 128 / 8 |
| **Total wall time** | **997 s (16.6 min)** | **2039 s (34.0 min)** | **227 s (3.8 min)** | **218 s (3.6 min)** |
| Load + text encode + start-frame encode | 58 s | 61 s | 38 s | 38 s |
| Sampling | 63 s | 787 s | 63 s | 63 s |
| Seconds per step | 15.8 | 39.4 | 15.6 | 15.7 |
| VAE decode | 867 s | 1181 s | 117 s | 109 s |
| Saving 121 PNGs | 8 s | 8 s | 8 s | 8 s |
| Peak VRAM, load + sampling | 7153 MiB | 7313 MiB | 7177 MiB | 7177 MiB |
| Peak VRAM, decode | 7889 MiB | 7931 MiB | 4937 MiB | 4905 MiB |
| Peak system RAM used | 15.7 GB | 15.7 GB | 15.3 GB | 15.2 GB |
| Lowest available RAM, load + sampling | 0.02 GB | 0.24 GB | 0.41 GB | 0.51 GB |
| Lowest available RAM, decode | 1.32 GB | 0.004 GB | 4.13 GB | 3.41 GB |
| Peak swap in use (before the run) | 2.35 GB (0.40) | 6.54 GB (1.00) | 0.93 GB (0.93) | 1.15 GB (0.88) |
| ComfyUI process, peak working set | not captured | 8.8 GB | 8.9 GB | 9.1 GB |
| ComfyUI process, peak committed | not captured | 26.7 GB | 15.5 GB | 15.5 GB |
| Out of memory | no | no | no | no |
| Paging | yes, during load | **heavy**, during decode | no growth | slight, during load |
| ComfyUI server start | 24 s | 13 s | 14 s | 13 s |

ComfyUI kept the diffusion model only partly on the GPU in every run
("loaded partially; 3388 MB loaded, 1848 MB offloaded"). It never fell back
to tiled decoding by itself.

Not included in "total": starting ComfyUI (13–24 s), fetching the frames over
HTTP and the ffmpeg encode.

### Quality metrics (metrics v1, against the derived start frame)

| Metric | (a) Turbo | (b) base | (c) Turbo, tiled | (d) Turbo, spatial-only | L4 smoke run* |
|---|---|---|---|---|---|
| motion_mean | 1.682 | 1.055 | 1.820 | 1.684 | 0.689 |
| motion_max | 3.138 | 11.682 | 3.709 | 3.138 | 1.286 |
| total_change | 27.64 | 3.23 | 27.56 | 27.64 | 4.15 |
| start_frame_psnr (dB) | 41.37 | 41.38 | 41.29 | 41.29 | 41.66 |
| temporal_pulse_ratio | 1.20 | 2.73 | 1.33 | 1.20 | 2.41 |
| luminance_flicker | 0.473 | 1.404 | 1.222 | 0.470 | 0.250 |
| luminance min .. max | 51.98 .. 54.87 | 49.38 .. 60.70 | 47.76 .. 54.64 | 52.02 .. 54.89 | 55.02 .. 57.55 |
| held_frame_ratio | 0.000 | 0.467 | 0.000 | 0.000 | 0.058 |

\* `comic_rooftop_01 / 20261004T145422Z-a264ad6e`: base model through
diffusers on an L4, **2 steps**. It is a smoke test, so it is context, not a
quality reference.

## What the measurements show

1. **The VAE decode, not the diffusion model, decides the run time.** In run
   (a) sampling took 63 s and the decode 867 s. During the untiled decode the
   GPU sat at 7.9 of 8.0 GB for the whole 14 minutes.
2. **Tiled decoding fixes that.** Decode time drops from 867 s to 109–117 s,
   peak VRAM in the decode from 7.9 GB to 4.9 GB, and the paging in the
   decode disappears.
3. **But the default temporal tiling damages the video.** With the default
   64-frame temporal tiles (run c) the mean luminance dips where the tiles
   join: 52.5 → 47.8 around frame 60 and 54.6 → 51.3 around frame 114.
   `luminance_flicker` goes from 0.47 to 1.22. Same seed, same sampling.
4. **One temporal tile avoids the damage.** With `temporal_size` 128 (run d)
   all 121 frames are one temporal tile and only spatial tiling remains.
   Every metric matches the untiled run (a) to the third decimal or close,
   there is no luminance dip, and it is the fastest run. The frames are not
   pixel-identical to (a): mean difference 0.54 on the 0–255 scale, evenly
   spread over all frames (maximum 0.67). Whether that comes from the spatial
   tile blending or from sampling not being bit-reproducible between runs was
   not separated.
5. **System RAM is the tight resource, not VRAM.** While the text encoder
   (6.4 GB) and the model load, available RAM falls to 0.0–0.5 GB of 15.7 GB
   in every run. It worked four times out of four, but with no margin:
   anything else using memory on the laptop would push it into the pagefile.
6. **The base model is slower per step and much slower overall.** 39.4 s per
   step against 15.7 (CFG 5 runs the model twice per step; CFG 1 once), times
   five as many steps. For comparison, the L4 ran the base model through
   diffusers at about 30 s per step.
7. **The base run's video is flawed.** Around frames 24–29 and 92–97 the
   picture turns green/blue-tinted and smeared for a few frames, then
   recovers (checked by eye on frames 91, 93, 94, 97; `motion_max` 11.7).
   Outside those episodes it is nearly static: 47% of transitions are held
   and the last frame differs from the first by only 3.2. **The cause is not
   established.** It could be the sampling (quantised model, this
   sampler/shift at 20 steps) or the untiled decode running in a machine
   that was paging 6.5 GB at the time. The saved latent was not re-decoded to
   find out.
8. **The Turbo run moves; the base run barely does.** `total_change` 27.6
   against 3.2. Frame 60 of run (a) was checked by eye: a coherent image of
   the shot (character from behind on the ledge, city at sunset). No frame
   sequence was watched as a video by a person.

## Problems hit

- **pip hung.** The first `pip install torch` sat for 56 minutes without
  downloading anything. Killed and restarted with `--timeout 60 --retries 10`;
  it then finished in about 15 minutes.
- **torch build.** ComfyUI v0.38.2 logs "You need pytorch with cu130 or
  higher to use optimized CUDA operations". cu128 was installed, so the
  timings may be pessimistic. Not retested with cu130.
- **Process memory missing for run (a).** The monitor watched the venv's
  `python.exe` launcher instead of the real interpreter. Fixed before run
  (b); run (a)'s system-wide RAM, swap and VRAM numbers are unaffected.
- **Run (a) loaded from a cold disk cache** (58 s against 38 s for the later
  Turbo runs), and its baseline RAM use was higher (9.1 GB against 6.4–6.9).
- **Run (d) was disturbed slightly:** a disk-usage command of mine ran during
  its decode. Its decode time (109 s) is therefore, if anything, a little
  high.
- **ComfyUI's own output folder** (`ai-tools\ComfyUI\output\spike_*`) holds a
  second copy of every frame set and the saved latents, about 0.8 GB. Safe
  to delete.

## Not tested

- base GGUF with tiled decode (expected roughly 15 min from the parts
  measured: 61 + 787 + ~110 s; an estimate, not a measurement)
- whether the base run's glitches come from sampling or from the decode
- more than one seed; run-to-run reproducibility of sampling
- torch cu130; smaller quantisations (Q4–Q6); the `euler` sampler; the "0.8
  latent multiplier" the Turbo card mentions against oversaturation
- a GGUF text encoder, which would ease the RAM pressure during load
- 960×544 (never needed)
- any human rating of the videos

## Recommendation for Task 6b

Turn **run (d)** into the local backend adapter:

| | |
|---|---|
| Model | `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` |
| Steps / CFG | 4 / 1 |
| Sampler / scheduler / shift | `uni_pc` / `simple` / 8 |
| Decode | `VAEDecodeTiled`, tile 512, overlap 64, **temporal_size ≥ number of frames** (128 for 121 frames), temporal overlap 8 |
| Text encoder / VAE | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` / `wan2.2_vae.safetensors` |
| Cost | 3.6 min per 121-frame clip, peak VRAM 7.2 GB, no paging in the decode |

Points the adapter must get right:

1. **It is a different model.** Turbo is a distilled finetune of TI2V-5B, not
   a quantised copy of it. It needs its own backend name and model identity,
   and its results must not be reported as "Wan 2.2 TI2V-5B". Whether its
   quality is close to the base model is an open benchmark question, not
   something this spike shows.
2. **Never use the default temporal tile size.** It must cover the whole
   clip; otherwise the luminance dips of run (c) come back. This should be
   computed from the frame count, not left as a parameter someone can forget.
3. **Record the file hashes and both git commits** in every run, as the
   table above does.
4. **RAM has no margin.** The adapter should refuse to start, or at least
   warn, when little RAM is free, and the run should be the only heavy thing
   on the laptop.
5. ComfyUI is a separate process with its own venv: the adapter talks HTTP
   to it and adds no dependency to the project, which is what the spike
   script already does.

The **base Q8_0 at 20 steps is not worth an adapter yet**: 34 minutes per
clip on this machine, heavy paging, and a flawed result whose cause is
unknown. If the base model is wanted locally, the first step is a rerun with
the tiled decode of run (d); that would show both the real time and whether
the glitches survive.
