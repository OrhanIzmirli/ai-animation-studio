# Backend: Wan 2.2 TI2V-5B (image-to-video)

Adapter: [`src/ai_studio/generation/backends/wan_ti2v.py`](../src/ai_studio/generation/backends/wan_ti2v.py),
class `WanTI2V5BBackend`. Contract: `VideoGenerationBackend`,
`supports_start_frame = True`.

> **Status: implemented and tested with an injected fake pipeline. Not yet run
> on a GPU.** The tests cover everything around the model (validation, frame
> rule, start-frame preprocessing, encoding, the runner's measurement) but not
> the diffusers calls themselves. The first real run is the next step.

| | |
|---|---|
| `backend_name` | `wan-ti2v-5b` |
| `backend_version` | `upstream-diffusers-0.1` |
| `model_name` | `Wan2.2-TI2V-5B` |
| `model_version` | resolved Hugging Face commit sha; before loading, the pinned revision (default `b8fff7315c768468a5333511427288870b2e9635`) |
| `workflow_version` | `i2v-v1` |
| Weights | Hugging Face `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, Apache-2.0 |

## Why this model first

- **Native 24 fps.** `comic_rooftop_01` pins 24 fps and the runner measures
  it. This model produces 24 fps directly, so nothing is resampled.
- **Real start-frame semantics.** It conditions frame 0 on an image, which is
  what the benchmark's `start_frame` means. It is not a reference-image model
  being asked to imitate one.
- **Fits one GPU.** A 5B model that runs on a single 24 GB+ card, so baseline
  runs are possible before renting larger hardware.
- **Apache-2.0** weights.

It is the first candidate, not the chosen model. No winner is picked in
advance; it is ranked by its benchmark results like any later backend.

## Install

```bash
# 1. the torch build for your CUDA version (see pytorch.org), then:
pip install -e ".[wan]"
```

The `wan` extra is `torch`, `diffusers>=0.35`, `transformers`, `accelerate`,
`ftfy`, `pillow`, `numpy`, `huggingface_hub`. **Use `diffusers==0.35.2`** for
benchmark runs: 0.35 is the release line that added Wan 2.2, and the adapter
was written against it. Newer releases are allowed by the extra but have not
been checked. The diffusers version actually used is recorded in every run's
`environment`. For Camber there is a fully pinned set,
`requirements/camber-wan.txt`; see [RUNNING_ON_CAMBER.md](RUNNING_ON_CAMBER.md).

Also required on `PATH`: `ffmpeg` (encoding) and `ffprobe` (the runner's
measurement).

Hardware: a CUDA GPU with 24 GB+ VRAM. `prepare()` raises
`BackendNotInstalledError` when the dependencies are missing or
`torch.cuda.is_available()` is false; such a run is not recorded. Weights are
downloaded by `prepare()` on first use into the Hugging Face cache
(34.20 GB, 29 files at the pinned revision).

**Offline use.** With `HF_HUB_OFFLINE=1` the weights are resolved from the
local cache only (`snapshot_download(..., local_files_only=True)`) and no
network request is made. If the pinned revision is not complete in the cache,
the run fails with a message naming `HF_HOME`.

**24 GB cards (e.g. L4).** On disk the transformer is ~20 GB in fp32 (~10 GB
in bf16 once loaded), the text encoder ~11.4 GB and the VAE ~2.8 GB. Together
they do not fit in 24 GB of VRAM, so use `cpu_offload=true` there. If decoding
still runs out of memory, add `vae_tiling=true`.

## What a request must look like

`validate_inputs()` rejects the request, and the runner records a failed run
with `failure_stage: validation`, when:

| Rule | Reason |
|---|---|
| `start_frame` is missing | text-only generation is not verified for this backend yet |
| `fps` is set and is not 24 | native rate; the adapter never resamples |
| `output_resolution` is not 1280×704 or 704×1280 | native sizes; both sides divisible by 32 (VAE spatial factor 16 × patch size 2) |
| the duration needs more than 121 frames | longer clips are not verified |
| `backend_parameters` has an unknown key | a typo must not silently fall back to a default |

Allowed `backend_parameters`:

| Key | Default | |
|---|---|---|
| `num_inference_steps` | 50 | positive int |
| `guidance_scale` | 5.0 | number ≥ 0 |
| `revision` | `b8fff7315c768468a5333511427288870b2e9635` (`WAN_TI2V_5B_REVISION`) | Hugging Face commit sha, tag or branch; the constructor's `revision` changes the default |
| `cpu_offload` | `false` | `true` uses `enable_model_cpu_offload()` instead of moving the pipeline to the GPU |
| `vae_tiling` | `false` | `true` calls `pipe.vae.enable_tiling()`: lower VRAM while decoding. A way to retry an out-of-memory failure without a code change |

## Frame rule

The model generates **4n+1** frames. The diffusers pipeline silently rounds
any other `num_frames`, so the adapter computes it itself and records it:

```
n = round((duration_seconds * 24 - 1) / 4)      (half rounds up, n >= 1)
num_frames = 4n + 1
```

5 s → 121 frames = 5.042 s, inside the benchmark's 0.1 s tolerance. 4 s → 97,
2 s → 49. If the pipeline returns a different number of frames than asked,
the adapter reports the real count with a warning and the runner's measured
duration check decides.

## Start-frame preprocessing

The start frame is scaled to cover the target size with LANCZOS, then
center-cropped. Aspect ratio is kept; nothing is stretched or padded.

For the canonical 1672×941 frame at 1280×704:

```
resize 1672x941 -> 1280x720 (lanczos, scale to cover)
center-crop 1280x720 -> 1280x704 (8 px top, 8 px bottom)
```

These exact strings go into the record's `start_frame.preprocessing`. A
non-RGB image additionally gets `convert <mode> -> RGB` first; an image that
is already RGB at the target size gets no steps. The result is saved as
`start_frame_<width>x<height>.png` inside the run's output directory and
recorded as `derived_file`. The original file is only opened for reading; the
runner's checksum check would fail the run otherwise.

## Generation

- Pipeline: `diffusers.WanImageToVideoPipeline`, `torch_dtype=bfloat16`, with
  `AutoencoderKLWan` loaded separately from the `vae` subfolder in `float32`.
  After loading, the adapter requires `pipe.config.expand_timesteps is True`
  and raises `RuntimeError` otherwise (wrong weights). The model card's
  example passes `image=` to `WanPipeline`; `WanPipeline.__call__` has no
  `image` parameter, so that is not used.
- Weights are resolved with `huggingface_hub.snapshot_download(repo, revision=...)`
  and loaded from that snapshot. The revision is pinned by default to commit
  `b8fff7315c768468a5333511427288870b2e9635`. The resolved commit sha is
  recorded as `model_version` and `model_parameters.revision`.
- Seed: `torch.Generator(device="cuda").manual_seed(seed)`.
- `prompt` and `negative_prompt` are passed **unchanged**. `camera_motion`,
  `subject_motion` and `visual_direction` are not folded into the prompt or
  mapped to anything; the run gets a warning saying so. The benchmark prompt
  already describes them.
- The pipeline is called with `output_type="np"`.

## Encoding

Frames are converted from float [0, 1] to 8-bit RGB (clip, ×255, round) and
piped as raw `rgb24` into ffmpeg:

```
ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r 24 -i - -an
       -c:v libx264 -pix_fmt yuv420p -crf 12 -preset slow -r 24 out.mp4
```

No audio, no frame-rate conversion. The settings are recorded in
`model_parameters.encoder`, the ffmpeg version in `environment`. A missing
ffmpeg is a `RuntimeError` and therefore a recorded failed run.

## What a run records

`model_parameters`: `model_repo`, `revision`, `pipeline`,
`num_inference_steps`, `guidance_scale`, `num_frames`, `width`, `height`,
`fps`, `cpu_offload`, `vae_tiling`, `torch_dtype`, `vae_dtype`, `encoder`.
`environment` additionally gets `torch_version`, `diffusers_version`,
`transformers_version`, `torch_cuda_version`, `ffmpeg_version`.
`fps` / `frame_count` / `output_media` come from the runner's ffprobe
measurement, not from the adapter.

## Known limits

- **Not run on a GPU yet** (see status above). Loading, VRAM use, timings and
  output quality are unverified.
- Image-to-video only; text-only requests are rejected.
- 24 fps and the two native resolutions only; at most 121 frames (~5 s).
- `reference_images` are ignored, with a warning. Only the start frame
  conditions the video.
- `metadata()` is read before the request is seen, so a run that fails before
  generation finishes records `model_version` as the adapter's default
  revision, even when `backend_parameters.revision` asked for another one.
  Successful runs record the resolved commit sha.
- `vae_tiling` may change the decoded pixels slightly (tile blending), so
  compare runs with the same setting. It is recorded in every run.
- The H.264 stream carries no colour-space tags; ffmpeg's default RGB→YUV
  conversion is used.
- Lossy output (CRF 12). Frames are not kept in a lossless form.

## Tests

`tests/test_wan_ti2v.py` needs no torch, diffusers or weights: the pipeline is
injected through `WanTI2V5BBackend(pipeline_factory=...)`. It includes two
end-to-end runs through the real `BenchmarkRunner` with real ffmpeg encoding
and the real ffprobe measurement (121 frames pass; 97 frames fail
verification). Those are skipped when ffmpeg is not installed.
