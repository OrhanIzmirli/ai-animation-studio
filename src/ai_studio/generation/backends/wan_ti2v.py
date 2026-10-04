"""Wan 2.2 TI2V-5B adapter: image-to-video through diffusers.

Everything model-specific lives here; the runner knows none of it.
See docs/BACKEND_WAN_TI2V_5B.md.

Heavy dependencies (torch, diffusers, huggingface_hub, Pillow, numpy) are
imported only inside prepare() and the helpers, never at module import, so
this module imports on a machine that has none of them.

Model facts the adapter relies on:
  - weights: Hugging Face "Wan-AI/Wan2.2-TI2V-5B-Diffusers" (Apache-2.0)
  - image-to-video uses WanImageToVideoPipeline, whose config has
    expand_timesteps=True for TI2V. (WanPipeline.__call__ has no image
    parameter, so it cannot condition on a start frame.)
  - VAE loaded separately in float32; the pipeline runs in bfloat16
  - native output: 24 fps, 1280x704 or 704x1280 (sides divisible by 32)
  - frame counts are 4n+1; the pipeline silently rounds anything else, so
    the adapter computes the count itself (frame_count_for)
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ai_studio.generation.backends.base import (
    BackendMetadata,
    BackendNotInstalledError,
    VideoGenerationBackend,
)
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import GenerationResult, StartFrameConditioning

MODEL_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# Default pinned commit of MODEL_REPO (34.20 GB, 29 files).
WAN_TI2V_5B_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
NATIVE_FPS = 24
NATIVE_RESOLUTIONS = ((1280, 704), (704, 1280))  # (width, height)
MAX_FRAMES = 121  # longer clips are not verified
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
ALLOWED_BACKEND_PARAMETERS = (
    "num_inference_steps", "guidance_scale", "revision", "cpu_offload", "vae_tiling",
)
ENCODER_SETTINGS = {
    "codec": "libx264", "pix_fmt": "yuv420p", "crf": 12, "preset": "slow",
    "fps": NATIVE_FPS, "audio": False,
}
# AutoencoderKLWan.tiled_encode / tiled_decode ignore config.patch_size before
# this release; the Wan 2.2 VAE uses it (2x2 patchify, 3 -> 12 channels).
MIN_DIFFUSERS_FOR_VAE_TILING = (0, 36, 0)
INSTALL_HINT = 'install the backend dependencies with: pip install -e ".[wan]"'


def frame_count_for(duration_seconds: float, fps: float = NATIVE_FPS) -> int:
    """Nearest valid frame count (4n+1) for a duration. 5 s at 24 fps -> 121."""
    n = math.floor((duration_seconds * fps - 1) / 4 + 0.5)  # round half up
    return 4 * max(n, 1) + 1


def preprocess_start_frame(image: Any, width: int, height: int) -> tuple[Any, list[str]]:
    """Scale-to-cover (LANCZOS) then center-crop a PIL image to width x height.

    Returns the new RGB image and the steps applied, in order. The input
    image object is not modified. 1672x941 -> 1280x720 -> crop 8 px top and
    bottom -> 1280x704.
    """
    from PIL import Image

    steps: list[str] = []
    if image.mode != "RGB":
        steps.append(f"convert {image.mode} -> RGB")
        image = image.convert("RGB")
    source_width, source_height = image.size
    if (source_width, source_height) == (width, height):
        return image.copy(), steps

    # The side that needs the larger scale factor lands exactly on the target;
    # the other side is >= its target and gets cropped.
    if width * source_height >= height * source_width:
        scaled = (width, max(height, round(source_height * width / source_width)))
    else:
        scaled = (max(width, round(source_width * height / source_height)), height)
    if scaled != (source_width, source_height):
        image = image.resize(scaled, Image.Resampling.LANCZOS)
        steps.append(
            f"resize {source_width}x{source_height} -> {scaled[0]}x{scaled[1]} (lanczos, scale to cover)"
        )
    if scaled != (width, height):
        left, top = (scaled[0] - width) // 2, (scaled[1] - height) // 2
        right, bottom = scaled[0] - width - left, scaled[1] - height - top
        image = image.crop((left, top, left + width, top + height))
        removed = ", ".join(
            f"{pixels} px {side}"
            for side, pixels in (("left", left), ("right", right), ("top", top), ("bottom", bottom))
            if pixels
        )
        steps.append(f"center-crop {scaled[0]}x{scaled[1]} -> {width}x{height} ({removed})")
    return image, steps


def prepare_start_frame(
    start_frame: str | Path, output_dir: str | Path, width: int, height: int,
) -> tuple[Any, StartFrameConditioning]:
    """Preprocess the start frame into a derived PNG inside output_dir. The
    original file is only ever opened for reading."""
    from PIL import Image

    with Image.open(start_frame) as original:
        original.load()
        image, steps = preprocess_start_frame(original, width, height)
    derived = Path(output_dir) / f"start_frame_{width}x{height}.png"
    derived.parent.mkdir(parents=True, exist_ok=True)
    image.save(derived, format="PNG")
    return image, StartFrameConditioning(
        preprocessing=steps, effective_width=width, effective_height=height, derived_file=str(derived),
    )


def frame_to_uint8(frame: Any) -> Any:
    """One H x W x 3 frame as uint8 RGB. Float frames are taken as [0, 1]:
    clipped, scaled and rounded (never truncated)."""
    import numpy as np

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape (height, width, 3), got {frame.shape}")
    if frame.dtype == np.uint8:
        return frame
    if not np.issubdtype(frame.dtype, np.floating):
        raise ValueError(f"frame must be float in [0, 1] or uint8, got {frame.dtype}")
    return np.rint(np.clip(np.nan_to_num(frame.astype(np.float32)), 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_frames(frames: Sequence[Any], output: str | Path, fps: int = NATIVE_FPS) -> dict[str, Any]:
    """Encode frames to H.264 by piping raw RGB into ffmpeg. Returns the
    encoder settings used. No audio, no frame-rate conversion."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode the output video")
    if len(frames) == 0:
        raise RuntimeError("the pipeline returned no frames")
    first = frame_to_uint8(frames[0])
    height, width = first.shape[:2]
    settings = dict(ENCODER_SETTINGS, fps=fps)
    command = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", settings["codec"], "-pix_fmt", settings["pix_fmt"],
        "-crf", str(settings["crf"]), "-preset", settings["preset"], "-r", str(fps), str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for index, frame in enumerate(frames):
            data = first if index == 0 else frame_to_uint8(frame)
            if data.shape != first.shape:
                raise ValueError(f"frame {index} has shape {data.shape}, expected {first.shape}")
            process.stdin.write(data.tobytes())
    except OSError:
        pass  # ffmpeg exited early; its exit code and stderr are reported below
    except BaseException:
        process.kill()
        process.communicate()
        raise
    _, stderr = process.communicate()  # closes stdin, waits
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({process.returncode}): {stderr.decode(errors='replace').strip()}")
    return settings


def version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric components: "0.35.2" -> (0, 35, 2), "0.36.0.dev0" -> (0, 36, 0)."""
    return tuple(int(part) for part in re.match(r"\d+(?:\.\d+)*", version).group().split("."))


def hf_hub_offline() -> bool:
    """True when HF_HUB_OFFLINE asks for the local cache only."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")


def _ffmpeg_version() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    try:
        words = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return words[2] if len(words) >= 3 and words[1] == "version" else None


class WanTI2V5BBackend(VideoGenerationBackend):
    """Image-to-video with Wan 2.2 TI2V-5B. Text-only generation is not
    verified yet, so a start_frame is required."""

    supports_start_frame = True

    def __init__(
        self,
        *,
        revision: str | None = WAN_TI2V_5B_REVISION,
        pipeline_factory: Callable[[], Any] | None = None,
    ) -> None:
        """`revision` pins the Hugging Face revision (commit sha, tag or
        branch; default: WAN_TI2V_5B_REVISION; None: unpinned);
        backend_parameters["revision"] does the same per request.
        `pipeline_factory` is for tests: it returns a callable taking the
        pipeline keyword arguments and returning an object with .frames[0];
        when given, prepare() imports neither torch nor diffusers."""
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ValueError("revision must be a non-empty string or None")
        self._revision = revision
        self._pipeline_factory = pipeline_factory
        self._cpu_offload = False
        self._vae_tiling = False
        self._pipe: Any = None
        # (revision, cpu_offload, vae_tiling) that _pipe was loaded with
        self._loaded: tuple[str | None, bool, bool] | None = None
        self._resolved_revision: str | None = None  # commit sha of the loaded weights
        self._make_generator: Callable[[int], Any] = lambda seed: None
        self._environment: dict[str, Any] = {}

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_name="wan-ti2v-5b",
            backend_version="upstream-diffusers-0.1",
            model_name="Wan2.2-TI2V-5B",
            model_version=self._resolved_revision or self._revision or "unpinned",
            workflow_version="i2v-v1",
        )

    # --- validation ---

    def validate_inputs(self, request: VideoGenerationRequest) -> None:
        if not isinstance(request, VideoGenerationRequest):
            raise ValueError(f"expected a VideoGenerationRequest, got {type(request).__name__}")
        if not request.start_frame:
            raise ValueError(
                "a start_frame is required: text-only generation is not verified for this backend yet"
            )
        if request.fps is not None and request.fps != NATIVE_FPS:
            raise ValueError(
                f"fps must be {NATIVE_FPS} (the model's native rate) or None; output is never resampled"
            )
        resolution = (request.output_resolution.width, request.output_resolution.height)
        if resolution not in NATIVE_RESOLUTIONS:
            allowed = " or ".join(f"{w}x{h}" for w, h in NATIVE_RESOLUTIONS)
            raise ValueError(f"output_resolution must be {allowed}, got {resolution[0]}x{resolution[1]}")
        frames = frame_count_for(request.duration_seconds)
        if frames > MAX_FRAMES:
            raise ValueError(
                f"duration {request.duration_seconds:g}s needs {frames} frames at {NATIVE_FPS} fps; "
                f"more than {MAX_FRAMES} frames is not verified"
            )
        settings = self._settings(request)
        # prepare() takes no request, so remember what it needs to load.
        self._revision, self._cpu_offload = settings["revision"], settings["cpu_offload"]
        self._vae_tiling = settings["vae_tiling"]

    def _settings(self, request: VideoGenerationRequest) -> dict[str, Any]:
        """Effective backend parameters: requested values over the defaults."""
        parameters = request.backend_parameters
        unknown = sorted(set(parameters) - set(ALLOWED_BACKEND_PARAMETERS))
        if unknown:
            raise ValueError(
                f"unknown backend_parameters {unknown}; allowed: {list(ALLOWED_BACKEND_PARAMETERS)}"
            )
        steps = parameters.get("num_inference_steps", DEFAULT_INFERENCE_STEPS)
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("num_inference_steps must be a positive int")
        guidance = parameters.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        if not isinstance(guidance, (int, float)) or isinstance(guidance, bool) or guidance < 0:
            raise ValueError("guidance_scale must be a number >= 0")
        revision = parameters.get("revision", self._revision)
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ValueError("revision must be a non-empty string")
        cpu_offload = parameters.get("cpu_offload", False)
        if not isinstance(cpu_offload, bool):
            raise ValueError("cpu_offload must be a bool")
        vae_tiling = parameters.get("vae_tiling", False)
        if not isinstance(vae_tiling, bool):
            raise ValueError("vae_tiling must be a bool")
        return {
            "num_inference_steps": steps, "guidance_scale": float(guidance),
            "revision": revision, "cpu_offload": cpu_offload, "vae_tiling": vae_tiling,
        }

    # --- model loading ---

    def prepare(self) -> None:
        wanted = (self._revision, self._cpu_offload, self._vae_tiling)
        if self._pipe is not None and self._loaded == wanted:
            return
        if self._pipeline_factory is not None:
            pipe = self._pipeline_factory()
            if self._vae_tiling:
                pipe.vae.enable_tiling()
            self._pipe, self._loaded = pipe, wanted
            return
        try:
            import diffusers
            import torch
            import transformers
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise BackendNotInstalledError(
                f"Backend 'wan-ti2v-5b' is not installed ({exc}); {INSTALL_HINT}"
            ) from exc
        if self._vae_tiling and version_tuple(diffusers.__version__) < MIN_DIFFUSERS_FOR_VAE_TILING:
            raise BackendNotInstalledError(
                f"vae_tiling needs diffusers >= 0.36.0, found {diffusers.__version__}: diffusers <0.36 "
                "tiles the Wan 2.2 VAE without patch_size; upgrade (pip install 'diffusers>=0.36.0')"
            )
        if not torch.cuda.is_available():
            raise BackendNotInstalledError(
                "Backend 'wan-ti2v-5b' is not installed for this machine: it needs a CUDA GPU "
                "and torch.cuda.is_available() is False"
            )

        # Resolve the revision to the commit actually on disk, then load from that
        # snapshot. With HF_HUB_OFFLINE=1 only the local cache is used: no network.
        try:
            snapshot = Path(snapshot_download(
                MODEL_REPO, revision=self._revision, local_files_only=hf_hub_offline(),
            ))
        except Exception as exc:
            if not hf_hub_offline():
                raise
            raise RuntimeError(
                f"HF_HUB_OFFLINE is set and {MODEL_REPO} at revision {self._revision!r} is not "
                f"complete in the local Hugging Face cache (HF_HOME={os.environ.get('HF_HOME')!r}): {exc}"
            ) from exc
        vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16)
        if getattr(pipe.config, "expand_timesteps", None) is not True:
            raise RuntimeError(
                f"wrong weights: {MODEL_REPO} at {snapshot.name} does not have expand_timesteps=True "
                "(not a TI2V checkpoint)"
            )
        if self._vae_tiling:
            pipe.vae.enable_tiling()
        if self._cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        self._pipe, self._loaded = pipe, wanted
        self._resolved_revision = snapshot.name  # snapshots/<commit sha>
        self._make_generator = lambda seed: torch.Generator(device="cuda").manual_seed(seed)
        self._environment = {
            "torch_version": torch.__version__,
            "diffusers_version": diffusers.__version__,
            "transformers_version": transformers.__version__,
            "torch_cuda_version": torch.version.cuda,
        }

    # --- generation ---

    def generate(self, request: VideoGenerationRequest, output_dir: Path) -> GenerationResult:
        if self._pipe is None:
            raise RuntimeError("prepare() must be called before generate()")
        settings = self._settings(request)
        width, height = request.output_resolution.width, request.output_resolution.height
        num_frames = frame_count_for(request.duration_seconds)
        output_dir = Path(output_dir)

        warnings: list[str] = []
        unused = [
            name for name in ("camera_motion", "subject_motion", "visual_direction")
            if getattr(request, name) is not None
        ]
        if unused:
            warnings.append(
                f"{', '.join(unused)} not consumed separately: only prompt and negative_prompt "
                "are passed to the model, unchanged"
            )
        if request.reference_images:
            warnings.append("reference_images are not used by this backend; only the start_frame conditions the video")

        image, conditioning = prepare_start_frame(request.start_frame, output_dir, width, height)
        result = self._pipe(
            image=image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=settings["num_inference_steps"],
            guidance_scale=settings["guidance_scale"],
            generator=self._make_generator(request.seed),
            output_type="np",
        )
        frames = result.frames[0]
        if len(frames) != num_frames:
            warnings.append(f"asked the pipeline for {num_frames} frames, it returned {len(frames)}")

        output = output_dir / "out.mp4"
        encoder = encode_frames(frames, output, NATIVE_FPS)

        revision = self._resolved_revision or settings["revision"]
        environment = dict(self._environment)
        ffmpeg_version = _ffmpeg_version()
        if ffmpeg_version:
            environment["ffmpeg_version"] = ffmpeg_version
        return GenerationResult(
            output_video=str(output),
            backend_name="wan-ti2v-5b",
            backend_version="upstream-diffusers-0.1",
            model_name="Wan2.2-TI2V-5B",
            model_version=revision or "unpinned",
            parameters={
                "model_repo": MODEL_REPO,
                "revision": revision,
                "pipeline": "WanImageToVideoPipeline",
                "num_inference_steps": settings["num_inference_steps"],
                "guidance_scale": settings["guidance_scale"],
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "fps": NATIVE_FPS,
                "cpu_offload": settings["cpu_offload"],
                "vae_tiling": settings["vae_tiling"],
                "torch_dtype": "bfloat16",
                "vae_dtype": "float32",
                "encoder": encoder,
            },
            environment=environment,
            warnings=warnings,
            fps=NATIVE_FPS,
            frame_count=len(frames),
            inference_steps=settings["num_inference_steps"],
            guidance=settings["guidance_scale"],
            start_frame_conditioning=conditioning,
        )
