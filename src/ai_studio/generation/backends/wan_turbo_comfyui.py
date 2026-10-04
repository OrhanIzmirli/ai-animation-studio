"""Wan 2.2 TI2V-5B *Turbo* (GGUF Q8_0) through a local ComfyUI.

NOT the base Wan 2.2 TI2V-5B: Turbo is quanhaol's distilled finetune of it
(4 steps, CFG 1). It has its own backend name and model identity so that its
results are never mixed with the base model's. See
docs/BACKEND_WAN_TI2V_5B_TURBO_COMFYUI.md and spikes/LOCAL_COMFYUI_SPIKE.md
(the settings are the spike's run (d)).

The adapter owns the ComfyUI process: prepare() starts it headless on
127.0.0.1 (or reuses one already answering on the configured port) and
close() stops a server it started. Use it as a context manager, or call
close(). All model work happens inside ComfyUI; this module only speaks
HTTP, so it needs no torch.

VAE decoding is tiled spatially only. The temporal tile always covers the
whole clip (temporal_tile_for): with a shorter temporal tile the video gets a
brightness dip at every tile seam (spike run (c)). It is therefore computed
from the frame count and is not a parameter.
"""

from __future__ import annotations

import io
import math
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from ai_studio.generation.backends.base import (
    BackendMetadata,
    BackendNotInstalledError,
    VideoGenerationBackend,
)
from ai_studio.generation.backends.comfyui import (
    MODEL_KINDS,
    ComfyUIConfig,
    ComfyUIServer,
    git_commit,
    load_config,
    sha256_file,
)
from ai_studio.generation.backends.wan_ti2v import (
    MAX_FRAMES,
    NATIVE_FPS,
    NATIVE_RESOLUTIONS,
    _ffmpeg_version,
    encode_frames,
    frame_count_for,
    prepare_start_frame,
)
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import GenerationResult
from ai_studio.system_memory import LowestAvailableRam, available_ram_gb, total_ram_gb

BACKEND_NAME = "wan-ti2v-5b-turbo-comfyui"
BACKEND_VERSION = "comfyui-0.1"
MODEL_NAME = "Wan2.2-TI2V-5B-Turbo (quanhaol) GGUF Q8_0"
WORKFLOW_VERSION = "i2v-turbo-tiled-v1"

CFG = 1.0  # the Turbo model is distilled for CFG 1; not a parameter
DEFAULTS = {
    "num_inference_steps": 4,
    "shift": 8.0,
    "sampler": "uni_pc",
    "scheduler": "simple",
    "spatial_tile": 512,
    "latent_scale": 1.0,
}
ALLOWED_BACKEND_PARAMETERS = tuple(DEFAULTS)
SPATIAL_TILE_OVERLAP = 64
TEMPORAL_TILE_OVERLAP = 8
# Below this much available RAM before loading, the run gets a warning. In the
# spike, a run that started with 6.6 GB available paged about 2 GB while
# loading; runs that started with 8.8 GB or more did not grow the pagefile.
LOW_RAM_WARNING_GB = 8.0
OUTPUT_PREFIX = "ai_studio_runs"

# Node classes the workflow needs (LatentMultiply only when latent_scale != 1).
REQUIRED_NODES = (
    "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "LoadImage", "CLIPTextEncode",
    "ModelSamplingSD3", "Wan22ImageToVideoLatent", "KSampler", "VAEDecodeTiled", "SaveImage",
)
LATENT_SCALE_NODE = "LatentMultiply"
# Which node input lists each model file.
MODEL_INPUTS = {
    "unet": ("UnetLoaderGGUF", "unet_name"),
    "text_encoder": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
}


def temporal_tile_for(num_frames: int) -> int:
    """Temporal tile size (in frames) for the tiled VAE decode: a multiple of
    4 strictly greater than the clip, so the whole clip is ONE temporal tile.
    121 frames -> 128."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    return 4 * (math.ceil(num_frames / 4) + 1)


def _choices(spec: Any) -> list[Any] | None:
    """Allowed values of a node input from its /object_info spec, if it is a choice."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if len(spec) > 1 and isinstance(spec[1], dict) and isinstance(spec[1].get("options"), list):
        return spec[1]["options"]
    return None


def build_workflow(
    *, models: dict[str, str], image_name: str, prompt: str, negative_prompt: str,
    width: int, height: int, num_frames: int, seed: int, settings: dict[str, Any], output_prefix: str,
) -> dict[str, Any]:
    """The ComfyUI API-format workflow (spike run (d))."""
    workflow: dict[str, Any] = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": models["unet"]}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": models["text_encoder"], "type": "wan", "device": "default",
        }},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"shift": settings["shift"], "model": ["1", 0]}},
        "8": {"class_type": "Wan22ImageToVideoLatent", "inputs": {
            "vae": ["3", 0], "width": width, "height": height, "length": num_frames,
            "batch_size": 1, "start_image": ["4", 0],
        }},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "seed": seed, "steps": settings["num_inference_steps"], "cfg": CFG,
            "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
            "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0], "denoise": 1.0,
        }},
    }
    latent = ["9", 0]
    if settings["latent_scale"] != 1.0:
        workflow["10"] = {"class_type": LATENT_SCALE_NODE, "inputs": {
            "samples": latent, "multiplier": settings["latent_scale"],
        }}
        latent = ["10", 0]
    workflow["11"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": latent, "vae": ["3", 0],
        "tile_size": settings["spatial_tile"], "overlap": SPATIAL_TILE_OVERLAP,
        "temporal_size": temporal_tile_for(num_frames), "temporal_overlap": TEMPORAL_TILE_OVERLAP,
    }}
    workflow["12"] = {"class_type": "SaveImage", "inputs": {
        "images": ["11", 0], "filename_prefix": f"{output_prefix}/frame",
    }}
    return workflow


class _RemoteFrames:
    """The run's PNG frames, fetched from ComfyUI one at a time (the encoder
    needs len(), indexing and iteration)."""

    def __init__(self, client: Any, images: list[dict[str, Any]]) -> None:
        self._client, self._images = client, images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Any:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(self._client.view(self._images[index]))) as image:
            return np.asarray(image.convert("RGB"))

    def __iter__(self):
        return (self[index] for index in range(len(self)))


class WanTurboComfyUIBackend(VideoGenerationBackend):
    """Image-to-video with the Wan 2.2 TI2V-5B Turbo GGUF in a local ComfyUI."""

    supports_start_frame = True

    def __init__(self, config: ComfyUIConfig | None = None) -> None:
        """`config`: None reads configs/comfyui.local.yaml when first needed."""
        self._config = config
        self._server: ComfyUIServer | None = None
        self._hashes: dict[str, str] = {}
        self._environment: dict[str, Any] = {}
        self._latent_scale_needed = False
        self._ram_before_gb: float | None = None

    def __enter__(self) -> WanTurboComfyUIBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the ComfyUI server if this adapter started it."""
        if self._server is not None:
            self._server.close()
            self._server = None

    def metadata(self) -> BackendMetadata:
        pinned = None
        try:
            pinned = self._hashes.get("unet") or self._load_config().sha256.get("unet")
        except BackendNotInstalledError:
            pass  # metadata must work without ComfyUI configured
        return BackendMetadata(
            backend_name=BACKEND_NAME, backend_version=BACKEND_VERSION, model_name=MODEL_NAME,
            model_version=pinned or "unverified", workflow_version=WORKFLOW_VERSION,
        )

    def _load_config(self) -> ComfyUIConfig:
        if self._config is None:
            self._config = load_config()
        return self._config

    # --- validation ---

    def validate_inputs(self, request: VideoGenerationRequest) -> None:
        if not isinstance(request, VideoGenerationRequest):
            raise ValueError(f"expected a VideoGenerationRequest, got {type(request).__name__}")
        if not request.start_frame:
            raise ValueError("a start_frame is required: this backend is image-to-video only")
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
        if request.seed < 0:
            raise ValueError("seed must be >= 0 for the ComfyUI sampler")
        settings = self._settings(request)
        # prepare() takes no request, so remember which optional node it must verify.
        self._latent_scale_needed = settings["latent_scale"] != 1.0

    @staticmethod
    def _settings(request: VideoGenerationRequest) -> dict[str, Any]:
        """Effective backend parameters: requested values over the defaults."""
        parameters = request.backend_parameters
        unknown = sorted(set(parameters) - set(ALLOWED_BACKEND_PARAMETERS))
        if unknown:
            raise ValueError(
                f"unknown backend_parameters {unknown}; allowed: {list(ALLOWED_BACKEND_PARAMETERS)}"
            )
        settings = {**DEFAULTS, **parameters}

        def is_number(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        steps = settings["num_inference_steps"]
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("num_inference_steps must be a positive int")
        if not is_number(settings["shift"]) or settings["shift"] <= 0:
            raise ValueError("shift must be a number > 0")
        for name in ("sampler", "scheduler"):
            if not isinstance(settings[name], str) or not settings[name]:
                raise ValueError(f"{name} must be a non-empty string")
        tile = settings["spatial_tile"]
        if not isinstance(tile, int) or isinstance(tile, bool) or not 64 <= tile <= 4096 or tile % 32:
            raise ValueError("spatial_tile must be an int between 64 and 4096, divisible by 32")
        if not is_number(settings["latent_scale"]) or not 0 < settings["latent_scale"] <= 10:
            raise ValueError("latent_scale must be a number in (0, 10]")
        settings["shift"], settings["latent_scale"] = float(settings["shift"]), float(settings["latent_scale"])
        return settings

    # --- ComfyUI lifecycle ---

    def prepare(self) -> None:
        if self._server is not None and self._server.client.is_up():
            self._check_nodes(self._server)  # the optional node may be new for this request
            return
        config = self._load_config()
        if not config.comfyui_dir.is_dir():
            raise BackendNotInstalledError(f"ComfyUI not found: {config.comfyui_dir} is not a directory")
        paths = {kind: config.model_path(kind) for kind in MODEL_KINDS}
        missing = [f"{kind}: {config.models[kind]}" for kind, path in paths.items() if path is None]
        if missing:
            raise BackendNotInstalledError(
                f"model files not found under {config.comfyui_dir / 'models'}: {', '.join(missing)}"
            )

        self._ram_before_gb = available_ram_gb()  # before anything is loaded
        started = time.perf_counter()
        hashes = {kind: sha256_file(path) for kind, path in paths.items()}
        hash_seconds = time.perf_counter() - started
        for kind, expected in config.sha256.items():
            if hashes[kind] != expected:
                raise BackendNotInstalledError(
                    f"{config.models[kind]} has sha256 {hashes[kind]}, but the configuration pins "
                    f"{expected}: it is not the expected {kind} file"
                )
        self._hashes = hashes

        server = ComfyUIServer(config)
        server.ensure_running()
        try:
            self._check_nodes(server)
            stats = server.client.system_stats().get("system", {})
        except Exception:
            server.close()
            raise
        self._server = server
        torch_version = stats.get("pytorch_version")
        cuda_build = re.search(r"\+(cu\d+|cpu)", torch_version or "")
        self._environment = {
            "comfyui_commit": git_commit(config.comfyui_dir),
            "comfyui_gguf_commit": git_commit(config.comfyui_dir / "custom_nodes" / "ComfyUI-GGUF"),
            "comfyui_version": stats.get("comfyui_version"),
            "comfyui_torch_version": torch_version,
            "comfyui_torch_cuda_build": cuda_build.group(1) if cuda_build else None,
            "comfyui_server_reused": not server.started_by_us,
            "model_sha256": dict(hashes),
            "model_hashing_seconds": round(hash_seconds, 1),
        }

    def _check_nodes(self, server: ComfyUIServer) -> None:
        """Every node class, input and model file the workflow needs must exist
        in the running ComfyUI; otherwise the backend is not installed."""
        config = server.config
        needed = REQUIRED_NODES + ((LATENT_SCALE_NODE,) if self._latent_scale_needed else ())
        info = {name: server.client.node_info(name) for name in needed}
        absent = sorted(name for name, entry in info.items() if entry is None)
        if absent:
            hint = " (is the ComfyUI-GGUF custom node installed?)" if "UnetLoaderGGUF" in absent else ""
            raise BackendNotInstalledError(f"ComfyUI at {server.client.base_url} lacks node classes {absent}{hint}")

        # The workflow's input names must match this ComfyUI version.
        probe = build_workflow(
            models=config.models, image_name="x.png", prompt="x", negative_prompt="", width=1280, height=704,
            num_frames=MAX_FRAMES, seed=0, output_prefix="x",
            settings={**DEFAULTS, "latent_scale": 0.5 if self._latent_scale_needed else 1.0},
        )
        for node in probe.values():
            inputs = info[node["class_type"]].get("input", {})
            required, optional = inputs.get("required", {}), inputs.get("optional", {})
            given = set(node["inputs"])
            missing, unknown = set(required) - given, given - set(required) - set(optional)
            if missing or unknown:
                raise BackendNotInstalledError(
                    f"this ComfyUI's {node['class_type']} node does not match the workflow "
                    f"(missing inputs {sorted(missing)}, unknown inputs {sorted(unknown)})"
                )
        for kind, (node_class, input_name) in MODEL_INPUTS.items():
            listed = _choices(info[node_class]["input"]["required"].get(input_name))
            if listed is not None and config.models[kind] not in listed:
                raise BackendNotInstalledError(
                    f"ComfyUI does not list {config.models[kind]!r} for {node_class}.{input_name}; "
                    "the running server may belong to a different install"
                )

    # --- generation ---

    def generate(self, request: VideoGenerationRequest, output_dir: Path) -> GenerationResult:
        if self._server is None:
            raise RuntimeError("prepare() must be called before generate()")
        server, client, config = self._server, self._server.client, self._server.config
        settings = self._settings(request)
        width, height = request.output_resolution.width, request.output_resolution.height
        num_frames = frame_count_for(request.duration_seconds)
        output_dir = Path(output_dir)

        warnings: list[str] = []
        if request.negative_prompt:
            warnings.append("negative_prompt has no effect: this model runs at CFG 1")
        unused = [
            name for name in ("camera_motion", "subject_motion", "visual_direction")
            if getattr(request, name) is not None
        ]
        if unused:
            warnings.append(
                f"{', '.join(unused)} not consumed separately: only the prompt is passed to the model, unchanged"
            )
        if request.reference_images:
            warnings.append("reference_images are not used by this backend; only the start_frame conditions the video")
        if self._ram_before_gb is not None and self._ram_before_gb < LOW_RAM_WARNING_GB:
            warnings.append(
                f"only {self._ram_before_gb:.1f} GB RAM was available before loading "
                f"(below {LOW_RAM_WARNING_GB:g} GB): the run may have paged and its timings may be slow"
            )
        if not server.started_by_us:
            warnings.append(
                f"reused a ComfyUI that was already running on port {config.port}; the recorded "
                "commits and file hashes are those of the configured install"
            )

        # Sampler / scheduler names are checked against what this ComfyUI offers.
        sampler_inputs = client.node_info("KSampler")["input"]["required"]
        for name, key in (("sampler", "sampler_name"), ("scheduler", "scheduler")):
            listed = _choices(sampler_inputs.get(key))
            if listed is not None and settings[name] not in listed:
                raise ValueError(f"{name} {settings[name]!r} is not available in this ComfyUI: {listed}")

        # Same preprocessing as the diffusers adapter (shared helper).
        _, conditioning = prepare_start_frame(request.start_frame, output_dir, width, height)
        run_key = uuid.uuid4().hex
        output_prefix = f"{OUTPUT_PREFIX}/{run_key}"
        image_name = client.upload_image(Path(conditioning.derived_file), f"ai_studio_{run_key}.png")
        workflow = build_workflow(
            models=config.models, image_name=image_name, prompt=request.prompt,
            negative_prompt=request.negative_prompt or "", width=width, height=height,
            num_frames=num_frames, seed=request.seed, settings=settings, output_prefix=output_prefix,
        )

        try:
            with LowestAvailableRam() as ram:
                started = time.perf_counter()
                outputs = client.wait_for_prompt(client.queue_prompt(workflow), config.generation_timeout_seconds)
                prompt_seconds = time.perf_counter() - started
                images = (outputs.get("12") or {}).get("images") or []
                if not images:
                    raise RuntimeError("ComfyUI finished the prompt but returned no frames")
                if len(images) != num_frames:
                    warnings.append(f"asked ComfyUI for {num_frames} frames, it returned {len(images)}")
                started = time.perf_counter()
                output = output_dir / "out.mp4"
                encoder = encode_frames(_RemoteFrames(client, images), output, NATIVE_FPS)
                encode_seconds = time.perf_counter() - started
        finally:
            self._clean_up(config, image_name, run_key)
            if server.started_by_us and server.log_path is not None and server.log_path.is_file():
                shutil.copyfile(server.log_path, output_dir / "comfyui.log")

        environment = dict(self._environment)
        environment.update({
            "ram_total_gb": _rounded(total_ram_gb()),
            "ram_available_before_gb": _rounded(self._ram_before_gb),
            "ram_available_min_gb": _rounded(ram.lowest_gb),
            "comfyui_prompt_seconds": round(prompt_seconds, 1),
            "fetch_and_encode_seconds": round(encode_seconds, 1),
        })
        ffmpeg_version = _ffmpeg_version()
        if ffmpeg_version:
            environment["ffmpeg_version"] = ffmpeg_version
        return GenerationResult(
            output_video=str(output),
            backend_name=BACKEND_NAME,
            backend_version=BACKEND_VERSION,
            model_name=MODEL_NAME,
            model_version=self._hashes["unet"],
            parameters={
                **settings,
                "cfg": CFG,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "fps": NATIVE_FPS,
                "unet": config.models["unet"],
                "text_encoder": config.models["text_encoder"],
                "vae": config.models["vae"],
                "vae_decode": "VAEDecodeTiled",
                "spatial_tile_overlap": SPATIAL_TILE_OVERLAP,
                "temporal_tile": temporal_tile_for(num_frames),
                "temporal_tile_overlap": TEMPORAL_TILE_OVERLAP,
                "encoder": encoder,
            },
            generation_time_seconds=prompt_seconds,
            environment=environment,
            warnings=warnings,
            fps=NATIVE_FPS,
            frame_count=len(images),
            inference_steps=settings["num_inference_steps"],
            guidance=CFG,
            start_frame_conditioning=conditioning,
        )

    @staticmethod
    def _clean_up(config: ComfyUIConfig, image_name: str, run_key: str) -> None:
        """Remove this run's copies from ComfyUI's own input / output folders
        (best effort; only paths this adapter created, inside comfyui_dir)."""
        frames_dir = config.comfyui_dir / "output" / OUTPUT_PREFIX / run_key
        uploaded = config.comfyui_dir / "input" / image_name
        try:
            if frames_dir.is_dir() and frames_dir.resolve().is_relative_to(config.comfyui_dir.resolve()):
                shutil.rmtree(frames_dir)
                if not any(frames_dir.parent.iterdir()):
                    frames_dir.parent.rmdir()
            if uploaded.is_file() and uploaded.name.startswith("ai_studio_"):
                uploaded.unlink()
        except OSError:
            pass


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
