"""Make keyframes in the local ComfyUI under a style lock.

    outputs/keyframes/<trailer>/<keyframe_id>/<lock>_seed<seed>_<key>.png
                                             /<lock>_seed<seed>_<key>.json

<key> is the first 8 hex digits of the SHA-256 of the request (prompt,
negative, size, seed, lock, reference), so a changed prompt never collides
with an earlier picture of the same seed. An existing file is never
overwritten: asking for it again skips it. Both files are made read-only.

Before anything is queued, every model file of the lock is checked against
its SHA-256 (the checkpoint's hash is cached by size and modification time,
it takes half a minute to compute).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_studio.generation.backends.comfyui import ComfyUIServer, load_config, sha256_file
from ai_studio.keyframes.lock import Prompt, StyleLock
from ai_studio.keyframes.plan import Keyframe, KeyframePlan
from ai_studio.paths import outputs_dir, resolve_path
from ai_studio.protection import make_read_only

SAVE_NODE = "9"
OUTPUT_PREFIX = "ai_studio_keyframes"


@dataclass(frozen=True)
class Request:
    lock_id: str
    keyframe_id: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    reference_image: str | None = None  # project-relative; through the IP-Adapter
    reference_region: tuple[float, float, float, float] | None = None  # where the reference acts; None: everywhere

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[:8]


def request_for(lock: StyleLock, plan: KeyframePlan, keyframe: Keyframe, seed: int) -> Request:
    prompt = keyframe.prompt(lock)
    reference = lock.characters[keyframe.character].reference_image if keyframe.character_reference else None
    return Request(lock.lock_id, keyframe.id, prompt.positive, prompt.negative,
                   plan.resolution.width, plan.resolution.height, seed, reference,
                   keyframe.character_region if reference else None)


def workflow(lock: StyleLock, prompt: Prompt, *, width: int, height: int, seed: int,
             reference_name: str | None = None, mask_name: str | None = None) -> dict[str, Any]:
    """SDXL text-to-image in ComfyUI's API format: checkpoint, the lock's LoRAs
    (model only), optionally the IP-Adapter (limited to a mask's white area
    when mask_name is given), KSampler, decode, save."""
    nodes: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": lock.checkpoint.file}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt.positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt.negative}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model = ["1", 0]
    for i, lora in enumerate(lock.loras):
        node = f"{20 + i}"
        nodes[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": lora.file, "strength_model": lora.strength,
        }}
        model = [node, 0]
    if reference_name is not None:
        ipa = lock.ipadapter
        nodes.update({
            "30": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
            "31": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipa.file}},
            "32": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": ipa.clip_vision}},
            "33": {"class_type": "IPAdapterAdvanced", "inputs": {
                "model": model, "ipadapter": ["31", 0], "image": ["30", 0], "clip_vision": ["32", 0],
                "weight": ipa.weight, "weight_type": ipa.weight_type, "combine_embeds": "concat",
                "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
            }},
        })
        if mask_name is not None:
            nodes["34"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
            nodes["33"]["inputs"]["attn_mask"] = ["34", 0]
        model = ["33", 0]
    nodes["7"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        "seed": seed, "steps": lock.steps, "cfg": lock.cfg, "sampler_name": lock.sampler,
        "scheduler": lock.scheduler, "denoise": 1.0,
    }}
    nodes["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    nodes[SAVE_NODE] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": OUTPUT_PREFIX}}
    return nodes


FEATHER = 0.04  # of the frame's shorter side


def region_mask(width: int, height: int, region: tuple[float, float, float, float]):
    """A white rectangle (the region) on black, its edges blurred."""
    from PIL import Image, ImageDraw, ImageFilter

    x0, y0, x1, y1 = region
    mask = Image.new("RGB", (width, height), (0, 0, 0))
    ImageDraw.Draw(mask).rectangle((x0 * width, y0 * height, x1 * width, y1 * height), fill=(255, 255, 255))
    return mask.filter(ImageFilter.GaussianBlur(FEATHER * min(width, height)))


def output_paths(root: Path, trailer_id: str, request: Request) -> tuple[Path, Path]:
    stem = f"{request.lock_id}_seed{request.seed}_{request.key}"
    folder = outputs_dir(root) / "keyframes" / trailer_id / request.keyframe_id
    return folder / f"{stem}.png", folder / f"{stem}.json"


class ModelCheck:
    """Verifies model files against the lock, caching hashes by size and mtime."""

    def __init__(self, cache_file: Path) -> None:
        self.cache_file = cache_file
        try:
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

    def sha256(self, path: Path) -> str:
        stat = path.stat()
        entry = self.cache.get(str(path))
        if entry and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
            return entry["sha256"]
        digest = sha256_file(path)
        self.cache[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
        return digest

    def verify(self, path: Path, expected: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"model file missing: {path}")
        actual = self.sha256(path)
        if actual != expected:
            raise ValueError(f"{path} has SHA-256 {actual}, the lock expects {expected}")


def verify_lock_files(lock: StyleLock, comfyui_dir: Path, check: ModelCheck, *, ipadapter: bool) -> dict[str, str]:
    models = comfyui_dir / "models"
    checked = {lock.checkpoint.file: lock.checkpoint.sha256}
    check.verify(models / "checkpoints" / lock.checkpoint.file, lock.checkpoint.sha256)
    for lora in lock.loras:
        check.verify(models / "loras" / lora.file, lora.sha256)
        checked[lora.file] = lora.sha256
    if ipadapter:
        check.verify(models / "ipadapter" / lock.ipadapter.file, lock.ipadapter.sha256)
        check.verify(models / "clip_vision" / lock.ipadapter.clip_vision, lock.ipadapter.clip_vision_sha256)
        checked[lock.ipadapter.file] = lock.ipadapter.sha256
        checked[lock.ipadapter.clip_vision] = lock.ipadapter.clip_vision_sha256
    return checked


class KeyframeMaker:
    """One ComfyUI session for many keyframes. Use as a context manager."""

    def __init__(self, lock: StyleLock, root: Path, *, log=print) -> None:
        self.lock, self.root, self.log = lock, root, log
        self.server = ComfyUIServer(load_config())
        self.check = ModelCheck(outputs_dir(root) / "keyframes" / ".model_hashes.json")
        self._verified: dict[bool, dict[str, str]] = {}
        self._uploaded: dict[str, str] = {}
        self._masks: dict[tuple, str] = {}

    def __enter__(self) -> KeyframeMaker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.close()

    def _models(self, ipadapter: bool) -> dict[str, str]:
        if ipadapter not in self._verified:
            self._verified[ipadapter] = verify_lock_files(
                self.lock, self.server.config.comfyui_dir, self.check, ipadapter=ipadapter)
        return self._verified[ipadapter]

    def _reference(self, path: str, expected: str) -> str:
        if path not in self._uploaded:
            source = resolve_path(path, self.root)
            if sha256_file(source) != expected:
                raise ValueError(f"reference image {path} does not match its SHA-256 in the lock")
            self._uploaded[path] = self.server.client.upload_image(source, f"ai_studio_ref_{expected[:12]}.png")
        return self._uploaded[path]

    def _mask(self, width: int, height: int, region: tuple[float, float, float, float]) -> str:
        """Upload a white-on-black mask of the region, edges feathered."""
        key = (width, height, region)
        if key not in self._masks:
            mask = region_mask(width, height, region)
            path = outputs_dir(self.root) / "keyframes" / ".masks" / f"mask_{width}x{height}_{hashlib.sha256(repr(key).encode()).hexdigest()[:12]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(path)
            self._masks[key] = self.server.client.upload_image(path, f"ai_studio_{path.name}")
        return self._masks[key]

    def render(self, request: Request) -> tuple[bytes, dict[str, str]]:
        """Generate one picture: ComfyUI's PNG bytes and the verified model
        hashes. Writes nothing."""
        models = self._models(request.reference_image is not None)
        self.server.ensure_running()
        reference_name = None
        if request.reference_image is not None:
            char = next(c for c in self.lock.characters.values() if c.reference_image == request.reference_image)
            reference_name = self._reference(request.reference_image, char.reference_sha256)
        mask_name = None
        if request.reference_region is not None:
            mask_name = self._mask(request.width, request.height, request.reference_region)
        graph = workflow(self.lock, Prompt(request.positive, request.negative), width=request.width,
                         height=request.height, seed=request.seed, reference_name=reference_name, mask_name=mask_name)
        client = self.server.client
        outputs = client.wait_for_prompt(client.queue_prompt(graph), self.server.config.generation_timeout_seconds)
        return client.view(outputs[SAVE_NODE]["images"][0]), models

    def make(self, trailer_id: str, request: Request) -> tuple[Path, bool]:
        """(path, made): made is False when the picture already existed."""
        png, sidecar = output_paths(self.root, trailer_id, request)
        if png.exists():
            return png, False
        started = time.perf_counter()
        data, models = self.render(request)
        seconds = round(time.perf_counter() - started, 1)
        png.parent.mkdir(parents=True, exist_ok=True)
        with png.open("xb") as f:
            f.write(data)
        record = {
            "keyframe_version": 1, "trailer_id": trailer_id, "request": asdict(request), "key": request.key,
            "lock": {"lock_id": self.lock.lock_id, "steps": self.lock.steps, "cfg": self.lock.cfg,
                     "sampler": self.lock.sampler, "scheduler": self.lock.scheduler,
                     "loras": [asdict(lora) for lora in self.lock.loras],
                     "ipadapter": asdict(self.lock.ipadapter) if request.reference_image and self.lock.ipadapter else None},
            "model_sha256": models, "png_sha256": hashlib.sha256(data).hexdigest(), "seconds": seconds,
            "made_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with sidecar.open("x", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        for path in (png, sidecar):
            make_read_only(path)
        return png, True
