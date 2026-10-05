"""Fake image backend: no model, no network, no dependencies.

It exists to exercise the image contracts end to end (requests, the runner,
records, evaluations) before any real image backend is chosen. It writes PNG
files whose pixels are a deterministic pattern derived from the request: the
same request and the same input files give byte-identical images, a different
seed, prompt or reference gives different ones. The pictures mean nothing.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

from ai_studio.generation.backends.base import BackendMetadata, ImageEditingBackend, ImageGenerationBackend
from ai_studio.generation.image import ImageEditingRequest, ImageGenerationRequest, ImageGenerationResult

BACKEND_NAME = "fake-image"
BACKEND_VERSION = "1.0"
MODEL_NAME = "fake-image-pattern"
MODEL_VERSION = "1"
MAX_SIDE = 4096
BANDS = 8


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png_bytes(width: int, height: int, digest: bytes) -> bytes:
    """An RGB PNG of horizontal colour bands taken from `digest` (>= 24 bytes)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = [b"\x00" + digest[3 * band: 3 * band + 3] * width for band in range(BANDS)]
    raw = b"".join(rows[min(BANDS - 1, y * BANDS // height)] for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


class FakeImageBackend(ImageGenerationBackend, ImageEditingBackend):
    """Implements both image contracts; dispatches on the request type."""

    max_reference_images = 8
    supports_mask = True

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_name=BACKEND_NAME, backend_version=BACKEND_VERSION, model_name=MODEL_NAME,
            model_version=MODEL_VERSION, workflow_version="none",
        )

    def validate_inputs(self, request: ImageGenerationRequest | ImageEditingRequest) -> None:
        size = request.output_resolution
        if max(size.width, size.height) > MAX_SIDE:
            raise ValueError(f"{BACKEND_NAME} writes images up to {MAX_SIDE} px per side")
        if request.backend_parameters:
            raise ValueError(f"{BACKEND_NAME} has no parameters, got {sorted(request.backend_parameters)}")

    def prepare(self) -> None:
        pass

    def generate(
        self, request: ImageGenerationRequest | ImageEditingRequest, output_dir: Path,
    ) -> ImageGenerationResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs = {
            "source_image": _sha256(request.source_image) if isinstance(request, ImageEditingRequest) else None,
            "mask": _sha256(request.mask) if getattr(request, "mask", None) else None,
            "reference_images": [_sha256(path) for path in request.reference_images],
        }
        size = request.output_resolution
        images, details = [], []
        for index in range(request.image_count):
            basis = json.dumps(
                [request.capability.value, request.prompt, request.negative_prompt, request.seed, index, inputs,
                 request.composition, request.subject_details, request.visual_direction],
                sort_keys=True,
            )
            path = output_dir / f"image_{index + 1:02d}.png"
            path.write_bytes(png_bytes(size.width, size.height, hashlib.sha256(basis.encode()).digest()))
            images.append(str(path))
            details.append({"file": path.name, "index": index})
        return ImageGenerationResult(
            output_images=images,
            backend_name=BACKEND_NAME, backend_version=BACKEND_VERSION,
            model_name=MODEL_NAME, model_version=MODEL_VERSION,
            parameters={},
            seed_used=request.seed,
            provenance={
                "generator": "deterministic colour bands from a hash of the request; no model",
                "task": request.capability.value,
                "input_sha256": inputs,
                "images": details,
            },
            warnings=["fake backend: the images are test patterns, not pictures"],
        )
