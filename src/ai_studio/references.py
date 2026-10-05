"""Import a hand-made image as a reference asset, with its provenance.

Some pictures are not made by a backend of this project: a person makes them
in another tool (for example an image app) and brings the file here. Such a
file enters the project through import_reference():

    benchmarks/references/<asset_set>/<name>                  the image, byte-identical
    benchmarks/references/<asset_set>/<name>.provenance.json  where it came from

The provenance file records what the PERSON states (tool, model, date, the
exact prompt, the reference images that were given to the tool) and what THIS
code measures (SHA-256, size in bytes, pixel dimensions, format, and a simple
check for a visible mark in the corners). Stated facts are not verified and
are labelled as stated.

Rules:
  - the source file is only read, never changed or moved
  - the copy is byte-identical (verified by hash); nothing is converted,
    cropped or cleaned, whatever the corner check says
  - nothing is overwritten: an existing name, or an image that is already in
    the asset set under another name, is refused
  - the copy and its provenance file are marked read-only

The corner check is a heuristic. It flags a small, compact, light or dark
shape that stands apart from its surroundings near a corner, which is what a
visible logo or "sparkle" mark looks like. It can miss a mark on a busy
background and it can flag picture content. It only reports; a person
decides. It cannot see invisible watermarks.

Needs Pillow (to read the pixels).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ai_studio.benchmarking.schema import check_file_name_safe
from ai_studio.paths import resolve_path, to_project_relative
from ai_studio.protection import make_read_only

REFERENCES_DIR = "benchmarks/references"
PROVENANCE_SUFFIX = ".provenance.json"
PROVENANCE_VERSION = 1
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
WATERMARK_CHECK_VERSION = "1"
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")

# Corner check: see find_corner_marks.
CORNER_FRACTION = 0.14  # side of the examined corner square, as a share of the shorter image side
MIN_CORNER_SIDE = 32
CONTRAST = 40.0  # luma distance from the corner's median that makes a pixel "stand apart"
MAX_CHROMA = 48  # a mark is near-neutral: max(R,G,B) - min(R,G,B) at most this
MIN_AREA, MAX_AREA = 0.003, 0.12  # share of the corner square covered by the shape
MAX_EXTENT = 0.6  # the shape's bounding box, as a share of the corner side
MIN_FILL = 0.18  # share of the bounding box covered (a four-point star: about 0.3)


class ReferenceImportError(RuntimeError):
    """The image was not imported; the message says why. Nothing was written."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ReferenceImportError("Pillow is needed to read the image: pip install pillow") from exc
    return Image


def read_image(path: str | Path) -> tuple[Any, str]:
    """(RGB pixels as a (H, W, 3) uint8 array, format name such as "PNG")."""
    import numpy as np

    image_module = _pillow()
    try:
        with image_module.open(path) as image:
            image_format = image.format or "unknown"
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # Pillow raises many types for a file that is not an image
        raise ReferenceImportError(f"{path} is not a readable image: {type(exc).__name__}: {exc}") from None
    return pixels, image_format


def _corner_mark(patch: Any, inner_rows: int, inner_cols: int) -> dict[str, Any] | None:
    """A likely mark in one corner square, or None. inner_rows / inner_cols
    are the patch edges (0 or -1) that face the inside of the picture."""
    import numpy as np

    side = patch.shape[0]
    luma = patch.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    chroma = patch.max(axis=2).astype(np.int16) - patch.min(axis=2)
    median = float(np.median(luma))
    best = None
    for tone, apart in (("light", luma - median > CONTRAST), ("dark", median - luma > CONTRAST)):
        mask = apart & (chroma <= MAX_CHROMA)
        count = int(mask.sum())
        if not MIN_AREA <= count / mask.size <= MAX_AREA:
            continue
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        top, bottom, left, right = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
        height, width = bottom - top + 1, right - left + 1
        if max(height, width) > MAX_EXTENT * side:
            continue  # spread out: picture content, not one small shape
        if mask[inner_rows].any() or mask[:, inner_cols].any():
            continue  # runs on into the picture
        fill = count / (height * width)
        if fill < MIN_FILL:
            continue
        if best is None or count > best["pixels"]:
            best = {"tone": tone, "pixels": count, "box_in_corner": [left, top, right + 1, bottom + 1],
                    "fill": round(fill, 3)}
    return best


def find_corner_marks(pixels: Any) -> dict[str, Any]:
    """Look for a likely visible logo / sparkle mark in each corner.

    Returns {"method", "version", "corner_side", "corners": {corner: None or
    {"tone", "box" (image pixels, [x0, y0, x1, y1]), "pixels", "fill"}},
    "flagged": [corners with a likely mark]}. Never changes the picture."""
    height, width = pixels.shape[:2]
    side = max(MIN_CORNER_SIDE, round(CORNER_FRACTION * min(width, height)))
    side = min(side, width, height)
    corners: dict[str, Any] = {}
    for corner in CORNERS:
        at_bottom, at_right = corner.startswith("bottom"), corner.endswith("right")
        y0, x0 = (height - side if at_bottom else 0), (width - side if at_right else 0)
        mark = _corner_mark(
            pixels[y0: y0 + side, x0: x0 + side],
            inner_rows=0 if at_bottom else -1, inner_cols=0 if at_right else -1,
        )
        if mark is not None:
            left, top, right, bottom = mark.pop("box_in_corner")
            mark["box"] = [x0 + left, y0 + top, x0 + right, y0 + bottom]
        corners[corner] = mark
    return {
        "method": "compact near-neutral shape that stands apart from its corner (heuristic, report only)",
        "version": WATERMARK_CHECK_VERSION,
        "corner_side": side,
        "corners": corners,
        "flagged": [corner for corner in CORNERS if corners[corner] is not None],
    }


def asset_set_dir(root: str | Path, asset_set: str) -> Path:
    check_file_name_safe("asset_set", asset_set)
    return resolve_path(f"{REFERENCES_DIR}/{asset_set}", root)


def provenance_path(image: Path) -> Path:
    return image.with_name(image.name + PROVENANCE_SUFFIX)


def _stated_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ReferenceImportError(f"date must look like 2026-10-06, got {value!r}") from None


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceImportError(f"{name} must be given (what you state is recorded as stated)")
    return value.strip()


def import_reference(
    source: str | Path,
    asset_set: str,
    *,
    tool: str,
    model: str,
    prompt: str,
    root: str | Path,
    name: str | None = None,
    prompt_source: str = "argument",
    made_on: str | None = None,
    input_references: list[str | Path] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Copy `source` into benchmarks/references/<asset_set>/ and write its
    provenance file. Returns the provenance record. ReferenceImportError if
    anything is wrong; then nothing was written.

    tool, model   what made the image, AS STATED by the person importing it
    prompt        the exact prompt text, kept character for character
    name          file name in the asset set (default: the source's name);
                  the extension must stay the same, nothing is converted
    made_on       the day the image was made, as stated (YYYY-MM-DD)
    input_references  the reference images that were given to the tool"""
    root = Path(root).resolve()
    source = Path(source)
    if not source.is_file():
        raise ReferenceImportError(f"no such file: {source}")
    tool, model = _text("tool", tool), _text("model", model)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ReferenceImportError("the prompt must not be empty")
    made_on = _stated_date(made_on)
    name = name or source.name
    try:
        check_file_name_safe("name", name)
        folder = asset_set_dir(root, asset_set)
    except ValueError as exc:
        raise ReferenceImportError(str(exc)) from None
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ReferenceImportError(f"{name}: the file name must end in one of {', '.join(IMAGE_SUFFIXES)}")
    if suffix != source.suffix.lower():
        raise ReferenceImportError(
            f"{name}: the extension must stay {source.suffix!r}; the file is copied, never converted"
        )
    if name.endswith(PROVENANCE_SUFFIX):
        raise ReferenceImportError(f"{name} is a provenance file name")

    target = folder / name
    sidecar = provenance_path(target)
    for path in (target, sidecar):
        if path.exists():
            raise ReferenceImportError(
                f"{to_project_relative(path, root)} already exists; a reference is never overwritten "
                "(choose another name)"
            )
    digest = sha256_file(source)
    if folder.is_dir():
        for other in sorted(folder.iterdir()):
            if other.is_file() and other.suffix.lower() in IMAGE_SUFFIXES and sha256_file(other) == digest:
                raise ReferenceImportError(
                    f"this image is already in the asset set as {to_project_relative(other, root)}"
                )
    inputs = []
    for item in input_references or []:
        path = resolve_path(item, root)
        if not path.is_file():
            raise ReferenceImportError(f"input reference not found: {item}")
        inputs.append({"path": to_project_relative(path, root), "sha256": sha256_file(path)})

    pixels, image_format = read_image(source)
    height, width = pixels.shape[:2]
    record = {
        "provenance_version": PROVENANCE_VERSION,
        "asset_set": asset_set,
        "file": to_project_relative(target, root),
        "sha256": digest,
        "bytes": source.stat().st_size,
        "width": int(width),
        "height": int(height),
        "format": image_format,
        "original_filename": source.name,
        "origin": "made by hand in an external tool and imported; not produced by a backend of this project",
        "stated_by_user": {  # recorded as stated, not verified
            "tool": tool,
            "model": model,
            "made_on": made_on,
            "input_references": inputs,
            "notes": notes,
        },
        "prompt": prompt,  # exact text
        "prompt_source": prompt_source,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "source_file_modified": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "visible_watermark_check": find_corner_marks(pixels),
    }

    folder.mkdir(parents=True, exist_ok=True)
    try:
        with open(source, "rb") as reader, open(target, "xb") as writer:  # exclusive create
            shutil.copyfileobj(reader, writer)
    except FileExistsError:
        raise ReferenceImportError(f"{to_project_relative(target, root)} appeared during the import") from None
    if sha256_file(target) != digest:
        raise ReferenceImportError(f"the copy {to_project_relative(target, root)} does not match the source")
    with sidecar.open("x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    for path in (target, sidecar):
        try:
            make_read_only(path)
        except OSError:
            pass
    return record


def load_provenance(image: str | Path) -> dict[str, Any]:
    """The provenance record of an imported reference image."""
    return json.loads(provenance_path(Path(image)).read_text(encoding="utf-8"))


def verify_reference(image: str | Path) -> bool:
    """True if the image still has the hash its provenance file recorded."""
    return sha256_file(image) == load_provenance(image)["sha256"]
