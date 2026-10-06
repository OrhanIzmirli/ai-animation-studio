"""Depth maps for painted plates with Depth-Anything-V2-Small. Run with the
Python of the Kokoro environment (it has torch and transformers):

    kokoro-venv/Scripts/python.exe -I scripts/depth_maps.py OUT_DIR IMAGE [IMAGE ...]

Writes OUT_DIR/<name>_depth.png: 16-bit greyscale at the image's size,
0 = farthest, 65535 = nearest (the model's relative depth, rescaled per image).
An existing file is skipped.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import pipeline


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    estimate = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu")
    for path in map(Path, sys.argv[2:]):
        target = out / f"{path.stem}_depth.png"
        if target.exists():
            print(f"exists, skipped: {target}")
            continue
        image = Image.open(path).convert("RGB")
        depth = np.asarray(estimate(image)["predicted_depth"].squeeze(), dtype=np.float32)
        depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
        img = Image.fromarray((depth * 65535).astype(np.uint16)).resize(image.size, Image.BICUBIC)
        img.save(target)
        print(f"made: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
