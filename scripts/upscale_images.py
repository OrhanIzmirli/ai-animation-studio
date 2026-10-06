"""Upscale images with an ESRGAN-type model in the local ComfyUI, then resize
to an exact size (Lanczos).

    python scripts/upscale_images.py --width 1920 --height 1056 --out DIR IMAGE [IMAGE ...]

Model: <ComfyUI>/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth (made for
line art and flat colour). Each result is DIR/<name>_<W>x<H>.png; an existing
file is skipped, never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_studio.generation.backends.comfyui import ComfyUIServer, load_config  # noqa: E402

MODEL = "RealESRGAN_x4plus_anime_6B.pth"


def workflow(image_name: str, width: int, height: int) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": MODEL}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "ImageScale", "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                                     "width": width, "height": height, "crop": "disabled"}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "ai_studio_upscale"}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    server = ComfyUIServer(load_config())
    try:
        server.ensure_running()
        client = server.client
        for path in map(Path, args.images):
            target = out / f"{path.stem}_{args.width}x{args.height}.png"
            if target.exists():
                print(f"exists, skipped: {target}")
                continue
            name = client.upload_image(path, f"ai_studio_up_{path.stem}.png")
            outputs = client.wait_for_prompt(client.queue_prompt(workflow(name, args.width, args.height)), 900)
            target.write_bytes(client.view(outputs["5"]["images"][0]))
            print(f"made: {target}")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
