"""Spike: SDXL base + IP-Adapter in the local ComfyUI, one test image.

Feasibility only (docs/research/LOCAL_IMAGE_BACKEND.md, part 2): does the
character from the approved turnaround sheet stay recognisable in a new pose,
and does the stack fit this laptop? Nothing is written under outputs/ or
benchmarks/; the reference is only read and a cleaned, cropped copy goes to
OUT (default spikes/output/sdxl_ipadapter/, which is git-ignored).

    python spikes/sdxl_ipadapter_spike.py [--weights 0.4 0.6 0.8] [--out FOLDER] [--copy-to FOLDER]

Starts ComfyUI (or reuses a running one), checks that the IPAdapter nodes
loaded, runs one prompt per IP-Adapter weight (same seed, prompt and
reference), and reports time and peak VRAM / RAM use per image.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_studio.generation.backends.comfyui import ComfyUIServer, load_config  # noqa: E402

REFERENCE = ROOT / "benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg"
# Front figure, head to waist, square (the CLIP image encoder sees a 224x224
# square): face, hair, jacket and backpack straps of the 1024x765 sheet. The
# box starts below the "PAGE 23" note; the handwritten note right of the hair
# (TEXT_BOX, sheet coordinates) is painted over with the background grey, so
# the reference carries no text.
CROP_BOX = (40, 46, 340, 346)
TEXT_BOX = (238, 46, 340, 118)
BACKGROUND_SAMPLE = (320, 140)  # a plain background pixel of the sheet
# What the simple IPAdapter node calls "prompt is more important".
WEIGHT_TYPE = "ease out"
CHECKPOINT = "sd_xl_base_1.0.safetensors"
IPADAPTER = "ip-adapter-plus_sdxl_vit-h.safetensors"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
WIDTH, HEIGHT, STEPS, CFG, SEED = 1024, 1024, 25, 6.0, 42
POSITIVE = (
    "graphic novel illustration, a young man with short curly dark hair sitting on a park bench, "
    "three-quarter view, daylight, black hooded jacket with dark-red accents, black backpack with dark-red "
    "straps, dark cargo trousers, white low-top sneakers, calm expression, hand-inked contours, cel shading, "
    "halftone shading"
)
NEGATIVE = "photorealistic, 3d render, blurry, deformed hands, extra fingers, text, watermark, signature"


def clean_reference() -> "Image.Image":
    """The text-free crop of the front figure; the sheet itself is only read."""
    from PIL import Image, ImageDraw

    with Image.open(REFERENCE) as sheet:
        sheet = sheet.convert("RGB")
        ImageDraw.Draw(sheet).rectangle(TEXT_BOX, fill=sheet.getpixel(BACKGROUND_SAMPLE))
        return sheet.crop(CROP_BOX)


def workflow(image_name: str, weight: float) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POSITIVE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEGATIVE}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IPADAPTER}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "8": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["6", 0], "image": ["5", 0], "clip_vision": ["7", 0],
            "weight": weight, "weight_type": WEIGHT_TYPE, "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only",
        }},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["8", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": SEED, "steps": STEPS, "cfg": CFG, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": 1.0,
        }},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "ai_studio_sdxl_ipa_spike"}},
    }


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual",
        )
    ]


def available_ram_gb() -> float:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1024**3


def vram_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    return int(out.strip().splitlines()[0])


class Monitor(threading.Thread):
    """Samples VRAM in use and available RAM twice a second."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.vram_peak, self.ram_low = 0, float("inf")

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.vram_peak = max(self.vram_peak, vram_used_mib())
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.ram_low = min(self.ram_low, available_ram_gb())
            self.stop.wait(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "spikes/output/sdxl_ipadapter"),
                        help="folder for the cropped reference, the results and the report")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.4, 0.6, 0.8], help="IP-Adapter weights, one image each")
    parser.add_argument("--copy-to", default=None, help="also copy the results here (e.g. the Desktop)")
    parser.add_argument("--crop-only", action="store_true", help="write the cleaned reference crop and stop")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    crop_path = out / "reference_crop_clean.png"
    clean_reference().save(crop_path)
    if args.crop_only:
        print(crop_path)
        return 0

    report: dict = {
        "checkpoint": CHECKPOINT, "ipadapter": IPADAPTER, "clip_vision": CLIP_VISION,
        "weight_type": WEIGHT_TYPE, "images": [],
        "width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfg": CFG, "seed": SEED,
        "positive": POSITIVE, "negative": NEGATIVE, "reference": str(REFERENCE.relative_to(ROOT)),
        "reference_crop_box": CROP_BOX, "reference_text_painted_over": TEXT_BOX,
        "ram_available_before_gb": round(available_ram_gb(), 2), "vram_used_before_mib": vram_used_mib(),
    }
    server = ComfyUIServer(load_config())
    monitor = Monitor()
    try:
        started = time.perf_counter()
        server.ensure_running()
        report["comfyui_started_by_spike"] = server.started_by_us
        report["comfyui_startup_seconds"] = round(time.perf_counter() - started, 1)
        client = server.client
        missing = [n for n in ("IPAdapterAdvanced", "IPAdapterModelLoader", "CLIPVisionLoader")
                   if client.node_info(n) is None]
        log = server.log_path.read_text(encoding="utf-8", errors="replace") if server.log_path else ""
        report["custom_node_import_failed"] = "IMPORT FAILED" in log and "IPAdapter" in log.split("IMPORT FAILED", 1)[1][:400]
        if missing or report["custom_node_import_failed"]:
            report["error"] = f"IPAdapter nodes not available: missing {missing}"
            print(server.log_tail(40))
            return 1

        image_name = client.upload_image(crop_path, "ai_studio_sdxl_ipa_spike_reference.png")
        monitor.start()
        for weight in args.weights:
            started = time.perf_counter()
            prompt_id = client.queue_prompt(workflow(image_name, weight))
            outputs = client.wait_for_prompt(prompt_id, timeout_seconds=1800)
            seconds = round(time.perf_counter() - started, 1)  # the first image includes loading the models
            images = [image for node in outputs.values() for image in node.get("images", [])]
            result = out / f"sdxl_ipadapter_test_w{round(weight * 10):02d}.png"
            result.write_bytes(client.view(images[0]))
            report["images"].append({
                "ipadapter_weight": weight, "seconds": seconds, "result": str(result),
                "vram_peak_so_far_mib": monitor.vram_peak, "ram_available_lowest_so_far_gb": round(monitor.ram_low, 2),
            })
        return 0
    except Exception as exc:  # a spike: report whatever went wrong, change nothing
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(server.log_tail(40))
        return 1
    finally:
        monitor.stop.set()
        report["vram_peak_mib"] = monitor.vram_peak
        report["ram_available_lowest_gb"] = None if monitor.ram_low == float("inf") else round(monitor.ram_low, 2)
        if server.log_path is not None:
            report["comfyui_log"] = str(server.log_path)
        server.close()
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.copy_to and not args.crop_only:
            import shutil

            target = Path(args.copy_to)
            target.mkdir(parents=True, exist_ok=True)
            for name in ["reference_crop_clean.png", "report.json", *(Path(i["result"]).name for i in report["images"])]:
                shutil.copy2(out / name, target / name)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
