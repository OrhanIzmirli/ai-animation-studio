"""Spike: SDXL base + IP-Adapter in the local ComfyUI, one test image.

Feasibility only (docs/research/LOCAL_IMAGE_BACKEND.md, part 2): does the
character from the approved turnaround sheet stay recognisable in a new pose,
and does the stack fit this laptop? Nothing is written under outputs/ or
benchmarks/; the reference is only read and a cleaned, cropped copy goes to
OUT (default spikes/output/sdxl_ipadapter/, which is git-ignored).

    python spikes/sdxl_ipadapter_spike.py [--weights 0.4 0.6 0.8] [--out FOLDER] [--copy-to FOLDER]
    python spikes/sdxl_ipadapter_spike.py --weights 0.8 --tag lineart_a
        --control-image spikes/output/sdxl_ipadapter/sdxl_ipadapter_test_w08.png
        [--control-strength 0.8] [--control-end 1.0] [--positive-add TEXT] [--negative-add TEXT]

Starts ComfyUI (or reuses a running one), checks that the IPAdapter nodes
loaded, runs one prompt per IP-Adapter weight (same seed, prompt and
reference), and reports time and peak VRAM / RAM use per image.

--init-image with --denoise runs image-to-image from that picture instead of
from noise (low denoise keeps its composition).

--lora applies a LoRA from <ComfyUI>/models/loras to the model (for the
character LoRA of spikes/train_hero_lora_spike.py); a weight of 0 leaves the
IP-Adapter out, so the LoRA can be judged on its own.

--control-image adds the Lineart T2I-Adapter: the lines of that image hold
the composition of the new one. The lines are extracted with ComfyUI's
built-in Canny node (no preprocessor model is installed) and saved beside
the result as <name>_lines.png.
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
T2I_LINEART = "t2i-adapter-lineart-sdxl-1.0.fp16.safetensors"
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


def workflow(
    image_name: str, weight: float, positive: str = POSITIVE, negative: str = NEGATIVE, seed: int = SEED,
    control: dict | None = None, init_image: str | None = None, denoise: float = 1.0,
    lora: str | None = None, lora_strength: float = 1.0,
) -> dict:
    """control: {"image": uploaded name, "strength", "end", "canny": (low, high)} or None.
    init_image: uploaded name of the picture to start from (image-to-image), with `denoise` < 1."""
    nodes = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative}},
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
            "seed": seed, "steps": STEPS, "cfg": CFG, "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": 1.0,
        }},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "ai_studio_sdxl_ipa_spike"}},
    }
    if lora is not None:
        nodes["40"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": lora, "strength_model": lora_strength,
        }}
        nodes["8"]["inputs"]["model"] = ["40", 0]
    if weight == 0:  # no IP-Adapter at all
        for key in ("5", "6", "7", "8"):
            del nodes[key]
        nodes["9"]["inputs"]["model"] = ["40", 0] if lora is not None else ["1", 0]
    if init_image is not None:
        nodes["30"] = {"class_type": "LoadImage", "inputs": {"image": init_image}}
        nodes["31"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["30", 0], "vae": ["1", 2]}}
        nodes["9"]["inputs"]["latent_image"] = ["31", 0]
        nodes["9"]["inputs"]["denoise"] = denoise
    if control is not None:
        low, high = control["canny"]
        nodes.update({
            "20": {"class_type": "LoadImage", "inputs": {"image": control["image"]}},
            "21": {"class_type": "Canny", "inputs": {"image": ["20", 0], "low_threshold": low, "high_threshold": high}},
            "22": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": T2I_LINEART}},
            "23": {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": ["2", 0], "negative": ["3", 0], "control_net": ["22", 0], "image": ["21", 0],
                "strength": control["strength"], "start_percent": 0.0, "end_percent": control["end"],
            }},
            "24": {"class_type": "SaveImage", "inputs": {"images": ["21", 0], "filename_prefix": "ai_studio_sdxl_ipa_spike_lines"}},
        })
        nodes["9"]["inputs"]["positive"], nodes["9"]["inputs"]["negative"] = ["23", 0], ["23", 1]
    return nodes


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
    parser.add_argument("--tag", default=None, help="name of this try: result sdxl_ipadapter_<tag>.png, report_<tag>.json (one weight)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--positive-add", default="", help="text appended to the positive prompt")
    parser.add_argument("--negative-add", default="", help="text appended to the negative prompt")
    parser.add_argument("--control-image", default=None, help="image whose lines hold the composition (Lineart T2I-Adapter)")
    parser.add_argument("--control-strength", type=float, default=0.8)
    parser.add_argument("--control-end", type=float, default=1.0, help="fraction of the steps the adapter acts on")
    parser.add_argument("--canny", type=float, nargs=2, default=[0.4, 0.8], metavar=("LOW", "HIGH"))
    parser.add_argument("--lora", default=None, help="LoRA file name in <ComfyUI>/models/loras")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--positive", default=None, help="replace the whole positive prompt")
    parser.add_argument("--init-image", default=None, help="start from this picture (image-to-image)")
    parser.add_argument("--denoise", type=float, default=1.0, help="with --init-image: how much of it is redrawn (0-1)")
    args = parser.parse_args()
    if args.tag and len(args.weights) != 1:
        parser.error("--tag needs exactly one weight")
    positive = (args.positive or POSITIVE) + (", " + args.positive_add if args.positive_add else "")
    negative = NEGATIVE + (", " + args.negative_add if args.negative_add else "")
    report_name = f"report_{args.tag}.json" if args.tag else "report.json"
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
        "width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfg": CFG, "seed": args.seed,
        "positive": positive, "negative": negative, "reference": str(REFERENCE.relative_to(ROOT)),
        "init_image": args.init_image, "denoise": args.denoise if args.init_image else 1.0,
        "lora": args.lora, "lora_strength": args.lora_strength if args.lora else None,
        "control": None if args.control_image is None else {
            "adapter": T2I_LINEART, "lines_from": args.control_image, "line_extractor": "ComfyUI Canny node",
            "canny": args.canny, "strength": args.control_strength, "end": args.control_end,
        },
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
        missing = [n for n in ("IPAdapterAdvanced", "IPAdapterModelLoader", "CLIPVisionLoader", "Canny",
                               "ControlNetApplyAdvanced") if client.node_info(n) is None]
        log = server.log_path.read_text(encoding="utf-8", errors="replace") if server.log_path else ""
        report["custom_node_import_failed"] = "IMPORT FAILED" in log and "IPAdapter" in log.split("IMPORT FAILED", 1)[1][:400]
        if missing or report["custom_node_import_failed"]:
            report["error"] = f"IPAdapter nodes not available: missing {missing}"
            print(server.log_tail(40))
            return 1

        image_name = client.upload_image(crop_path, "ai_studio_sdxl_ipa_spike_reference.png")
        control = None
        if args.control_image is not None:
            control = {
                "image": client.upload_image(Path(args.control_image), "ai_studio_sdxl_ipa_spike_control.png"),
                "strength": args.control_strength, "end": args.control_end, "canny": tuple(args.canny),
            }
        init_name = None
        if args.init_image is not None:
            init_name = client.upload_image(Path(args.init_image), "ai_studio_sdxl_ipa_spike_init.png")
        monitor.start()
        for weight in args.weights:
            started = time.perf_counter()
            prompt_id = client.queue_prompt(workflow(
                image_name, weight, positive, negative, args.seed, control, init_name, args.denoise,
                args.lora, args.lora_strength,
            ))
            outputs = client.wait_for_prompt(prompt_id, timeout_seconds=1800)
            seconds = round(time.perf_counter() - started, 1)  # the first image includes loading the models
            name = f"sdxl_ipadapter_{args.tag}" if args.tag else f"sdxl_ipadapter_test_w{round(weight * 10):02d}"
            result = out / f"{name}.png"
            result.write_bytes(client.view(outputs["11"]["images"][0]))
            if "24" in outputs:
                (out / f"{name}_lines.png").write_bytes(client.view(outputs["24"]["images"][0]))
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
        (out / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.copy_to and not args.crop_only:
            import shutil

            target = Path(args.copy_to)
            target.mkdir(parents=True, exist_ok=True)
            for name in ["reference_crop_clean.png", report_name, *(Path(i["result"]).name for i in report["images"])]:
                shutil.copy2(out / name, target / name)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
