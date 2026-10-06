"""Spike: a short image-to-video clip with the local Wan 2.2 TI2V-5B Turbo
(ComfyUI) from a keyframe made by the SDXL + LoRA + IP-Adapter pipeline.

    python spikes/wan_video_spike.py --start-frame spikes/output/sdxl_ipadapter/x.png
        --prompt "..." --tag bench [--seconds 2] [--steps 4] [--seed 42]

Uses the project's own backend (`wan-ti2v-5b-turbo-comfyui`) exactly as the
benchmark runner does, but writes into spikes/output/wan_video/<tag>/
(git-ignored) and records nothing under outputs/ or benchmarks/. Reports
time and peak VRAM / RAM beside the clip, plus a contact sheet of frames.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_studio.generation.backends.wan_comfyui import WanTurboComfyUIBackend  # noqa: E402
from ai_studio.generation.request import Resolution, VideoGenerationRequest  # noqa: E402
from ai_studio.quality.frames import decode_video  # noqa: E402
from sdxl_ipadapter_spike import Monitor, available_ram_gb, vram_used_mib  # noqa: E402

OUT = ROOT / "spikes/output/wan_video"
NEGATIVE = (
    "photorealistic, live action, 3D render look, glossy plastic shading, soft airbrushed gradients, blurry, "
    "motion blur smear, flickering lines, boiling line art, morphing face, changing hairstyle, changing clothing, "
    "extra limbs, deformed hands, text, logo, watermark, signature, speech bubbles, mask, superhero costume"
)


def contact_sheet(video: Path, target: Path, columns: int = 4, tiles: int = 8) -> None:
    from PIL import Image

    frames = decode_video(video, analysis_width=None, gray=False)
    picks = [round(i * (len(frames) - 1) / (tiles - 1)) for i in range(tiles)]
    width, height = 320, round(320 * frames.shape[1] / frames.shape[2])
    rows = -(-tiles // columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (40, 40, 40))
    for index, frame_index in enumerate(picks):
        tile = Image.fromarray(frames[frame_index]).resize((width, height))
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    sheet.save(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-frame", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=None, help="num_inference_steps (backend default: 4 for Turbo)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    request = VideoGenerationRequest(
        prompt=args.prompt, negative_prompt=NEGATIVE, duration_seconds=args.seconds,
        output_resolution=Resolution(width=1280, height=704), seed=args.seed,
        start_frame=str(Path(args.start_frame).resolve()),
        backend_parameters={} if args.steps is None else {"num_inference_steps": args.steps},
    )
    report: dict = {
        "tag": args.tag, "start_frame": args.start_frame, "prompt": args.prompt, "negative_prompt": NEGATIVE,
        "seconds_requested": args.seconds, "seed": args.seed, "steps": args.steps or "backend default",
        "ram_available_before_gb": round(available_ram_gb(), 2), "vram_used_before_mib": vram_used_mib(),
    }
    backend = WanTurboComfyUIBackend()
    monitor = Monitor()
    try:
        backend.validate_inputs(request)
        started = time.perf_counter()
        backend.prepare()
        report["prepare_seconds"] = round(time.perf_counter() - started, 1)
        monitor.start()
        started = time.perf_counter()
        result = backend.generate(request, out)
        report["generate_seconds"] = round(time.perf_counter() - started, 1)
        report["result"] = {
            "output_video": result.output_video, "frame_count": result.frame_count, "fps": result.fps,
            "inference_steps": result.inference_steps, "guidance": result.guidance,
            "backend_time_seconds": result.generation_time_seconds, "warnings": result.warnings,
            "parameters": result.parameters,
            "start_frame_preprocessing": dataclasses.asdict(result.start_frame_conditioning)
            if result.start_frame_conditioning else None,
        }
        contact_sheet(Path(result.output_video), out / "contact_sheet.png")
        return 0
    except Exception as exc:  # a spike: report whatever went wrong, change nothing
        report["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        monitor.stop.set()
        report["vram_peak_mib"] = monitor.vram_peak
        report["ram_available_lowest_gb"] = None if monitor.ram_low == float("inf") else round(monitor.ram_low, 2)
        backend.close()
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
