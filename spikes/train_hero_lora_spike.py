"""Spike: train a small COMIC_HERO_01 character LoRA on SDXL base, locally,
with ComfyUI's built-in training nodes (no extra tool is installed).

    python spikes/build_hero_lora_dataset.py --install-to hero01_lora
    python spikes/train_hero_lora_spike.py --steps 10 --name smoke      # does it fit, how fast
    python spikes/train_hero_lora_spike.py --steps 800 --name hero01_r8

In parts (the node saves only at the end, so a long run that is interrupted
loses everything): give each part the file the previous one printed as
"installed_as" and a new seed.

    python spikes/train_hero_lora_spike.py --steps 100 --name hero01_r8 --seed 1
    python spikes/train_hero_lora_spike.py --steps 100 --name hero01_r8 --seed 2
        --existing ai_studio_hero01_r8_100_steps_00001_.safetensors

Writes the LoRA to spikes/output/hero_lora/<name>.safetensors (git-ignored)
and copies it to <ComfyUI>/models/loras/, with a report beside it. Nothing is
written under outputs/ or benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_studio.generation.backends.comfyui import ComfyUIServer, load_config  # noqa: E402
from sdxl_ipadapter_spike import CHECKPOINT, Monitor, available_ram_gb, vram_used_mib  # noqa: E402

OUT = ROOT / "spikes/output/hero_lora"


def workflow(
    folder: str, name: str, steps: int, learning_rate: float, rank: int, algorithm: str, offloading: bool,
    checkpoint_depth: int = 1, existing: str = "[None]", seed: int = 42,
) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "LoadImageTextDataSetFromFolder", "inputs": {"folder": folder}},
        "3": {"class_type": "MakeTrainingDataset", "inputs": {
            "images": ["2", 0], "vae": ["1", 2], "clip": ["1", 1], "texts": ["2", 1],
        }},
        "4": {"class_type": "TrainLoraNode", "inputs": {
            "model": ["1", 0], "latents": ["3", 0], "positive": ["3", 1],
            "batch_size": 1, "grad_accumulation_steps": 1, "steps": steps, "learning_rate": learning_rate,
            "rank": rank, "optimizer": "AdamW", "loss_function": "MSE", "seed": seed,
            "training_dtype": "bf16", "lora_dtype": "bf16", "quantized_backward": False, "algorithm": algorithm,
            "gradient_checkpointing": True, "checkpoint_depth": checkpoint_depth, "offloading": offloading,
            "existing_lora": existing, "bucket_mode": False, "bypass_mode": False,
        }},
        "5": {"class_type": "SaveLoRA", "inputs": {"lora": ["4", 0], "prefix": f"loras/ai_studio_{name}", "steps": ["4", 2]}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--folder", default="hero01_lora", help="dataset subfolder of ComfyUI's input folder")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--offloading", action="store_true", help="keep part of the model in RAM while training")
    parser.add_argument("--checkpoint-depth", type=int, default=1, help="gradient checkpointing depth (1-5); deeper saves memory")
    parser.add_argument("--existing", default="[None]",
                        help="continue this LoRA (a file in <ComfyUI>/models/loras named ..._<N>_steps_...)")
    parser.add_argument("--seed", type=int, default=42, help="use a new one for every part")
    parser.add_argument("--timeout-minutes", type=float, default=240)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = load_config()
    report: dict = {
        "name": args.name, "checkpoint": CHECKPOINT, "dataset_folder": args.folder, "steps": args.steps,
        "learning_rate": args.learning_rate, "rank": args.rank, "offloading": args.offloading,
        "checkpoint_depth": args.checkpoint_depth, "continued_from": args.existing, "seed": args.seed,
        "batch_size": 1, "optimizer": "AdamW", "training_dtype": "bf16", "gradient_checkpointing": True,
        "ram_available_before_gb": round(available_ram_gb(), 2), "vram_used_before_mib": vram_used_mib(),
    }
    server = ComfyUIServer(config)
    monitor = Monitor()
    try:
        server.ensure_running()
        client = server.client
        info = client.node_info("TrainLoraNode")
        if info is None:
            report["error"] = "this ComfyUI has no TrainLoraNode"
            return 1
        algorithm = info["input"]["required"]["algorithm"][0]
        algorithm = (algorithm if isinstance(algorithm[0], str) else algorithm[0])[0] if isinstance(algorithm, list) else "LoRA"
        report["algorithm"] = algorithm
        before = set((config.comfyui_dir / "output" / "loras").glob(f"ai_studio_{args.name}_*.safetensors"))
        monitor.start()
        started = time.perf_counter()
        prompt_id = client.queue_prompt(workflow(
            args.folder, args.name, args.steps, args.learning_rate, args.rank, algorithm, args.offloading,
            args.checkpoint_depth, args.existing, args.seed,
        ))
        client.wait_for_prompt(prompt_id, timeout_seconds=args.timeout_minutes * 60, poll_seconds=5)
        report["seconds"] = round(time.perf_counter() - started, 1)
        report["seconds_per_step_including_setup"] = round(report["seconds"] / args.steps, 2)
        made = sorted(set((config.comfyui_dir / "output" / "loras").glob(f"ai_studio_{args.name}_*.safetensors")) - before)
        if not made:
            report["error"] = "training finished but no LoRA file was written"
            return 1
        # Keep ComfyUI's file name: its "<N>_steps_" part is how the next part finds the step count.
        target = OUT / made[-1].name
        shutil.copy2(made[-1], target)
        installed = config.comfyui_dir / "models" / "loras" / made[-1].name
        shutil.copy2(made[-1], installed)
        report.update(lora=str(target), lora_bytes=target.stat().st_size, installed_as=installed.name)
        return 0
    except Exception as exc:  # a spike: report whatever went wrong, change nothing
        report["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        monitor.stop.set()
        report["vram_peak_mib"] = monitor.vram_peak
        report["ram_available_lowest_gb"] = None if monitor.ram_low == float("inf") else round(monitor.ram_low, 2)
        if server.log_path is not None:
            report["comfyui_log"] = str(server.log_path)
            report["log_tail"] = server.log_tail(12).splitlines()
        server.close()
        part = Path(report["lora"]).stem if "lora" in report else f"{args.name}_failed_seed{args.seed}"
        (OUT / f"report_train_{part}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
