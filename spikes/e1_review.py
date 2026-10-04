"""Review material for experiment E1 (throwaway analysis script, not src/).

    python spikes/e1_review.py

Reads the run records of comic_rooftop_01 (the E1 runs, purpose "experiment",
plus the earlier seed-42 Turbo run) and writes to spikes/output/E1/
(git-ignored):

    contact_<label>.png   frames 0, 24, 48, 72, 96, 120, labelled
    face_<label>.png      the same frames cropped to HEAD_BOX
    compare_grid.mp4      all clips tiled in one 24 fps video, labelled
    table.md              the table used in docs/experiments/E1_REPORT.md
    numbers.json          a few extra per-run numbers (saturation, crop change)

HEAD_BOX is one fixed box for every run, chosen by eye on the derived start
frame (1280x704): the character's head and shoulders with some margin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ai_studio.generation.backends.wan_ti2v import encode_frames  # noqa: E402
from ai_studio.quality.frames import decode_video  # noqa: E402

BENCHMARK = "comic_rooftop_01"
BASELINE_RUN = "20261004T210526Z-92e20c65"  # Turbo defaults, seed 42 (before E1)
OUT = REPO / "spikes" / "output" / "E1"
FRAMES = (0, 24, 48, 72, 96, 120)
HEAD_BOX = (160, 100, 660, 420)  # left, top, right, bottom in the 1280x704 frame
TILE = (640, 352)
LABEL_HEIGHT = 30


def font(size: int = 20):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: fixed-size bitmap font
        return ImageFont.load_default()


def labelled(image: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + LABEL_HEIGHT), (0, 0, 0))
    ImageDraw.Draw(out).text((6, 3), text, fill=(255, 255, 255), font=font())
    out.paste(image, (0, LABEL_HEIGHT))
    return out


def sheet(images: list[Image.Image], columns: int) -> Image.Image:
    width, height = images[0].size
    rows = -(-len(images) // columns)
    out = Image.new("RGB", (columns * width, rows * height), (0, 0, 0))
    for index, image in enumerate(images):
        out.paste(image, ((index % columns) * width, (index // columns) * height))
    return out


def settings_text(record: dict) -> str:
    p = record["model_parameters"]
    if not p:
        return "-"
    text = f"{p.get('num_inference_steps')} steps, CFG {p.get('cfg'):g}, {p.get('sampler')}/{p.get('scheduler')}, shift {p.get('shift'):g}"
    if p.get("latent_scale", 1.0) != 1.0:
        text += f", latent_scale {p['latent_scale']:g}"
    return text


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((REPO / "benchmarks" / "results" / BENCHMARK).glob("*.json"))
    ]
    chosen = [
        r for r in records
        if r["run_id"] == BASELINE_RUN or r["task_parameters"].get("run_purpose") == "experiment"
    ]
    # Order: Turbo defaults by seed (42 first), then the Turbo variant, then base.
    def order(r: dict) -> tuple:
        turbo = "turbo" in (r.get("backend") or "")
        variant = (r["model_parameters"] or {}).get("latent_scale", 1.0) != 1.0
        return (not turbo, variant, r["seed"] != 42, r["seed"], r["run_id"])

    chosen.sort(key=order)
    rows, numbers, tiles = [], {}, []
    for record in chosen:
        run_id = record["run_id"]
        short = run_id.split("-")[-1]
        kind = "base" if "turbo" not in (record.get("backend") or "") else "turbo"
        variant = "_ls0.8" if (record["model_parameters"] or {}).get("latent_scale", 1.0) != 1.0 else ""
        label = f"{kind}_seed{record['seed']}{variant}_{short}"
        env, tp = record["environment"], record["model_parameters"] or {}
        metrics_path = REPO / "benchmarks" / "metrics" / BENCHMARK / run_id / "metrics_v1.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"] if metrics_path.is_file() else {}
        lowest, total = env.get("ram_available_min_gb"), env.get("ram_total_gb")
        rows.append({
            "label": label, "run_id": run_id, "backend": record.get("backend"), "seed": record["seed"],
            "settings": settings_text(record), "status": record["status"],
            "error": record.get("error_message"),
            "total_s": record.get("total_time_seconds"), "prepare_s": record.get("prepare_time_seconds"),
            "generation_s": record.get("generation_time_seconds"),
            "comfyui_prompt_s": env.get("comfyui_prompt_seconds"),
            "ram_before_gb": env.get("ram_available_before_gb"), "ram_lowest_gb": lowest,
            "ram_peak_used_gb": None if lowest is None or total is None else round(total - lowest, 2),
            "git_dirty": env.get("git_dirty"), "git_commit": (env.get("git_commit") or "")[:7],
            **{k: metrics.get(k) for k in (
                "motion_mean", "motion_max", "total_change", "start_frame_psnr", "temporal_pulse_ratio",
                "luminance_flicker", "luminance_min", "luminance_max", "held_frame_ratio",
            )},
        })
        if record["status"] != "success":
            continue

        video = decode_video(REPO / record["output_video"], analysis_width=1280, gray=False)
        picks = [Image.fromarray(video[min(i, len(video) - 1)]) for i in FRAMES]
        sheet([labelled(im.resize(TILE, Image.Resampling.LANCZOS), f"frame {i}") for im, i in zip(picks, FRAMES)], 3) \
            .save(OUT / f"contact_{label}.png")
        sheet([labelled(im.crop(HEAD_BOX), f"frame {i}") for im, i in zip(picks, FRAMES)], 3) \
            .save(OUT / f"face_{label}.png")

        # A few plain numbers to support the written description.
        hsv = [np.asarray(im.convert("HSV"), dtype=np.float64) for im in picks]
        crops = [np.asarray(im.crop(HEAD_BOX).convert("L"), dtype=np.float64) for im in picks]
        numbers[label] = {
            "saturation_mean_per_frame": [round(float(h[..., 1].mean()), 1) for h in hsv],
            "value_mean_per_frame": [round(float(h[..., 2].mean()), 1) for h in hsv],
            "head_box_change_vs_frame0": [round(float(np.abs(c - crops[0]).mean()), 2) for c in crops],
            "full_frame_change_vs_frame0": [
                round(float(np.abs(np.asarray(im.convert("L"), dtype=np.float64)
                                   - np.asarray(picks[0].convert("L"), dtype=np.float64)).mean()), 2)
                for im in picks
            ],
        }

        small = [np.asarray(Image.fromarray(frame).resize(TILE, Image.Resampling.BILINEAR)) for frame in video]
        text = f"{short}  seed {record['seed']}  {kind}  {tp.get('num_inference_steps')} st CFG {tp.get('cfg'):g}" \
               + (f"  ls {tp['latent_scale']:g}" if variant else "")
        bar = np.asarray(labelled(Image.new("RGB", (TILE[0], 0)), text))  # just the label strip
        tiles.append([np.vstack([bar, frame]) for frame in small])
        del video

    if tiles:
        columns, count = 4, min(len(t) for t in tiles)
        tile_h, tile_w = tiles[0][0].shape[:2]
        rows_n = -(-len(tiles) // columns)

        def grid_frame(index: int) -> np.ndarray:
            canvas = np.zeros((rows_n * tile_h, columns * tile_w, 3), dtype=np.uint8)
            for n, tile in enumerate(tiles):
                y, x = (n // columns) * tile_h, (n % columns) * tile_w
                canvas[y:y + tile_h, x:x + tile_w] = tile[index]
            return canvas

        class Grid:
            def __len__(self):
                return count

            def __getitem__(self, index):
                return grid_frame(index)

            def __iter__(self):
                return (grid_frame(i) for i in range(count))

        encode_frames(Grid(), OUT / "compare_grid.mp4")

    def cell(value, digits=3):
        if value is None:
            return "-"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    header = ["label", "backend", "seed", "settings", "status", "total s", "prepare s", "generation s",
              "RAM before GB", "RAM lowest GB", "RAM peak used GB", "motion_mean", "motion_max", "total_change",
              "start_psnr dB", "pulse_ratio", "flicker", "lum min", "lum max", "held_ratio"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join([
            f"`{r['label']}`", f"`{r['backend']}`", str(r["seed"]), r["settings"], r["status"],
            cell(r["total_s"], 1), cell(r["prepare_s"], 1), cell(r["generation_s"], 1),
            cell(r["ram_before_gb"], 2), cell(r["ram_lowest_gb"], 2), cell(r["ram_peak_used_gb"], 2),
            cell(r["motion_mean"]), cell(r["motion_max"]), cell(r["total_change"]), cell(r["start_frame_psnr"], 2),
            cell(r["temporal_pulse_ratio"], 2), cell(r["luminance_flicker"]), cell(r["luminance_min"], 1),
            cell(r["luminance_max"], 1), cell(r["held_frame_ratio"]),
        ]) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "numbers.json").write_text(json.dumps({"rows": rows, "numbers": numbers}, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(numbers, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
