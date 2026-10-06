"""Check a generated clip for the defects found in trailer milo_01.

    python scripts/check_clip.py VIDEO [--character] [--sheet OUT.png] [--json OUT.json]

Checks (each PASS / FAIL with the number behind it):

  colour_flicker   the background's colour pumping from frame to frame:
                   standard deviation of the frame-to-frame change of the mean
                   blue-minus-green and red-minus-green. milo_01 shot_01 (sky
                   pulsing blue/teal) measured 1.56; clean clips 0.13-0.55.
  pop              a sudden jump: the largest frame-to-frame difference
                   against the median one (something appears or vanishes at once).
                   The step from frame 0 (the keyframe) to frame 1 is reported
                   as start_step and not judged.
  character_area   with --character: the pale blue-white of the ghost, as a
                   share of the frame. A jump between two frames (limbs
                   appearing, the body splitting) or a large change over the
                   clip (melting into a wall, growing, shrinking) fails.

These are thresholds on simple measurements, set from milo_01's defects.
They catch flicker, pops and changes of the character's size; they do NOT
see a wrong face, a leg instead of a tail or a strange drawing. Look at the
contact sheet as well.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H = 320, 176
LIMITS = {
    "colour_flicker": 0.8,
    "pop_ratio": 3.5,           # largest / median frame difference ...
    "pop_min": 6.0,             # ... and only when the largest is visible at all
    "area_step": 0.15,          # largest frame-to-frame relative change of the ghost's area
    "area_range": 0.40,         # (max - min) / median over the clip
}


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"scale={W}:{H}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def ghost_mask(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (b > 150) & (g > 140) & (b - r > 15)


def check(frames: np.ndarray, character: bool) -> dict:
    f = frames.astype(np.float32)
    results: dict = {"frames": len(frames)}
    mask = ghost_mask(frames) if character else np.zeros(frames.shape[:3], bool)
    background = ~mask
    def mean_of(channel):
        return np.array([f[i, ..., channel][background[i]].mean() for i in range(len(f))])
    r, g, b = mean_of(0), mean_of(1), mean_of(2)
    flicker = float(max(np.diff(b - g).std(), np.diff(r - g).std()))
    results["colour_flicker"] = {"value": round(flicker, 2), "limit": LIMITS["colour_flicker"],
                                 "pass": flicker <= LIMITS["colour_flicker"]}

    all_diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))
    # Frame 0 is the keyframe itself; the model redraws it from frame 1, so
    # every clip has a step there. Reported, not judged: the shot's fade-in
    # covers it.
    results["start_step"] = round(float(all_diff[0]), 2)
    diff = all_diff[1:]
    moving = diff[diff > 0.3]  # holds of the style layer are not motion
    median = float(np.median(moving)) if len(moving) else 0.0
    worst = int(np.argmax(diff))
    ratio = float(diff[worst] / median) if median else 0.0
    pop = ratio > LIMITS["pop_ratio"] and diff[worst] > LIMITS["pop_min"]
    results["pop"] = {"value": round(ratio, 2), "largest": round(float(diff[worst]), 2), "at_frame": worst + 2,
                      "limit": LIMITS["pop_ratio"], "pass": not pop}

    if character:
        area = mask.mean(axis=(1, 2))
        if area.max() < 0.005:
            results["character_area"] = {"pass": False, "note": "no ghost found in any frame"}
        else:
            steps = np.abs(np.diff(area)) / np.maximum(area[:-1], 1e-6)
            spread = float((area.max() - area.min()) / max(np.median(area), 1e-6))
            at = int(np.argmax(steps))
            results["character_area"] = {
                "first": round(float(area[0]), 4), "min": round(float(area.min()), 4), "max": round(float(area.max()), 4),
                "largest_step": round(float(steps[at]), 3), "at_frame": at + 1, "range": round(spread, 3),
                "limits": [LIMITS["area_step"], LIMITS["area_range"]],
                "pass": bool(steps[at] <= LIMITS["area_step"] and spread <= LIMITS["area_range"]),
            }
    results["pass"] = all(v["pass"] for v in results.values() if isinstance(v, dict))
    return results


def contact_sheet(frames: np.ndarray, target: Path, count: int = 8) -> None:
    from PIL import Image, ImageDraw

    picks = np.linspace(0, len(frames) - 1, count).round().astype(int)
    sheet = Image.new("RGB", (4 * W, 2 * H))
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(picks):
        x, y = (k % 4) * W, (k // 4) * H
        sheet.paste(Image.fromarray(frames[i]), (x, y))
        draw.text((x + 4, y + 2), f"frame {i}", fill=(255, 255, 0))
    sheet.save(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--character", action="store_true", help="the clip shows the pale ghost")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    frames = decode(Path(args.video))
    results = check(frames, args.character)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    if args.sheet:
        contact_sheet(frames, Path(args.sheet))
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
