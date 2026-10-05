"""Build the small training set for a COMIC_HERO_01 character LoRA (spike).

Reads the approved reference images (never writes to them) and writes
1024x1024 crops with one caption file each to
spikes/output/hero_lora_dataset/ (git-ignored), plus contact_sheet.png to
check the crops by eye.

    python spikes/build_hero_lora_dataset.py [--install-to COMFYUI_INPUT_SUBFOLDER]

--install-to also copies the set into <ComfyUI>/input/<name>/, where
ComfyUI's "Load Image-Text (from Folder)" node reads it.

Sources: the approved turnaround sheet (three figures; its handwritten notes,
swatches, shoe drawing and signature are painted over with the background
grey before cropping), the shot_03 start keyframe and the canonical rooftop
frame. The spike's own generations (w08, prompt_only_c) are left out: their
jacket is grey and they carry invented details, which is what the LoRA is
meant to fix.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHEET = ROOT / "benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg"
ROOFTOP = ROOT / "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
EYES = ROOT / "benchmarks/references/trailer_01_keyframes/shot_03_start.jpg"
OUT = ROOT / "spikes/output/hero_lora_dataset"
SIZE = 1024
TRIGGER = "hero01"
STYLE = "hand-inked comic illustration"
OUTFIT = "black hooded jacket with dark-red accents, black backpack with dark-red straps"

# Things on the 1024x765 sheet that are not the character: (box, background sample pixel).
SHEET_PAINT = [
    ((0, 0, 135, 46), (140, 60)),        # "PAGE 23"
    ((238, 46, 420, 120), (330, 140)),   # "HAIR VOLUME CONSISTENCY"
    ((20, 530, 68, 685), (40, 500)),     # colour swatches
    ((288, 572, 432, 702), (350, 540)),  # "SHOE DETAIL", the shoe drawing and its pointer
    ((640, 140, 748, 258), (690, 300)),  # "BACKPACK POSITION" and its pointer
    ((934, 702, 1024, 765), (990, 680)), # signature
]
# name, crop box on the sheet, caption
SHEET_CROPS = [
    ("sheet_front_full", (66, 44, 312, 750),
     f"{TRIGGER}, a young man with short curly dark hair, full body, front view, standing, arms at his sides, "
     f"{OUTFIT}, dark cargo trousers, white low-top sneakers, plain light grey background, {STYLE}"),
    ("sheet_three_quarter_full", (396, 44, 632, 750),
     f"{TRIGGER}, a young man with short curly dark hair, full body, side three-quarter view facing right, standing, "
     f"{OUTFIT}, dark cargo trousers, white low-top sneakers, plain light grey background, {STYLE}"),
    ("sheet_back_full", (736, 44, 966, 750),
     f"{TRIGGER}, a young man with short curly dark hair, full body, seen from behind, standing, "
     f"{OUTFIT}, dark cargo trousers, white low-top sneakers, plain light grey background, {STYLE}"),
    ("sheet_front_upper", (40, 46, 340, 346),
     f"{TRIGGER}, a young man with short curly dark hair, upper body, front view, calm expression, {OUTFIT}, "
     f"plain light grey background, {STYLE}"),
    ("sheet_three_quarter_upper", (372, 46, 672, 346),
     f"{TRIGGER}, a young man with short curly dark hair, upper body, side three-quarter view facing right, "
     f"calm expression, {OUTFIT}, plain light grey background, {STYLE}"),
    ("sheet_back_upper", (704, 46, 1004, 346),
     f"{TRIGGER}, a young man with short curly dark hair, upper body, seen from behind, {OUTFIT}, "
     f"plain light grey background, {STYLE}"),
]
ROOFTOP_CROP = ("rooftop_from_behind", (170, 150, 960, 940),
                f"{TRIGGER}, a young man with short curly dark hair, seen from behind, sitting on a rooftop ledge, "
                f"{OUTFIT}, dark trousers, white sneakers, city skyline at sunset, orange and violet light, {STYLE}")
EYES_CROP = ("eyes_close_up", (912, 0, 2256, 1344),
             f"{TRIGGER}, a young man with short curly dark hair, extreme close-up of his eyes, dark eyes and dark "
             f"eyebrows, warm orange light from one side and violet shadow, {STYLE}")


def square(image, background):
    """Pad to a square with `background`, then scale to SIZE x SIZE (Lanczos)."""
    from PIL import Image

    side = max(image.size)
    canvas = Image.new("RGB", (side, side), background)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def build() -> list[Path]:
    from PIL import Image, ImageDraw

    OUT.mkdir(parents=True, exist_ok=True)
    made: list[tuple[str, object, str]] = []
    with Image.open(SHEET) as sheet:
        sheet = sheet.convert("RGB")
    draw = ImageDraw.Draw(sheet)
    for box, sample in SHEET_PAINT:
        draw.rectangle(box, fill=sheet.getpixel(sample))
    for name, box, caption in SHEET_CROPS:
        crop = sheet.crop(box)
        made.append((name, square(crop, crop.getpixel((2, 2))), caption))
    for path, (name, box, caption) in ((ROOFTOP, ROOFTOP_CROP), (EYES, EYES_CROP)):
        with Image.open(path) as image:
            crop = image.convert("RGB").crop(box)
        made.append((name, square(crop, (0, 0, 0)), caption))

    files = []
    for name, image, caption in made:
        target = OUT / f"{name}.png"
        image.save(target)
        (OUT / f"{name}.txt").write_text(caption + "\n", encoding="utf-8")
        files.append(target)
    columns = 4
    rows = -(-len(made) // columns)
    sheet_image = Image.new("RGB", (columns * 256, rows * 256), (40, 40, 40))
    for index, (_, image, _) in enumerate(made):
        sheet_image.paste(image.resize((256, 256)), ((index % columns) * 256, (index // columns) * 256))
    sheet_image.save(OUT / "contact_sheet.png")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-to", default=None, help="subfolder name under ComfyUI's input folder")
    args = parser.parse_args()
    files = build()
    print(f"{len(files)} images with captions in {OUT}")
    if args.install_to:
        from ai_studio.generation.backends.comfyui import load_config

        target = load_config().comfyui_dir / "input" / args.install_to
        target.mkdir(parents=True, exist_ok=True)
        for image in files:
            shutil.copy2(image, target / image.name)
            shutil.copy2(image.with_suffix(".txt"), target / image.with_suffix(".txt").name)
        print(f"copied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
