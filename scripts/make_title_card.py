"""Draw a retro glowing title card on a background plate (CPU, Pillow).

    python scripts/make_title_card.py --plate PLATE.png --title MILO
        --subtitle "Coming this Saturday Morning" --out outputs/title_cards/milo_02.png

The plate is scaled to cover 1280x704 and darkened; the title is set in a
heavy font with an orange glow and a turquoise face, the subtitle in a
handwriting font. Fonts are looked up in the Windows font folder. An existing
output file is never overwritten. Use the result as a shot's title_card.image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONTS = Path("C:/Windows/Fonts")
TITLE_FONT, SUBTITLE_FONT = "impact.ttf", "segoescb.ttf"
TURQUOISE, ORANGE, INK = (64, 224, 208), (255, 140, 40), (12, 10, 30)


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def glow_text(base: Image.Image, xy: tuple[int, int], text: str, font, face, glow, radius: int, outline: int) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, text, font=font, fill=glow + (255,), anchor="mm",
                               stroke_width=outline * 3, stroke_fill=glow + (255,))
    halo = layer.filter(ImageFilter.GaussianBlur(radius))
    base.alpha_composite(halo)
    base.alpha_composite(halo)
    ImageDraw.Draw(base).text(xy, text, font=font, fill=face + (255,), anchor="mm",
                              stroke_width=outline, stroke_fill=INK + (255,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plate", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, nargs=2, default=[1280, 704])
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"ERROR: {out} exists; choose another name")
        return 1
    width, height = args.size
    plate = cover(Image.open(args.plate).convert("RGB"), width, height)
    card = ImageEnhance.Brightness(plate).enhance(0.45).convert("RGBA")
    glow_text(card, (width // 2, int(height * 0.42)), args.title, ImageFont.truetype(str(FONTS / TITLE_FONT), int(height * 0.34)),
              TURQUOISE, ORANGE, radius=18, outline=6)
    if args.subtitle:
        glow_text(card, (width // 2, int(height * 0.74)), args.subtitle,
                  ImageFont.truetype(str(FONTS / SUBTITLE_FONT), int(height * 0.075)), (255, 236, 200), ORANGE,
                  radius=8, outline=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    card.convert("RGB").save(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
