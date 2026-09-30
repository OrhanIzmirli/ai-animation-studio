# COMIC_HERO_01 — reference assets

Original character for benchmark `comic_rooftop_01`. Not based on any existing
character, franchise or costume.

Media files in this folder are git-ignored. Their SHA-256 checksums are
recorded in every benchmark run that uses them. **Once a file has been used in
a run, never edit or replace it.** Add a new file under a new name instead.

## Design (canonical, as visible in rooftop_frame_v001.png)

- young man
- curly dark hair
- dark hooded streetwear with subtle dark-red accents
- dark trousers
- light sneakers
- dark backpack with subtle red detailing
- no mask, no logo, no emblem, no text on the clothing

Only features visible in the canonical frame are part of the design.

## Canonical files

| File | Content | Dimensions | SHA-256 |
|---|---|---|---|
| `rooftop_frame_v001.png` | canonical `start_frame` (frame 0) of `comic_rooftop_01` | 1672×941, RGB 8-bit PNG, 2,497,820 bytes | `3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb` |

Verify a copy with: `python -c "from ai_studio.benchmarking.runner import sha256_file; print(sha256_file('benchmarks/references/comic_hero_01/rooftop_frame_v001.png'))"`

## Other files (planned, not yet used by any benchmark)

| File | Content |
|---|---|
| `front.png` | full body, front view, neutral pose |
| `three_quarter.png` | full body, three-quarter view |
| `profile.png` | full body, side view |
| `face_closeup.png` | head and shoulders, neutral expression |
| `style_frame.png` | one finished frame in the target look: ink contours, cel shading, halftone, hatching |

All images should be drawn in the target graphic-novel style, on a plain
background, at a consistent scale.
