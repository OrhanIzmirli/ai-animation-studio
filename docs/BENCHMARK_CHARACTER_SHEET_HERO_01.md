# Benchmark: character_sheet_hero_01

Definition: [`benchmarks/definitions/character_sheet_hero_01.yaml`](../benchmarks/definitions/character_sheet_hero_01.yaml)
Task type: **`image_generation`** (reference image + prompt -> one image).
Status: **`draft`**. It cannot be run until a human sets `status: ready`.
No image backend is chosen yet; see `docs/research/IMAGE_BACKENDS.md`.

## Goal

Test whether an image backend can turn the **one canonical frame** of
`COMIC_HERO_01` into a **five-view character sheet** without changing who he
is, what he wears, or how the picture is drawn. The sheet becomes the identity
reference for per-shot keyframes (`trailers/trailer_01/shotlist.yaml`, the
shots with `needs_character_keyframe: true`).

## Input

`benchmarks/references/comic_hero_01/rooftop_frame_v001.png` (1672x941 PNG,
SHA-256 `3a3e7675...dbf1e0fb`), given as a **reference image**. It is not a
start frame and not an image to edit. The backend must really use reference
images (`max_reference_images >= 1`), otherwise the runner refuses the run.

## Output

One image per run: the whole sheet. Size and seed are chosen per run and
recorded. Suggested: a wide sheet such as 1536x1024.

## Required views

All five, in one row, same scale, common ground line, neutral standing pose,
plain light-grey background, whole figure visible:

| View | Meaning |
|---|---|
| `front` | facing the viewer |
| `three_quarter_left` | body turned 45 degrees to his left |
| `three_quarter_right` | body turned 45 degrees to his right |
| `profile` | exact side view, facing the left edge of the sheet |
| `back` | seen from behind |

## Fixed outfit (from the canonical frame)

Only what the canonical frame shows is fixed:

- young man, slim build, short curly dark hair with volume on top
- black hooded jacket, hood down, subtle dark-red accents
- black backpack on both shoulders, dark-red straps, small light buckles
- dark, loose cargo trousers with a side pocket
- white low-top sneakers with dark details
- dark red is the only accent colour
- no mask, no face covering, no logo, no emblem, no letter, no text

The frame shows him **from behind**. His **face is not defined by it**: the
first accepted sheet defines the face, which is a human decision.

## Style constraints

The look of the canonical frame: strong ink contours with varying line weight,
controlled cel shading, halftone texture in the shadows, hand-drawn hatching.
Flat, even lighting for the sheet; the frame's sunset light is not part of the
character design. No photorealism, no 3D-render look.

## No franchise motifs

Original character. Nothing that evokes an existing character, title or
costume: no mask, no emblem, no logo, no cape, no armor, no weapon, no
signature symbol.

## Evaluation (manual)

`CharacterSheetEvaluation`, stored by `CharacterSheetEvaluationStore` under
`benchmarks/character_sheet_evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json`.
Integer scores 1-10, **higher is always better**. No overall score, no winner.
Files are never overwritten; re-rating is an explicit new revision. Only
successful `image_generation` runs can be rated.

| Criterion | 10 | 1 |
|---|---|---|
| `identity_consistency` | clearly the same person in every view (face, hair, build, proportions) | the views show different people |
| `style_match` | drawn in the look of the canonical frame | a different style, e.g. photorealistic or 3D render |
| `outfit_fidelity` | every fixed outfit detail present and identical in every view, nothing added | wrong outfit, changes between views, or a logo, emblem or mask |
| `line_quality` | clean, confident, closed contours with deliberate line weight | broken, muddy or smeared lines |
| `anatomy` | correct proportions, hands, feet and a plausible turn in every view | deformed or impossible anatomy |

A sheet with a missing or extra view should say so in `notes`; it lowers
`identity_consistency` and `anatomy` only as far as the remaining views
deserve. Whether a missing view should fail a run outright is an open point.

## Canonical character reference (hand-made, approved 2026-10-05)

`benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg`
(1024x765 JPEG, SHA-256
`7a7b391bc05a948e57b52b6173a03ef912f890875b0865c07e90dacf47d09c8a`), made by
hand in the Gemini app and imported with `scripts/import_reference.py`. Its
provenance file lies beside it; the prompt chain is
[`docs/prompts/character_turnaround_hero_01.prompt.txt`](prompts/character_turnaround_hero_01.prompt.txt).

It is **one sheet image**, as this benchmark assumes, with **three views**
(front, three-quarter, back), not the five listed above. It is the canonical
reference for who `COMIC_HERO_01` is, including his face. It is not an input
of this benchmark: `reference_images` still holds only the canonical frame.

## Open points (why this is a draft)

1. Decided: the face is the one on the approved turnaround sheet.
2. The red marks on the jacket back in the canonical frame are read here as
   plain accents, not as a symbol. If they are meant as a design element, the
   outfit list must say exactly what they are.
3. Decided: one sheet image, not one image per view.
4. Which image backend, and at what size.
5. This draft asks for five views; the approved reference has three. Either
   the benchmark follows the reference (three views) or stays at five.
6. The style constraints above ask for flat, even lighting and clean, closed
   line work; the prompts that produced the approved reference ask for
   directional and rim lighting and visible under-sketch lines. Which of the
   two the benchmark should rate against is not decided.
