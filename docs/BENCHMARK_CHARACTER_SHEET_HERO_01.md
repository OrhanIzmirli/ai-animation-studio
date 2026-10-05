# Benchmark: character_sheet_hero_01

Definition: [`benchmarks/definitions/character_sheet_hero_01.yaml`](../benchmarks/definitions/character_sheet_hero_01.yaml)
Task type: **`image_generation`** (reference image + prompt -> one image).
Status: **`draft`**. It cannot be run until a human sets `status: ready`.
No image backend is chosen yet; see `docs/research/IMAGE_BACKENDS.md`.

## Goal

Test whether an image backend can turn the **one canonical frame** of
`COMIC_HERO_01` into a **three-view character sheet** without changing who he
is or what he wears, drawn like the approved hand-made turnaround sheet (see
"Canonical character reference" below). The sheet becomes the identity
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

All three, side by side, same scale, common ground line, neutral standing
pose, plain light-grey background, whole figure visible. They are the views
of the approved reference:

| View | Meaning |
|---|---|
| `front` | facing the viewer |
| `three_quarter` | body turned to his left, facing the right side of the sheet |
| `back` | seen from behind |

Dropped on 2026-10-05 (they were in the earlier five-view plan and are not on
the approved reference): `three_quarter_right`, `profile`.

## Fixed outfit (from the canonical frame)

Only what the canonical frame shows is fixed:

- young man, slim build, short curly dark hair with volume on top
- black hooded jacket, hood down, subtle dark-red accents
- black backpack on both shoulders, dark-red straps, small light buckles
- dark, loose cargo trousers with a side pocket
- white low-top sneakers with dark details
- dark red is the only accent colour
- no mask, no face covering, no logo, no emblem, no letter, no text

The frame shows him **from behind**. His **face is not defined by it**: it is
the face on the approved turnaround sheet.

## Style constraints

The style target is the approved turnaround sheet, not the earlier "flat,
even lighting and clean, closed line work". The wording follows turns 3 to 5
of the prompt chain that produced it
([`docs/prompts/character_turnaround_hero_01.prompt.txt`](prompts/character_turnaround_hero_01.prompt.txt)):

- **Lighting:** directional studio lighting with real highlights, core
  shadows and reflected light, identical in all views; not flat, not even. A
  touch of dark-red rim light on the edges of the hair and the jacket
  silhouette. The canonical frame's sunset light is not part of the design.
- **Line work:** hand-inked black ink contours with naturally varying,
  pressure-dependent line weight; thick outer contours, thinner interior
  details; ink pooling and nib texture may show. Not uniform.
- **Under-sketch:** faint construction lines visible in a few places, as if
  not fully erased.
- **Shading:** bold flat cel shading blocks on the clothing, coarse halftone
  screen-tone dots in the mid-tones (not a perfectly uniform dot grid), rough
  hand-drawn hatching and cross-hatching.
- **Finish:** deliberately varied across the image, face and torso tighter,
  other areas looser and more gestural; not uniformly polished.
- No photorealism, no 3D-render look, no smooth vector render, no soft
  airbrushed gradients.

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

It is **one sheet image** with **three views** (front, three-quarter, back).
It is the canonical reference for who `COMIC_HERO_01` is, including his face,
and the benchmark's views and style target follow it. It is not an input of
this benchmark: `reference_images` still holds only the canonical frame.

## Open points (why this is a draft)

1. Decided: the face is the one on the approved turnaround sheet.
2. The red marks on the jacket back in the canonical frame are read here as
   plain accents, not as a symbol. If they are meant as a design element, the
   outfit list must say exactly what they are.
3. Decided: one sheet image, not one image per view.
4. Which image backend, and at what size.
5. Decided (2026-10-05): three views (front, three-quarter, back), matching
   the approved reference. `three_quarter_right` and `profile` are dropped.
6. Decided (2026-10-05): the style target is the approved turnaround sheet
   (directional lighting, visible under-sketch, varying hand-inked line
   weight, halftone mid-tones, varied finish), not flat, even lighting and
   clean, closed line work.
