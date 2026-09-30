# Benchmark: comic_rooftop_01

**Phase 1B.1 — first comic animation benchmark.**
Definition: [`benchmarks/definitions/comic_rooftop_01.yaml`](../benchmarks/definitions/comic_rooftop_01.yaml)
Task type: **`video_generation`** (no source video).
Status: defined; reference assets not yet created; no model installed.

## Goal

Test whether a video pipeline can keep **one character** and a **graphic-novel
/ hybrid 2D-3D visual direction** stable across a short animated shot.

This is deliberately the easiest possible comic shot (Level 1): almost no
character motion, slow camera. If a backend cannot hold identity, line work
and style here, harder shots are not worth testing on it yet.

## Shot

| | |
|---|---|
| Duration target | ~5 seconds |
| Character | `COMIC_HERO_01` (original character, see below) |
| Scene | Young man sitting on the edge of a high rooftop, overlooking a dense city at late sunset, city lights starting to glow |
| Character motion | Mostly stationary. Subtle wind in hair and clothing. Near the end, a slight head turn toward the camera |
| Camera | Slow cinematic push-in from a low three-quarter angle behind him |

The brief allowed "stands/sits" and "night/sunset". A fixed benchmark needs a
single choice, so the definition fixes **sitting** (the most stationary pose,
matching Level 1) and **late sunset** (one lighting condition, not two).

## Character: COMIC_HERO_01

Original design, not based on any existing character, title or costume.
The canonical frame defines it. Only features visible there count: young man
with curly dark hair, dark hooded streetwear with subtle dark-red accents,
dark trousers, light sneakers, dark backpack with subtle red detailing. No
mask, logo, emblem or text.

**Canonical start frame:** `benchmarks/references/comic_hero_01/rooftop_frame_v001.png`
(1672×941 PNG, SHA-256 `3a3e7675…dbf1e0fb`). It is the benchmark's
`start_frame`, the exact image conditioning frame 0 (image-to-video), and
is not listed in `reference_images`. The benchmark pins `duration_seconds: 5`
and `fps: 24`. Only backends that declare start-frame conditioning can run it.
Further sheets (front, profile, ...) are planned. See that folder's README.

## Visual direction

- graphic novel aesthetic
- hybrid 2D/3D animation
- strong ink contours
- controlled cel shading
- halftone print texture
- hand-drawn hatch details
- layered city background
- cinematic perspective
- intentional limited / stepped animation feel
- stable facial structure
- stable clothing design
- stable background geometry

## Prompt policy

The prompt describes the look in generic art terms only. It must never contain
copyrighted character names, film or series titles, studio names, logos or
franchise-specific costumes, and the negative prompt excludes text, logos,
masks and superhero costumes. The exact prompt and negative prompt live in the
YAML definition. That file is the single source of truth, and it is frozen
once the first run exists.

## Manual success criteria (experimental)

Rated **1–10, higher is always better**. These comic/anime-specific criteria
are **experimental**. They live in their own schema,
`ComicAnimationEvaluation` (`src/ai_studio/benchmarking/comic_evaluation.py`),
separate from the Phase 1 object-editing `HumanEvaluation` (unchanged) and
from any future automatic Quality Gate metrics.

| Criterion | 10 | 1 |
|---|---|---|
| `character_identity` | unmistakably COMIC_HERO_01 throughout: hair, face, outfit match the reference | a different or generic character |
| `face_stability` | facial structure and proportions constant, including through the head turn | face morphs, drifts or melts |
| `line_stability` | ink contours hold their shape and weight frame to frame | lines boil, crawl, break or vanish |
| `style_consistency` | ink, cel shading, halftone and hatching present and uniform for the whole shot | style drifts toward generic 3D / photoreal, or changes mid-shot |
| `background_stability` | buildings keep their geometry and layering; parallax is plausible | architecture warps, melts or pops |
| `motion_quality` | wind and head turn read as deliberate and natural, and the stepped timing is clearly intentional | jitter, sliding, unmotivated movement, or motion missing entirely |
| `camera_motion` | smooth, steady push-in with consistent perspective | camera drifts, shakes, or zooms without matching perspective |
| `temporal_consistency` | no unintended flicker; lighting and colour constant | heavy flicker or colour/light pumping |
| `artifact_visibility` | **no visible distracting artifacts** | **severe obvious artifacts** |
| `overall_visual_quality` | could be a frame-accurate shot from a finished production | unusable |

`artifact_visibility`: a **high score means few artifacts**, the same as in
the Phase 1 schema.

**Stepped animation ≠ temporal inconsistency.** The shot intentionally uses
limited/stepped timing (holding drawings for 2+ frames). Evaluators must not
penalise deliberate holds under `temporal_consistency` or `motion_quality`.
Penalise only *unintended* change: flicker, boiling, drift.

No weighted score and no automatic winner, as for all Phase 1 benchmarks.

### Recording ratings

```python
from ai_studio.benchmarking import BenchmarkLogger, ComicAnimationEvaluation, ComicAnimationEvaluationStore
from ai_studio.paths import comic_evaluations_dir, results_dir

store = ComicAnimationEvaluationStore(comic_evaluations_dir(), results=BenchmarkLogger(results_dir()))
store.save(ComicAnimationEvaluation(
    benchmark_id="comic_rooftop_01", run_id="<run_id>", evaluator="<name>",
    character_identity=7, face_stability=6, line_stability=5, style_consistency=8,
    background_stability=7, motion_quality=6, camera_motion=8,
    temporal_consistency=6, artifact_visibility=7, overall_visual_quality=6,
    notes="line boil on the jacket around 3s",
))
```

- **Storage:** `benchmarks/comic_evaluations/<benchmark_id>/<run_id>/<evaluator>.r001.json`.
- **Overwriting:** `save()` refuses to overwrite. Re-rating is explicit with `save_revision()` (r002, ...), and earlier revisions are kept.
- **Linked runs:** with the results logger attached, only successful `video_generation` runs can be rated.

## Progressive difficulty

Each level adds one major source of failure. A backend should be tried at
level N+1 only once it is acceptable at level N, and each level will get its
own fixed benchmark definition.

| Level | Shot | New difficulty introduced |
|---|---|---|
| **1** | stationary character + slow camera *(this benchmark)* | baseline identity, line and style stability |
| 2 | walking | full-body articulation, cyclic motion, feet contact, clothing follow-through |
| 3 | jump / landing | large vertical displacement, squash/stretch, impact timing, silhouette changes |
| 4 | fast camera motion | rapid perspective change, background re-layout, motion blur vs. clean line art |
| 5 | fight movement | fast limbs, self-occlusion, hands, extreme poses, readable action |
| 6 | two interacting characters | two identities at once, contact, occlusion between characters, no identity bleeding |

## Constants for fair comparison

In addition to the Phase 1 rules (docs/PHASE_1_BENCHMARK.md), all compared
runs must use:

- the same YAML definition, prompt and negative prompt
- the same reference images (verified by checksum)
- the same set of seeds per backend
- the same output resolution, and the same fps and duration target where the
  backend allows it. Backends that cannot hit ~5 s record their actual
  `fps` / `frame_count`.

## Resolved and open gaps

Resolved in Phase 1B:
- **Generation vs. editing.** The benchmark is declared
  `task_type: video_generation` and runs through `VideoGenerationRequest` /
  `VideoGenerationBackend`. There is no fake source video.
- **Structured shot fields.** `duration_seconds`, `camera_motion`,
  `subject_motion` and `visual_direction` are real definition fields and are
  recorded with every run.
- **Criteria.** The comic criteria have their own experimental
  `ComicAnimationEvaluation` schema and store.

Open:
1. **The definition stays `status: draft` until the reference sheet exists.**
   It is complete enough to run text-only, because `reference_images` may
   legitimately be empty for generation, but the runner refuses drafts. A
   human marks it `ready` once the COMIC_HERO_01 sheet is final and listed in
   `reference_images`.
2. **Reference assets are external reproducibility dependencies.** They are
   checksummed, not version-controlled. Shared object storage comes before
   multi-machine reproduction (see PHASE_1_BENCHMARK.md).
