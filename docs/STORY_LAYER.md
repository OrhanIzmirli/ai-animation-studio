# Story / director layer

Status: **skeleton** (2026-10-05). Schemas, validators
and a shot-to-request converter exist and are tested. Nothing in this layer
generates an image or a video, and nothing in the rest of the engine calls it
yet.

## What it is for

The engine can run one benchmark shot. A trailer or a film needs a level
above that: what the story is, who is in it, which images define each
character, and which shots tell it. This layer holds those decisions as
three YAML files that are checked by code, so that a plan cannot quietly
refer to a character who was never designed or to a keyframe nobody has made.

| File | Type | Answers |
|---|---|---|
| `story_bible.yaml` | `StoryBible` | What is the story? What does it look like? |
| `character_bible.yaml` | `CharacterBible` | Who appears? Which reference images define them (by SHA-256)? |
| `shot_plan.yaml` | `ShotPlan` | Which scenes and shots? Shot type, camera, motion, duration, characters, keyframes, continuity. |

Code: `src/ai_studio/story/` (`schema.py` types, `loading.py` YAML,
`validation.py` cross-file checks, `convert.py` shot → request).
Tests: `tests/test_story.py`. Worked example: `stories/example_rooftop/`.

Suggested layout: one folder per story, `stories/<story_id>/`, holding the
three files. The loader does not enforce the folder name.

## Rules the layer enforces

1. **Draft until a human says ready.** The story bible, the shot plan and
   every shot have `status: draft | ready`, default `draft`. The converter
   refuses a draft shot (`DraftShotError`). This is the same rule as for
   benchmark definitions (`benchmarking/definitions.py`): readiness is an
   explicit human edit and is never inferred.
2. **Assets are identified by hash.** A reference asset is `available` with a
   64-character lowercase SHA-256, or `missing` with no hash. A keyframe
   requirement points to an asset by that hash (`asset_sha256`), or is itself
   marked `missing`. There is no third state, so the plan always says openly
   what does not exist yet.
3. **Unknown keys are errors.** A typo cannot silently drop a field.
4. **Durations add up.** Every shot is a whole number of frames at the plan's
   `fps`; a scene's `duration_seconds` equals the sum of its shots; the plan's
   `total_duration_seconds` equals the sum of its scenes.
5. **No shot without its keyframes.** The converter refuses a shot while one
   of its keyframes is missing (`MissingKeyframeError`). Generating without
   the planned start frame would produce a different shot under the same name.

## The files

All paths inside the files are relative to the project root, as everywhere in
this project (`ai_studio/paths.py`). The examples below are shortened; the
complete ones are in `stories/example_rooftop/`.

### story_bible.yaml

```yaml
story_id: example_rooftop          # required; usable as a file name
title: Example - the rooftop       # required
status: draft
logline: >-                        # required, one sentence
  A young man sits on a rooftop ledge at sunset and watches his city light up.
synopsis: ...                      # optional prose
themes: [stillness before something begins]
tone: quiet, cinematic
setting: a dense city seen from its rooftops, late sunset
world_rules:
  - the hero's face is never covered
visual_style:                      # required
  summary: Graphic novel style animated shot, hybrid 2D and 3D animation
  direction: [strong ink contours, controlled cel shading, ...]
  negative: [photorealistic, live action, ...]
  artist_profile: comic_finish_v1  # optional; configs/style_recipes/<name>.yaml
characters: [COMIC_HERO_01]        # ids that must exist in the character bible
originality_notes: ...
notes: ...
```

`visual_style.summary` opens every composed prompt, `direction` becomes the
request's `visual_direction`, and `negative` is the base of the negative
prompt. `artist_profile` is recorded for the post-processing step; this layer
does not check that the recipe file exists.

### character_bible.yaml

```yaml
characters:
  - id: COMIC_HERO_01
    name: The young man on the rooftop
    role: protagonist              # protagonist | antagonist | supporting | background
    description: >-                # required; one prompt-ready sentence
      A young man with curly dark hair, wearing dark hooded streetwear ...
    appearance: [young man, curly dark hair]
    wardrobe: [dark hooded streetwear with subtle dark-red accents, ...]
    never: [mask, logo, emblem]    # added to the negative prompt of his shots
    reference_assets:
      - id: rooftop_frame_v001
        kind: start_frame
        path: benchmarks/references/comic_hero_01/rooftop_frame_v001.png
        status: available
        sha256: 3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb
      - id: front
        kind: front
        path: benchmarks/references/comic_hero_01/front.png
        status: missing            # planned, not made: no sha256
```

Asset kinds: `start_frame`, `front`, `three_quarter`, `profile`, `back`,
`face_closeup`, `expression`, `style_frame`, `environment`, `prop`, `other`.

### shot_plan.yaml

```yaml
plan_id: example_rooftop_plan
story_id: example_rooftop          # must equal the story bible's story_id
status: draft
fps: 24                            # a whole number
resolution: {width: 1280, height: 704}
total_duration_seconds: 9
assets:                            # optional: shared, non-character assets
  - {id: city_sunset_plate, kind: environment, path: ..., status: missing}
scenes:
  - id: scene_01
    name: The rooftop at sunset
    location: a high rooftop over a dense city
    time_of_day: late sunset, city lights beginning to glow
    duration_seconds: 9
    shots:
      - id: shot_02                # unique in the whole plan
        name: The young man on the ledge
        shot_type: wide
        camera: slow cinematic push-in from a low three-quarter angle behind the character
        motion:
          - character sits on the rooftop ledge and remains mostly stationary
        duration_seconds: 5
        characters: [COMIC_HERO_01]
        action: >-                 # what happens, prompt-ready
          Seen from behind, he sits on the edge of the rooftop ledge ...
        keyframes:
          - id: start
            role: start_frame      # start_frame | reference
            description: the hero on the ledge from behind, frame 0
            character: COMIC_HERO_01
            status: available
            asset_sha256: 3a3e7675...e0fb
        continuity:
          - clothing and backpack exactly as in rooftop_frame_v001.png
        prompt: null               # optional full prompt; null = composed
        negative_prompt: null
        status: draft
```

Shot types: `establishing`, `wide`, `full`, `medium`, `medium_close_up`,
`close_up`, `extreme_close_up`, `over_the_shoulder`, `point_of_view`,
`two_shot`, `insert`.

`camera` takes the same shapes as the request's shot controls: a string, a
list of strings, or a mapping.

## Using it

```python
from ai_studio.story import (
    load_story_bible, load_character_bible, load_shot_plan,
    validate_story, missing_keyframes, verify_assets,
    shot_to_request_dict, request_from_dict,
)

story = load_story_bible("stories/example_rooftop/story_bible.yaml")
characters = load_character_bible("stories/example_rooftop/character_bible.yaml")
plan = load_shot_plan("stories/example_rooftop/shot_plan.yaml")

validate_story(plan, characters, story)   # [] when consistent, else every problem
missing_keyframes(plan)                   # the images still to be made
verify_assets(characters, plan)           # recorded hashes vs. files on this machine

data = shot_to_request_dict(plan, "shot_02", characters, story, seed=42, preview=True)
request = request_from_dict(data)         # a real VideoGenerationRequest
```

- `load_*` checks one file's shape and raises `ValueError` naming the file.
- `validate_story` checks across the files and returns a list of sentences;
  `check_story` raises `StoryValidationError` (with `.problems`) instead.
- `verify_assets` is the only function that reads media files. Media is
  git-ignored, so on a fresh clone every asset is reported as "not on this
  machine"; that is information, not a broken plan.
- `shot_to_request_dict` returns plain data whose keys are exactly the fields
  of `VideoGenerationRequest`; `output_resolution` is a `{width, height}`
  mapping so the dict can be saved. `preview=True` lifts the draft and
  missing-keyframe refusals, for reading a prompt before the shot is ready.

### How a prompt is composed

If a shot has its own `prompt`, it is used unchanged. Otherwise the prompt is
built, always in the same order: style summary · shot type · scene location
and time · each character's `description` · the shot's `action` · the camera
(when it is text) · the style `direction` keywords. For `shot_02` of the
example this gives:

> Graphic novel style animated shot, hybrid 2D and 3D animation. Wide shot. A
> high rooftop over a dense city, late sunset, city lights beginning to glow.
> A young man with curly dark hair, wearing dark hooded streetwear … Seen from
> behind, he sits on the edge of the rooftop ledge … Slow cinematic push-in
> from a low three-quarter angle behind the character. Strong ink contours,
> controlled cel shading, …

The negative prompt is the shot's own, or else the style's `negative` terms
plus every listed character's `never` terms, without repeats.

This composed text has **not** been tested against any model. Whether it
works as well as the hand-written benchmark prompt is an open question; until
that is measured, a shot can carry a hand-written `prompt`.

## How the director fills these files

The intended division of work: the human owns the story, the taste and every
`ready` decision; the director writes and maintains the three files and keeps
them consistent; the engine generates. The director never generates images here
(image generation is manual in the Gemini app for now) and never marks
anything `ready`.

1. **Brief → story bible.** From the human's brief, write `story_bible.yaml`:
   logline, synopsis, tone, setting, world rules, and the visual style in the
   project's existing vocabulary (the same phrases the benchmark prompts use,
   so results stay comparable). Fill `originality_notes`: no existing
   characters, titles, logos or costumes. Leave `status: draft`.
2. **Characters → character bible.** One entry per character. Write only
   what a reference image actually shows; where no image exists, say so in
   `notes` and list the needed images as `missing` assets (front,
   three-quarter, profile, face close-up). Do not describe a face that has
   never been drawn — an invented description becomes an invented face in
   every shot.
3. **Record assets by hash.** When the human supplies an image, compute its
   SHA-256, set `status: available` and the hash. Never type a hash from
   memory. Run `verify_assets` to confirm that file and record agree. A file
   that has been used in a run is never edited or replaced; a new version gets
   a new file name and a new entry.
4. **Story → shot plan.** Break the story into scenes and shots. For each
   shot choose the shot type, camera, motion and a whole-frame duration, list
   the characters, and write `action` as prompt-ready prose. Keep within what
   the current backend can do (about 5 seconds per shot at 24 fps,
   1280×704 — see `docs/BACKEND_WAN_TI2V_5B_TURBO_COMFYUI.md`).
5. **Keyframes.** Every shot that shows a character needs a `start_frame`
   keyframe showing that character, because the video model invents
   whatever the first frame does not fix. Point it at an available asset, or
   mark it `missing` with a precise description of the image to make. The
   list from `missing_keyframes(plan)` is then the image to-do list for the
   human.
6. **Continuity.** For each shot write what must match its neighbours: light
   direction, time of day, wardrobe, props, screen direction.
7. **Validate, then report.** Run `validate_story` until it returns nothing.
   Give the human: the plan, the list of missing keyframes, and the open
   decisions. Do not paper over a problem by deleting the check's trigger
   (for example by removing a character from a shot so that it validates).
8. **After the human marks shots ready**, `shot_to_request_dict` gives the
   request for each shot. Running it is a separate, explicit step.

Things the director must not do: set any `status` to `ready`; invent hashes,
measurements or file contents; describe features that no reference shows;
use names, likenesses or signature designs from existing franchises.

## Relation to the trailer shot list

`src/ai_studio/trailer/shotlist.py` and `trailers/trailer_01/shotlist.yaml`
do a different job from the story layer; the two were written independently:

| | Trailer shot list | Story layer |
|---|---|---|
| Purpose | **Assemble** a trailer: which finished clip plays for each shot, fades, title cards | **Plan** before generation: story, characters, assets by hash, keyframe requirements |
| Shot fields | `camera`, `motion`, `prompt`, `clip`, `needs_character_keyframe: true/false` | `shot_type`, `camera`, `motion`, `action`, `characters`, `keyframes` (by hash), `continuity` |
| Output | an edited video file | a `VideoGenerationRequest` per shot |

They overlap in `id`, `name`, `camera`, `motion`, `duration_seconds`,
`prompt`, `negative_prompt` and `status`, and both require whole-frame
durations. Nothing connects them yet. **Open decision:** either keep both and generate the trailer's shot
list entries from a shot plan, or fold the story layer's extra fields into
the trailer shot list. Until then, a trailer's shots would have to be kept in
step by hand in two files.

## Not done yet

- No command-line entry point.
- No link from a shot to the benchmark run or clip that realised it.
- No title-card or non-generated shots (the trailer shot list has these).
- No transitions between shots, no audio, no dialogue.
- The composed prompt is untested against a model (see above).
- Image requests: `generation/image.py` has an image-generation contract.
  Turning a `missing` keyframe into such a request is the natural next
  converter.
- Multi-keyframe conditioning (for example an end frame) has no role here,
  because no current backend declares support for it.
