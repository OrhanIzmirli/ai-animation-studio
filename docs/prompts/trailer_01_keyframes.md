# Prompt pack: keyframes for trailer_01 (manual, Gemini app)

One keyframe prompt for every shot of
[`trailers/trailer_01/shotlist.yaml`](../../trailers/trailer_01/shotlist.yaml)
that has `needs_character_keyframe: true`. A keyframe is the still picture a
shot starts from (the start frame of an image-to-video run). For two shots an
optional END keyframe is given as well, for models that take a first and a
last frame.

Text only. Nothing here calls a model. The prompts are the fenced `text`
blocks; copy a block exactly.

The character reference is the approved character sheet, one image with three
views (front, three-quarter, back):
`benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg`.
Every prompt refers to it; it defines his face, hair and outfit, and the
drawing style. The style paragraphs follow `visual_direction` in
`benchmarks/definitions/character_sheet_hero_01.yaml`, adapted to a wide
cinematic frame and to the light of each scene.

Original character and original story. The prompts ask for nothing that
belongs to an existing character, title or costume.

## Before you start

1. One fresh chat per shot.
2. Attach the images named under "Attach" for that shot: the character sheet
   `benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg`
   (face, hair, outfit, drawing style) and the canonical rooftop frame
   `benchmarks/references/comic_hero_01/rooftop_frame_v001.png` (environment,
   lighting, city). The sheet carries handwritten notes, colour swatches and
   a small shoe drawing; the prompts tell the model to use only the three
   figures.
3. Ask for 16:9. The video runs are 1280x704; the app's 16:9 size is close to
   that and the video backend resizes the start frame and records how.
4. Download at full size. Do not crop, resize, convert or edit. Leave any
   visible corner mark; the import reports it.
5. Check each result with the checklist at the end, then import it with
   `scripts/import_reference.py` into the asset set `trailer_01_keyframes`
   (example at the end).

## Shots and what they need

| Shot | Name | Character keyframe |
|---|---|---|
| `shot_01` | The city at sunset | not needed (no character) |
| `shot_02` | The rooftop | already exists: the canonical frame `benchmarks/references/comic_hero_01/rooftop_frame_v001.png` |
| `shot_03` | His eyes | prompt below |
| `shot_04` | The red glow | prompt below (+ optional end keyframe) |
| `shot_05` | The lights go out | not needed (no character) |
| `shot_06` | He stands | prompt below (+ optional end keyframe) |
| `shot_07` | Title card | not needed (made in editing) |

## Shot: shot_02

No prompt. This shot starts from the canonical frame, which already exists
and must not be regenerated or edited.

## Shot: shot_03

His eyes, extreme close-up, 4 s. The shot: city lights shimmer in his eyes,
one slow blink.

Attach: `benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg` (face, hair, outfit) and `benchmarks/references/comic_hero_01/rooftop_frame_v001.png` (environment, lighting, city).
Save the prompt as `shot_03_start.prompt.txt`, the image as `shot_03_start.png`.

```text
Using the attached character sheet as the reference for the young man's face and hair and for the drawing style, and the attached rooftop image as the reference for the city and the light, draw one wide cinematic frame. On the character sheet use only the three figures; ignore its handwritten notes, colour swatches and the small shoe drawing.

Extreme close-up of his eyes, filling the frame from just above his eyebrows to the bridge of his nose and from temple to temple. It is the same young man as in the character sheet: the same eye shape, the same dark eyes and dark eyebrows, the same skin tone, with a few short curls of his almost black hair falling over his forehead at the top edge of the frame. His face is fully visible and uncovered. He looks slightly past the camera into the distance with a calm, watchful expression, eyes open.

The last warm light of a late sunset falls on his face from one side; the other side is in cool violet shadow. In each dark eye, the lights of a distant city at dusk are reflected as a small cluster of tiny bright warm points.

Drawn as hand-made graphic novel concept art, in the same hand as the attached character sheet: hand-inked black ink contours with naturally varying, pressure-dependent line weight, thick outer contours and thinner interior details. The sunset is a real directional light: clear highlights on the lit side of his face, core shadows on the other, and a little reflected light inside the shadows; not flat, not even. Bold flat cel shading blocks, coarse halftone screen-tone dots in the mid-tones, rough hand-drawn hatching in the shadows. Faint construction lines of the under-sketch show through in a few places. The level of finish varies on purpose: his eyes are rendered most tightly, the skin, eyebrows and curls around them stay looser and more gestural. His skin is drawn with the same inked lines and flat tones as everything else. Not photorealistic, not a 3D render, no smooth vector look, no soft airbrushed gradients.

One single picture without borders or captions. Wide format, 16:9.
```

## Shot: shot_04

The red glow, close-up of his hand on the ledge, 4 s. The shot: his fingers
tighten, then a soft red glow appears under his palm and spreads along the
cracks in the concrete. The START keyframe therefore has NO glow yet.

Attach: `benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg` (outfit: sleeve and cuff, skin tone) and `benchmarks/references/comic_hero_01/rooftop_frame_v001.png` (environment, lighting, city: the ledge).
Save the prompt as `shot_04_start.prompt.txt`, the image as `shot_04_start.png`.

```text
Using the attached character sheet as the reference for the young man's clothing and skin tone and for the drawing style, and the attached rooftop image as the reference for the rooftop ledge, the city and the light, draw one wide cinematic frame. On the character sheet use only the three figures; ignore its handwritten notes, colour swatches and the small shoe drawing.

Close-up, from a low angle at the level of the ledge, of his bare right hand resting on the edge of a concrete rooftop ledge. The hand comes into the frame from the left: five fingers curled over the edge of the weathered concrete, relaxed, not yet gripping hard. At his wrist we see the end of the black sleeve of his hooded jacket with its subtle dark-red cuff. The concrete has fine cracks and a rough, stained surface. Behind the ledge, far below and out of focus, lies the dense city at dusk with small warm lights.

The light is the cool blue-violet of early dusk with a last trace of warm sunset on the edge of the hand. There is no glow and no other light source on the ledge.

Drawn as hand-made graphic novel concept art, in the same hand as the attached character sheet: hand-inked black ink contours with naturally varying, pressure-dependent line weight, thick outer contours and thinner interior details. The dusk light is directional: a clear highlight on the edge of the hand, core shadows between the fingers and under the palm, and a little reflected light from the concrete; not flat, not even. Bold flat cel shading blocks, coarse halftone screen-tone dots in the mid-tones, rough hand-drawn hatching in the shadows. Faint construction lines of the under-sketch show through in a few places. The level of finish varies on purpose: the hand and the edge of the ledge are rendered most tightly, the city behind them stays looser and more gestural. Not photorealistic, not a 3D render, no smooth vector look, no soft airbrushed gradients.

One single picture without borders or captions. Wide format, 16:9.
```

Optional END keyframe (only for a model that takes a first and a last frame).
Make it in the SAME chat, right after you accepted the start keyframe, so the
hand, the ledge and the framing stay the same.

Attach: your accepted `shot_04_start.png`.
Save the prompt as `shot_04_end.prompt.txt`, the image as `shot_04_end.png`.

```text
Using the attached picture as the exact starting point, draw the same frame a few seconds later. Keep the camera position, the framing, the hand, the sleeve, the ledge, the city and the drawing style exactly as they are.

Change only this: his fingers now grip the edge of the concrete firmly, the knuckles slightly raised, and a soft red glow has appeared under his palm. The glow spreads outward from under the hand along the fine cracks in the concrete as thin glowing red lines, and lights his fingers and the cuff of his sleeve softly from below. The glow is calm and even, like embers under the surface. A little dust lifts from the ledge near the hand. The city in the background is unchanged.

Same hand-made graphic novel style as the starting picture: hand-inked black ink contours with naturally varying line weight, directional light with real highlights, core shadows and reflected light, bold flat cel shading blocks, coarse halftone screen-tone dots in the mid-tones, rough hand-drawn hatching, faint under-sketch lines in a few places, a deliberately varied level of finish. The glow is drawn as flat red tones with a hard edge and a halftone falloff. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single picture without borders or captions. Wide format, 16:9.
```

## Shot: shot_06

He stands, medium shot from behind, 5 s. The shot: he rises from the ledge,
pulls his hood up with one hand, and holds still as a silhouette against the
moon. The START keyframe shows him beginning to rise, hood still DOWN.

Attach: `benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg` (face, hair, outfit; the back view matters most) and `benchmarks/references/comic_hero_01/rooftop_frame_v001.png` (environment, lighting, city).
Save the prompt as `shot_06_start.prompt.txt`, the image as `shot_06_start.png`.

```text
Using the attached character sheet as the reference for the young man, his hair and his clothing and for the drawing style, and the attached rooftop image as the reference for the rooftop ledge and the city, draw one wide cinematic frame. On the character sheet use only the three figures, above all the back view; ignore its handwritten notes, colour swatches and the small shoe drawing.

Medium shot from behind and slightly below. The same young man is on the rooftop ledge, caught in the first moment of getting up: one foot already flat on the ledge, one knee still bent low, one hand pushing off the concrete, his back to the camera. We see him from the knees up. His hood is down, so the back of his short curly, almost black hair is visible.

He wears a black hooded jacket with the hood down and subtle dark-red accents along the seams and cuffs, a black backpack on both shoulders with dark-red straps and small light buckles, dark loose cargo trousers with a side pocket, and white low-top sneakers with dark details and white laces. Dark red is the only accent colour. All clothing is plain fabric with no printed graphics, no writing and no symbols.

Beyond the ledge lies the same dense city as in the rooftop reference, but now it is night and every window and street light is dark: the buildings are only dark shapes with thin pale edges. A large full moon hangs low in a deep blue sky directly behind him and is the only light, giving his figure a thin cold rim of light. A light wind moves the hem of his jacket.

Drawn as hand-made graphic novel concept art, in the same hand as the attached character sheet: hand-inked black ink contours with naturally varying, pressure-dependent line weight, thick outer contours and thinner interior details. The moon is a real directional light from behind: a thin bright rim highlight on his figure, deep core shadows on the side facing us, and a little reflected light from the concrete; not flat, not even. Bold flat cel shading blocks, coarse halftone screen-tone dots in the mid-tones, rough hand-drawn hatching in the shadows. Faint construction lines of the under-sketch show through in a few places. The level of finish varies on purpose: his figure is rendered most tightly, the dark city and the sky stay looser and more gestural. Not photorealistic, not a 3D render, no smooth vector look, no soft airbrushed gradients.

One single picture without borders or captions. Wide format, 16:9.
```

Optional END keyframe (only for a model that takes a first and a last frame).
Make it in the SAME chat, right after you accepted the start keyframe.

Attach: your accepted `shot_06_start.png`.
Save the prompt as `shot_06_end.prompt.txt`, the image as `shot_06_end.png`.

```text
Using the attached picture as the exact starting point, draw the same frame a few seconds later. Keep the camera position, the framing, the rooftop ledge, the dark city, the full moon, the sky and the drawing style exactly as they are.

Change only this: he now stands fully upright on the ledge with his back to the camera, feet apart, arms relaxed at his sides, and he has pulled the hood of his jacket up over his hair. He stands still, a dark figure outlined against the large full moon, with a thin cold rim of moonlight along his shoulders, his hood and his backpack. The backpack with its dark-red straps is still on his back, and the dark-red accents of the jacket are just visible in the rim light. The wind moves the hem of his jacket. We still see him from behind; his face is not shown.

Same hand-made graphic novel style as the starting picture: hand-inked black ink contours with naturally varying line weight, directional light with real highlights, core shadows and reflected light, bold flat cel shading blocks, coarse halftone screen-tone dots in the mid-tones, rough hand-drawn hatching, faint under-sketch lines in a few places, a deliberately varied level of finish. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single picture without borders or captions. Wide format, 16:9.
```

## Checklist: accept or reject each keyframe

Reject the image if any line is a "no".

- [ ] It shows the moment the shot STARTS from (shot_04: no glow yet; shot_06: rising, hood down). An end keyframe shows the end state and nothing else has changed.
- [ ] Same young man as the accepted character sheet: face (shot_03), skin tone, hair.
- [ ] Outfit as on the sheet, as far as it is visible: black hooded jacket with dark-red seams and cuffs, black backpack with dark-red straps, dark cargo trousers, white low-top sneakers.
- [ ] Nothing on the clothing: no logo, no emblem, no letter, no symbol. Face uncovered wherever it is visible: no mask, no goggles.
- [ ] Same hand as the approved character sheet: hand-inked contours with varying line weight, directional light with real highlights and core shadows (not flat), cel shading blocks, halftone dots in the mid-tones, hatching, faint under-sketch lines in places, a varied level of finish. Skin drawn like everything else, not a soft rendered face or hand.
- [ ] Hands: five fingers, plausible grip (shot_04 especially).
- [ ] The city and the ledge look like the canonical frame's city and ledge.
- [ ] Light as described for the shot (shot_03 sunset from one side; shot_04 dusk; shot_06 moon only, city dark).
- [ ] 16:9, nothing important at the very edge (the video backend resizes the frame slightly).
- [ ] No text, caption, speech bubble, signature or frame in the picture. In particular nothing copied from the notes, swatches or shoe drawing on the character sheet.
- [ ] Nothing that reminds you of an existing character, title or costume. In particular the red glow is a plain glow in cracked concrete, not a pattern, and the moonlit figure wears ordinary streetwear.
- [ ] End keyframes only: put start and end side by side. Camera, framing and background must match; only the described change differs.

## Importing an accepted keyframe

```
python scripts/import_reference.py "C:\path\to\download.png" --asset-set trailer_01_keyframes ^
    --name shot_03_start.png --tool "Gemini app" --model "<model name exactly as the app shows it>" ^
    --prompt-file shot_03_start.prompt.txt --date 2026-10-06 ^
    --input-reference benchmarks/references/comic_hero_01/character_turnaround_hero_01.jpg ^
    --input-reference benchmarks/references/comic_hero_01/rooftop_frame_v001.png
```

List every image you attached as an `--input-reference`. Follow-up
corrections you typed in the chat go into `--notes`, in order.
