# Prompt pack: character sheet for COMIC_HERO_01 (manual, Gemini app)

For making the views required by
[`benchmarks/definitions/character_sheet_hero_01.yaml`](../../benchmarks/definitions/character_sheet_hero_01.yaml)
by hand in the Gemini app, **one view per image**, from the canonical frame
`benchmarks/references/comic_hero_01/rooftop_frame_v001.png`.

Text only. Nothing here calls a model. The prompts are the fenced `text`
blocks; copy a block exactly, without changing a word, so that the prompt you
used is the prompt on record.

Original character. The prompts ask for nothing that belongs to an existing
character, title or costume.

Note: the benchmark draft describes ONE sheet image with five views. This pack
makes five single images instead, because that is what you asked for and what
works by hand. Which of the two the benchmark should fix is still open.

## Before you start

1. Use one fresh chat for the whole sheet. Pick the image model in the app
   and write down its name exactly as the app shows it (you need it for the
   import).
2. Attach the canonical frame to the first prompt.
3. Make the **front view first**. The canonical frame shows him from behind,
   so the front view is where his face is decided. Do not go on until you have
   a front view you accept.
4. For every later view, attach **two** images: the canonical frame and your
   accepted front view. From the third view on you may add one more accepted
   view (for the back view, the canonical frame matters most).
5. Download each result at full size. Do not crop, resize, convert or edit
   it. If the app adds a visible mark in a corner, leave it; the import
   reports it and you decide later.
6. If a result fails the checklist below, do not argue with it for long:
   after two corrections in the same chat, start a new chat with the same
   prompt and the same attachments.

## The two blocks every prompt shares

Every prompt below contains these two paragraphs word for word. They are
shown here once so you can see what is fixed.

Outfit:

> He has short curly dark hair, almost black, with volume on top, and a slim
> build. He wears a black hooded jacket with the hood down and subtle dark-red
> accents along the seams and cuffs, a black backpack on both shoulders with
> dark-red straps and small light buckles, dark loose cargo trousers with a
> side pocket, and white low-top sneakers with dark details and white laces.
> Dark red is the only accent colour. All clothing is plain fabric with no
> printed graphics, no writing and no symbols.

Style:

> Drawn in the same style as the rooftop reference image: a graphic novel
> illustration with strong black ink contours of varying line weight,
> controlled cel shading with only a few flat tones per surface, halftone dot
> texture in the shadows and hand-drawn hatching. Flat, even studio lighting
> instead of the sunset light of the reference. Not photorealistic, not a 3D
> render, no soft airbrushed gradients.

## View: front

Attach: the canonical frame.
Save the prompt as `front.prompt.txt`, the accepted image as `front.png`.

```text
Using the attached rooftop image as the reference for the character and for the drawing style, draw one full-body illustration of the same young man, seen from the front.

He stands upright in a neutral pose, facing the viewer directly, arms relaxed at his sides, feet slightly apart. The whole figure is visible from the top of his hair to the soles of his shoes, centred, with some empty space around him. His face is fully visible and uncovered: a young man of about twenty with a calm, neutral expression, dark eyes and dark eyebrows.

He has short curly dark hair, almost black, with volume on top, and a slim build. He wears a black hooded jacket with the hood down and subtle dark-red accents along the seams and cuffs, a black backpack on both shoulders with dark-red straps and small light buckles, dark loose cargo trousers with a side pocket, and white low-top sneakers with dark details and white laces. Dark red is the only accent colour. All clothing is plain fabric with no printed graphics, no writing and no symbols.

Background: plain flat light grey, nothing else in the picture, no scenery, no floor pattern, only a small soft contact shadow under the shoes.

Drawn in the same style as the rooftop reference image: a graphic novel illustration with strong black ink contours of varying line weight, controlled cel shading with only a few flat tones per surface, halftone dot texture in the shadows and hand-drawn hatching. Flat, even studio lighting instead of the sunset light of the reference. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single figure, one single view. Portrait format, 3:4.
```

## View: three_quarter_left

Attach: the canonical frame and the accepted `front.png`.
Save the prompt as `three_quarter_left.prompt.txt`, the image as `three_quarter_left.png`.

```text
Using the attached images as reference, draw one full-body illustration of the same young man as in the front view, with exactly the same face, hair, build and clothing, in a three-quarter view.

His whole body is turned 45 degrees toward his own left, so that he faces the right side of the picture and we see his front and his right side. He stands upright in a neutral pose, arms relaxed at his sides, feet slightly apart, looking straight ahead in the direction his body faces. The whole figure is visible from the top of his hair to the soles of his shoes, centred, at the same scale as the front view. His face is fully visible and uncovered, with the same calm, neutral expression.

He has short curly dark hair, almost black, with volume on top, and a slim build. He wears a black hooded jacket with the hood down and subtle dark-red accents along the seams and cuffs, a black backpack on both shoulders with dark-red straps and small light buckles, dark loose cargo trousers with a side pocket, and white low-top sneakers with dark details and white laces. Dark red is the only accent colour. All clothing is plain fabric with no printed graphics, no writing and no symbols.

Background: plain flat light grey, nothing else in the picture, no scenery, no floor pattern, only a small soft contact shadow under the shoes.

Drawn in the same style as the rooftop reference image: a graphic novel illustration with strong black ink contours of varying line weight, controlled cel shading with only a few flat tones per surface, halftone dot texture in the shadows and hand-drawn hatching. Flat, even studio lighting instead of the sunset light of the reference. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single figure, one single view. Portrait format, 3:4.
```

## View: three_quarter_right

Attach: the canonical frame and the accepted `front.png`.
Save the prompt as `three_quarter_right.prompt.txt`, the image as `three_quarter_right.png`.

```text
Using the attached images as reference, draw one full-body illustration of the same young man as in the front view, with exactly the same face, hair, build and clothing, in a three-quarter view.

His whole body is turned 45 degrees toward his own right, so that he faces the left side of the picture and we see his front and his left side. He stands upright in a neutral pose, arms relaxed at his sides, feet slightly apart, looking straight ahead in the direction his body faces. The whole figure is visible from the top of his hair to the soles of his shoes, centred, at the same scale as the front view. His face is fully visible and uncovered, with the same calm, neutral expression.

He has short curly dark hair, almost black, with volume on top, and a slim build. He wears a black hooded jacket with the hood down and subtle dark-red accents along the seams and cuffs, a black backpack on both shoulders with dark-red straps and small light buckles, dark loose cargo trousers with a side pocket, and white low-top sneakers with dark details and white laces. Dark red is the only accent colour. All clothing is plain fabric with no printed graphics, no writing and no symbols.

Background: plain flat light grey, nothing else in the picture, no scenery, no floor pattern, only a small soft contact shadow under the shoes.

Drawn in the same style as the rooftop reference image: a graphic novel illustration with strong black ink contours of varying line weight, controlled cel shading with only a few flat tones per surface, halftone dot texture in the shadows and hand-drawn hatching. Flat, even studio lighting instead of the sunset light of the reference. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single figure, one single view. Portrait format, 3:4.
```

## View: profile

Attach: the canonical frame and the accepted `front.png` (optionally also an accepted three-quarter view).
Save the prompt as `profile.prompt.txt`, the image as `profile.png`.

```text
Using the attached images as reference, draw one full-body illustration of the same young man as in the front view, with exactly the same face, hair, build and clothing, in an exact side view.

He stands in strict profile, his body and head turned 90 degrees so that he faces the left edge of the picture. We see only one side of him: one eye, the outline of his nose, lips and chin, one arm, and the backpack on his back in side view. He stands upright in a neutral pose, arms relaxed at his sides, feet together. The whole figure is visible from the top of his hair to the soles of his shoes, centred, at the same scale as the front view. His face is uncovered, with the same calm, neutral expression.

He has short curly dark hair, almost black, with volume on top, and a slim build. He wears a black hooded jacket with the hood down and subtle dark-red accents along the seams and cuffs, a black backpack on both shoulders with dark-red straps and small light buckles, dark loose cargo trousers with a side pocket, and white low-top sneakers with dark details and white laces. Dark red is the only accent colour. All clothing is plain fabric with no printed graphics, no writing and no symbols.

Background: plain flat light grey, nothing else in the picture, no scenery, no floor pattern, only a small soft contact shadow under the shoes.

Drawn in the same style as the rooftop reference image: a graphic novel illustration with strong black ink contours of varying line weight, controlled cel shading with only a few flat tones per surface, halftone dot texture in the shadows and hand-drawn hatching. Flat, even studio lighting instead of the sunset light of the reference. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single figure, one single view. Portrait format, 3:4.
```

## View: back

Attach: the canonical frame and the accepted `front.png` (optionally also the accepted profile).
Save the prompt as `back.prompt.txt`, the image as `back.png`.

```text
Using the attached images as reference, draw one full-body illustration of the same young man as in the front view, with exactly the same hair, build and clothing, seen from directly behind.

He stands upright with his back to the viewer, in a neutral pose, arms relaxed at his sides, feet slightly apart, head facing away so that only the back of his curly hair is visible. The backpack sits on both shoulders exactly as in the rooftop reference image, with the hood of the jacket lying down over its top edge. The whole figure is visible from the top of his hair to the soles of his shoes, centred, at the same scale as the front view.

He has short curly dark hair, almost black, with volume on top, and a slim build. He wears a black hooded jacket with the hood down and subtle dark-red accents along the seams and cuffs, a black backpack on both shoulders with dark-red straps and small light buckles, dark loose cargo trousers with a side pocket, and white low-top sneakers with dark details and white laces. Dark red is the only accent colour. All clothing is plain fabric with no printed graphics, no writing and no symbols.

Background: plain flat light grey, nothing else in the picture, no scenery, no floor pattern, only a small soft contact shadow under the shoes.

Drawn in the same style as the rooftop reference image: a graphic novel illustration with strong black ink contours of varying line weight, controlled cel shading with only a few flat tones per surface, halftone dot texture in the shadows and hand-drawn hatching. Flat, even studio lighting instead of the sunset light of the reference. Not photorealistic, not a 3D render, no soft airbrushed gradients.

One single figure, one single view. Portrait format, 3:4.
```

## Checklist: accept or reject each result

Look at the image at full size. Reject it if any line below is a "no".

Identity
- [ ] It is one figure, one view, the view that was asked for (front / turned to HIS left / turned to HIS right / strict profile facing the left edge / back).
- [ ] Same person as the accepted front view: same face shape, eyes, eyebrows, nose, skin tone. (Front view: a face you are willing to keep for the whole trailer.)
- [ ] Same hair: short, curly, almost black, volume on top. Not longer, not straighter, not a different cut.
- [ ] Same build and proportions; same height in the frame as the other views.

Outfit
- [ ] Black hooded jacket, hood DOWN, dark-red accents only along seams and cuffs.
- [ ] Black backpack on both shoulders, dark-red straps, small light buckles (two straps visible from the front, the pack itself from the side and back).
- [ ] Dark loose cargo trousers with a side pocket.
- [ ] White low-top sneakers with dark details and white laces.
- [ ] No other accent colour than dark red. Nothing added: no hat, gloves, jewellery, headphones, scarf, belt bag.
- [ ] Nothing on the clothing: no logo, no emblem, no letter or number, no symbol, no pattern. Check the chest, the back of the jacket and the backpack.
- [ ] Face uncovered: no mask, no face covering, no goggles, hood not up.

Style
- [ ] Reads as the same hand as the canonical frame: clear black ink contours with varying weight.
- [ ] Cel shading with few flat tones; halftone dots or hatching in the shadows.
- [ ] The FACE is drawn like the body: inked lines and flat tones, not a soft, smooth, rendered face on an inked body.
- [ ] Not photorealistic, not a 3D render, no airbrushed gradients.
- [ ] Even lighting; no sunset colours on the figure.

Drawing
- [ ] Whole figure in the picture, head and shoes not cut off.
- [ ] Two arms, two legs, five fingers per visible hand, feet pointing the right way for the view.
- [ ] Clean lines: no smeared, doubled or broken contours.

Picture
- [ ] Plain light-grey background, nothing else in it.
- [ ] No text, caption, label, signature or frame anywhere in the picture.
- [ ] Nothing that reminds you of an existing character, title or costume.
- [ ] Note for yourself whether the app left a visible mark in a corner (do not crop it now).

When all five views are accepted, compare them side by side once more for
the identity and outfit lines, then import them.

## Importing an accepted image

One command per accepted image; it copies the file unchanged and records
where it came from:

```
python scripts/import_reference.py "C:\path\to\download.png" --asset-set comic_hero_01 ^
    --name front.png --tool "Gemini app" --model "<model name exactly as the app shows it>" ^
    --prompt-file front.prompt.txt --date 2026-10-06 ^
    --input-reference benchmarks/references/comic_hero_01/rooftop_frame_v001.png
```

For the later views add a second
`--input-reference benchmarks/references/comic_hero_01/front.png` (and any
other view you attached). If you corrected a result with follow-up messages in
the chat, put those messages into `--notes`, in order: the prompt file alone
would no longer be the whole truth.
