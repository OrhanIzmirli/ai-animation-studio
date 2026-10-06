# Style LoRA dataset: what to gather (retro 1980s western animation)

For a local style LoRA on SDXL base, trained like the character LoRA
(`spikes/train_hero_lora_spike.py`, ComfyUI's `TrainLoraNode`).

## Rights (the original-material rule)

Every image must be one of:
- drawn or painted by you;
- commissioned, with a written note that it may be used for training;
- published under a licence that allows this use (CC0, public domain, or
  CC-BY with commercial use allowed), with the source link kept;
- made by you in another AI tool whose terms allow it (record tool, model
  and prompt, as for the character sheet).

Not allowed: screenshots, frames or background art from existing shows or
films, fan art of existing characters, images found by search with no
licence. A model trained on them learns that show, which is exactly what the
rule forbids.

## Images

| | |
|---|---|
| Count | 15 minimum, 20–30 better |
| Format | PNG preferred; high-quality JPG (little compression) also fine. WEBP accepted |
| Size | short side at least 1024 px (768 px is the floor). Any aspect ratio: I crop to 1024x1024 |
| Content mix | about half environments (exteriors, interiors, night and day), about a third characters (several different ones, full body and half body), the rest props and close-ups |
| Must share | the look you want learned: rough, uneven ink outlines, painted backgrounds, muted palette, film grain |
| Must vary | subjects, colours, lighting, framing, so that only the style is common to all |
| Avoid | text, logos, signatures, watermarks, borders, speech bubbles, UI, letterbox bars; near-duplicates; heavy JPEG blocks or blur (the LoRA learns those too). Film grain is fine |

Keep the subject important to the composition near the centre: the 1024x1024
square crop may cut the edges of wide pictures. If a picture must not be
cropped, say so and I will pad it instead.

## Folder

Put the originals here, untouched:

```
benchmarks/references/style_retro80s/
  originals/
    001_house_on_cliff.png
    002_kitchen_night.jpg
    ...
  SOURCES.md     one line per file: file name, who made it, licence / permission, link or tool + prompt
```

Name files `NNN_short_description.ext` (three digits, lower case, no spaces).
The description helps me write captions; it does not have to be long.

I then import each original with its provenance, write the 1024x1024 crops
and one caption file per image (same name, `.txt`) to
`spikes/output/style_retro80s_dataset/` and ComfyUI's `input/` folder, plus a
contact sheet for you to check before training. Captions describe the
content only ("a wooden house on a rocky cliff at night, lightning") with a
trigger word in front; the style is what the trigger word comes to mean.

## Then

Training as for the character LoRA: rank 8, learning rate 1e-4, in parts of
100 steps (about 7 minutes each on this laptop), checkpoints kept per part,
then a grid of strengths and checkpoints on fixed prompts and seeds.
