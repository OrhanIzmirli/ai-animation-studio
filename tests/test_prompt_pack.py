"""The manual prompt packs in docs/prompts/ must stay in step with the
character sheet benchmark and the trailer shot list."""

import re
from pathlib import Path

import pytest

from ai_studio.benchmarking import load_definition
from ai_studio.trailer.shotlist import load_shotlist

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "docs" / "prompts"
SHEET = PROMPTS / "character_sheet_hero_01.md"
KEYFRAMES = PROMPTS / "trailer_01_keyframes.md"
# Nothing a prompt ASKS for may contain these (checklists may name them as things to reject).
BANNED = ("spider", "web", "mask", "letter", "emblem", "logo", "superhero", "costume", "cape", "armor", "weapon")
STYLE_TERMS = ("graphic novel", "ink contours", "cel shading", "halftone", "hatching")
# The style of the approved turnaround sheet (character_sheet_hero_01.yaml, visual_direction).
APPROVED_STYLE_TERMS = (
    "line weight", "highlight", "core shadows", "reflected light", "halftone screen-tone dots in the mid-tones",
    "under-sketch", "level of finish",
)
OUTFIT_TERMS = (
    "short curly dark hair", "black hooded jacket", "dark-red accents", "black backpack", "dark-red straps",
    "cargo trousers", "white low-top sneakers",
)


def sections(path, heading):
    """{name: section text} for every '## <heading>: <name>' section of a file."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(rf"^## {heading}: (\S+)\s*$", text, flags=re.MULTILINE)
    found = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        found[name] = re.split(r"^## ", body, maxsplit=1, flags=re.MULTILINE)[0]
    return found


def prompt_blocks(text):
    return re.findall(r"^```text\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)


def all_prompts(path):
    return prompt_blocks(path.read_text(encoding="utf-8"))


# --- character sheet ---

@pytest.fixture(scope="module")
def definition():
    return load_definition(REPO_ROOT / "benchmarks" / "definitions" / "character_sheet_hero_01.yaml")


def test_one_prompt_per_view_of_the_superseded_pack(definition):
    views = sections(SHEET, "View")
    # The pack is superseded and kept for reference: it still holds the five
    # views of the earlier plan, while the benchmark now asks for three.
    assert list(views) == ["front", "three_quarter_left", "three_quarter_right", "profile", "back"]
    assert definition.composition["views"] == ["front", "three_quarter", "back"]
    assert "SUPERSEDED" in SHEET.read_text(encoding="utf-8")
    for name, body in views.items():
        assert len(prompt_blocks(body)) == 1, name
        assert f"`{name}.png`" in body and f"`{name}.prompt.txt`" in body, name
    assert len(all_prompts(SHEET)) == 5


def test_every_view_prompt_fixes_the_same_outfit_and_style():
    prompts = all_prompts(SHEET)

    def paragraph(prompt, start):
        (found,) = [p for p in prompt.split("\n\n") if p.startswith(start)]
        return found

    outfits = {paragraph(prompt, "He has short curly dark hair") for prompt in prompts}
    styles = {paragraph(prompt, "Drawn in the same style") for prompt in prompts}
    assert len(outfits) == 1 and len(styles) == 1  # word for word identical in all five
    (outfit,), (style,) = outfits, styles
    for term in OUTFIT_TERMS:
        assert term in outfit, term
    assert "hood down" in outfit and "no printed graphics, no writing and no symbols" in outfit
    for term in STYLE_TERMS:
        assert term in style, term
    assert "Not photorealistic" in style and "not a 3D render" in style


def test_view_prompts_ask_for_one_view_on_a_neutral_background():
    for prompt in all_prompts(SHEET):
        assert "plain flat light grey" in prompt and "no scenery" in prompt
        assert "One single figure, one single view." in prompt
        assert "full-body" in prompt and "rooftop" in prompt  # the canonical frame is the reference
    views = sections(SHEET, "View")
    assert "seen from the front" in views["front"]
    assert "toward his own left" in views["three_quarter_left"]
    assert "toward his own right" in views["three_quarter_right"]
    assert "strict profile" in views["profile"] and "left edge" in views["profile"]
    assert "from directly behind" in views["back"]
    for name in ("three_quarter_left", "three_quarter_right", "profile", "back"):
        assert "same young man as in the front view" in views[name], name


def test_sheet_outfit_agrees_with_the_benchmark(definition):
    details = " ".join(str(value) for value in definition.subject_details.values()).lower()
    for term in ("curly dark hair", "black hooded jacket", "black backpack", "cargo trousers", "white low-top sneakers"):
        assert term in details, term


def test_sheet_has_an_acceptance_checklist():
    text = SHEET.read_text(encoding="utf-8")
    checklist = text.split("## Checklist", 1)[1].split("\n## ", 1)[0]
    boxes = re.findall(r"^- \[ \] (.+)$", checklist, flags=re.MULTILINE)
    assert len(boxes) >= 15
    joined = " ".join(boxes).lower()
    for topic in ("same person", "hood down", "backpack", "sneakers", "logo", "mask", "ink contours",
                  "five fingers", "light-grey background", "existing character"):
        assert topic in joined, topic
    assert "soft, smooth, rendered face on an inked body" in joined


# --- trailer keyframes ---

@pytest.fixture(scope="module")
def shotlist():
    return load_shotlist(REPO_ROOT / "trailers" / "trailer_01" / "shotlist.yaml")


def test_every_shot_that_needs_a_keyframe_has_a_section(shotlist):
    needed = [shot.id for shot in shotlist.shots if shot.needs_character_keyframe]
    shots = sections(KEYFRAMES, "Shot")
    assert list(shots) == needed == ["shot_02", "shot_03", "shot_04", "shot_06"]
    # shot_02 starts from the canonical frame: no prompt, nothing is regenerated
    assert prompt_blocks(shots["shot_02"]) == [] and "rooftop_frame_v001.png" in KEYFRAMES.read_text(encoding="utf-8")
    assert len(prompt_blocks(shots["shot_03"])) == 1
    assert len(prompt_blocks(shots["shot_04"])) == 2 and len(prompt_blocks(shots["shot_06"])) == 2  # start + optional end
    for shot_id in ("shot_03", "shot_04", "shot_06"):
        assert f"`{shot_id}_start.png`" in shots[shot_id], shot_id


def test_keyframe_prompts_refer_to_the_accepted_character_sheet_and_the_style():
    shots = sections(KEYFRAMES, "Shot")
    for shot_id in ("shot_03", "shot_04", "shot_06"):
        start = prompt_blocks(shots[shot_id])[0]
        assert "attached character sheet" in start and "use only the three figures" in start, shot_id
        assert "rooftop" in start and "16:9" in start, shot_id
        for term in STYLE_TERMS + APPROVED_STYLE_TERMS:
            assert term in start, (shot_id, term)
        assert "few flat tones" not in start and "same style as the rooftop" not in start, shot_id
        # the approved turnaround sheet and the canonical frame, not the per-view images of the old plan
        assert re.search(
            r"Attach: `benchmarks/references/comic_hero_01/character_turnaround_hero_01\.jpg`.*"
            r"`benchmarks/references/comic_hero_01/rooftop_frame_v001\.png`", shots[shot_id],
        ), shot_id
        assert not re.search(r"`(front|back|three_quarter_left|three_quarter_right)\.png`", shots[shot_id]), shot_id
    for shot_id in ("shot_04", "shot_06"):
        end = prompt_blocks(shots[shot_id])[1]
        assert "exact starting point" in end and "Change only this" in end, shot_id
        assert "16:9" in end
        for term in APPROVED_STYLE_TERMS:
            assert term in end, (shot_id, term)


def test_shot_list_uses_the_approved_style_and_the_latest_profile(shotlist):
    assert shotlist.artist_profile == "comic_finish_v3"
    for shot in shotlist.shots:
        if shot.source != "generated" or shot.id == "shot_02":  # shot_02 keeps the frozen benchmark prompt
            continue
        for term in APPROVED_STYLE_TERMS:
            assert term in shot.prompt, (shot.id, term)
        assert "controlled cel shading" not in shot.prompt, shot.id


def test_keyframes_show_the_start_of_each_shot(shotlist):
    shots = sections(KEYFRAMES, "Shot")
    eyes, hand, stands = (prompt_blocks(shots[name]) for name in ("shot_03", "shot_04", "shot_06"))
    assert "Extreme close-up of his eyes" in eyes[0] and "reflected" in eyes[0]
    assert "There is no glow" in hand[0] and "red glow has appeared" in hand[1]
    assert "hood is down" in stands[0] and "pulled the hood of his jacket up" in stands[1]
    assert "full moon" in stands[0] and "full moon" in stands[1]
    for term in OUTFIT_TERMS[1:]:
        assert term in stands[0], term
    by_id = {shot.id: shot for shot in shotlist.shots}
    assert "eyes" in by_id["shot_03"].camera and "hand" in by_id["shot_04"].camera


def test_keyframes_have_an_acceptance_checklist():
    text = KEYFRAMES.read_text(encoding="utf-8")
    boxes = re.findall(r"^- \[ \] (.+)$", text.split("## Checklist", 1)[1], flags=re.MULTILINE)
    assert len(boxes) >= 10
    joined = " ".join(boxes).lower()
    for topic in ("character sheet", "five fingers", "no glow yet", "existing character", "16:9"):
        assert topic in joined, topic


# --- both packs ---

@pytest.mark.parametrize("path", [SHEET, KEYFRAMES], ids=["sheet", "keyframes"])
def test_prompts_ask_for_no_franchise_motif(path):
    prompts = all_prompts(path)
    assert prompts
    for prompt in prompts:
        lowered = prompt.lower()
        for word in BANNED:
            assert word not in lowered, (word, prompt[:60])


@pytest.mark.parametrize("path", [SHEET, KEYFRAMES], ids=["sheet", "keyframes"])
def test_packs_are_plain_text_instructions(path):
    text = path.read_text(encoding="utf-8")
    assert text.isascii()  # copy-paste safe
    assert "scripts/import_reference.py" in text
    assert "Do not crop" in text
    for prompt in prompt_blocks(text):
        assert "\n\n\n" not in prompt and prompt == prompt.strip() + "\n"
