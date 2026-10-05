"""Importing hand-made images as reference assets with provenance."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.protection import is_read_only
from ai_studio.references import (
    CORNERS,
    ReferenceImportError,
    find_corner_marks,
    import_reference,
    load_provenance,
    provenance_path,
    verify_reference,
)

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
PROMPT = "Front view of the young man from the reference image.\nPlain light-grey background.  Two spaces kept."
W, H = 640, 360


# --- synthetic pictures ---

def picture(seed=0):
    """A soft, busy-enough picture with no mark in any corner."""
    y, x = np.mgrid[0:H, 0:W]
    base = 90 + 50 * np.sin(x / 70.0 + seed) + 40 * np.cos(y / 50.0)
    image = np.stack([base, base * 0.9 + 10, base * 0.8 + 30], axis=2)
    return np.clip(image, 0, 255).astype(np.uint8)


def add_sparkle(image, corner="bottom_right", size=26, margin=14, value=245):
    """Draw a four-point star (a "sparkle") near a corner."""
    image = image.copy()
    height, width = image.shape[:2]
    cy = height - margin - size // 2 if corner.startswith("bottom") else margin + size // 2
    cx = width - margin - size // 2 if corner.endswith("right") else margin + size // 2
    half = size // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            # astroid-like star: |dx|^0.6 + |dy|^0.6 <= half^0.6
            if abs(dx) ** 0.6 + abs(dy) ** 0.6 <= half ** 0.6:
                image[cy + dy, cx + dx] = value
    return image


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


@pytest.fixture
def made(tmp_path):
    """An image 'made in another tool', outside the project."""
    return save(tmp_path / "downloads" / "Gemini_Generated_Image_abc123.png", picture())


def do_import(project, source, **overrides):
    fields = dict(tool="Gemini app", model="Nano Banana Pro, as shown in the app", prompt=PROMPT, root=project)
    fields.update(overrides)
    return import_reference(source, "comic_hero_01", **fields)


# --- the corner check ---

def test_clean_picture_has_no_flag():
    result = find_corner_marks(picture())
    assert result["flagged"] == [] and set(result["corners"]) == set(CORNERS)
    assert all(mark is None for mark in result["corners"].values())
    assert "heuristic" in result["method"] and "report only" in result["method"]


@pytest.mark.parametrize("corner", CORNERS)
def test_sparkle_is_flagged_in_its_corner_only(corner):
    result = find_corner_marks(add_sparkle(picture(), corner))
    assert result["flagged"] == [corner]
    mark = result["corners"][corner]
    assert mark["tone"] == "light"
    x0, y0, x1, y1 = mark["box"]
    assert 20 <= x1 - x0 <= 30 and 20 <= y1 - y0 <= 30  # about the drawn 26 px star
    if corner == "bottom_right":
        assert x0 > W - 60 and y0 > H - 60


def test_dark_mark_on_a_light_picture_is_flagged():
    light = np.full((H, W, 3), 225, dtype=np.uint8)
    result = find_corner_marks(add_sparkle(light, "bottom_right", value=20))
    assert result["flagged"] == ["bottom_right"] and result["corners"]["bottom_right"]["tone"] == "dark"


def test_picture_content_running_through_a_corner_is_not_a_mark():
    image = picture()
    for offset in range(-3, 4):  # a bright diagonal stroke through the bottom-right corner
        for i in range(120):
            image[H - 1 - i, min(W - 1, max(0, W - 120 + i + offset))] = 250
    image[:40, :40] = 250  # a big bright block filling the top-left corner
    assert find_corner_marks(image)["flagged"] == []


def test_coloured_shape_is_not_a_mark():
    image = picture()
    image[H - 40: H - 14, W - 40: W - 14] = (250, 40, 40)  # saturated red: picture content, not a neutral logo
    assert find_corner_marks(image)["flagged"] == []


def test_check_does_not_change_the_pixels():
    image = add_sparkle(picture())
    before = image.copy()
    find_corner_marks(image)
    assert np.array_equal(image, before)


def test_check_handles_tiny_images():
    assert find_corner_marks(np.zeros((20, 24, 3), dtype=np.uint8))["corner_side"] == 20


# --- importing ---

def test_import_copies_the_file_and_writes_provenance(project, made):
    frame = project / "benchmarks" / "references" / "comic_hero_01" / "rooftop_frame_v001.png"
    save(frame, picture(seed=3))
    source_hash = hashlib.sha256(made.read_bytes()).hexdigest()
    source_bytes = made.read_bytes()

    record = do_import(
        project, made, name="front.png", made_on="2026-10-06", notes="third attempt",
        input_references=["benchmarks/references/comic_hero_01/rooftop_frame_v001.png"],
    )

    target = project / "benchmarks" / "references" / "comic_hero_01" / "front.png"
    assert target.read_bytes() == source_bytes  # byte-identical copy
    assert made.read_bytes() == source_bytes and not is_read_only(made)  # the original is untouched
    assert is_read_only(target) and is_read_only(provenance_path(target))
    assert provenance_path(target).name == "front.png.provenance.json"
    assert json.loads(provenance_path(target).read_text(encoding="utf-8")) == record == load_provenance(target)

    assert record["file"] == "benchmarks/references/comic_hero_01/front.png"
    assert record["asset_set"] == "comic_hero_01"
    assert record["sha256"] == source_hash and record["bytes"] == len(source_bytes)
    assert (record["width"], record["height"], record["format"]) == (W, H, "PNG")
    assert record["original_filename"] == "Gemini_Generated_Image_abc123.png"
    assert record["prompt"] == PROMPT  # exact: line break and double space kept
    assert record["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert record["prompt_source"] == "argument"
    stated = record["stated_by_user"]
    assert stated["tool"] == "Gemini app" and stated["model"].startswith("Nano Banana Pro")
    assert stated["made_on"] == "2026-10-06" and stated["notes"] == "third attempt"
    assert stated["input_references"] == [{
        "path": "benchmarks/references/comic_hero_01/rooftop_frame_v001.png",
        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
    }]
    assert "not produced by a backend" in record["origin"]
    assert record["imported_at"] and record["source_file_modified"]
    assert record["visible_watermark_check"]["flagged"] == []
    assert verify_reference(target)


def test_default_name_is_the_original_name(project, made):
    record = do_import(project, made)
    assert record["file"] == "benchmarks/references/comic_hero_01/Gemini_Generated_Image_abc123.png"
    assert record["stated_by_user"]["made_on"] is None and record["stated_by_user"]["input_references"] == []


def test_watermarked_image_is_imported_unchanged_and_reported(project, tmp_path):
    marked = save(tmp_path / "marked.png", add_sparkle(picture()))
    record = do_import(project, marked)
    target = project / record["file"]
    assert target.read_bytes() == marked.read_bytes()  # never cropped, never cleaned
    check = record["visible_watermark_check"]
    assert check["flagged"] == ["bottom_right"] and check["corners"]["bottom_right"]["box"]


def test_nothing_is_overwritten(project, made, tmp_path):
    do_import(project, made, name="front.png")
    target = project / "benchmarks" / "references" / "comic_hero_01" / "front.png"
    before = target.read_bytes(), provenance_path(target).read_bytes()
    other = save(tmp_path / "other.png", picture(seed=5))
    with pytest.raises(ReferenceImportError, match="already exists"):
        do_import(project, other, name="front.png")
    assert (target.read_bytes(), provenance_path(target).read_bytes()) == before


def test_the_same_image_is_not_imported_twice(project, made):
    do_import(project, made, name="front.png")
    with pytest.raises(ReferenceImportError, match="already in the asset set as benchmarks/references/comic_hero_01/front.png"):
        do_import(project, made, name="front_again.png")
    folder = project / "benchmarks" / "references" / "comic_hero_01"
    assert sorted(p.name for p in folder.iterdir()) == ["front.png", "front.png.provenance.json"]


def test_a_leftover_provenance_file_blocks_the_name(project, made):
    folder = project / "benchmarks" / "references" / "comic_hero_01"
    folder.mkdir(parents=True)
    (folder / "front.png.provenance.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReferenceImportError, match="already exists"):
        do_import(project, made, name="front.png")
    assert not (folder / "front.png").exists()


@pytest.mark.parametrize("overrides, match", [
    (dict(tool="  "), "tool must be given"),
    (dict(model=""), "model must be given"),
    (dict(prompt=" \n"), "prompt must not be empty"),
    (dict(made_on="tomorrow"), "2026-10-06"),
    (dict(made_on="06.10.2026"), "2026-10-06"),
    (dict(name="../front.png"), "file name"),
    (dict(name="front.jpg"), "never converted"),
    (dict(name="front.gif"), "must end in one of"),
    (dict(name="front"), "must end in one of"),
    (dict(input_references=["benchmarks/references/none.png"]), "input reference not found"),
])
def test_bad_input_is_refused_and_nothing_is_written(project, made, overrides, match):
    with pytest.raises(ReferenceImportError, match=match):
        do_import(project, made, **overrides)
    assert not (project / "benchmarks" / "references" / "comic_hero_01").exists()


def test_bad_asset_set_missing_file_and_non_image(project, made, tmp_path):
    with pytest.raises(ReferenceImportError, match="asset_set"):
        import_reference(made, "../x", tool="t", model="m", prompt="p", root=project)
    with pytest.raises(ReferenceImportError, match="no such file"):
        do_import(project, tmp_path / "missing.png")
    fake = tmp_path / "notes.png"
    fake.write_text("this is not a picture", encoding="utf-8")
    with pytest.raises(ReferenceImportError, match="not a readable image"):
        do_import(project, fake)
    assert not (project / "benchmarks").exists()


def test_jpeg_keeps_its_bytes_and_format(project, tmp_path):
    source = tmp_path / "shot.JPG"
    Image.fromarray(picture()).save(source, format="JPEG", quality=90)
    record = do_import(project, source, name="shot_03_keyframe.jpg")
    assert record["format"] == "JPEG" and (record["width"], record["height"]) == (W, H)
    assert (project / record["file"]).read_bytes() == source.read_bytes()


def test_changed_file_fails_verification(project, made):
    record = do_import(project, made, name="front.png")
    target = project / record["file"]
    assert verify_reference(target)
    target.chmod(0o666)
    target.write_bytes(target.read_bytes() + b"x")
    assert not verify_reference(target)


def test_provenance_files_are_tracked_by_git_but_the_media_is_not():
    if shutil.which("git") is None:
        pytest.skip("git not installed")

    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT).returncode == 0

    assert ignored("benchmarks/references/comic_hero_01/front.png")
    assert not ignored("benchmarks/references/comic_hero_01/front.png.provenance.json")


# --- the script ---

@pytest.fixture
def script(project, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    spec = importlib.util.spec_from_file_location("script_import_reference", SCRIPTS / "import_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(script, *argv):
    lines = []
    code = script.main([str(a) for a in argv], out=lines.append)
    return code, "\n".join(lines)


def test_script_imports_with_a_prompt_file(script, project, made, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(PROMPT + "\n", encoding="utf-8")
    code, text = run_script(
        script, made, "--asset-set", "comic_hero_01", "--tool", "Gemini app", "--model", "Nano Banana Pro",
        "--prompt-file", prompt_file, "--name", "front.png", "--date", "2026-10-06",
    )
    assert code == 0
    assert "imported: benchmarks/references/comic_hero_01/front.png" in text
    assert f"size: {W}x{H} PNG" in text and "no likely visible mark" in text
    record = load_provenance(project / "benchmarks" / "references" / "comic_hero_01" / "front.png")
    assert record["prompt"] == PROMPT  # only the final line break of the file is dropped
    assert record["prompt_source"] == "file prompt.txt"


def test_script_reports_a_likely_mark_but_still_imports(script, project, tmp_path):
    marked = save(tmp_path / "marked.png", add_sparkle(picture()))
    code, text = run_script(
        script, marked, "--asset-set", "comic_hero_01", "--tool", "Gemini app", "--model", "x", "--prompt", "p",
    )
    assert code == 0
    assert "WARNING: likely visible mark in the bottom right corner" in text
    assert "nothing is cropped automatically" in text
    assert (project / "benchmarks" / "references" / "comic_hero_01" / "marked.png").read_bytes() == marked.read_bytes()


def test_script_reports_errors_and_writes_nothing(script, project, made, tmp_path):
    code, text = run_script(
        script, made, "--asset-set", "comic_hero_01", "--tool", "Gemini app", "--model", "x",
        "--prompt-file", tmp_path / "missing.txt",
    )
    assert code == 1 and "cannot read the prompt file" in text
    code, text = run_script(
        script, made, "--asset-set", "comic_hero_01", "--tool", "Gemini app", "--model", "x", "--prompt", "p",
        "--date", "soon",
    )
    assert code == 1 and text.startswith("ERROR:")
    assert not (project / "benchmarks").exists()


def test_script_needs_exactly_one_prompt_source(script, made):
    for extra in ([], ["--prompt", "a", "--prompt-file", "b"]):
        with pytest.raises(SystemExit) as info:
            script.main([str(made), "--asset-set", "s", "--tool", "t", "--model", "m", *extra], out=lambda line: None)
        assert info.value.code == 2
