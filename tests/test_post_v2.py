"""Style layer v2: accent-preserving palette, luminance-neutral ink and
halftone, the v2 profiles, and the two comparison measures."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ai_studio.post import operations as ops
from ai_studio.post.measures import accent_mask, accent_retention, luminance_shift
from ai_studio.post.noise import luma
from ai_studio.post.profile import POST_VERSION, apply_profile, load_profile, parse_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES = REPO_ROOT / "configs" / "style_recipes"
H, W = 96, 128
MAGENTA, CYAN = (230, 30, 200), (20, 220, 230)


def muted_scene(accents=((MAGENTA, (10, 10)),), count=1):
    """A large area of dull reds and greys with small vivid patches (6x6 px,
    about 0.3 % of the picture each)."""
    y, x = np.indices((H, W))
    frame = np.stack([110 + x // 3, 90 + y // 4, 90 + (x + y) // 10], axis=-1).astype(np.uint8)
    for colour, (top, left) in accents:
        frame[top:top + 6, left:left + 6] = colour
    return np.repeat(frame[None], count, axis=0)


def nearest_distance(palette, colour):
    return float(np.sqrt(((palette - np.array(colour, dtype=np.float32)) ** 2).sum(axis=1)).min())


# --- palette: reserved accent slots ---

def test_plain_kmeans_loses_a_small_accent_and_accent_slots_keep_it():
    source = muted_scene()[0]
    plain = ops.learn_palette(source, 6, seed=1)
    with_accents = ops.learn_palette(source, 6, seed=1, accent_slots=2)
    assert len(plain) == len(with_accents) == 6  # the palette does not grow: slots are reserved inside k
    assert nearest_distance(plain, MAGENTA) > 60  # merged into some dull colour
    assert nearest_distance(with_accents, MAGENTA) < 5  # kept as its own colour


def test_quantized_accent_keeps_its_hue_only_with_accent_slots():
    frames = muted_scene()
    patch = (slice(None), slice(10, 16), slice(10, 16))
    lost = ops.palette_limit(frames, 6, frames[0], seed=1)[patch].reshape(-1, 3).mean(axis=0)
    kept = ops.palette_limit(frames, 6, frames[0], seed=1, accent_slots=2)[patch].reshape(-1, 3).mean(axis=0)
    assert np.abs(kept - np.array(MAGENTA)).max() < 6
    assert np.abs(lost - np.array(MAGENTA)).max() > 40


def test_slots_go_to_the_accents_the_palette_represents_worst():
    source = muted_scene(accents=((MAGENTA, (10, 10)), (CYAN, (60, 90))))[0]
    one = ops.learn_palette(source, 8, seed=1, accent_slots=1)
    two = ops.learn_palette(source, 8, seed=1, accent_slots=2)
    assert min(nearest_distance(one, MAGENTA), nearest_distance(one, CYAN)) < 5  # one slot: one of them
    assert nearest_distance(two, MAGENTA) < 5 and nearest_distance(two, CYAN) < 5  # two slots: both


def test_unused_accent_slots_are_given_back_to_kmeans():
    y, x = np.indices((H, W))
    grey = np.stack([40 + x, 40 + x, 40 + x], axis=-1).astype(np.uint8)  # nothing saturated at all
    assert np.array_equal(ops.learn_palette(grey, 6, seed=1, accent_slots=3), ops.learn_palette(grey, 6, seed=1))


def test_accent_palette_is_deterministic_sorted_and_validated():
    source = muted_scene()[0]
    a = ops.learn_palette(source, 6, seed=1, accent_slots=2)
    assert np.array_equal(a, ops.learn_palette(source, 6, seed=1, accent_slots=2))
    brightness = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    assert (np.diff(brightness) >= 0).all()
    for bad in (-1, 6, 1.5):
        with pytest.raises(ValueError):
            ops.learn_palette(source, 6, accent_slots=bad)


def test_a_large_saturated_area_is_not_an_accent_candidate_worth_a_slot():
    # Half the picture vivid red: k-means already has it, so the slot goes to the small cyan patch.
    frame = muted_scene(accents=((CYAN, (60, 90)),))[0].copy()
    frame[:, : W // 2] = (220, 30, 30)
    frame[60:66, 90:96] = CYAN
    palette = ops.learn_palette(frame, 5, seed=1, accent_slots=1)
    assert nearest_distance(palette, CYAN) < 5 and nearest_distance(palette, (220, 30, 30)) < 5


# --- ink and halftone without global darkening ---

def busy_frames(level=120):
    y, x = np.indices((H, W))
    return (np.stack([((x // 32 + y // 32) % 2) * 80 + level - 40] * 3, axis=-1).astype(np.uint8))[None]


def mean_luma(frames):
    return float(luma(frames).mean())


def test_ink_darkens_the_picture_unless_luminance_is_preserved():
    frames = busy_frames()
    plain = ops.ink_lines(frames, 0.8, (1.0, 3.0), threshold=10, pressure_variation=0.0)
    kept = ops.ink_lines(frames, 0.8, (1.0, 3.0), threshold=10, pressure_variation=0.0, preserve_luminance=True)
    assert mean_luma(plain) < mean_luma(frames) - 3  # v1 behaviour: clearly darker
    assert abs(mean_luma(kept) - mean_luma(frames)) <= 1.0  # v2: within one level
    assert (kept.astype(int) - plain.astype(int)).min() >= 0  # the lines are still there, the fills lighter
    assert np.array_equal(ops.ink_lines(busy_frames() * 0 + 128, 0.8, 2.0, preserve_luminance=True),
                          busy_frames() * 0 + 128)  # nothing to ink: nothing changes


def test_halftone_keeps_the_tone_of_each_area_when_asked():
    for level in (15, 35, 60):
        frames = np.full((1, H, W, 3), level, dtype=np.uint8)
        plain = ops.halftone(frames, 6, 80, strength=0.6, variation=0.0)
        kept = ops.halftone(frames, 6, 80, strength=0.6, variation=0.0, preserve_tone=True)
        assert mean_luma(plain) < level - 1.5
        assert abs(mean_luma(kept) - level) <= 1.0
        assert kept.max() > level > kept.min()  # dark dots on lighter ground, not a flat tone
    bright = np.full((1, H, W, 3), 200, dtype=np.uint8)
    assert np.array_equal(ops.halftone(bright, 6, 80, preserve_tone=True), bright)  # highlights untouched


# --- profiles ---

V1_SHA256 = {
    "comic_finish_v1": "cb8a121628c82a73b8a8758c43e4986fa705cdecd124e165ee751ac40c12f8fc",
    "retro_cel_v1": "8439b401c0fe44be285ed527a0eca28e401ae85d5dc304a3304769824ed2ded6",
}


@pytest.mark.parametrize("name", sorted(V1_SHA256))
def test_v1_profiles_are_untouched(name):
    # These are the hashes recorded in the manifests of the videos made with v1.
    assert hashlib.sha256((RECIPES / f"{name}.yaml").read_bytes()).hexdigest() == V1_SHA256[name]


def test_v1_profiles_keep_their_v1_behaviour():
    for name in V1_SHA256:
        habits = load_profile(RECIPES / f"{name}.yaml").habits
        assert habits["palette"]["accent_slots"] == 0 and habits["ink"]["preserve_luminance"] is False
        if "halftone" in habits:
            assert habits["halftone"]["preserve_tone"] is False


def test_v2_profiles_are_the_same_hand_with_changed_habits():
    assert POST_VERSION == "3"
    for name in ("comic_finish", "retro_cel"):
        v1, v2 = load_profile(RECIPES / f"{name}_v1.yaml"), load_profile(RECIPES / f"{name}_v2.yaml")
        assert v2.profile_id == f"{name}_v2" and v2.version == 2
        assert v2.seed == v1.seed and v2.habits["timing"] == v1.habits["timing"]  # same hand, same timing
        assert set(v2.habits) == set(v1.habits)
        assert v2.habits["palette"]["accent_slots"] > 0
        assert v2.habits["ink"]["preserve_luminance"] is True
        for key in ("thickness", "strength", "taper", "boil_px", "pressure_variation"):
            assert v2.habits["ink"][key] == v1.habits["ink"][key]
    assert load_profile(RECIPES / "comic_finish_v2.yaml").habits["halftone"]["preserve_tone"] is True
    assert sorted(p.stem for p in RECIPES.glob("*.yaml")) == [
        "comic_finish_v1", "comic_finish_v2", "comic_finish_v3", "retro_cel_v1", "retro_cel_v2", "retro_cel_v3",
    ]


V2_SHA256 = {
    "comic_finish_v2": "7660f179d2919a8e2f42cfcf8e970480dc7dd4de88eebeb4d3e18d241efbb74e",
    "retro_cel_v2": "d2b34f95dc68b5ded8cbb9cf2577eeb935dfc38acf485b1666a2abe40a27f787",
}


@pytest.mark.parametrize("name", sorted(V2_SHA256))
def test_v2_profiles_are_frozen_too(name):
    assert hashlib.sha256((RECIPES / f"{name}.yaml").read_bytes()).hexdigest() == V2_SHA256[name]


def test_v3_is_v2_plus_the_accent_guard():
    for name in ("comic_finish", "retro_cel"):
        v2, v3 = load_profile(RECIPES / f"{name}_v2.yaml"), load_profile(RECIPES / f"{name}_v3.yaml")
        assert v3.seed == v2.seed and v3.version == 3
        assert v2.habits["palette"]["accent_pixel_saturation"] == 0.0
        assert v3.habits["palette"]["accent_pixel_saturation"] == 0.3
        changed = {**v3.habits["palette"], "accent_pixel_saturation": 0.0}
        assert changed == v2.habits["palette"]
        assert {h: p for h, p in v3.habits.items() if h != "palette"} == \
               {h: p for h, p in v2.habits.items() if h != "palette"}


PALE_PINK, WHITE = (245, 100, 175), (232, 232, 232)


def pale_scene():
    """Dull surroundings, a small pale-pink accent, and a white patch that is
    closer to that pink than to anything else in the picture."""
    frame = muted_scene(accents=((PALE_PINK, (10, 10)),))[0].copy()
    later = frame.copy()
    later[60:72, 90:102] = WHITE  # appears after the start frame, e.g. a shoe coming into view
    return frame, later[None]


def test_neutral_pixels_take_an_accent_colour_unless_guarded():
    source, frames = pale_scene()
    patch = (0, slice(60, 72), slice(90, 102))
    unguarded = ops.palette_limit(frames, 6, source, seed=1, accent_slots=1)[patch].reshape(-1, 3).mean(axis=0)
    guarded = ops.palette_limit(frames, 6, source, seed=1, accent_slots=1,
                                accent_pixel_saturation=0.3)[patch].reshape(-1, 3).mean(axis=0)
    assert np.abs(unguarded - np.array(PALE_PINK)).max() < 8  # the white patch has turned pink
    assert guarded.max() - guarded.min() < 60 and np.abs(guarded - np.array(PALE_PINK)).max() > 30  # it has not:
    # it took the nearest NON-accent colour, a dull one (there is no white in this palette)
    accent = (0, slice(10, 16), slice(10, 16))  # the real accent still gets its colour
    kept = ops.palette_limit(frames, 6, source, seed=1, accent_slots=1, accent_pixel_saturation=0.3)[accent]
    assert np.abs(kept.reshape(-1, 3).mean(axis=0) - np.array(PALE_PINK)).max() < 8


def test_accent_guard_is_off_by_default_and_validated():
    source, frames = pale_scene()
    assert np.array_equal(ops.palette_limit(frames, 6, source, seed=1, accent_slots=1),
                          ops.palette_limit(frames, 6, source, seed=1, accent_slots=1, accent_pixel_saturation=0.0))
    assert np.array_equal(ops.palette_limit(frames, 6, source, seed=1, accent_pixel_saturation=0.3),
                          ops.palette_limit(frames, 6, source, seed=1))  # no accent slots: nothing to guard
    with pytest.raises(ValueError):
        ops.palette_limit(frames, 6, source, accent_pixel_saturation=1.5)
    palette, flags = ops.learn_palette_with_accents(source, 6, seed=1, accent_slots=1)
    assert flags.sum() == 1 and np.abs(palette[flags][0] - np.array(PALE_PINK)).max() < 5


def test_v2_profile_keeps_luminance_and_accents_on_a_synthetic_shot():
    y, x = np.indices((H, W))
    base = np.stack([60 + x, 40 + y, 50 + (x + y) // 4], axis=-1).astype(np.uint8)
    base[: H // 3] = (base[: H // 3] * 0.4).astype(np.uint8)
    base[40:48, 20:28] = CYAN
    frames = np.stack([np.roll(base, i, axis=1) for i in range(12)])

    def styled(version):
        profile = load_profile(RECIPES / f"comic_finish_v{version}.yaml")
        out, info = apply_profile(frames, profile, frames[0])
        shown = frames[np.array(info["drawings"])]  # compare each styled frame with the drawing it shows
        return luminance_shift(shown, out), accent_retention(shown, out)

    (shift_v1, kept_v1), (shift_v2, kept_v2) = styled(1), styled(2)
    assert abs(shift_v2) <= 1.5 and abs(shift_v2) < abs(shift_v1)
    assert kept_v2 > 0.6 and kept_v2 > kept_v1


# --- measures ---

def test_luminance_shift():
    frames = muted_scene(count=2)
    assert luminance_shift(frames, frames) == 0.0
    darker = (frames * 0.5).astype(np.uint8)
    assert luminance_shift(frames, darker) == pytest.approx(-mean_luma(frames) / 2, abs=1.0)
    with pytest.raises(ValueError):
        luminance_shift(frames, frames[:1])


def test_accent_mask_finds_small_vivid_areas_only():
    frame = muted_scene(accents=((CYAN, (60, 90)),))[0].copy()
    frame[:, : W // 2] = (230, 20, 20)  # a large vivid area in another hue
    frame[60:66, 90:96] = CYAN
    mask = accent_mask(frame)
    assert mask[60:66, 90:96].all()  # the small cyan patch
    assert not mask[:, : W // 2].any()  # not the big red area
    assert mask.sum() == 36
    assert not accent_mask(np.full((H, W, 3), 128, dtype=np.uint8)).any()


def test_accent_retention():
    raw = muted_scene(accents=((MAGENTA, (10, 10)), (CYAN, (60, 90))), count=2)
    assert accent_retention(raw, raw) == 1.0
    grey = raw.copy()
    grey[:, 10:16, 10:16] = 120  # the magenta accent washed out to grey
    assert accent_retention(raw, grey) == pytest.approx(0.5)
    shifted = raw.copy()
    shifted[:, 60:66, 90:96] = (230, 220, 20)  # cyan repainted yellow: another hue
    assert accent_retention(raw, shifted) == pytest.approx(0.5)
    darker = (raw * 0.6).astype(np.uint8)  # darker but the same colours: still retained
    assert accent_retention(raw, darker) == 1.0
    assert accent_retention(raw[0], raw[0]) == 1.0  # single frames work too
    plain = np.full((2, H, W, 3), 128, dtype=np.uint8)
    assert accent_retention(plain, plain) is None  # no accents: undefined
