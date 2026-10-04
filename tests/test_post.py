"""The style layer: timing, operations, artist profiles and run variants.
Synthetic frames only; no model, no GPU."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from ai_studio import paths
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkResult, Resolution
from ai_studio.post import operations as ops
from ai_studio.post import timing
from ai_studio.post.noise import SALT_PRESSURE, SALT_WOBBLE_X, box_blur, fine_noise, max_filter, smooth_noise
from ai_studio.post.profile import (
    HABIT_ORDER,
    apply_profile,
    choose_drawings,
    load_profile,
    motion_of,
    parse_profile,
)
from ai_studio.post.variant import postprocess_run
from ai_studio.quality.frames import decode_video
from ai_studio.quality.metrics import (
    METRICS_VERSION,
    compute_metrics,
    hold_pattern_periodicity,
    transition_diffs,
)
from ai_studio.quality.store import VideoMetricsRecord, VideoMetricsStore, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES = REPO_ROOT / "configs" / "style_recipes"
H, W = 96, 128
BENCH = "bench_01"

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


# --- synthetic pictures ---

def flat(value, count=1):
    return np.full((count, H, W, 3), value, dtype=np.uint8)


def split(left, right, count=1):
    """Left half one grey, right half another: one vertical edge at W/2."""
    frames = np.empty((count, H, W, 3), dtype=np.uint8)
    frames[:, :, : W // 2] = left
    frames[:, :, W // 2:] = right
    return frames


def scene(count=12, step=3):
    """A coloured picture with edges, shadows and highlights that scrolls."""
    y, x = np.indices((H, W))
    base = np.stack([
        (x * 2) % 256, (y * 3) % 256, ((x // 16 + y // 16) % 2) * 200 + 20,
    ], axis=-1).astype(np.uint8)
    base[: H // 3] = (base[: H // 3] * 0.25).astype(np.uint8)  # a shadow band at the top
    return np.stack([np.roll(base, i * step, axis=1) for i in range(count)])


def changed(a, b):
    return np.abs(a.astype(int) - b.astype(int)).sum(axis=-1) > 0


# --- timing: fixed holds ---

@pytest.mark.parametrize("count", [0, 1, 2, 7, 24, 121])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_hold_frames_never_changes_the_frame_count(count, n):
    frames = np.arange(count, dtype=np.uint8).reshape(count, 1, 1, 1).repeat(3, axis=-1)
    held = timing.hold_frames(frames, n)
    assert held.shape == frames.shape
    assert held[:, 0, 0, 0].tolist() == [(i // n) * n for i in range(count)]


def test_on_twos_and_on_threes_patterns():
    assert timing.fixed_holds(7, 2).tolist() == [0, 0, 2, 2, 4, 4, 6]
    assert timing.fixed_holds(7, 3).tolist() == [0, 0, 0, 3, 3, 3, 6]
    assert timing.hold_lengths(timing.fixed_holds(7, 3)) == [3, 3, 1]
    assert timing.hold_lengths(np.array([0, 0, 2, 2, 2, 5])) == [2, 3, 1]
    with pytest.raises(ValueError):
        timing.fixed_holds(5, 0)


def test_hold_frames_does_not_modify_its_input():
    frames = scene(6)
    before = frames.copy()
    timing.hold_frames(frames, 2)
    assert np.array_equal(frames, before)


# --- timing: animator holds ---

TIMING = dict(still_below=1.0, moderate_below=3.0, hysteresis=0.2, lookahead=3, rhythm_break=(3, 6))


def holds(motion, **overrides):
    return timing.hold_lengths(timing.animator_holds(np.asarray(motion, dtype=float), **{**TIMING, **overrides}))


def test_hold_follows_the_measured_motion():
    still, moderate, fast = holds([0.3] * 60), holds([2.0] * 60), holds([6.0] * 60)
    assert max(set(still), key=still.count) == 3  # nearly still: mostly on threes
    assert max(set(moderate), key=moderate.count) == 2  # moderate: mostly on twos
    assert set(fast) == {1}  # fast: on ones, every frame a new drawing


def test_timing_changes_within_a_clip_when_the_motion_changes():
    drawings = timing.animator_holds(np.r_[np.full(40, 0.3), np.full(40, 6.0), np.full(40, 2.0)], **TIMING)
    lengths = timing.hold_lengths(drawings)
    assert len(drawings) == 121 and sum(lengths) == 121
    position, by_part = 0, {0: [], 1: [], 2: []}
    for length in lengths:
        by_part[min(position // 40, 2)].append(length)
        position += length
    assert np.mean(by_part[0]) > 2.5 and np.mean(by_part[1]) < 1.3 and 1.7 < np.mean(by_part[2]) < 2.4


def test_drawings_are_a_valid_hold_pattern():
    motion = np.abs(np.sin(np.arange(120) / 9.0)) * 4
    drawings = timing.animator_holds(motion, **TIMING)
    assert len(drawings) == 121 and drawings[0] == 0
    assert (np.diff(drawings) >= 0).all()
    starts = np.flatnonzero(np.r_[True, np.diff(drawings) > 0])
    assert (drawings[starts] == starts).all()  # a drawing is the source frame where it starts
    assert max(timing.hold_lengths(drawings)) <= 3


def test_hysteresis_keeps_the_timing_from_flipping():
    hovering = 1.0 + 0.08 * np.sin(np.arange(120))  # motion wobbling around the "still" threshold
    with_margin = holds(hovering, rhythm_break=(50, 50))
    without = holds(hovering, hysteresis=0.0, rhythm_break=(50, 50))
    switches = lambda lengths: int((np.diff(lengths) != 0).sum())  # noqa: E731
    # with the margin: one timing throughout (plus the single forced rhythm break)
    assert switches(with_margin) <= 2 and switches(without) > 10


def test_animator_timing_is_deterministic_and_seeded():
    motion = np.full(120, 2.0)
    assert holds(motion, seed=7) == holds(motion, seed=7)
    assert holds(motion, seed=7) != holds(motion, seed=8)  # another hand breaks the rhythm elsewhere


@pytest.mark.parametrize("level", [0.3, 2.0])
def test_animator_hold_pattern_is_not_periodic(level):
    drawings = timing.animator_holds(np.full(120, level), **TIMING)
    held = drawings[1:] == drawings[:-1]
    assert hold_pattern_periodicity(held) < 0.95
    assert len(set(timing.hold_lengths(drawings))) >= 2  # the even rhythm is broken now and then


def test_animator_timing_rejects_bad_habits():
    for bad in ({"still_below": 3.0}, {"hysteresis": 1.0}, {"lookahead": 0}, {"rhythm_break": (1, 4)}):
        with pytest.raises(ValueError):
            timing.animator_holds(np.ones(10), **{**TIMING, **bad})


# --- metric: hold_pattern_periodicity ---

def held_of(drawings):
    return drawings[1:] == drawings[:-1]


@pytest.mark.parametrize("n", [2, 3, 4])
def test_fixed_holds_are_perfectly_periodic(n):
    assert hold_pattern_periodicity(held_of(timing.fixed_holds(121, n))) == pytest.approx(1.0)


def test_periodicity_is_undefined_without_a_pattern():
    assert hold_pattern_periodicity(np.zeros(120, dtype=bool)) is None  # on ones: no holds
    assert hold_pattern_periodicity(np.ones(120, dtype=bool)) is None  # a frozen video
    assert hold_pattern_periodicity(np.array([True, False, True])) is None  # too short


def test_irregular_holds_score_lower_than_mechanical_ones():
    irregular = np.random.default_rng(0).random(120) < 0.5
    assert hold_pattern_periodicity(irregular) < 0.5


def test_periodicity_is_part_of_the_metrics_and_bumps_the_version():
    assert METRICS_VERSION == "2"
    moving = scene(24)  # RGB stack, reduced to luma by the metrics
    assert compute_metrics(moving)["hold_pattern_periodicity"] is None  # no holds in a raw clip
    on_twos = timing.hold_frames(moving, 2)
    metrics = compute_metrics(on_twos)
    assert metrics["hold_pattern_periodicity"] == pytest.approx(1.0, abs=0.01)
    assert metrics["held_frame_ratio"] == pytest.approx(12 / 23)  # holds, not a freeze
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


# --- coherent noise ---

def test_smooth_noise_is_deterministic_bounded_and_spatially_coherent():
    field = smooth_noise((H, W), 32, seed=5, salt=SALT_PRESSURE, index=3)
    assert np.array_equal(field, smooth_noise((H, W), 32, seed=5, salt=SALT_PRESSURE, index=3))
    assert field.shape == (H, W) and -1.0 <= field.min() and field.max() <= 1.0
    assert field.std() > 0.1  # it does vary
    neighbour_step = np.abs(np.diff(field, axis=1)).mean()
    fine_step = np.abs(np.diff(fine_noise((H, W), 5, SALT_PRESSURE, 3), axis=1)).mean()
    assert neighbour_step < 0.05 and neighbour_step < fine_step / 20  # no per-pixel jumps


def test_noise_fields_differ_by_seed_salt_and_drawing():
    base = smooth_noise((H, W), 32, 5, SALT_PRESSURE, 3)
    for other in (smooth_noise((H, W), 32, 6, SALT_PRESSURE, 3), smooth_noise((H, W), 32, 5, SALT_WOBBLE_X, 3),
                  smooth_noise((H, W), 32, 5, SALT_PRESSURE, 4)):
        assert not np.array_equal(base, other)


def test_image_helpers():
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 9.0
    assert box_blur(image, 1)[4, 4] == pytest.approx(1.0) and box_blur(image, 1).sum() == pytest.approx(9.0)
    assert max_filter(image, 1)[3:6, 3:6].min() == 9.0 and max_filter(image, 1)[2, 2] == 0.0
    assert np.array_equal(box_blur(image, 0), image)


# --- operations: common contract ---

SOURCE = scene(1)[0]
OPERATIONS = {
    "palette_limit": lambda f: ops.palette_limit(f, 8, SOURCE, seed=1),
    "ink_lines": lambda f: ops.ink_lines(f, 0.8, (1.0, 3.0), boil_px=0.6, seed=1),
    "halftone": lambda f: ops.halftone(f, 6, 90, seed=1),
    "grain": lambda f: ops.grain(f, 4.0, seed=1),
    "misregistration": lambda f: ops.misregistration(f, 1),
    "bloom": lambda f: ops.bloom(f, 120, 8, 0.5),
}


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_every_operation_is_pure_deterministic_and_has_an_effect(name):
    frames = scene(3)
    before = frames.copy()
    first, second = OPERATIONS[name](frames), OPERATIONS[name](frames)
    assert np.array_equal(frames, before)  # input untouched
    assert first.shape == frames.shape and first.dtype == np.uint8
    assert np.array_equal(first, second)  # same input + parameters -> identical output
    assert not np.array_equal(first, frames)  # and it does something


@pytest.mark.parametrize("bad", [np.zeros((2, H, W, 3), dtype=np.float32), np.zeros((H, W, 3), dtype=np.uint8)])
def test_operations_reject_wrong_arrays(bad):
    with pytest.raises(ValueError):
        ops.grain(bad, 1.0)


# --- palette ---

def test_palette_limit_uses_one_k_colour_palette_for_every_frame():
    frames = scene(4)
    limited = ops.palette_limit(frames, 6, SOURCE, seed=3)
    palette = {tuple(colour) for colour in limited.reshape(-1, 3)}
    assert len(palette) <= 6
    for frame in limited:
        assert {tuple(colour) for colour in frame.reshape(-1, 3)} <= palette


def test_palette_comes_from_the_source_image_not_from_the_frames():
    frames = scene(2)
    grey_source = np.stack([np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))] * 3, axis=-1)
    limited = ops.palette_limit(frames, 5, grey_source)
    assert (limited[..., 0] == limited[..., 1]).all() and (limited[..., 1] == limited[..., 2]).all()  # only greys exist


def test_palette_strength_blends_and_seed_is_fixed():
    frames = scene(1)
    assert np.array_equal(ops.palette_limit(frames, 6, SOURCE, strength=0.0), frames)
    hard, soft = ops.palette_limit(frames, 6, SOURCE), ops.palette_limit(frames, 6, SOURCE, strength=0.5)
    distance = lambda a: np.abs(a.astype(int) - frames.astype(int)).mean()  # noqa: E731
    assert 0 < distance(soft) < distance(hard)
    assert np.array_equal(ops.learn_palette(SOURCE, 6, seed=2), ops.learn_palette(SOURCE, 6, seed=2))
    with pytest.raises(ValueError):
        ops.learn_palette(SOURCE, 1)


# --- ink ---

INK = dict(weight_min=1.0, weight_max=5.0, pressure_variation=0.0, taper=0.0)


def ink_width(frame, **kwargs):
    """Inked pixels per row along a vertical edge."""
    mask = ops.ink_mask(frame, **{**INK, **kwargs})
    return float((mask[20:-20] > 0.5).sum(axis=1).mean())


def test_ink_darkens_edges_and_leaves_flat_areas_alone():
    frames = split(60, 200)
    inked = ops.ink_lines(frames, 0.9, (1.0, 3.0), pressure_variation=0.0)
    touched = changed(inked, frames)[0]
    assert touched[:, W // 2 - 2: W // 2 + 2].any()
    assert not touched[:, : W // 2 - 8].any() and not touched[:, W // 2 + 8:].any()
    assert (inked <= frames).all()  # ink only darkens
    assert np.array_equal(ops.ink_lines(flat(128), 0.9, (1.0, 3.0)), flat(128))  # no edge, no line


def test_ink_is_thicker_in_shadow_than_in_highlight():
    in_shadow, in_light = split(10, 90)[0], split(165, 245)[0]  # the same step of 80, dark vs bright
    assert ink_width(in_shadow, contrast_bias=0.0, shadow_bias=1.0) > ink_width(in_light, contrast_bias=0.0, shadow_bias=1.0) + 1


def test_ink_is_thicker_at_strong_contrast():
    strong, weak = split(40, 220)[0], split(100, 160)[0]  # same average brightness
    assert ink_width(strong, shadow_bias=0.0, contrast_bias=1.0, threshold=10) > \
        ink_width(weak, shadow_bias=0.0, contrast_bias=1.0, threshold=10) + 0.5


def test_ink_tapers_toward_the_end_of_a_line():
    frame = np.full((H, W, 3), 60, dtype=np.uint8)
    frame[46:49, 20:90] = 220  # a thin stroke from x=20 to x=90

    def weight_along(taper):  # inked pixels across the stroke, per column
        return (ops.ink_mask(frame, **{**INK, "taper": taper, "shadow_bias": 1.0}) > 0.5).sum(axis=0)

    tapered, untapered = weight_along(0.9), weight_along(0.0)
    middle = tapered[50:60].min()
    assert middle == untapered[50:60].min()  # the body of the stroke keeps its weight
    for end in (slice(17, 20), slice(90, 93)):  # the last pixels at either end
        assert tapered[end].sum() < untapered[end].sum()  # thinner at the tip than without taper
        assert tapered[end].min() < middle
    assert tapered[17:24].tolist() == sorted(tapered[17:24].tolist())  # weight grows away from the tip


def test_pressure_variation_is_smooth_and_belongs_to_the_drawing():
    frame = scene(1)[0]
    plain = ops.ink_mask(frame, **INK)
    a = ops.ink_mask(frame, **{**INK, "pressure_variation": 0.5}, seed=1, drawing_id=10)
    again = ops.ink_mask(frame, **{**INK, "pressure_variation": 0.5}, seed=1, drawing_id=10)
    other = ops.ink_mask(frame, **{**INK, "pressure_variation": 0.5}, seed=1, drawing_id=11)
    assert np.array_equal(a, again) and not np.array_equal(a, other) and not np.array_equal(a, plain)


def test_line_boil_is_off_by_default_and_moves_lines_only_between_drawings():
    frames = split(60, 200, count=2)
    assert np.array_equal(ops.ink_lines(frames, 0.9, 2.0, pressure_variation=0.0, ids=[0, 7])[0],
                          ops.ink_lines(frames, 0.9, 2.0, pressure_variation=0.0, ids=[0, 7])[1])  # boil off: identical
    boiled = ops.ink_lines(frames, 0.9, 2.0, pressure_variation=0.0, boil_px=1.0, ids=[0, 7])
    assert not np.array_equal(boiled[0], boiled[1])  # two drawings: the line sits slightly differently
    same_drawing = ops.ink_lines(frames, 0.9, 2.0, pressure_variation=0.0, boil_px=1.0, ids=[7, 7])
    assert np.array_equal(same_drawing[0], same_drawing[1])  # one drawing: no change

    mask = ops.ink_mask(frames[0], **INK)
    assert ops.line_boil(mask, 0.0) is mask
    wobbled = ops.line_boil(mask, 1.0, seed=1, drawing_id=3)
    assert np.abs(wobbled - mask).max() > 0.01
    assert np.abs(wobbled - mask)[:, : W // 2 - 12].max() == 0  # only where there is ink to move
    with pytest.raises(ValueError):
        ops.line_boil(mask, -1.0)


# --- halftone ---

def test_halftone_puts_dots_in_shadows_only():
    frames = split(30, 220)
    dotted = ops.halftone(frames, 6, 80, strength=0.6, variation=0.0)
    touched = changed(dotted, frames)[0]
    assert not touched[:, W // 2:].any()  # highlights untouched
    shadow = touched[:, : W // 2]
    assert 0.05 < shadow.mean() < 0.95  # a pattern of dots, not a flat tint
    assert (dotted <= frames).all()
    assert np.array_equal(ops.halftone(flat(200), 6, 80), flat(200))


def test_halftone_dots_grow_with_darkness_and_the_screen_does_not_move():
    light_shadow = changed(ops.halftone(flat(70), 6, 80, variation=0.0), flat(70)).mean()
    deep_shadow = changed(ops.halftone(flat(10), 6, 80, variation=0.0), flat(10)).mean()
    assert deep_shadow > light_shadow > 0
    two = ops.halftone(flat(30, count=2), 6, 80)
    assert np.array_equal(two[0], two[1])  # fixed to the screen, the same for every drawing
    with pytest.raises(ValueError):
        ops.halftone(flat(30), 1, 80)


# --- grain ---

def test_grain_is_stronger_in_shadows():
    frames = split(40, 215)
    noisy = ops.grain(frames, 4.0, seed=1, shadow_gain=1.0).astype(float)[0]
    dark_std, bright_std = noisy[:, : W // 2].std(), noisy[:, W // 2:].std()
    assert dark_std > 1.4 * bright_std > 0
    even = ops.grain(frames, 4.0, seed=1, shadow_gain=0.0).astype(float)[0]
    assert even[:, : W // 2].std() == pytest.approx(even[:, W // 2:].std(), rel=0.15)


def test_grain_is_deterministic_per_drawing():
    frames = flat(128, count=2)
    a, b = ops.grain(frames, 3.0, seed=1, ids=[4, 4]), ops.grain(frames, 3.0, seed=1, ids=[4, 5])
    assert np.array_equal(a[0], a[1])  # the same drawing keeps its grain (no change inside a hold)
    assert not np.array_equal(b[0], b[1])  # a new drawing, new grain
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(ops.grain(frames, 3.0, seed=2, ids=[4, 4])[0], a[0])
    assert np.array_equal(ops.grain(frames, 0.0), frames)
    assert (a[0][..., 0] == a[0][..., 1]).all()  # monochrome grain


# --- misregistration ---

def test_misregistration_shifts_red_right_and_blue_down():
    frames = np.zeros((1, H, W, 3), dtype=np.uint8)
    frames[0, 40, 50] = (255, 255, 255)
    shifted = ops.misregistration(frames, 2)[0]
    assert shifted[40, 52, 0] == 255 and shifted[40, 50, 0] == 0  # red moved right
    assert shifted[42, 50, 2] == 255 and shifted[40, 50, 2] == 0  # blue moved down
    assert shifted[40, 50, 1] == 255  # green stays
    assert np.array_equal(ops.misregistration(frames, 0), frames)
    with pytest.raises(ValueError):
        ops.misregistration(frames, 1.5)


# --- bloom ---

def test_bloom_glows_around_bright_areas_only():
    frames = flat(30)
    frames[0, 40:56, 56:72] = 250
    glowing = ops.bloom(frames, 150, 12, 0.6)[0].astype(int)
    assert glowing[36, 64, 0] > 30  # light spills just outside the bright square
    assert glowing[5, 5, 0] == 30  # far away: unchanged
    assert (glowing >= frames[0]).all()  # bloom only brightens
    assert np.array_equal(ops.bloom(flat(30), 150, 12, 0.6), flat(30))


# --- artist profiles ---

PROFILE = {
    "name": "test_finish", "version": 1, "description": "for tests", "seed": 11,
    "habits": {
        "timing": {"mode": "animator", "still_below": 1.0, "moderate_below": 4.0},
        "palette": {"k": 8},
        "ink": {"strength": 0.8, "thickness": [1.0, 3.0], "boil_px": 0.6},
        "halftone": {"dot_size": 6, "threshold": 80},
        "grain": {"amount": 3.0},
        "misregistration": {"px": 1},
    },
}


def test_repository_profiles_load_and_are_named_by_technique():
    names = sorted(path.stem for path in RECIPES.glob("*.yaml"))
    assert names == ["comic_finish_v1", "retro_cel_v1"]
    comic, retro = load_profile(RECIPES / "comic_finish_v1.yaml"), load_profile(RECIPES / "retro_cel_v1.yaml")
    assert comic.profile_id == "comic_finish_v1" and retro.profile_id == "retro_cel_v1"
    assert set(comic.habits) == {"timing", "palette", "ink", "halftone", "grain", "misregistration"}
    assert set(retro.habits) == {"timing", "palette", "ink", "bloom", "grain"}  # no halftone
    assert comic.habits["timing"]["mode"] == retro.habits["timing"]["mode"] == "animator"
    assert comic.habits["misregistration"]["px"] == 1
    assert retro.habits["palette"]["strength"] < comic.habits["palette"]["strength"]  # softer palette
    assert comic.seed != retro.seed  # two different hands


def test_profile_resolves_every_parameter():
    profile = parse_profile(PROFILE)
    assert list(profile.habits) == [h for h in HABIT_ORDER if h in PROFILE["habits"]]
    assert profile.habits["ink"]["taper"] == 0.6 and profile.habits["ink"]["shadow_bias"] == 0.6  # defaults filled in
    assert profile.habits["timing"]["hysteresis"] == 0.2
    assert json.loads(json.dumps(profile.to_dict())) == profile.to_dict()


@pytest.mark.parametrize(
    "change, match",
    [
        ({"habits": {"sparkle": {}}}, "unknown habits"),
        ({"habits": {"ink": {"strength": 1, "thickness": 2, "glow": 1}}}, "unknown parameters"),
        ({"habits": {"ink": {"strength": 1}}}, "missing parameters \\['thickness'\\]"),
        ({"habits": {}}, "non-empty"),
        ({"habits": {"timing": {"mode": "random"}}}, "timing.mode"),
        ({"version": 0}, "version"),
        ({"seed": -1}, "seed"),
        ({"name": "bad name"}, "name"),
        ({"studio": "x"}, "unknown profile keys"),
    ],
)
def test_invalid_profiles_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        parse_profile({**PROFILE, **change})


def test_profile_file_name_must_match_name_and_version(tmp_path):
    good = tmp_path / "test_finish_v1.yaml"
    good.write_text(yaml.safe_dump(PROFILE), encoding="utf-8")
    assert load_profile(good).profile_id == "test_finish_v1"
    bad = tmp_path / "test_finish_v2.yaml"
    bad.write_text(yaml.safe_dump(PROFILE), encoding="utf-8")
    with pytest.raises(ValueError, match="must be named test_finish_v1.yaml"):
        load_profile(bad)


def test_apply_profile_keeps_the_frame_count_and_nothing_changes_inside_a_hold():
    frames = scene(30, step=1)
    styled, info = apply_profile(frames, parse_profile(PROFILE), SOURCE)
    assert styled.shape == frames.shape and styled.dtype == np.uint8
    drawings = np.array(info["drawings"])
    assert len(drawings) == 30 and sum(info["hold_lengths"]) == 30
    assert info["drawing_count"] == len(set(drawings.tolist())) < 30  # some frames are held
    for index in range(1, 30):
        same = np.array_equal(styled[index], styled[index - 1])
        assert same == (drawings[index] == drawings[index - 1])  # identical inside a hold, new on a new drawing
    assert 0 < info["held_frame_ratio"] < 1


def test_apply_profile_is_deterministic_and_the_seed_is_the_hand():
    frames = scene(12, step=1)
    a, info_a = apply_profile(frames, parse_profile(PROFILE), SOURCE)
    b, info_b = apply_profile(frames, parse_profile(PROFILE), SOURCE)
    assert np.array_equal(a, b) and info_a == info_b
    other_hand, _ = apply_profile(frames, parse_profile({**PROFILE, "seed": 12}), SOURCE)
    assert not np.array_equal(a, other_hand)


def test_same_profile_gives_the_same_habits_on_another_shot():
    profile = parse_profile(PROFILE)
    shot_a, shot_b = scene(12, step=1), scene(12, step=1)[:, ::-1].copy()
    styled_a, _ = apply_profile(shot_a, profile, shot_a[0])
    styled_b, _ = apply_profile(shot_b, profile, shot_b[0])
    # One hand: the grain field of drawing 0 is the same on both shots.
    plain = parse_profile({**PROFILE, "habits": {"grain": {"amount": 3.0, "shadow_gain": 0.0}}})
    grey = flat(128, count=2)
    grain_a, _ = apply_profile(grey, plain, grey[0])
    grain_b, _ = apply_profile(grey.copy(), plain, grey[0])
    assert np.array_equal(grain_a, grain_b)
    assert styled_a.shape == styled_b.shape


def test_fixed_timing_mode_and_no_timing_habit():
    frames = scene(9, step=1)
    fixed = parse_profile({**PROFILE, "habits": {"timing": {"mode": "fixed", "n": 3}}})
    assert choose_drawings(frames, fixed).tolist() == [0, 0, 0, 3, 3, 3, 6, 6, 6]
    styled, info = apply_profile(frames, fixed, SOURCE)
    assert info["hold_lengths"] == [3, 3, 3] and np.array_equal(styled, frames[[0, 0, 0, 3, 3, 3, 6, 6, 6]])
    untimed = parse_profile({**PROFILE, "habits": {"misregistration": {"px": 1}}})
    assert choose_drawings(frames, untimed).tolist() == list(range(9))


def test_motion_is_measured_like_the_quality_metrics():
    frames = scene(6, step=2)
    assert motion_of(frames) == pytest.approx(transition_diffs(frames), rel=0.02)
    assert len(motion_of(frames)) == 5


# --- run variants ---

def write_video(path, frames, fps=24):
    height, width = frames.shape[1:3]
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
        input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True,
    )


def write_png(path, frame):
    height, width = frame.shape[:2]
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-i", "-", "-frames:v", "1", str(path)],
        input=np.ascontiguousarray(frame).tobytes(), check=True, capture_output=True,
    )


@pytest.fixture
def run_project(project):
    """A project with one successful run (video + derived start frame), one
    failed run, and a profile file."""
    frames = scene(30, step=1)
    run_dir = project / "outputs" / BENCH / "run-1"
    write_video(run_dir / "out.mp4", frames)
    write_png(run_dir / "start_frame.png", frames[0])
    logger = BenchmarkLogger(project / "benchmarks" / "results")
    common = dict(benchmark_id=BENCH, task_type="video_generation", model="m", model_version="1",
                  workflow_version="w", seed=1, inference_steps=4, guidance=1.0, prompt="p",
                  resolution=Resolution(width=W, height=H))
    logger.log(BenchmarkResult(run_id="run-1", output_video=f"outputs/{BENCH}/run-1/out.mp4", start_frame={
        "path": "benchmarks/references/f.png", "sha256": "ab" * 32, "original_width": W, "original_height": H,
        "preprocessing": [], "effective_width": W, "effective_height": H,
        "derived_file": f"outputs/{BENCH}/run-1/start_frame.png",
    }, **common))
    logger.log(BenchmarkResult(run_id="failed", status="failed", failure_stage="generation",
                               error_type="MemoryError", **common))
    recipes = project / "configs" / "style_recipes"
    recipes.mkdir(parents=True)
    (recipes / "test_finish_v1.yaml").write_text(yaml.safe_dump(PROFILE), encoding="utf-8")
    store = VideoMetricsStore(project / "benchmarks" / "metrics", logger)
    return project, logger, store, recipes / "test_finish_v1.yaml"


@needs_ffmpeg
def test_postprocess_run_writes_video_manifest_and_variant_metrics(run_project):
    project, logger, store, recipe = run_project
    record_bytes = logger.path_for(BENCH, "run-1").read_bytes()
    source = project / "outputs" / BENCH / "run-1" / "out.mp4"
    source_hash = sha256_file(source)

    manifest = postprocess_run(logger, store, project, BENCH, "run-1", recipe)

    target = project / "outputs" / BENCH / "run-1" / "post" / "test_finish_v1"
    assert sorted(p.name for p in target.iterdir()) == ["manifest.json", f"metrics_v{METRICS_VERSION}.json", "out.mp4"]
    assert json.loads((target / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert set(manifest) == {
        "benchmark_id", "run_id", "variant", "recipe", "profile", "recipe_file", "recipe_sha256",
        "post_version", "source_video", "source_video_sha256", "start_frame", "start_frame_sha256",
        "output_video", "output_video_sha256", "frame_count", "fps", "width", "height", "encoder",
        "timing", "metrics_file", "tool_versions", "processing_seconds", "timestamp",
    }
    assert manifest["recipe"] == "test_finish_v1" and manifest["variant"] == "post/test_finish_v1"
    assert manifest["profile"] == parse_profile(PROFILE).to_dict()  # every parameter, defaults included
    assert manifest["recipe_sha256"] == sha256_file(recipe)
    assert manifest["source_video"] == f"outputs/{BENCH}/run-1/out.mp4"
    assert manifest["source_video_sha256"] == source_hash
    assert manifest["output_video"] == f"outputs/{BENCH}/run-1/post/test_finish_v1/out.mp4"
    assert manifest["output_video_sha256"] == sha256_file(target / "out.mp4")
    assert manifest["encoder"] == {"codec": "libx264", "pix_fmt": "yuv420p", "crf": 12, "preset": "slow",
                                   "fps": 24, "audio": False}
    assert set(manifest["tool_versions"]) == {"python", "numpy", "ffmpeg"} and all(manifest["tool_versions"].values())
    assert manifest["frame_count"] == 30 and sum(manifest["timing"]["hold_lengths"]) == 30

    styled = decode_video(target / "out.mp4", analysis_width=None, gray=False)
    assert styled.shape == (30, H, W, 3)  # same frame count and size as the source

    # Metrics: next to the manifest, marked as a variant; the raw run's store is untouched.
    metrics = VideoMetricsRecord.from_dict(json.loads((target / f"metrics_v{METRICS_VERSION}.json").read_text()))
    assert metrics.variant == "post/test_finish_v1" and metrics.video_sha256 == manifest["output_video_sha256"]
    assert metrics.metrics["held_frame_ratio"] > 0.2  # holds are visible in the measured video
    assert not store.exists(BENCH, "run-1") and not (project / "benchmarks" / "metrics").exists()

    # The run itself is untouched.
    assert logger.path_for(BENCH, "run-1").read_bytes() == record_bytes
    assert sha256_file(source) == source_hash


@needs_ffmpeg
def test_postprocess_is_reproducible_and_never_overwrites(run_project, tmp_path):
    project, logger, store, recipe = run_project
    first = postprocess_run(logger, store, project, BENCH, "run-1", recipe)
    target = project / "outputs" / BENCH / "run-1" / "post" / "test_finish_v1"
    kept = (target / "out.mp4").read_bytes()
    with pytest.raises(FileExistsError):
        postprocess_run(logger, store, project, BENCH, "run-1", recipe)
    assert (target / "out.mp4").read_bytes() == kept

    shutil.move(str(target), str(tmp_path / "first"))  # do it again from scratch: the same video
    second = postprocess_run(logger, store, project, BENCH, "run-1", recipe)
    assert second["output_video_sha256"] == first["output_video_sha256"]
    assert second["timing"] == first["timing"]


@needs_ffmpeg
def test_postprocess_refuses_failed_runs_and_missing_files_without_leaving_debris(run_project):
    project, logger, store, recipe = run_project
    with pytest.raises(ValueError, match="did not succeed"):
        postprocess_run(logger, store, project, BENCH, "failed", recipe)
    with pytest.raises(FileNotFoundError):
        postprocess_run(logger, store, project, BENCH, "no-such-run", recipe)
    (project / "outputs" / BENCH / "run-1" / "start_frame.png").unlink()
    with pytest.raises(FileNotFoundError, match="palette from the start frame"):
        postprocess_run(logger, store, project, BENCH, "run-1", recipe)
    (project / "outputs" / BENCH / "run-1" / "out.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="not on this machine"):
        postprocess_run(logger, store, project, BENCH, "run-1", recipe)
    assert not (project / "outputs" / BENCH / "run-1" / "post" / "test_finish_v1").exists()


def test_metrics_store_keeps_raw_and_variant_apart(run_project, tmp_path):
    _, _, store, _ = run_project
    fields = dict(benchmark_id=BENCH, run_id="run-1", metrics_version=METRICS_VERSION,
                  analysis_size={"width": 320, "height": 176}, video_sha256="ab" * 32,
                  metrics={"motion_mean": 1.0}, per_transition_diffs=[1.0],
                  tool_versions={"numpy": "1", "ffmpeg": "8"})
    raw, variant = VideoMetricsRecord(**fields), VideoMetricsRecord(**fields, variant="post/x_v1")
    assert raw.variant is None
    with pytest.raises(ValueError, match="use save_variant"):
        store.save(variant)  # a variant can never land in the raw run's place
    with pytest.raises(ValueError, match="needs a VideoMetricsRecord with a variant"):
        store.save_variant(raw, tmp_path)
    path = store.save_variant(variant, tmp_path)
    assert path == tmp_path / f"metrics_v{METRICS_VERSION}.json"
    with pytest.raises(FileExistsError):
        store.save_variant(variant, tmp_path)
    old = {k: v for k, v in raw.to_dict().items() if k != "variant"}  # a file written before variants existed
    assert VideoMetricsRecord.from_dict(old).variant is None
    with pytest.raises(ValueError):
        VideoMetricsRecord(**fields, variant="  ")


@needs_ffmpeg
def test_postprocess_script(run_project, monkeypatch):
    project, _, _, _ = run_project
    spec = importlib.util.spec_from_file_location("script_postprocess_run", REPO_ROOT / "scripts" / "postprocess_run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))

    def run(*argv):
        lines = []
        return script.main(list(argv), out=lines.append), "\n".join(lines)

    code, text = run("--benchmark", BENCH, "--run-id", "run-1", "--recipe", "test_finish_v1")
    assert code == 0 and "variant: post/test_finish_v1" in text and "hold_pattern_periodicity" in text
    assert (project / "outputs" / BENCH / "run-1" / "post" / "test_finish_v1" / "manifest.json").is_file()
    code, text = run("--benchmark", BENCH, "--run-id", "run-1", "--recipe", "test_finish_v1")
    assert code == 1 and "never overwritten" in text
    code, text = run("--benchmark", BENCH, "--run-id", "run-1", "--recipe", "nope")
    assert code == 2 and "available: test_finish_v1" in text
    code, text = run("--benchmark", BENCH, "--run-id", "failed", "--recipe", "test_finish_v1")
    assert code == 1 and "did not succeed" in text


@needs_ffmpeg
def test_decode_video_at_native_size(tmp_path):
    frames = scene(4)
    write_video(tmp_path / "v.mp4", frames)
    native = decode_video(tmp_path / "v.mp4", analysis_width=None, gray=False)
    assert native.shape == frames.shape
    assert np.abs(native.astype(int) - frames.astype(int)).mean() < 12  # lossy, but the same picture
