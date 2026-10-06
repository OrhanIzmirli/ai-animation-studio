"""Keyframes: style locks, prompt composition, keyframe plans, the ComfyUI
graph and output naming. No model, no ComfyUI."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from ai_studio.keyframes.generate import Request, output_paths, request_for, workflow
from ai_studio.keyframes.lock import compose, load_lock, parse_lock
from ai_studio.keyframes.plan import load_plan, parse_plan
from ai_studio.trailer.shotlist import load_shotlist

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "configs/style_locks/milo_80s_v1.yaml"
TRAILER = REPO_ROOT / "trailers/milo_01"
SHA = "a" * 64


def lock_data():
    return {
        "lock_id": "test_lock",
        "checkpoint": {"file": "base.safetensors", "sha256": SHA},
        "loras": [{"file": "style.safetensors", "sha256": SHA, "strength": 0.8}],
        "sampler": {"steps": 25, "cfg": 6.0, "sampler": "dpmpp_2m", "scheduler": "karras"},
        "trigger": "TRIGGER", "style": "STYLE", "tail": "TAIL", "negative": "BASE NEG",
        "characters": {"hero": {
            "name": "NAME", "solo": "SOLO", "solo_negative": "SOLO NEG", "body": "BODY", "negative": "CHAR NEG",
            "reference_image": "ref.png", "reference_sha256": SHA,
        }},
        "ipadapter": {"file": "ipa.safetensors", "sha256": SHA, "clip_vision": "clip.safetensors",
                      "clip_vision_sha256": SHA, "weight": 0.4, "weight_type": "ease out"},
    }


def plan_data():
    return {
        "trailer_id": "t", "style_lock": "test_lock", "resolution": {"width": 1280, "height": 704},
        "seeds": [42, 43],
        "keyframes": [
            {"id": "a_start", "shot": "a", "character": "hero", "pose": "POSE", "scene": "SCENE"},
            {"id": "b_plate", "shot": "b", "character": None, "scene": "PLATE", "negative": "EXTRA", "seeds": [7]},
        ],
    }


# --- composition ---

def test_compose_order_with_a_character():
    prompt = compose(parse_lock(lock_data()), character="hero", pose="POSE", scene="SCENE", negative="EXTRA")
    assert prompt.positive == "TRIGGER, NAME, SOLO, BODY, POSE, STYLE, SCENE, TAIL"
    assert prompt.negative == "BASE NEG, SOLO NEG, CHAR NEG, EXTRA"


def test_compose_without_solo_or_character():
    lock = parse_lock(lock_data())
    duo = compose(lock, character="hero", scene="SCENE", solo=False)
    assert duo.positive == "TRIGGER, NAME, BODY, STYLE, SCENE, TAIL" and duo.negative == "BASE NEG, CHAR NEG"
    plate = compose(lock, character=None, scene="  a   wide\n plate ")
    assert plate.positive == "TRIGGER, STYLE, a wide plate, TAIL" and plate.negative == "BASE NEG"


def test_compose_refuses_unknown_character_and_pose_without_character():
    lock = parse_lock(lock_data())
    with pytest.raises(ValueError, match="not in lock"):
        compose(lock, character="villain", scene="S")
    with pytest.raises(ValueError, match="pose needs a character"):
        compose(lock, character=None, scene="S", pose="P")


@pytest.mark.parametrize("change, match", [
    (lambda d: d.update(colour="red"), "unknown"),
    (lambda d: d["checkpoint"].update(sha256="ABC"), "SHA-256"),
    (lambda d: d["sampler"].update(steps=0), "steps"),
    (lambda d: d["characters"]["hero"].pop("reference_sha256"), "go together"),
    (lambda d: d.pop("trigger"), "missing"),
])
def test_invalid_locks_rejected(change, match):
    data = lock_data()
    change(data)
    with pytest.raises(ValueError, match=match):
        parse_lock(data)


def test_milo_lock_reproduces_the_approved_prompt():
    """The design check composes exactly the prompt the approved image was made with."""
    lock = load_lock(LOCK_FILE)
    check = lock.design_check
    prompt = compose(lock, character=check.character, scene=check.scene, pose=check.pose, solo=check.solo)
    provenance = json.loads((REPO_ROOT / (check.expected_image + ".provenance.json")).read_text(encoding="utf-8"))
    assert provenance["sha256"] == check.expected_sha256
    assert prompt.positive == provenance["prompt"]


# --- plans ---

def test_plan_parses_and_applies_default_seeds():
    plan = parse_plan(plan_data(), parse_lock(lock_data()), {"a", "b"})
    assert [k.id for k in plan.keyframes] == ["a_start", "b_plate"]
    assert plan.seeds == (42, 43) and plan.get("b_plate").seeds == (7,)
    assert plan.get("a_start").prompt(parse_lock(lock_data())).positive.startswith("TRIGGER, NAME, SOLO")


@pytest.mark.parametrize("change, match", [
    (lambda d: d["keyframes"][0].update(shot="z"), "not in the shot list"),
    (lambda d: d["keyframes"].append(copy.deepcopy(d["keyframes"][0])), "duplicate id"),
    (lambda d: d["keyframes"][1].update(pose="P"), "pose needs a character"),
    (lambda d: d["keyframes"][1].update(character_reference=True), "character_reference needs"),
    (lambda d: d["keyframes"][0].update(solo="yes"), "true or false"),
    (lambda d: d.update(seeds=[1, 1]), "seed twice"),
    (lambda d: d["keyframes"][0].update(character_region=[0.1, 0.1, 0.5, 0.5]), "needs character_reference"),
    (lambda d: d["keyframes"][0].update(character_reference=True, character_region=[0.5, 0.1, 0.4, 0.5]), "x0 < x1"),
    (lambda d: d.update(style_lock="other"), "uses lock"),
    (lambda d: d["keyframes"][0].update(id="../x"), "usable as a file name"),
])
def test_invalid_plans_rejected(change, match):
    data = plan_data()
    change(data)
    with pytest.raises(ValueError, match=match):
        parse_plan(data, parse_lock(lock_data()), {"a", "b"})


def test_milo_plan_matches_its_shot_list():
    lock_id = yaml.safe_load((TRAILER / "keyframes.yaml").read_text(encoding="utf-8"))["style_lock"]
    lock = load_lock(REPO_ROOT / "configs/style_locks" / f"{lock_id}.yaml")
    shotlist = load_shotlist(TRAILER / "shotlist.yaml")
    plan = load_plan(TRAILER / "keyframes.yaml", lock, {shot.id for shot in shotlist.shots})
    assert shotlist.duration_seconds == 30
    generated = {s.id for s in shotlist.shots if s.source == "generated"}
    assert generated <= {k.shot for k in plan.keyframes}  # every generated shot has a start frame
    assert (plan.resolution.width, plan.resolution.height) == (shotlist.resolution.width, shotlist.resolution.height)


# --- requests, graph, paths ---

def test_request_key_follows_every_input():
    lock, plan = parse_lock(lock_data()), parse_plan(plan_data(), parse_lock(lock_data()))
    request = request_for(lock, plan, plan.get("a_start"), 42)
    assert request.key == request_for(lock, plan, plan.get("a_start"), 42).key
    assert request.key != request_for(lock, plan, plan.get("a_start"), 43).key
    changed = Request(**{**request.__dict__, "positive": request.positive + "!"})
    assert changed.key != request.key


def test_workflow_chains_the_loras_and_uses_the_lock_sampler():
    lock = parse_lock(lock_data())
    graph = workflow(lock, compose(lock, character=None, scene="S"), width=1280, height=704, seed=5)
    assert graph["20"]["inputs"] == {"model": ["1", 0], "lora_name": "style.safetensors", "strength_model": 0.8}
    sampler = graph["7"]["inputs"]
    assert sampler["model"] == ["20", 0]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (
        5, 25, 6.0, "dpmpp_2m", "karras")
    assert graph["4"]["inputs"]["width"] == 1280 and "33" not in graph


def test_workflow_with_a_reference_goes_through_the_ipadapter():
    lock = parse_lock(lock_data())
    graph = workflow(lock, compose(lock, character="hero", scene="S"), width=1024, height=1024, seed=1,
                     reference_name="ref.png")
    assert graph["33"]["inputs"]["model"] == ["20", 0] and graph["7"]["inputs"]["model"] == ["33", 0]
    assert graph["33"]["inputs"]["weight"] == 0.4 and graph["30"]["inputs"]["image"] == "ref.png"


def test_output_paths(tmp_path):
    request = Request("lk", "shot_01_start", "P", "N", 1280, 704, 42)
    png, sidecar = output_paths(tmp_path, "t", request)
    assert png.parent == tmp_path / "outputs" / "keyframes" / "t" / "shot_01_start"
    assert png.name == f"lk_seed42_{request.key}.png" and sidecar.suffix == ".json" and png.stem == sidecar.stem


def test_dry_run_needs_no_comfyui(capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_keyframes", REPO_ROOT / "scripts/make_keyframes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = []
    assert module.main(["--trailer", "milo_01", "--dry-run", "--keyframe", "shot_03_start", "--seed", "42"],
                       out=lines.append) == 0
    assert any("style of 80scartoon, An adorable" in line for line in lines)
    assert lines[-1].startswith("1 picture(s)")


def test_region_limits_the_reference_to_a_mask():
    lock = parse_lock(lock_data())
    data = plan_data()
    data["keyframes"][0].update(character_reference=True, character_region=[0.1, 0.2, 0.5, 0.9])
    plan = parse_plan(data, lock)
    request = request_for(lock, plan, plan.get("a_start"), 42)
    assert request.reference_region == (0.1, 0.2, 0.5, 0.9)
    plain = request_for(lock, parse_plan(plan_data(), lock), parse_plan(plan_data(), lock).get("a_start"), 42)
    assert plain.reference_region is None and plain.key != request.key
    graph = workflow(lock, compose(lock, character="hero", scene="S"), width=1280, height=704, seed=1,
                     reference_name="ref.png", mask_name="mask.png")
    assert graph["33"]["inputs"]["attn_mask"] == ["34", 0] and graph["34"]["inputs"]["image"] == "mask.png"


def test_region_mask_is_white_inside_and_black_outside():
    from ai_studio.keyframes.generate import region_mask

    mask = region_mask(1280, 704, (0.25, 0.25, 0.75, 0.75)).convert("L")
    assert mask.getpixel((640, 352)) == 255 and mask.getpixel((10, 10)) == 0
