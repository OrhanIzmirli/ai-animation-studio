"""Make the keyframes of a trailer under its style lock (local ComfyUI).

    python scripts/make_keyframes.py --trailer milo_01 --dry-run
    python scripts/make_keyframes.py --trailer milo_01
    python scripts/make_keyframes.py --trailer milo_01 --keyframe shot_03_start --seed 42 --seed 43
    python scripts/make_keyframes.py --trailer milo_01 --check-lock

Reads trailers/<trailer>/keyframes.yaml, its shotlist.yaml and the style lock
it names (configs/style_locks/<lock>.yaml). Without --keyframe every keyframe
is made; without --seed each keyframe's seeds (or the plan's). Pictures go to
outputs/keyframes/<trailer>/<keyframe>/ with a JSON record each; existing
ones are skipped, never overwritten. A contact sheet of every keyframe made
or found in this call is written to outputs/keyframes/<trailer>/.

--dry-run prints the composed prompts and output paths and stops; it needs
no ComfyUI. --check-lock regenerates the lock's design_check and compares it
pixel by pixel with the approved image; it writes nothing.

Close the browser and other heavy programs first: SDXL needs about 7 GB of
VRAM and several GB of RAM.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.keyframes.generate import KeyframeMaker, Request, output_paths, request_for  # noqa: E402
from ai_studio.keyframes.lock import compose, load_lock  # noqa: E402
from ai_studio.keyframes.plan import load_plan  # noqa: E402
from ai_studio.paths import outputs_dir, project_root, resolve_path, to_project_relative  # noqa: E402
from ai_studio.trailer.shotlist import load_shotlist  # noqa: E402


def check_lock(maker: KeyframeMaker, lock, root: Path, out) -> int:
    import numpy as np
    from PIL import Image

    from ai_studio.generation.backends.comfyui import sha256_file

    check = lock.design_check
    if check is None:
        out(f"ERROR: lock {lock.lock_id} has no design_check")
        return 2
    expected_path = resolve_path(check.expected_image, root)
    if sha256_file(expected_path) != check.expected_sha256:
        out(f"ERROR: {check.expected_image} is not the file the lock names (SHA-256 differs)")
        return 2
    prompt = compose(lock, character=check.character, scene=check.scene, pose=check.pose, solo=check.solo)
    request = Request(lock.lock_id, "design_check", prompt.positive, prompt.negative, check.width, check.height, check.seed)
    started = time.perf_counter()
    data, _ = maker.render(request)
    made = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)
    expected = np.asarray(Image.open(expected_path).convert("RGB"), dtype=np.int16)
    if made.shape != expected.shape:
        out(f"FAIL: size {made.shape} against {expected.shape}")
        return 1
    diff = np.abs(made - expected)
    out(f"rendered in {time.perf_counter() - started:.1f} s; pixels differing: {int((diff.max(axis=2) > 0).sum())} "
        f"of {diff.shape[0] * diff.shape[1]}; largest difference {int(diff.max())}")
    if diff.max() == 0:
        out(f"OK: lock {lock.lock_id} reproduces {check.expected_image} pixel for pixel")
        return 0
    out("FAIL: the lock does not reproduce the approved image")
    return 1


def contact_sheet(paths: list[tuple[str, int, Path]], target: Path) -> None:
    from PIL import Image, ImageDraw

    cell_w, cell_h, label = 480, 264, 22
    columns = 4
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label)), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (keyframe_id, seed, path) in enumerate(paths):
        x, y = (i % columns) * cell_w, (i // columns) * (cell_h + label)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_w, cell_h))
            sheet.paste(image, (x, y + label))
        draw.text((x + 6, y + 4), f"{keyframe_id}  seed {seed}", fill=(230, 230, 230))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trailer", required=True)
    parser.add_argument("--keyframe", action="append", default=[], help="only these keyframe ids")
    parser.add_argument("--seed", action="append", type=int, default=[], help="only these seeds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-lock", action="store_true")
    args = parser.parse_args(argv)

    root = project_root()
    folder = root / "trailers" / args.trailer
    import yaml

    with open(folder / "keyframes.yaml", encoding="utf-8") as f:
        lock_id = yaml.safe_load(f)["style_lock"]
    lock = load_lock(root / "configs" / "style_locks" / f"{lock_id}.yaml")
    shots = {shot.id for shot in load_shotlist(folder / "shotlist.yaml").shots}
    plan = load_plan(folder / "keyframes.yaml", lock, shots)
    keyframes = [plan.get(k) for k in args.keyframe] if args.keyframe else list(plan.keyframes)
    jobs = [(k, request_for(lock, plan, k, seed)) for k in keyframes for seed in (args.seed or k.seeds or plan.seeds)]

    if args.dry_run:
        for keyframe, request in jobs:
            png, _ = output_paths(root, plan.trailer_id, request)
            state = "exists" if png.exists() else "to make"
            out(f"--- {keyframe.id} seed {request.seed} ({state}): {to_project_relative(png, root)}")
            out(f"    +  {request.positive}")
            out(f"    -  {request.negative}")
            if request.reference_image:
                out(f"    reference: {request.reference_image}")
        out(f"{len(jobs)} picture(s); lock {lock.lock_id}; {plan.resolution.width}x{plan.resolution.height}")
        return 0

    with KeyframeMaker(lock, root, log=out) as maker:
        if args.check_lock:
            return check_lock(maker, lock, root, out)
        done: list[tuple[str, int, Path]] = []
        for i, (keyframe, request) in enumerate(jobs, 1):
            started = time.perf_counter()
            path, made = maker.make(plan.trailer_id, request)
            what = f"made in {time.perf_counter() - started:.0f} s" if made else "exists, skipped"
            out(f"[{i}/{len(jobs)}] {keyframe.id} seed {request.seed}: {what}: {to_project_relative(path, root)}")
            done.append((keyframe.id, request.seed, path))
    sheet = outputs_dir(root) / "keyframes" / plan.trailer_id / f"contact_{time.strftime('%Y%m%dT%H%M%S')}.png"
    contact_sheet(done, sheet)
    out(f"contact sheet: {to_project_relative(sheet, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
