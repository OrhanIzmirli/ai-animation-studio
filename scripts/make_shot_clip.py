"""Animate one trailer shot from a chosen keyframe (Wan 2.2 TI2V-5B Turbo, local ComfyUI).

    python scripts/make_shot_clip.py --trailer milo_01 --shot shot_03
        --keyframe outputs/keyframes/milo_01/shot_03_start/milo_80s_v1_seed42_3941688e.png [--seed 42]

The keyframe is copied (never overwritten) to
benchmarks/references/<trailer>_keyframes/<shot>_<keyframe file name> with a
provenance file made from its keyframe record, so the run's start frame
points at a durable file. The shot's prompt, negative prompt and duration
come from trailers/<trailer>/shotlist.yaml; its camera and motion go into the
request as shot controls (recorded; the Turbo backend uses only the prompt).

The run is recorded under benchmark id <trailer>_<shot> with purpose
"experiment" and measured like every run. Put the printed clip line into the
shot list to use it in the trailer.

Close the browser first: the video model needs nearly all of the RAM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRunner  # noqa: E402
from ai_studio.generation.backends.wan_comfyui import WanTurboComfyUIBackend  # noqa: E402
from ai_studio.generation.request import Resolution, VideoGenerationRequest  # noqa: E402
from ai_studio.paths import outputs_dir, resolve_path, results_dir, to_project_relative  # noqa: E402
from ai_studio.protection import make_read_only  # noqa: E402
from ai_studio.references import sha256_file  # noqa: E402
from ai_studio.trailer.shotlist import load_shotlist  # noqa: E402


def install_keyframe(trailer_id: str, shot_id: str, keyframe: Path) -> str:
    """Copy the keyframe under benchmarks/references with provenance; return its project path."""
    target = ROOT / "benchmarks" / "references" / f"{trailer_id}_keyframes" / f"{shot_id}_{keyframe.name}"
    relative = to_project_relative(target, ROOT)
    if target.exists():
        if sha256_file(target) != sha256_file(keyframe):
            raise FileExistsError(f"{relative} exists with different content")
        return relative
    record_file = keyframe.with_suffix(".json")
    record = json.loads(record_file.read_text(encoding="utf-8")) if record_file.is_file() else None
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(keyframe, "rb") as reader, open(target, "xb") as writer:
        shutil.copyfileobj(reader, writer)
    provenance = {
        "provenance_version": 1, "asset_set": f"{trailer_id}_keyframes", "file": relative,
        "sha256": sha256_file(target), "original_file": to_project_relative(keyframe, ROOT),
        "origin": "generated on this machine by scripts/make_keyframes.py under a style lock",
        "keyframe_record": record, "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = target.with_name(target.name + ".provenance.json")
    with sidecar.open("x", encoding="utf-8") as f:
        f.write(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    for path in (target, sidecar):
        make_read_only(path)
    return relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trailer", required=True)
    parser.add_argument("--shot", required=True)
    parser.add_argument("--keyframe", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shotlist = load_shotlist(ROOT / "trailers" / args.trailer / "shotlist.yaml")
    shot = next((s for s in shotlist.shots if s.id == args.shot), None)
    if shot is None or shot.source != "generated":
        print(f"ERROR: {args.shot} is not a generated shot of {args.trailer}")
        return 2
    start_frame = install_keyframe(args.trailer, shot.id, resolve_path(args.keyframe, ROOT))
    request = VideoGenerationRequest(
        prompt=shot.prompt, negative_prompt=shot.negative_prompt, duration_seconds=shot.duration_seconds,
        output_resolution=Resolution(width=shotlist.resolution.width, height=shotlist.resolution.height),
        seed=args.seed, start_frame=start_frame, fps=shotlist.fps,
        camera_motion=shot.camera, subject_motion=list(shot.motion) or None,
    )
    benchmark_id = f"{args.trailer}_{shot.id}"
    runner = BenchmarkRunner(BenchmarkLogger(results_dir(ROOT)), outputs_dir(ROOT), root=ROOT)
    with WanTurboComfyUIBackend() as backend:
        record = runner.run(request, backend, benchmark_id=benchmark_id, purpose="experiment",
                            notes=f"trailer {args.trailer}, {shot.id} ({shot.name}); keyframe {start_frame}")
    print(f"status: {record.status}")
    if record.status != "success":
        print(f"failure: {record.failure_stage}: {record.error_message}")
        return 1
    print(f"video: {record.output_video} ({record.frame_count} frames, {record.total_time_seconds:.0f} s)")
    measured = subprocess.run([sys.executable, str(ROOT / "scripts/measure_run.py"), "--benchmark", benchmark_id,
                               "--run-id", record.run_id], capture_output=True, text=True, cwd=ROOT)
    print(measured.stdout.strip().splitlines()[-1] if measured.stdout.strip() else measured.stderr.strip()[-500:])
    print(f"shot list: clip: {{run: {benchmark_id}/{record.run_id}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
