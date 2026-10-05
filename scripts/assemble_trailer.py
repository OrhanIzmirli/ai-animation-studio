"""Assemble a trailer from its shot list. No generation, CPU only.

    python scripts/assemble_trailer.py --trailer trailer_01
    python scripts/assemble_trailer.py --trailer trailer_01 --profile retro_cel_v1

Reads trailers/<trailer>/shotlist.yaml. Each shot contributes its chosen clip
("clip:" in the shot list), cut to the shot's length and finished with the
trailer's one artist profile; a shot without a clip becomes a labelled grey
card; the title card is drawn in editing. Cuts and fades as the shot list says.

Writes outputs/trailers/<trailer>/<timestamp>/trailer.mp4 and manifest.json.
An existing assembly is never overwritten. Needs ffmpeg, ffprobe and Pillow.

--profile overrides the shot list's artist_profile (a name in
configs/style_recipes/, without .yaml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.paths import project_root, resolve_path  # noqa: E402
from ai_studio.quality.frames import FrameDecodeError  # noqa: E402
from ai_studio.trailer.assemble import RECIPES_DIR, TrailerError, assemble_trailer, shotlist_path  # noqa: E402


def _is_name(value: str) -> bool:
    return bool(value) and not any(c in value for c in "/\\") and value not in (".", "..")


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trailer", required=True, help="trailer id, e.g. trailer_01")
    parser.add_argument("--profile", default=None, help=f"profile name in {RECIPES_DIR}/, without .yaml")
    args = parser.parse_args(argv)
    root = project_root()
    if not _is_name(args.trailer) or not shotlist_path(root, args.trailer).is_file():
        known = sorted(p.parent.name for p in resolve_path("trailers", root).glob("*/shotlist.yaml"))
        out(f"ERROR: no trailer {args.trailer!r}; available: {', '.join(known) or 'none'}")
        return 2
    if args.profile is not None and (
        not _is_name(args.profile) or not resolve_path(f"{RECIPES_DIR}/{args.profile}.yaml", root).is_file()
    ):
        known = sorted(p.stem for p in resolve_path(RECIPES_DIR, root).glob("*.yaml"))
        out(f"ERROR: no profile {args.profile!r}; available: {', '.join(known) or 'none'}")
        return 2
    try:
        manifest = assemble_trailer(root, args.trailer, profile_name=args.profile)
    except FileExistsError as exc:
        out(f"ERROR: {exc}; an assembly is never overwritten")
        return 1
    except (TrailerError, FileNotFoundError, ValueError, FrameDecodeError, RuntimeError) as exc:
        out(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    out(f"trailer: {manifest['trailer_id']} ({manifest['working_title']}), assembly {manifest['assembly']}")
    out(f"video: {manifest['output_video']}")
    out(f"length: {manifest['duration_seconds']:g} s, {manifest['frame_count']} frames at {manifest['fps']} fps, "
        f"{manifest['width']}x{manifest['height']}")
    out(f"artist profile: {manifest['artist_profile']}")
    for shot in manifest["shots"]:
        detail = shot.get("video") or (shot.get("title_card") or {}).get("text") or "grey card"
        out(f"  {shot['id']}  {shot['duration_seconds']:g} s  {shot['kind']:<11}  {detail}")
    if not manifest["complete"]:
        out(f"INCOMPLETE: {len(manifest['placeholder_shots'])} placeholder shot(s): "
            f"{', '.join(manifest['placeholder_shots'])}")
    out(f"processing time: {manifest['processing_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
