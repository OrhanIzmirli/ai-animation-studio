"""Import an image you made in another tool as a reference asset, with its provenance.

    python scripts/import_reference.py IMAGE --asset-set comic_hero_01
        --tool "Gemini app" --model "Nano Banana Pro, as shown in the app"
        --prompt-file prompt.txt            (or --prompt "exact prompt text")
        [--name front.png] [--date 2026-10-06]
        [--input-reference benchmarks/references/comic_hero_01/rooftop_frame_v001.png ...]
        [--notes "second attempt, first had six fingers"]

Copies IMAGE, byte for byte, to benchmarks/references/<asset-set>/<name> and
writes <name>.provenance.json beside it: the tool, model and date as YOU state
them, the exact prompt, the original file name, and what is measured here
(SHA-256, dimensions, format, a check for a visible mark in the corners).

The original is only read. Nothing is converted or cropped, nothing is ever
overwritten, and an image that is already in the asset set is refused.

The corner check only REPORTS a likely visible logo / sparkle; look at the
picture yourself. The import succeeds either way. Needs Pillow.

Exit code 0 if imported, 1 if not (nothing written), 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Works without an installed package: use the source tree next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_studio.paths import project_root  # noqa: E402
from ai_studio.references import ReferenceImportError, import_reference  # noqa: E402


def main(argv: list[str] | None = None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="the image file to import (it is only read)")
    parser.add_argument("--asset-set", required=True, help="folder under benchmarks/references/, e.g. comic_hero_01")
    parser.add_argument("--tool", required=True, help="the tool that made it, as you state it")
    parser.add_argument("--model", required=True, help="the model, as the tool names it")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="the exact prompt text")
    prompt.add_argument("--prompt-file", help="a UTF-8 text file holding the exact prompt")
    parser.add_argument("--name", default=None, help="file name in the asset set (default: the image's own name)")
    parser.add_argument("--date", default=None, help="the day the image was made, YYYY-MM-DD")
    parser.add_argument("--input-reference", action="append", default=[],
                        help="a reference image you gave to the tool (repeatable)")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    if args.prompt_file is not None:
        try:
            # Exact text; only the line break an editor adds at the very end is dropped.
            text = Path(args.prompt_file).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            out(f"ERROR: cannot read the prompt file: {exc}")
            return 1
        prompt_text, prompt_source = text.rstrip("\r\n"), f"file {Path(args.prompt_file).name}"
    else:
        prompt_text, prompt_source = args.prompt, "argument"

    root = project_root()
    try:
        record = import_reference(
            args.image, args.asset_set, tool=args.tool, model=args.model, prompt=prompt_text, root=root,
            name=args.name, prompt_source=prompt_source, made_on=args.date,
            input_references=args.input_reference, notes=args.notes,
        )
    except ReferenceImportError as exc:
        out(f"ERROR: {exc}")
        return 1

    out(f"imported: {record['file']}")
    out(f"provenance: {record['file']}.provenance.json")
    out(f"sha256: {record['sha256']}")
    out(f"size: {record['width']}x{record['height']} {record['format']}, {record['bytes']} bytes")
    check = record["visible_watermark_check"]
    if check["flagged"]:
        for corner in check["flagged"]:
            mark = check["corners"][corner]
            out(f"WARNING: likely visible mark in the {corner.replace('_', ' ')} corner "
                f"({mark['tone']} shape at x {mark['box'][0]}-{mark['box'][2]}, y {mark['box'][1]}-{mark['box'][3]})")
        out("The image was imported unchanged. Look at it and decide; nothing is cropped automatically.")
    else:
        out("corner check: no likely visible mark found (a heuristic; look at the corners yourself)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
