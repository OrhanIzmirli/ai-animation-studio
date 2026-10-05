"""Review material for the style layer (throwaway script, not src/).

    python spikes/style_review.py RUN_ID [RUN_ID ...]

For every post variant found under outputs/comic_rooftop_01/<run>/post/ it
writes to spikes/output/style/ (git-ignored):

    before_after_<run>_<recipe>.mp4   raw left, styled right, labelled, 24 fps
    contact_<run>_<recipe>.png        frames 0 / 48 / 96 / 120: raw row, styled row
    face_<run>_<recipe>.png           frame 120, head-and-shoulders crop, raw | styled
    sign_<run>_<recipe>.png           frame 60, the billboard (an accent colour), raw | styled
    shoe_<run>_<recipe>.png           frame 120, lower right corner (a pale neutral), raw | styled
    table.md                          metrics raw vs styled, with luminance_shift and
                                      accent_retention (ai_studio.post.measures) on every 8th frame

The crop box is the one used in experiment E1 (docs/experiments/E1_REPORT.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ai_studio.post.measures import accent_retention, luminance_shift  # noqa: E402
from ai_studio.quality.frames import decode_video  # noqa: E402
from ai_studio.quality.metrics import HELD_FRAME_THRESHOLD, hold_pattern_periodicity  # noqa: E402
from ai_studio.video_encode import encode_frames  # noqa: E402

BENCHMARK = "comic_rooftop_01"
OUT = REPO / "spikes" / "output" / "style"
FRAMES = (0, 48, 96, 120)
HEAD_BOX = (160, 100, 660, 420)
SIGN_BOX = (1130, 300, 1280, 460)  # the pink billboard on the right (an accent colour), chosen by eye
SHOE_BOX = (600, 560, 800, 704)  # in seed 3 the white shoe at frame 120; elsewhere the ledge
LABEL = 30
KEYS = ("motion_mean", "motion_max", "total_change", "start_frame_psnr", "temporal_pulse_ratio",
        "luminance_flicker", "luminance_min", "luminance_max", "held_frame_ratio", "hold_pattern_periodicity")


def font(size=20):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def labelled(image: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + LABEL), (0, 0, 0))
    ImageDraw.Draw(out).text((6, 3), text, fill=(255, 255, 255), font=font())
    out.paste(image, (0, LABEL))
    return out


def raw_metrics(run_id: str) -> tuple[dict, str]:
    folder = REPO / "benchmarks" / "metrics" / BENCHMARK / run_id
    for version in ("2", "1"):
        path = folder / f"metrics_v{version}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            metrics = dict(record["metrics"])
            if "hold_pattern_periodicity" not in metrics:  # v1: derive it from the stored transition diffs
                held = np.array(record["per_transition_diffs"]) < HELD_FRAME_THRESHOLD
                metrics["hold_pattern_periodicity"] = hold_pattern_periodicity(held)
            return metrics, f"v{version}"
    return {}, "-"


def main(run_ids: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_id in run_ids:
        short = run_id.split("-")[-1]
        run_dir = REPO / "outputs" / BENCHMARK / run_id
        metrics, version = raw_metrics(run_id)
        rows.append((f"`{short}` raw", version, "-", "-", None, None, metrics))
        if not (run_dir / "out.mp4").is_file():
            print(f"{run_id}: raw video missing, skipped")
            continue
        raw = decode_video(run_dir / "out.mp4", analysis_width=None, gray=False)
        for variant in sorted((run_dir / "post").glob("*/manifest.json")):
            manifest = json.loads(variant.read_text(encoding="utf-8"))
            recipe = manifest["recipe"]
            styled = decode_video(variant.parent / "out.mp4", analysis_width=None, gray=False)
            measured = json.loads((REPO / manifest["metrics_file"]).read_text(encoding="utf-8"))
            timing = manifest["timing"]
            sample = list(range(0, len(raw), 8))
            shown = raw[np.array(timing["drawings"])[sample]]  # the raw drawing each styled frame shows
            rows.append((f"`{short}` + `{recipe}`", f"v{measured['metrics_version']}",
                         str(timing["drawing_count"]), str(timing["hold_pattern_periodicity"]),
                         luminance_shift(shown, styled[sample]), accent_retention(shown, styled[sample]),
                         measured["metrics"]))
            for tag, frame_index, (left, top, right, bottom) in (
                ("sign", 60, SIGN_BOX), ("shoe", 120, SHOE_BOX),
            ):
                crops = [Image.fromarray(clip[frame_index][top:bottom, left:right]).resize((760, 720), Image.Resampling.LANCZOS)
                         for clip in (raw, styled)]
                both = Image.new("RGB", (1520, 720 + LABEL))
                both.paste(labelled(crops[0], f"raw, frame {frame_index}"), (0, 0))
                both.paste(labelled(crops[1], f"{recipe}, frame {frame_index}"), (760, 0))
                both.save(OUT / f"{tag}_{short}_{recipe}.png")

            bar_raw = np.asarray(labelled(Image.new("RGB", (raw.shape[2], 0)), f"{short}  raw"))
            bar_styled = np.asarray(labelled(Image.new("RGB", (raw.shape[2], 0)), f"{short}  {recipe}"))

            class SideBySide:
                def __len__(self):
                    return len(raw)

                def __getitem__(self, i):
                    return np.hstack([np.vstack([bar_raw, raw[i]]), np.vstack([bar_styled, styled[i]])])

                def __iter__(self):
                    return (self[i] for i in range(len(self)))

            encode_frames(SideBySide(), OUT / f"before_after_{short}_{recipe}.mp4")

            def tile(frame, text):
                return labelled(Image.fromarray(frame).resize((640, 352), Image.Resampling.LANCZOS), text)

            sheet = Image.new("RGB", (640 * len(FRAMES), 2 * (352 + LABEL)))
            for column, index in enumerate(FRAMES):
                sheet.paste(tile(raw[index], f"raw, frame {index}"), (column * 640, 0))
                sheet.paste(tile(styled[index], f"{recipe}, frame {index}"), (column * 640, 352 + LABEL))
            sheet.save(OUT / f"contact_{short}_{recipe}.png")

            left, top, right, bottom = HEAD_BOX
            crops = [
                labelled(Image.fromarray(clip[120][top:bottom, left:right]).resize((1000, 640), Image.Resampling.LANCZOS), text)
                for clip, text in ((raw, "raw, frame 120"), (styled, f"{recipe}, frame 120"))
            ]
            face = Image.new("RGB", (2000, 640 + LABEL))
            face.paste(crops[0], (0, 0))
            face.paste(crops[1], (1000, 0))
            face.save(OUT / f"face_{short}_{recipe}.png")

    def cell(value):
        return "-" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))

    header = ["video", "metrics", "drawings", "periodicity (true pattern)", "luminance_shift",
              "accent_retention", *KEYS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, version, drawings, true_periodicity, shift, retention, metrics in rows:
        lines.append("| " + " | ".join([
            name, version, drawings, true_periodicity,
            "-" if shift is None else f"{shift:+.2f}", cell(retention), *(cell(metrics.get(k)) for k in KEYS),
        ]) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
