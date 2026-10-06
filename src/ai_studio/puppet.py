"""Cut-out ("puppet") animation of a character drawn once.

The approved drawing is cut into layers - the body and the two arms - so the
arms can turn at the shoulder while the body stays the same drawing. The
face is not taken from the drawing: the eyes and the mouth are erased and
redrawn every frame in the drawing's own style, so they can look around,
open wide, blink and change shape. The body bends like cloth (its lower
part and the curl on its head lag behind and ripple), and the whole figure
stretches along its motion when it moves fast.

A pose is a handful of numbers; render() turns it into an RGBA picture to
composite over a plate. Everything is deterministic (Pillow and numpy).

How the arms stay attached: an arm layer holds the arm plus body colour from
deep inside the body behind its root and is drawn BEHIND the body, so when it
turns, the body covers the root. Small notches left at the shoulder are
filled with body colour and inked.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

INK = (18, 16, 28, 255)
IRIS = (38, 206, 204, 255)
MOUTH_DARK = (60, 18, 40, 255)
TONGUE = (232, 120, 150, 255)


@dataclass(frozen=True)
class RigSpec:
    """Where the parts are in the source drawing (pixel coordinates)."""
    background: tuple[int, int, int]                 # the flat colour around the figure
    arms: dict[str, list[tuple[int, int]]]            # name -> polygon of the arm, cut at the body
    pivots: dict[str, tuple[int, int]]                # name -> shoulder point the arm turns around
    cuts: dict[str, tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=dict)  # name -> the cut line
    cut_clear: int = 16                               # ink this close to the cut is removed from the arm
    root_margin: int = 22                             # how far the arm layer reaches under the body
    joint_radius: int = 110                           # around each pivot, notches between arm and body are filled
    joint_close: int = 13                             # size of the notches that get filled (pixels)
    eyes: list[tuple[int, int]] = field(default_factory=list)  # a point inside each eye
    mouth: tuple[int, int] | None = None              # a point inside the mouth
    body_sample: tuple[int, int] = (520, 600)         # a point of plain body colour


@dataclass(frozen=True)
class Pose:
    x: float                 # where the figure's anchor (bottom centre) goes, in output pixels
    y: float
    height: float            # figure height in output pixels
    squash: float = 0.0      # > 0 wider and shorter, < 0 taller and thinner
    tilt: float = 0.0        # degrees, positive = counter-clockwise
    arms: dict[str, float] = field(default_factory=dict)  # name -> degrees, positive = counter-clockwise
    blink: bool = False      # same as eye_open = 0
    eye_open: float = 1.0    # 1 open, 0 shut
    wink: int | None = None  # index of an eye that is shut while the other stays open
    eye_scale: float = 1.0   # > 1: eyes wide (fright)
    gaze: tuple[float, float] = (0.0, 0.0)  # pupils: -1..1 left/right, -1..1 up/down
    mouth: str = "smile"     # smile (the drawing's own) | o (shout) | closed (content smile)
    drag: float = 0.0        # lower body trails sideways, fraction of the height (+ to the right)
    wave: float = 0.0        # ripple of the lower body, fraction of the height
    wave_phase: float = 0.0  # radians; advance it with time for a travelling ripple
    tip: float = 0.0         # the curl on the head swings sideways, fraction of the height
    stretch: float = 0.0     # stretch along stretch_angle (0.3 = 30 % longer), area kept
    stretch_angle: float = 0.0  # degrees, 0 = horizontal
    glow: float = 0.0        # 0..1
    opacity: float = 1.0


def _flood(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    out = np.zeros_like(mask)
    q = deque([seed])
    h, w = mask.shape
    while q:
        y, x = q.popleft()
        if 0 <= y < h and 0 <= x < w and mask[y, x] and not out[y, x]:
            out[y, x] = True
            q.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    return out


def _near(mask: np.ndarray, seed: tuple[int, int], radius: int = 8) -> tuple[int, int]:
    """The (row, column) of the mask pixel closest to seed (x, y), so a seed point
    survives the rounding of a resized drawing."""
    x, y = seed
    h, w = mask.shape
    y0, y1, x0, x1 = max(0, y - radius), min(h, y + radius + 1), max(0, x - radius), min(w, x + radius + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if not len(ys):
        raise ValueError(f"no feature pixel within {radius} px of {seed}")
    k = int(np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2))
    return int(ys[k] + y0), int(xs[k] + x0)


def _polygon_mask(size: tuple[int, int], polygon, grow: int = 0) -> np.ndarray:
    img = Image.new("L", size, 0)
    ImageDraw.Draw(img).polygon(polygon, fill=255)
    if grow:
        img = img.filter(ImageFilter.MaxFilter(2 * grow + 1))
    return np.asarray(img) > 127


def _filter(mask: np.ndarray, kind, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(kind(size))) > 127


def _box_count(mask: np.ndarray, size: int) -> np.ndarray:
    """How many pixels of mask fall in the size x size square around each pixel (integral image)."""
    r = size // 2
    m = np.pad(mask.astype(np.int32), r + 1)
    c = m.cumsum(0).cumsum(1)
    h, w = mask.shape
    y, x = np.arange(h) + r + 1, np.arange(w) + r + 1
    return (c[(y + r)[:, None], (x + r)[None, :]] - c[(y - r - 1)[:, None], (x + r)[None, :]]
            - c[(y + r)[:, None], (x - r - 1)[None, :]] + c[(y - r - 1)[:, None], (x - r - 1)[None, :]])


def _dilate(mask: np.ndarray, size: int) -> np.ndarray:
    return _box_count(mask, size) > 0


def _erode(mask: np.ndarray, size: int) -> np.ndarray:
    return _box_count(mask, size) == size * size


def scaled(spec: RigSpec, factor: float) -> RigSpec:
    """The same spec for the drawing resized by factor."""
    pt = lambda p: (round(p[0] * factor), round(p[1] * factor))  # noqa: E731
    return RigSpec(
        background=spec.background,
        arms={k: [pt(p) for p in v] for k, v in spec.arms.items()},
        pivots={k: pt(v) for k, v in spec.pivots.items()},
        cuts={k: (pt(a), pt(b)) for k, (a, b) in spec.cuts.items()},
        cut_clear=max(2, round(spec.cut_clear * factor)), root_margin=max(2, round(spec.root_margin * factor)),
        joint_radius=round(spec.joint_radius * factor), joint_close=max(2, round(spec.joint_close * factor)),
        eyes=[pt(e) for e in spec.eyes], mouth=pt(spec.mouth) if spec.mouth else None,
        body_sample=pt(spec.body_sample),
    )


@dataclass(frozen=True)
class _Feature:
    cx: float
    cy: float
    w: float
    h: float


class Rig:
    def __init__(self, drawing: Image.Image, spec: RigSpec, scale: float = 1.0) -> None:
        """scale < 1 works on a smaller copy of the drawing (faster; use about
        the largest size the figure will have on screen)."""
        if scale != 1.0:
            drawing = drawing.resize((round(drawing.width * scale), round(drawing.height * scale)), Image.LANCZOS)
            spec = scaled(spec, scale)
        self.spec = spec
        rgb = np.asarray(drawing.convert("RGB")).astype(np.int16)
        h, w = rgb.shape[:2]
        self.size = (w, h)
        figure = np.abs(rgb - np.array(spec.background)).sum(axis=2) > 24
        alpha = np.asarray(Image.fromarray((figure * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)))
        self.body_colour = tuple(int(v) for v in rgb[spec.body_sample[1], spec.body_sample[0]]) + (255,)

        # --- the face: find the eyes and the mouth, then erase them from the drawing
        dark = rgb.sum(axis=2) < 200
        iris = (rgb[..., 1] > 170) & (rgb[..., 0] < 120)
        white = rgb.sum(axis=2) > 650
        pink = (rgb[..., 0] > 140) & (rgb[..., 1] < 140) & (rgb[..., 2] > 100)
        erase = np.zeros((h, w), bool)
        self.eyes: list[_Feature] = []
        for seed in spec.eyes:
            eye = _flood(dark | iris | white, _near(dark | iris, seed))
            ys, xs = np.nonzero(eye)
            self.eyes.append(_Feature(xs.mean(), ys.mean(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            erase |= eye
        self.mouth_feature, self.mouth_smile = None, None
        if spec.mouth:
            m = _flood(dark | pink, _near(dark | pink, spec.mouth))
            ys, xs = np.nonzero(m)
            self.mouth_feature = _Feature(xs.mean(), ys.mean(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            grown = _filter(m, ImageFilter.MaxFilter, 3)
            crop = np.zeros((h, w, 4), np.uint8)
            crop[..., :3] = rgb
            crop[..., 3] = np.where(grown, 255, 0)
            box = (xs.min() - 3, ys.min() - 3, xs.max() + 4, ys.max() + 4)
            self.mouth_smile = (Image.fromarray(crop, "RGBA").crop(box), box[:2])
            erase |= m
        erase = _filter(erase, ImageFilter.MaxFilter, 5)
        clean = rgb.copy()
        clean[erase] = self.body_colour[:3]

        base = np.zeros((h, w, 4), np.uint8)
        base[..., :3] = clean
        base[..., 3] = alpha

        # --- arms
        arms_union = np.zeros((h, w), bool)
        deep = _filter(figure, ImageFilter.MinFilter, 2 * spec.root_margin + 1)
        self.arms: dict[str, Image.Image] = {}
        yy, xx = np.mgrid[0:h, 0:w]
        for name, polygon in spec.arms.items():
            exact = _polygon_mask((w, h), polygon)
            arms_union |= exact
            reach = exact | (_polygon_mask((w, h), polygon, grow=spec.root_margin) & deep)
            layer = base.copy()
            layer[..., 3] = np.where(reach, alpha, 0)
            if name in spec.cuts:
                # the arm's outline ends at the cut; once the arm turns those ends would stick
                # out as small ticks, so the ink next to the cut is painted body colour
                (x0, y0), (x1, y1) = spec.cuts[name]
                dx, dy = x1 - x0, y1 - y0
                t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
                near = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy)) < spec.cut_clear
                layer[near & (rgb.sum(axis=2) < 250) & reach, :3] = self.body_colour[:3]
            self.arms[name] = Image.fromarray(layer, "RGBA")
        body = base.copy()
        body[arms_union, 3] = 0
        self.body = Image.fromarray(body, "RGBA")
        self.pivots = spec.pivots
        self.joint_area = Image.new("L", (w, h), 0)
        for px, py in spec.pivots.values():
            r = spec.joint_radius
            ImageDraw.Draw(self.joint_area).ellipse((px - r, py - r, px + r, py + r), fill=255)

        ys, xs = np.nonzero(figure)
        self.top, self.bottom = ys.min(), ys.max()
        self.anchor = ((xs.min() + xs.max()) / 2, ys.max())  # bottom centre of the figure
        self.figure_height = ys.max() - ys.min()

    # --- the face -------------------------------------------------------------------

    def _draw_eye(self, layer: Image.Image, eye: _Feature, pose: Pose, colour_lid=None) -> None:
        lid_colour = colour_lid or self.body_colour
        open_ = 0.0 if pose.blink else min(max(pose.eye_open, 0.0), 1.0)
        ew, eh = eye.w * pose.eye_scale, eye.h * pose.eye_scale
        lw = max(2, round(eye.w * 0.13))
        pad = lw + 2
        size = (round(ew + 2 * pad), round(eh + 2 * pad))
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = (pad, pad, pad + ew, pad + eh)
        if open_ <= 0.08:  # shut: a lid line, curved like a happy closed eye
            cy = pad + eh * 0.55
            d.arc((pad - ew * 0.1, cy - eh * 0.3, pad + ew * 1.1, cy + eh * 0.3), 20, 160, fill=INK, width=lw)
        else:
            d.ellipse(box, fill=IRIS, outline=INK, width=lw)
            gx, gy = pose.gaze
            pw, ph = ew * 0.56, eh * 0.5
            px = pad + ew / 2 + gx * ew * 0.2 - pw / 2
            py = pad + eh * 0.55 + gy * eh * 0.18 - ph / 2
            pupil = Image.new("RGBA", size, (0, 0, 0, 0))
            pd = ImageDraw.Draw(pupil)
            pd.ellipse((px, py, px + pw, py + ph), fill=INK)
            r = ew * 0.11
            pd.ellipse((px + pw * 0.22 - r, py + ph * 0.2 - r, px + pw * 0.22 + r, py + ph * 0.2 + r),
                       fill=(250, 252, 252, 255))
            inside = Image.new("L", size, 0)
            ImageDraw.Draw(inside).ellipse((pad + lw, pad + lw, pad + ew - lw, pad + eh - lw), fill=255)
            img.paste(pupil, (0, 0), Image.fromarray(np.minimum(np.asarray(pupil.getchannel("A")), np.asarray(inside))))
            if open_ < 1:  # the lid comes down from the top
                lid_y = pad + eh * (1 - open_)
                cover = Image.new("L", size, 0)
                cd = ImageDraw.Draw(cover)
                cd.rectangle((0, 0, size[0], lid_y), fill=255)
                ellipse = Image.new("L", size, 0)
                ImageDraw.Draw(ellipse).ellipse(box, fill=255)
                m = np.minimum(np.asarray(cover), np.asarray(ellipse.filter(ImageFilter.MaxFilter(3))))
                img.paste(Image.new("RGBA", size, lid_colour), (0, 0), Image.fromarray(m))
                d = ImageDraw.Draw(img)
                half = math.sqrt(max(0.0, 1 - ((lid_y - pad - eh / 2) / (eh / 2)) ** 2)) * ew / 2
                d.line((pad + ew / 2 - half, lid_y, pad + ew / 2 + half, lid_y), fill=INK, width=lw)
        layer.alpha_composite(img, (round(eye.cx - size[0] / 2), round(eye.cy - size[1] / 2)))

    def _draw_mouth(self, layer: Image.Image, pose: Pose) -> None:
        m = self.mouth_feature
        if m is None:
            return
        if pose.mouth == "smile":
            image, at = self.mouth_smile
            layer.alpha_composite(image, (int(at[0]), int(at[1])))
            return
        d = ImageDraw.Draw(layer)
        lw = max(2, round(m.w * 0.07))
        if pose.mouth == "o":
            ow, oh = m.w * 0.42, m.h * 1.05
            box = (m.cx - ow / 2, m.cy - oh / 2, m.cx + ow / 2, m.cy + oh / 2)
            d.ellipse(box, fill=MOUTH_DARK, outline=INK, width=lw)
            d.chord((box[0] + lw, m.cy, box[2] - lw, box[3] - lw), 0, 180, fill=TONGUE)
        else:  # closed, content
            d.arc((m.cx - m.w * 0.45, m.cy - m.h * 0.9, m.cx + m.w * 0.45, m.cy + m.h * 0.3), 25, 155,
                  fill=INK, width=lw)

    def face_only(self, pose: Pose, canvas: tuple[int, int], lid_colour=(8, 8, 14, 255)) -> Image.Image:
        """Just the eyes, as they would sit on the figure in this pose (for eyes in the dark)."""
        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        for eye in self.eyes:
            self._draw_eye(layer, eye, pose, colour_lid=lid_colour)
        return self._place(layer, pose, canvas)

    # --- the body -------------------------------------------------------------------

    def _bend(self, layer: Image.Image, pose: Pose) -> Image.Image:
        """Shift each row sideways: the lower body trails and ripples, the head's curl swings."""
        if not (pose.drag or pose.wave or pose.tip):
            return layer
        a = np.asarray(layer).astype(np.float32)
        h, w = a.shape[:2]
        rows = np.arange(h)
        v = np.clip((rows - self.top) / max(self.bottom - self.top, 1), 0, 1)
        H = self.figure_height
        shift = H * (pose.drag * v ** 2 + pose.wave * np.sin(pose.wave_phase - 4 * v) * v ** 1.5)
        shift += H * pose.tip * np.clip((0.18 - v) / 0.18, 0, 1) ** 1.5
        src = np.arange(w)[None, :] - shift[:, None]
        x0 = np.floor(src).astype(int)
        f = (src - x0)[..., None]
        x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
        out = a[rows[:, None], x0c] * (1 - f) + a[rows[:, None], x1c] * f
        outside = (x0 < 0) | (x0 + 1 > w - 1)
        out[outside] = 0
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")

    def _place(self, layer: Image.Image, pose: Pose, canvas: tuple[int, int]) -> Image.Image:
        scale = pose.height / self.figure_height
        sx, sy = scale * (1 + pose.squash), scale * (1 - pose.squash)
        figure = layer.resize((max(1, round(self.size[0] * sx)), max(1, round(self.size[1] * sy))), Image.LANCZOS)
        ax, ay = self.anchor[0] * sx, self.anchor[1] * sy
        if pose.stretch:
            # stretch along a direction about the figure's middle, area kept
            pad = round(max(figure.size) * 0.35)
            big = Image.new("RGBA", (figure.width + 2 * pad, figure.height + 2 * pad), (0, 0, 0, 0))
            big.paste(figure, (pad, pad))
            cx, cy = big.width / 2, big.height / 2
            s = 1 + pose.stretch
            th = math.radians(-pose.stretch_angle)  # image y points down
            c, si = math.cos(th), math.sin(th)
            # inverse of R diag(s, 1/s) R^T  is  R diag(1/s, s) R^T
            a11 = c * c / s + si * si * s
            a12 = c * si * (1 / s - s)
            a22 = si * si / s + c * c * s
            big = big.transform(big.size, Image.AFFINE,
                                (a11, a12, cx - a11 * cx - a12 * cy, a12, a22, cy - a12 * cx - a22 * cy),
                                resample=Image.BICUBIC)
            figure, ax, ay = big, ax + pad, ay + pad
        if pose.tilt:
            figure = figure.rotate(pose.tilt, resample=Image.BICUBIC, center=(ax, ay))
        if pose.opacity < 1:
            figure.putalpha(figure.getchannel("A").point(lambda v: int(v * pose.opacity)))
        out = Image.new("RGBA", canvas, (0, 0, 0, 0))
        offset = (round(pose.x - ax), round(pose.y - ay))
        if pose.glow > 0:
            halo = Image.new("RGBA", figure.size, (190, 245, 255, 0))
            halo.putalpha(figure.getchannel("A").filter(ImageFilter.GaussianBlur(max(2, pose.height * 0.04)))
                          .point(lambda v: int(v * 0.55 * pose.glow)))
            paste(out, halo, offset)
        paste(out, figure, offset)
        return out

    def render(self, pose: Pose, canvas: tuple[int, int]) -> Image.Image:
        """The figure in this pose on a transparent canvas of the output size."""
        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        for name, arm in self.arms.items():
            angle = pose.arms.get(name, 0.0)
            layer.alpha_composite(arm.rotate(angle, resample=Image.BICUBIC, center=self.pivots[name]) if angle else arm)
        layer.alpha_composite(self.body)
        if any(pose.arms.values()):
            layer = _close_joints(self, layer)
        for i, eye in enumerate(self.eyes):
            self._draw_eye(layer, eye, replace(pose, eye_open=0.0) if pose.wink == i else pose)
        self._draw_mouth(layer, pose)
        return self._place(self._bend(layer, pose), pose, canvas)


def _close_joints(rig: Rig, layer: Image.Image) -> Image.Image:
    """Fill the small notches a turned arm leaves at the shoulder with body
    colour and ink their new edge. Works only in a box around each pivot (the
    rank filters are slow on the whole drawing)."""
    a = np.asarray(layer).copy()
    size = 2 * rig.spec.joint_close + 1
    r = rig.spec.joint_radius
    pad = size
    h, w = a.shape[:2]
    for px, py in rig.pivots.values():
        x0, y0, x1, y1 = max(0, px - r - pad), max(0, py - r - pad), min(w, px + r + pad), min(h, py + r + pad)
        box = a[y0:y1, x0:x1]
        solid = box[..., 3] > 127
        closed = _erode(_dilate(solid, size), size)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        joint = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
        new = closed & ~solid & joint
        if not new.any():
            continue
        box[new] = rig.body_colour
        edge = closed & ~_erode(closed, 5) & _dilate(new, 7) & joint
        box[edge] = INK
    return Image.fromarray(a, "RGBA")


def paste(dst: Image.Image, src: Image.Image, offset: tuple[int, int]) -> None:
    """Alpha-composite src onto dst at offset; parts outside dst are dropped."""
    x, y = offset
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(src.width, dst.width - x), min(src.height, dst.height - y)
    if right > left and bottom > top:
        dst.alpha_composite(src.crop((left, top, right, bottom)), (x + left, y + top))


def ease(t: float) -> float:
    """Ease in and out, 0..1 -> 0..1."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bob(t: float, period: float = 1.6, amount: float = 1.0) -> float:
    return math.sin(2 * math.pi * t / period) * amount


def spring(t: float, frequency: float = 3.0, damping: float = 5.0) -> float:
    """0 -> 1 with an overshoot that settles: for landings and takes."""
    if t <= 0:
        return 0.0
    return 1 - math.exp(-damping * t) * math.cos(2 * math.pi * frequency * t)
