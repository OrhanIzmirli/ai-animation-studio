"""The MILO teaser in HD: 1920 x 1056, 24 new drawings a second ("on ones"),
painted plates upscaled with Real-ESRGAN, 2.5D depth parallax, cel-shaded
Milo, motion blur, fireflies, moon rays and bloom.

    python scripts/make_milo_teaser_hd.py [--preview]

The acting, the timing and the cues are those of scripts/make_milo_teaser.py
(imported, so the two stay the same film); this script renders them at
cinema quality:
  - plates: outputs/hd_plates/*_1920x1056.png (scripts/upscale_images.py),
    their depth maps: outputs/hd_plates/depth/ (scripts/depth_maps.py);
  - camera moves become 2.5D: near pixels of a plate move and grow more than
    far ones, so trees slide past the sky;
  - Milo from the upscaled drawing (outputs/hd_plates/milo_cutout_v3_2048_clean.png),
    shaded in two cel tones (a shadow band away from the light, a highlight
    towards it) and blurred along his motion when he moves fast;
  - every frame is a new drawing, film grain and gate weave per frame.

Writes outputs/trailers/milo_03_hd/<timestamp>/teaser.mp4 and cues.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_studio.post.noise import smooth_noise  # noqa: E402
from ai_studio.puppet import Pose, Rig, RigSpec, ease, ease_in, lerp, paste, scaled  # noqa: E402

_spec = importlib.util.spec_from_file_location("milo_teaser", ROOT / "scripts/make_milo_teaser.py")
base = importlib.util.module_from_spec(_spec)
sys.modules["milo_teaser"] = base
_spec.loader.exec_module(base)

W, H = base.W, base.H              # world coordinates: the acting is written in these
OW, OH, FPS = 1920, 1056, 24       # output
K = OW / W                         # world -> output pixels at zoom 1
HD = ROOT / "outputs/hd_plates"
PLATES = {
    "forest": "milo_80s_v3_seed45_fc6b48b7", "bark": "milo_80s_v3_seed42_4dd781cf",
    "clearing": "milo_80s_v3_seed42_6008d06e", "sky": "milo_80s_v3_seed43_1b27d004",
    "castle": "milo_80s_v3_seed42_df9ad671",
}
DRAWING = HD / "milo_cutout_v3_2048_clean.png"
SPEC = scaled(base.SPEC, 2.0)      # the cut, on the 2048-pixel drawing
PARALLAX = 0.45                    # how much more the nearest pixels move than the plate as a whole


# --- plates with depth ------------------------------------------------------------------

@dataclass
class Plate:
    image: np.ndarray   # (OH, OW, 3) float32
    depth: np.ndarray   # (OH/2, OW/2) float32, 0 far .. 1 near


def load_plate(name: str) -> Plate:
    stem = PLATES[name]
    img = np.asarray(Image.open(HD / f"{stem}_1920x1056.png").convert("RGB"), dtype=np.float32)
    depth = np.asarray(Image.open(HD / "depth" / f"{stem}_1920x1056_depth.png"), dtype=np.float32) / 65535
    depth = np.asarray(Image.fromarray((depth * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                       dtype=np.float32) / 255
    return Plate(img, depth[::2, ::2].copy())  # half resolution: the map is smooth


def sample(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Bilinear sampling of img at float coordinates (clamped at the edges)."""
    h, w = img.shape[:2]
    sx, sy = np.clip(sx, 0, w - 1.001), np.clip(sy, 0, h - 1.001)
    x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@dataclass
class Camera:
    zoom: float = 1.0
    fx: float = W / 2
    fy: float = H / 2
    rest: tuple = (1.0, W / 2, H / 2)   # where the shot's camera started: parallax is the move away from it

    def window(self) -> tuple[float, float, float, float]:
        w, h = W / self.zoom, H / self.zoom
        return min(max(self.fx - w / 2, 0), W - w), min(max(self.fy - h / 2, 0), H - h), w, h

    def view(self, plate: Plate) -> Image.Image:
        """The plate through the camera, with depth parallax: near pixels move and grow more."""
        left, top, w, h = self.window()
        ys, xs = np.mgrid[0:OH, 0:OW].astype(np.float32)
        u, v = xs / OW, ys / OH
        # where each output pixel looks in the plate, before parallax (plate pixels)
        px, py = (left + u * w) * K, (top + v * h) * K
        z0, fx0, fy0 = self.rest
        mx, my = (self.fx - fx0) * K, (self.fy - fy0) * K   # camera travel, plate pixels
        grow = (self.zoom - z0) * 0.35
        d = sample(plate.depth[..., None], px[::2, ::2] / 2, py[::2, ::2] / 2)[..., 0]
        d = np.repeat(np.repeat(d, 2, axis=0), 2, axis=1)[:OH, :OW]
        cx, cy = (left + w / 2) * K, (top + h / 2) * K
        sx = cx + (px - cx) / (1 + grow * d) + mx * PARALLAX * d
        sy = cy + (py - cy) / (1 + grow * d) + my * PARALLAX * d
        out = sample(plate.image, sx, sy)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

    def pose(self, pose: Pose) -> Pose:
        left, top, _, _ = self.window()
        return replace(pose, x=(pose.x - left) * self.zoom * K, y=(pose.y - top) * self.zoom * K,
                       height=pose.height * self.zoom * K)

    def point(self, x: float, y: float) -> tuple[float, float]:
        left, top, _, _ = self.window()
        return (x - left) * self.zoom * K, (y - top) * self.zoom * K

    @property
    def scale(self) -> float:
        return self.zoom * K

    def foreground(self, frame: Image.Image, layer: Image.Image, depth: float = 1.6) -> None:
        left, top, w, h = self.window()
        z = 1 + (self.zoom - 1) * depth
        fw, fh = layer.size
        lk = fw / (W * 1.3)
        ox = (fw - W * lk) / 2 + (left - (W - w) / 2) * depth * lk
        oy = (fh - H * lk) / 2 + (top - (H - h) / 2) * depth * lk
        bw, bh = W * lk / z, H * lk / z
        box = (ox + (W * lk - bw) / 2, oy + (H * lk - bh) / 2, ox + (W * lk + bw) / 2, oy + (H * lk + bh) / 2)
        frame.alpha_composite(layer.resize((OW, OH), Image.BICUBIC, box=box))


def grade_plates(raw: dict) -> dict:
    ref = raw["forest"].image.reshape(-1, 3)
    stats = (ref.mean(0), ref.std(0))
    out = {}
    for name, plate in raw.items():
        img = base.grade(Image.fromarray(plate.image.astype(np.uint8)), stats, 0.0 if name == "forest" else 0.35)
        out[name] = Plate(np.asarray(img.convert("RGB"), dtype=np.float32), plate.depth)
    return out


def derived(plate: Plate, fn) -> Plate:
    img = fn(Image.fromarray(plate.image.astype(np.uint8)).convert("RGBA"))
    return Plate(np.asarray(img.convert("RGB"), dtype=np.float32), plate.depth)


# --- Milo -------------------------------------------------------------------------------

def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur in float (no 8-bit steps, which the shading would turn into ripples)."""
    r = max(1, int(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, x)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, x)


def cel_shade(figure: Image.Image, light=(-0.6, -0.8), shadow=(0.72, 0.78, 0.94), highlight=1.1,
              rim=(215, 245, 255), rim_strength=0.6, ambient=(1.0, 1.0, 1.0)) -> Image.Image:
    """Two-tone cel shading from the silhouette: a crisp shadow shape on the side away from
    the light, a highlight on the side towards it, a thin rim of light. Worked out in float at
    a quarter of the size, then scaled back up."""
    a = np.asarray(figure).astype(np.float32)
    alpha = a[..., 3] / 255
    ys, xs = np.nonzero(alpha > 0.5)
    if not len(ys):
        return figure
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + pad)
    crop = alpha[y0:y1, x0:x1]
    q = 4
    small = np.asarray(Image.fromarray(crop).resize((max(1, (x1 - x0) // q), max(1, (y1 - y0) // q)), Image.BOX),
                       dtype=np.float32)
    size = max(ys.max() - ys.min(), 1) / q
    sigma = max(1.0, size * 0.09)
    soft = _gauss(small, sigma)
    gy, gx = np.gradient(soft)
    norm = np.hypot(gx, gy) + 1e-9
    lx, ly = light
    near_edge = np.clip(norm * sigma * 2.5, 0, 1)
    facing = -(gx * lx + gy * ly) / norm * near_edge
    shade = np.clip((-facing - 0.08) / 0.06, 0, 1)
    lit = np.clip((facing - 0.3) / 0.08, 0, 1)
    up = lambda m: np.asarray(Image.fromarray(m.astype(np.float32)).resize((x1 - x0, y1 - y0), Image.BILINEAR))  # noqa: E731
    shade_full, lit_full = np.zeros_like(alpha), np.zeros_like(alpha)
    shade_full[y0:y1, x0:x1], lit_full[y0:y1, x0:x1] = up(shade), up(lit)
    rgb = a[..., :3] * np.array(ambient, np.float32)
    ink = a[..., :3].sum(axis=2) < 120
    tone = 1 + (np.array(shadow, np.float32) - 1) * shade_full[..., None]
    tone = tone * (1 + (highlight - 1) * lit_full[..., None])
    rgb = np.where(ink[..., None], rgb, rgb * tone)
    edge = np.clip(alpha - np.roll(alpha, (int(-ly * 5), int(-lx * 5)), axis=(0, 1)), 0, 1)
    rgb = rgb + (np.array(rim, np.float32) - rgb) * (edge * rim_strength)[..., None]
    a[..., :3] = np.clip(rgb, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def draw_milo(s, frame: Image.Image, cam: Camera, pose_at, t: float, light: dict, ground: float | None = None,
              blur_from: float = 0.15) -> None:
    """Milo at time t through the camera: cel-shaded, glowing on the ground, and blurred along
    his motion (three sub-frame poses averaged) when he moves fast."""
    pose = pose_at(t)
    if pose is None:
        return
    p = cam.pose(pose)
    if ground is not None:
        gx, gy = cam.point(pose.x, ground)
        lift = max(0.0, 1 - (ground - pose.y) / 300)
        base_glow(frame, gx, gy, p.height, 0.9 * lift * pose.opacity)
    prev, nxt = pose_at(t - 1 / 72), pose_at(t + 1 / 72)
    speed = 0.0 if prev is None or nxt is None else math.hypot(nxt.x - prev.x, nxt.y - prev.y) / max(pose.height, 1)
    if speed > blur_from:
        layers = [s.rig.render(cam.pose(q), (OW, OH)) for q in (prev, pose, nxt)]
        acc = sum(np.asarray(l, dtype=np.float32) for l in layers) / 3
        figure = Image.fromarray(acc.astype(np.uint8), "RGBA")
    else:
        figure = s.rig.render(p, (OW, OH))
    frame.alpha_composite(cel_shade(figure, **light))


def base_glow(frame: Image.Image, x: float, y: float, height: float, strength: float) -> None:
    if strength <= 0:
        return
    light = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    rx, ry = height * 0.55, height * 0.12
    ImageDraw.Draw(light).ellipse((x - rx, y - ry, x + rx, y + ry), fill=(170, 240, 255, int(110 * strength)))
    frame.alpha_composite(light.filter(ImageFilter.GaussianBlur(max(2, height * 0.08))))


# --- atmosphere ---------------------------------------------------------------------------

def add_fog(frame: Image.Image, fog: np.ndarray, t: float, strength: float, speed: float = 40) -> None:
    n = np.roll(fog, -int(t * speed * K) % OW, axis=1)[:, :OW]
    ramp = np.clip((np.arange(OH) / OH - 0.35) / 0.65, 0, 1)[:, None]
    layer = Image.new("RGBA", (OW, OH), (175, 205, 215, 0))
    layer.putalpha(Image.fromarray(np.clip(n * ramp * strength * 255, 0, 255).astype(np.uint8)))
    frame.alpha_composite(layer)


def add_stars(frame: Image.Image, cam: Camera, stars: list, t: float, region_bottom: float) -> None:
    d = ImageDraw.Draw(frame)
    for x, y, phase, size in stars:
        if y > region_bottom:
            continue
        sx, sy = cam.point(x, y)
        if not (0 <= sx < OW and 0 <= sy < OH):
            continue
        b = 0.5 + 0.5 * math.sin(2 * math.pi * (t * 0.8 + phase))
        if b < 0.55:
            continue
        r, c = size * b * K, (255, 250, 220, int(255 * b))
        d.line((sx - 2 * r, sy, sx + 2 * r, sy), fill=c, width=1)
        d.line((sx, sy - 2 * r, sx, sy + 2 * r), fill=c, width=1)
        d.ellipse((sx - r / 2, sy - r / 2, sx + r / 2, sy + r / 2), fill=c)


def fireflies(frame: Image.Image, cam: Camera, flies: list, t: float, area: tuple) -> None:
    """Small warm lights that drift on lazy curves and pulse."""
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = area
    for fx, fy, ph, sp in flies:
        x = x0 + (fx + 0.04 * math.sin(t * sp + ph * 6)) % 1 * (x1 - x0)
        y = y0 + (fy + 0.05 * math.sin(t * sp * 0.7 + ph * 9)) % 1 * (y1 - y0)
        b = max(0.0, math.sin(t * 2.2 + ph * 20))
        if b < 0.1:
            continue
        sx, sy = cam.point(x, y)
        r = 6.5 * cam.scale * (0.6 + 0.4 * b)
        d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 240, 150, int(255 * b)))
    glow = layer.filter(ImageFilter.GaussianBlur(6 * K))
    frame.alpha_composite(glow)
    frame.alpha_composite(glow)
    frame.alpha_composite(layer)


def moon_rays(frame: Image.Image, cam: Camera, moon: tuple, t: float, strength: float = 1.0) -> None:
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    mx, my = cam.point(*moon)
    for k in range(9):
        a = 2.0 + 0.14 * k + 0.03 * math.sin(t * 0.6 + k)
        spread, length = 0.035, 1400 * K
        d.polygon([(mx, my), (mx + math.cos(a - spread) * length, my + math.sin(a - spread) * length),
                   (mx + math.cos(a + spread) * length, my + math.sin(a + spread) * length)],
                  fill=(255, 245, 210, int(20 * strength * (0.6 + 0.4 * math.sin(t * 1.3 + k * 2)))))
    frame.alpha_composite(layer.filter(ImageFilter.GaussianBlur(14 * K)))


def bloom(frame: np.ndarray) -> np.ndarray:
    """Light that spills from the brightest parts (moon, glows, logo)."""
    lum = frame.mean(axis=2)
    bright = np.clip((lum - 175) / 80, 0, 1)[..., None] * frame
    small = Image.fromarray(bright.astype(np.uint8)).resize((OW // 4, OH // 4), Image.BILINEAR)
    halo = np.asarray(small.filter(ImageFilter.GaussianBlur(10)).resize((OW, OH), Image.BILINEAR), dtype=np.float32)
    return np.clip(frame + halo * 0.6, 0, 255)


def film(frame: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(5000 + n)
    if n % 2 == 0:  # gate weave changes every other frame: a gentle float, not a jitter
        dx, dy = np.random.default_rng(9000 + n // 2).integers(-1, 2, size=2)
        frame = np.roll(frame, (dy, dx), axis=(0, 1))
    grain = rng.normal(0, 4.5, frame.shape[:2])[..., None]
    return np.clip(frame * VIGNETTE + grain, 0, 255)


yy, xx = np.mgrid[0:OH, 0:OW]
VIGNETTE = (1 - 0.24 * (((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2))[..., None].astype(np.float32)
del yy, xx


# --- shots --------------------------------------------------------------------------------

@dataclass
class Scene:
    plates: dict
    rig: Rig
    tc: object
    fonts: tuple
    fog: np.ndarray
    stars: list
    flies: list
    fg: dict
    letters: list


def shot_forest(s: Scene, t: float) -> Image.Image:
    k = ease(t / 3.0)
    cam = Camera(zoom=lerp(1.25, 1.1, k), fx=lerp(700, 640, k), fy=lerp(230, 400, k), rest=(1.25, 700, 230))
    frame = cam.view(s.plates["forest"])
    moon_rays(frame, cam, (1015, 110), t)
    add_stars(frame, cam, s.stars, t, 260)
    add_fog(frame, s.fog, t, 0.5)
    fireflies(frame, cam, s.flies[:14], t, (150, 380, 1150, 690))
    cam.foreground(frame, s.fg["forest"])
    return frame


def shot_hollow(s: Scene, t: float) -> Image.Image:
    cam = Camera(zoom=1 + 0.06 * ease(t / 3.5), fx=640, fy=345, rest=(1.0, 640, 345))
    frame = cam.view(s.plates["bark_hollow"])
    if t >= 0.75:
        open_ = ease((t - 0.75) / 0.3) * base.blinking(t, [1.6])
        gaze = (0.0, 0.0)
        if 1.9 <= t < 2.4:
            gaze = (-ease((t - 1.9) / 0.15), 0.0)
        elif 2.4 <= t < 2.9:
            gaze = (ease((t - 2.4) / 0.15), 0.0)
        if t >= 3.0:
            open_ = lerp(1, 0.5, ease((t - 3.0) / 0.2))
        x, y = base.place_eyes(s.rig, (640, 330), 1500)
        pose = cam.pose(Pose(x, y, 1500, eye_open=max(open_, 0.0), gaze=gaze))
        glow = s.rig.face_only(pose, (OW, OH))
        halo = Image.new("RGBA", (OW, OH), (60, 230, 220, 0))
        halo.putalpha(glow.getchannel("A").filter(ImageFilter.GaussianBlur(20)).point(lambda v: int(v * 0.75)))
        frame.alpha_composite(halo)
        frame.alpha_composite(glow)
    return frame


ARRIVE_LIGHT = {"ambient": (0.96, 1.0, 1.03), "rim": (215, 245, 255), "light": (0.55, -0.83), "rim_strength": 0.7}
SKY_LIGHT = {"ambient": (1.0, 0.95, 1.02), "rim": (255, 215, 235), "light": (0.6, -0.8), "shadow": (0.74, 0.70, 0.92)}
DANGER_LIGHT = {"ambient": (0.80, 0.76, 0.97), "rim": (200, 140, 255), "light": (1.0, -0.2),
                "shadow": (0.62, 0.58, 0.86), "rim_strength": 0.85}
CASTLE_LIGHT = {"ambient": (0.93, 0.93, 1.0), "rim": (255, 235, 200), "light": (-0.4, -0.9)}


def shot_arrive(s: Scene, t: float) -> Image.Image:
    push = ease((t - 0.9) / 0.8) - ease((t - 3.4) / 0.6)
    cam = Camera(zoom=lerp(1.05, 1.75, push), fx=640, fy=lerp(352, 395, push), rest=(1.05, 640, 352))
    frame = cam.view(s.plates["clearing"])
    add_stars(frame, cam, s.stars, t, 200)
    add_fog(frame, s.fog, t + 3, 0.3, speed=25)
    fireflies(frame, cam, s.flies[14:30], t, (80, 300, 1200, 600))
    draw_milo(s, frame, cam, base.arrive_pose, t, ARRIVE_LIGHT, ground=665)
    cam.foreground(frame, s.fg["clearing"])
    return frame


def shot_sky(s: Scene, t: float) -> Image.Image:
    cam = Camera(zoom=1.04, fx=640 + 20 * math.sin(t), fy=352, rest=(1.04, 640, 352))
    frame = cam.view(s.plates["sky"])
    add_stars(frame, cam, s.stars, t, 420)
    for k in range(1, 9):
        back = base.sky_pose(max(0.0, t - k * 0.06))
        px, py = cam.point(back.x, back.y - 125)
        base.sparkle(frame, px, py, (12 - k) * K, alpha=1 - k / 9)
    draw_milo(s, frame, cam, base.sky_pose, t, SKY_LIGHT, blur_from=0.35)
    return frame


def shot_danger(s: Scene, t: float) -> Image.Image:
    tension = ease(t / 2.4)
    cam = Camera(zoom=lerp(1.05, 1.14, tension), fx=lerp(640, 600, tension), fy=lerp(352, 380, tension),
                 rest=(1.05, 640, 352))
    frame = cam.view(s.plates["clearing_dark"])
    add_fog(frame, s.fog, t + 7, 0.5, speed=60)
    if t >= 0.6:
        reach_in = ease((t - 0.6) / 1.6)
        wx, wy = lerp(W + 250, 860, reach_in), lerp(-150, 330, reach_in)
        curl, reach = 0.15 + 0.12 * math.sin(t * 5), 0.8 + 0.2 * reach_in
        if t >= 3.0:
            k = ease((t - 3.0) / 0.3)
            wx, wy, curl = lerp(860, 700, k), lerp(330, 400, k), lerp(0.3, 1.0, k)
        if t >= 3.7:
            k = ease((t - 3.7) / 0.8)
            wx, wy, curl = lerp(700, W + 300, k), lerp(400, -200, k), 1.0
        hand = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        base.shadow_hand(hand, base.Camera(zoom=cam.zoom, fx=cam.fx, fy=cam.fy).point(wx, wy + 12 * math.sin(t * 2.3)),
                         reach * cam.zoom, curl, t * 6)
        frame.alpha_composite(hand.resize((OW, OH), Image.BICUBIC))
    pose, info = base.danger_pose(t)
    if info["lines"]:
        for k, a in ((2, 0.18), (1, 0.32)):
            back, _ = base.danger_pose(t - k * 0.05)
            if back is not None:
                frame.alpha_composite(s.rig.render(cam.pose(replace(back, opacity=a, glow=0)), (OW, OH)))
        lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sx, sy = base.Camera(zoom=cam.zoom, fx=cam.fx, fy=cam.fy).point(info["x"] + 200, info["y"])
        base.speed_lines(lines, sx, sy, 400 * cam.zoom, info["vx"], info["vy"], seed=int(t * 48))
        frame.alpha_composite(lines.resize((OW, OH), Image.BICUBIC))
    draw_milo(s, frame, cam, lambda q: base.danger_pose(q)[0], t, DANGER_LIGHT, ground=655, blur_from=0.3)
    dust = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    base.dust(dust, *base.Camera(zoom=cam.zoom, fx=cam.fx, fy=cam.fy).point(470, 650), t - 2.9)
    frame.alpha_composite(dust.resize((OW, OH), Image.BICUBIC))
    cam.foreground(frame, s.fg["danger"])
    if 2.55 <= t < 2.95:
        frame = base.shaken(frame, 16 * K * (1 - (t - 2.55) / 0.4), seed=int(t * 48))
    return frame


def castle_pose(t: float) -> Pose | None:
    fly = 1.3
    if t >= fly:
        return None
    u = ease_in(t / fly) * 0.6 + 0.4 * (t / fly)
    return Pose(lerp(1450, base.WINDOW[0], u) + 120 * math.sin(math.pi * u),
                lerp(120, base.WINDOW[1] + 20, u) - 120 * math.sin(math.pi * u),
                lerp(300, 40, u), stretch=0.3, stretch_angle=200, drag=0.15, wave=0.05, wave_phase=t * 12,
                gaze=(-1, 0.3), arms={"left": -25, "right": 25}, glow=0.8, opacity=1 - max(0, (u - 0.9) / 0.1))


def shot_castle(s: Scene, t: float) -> Image.Image:
    k0 = ease(t / 2.5)
    cam = Camera(zoom=lerp(1.04, 1.12, k0), fx=lerp(640, base.WINDOW[0] + 60, k0), fy=lerp(352, 420, k0),
                 rest=(1.04, 640, 352))
    frame = cam.view(s.plates["castle_lit"])
    moon_rays(frame, cam, (575, 190), t, 0.7)
    add_stars(frame, cam, s.stars, t, 300)
    if t < 1.3:
        draw_milo(s, frame, cam, castle_pose, t, CASTLE_LIGHT, blur_from=0.6)
    else:
        k = t - 1.3
        wx, wy = cam.point(*base.WINDOW)
        glow = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
        r = 36 * cam.scale
        ImageDraw.Draw(glow).ellipse((wx - r, wy - r * 1.2, wx + r, wy + r * 0.8), fill=(255, 210, 120, int(210 * math.exp(-k))))
        frame.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14 * K)))
        if k < 0.6:
            for j in range(8):
                ang, rr = 2 * math.pi * j / 8 + 0.3, (25 + 170 * k) * cam.scale
                base.sparkle(frame, wx + rr * math.cos(ang), wy + rr * math.sin(ang), (14 * (1 - k / 0.6) + 3) * K,
                             alpha=1 - k / 0.6)
    return frame


def shot_title(s: Scene, t: float) -> Image.Image:
    cam = Camera(zoom=1.0, fx=W / 2, fy=H / 2, rest=(1.0, W / 2, H / 2))
    frame = cam.view(s.plates["castle_dark"])
    add_stars(frame, cam, s.stars, t, 380)
    pose = base.title_pose(t)
    behind = t < 1.2
    if behind:
        draw_milo(s, frame, cam, base.title_pose, t, CASTLE_LIGHT, blur_from=0.5)
    if t < 1.9:
        for k in range(1, 7):
            back = base.title_pose(max(0.0, t - k * 0.05))
            px, py = cam.point(back.x - 30, back.y - 130)
            base.sparkle(frame, px, py, (10 - k) * K, alpha=1 - k / 7)
    for (image, offset), at in zip(s.letters, base.LETTER_TIMES):
        if t < at:
            continue
        u = t - at
        scale = 1 - math.exp(-7 * u) * math.cos(2 * math.pi * 2.2 * u)
        if scale <= 0.02:
            continue
        img = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
        lx, ly = OW // 2 + offset[0], int(OH * 0.42) + offset[1]
        paste(frame, img, (round(lx - img.width / 2), round(ly - img.height / 2)))
    if t >= 0.9:
        sub = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
        s.tc.glow_text(sub, (OW // 2, int(OH * 0.74)), "Coming this Saturday Morning", s.fonts[1], (255, 236, 200),
                       s.tc.ORANGE, radius=12, outline=3)
        sub.putalpha(sub.getchannel("A").point(lambda v: int(v * ease((t - 0.9) / 0.4))))
        frame.alpha_composite(sub)
    if not behind:
        draw_milo(s, frame, cam, base.title_pose, t, CASTLE_LIGHT)
    if t >= 2.6:
        k = ease_in((t - 2.6) / 0.7)
        p = cam.pose(pose)
        fx, fy = p.x, p.y - p.height * 0.72
        r = lerp(1400, 0, k)
        mask = Image.new("L", (OW, OH), 255)
        if r > 1:
            ImageDraw.Draw(mask).ellipse((fx - r, fy - r, fx + r, fy + r), fill=0)
        frame.paste(Image.new("RGBA", (OW, OH), (0, 0, 0, 255)), (0, 0), mask)
    return frame


SHOT_FUNCTIONS = {"forest": shot_forest, "hollow": shot_hollow, "arrive": shot_arrive, "sky": shot_sky,
                  "danger": shot_danger, "castle": shot_castle, "title": shot_title}


def render(s: Scene, t: float) -> Image.Image:
    name, s0, _ = next(x for x in base.SHOTS if x[1] <= t < x[2])
    return SHOT_FUNCTIONS[name](s, t - s0)


def build_scene() -> Scene:
    spec = importlib.util.spec_from_file_location("title_card", ROOT / "scripts/make_title_card.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    plates = grade_plates({name: load_plate(name) for name in PLATES})

    def hollow(img: Image.Image) -> Image.Image:  # the hollow is drawn in world size, then scaled up
        world = base.draw_hollow(img.resize((W, H), Image.LANCZOS))
        mask = Image.new("L", (W, H), 0)
        diff = np.abs(np.asarray(world.convert("RGB"), dtype=np.int16)
                      - np.asarray(img.resize((W, H), Image.LANCZOS).convert("RGB"), dtype=np.int16)).sum(axis=2)
        mask = Image.fromarray((diff > 6).astype(np.uint8) * 255).resize((OW, OH), Image.BICUBIC)
        out = img.copy()
        out.paste(world.resize((OW, OH), Image.LANCZOS), (0, 0), mask.filter(ImageFilter.GaussianBlur(1)))
        return out

    plates["bark_hollow"] = derived(plates["bark"], hollow)

    def dark(img: Image.Image) -> Image.Image:
        d = ImageEnhance.Brightness(img).enhance(0.55)
        d.alpha_composite(Image.new("RGBA", img.size, (70, 20, 110, 70)))
        return d

    plates["clearing_dark"] = derived(plates["clearing"], dark)
    plates["castle_lit"] = derived(plates["castle"], lambda i: ImageEnhance.Brightness(i).enhance(0.9))
    plates["castle_dark"] = derived(plates["castle"], lambda i: ImageEnhance.Brightness(i).enhance(0.42))
    fog = smooth_noise((OH, 2 * OW), 330, seed=11, salt=3)
    fog = np.concatenate([fog, fog[:, ::-1]], axis=1)
    fog = np.clip((fog - fog.min()) / (fog.max() - fog.min()) * 1.4 - 0.45, 0, 1)
    rng = np.random.default_rng(5)
    stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0, 1), rng.uniform(2, 4)) for _ in range(55)]
    flies = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.4, 1.0)) for _ in range(30)]
    fonts = (ImageFont.truetype(str(tc.FONTS / tc.TITLE_FONT), int(OH * 0.34)),
             ImageFont.truetype(str(tc.FONTS / tc.SUBTITLE_FONT), int(OH * 0.075)))
    fg = {}
    for name, args in {"forest": (21, True, False, True), "clearing": (22, True, True, True),
                       "danger": (23, False, True, True)}.items():
        layer = base.foreground(*args)
        fg[name] = layer.resize((round(layer.width * K), round(layer.height * K)), Image.LANCZOS)
    letters = []
    for image, (ox, oy) in base.letter_images(tc, fonts[0]):
        letters.append((image, (ox, oy)))
    return Scene(plates, Rig(Image.open(DRAWING), SPEC, scale=0.75), tc, fonts, fog, stars, flies, fg, letters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--only", type=float, nargs=2, default=None, metavar=("FROM", "TO"),
                        help="render only these seconds (with --preview: every 6th frame)")
    args = parser.parse_args()
    scene = build_scene()
    total = int(base.SHOTS[-1][2] * FPS)
    out_dir = ROOT / "outputs/trailers/milo_03_hd" / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True)
    (out_dir / "cues.json").write_text(json.dumps(base.CUES, indent=2) + "\n", encoding="utf-8")
    first, last = (0, total) if args.only is None else (int(args.only[0] * FPS), int(args.only[1] * FPS))
    step = (6 if args.only else 12) if args.preview else 1
    encoder = None
    if not args.preview:
        encoder = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
             str(out_dir / "teaser.mp4")], stdin=subprocess.PIPE)
    previews = []
    started = time.perf_counter()
    for n in range(first, last, step):
        t = n / FPS
        frame = np.asarray(render(scene, t).convert("RGB"), dtype=np.float32)
        frame = film(bloom(frame), n)
        level = 1.0
        for name, s0, s1 in base.SHOTS:
            a, b = round(s0 * FPS), round(s1 * FPS)
            if a <= n < b:
                level = min(1.0, (n - a + 1) / (12 if a == 0 else base.DIP))
                if name != "title":
                    level = min(level, (b - n) / base.DIP)
        frame = (frame * level).astype(np.uint8)
        if args.preview:
            previews.append(frame)
        else:
            encoder.stdin.write(frame.tobytes())
        if n % 48 == 0:
            print(f"  {t:5.1f}s  {time.perf_counter() - started:5.0f}s elapsed", flush=True)
    if encoder:
        encoder.stdin.close()
        encoder.wait()
        print(f"video: {(out_dir / 'teaser.mp4').relative_to(ROOT)}  ({last - first} frames, {time.perf_counter() - started:.0f} s)")
    else:
        cols = 4
        sheet = Image.new("RGB", (cols * 480, ((len(previews) + cols - 1) // cols) * 264))
        for i, f in enumerate(previews):
            sheet.paste(Image.fromarray(f).resize((480, 264)), ((i % cols) * 480, (i // cols) * 264))
        sheet.save(out_dir / "preview.png")
        print(f"preview: {(out_dir / 'preview.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
