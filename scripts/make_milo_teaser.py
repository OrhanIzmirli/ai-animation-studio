"""Make the MILO teaser as cut-out (cel) animation: painted plates plus Milo's
approved drawing animated as a puppet (ai_studio.puppet). No video model.

    python scripts/make_milo_teaser.py [--preview]

Direction, beat by beat, like a 1980s TV cartoon:
  - one colour grade over every plate, so the film reads as one piece;
  - Milo lit by each scene: ambient tint, a moonlight rim, his glow on the ground;
  - a multiplane camera: out-of-focus foreground branches and grass move
    faster than the plate when the camera moves (parallax);
  - acting: anticipation before big moves, squash and stretch, cloth-like
    follow-through, eyes that look, blink and wink, a fright take, a zip-off
    with speed lines and after-images, a loop-the-loop;
  - a push-in to a close-up for "Meet Milo!", a tilt down from the moon;
  - title letters that bounce in, and an iris-out on Milo's wink.
12 drawings a second, held on twos.

Writes outputs/trailers/milo_03/<timestamp>/teaser.mp4 (silent) and
cues.json (sound effects, narration, music and ambience times; read by
scripts/make_milo_audio.py). --preview writes a contact sheet of every 12th frame.
"""

from __future__ import annotations

import argparse
import glob
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
from ai_studio.puppet import Pose, Rig, RigSpec, bob, ease, ease_in, ease_out, lerp, paste  # noqa: E402

W, H, FPS = 1280, 704, 24
DRAWING = ROOT / "benchmarks/references/milo/milo_cutout_v3.png"
SPEC = RigSpec(  # the cut of milo_cutout_v3.png (1024 x 1024)
    background=(128, 128, 128),
    arms={"left": [(180, 280), (330, 280), (410, 330), (445, 352), (388, 512), (180, 512)],
          "right": [(625, 342), (700, 270), (840, 270), (840, 500), (640, 485)]},
    pivots={"left": (425, 440), "right": (632, 420)},
    cuts={"left": ((445, 352), (388, 512)), "right": ((625, 342), (640, 485))},
    eyes=[(505, 252), (576, 260)], mouth=(510, 350), body_sample=(520, 600),
)
PLATES = {
    "forest": "outputs/keyframes/milo_02/shot_01_start/milo_80s_v3_seed45_*.png",
    "bark": "outputs/keyframes/milo_03/plate_hollow/milo_80s_v3_seed42_*.png",
    "clearing": "outputs/keyframes/milo_03/plate_clearing/milo_80s_v3_seed42_*.png",
    "sky": "outputs/keyframes/milo_03/plate_sky/milo_80s_v3_seed43_*.png",
    "castle": "outputs/keyframes/milo_02/shot_05_plate/milo_80s_v3_seed42_*.png",
}
SHOTS = [  # (name, start s, end s)
    ("forest", 0.0, 3.0), ("hollow", 3.0, 6.5), ("arrive", 6.5, 11.5), ("sky", 11.5, 14.5),
    ("danger", 14.5, 19.0), ("castle", 19.0, 21.5), ("title", 21.5, 25.0),
]
DIP = 4
WINDOW = (505, 500)  # the arched window in the castle plate
LETTER_TIMES = [0.15, 0.3, 0.45, 0.6]
CUES = {
    "narration": [
        [0.2, "Deep in the shadowy forest, a secret is hiding..."],
        [4.1, "...but it's not as scary as you think!"],
        [7.6, "Meet Milo!"],
        [14.9, "But dark forces are after him..."],
        [18.6, "...and Milo's heart is bigger than any fear!"],
        [21.7, "Milo! Coming this Saturday morning!"],
    ],
    "sfx": [
        [1.0, "owl"], [3.8, "twinkle"], [6.5, "whoosh_up"], [6.9, "boing"], [9.65, "ding"],
        [10.9, "whoosh_up"], [11.6, "whoosh"], [12.6, "loop"], [14.0, "whoosh"], [15.1, "sting"],
        [16.5, "uh_oh"], [17.05, "boing_take"], [17.4, "zip"], [17.8, "grab"],
        [19.2, "whoosh"], [20.35, "pop"],
        *[[21.5 + t, "pop"] for t in LETTER_TIMES], [22.4, "chime"], [23.9, "ding"], [24.1, "iris"],
    ],
    "music": {"mystery": [0.0, 6.5], "magic": [6.5, 14.5], "danger": [14.5, 19.0], "hero": [19.0, 21.5],
              "title": [21.5, 25.0]},
    "ambience": {"wind": [[0.0, 6.5], [14.5, 19.0]], "crickets": [[0.0, 11.5]]},
}


# --- the look -----------------------------------------------------------------------------

def grade(img: Image.Image, reference_stats, strength: float = 0.35) -> Image.Image:
    """Pull a plate's colours part of the way to the reference plate's (mean and spread per
    channel), then one shared night curve: deep blue shadows, slightly warm highlights."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    mean, std = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
    rmean, rstd = reference_stats
    a = a * (1 - strength) + ((a - mean) / std * rstd + rmean) * strength
    lum = a.mean(axis=2, keepdims=True) / 255
    shadows = np.array([10, 16, 38], np.float32)
    highlights = np.array([255, 240, 215], np.float32)
    a = a + (shadows - a) * 0.18 * (1 - lum) ** 2 + (highlights - a) * 0.06 * lum ** 3
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")


def integrate(figure: Image.Image, ambient=(1.0, 1.0, 1.0), rim=(210, 240, 255), rim_dir=(-1, -1),
              rim_strength: float = 0.8) -> Image.Image:
    """Light Milo with the scene: tint by its ambient light, add a rim on the moon side."""
    a = np.asarray(figure).astype(np.float32)
    rgb, alpha = a[..., :3] * np.array(ambient, np.float32), a[..., 3] / 255
    dx, dy = rim_dir
    edge = np.clip(alpha - np.roll(alpha, (int(-dy * 4), int(-dx * 4)), axis=(0, 1)), 0, 1)
    edge = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))) / 255
    rgb = rgb + (np.array(rim, np.float32) - rgb) * (edge * rim_strength)[..., None]
    a[..., :3] = np.clip(rgb, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def ground_glow(frame: Image.Image, x: float, y: float, height: float, strength: float) -> None:
    if strength <= 0:
        return
    light = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rx, ry = height * 0.55, height * 0.12
    ImageDraw.Draw(light).ellipse((x - rx, y - ry, x + rx, y + ry), fill=(170, 240, 255, int(110 * strength)))
    frame.alpha_composite(light.filter(ImageFilter.GaussianBlur(max(2, height * 0.08))))


def branch(d: ImageDraw.ImageDraw, x, y, angle, length, width, depth, rng) -> None:
    if depth == 0 or length < 6:
        return
    x2, y2 = x + math.cos(angle) * length, y + math.sin(angle) * length
    d.line((x, y, x2, y2), fill=(6, 8, 18, 255), width=max(1, int(width)))
    d.ellipse((x2 - width / 2, y2 - width / 2, x2 + width / 2, y2 + width / 2), fill=(6, 8, 18, 255))
    for turn in (-1, 1):
        branch(d, x2, y2, angle + turn * rng.uniform(0.25, 0.6), length * rng.uniform(0.6, 0.78), width * 0.66,
               depth - 1, rng)


def foreground(seed: int, left: bool, right: bool, grass: bool) -> Image.Image:
    """An out-of-focus dark foreground (branches at the top corners, grass at the bottom),
    larger than the frame so it can slide for parallax."""
    fw, fh = int(W * 1.3), int(H * 1.3)
    layer = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rng = np.random.default_rng(seed)
    top, bottom = (fh - H) // 2, fh - (fh - H) // 2  # the part of the layer the camera sees at rest
    if left:
        branch(d, (fw - W) // 2 - 60, top - 20, 0.35, 230, 34, 7, rng)
    if right:
        branch(d, fw - (fw - W) // 2 + 60, top - 40, math.pi - 0.4, 250, 36, 7, rng)
    if grass:
        for _ in range(160):
            x = rng.uniform(0, fw)
            edge = x < fw * 0.25 or x > fw * 0.75
            h = rng.uniform(40, 130) * (1.3 if edge else 0.6)
            lean = rng.uniform(-0.35, 0.35)
            d.polygon([(x - 7, bottom + 30), (x + 7, bottom + 30), (x + lean * h, bottom - h)], fill=(6, 8, 18, 255))
    return layer.filter(ImageFilter.GaussianBlur(3.0))


@dataclass
class Camera:
    zoom: float = 1.0
    fx: float = W / 2        # the world point at the centre of the screen
    fy: float = H / 2

    def window(self) -> tuple[float, float, float, float]:
        w, h = W / self.zoom, H / self.zoom
        left = min(max(self.fx - w / 2, 0), W - w)
        top = min(max(self.fy - h / 2, 0), H - h)
        return left, top, w, h

    def view(self, plate: Image.Image) -> Image.Image:
        left, top, w, h = self.window()
        return plate.resize((W, H), Image.BICUBIC, box=(left, top, left + w, top + h))

    def pose(self, pose: Pose) -> Pose:
        left, top, _, _ = self.window()
        return replace(pose, x=(pose.x - left) * self.zoom, y=(pose.y - top) * self.zoom, height=pose.height * self.zoom)

    def point(self, x: float, y: float) -> tuple[float, float]:
        left, top, _, _ = self.window()
        return (x - left) * self.zoom, (y - top) * self.zoom

    def foreground(self, frame: Image.Image, layer: Image.Image, depth: float = 1.6) -> None:
        """Composite a foreground layer that moves `depth` times as much as the plate."""
        left, top, w, h = self.window()
        z = 1 + (self.zoom - 1) * depth
        fw, fh = layer.size
        ox = (fw - W) / 2 + (left - (W - w) / 2) * depth
        oy = (fh - H) / 2 + (top - (H - h) / 2) * depth
        box = (ox + (W - W / z) / 2, oy + (H - H / z) / 2, ox + (W + W / z) / 2, oy + (H + H / z) / 2)
        frame.alpha_composite(layer.resize((W, H), Image.BICUBIC, box=box))


@dataclass
class Scene:
    plates: dict
    rig: Rig
    tc: object
    fonts: tuple
    fog: np.ndarray
    stars: list
    fg: dict
    letters: list


def load_plate(name: str) -> Image.Image:
    (path,) = glob.glob(str(ROOT / PLATES[name]))
    return Image.open(path).convert("RGBA").resize((W, H), Image.LANCZOS)


# --- effects --------------------------------------------------------------------------

def add_fog(frame: Image.Image, fog: np.ndarray, t: float, strength: float, speed: float = 40) -> None:
    n = np.roll(fog, -int(t * speed) % W, axis=1)[:, :W]
    ramp = np.clip((np.arange(H) / H - 0.35) / 0.65, 0, 1)[:, None]
    layer = Image.new("RGBA", (W, H), (175, 205, 215, 0))
    layer.putalpha(Image.fromarray(np.clip(n * ramp * strength * 255, 0, 255).astype(np.uint8)))
    frame.alpha_composite(layer)


def add_stars(frame: Image.Image, stars: list, t: float, region_bottom: int) -> None:
    d = ImageDraw.Draw(frame)
    for x, y, phase, size in stars:
        if y > region_bottom or not (0 <= x < W):
            continue
        b = 0.5 + 0.5 * math.sin(2 * math.pi * (t * 0.8 + phase))
        if b < 0.55:
            continue
        r, c = size * b, (255, 250, 220, int(255 * b))
        d.line((x - 2 * r, y, x + 2 * r, y), fill=c, width=1)
        d.line((x, y - 2 * r, x, y + 2 * r), fill=c, width=1)
        d.ellipse((x - r / 2, y - r / 2, x + r / 2, y + r / 2), fill=c)


def sparkle(frame: Image.Image, x: float, y: float, r: float, alpha: float = 1.0) -> None:
    if r <= 0:
        return
    c = (255, 250, 200, int(255 * max(0.0, min(alpha, 1.0))))
    ImageDraw.Draw(frame).polygon(
        [(x, y - r), (x + r * 0.25, y - r * 0.25), (x + r, y), (x + r * 0.25, y + r * 0.25),
         (x, y + r), (x - r * 0.25, y + r * 0.25), (x - r, y), (x - r * 0.25, y - r * 0.25)], fill=c)


def speed_lines(frame: Image.Image, x: float, y: float, height: float, vx: float, vy: float, seed: int,
                count: int = 8, length: float = 280) -> None:
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return
    ux, uy, rng = vx / speed, vy / speed, np.random.default_rng(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        off, start, ln = rng.uniform(-0.45, 0.45) * height, rng.uniform(0.1, 0.4) * height, rng.uniform(0.5, 1) * length
        sx, sy = x - uy * off - ux * start, y - height * 0.45 + ux * off - uy * start
        d.line((sx, sy, sx - ux * ln, sy - uy * ln), fill=(235, 250, 255, 170), width=3)
    frame.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.6)))


def dust(frame: Image.Image, x: float, y: float, age: float) -> None:
    if not 0 <= age < 0.8:
        return
    d = ImageDraw.Draw(frame)
    a = int(150 * (1 - age / 0.8))
    for k, (dx, dy) in enumerate([(-60, -10), (-20, -25), (25, -20), (60, -8), (0, -40)]):
        r = 18 + 70 * age + 6 * (k % 3)
        cx, cy = x + dx * (0.5 + age), y + dy * (0.5 + age)
        d.ellipse((cx - r, cy - r * 0.7, cx + r, cy + r * 0.7), fill=(190, 205, 215, a))


def shadow_hand(frame: Image.Image, wrist: tuple[float, float], reach: float, curl: float, wiggle: float) -> None:
    """A black shadow arm reaching down from the upper right, long bony fingers spread
    towards the left; curl 0 = open claw, 1 = closed fist."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    wx, wy = wrist
    ink = (10, 4, 20, 255)
    d.polygon([(W + 120, -200), (W + 330, -120), (wx + 55, wy + 30), (wx - 5, wy - 40)], fill=ink)
    d.ellipse((wx - 95, wy - 70, wx + 70, wy + 85), fill=ink)
    for k, base in enumerate([-62, -32, -4, 24]):
        ang = math.radians(180 + base)
        length = 230 * reach * (0.85 if k in (0, 3) else 1.0)
        bend = math.radians(80 * curl + 10 * math.sin(wiggle + k * 1.3))
        pts, a = [(wx - 55 + 20 * math.cos(ang), wy + 10 + 40 * math.sin(ang))], ang
        for seg, (frac, width) in enumerate(((0.45, 30), (0.33, 22), (0.22, 14))):
            x1, y1 = pts[-1][0] + math.cos(a) * length * frac, pts[-1][1] + math.sin(a) * length * frac
            d.line((pts[-1][0], pts[-1][1], x1, y1), fill=ink, width=width)
            d.ellipse((x1 - width / 2, y1 - width / 2, x1 + width / 2, y1 + width / 2), fill=ink)
            pts.append((x1, y1))
            a -= bend * (0.6 + 0.4 * seg)
        tip = pts[-1]
        d.polygon([(tip[0] - 7, tip[1] - 5), (tip[0] + 7, tip[1] + 5),
                   (tip[0] + math.cos(a) * 26, tip[1] + math.sin(a) * 26)], fill=ink)
    tx, ty = wx - 20, wy + 70
    d.line((tx, ty, tx - 90 * reach, ty + 50 - 70 * curl), fill=ink, width=26)
    rim = layer.getchannel("A").filter(ImageFilter.MaxFilter(11)).filter(ImageFilter.GaussianBlur(7))
    glow = Image.new("RGBA", (W, H), (150, 60, 200, 0))
    glow.putalpha(rim.point(lambda v: int(v * 0.65)))
    frame.alpha_composite(glow)
    frame.alpha_composite(layer)


def draw_hollow(plate: Image.Image) -> Image.Image:
    out = plate.copy()
    cx, cy, rx, ry = 640, 360, 175, 215
    rng = np.random.default_rng(7)
    pts = []
    for k in range(48):
        a = 2 * math.pi * k / 48
        r = 1 + 0.06 * math.sin(3 * a + 0.7) + 0.04 * math.sin(7 * a) + rng.uniform(-0.02, 0.02)
        pts.append((cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)))
    rim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(rim).polygon([(cx + (x - cx) * 1.09, cy + (y - cy) * 1.07) for x, y in pts], fill=(150, 175, 160, 160))
    out.alpha_composite(rim.filter(ImageFilter.GaussianBlur(6)))
    hole = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(hole).polygon(pts, fill=(6, 6, 12, 255))
    out.alpha_composite(hole)
    ImageDraw.Draw(out).line(pts + [pts[0]], fill=(14, 12, 20, 255), width=7)
    return out


def shaken(frame: Image.Image, amount: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    dx, dy = (rng.uniform(-1, 1, 2) * amount).round().astype(int)
    out = Image.new("RGBA", frame.size, (0, 0, 0, 255))
    out.paste(frame, (int(dx), int(dy)))
    return out


# --- acting ---------------------------------------------------------------------------

def place_eyes(rig: Rig, centre: tuple[float, float], height: float) -> tuple[float, float]:
    ex = sum(e.cx for e in rig.eyes) / len(rig.eyes)
    ey = sum(e.cy for e in rig.eyes) / len(rig.eyes)
    k = height / rig.figure_height
    return centre[0] - (ex - rig.anchor[0]) * k, centre[1] - (ey - rig.anchor[1]) * k


def cloth(t: float, period: float = 1.3, wave: float = 0.035) -> dict:
    return {"wave": wave, "wave_phase": 2 * math.pi * t / period, "tip": 0.03 * math.sin(2 * math.pi * t / 1.7)}


def blinking(t: float, times: list[float]) -> float:
    return 0.0 if any(s <= t < s + 0.17 for s in times) else 1.0


def draw_milo(s: Scene, frame: Image.Image, cam: Camera, pose: Pose, light: dict, ground: float | None = None) -> None:
    """Milo through the camera, lit by the scene, with his glow on the ground."""
    p = cam.pose(pose)
    if ground is not None:
        gx, gy = cam.point(pose.x, ground)
        lift = max(0.0, 1 - (ground - pose.y) / 300)
        ground_glow(frame, gx, gy, p.height, 0.9 * lift * pose.opacity)
    frame.alpha_composite(integrate(s.rig.render(p, (W, H)), **light))


def shot_forest(s: Scene, t: float) -> Image.Image:
    """A tilt down from the moon into the forest."""
    k = ease(t / 3.0)
    cam = Camera(zoom=lerp(1.25, 1.1, k), fx=lerp(700, 640, k), fy=lerp(230, 400, k))
    frame = cam.view(s.plates["forest"])
    add_stars(frame, [(*cam.point(x, y), ph, sz) for x, y, ph, sz in s.stars], t, 260)
    add_fog(frame, s.fog, t, 0.5)
    cam.foreground(frame, s.fg["forest"])
    return frame


def shot_hollow(s: Scene, t: float) -> Image.Image:
    """Eyes open in the dark hollow, blink, look around, smile."""
    cam = Camera(zoom=1 + 0.06 * ease(t / 3.5), fx=640, fy=345)
    frame = cam.view(s.plates["bark_hollow"])
    if t >= 0.75:
        open_ = ease((t - 0.75) / 0.3) * blinking(t, [1.6])
        gaze = (0.0, 0.0)
        if 1.9 <= t < 2.4:
            gaze = (-ease((t - 1.9) / 0.15), 0.0)
        elif 2.4 <= t < 2.9:
            gaze = (ease((t - 2.4) / 0.15), 0.0)
        if t >= 3.0:
            open_ = lerp(1, 0.5, ease((t - 3.0) / 0.2))
        x, y = place_eyes(s.rig, (640, 330), 1500)
        pose = cam.pose(Pose(x, y, 1500, eye_open=max(open_, 0.0), gaze=gaze))
        glow = s.rig.face_only(pose, (W, H))
        halo = Image.new("RGBA", (W, H), (60, 230, 220, 0))
        halo.putalpha(glow.getchannel("A").filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.7)))
        frame.alpha_composite(halo)
        frame.alpha_composite(glow)
    return frame


ARRIVE_LIGHT = {"ambient": (0.95, 1.0, 1.02), "rim": (215, 245, 255), "rim_dir": (1, -1), "rim_strength": 0.75}


def arrive_pose(t: float) -> Pose:
    base_x, base_y, height = 640, 650, 430
    c = cloth(t)
    if t < 0.4:  # shoots up from below, stretched
        u = ease_out(t / 0.4)
        return Pose(base_x, lerp(H + 420, base_y - 40, u), height, stretch=lerp(0.45, 0.1, u), stretch_angle=90,
                    arms={"left": -30, "right": 30}, mouth="o", eye_scale=1.2, **c)
    u = t - 0.4
    squash = 0.2 * math.exp(-6 * u) * math.cos(2 * math.pi * 2.2 * u)
    y = base_y + bob(u, 1.5, 14) - 40 * math.exp(-5 * u)
    gaze, mouth = (0.0, 0.0), "smile"
    arms = {"left": 4 * math.sin(2 * math.pi * u / 1.5), "right": 0.0}
    if 0.55 <= t < 0.85:
        gaze = (-1.0, -0.1)
    elif 0.85 <= t < 1.15:
        gaze = (1.0, -0.1)
    if t < 0.75:
        mouth = "o"
    if 1.8 <= t < 3.1:  # waves at the camera
        w = (t - 1.8) / 1.3
        arms["right"] = (32 + 20 * math.sin(2 * math.pi * (t - 1.8) / 0.45)) * math.sin(math.pi * w) ** 0.4
    wink = 0 if 3.15 <= t < 3.5 else None
    x, stretch, angle, extra = base_x, 0.0, 90.0, {}
    if 4.1 <= t < 4.4:  # anticipation
        k = ease((t - 4.1) / 0.3)
        squash, y = 0.16 * k, y + 25 * k
        arms, mouth = {"left": 15 * k, "right": -15 * k}, "closed"
    elif t >= 4.4:  # off up and to the right
        k = ease_in((t - 4.4) / 0.4)
        x, y = lerp(base_x, base_x + 700, k), lerp(y, -400, k)
        stretch, angle = 0.5 * min(1, k * 3), 55
        arms, mouth, extra = {"left": -35, "right": 35}, "o", {"drag": -0.15 * min(1, k * 3)}
    return Pose(x, y, height, squash=squash, arms=arms, eye_open=blinking(t, [1.4, 3.8]), wink=wink, gaze=gaze,
                mouth=mouth, glow=0.5, stretch=stretch, stretch_angle=angle, **{**c, **extra})


def shot_arrive(s: Scene, t: float) -> Image.Image:
    """Wide, then a push-in to his face for "Meet Milo!", the wave and the wink, a pull back."""
    push = ease((t - 0.9) / 0.8) - ease((t - 3.4) / 0.6)
    cam = Camera(zoom=lerp(1.05, 1.75, push), fx=640, fy=lerp(352, 395, push))
    frame = cam.view(s.plates["clearing"])
    add_stars(frame, [(*cam.point(x, y), ph, sz) for x, y, ph, sz in s.stars], t, 200)
    add_fog(frame, s.fog, t + 3, 0.3, speed=25)
    draw_milo(s, frame, cam, arrive_pose(t), ARRIVE_LIGHT, ground=665)
    cam.foreground(frame, s.fg["clearing"])
    return frame


def sky_path(t: float) -> tuple[float, float, float]:
    """Position and how far round the loop he is (degrees) at time t of the 3 s shot."""
    u = t / 3.0
    loop_a, loop_b, cx, cy, r = 0.35, 0.68, W * 0.5, 420, 130
    if u < loop_a:
        return lerp(-300, cx, u / loop_a), cy + 25 * math.sin(u * 9), 0.0
    if u < loop_b:
        phi = 2 * math.pi * ease((u - loop_a) / (loop_b - loop_a))
        return cx + r * math.sin(phi), cy - r * (1 - math.cos(phi)), math.degrees(phi)
    return lerp(cx, W + 320, (u - loop_b) / (1 - loop_b)), cy + 25 * math.sin(u * 9), 360.0


def sky_pose(t: float) -> Pose:
    x, y, turn = sky_path(t)
    x2, y2, _ = sky_path(t + 0.03)
    angle = math.degrees(math.atan2(-(y2 - y), x2 - x))
    flap = 10 * math.sin(2 * math.pi * t / 0.4)
    looping = 0 < turn < 360
    return Pose(x, y, 250, stretch=0.3, stretch_angle=angle, tilt=angle * (1.0 if looping else 0.5),
                drag=-0.2, wave=0.05, wave_phase=2 * math.pi * t / 0.5, tip=-0.06,
                arms={"left": 25 + flap, "right": -25 - flap}, gaze=(1.0, -0.3 if looping else 0.0),
                mouth="o" if looping else "smile", eye_open=blinking(t, [2.4]), glow=0.85)


def shot_sky(s: Scene, t: float) -> Image.Image:
    """A flight across the sky with a loop-the-loop."""
    cam = Camera(zoom=1.04, fx=640 + 20 * math.sin(t), fy=352)
    frame = cam.view(s.plates["sky"])
    add_stars(frame, [(*cam.point(x, y), ph, sz) for x, y, ph, sz in s.stars], t, 420)
    for k in range(1, 8):
        back = sky_pose(max(0.0, t - k * 0.07))
        px, py = cam.point(back.x, back.y - 125)
        sparkle(frame, px, py, 11 - k, alpha=1 - k / 8)
    draw_milo(s, frame, cam, sky_pose(t), {"ambient": (1.0, 0.95, 1.02), "rim": (255, 215, 235), "rim_dir": (1, -1),
                                            "rim_strength": 0.7})
    return frame


DANGER_LIGHT = {"ambient": (0.78, 0.74, 0.96), "rim": (200, 140, 255), "rim_dir": (1, 0), "rim_strength": 0.9}


def danger_pose(t: float) -> tuple[Pose | None, dict]:
    base_x, base_y, height = 470, 640, 400
    c = cloth(t, 1.6, 0.03)
    info = {"x": base_x, "y": base_y, "vx": 0.0, "vy": 0.0, "lines": False}
    if t < 2.0:  # floats, eyes shut, humming
        return Pose(base_x, base_y + bob(t, 1.6, 12), height, eye_open=0.0, mouth="closed", glow=0.4,
                    arms={"left": 5 * math.sin(t * 3), "right": -5 * math.sin(t * 3)}, **c), info
    if t < 2.4:  # opens his eyes, looks at the hand
        return Pose(base_x, base_y + bob(t, 1.6, 12), height, eye_open=ease((t - 2.0) / 0.12), gaze=(1.0, -0.2),
                    mouth="closed", glow=0.4, **c), info
    if t < 2.55:  # anticipation
        k = ease((t - 2.4) / 0.15)
        return Pose(base_x, base_y + 20 * k, height, squash=0.2 * k, eye_scale=1.1, gaze=(1.0, 0), mouth="o",
                    arms={"left": 20 * k, "right": -20 * k}, glow=0.4, **c), info
    if t < 2.9:  # the take
        k = (t - 2.55) / 0.35
        st = 0.45 * math.exp(-4 * k) * math.cos(2 * math.pi * 1.5 * k)
        return Pose(base_x, base_y - math.sin(math.pi * min(1, k * 1.4)) * 110, height, stretch=max(st, -0.1),
                    stretch_angle=90, eye_scale=1.65, gaze=(0.6, -0.3), mouth="o", arms={"left": -45, "right": 45},
                    glow=0.6, tip=0.08 * math.sin(k * 20), wave=0.05, wave_phase=k * 12), info
    if t < 3.25:  # zip!
        k = ease_in((t - 2.9) / 0.35)
        x = lerp(base_x, -500, k)
        info.update(x=x, y=base_y - 60, vx=-2600, lines=True)
        return Pose(x, base_y - 60, height, stretch=0.7, stretch_angle=0, drag=0.35, eye_scale=1.4, gaze=(-1, 0),
                    mouth="o", arms={"left": -20, "right": 50}, glow=0.6, wave=0.04, wave_phase=t * 40), info
    return None, info


def shot_danger(s: Scene, t: float) -> Image.Image:
    """The shadow hand, the take, the zip-off; the camera creeps in with the tension."""
    tension = ease(t / 2.4)
    cam = Camera(zoom=lerp(1.05, 1.14, tension), fx=lerp(640, 600, tension), fy=lerp(352, 380, tension))
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
        shadow_hand(frame, cam.point(wx, wy + 12 * math.sin(t * 2.3)), reach * cam.zoom, curl, t * 6)
    pose, info = danger_pose(t)
    if info["lines"]:
        for k, a in ((2, 0.18), (1, 0.32)):
            back, _ = danger_pose(t - k * 0.05)
            if back is not None:
                frame.alpha_composite(s.rig.render(cam.pose(replace(back, opacity=a, glow=0)), (W, H)))
        sx, sy = cam.point(info["x"] + 200, info["y"])
        speed_lines(frame, sx, sy, 400 * cam.zoom, info["vx"], info["vy"], seed=int(t * 24))
    if pose is not None:
        draw_milo(s, frame, cam, pose, DANGER_LIGHT, ground=655)
    dx, dy = cam.point(470, 650)
    dust(frame, dx, dy, t - 2.9)
    cam.foreground(frame, s.fg["danger"])
    if 2.55 <= t < 2.95:
        frame = shaken(frame, 16 * (1 - (t - 2.55) / 0.4), seed=int(t * 24))
    return frame


def shot_castle(s: Scene, t: float) -> Image.Image:
    """Milo flies into the castle window: pop, sparkles, a warm glow."""
    k0 = ease(t / 2.5)
    cam = Camera(zoom=lerp(1.04, 1.12, k0), fx=lerp(640, WINDOW[0] + 60, k0), fy=lerp(352, 420, k0))
    frame = cam.view(s.plates["castle_lit"])
    add_stars(frame, [(*cam.point(x, y), ph, sz) for x, y, ph, sz in s.stars], t, 300)
    fly = 1.3
    if t < fly:
        u = ease_in(t / fly) * 0.6 + 0.4 * (t / fly)
        pose = Pose(lerp(1450, WINDOW[0], u) + 120 * math.sin(math.pi * u),
                    lerp(120, WINDOW[1] + 20, u) - 120 * math.sin(math.pi * u),
                    lerp(300, 40, u), stretch=0.3, stretch_angle=200, drag=0.15, wave=0.05, wave_phase=t * 12,
                    gaze=(-1, 0.3), arms={"left": -25, "right": 25}, glow=0.8, opacity=1 - max(0, (u - 0.9) / 0.1))
        draw_milo(s, frame, cam, pose, {"ambient": (0.92, 0.92, 1.0), "rim": (255, 235, 200), "rim_dir": (-1, -1)})
    else:
        k = t - fly
        wx, wy = cam.point(*WINDOW)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((wx - 36, wy - 44, wx + 36, wy + 28), fill=(255, 210, 120, int(210 * math.exp(-k))))
        frame.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)))
        if k < 0.6:
            for j in range(6):
                ang, r = 2 * math.pi * j / 6 + 0.3, 25 + 160 * k
                sparkle(frame, wx + r * math.cos(ang), wy + r * math.sin(ang), 14 * (1 - k / 0.6) + 3, alpha=1 - k / 0.6)
    return frame


def title_pose(t: float) -> Pose:
    if t < 1.8:  # flies in behind the letters and comes to rest beside the logo
        u = ease_out(t / 1.8)
        return Pose(lerp(-150, 1175, u), lerp(620, 690, u) - 160 * math.sin(math.pi * u), 220, tilt=-8 * (1 - u),
                    stretch=0.2 * (1 - u), stretch_angle=10, drag=-0.12 * (1 - u), wave=0.05, wave_phase=t * 8,
                    gaze=(1, -0.2), glow=1.0, mouth="smile")
    k = t - 1.8
    wink = 0 if 2.4 <= t < 2.9 else None
    return Pose(1175, 690 + bob(k, 1.2, 8), 220, squash=0.12 * math.exp(-6 * k) * math.cos(2 * math.pi * 2.5 * k),
                gaze=(0, 0), glow=1.0, wink=wink, mouth="smile", arms={"right": 30 if wink == 0 else 0}, **cloth(t))


def shot_title(s: Scene, t: float) -> Image.Image:
    """Letters bounce in, Milo comes to rest beside them, winks, and the iris closes on him."""
    frame = s.plates["castle_dark"].copy()
    add_stars(frame, s.stars, t, 380)
    pose = title_pose(t)
    behind = t < 1.2  # passes behind the letters, then rests in front, beside them
    if behind:
        frame.alpha_composite(integrate(s.rig.render(pose, (W, H))))
    if t < 1.9:
        for k in range(1, 6):
            back = title_pose(max(0.0, t - k * 0.05))
            sparkle(frame, back.x - 30, back.y - 130, 9 - k, alpha=1 - k / 6)
    for (image, offset), at in zip(s.letters, LETTER_TIMES):
        if t < at:
            continue
        u = t - at
        scale = 1 - math.exp(-7 * u) * math.cos(2 * math.pi * 2.2 * u)
        if scale <= 0.02:
            continue
        img = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
        lx, ly = W // 2 + offset[0], int(H * 0.42) + offset[1]
        paste(frame, img, (round(lx - img.width / 2), round(ly - img.height / 2)))
    if t >= 0.9:
        sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        s.tc.glow_text(sub, (W // 2, int(H * 0.74)), "Coming this Saturday Morning", s.fonts[1], (255, 236, 200),
                       s.tc.ORANGE, radius=8, outline=2)
        sub.putalpha(sub.getchannel("A").point(lambda v: int(v * ease((t - 0.9) / 0.4))))
        frame.alpha_composite(sub)
    if not behind:
        frame.alpha_composite(integrate(s.rig.render(pose, (W, H))))
    if t >= 2.6:  # iris-out on Milo's face
        k = ease_in((t - 2.6) / 0.7)
        fx, fy = pose.x, pose.y - pose.height * 0.72
        r = lerp(900, 0, k)
        mask = Image.new("L", (W, H), 255)
        if r > 1:
            ImageDraw.Draw(mask).ellipse((fx - r, fy - r, fx + r, fy + r), fill=0)
        frame.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), mask)
    return frame


def letter_images(tc, font) -> list:
    """Each title letter with the logo's glow on its own image, and its offset from the logo's centre."""
    word = "MILO"
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    widths = [probe.textlength(ch, font=font) for ch in word]
    total = sum(widths) + 8 * (len(word) - 1)
    out, x = [], -total / 2
    for ch, w in zip(word, widths):
        size = (int(w + 140), int(font.size * 1.6))
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        tc.glow_text(img, (size[0] // 2, size[1] // 2), ch, font, tc.TURQUOISE, tc.ORANGE, radius=18, outline=6)
        out.append((img, (x + w / 2, 0)))
        x += w + 8
    return out


def film(frame: np.ndarray, drawing_index: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + drawing_index)
    dx, dy = rng.integers(-1, 2, size=2)
    frame = np.roll(frame, (dy, dx), axis=(0, 1))
    grain = rng.normal(0, 5, frame.shape[:2])[..., None]
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 1 - 0.24 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(frame * vignette[..., None] + grain, 0, 255).astype(np.uint8)


SHOT_FUNCTIONS = {"forest": shot_forest, "hollow": shot_hollow, "arrive": shot_arrive, "sky": shot_sky,
                  "danger": shot_danger, "castle": shot_castle, "title": shot_title}


def render_drawing(s: Scene, t: float) -> Image.Image:
    name, s0, _ = next(x for x in SHOTS if x[1] <= t < x[2])
    return SHOT_FUNCTIONS[name](s, t - s0)


def build_scene() -> Scene:
    spec = importlib.util.spec_from_file_location("title_card", ROOT / "scripts/make_title_card.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    raw = {name: load_plate(name) for name in PLATES}
    ref = np.asarray(raw["forest"].convert("RGB")).reshape(-1, 3).astype(np.float32)
    stats = (ref.mean(0), ref.std(0))
    plates = {name: grade(img, stats, 0.0 if name == "forest" else 0.35) for name, img in raw.items()}
    plates["bark_hollow"] = draw_hollow(plates["bark"])
    dark = ImageEnhance.Brightness(plates["clearing"]).enhance(0.55)
    dark.alpha_composite(Image.new("RGBA", (W, H), (70, 20, 110, 70)))
    plates["clearing_dark"] = dark
    plates["castle_lit"] = ImageEnhance.Brightness(plates["castle"]).enhance(0.9)
    plates["castle_dark"] = ImageEnhance.Brightness(plates["castle"]).enhance(0.42)
    fog = smooth_noise((H, 2 * W), 220, seed=11, salt=3)
    fog = np.concatenate([fog, fog[:, ::-1]], axis=1)
    fog = np.clip((fog - fog.min()) / (fog.max() - fog.min()) * 1.4 - 0.45, 0, 1)
    rng = np.random.default_rng(5)
    stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0, 1), rng.uniform(2, 4)) for _ in range(45)]
    fonts = (ImageFont.truetype(str(tc.FONTS / tc.TITLE_FONT), int(H * 0.34)),
             ImageFont.truetype(str(tc.FONTS / tc.SUBTITLE_FONT), int(H * 0.075)))
    fg = {"forest": foreground(21, True, False, True), "clearing": foreground(22, True, True, True),
          "danger": foreground(23, False, True, True)}
    return Scene(plates, Rig(Image.open(DRAWING), SPEC, scale=0.9), tc, fonts, fog, stars, fg,
                 letter_images(tc, fonts[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    scene = build_scene()
    total = int(SHOTS[-1][2] * FPS)
    out_dir = ROOT / "outputs/trailers/milo_03" / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True)
    (out_dir / "cues.json").write_text(json.dumps(CUES, indent=2) + "\n", encoding="utf-8")
    frames_needed = range(0, total, 12) if args.preview else range(total)
    encoder = None
    if not args.preview:
        encoder = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
             str(out_dir / "teaser.mp4")], stdin=subprocess.PIPE)
    previews, cache = [], {}
    started = time.perf_counter()
    for n in frames_needed:
        drawing = n // 2
        if drawing not in cache:
            frame = render_drawing(scene, drawing * 2 / FPS)
            cache = {drawing: film(np.asarray(frame.convert("RGB"), dtype=np.float32), drawing)}
        frame = cache[drawing].astype(np.float32)
        level = 1.0
        for name, s0, s1 in SHOTS:
            a, b = round(s0 * FPS), round(s1 * FPS)
            if a <= n < b:
                fade_in = 12 if a == 0 else DIP
                level = min(1.0, (n - a + 1) / fade_in)
                if name != "title":  # the iris closes the film
                    level = min(level, (b - n) / DIP)
        frame = (frame * level).astype(np.uint8)
        if args.preview:
            previews.append(frame)
        else:
            encoder.stdin.write(frame.tobytes())
    if encoder:
        encoder.stdin.close()
        encoder.wait()
        print(f"video: {(out_dir / 'teaser.mp4').relative_to(ROOT)}  ({total} frames, {time.perf_counter() - started:.0f} s)")
    else:
        cols = 6
        sheet = Image.new("RGB", (cols * 320, ((len(previews) + cols - 1) // cols) * 176))
        for i, f in enumerate(previews):
            sheet.paste(Image.fromarray(f).resize((320, 176)), ((i % cols) * 320, (i // cols) * 176))
        sheet.save(out_dir / "preview.png")
        print(f"preview: {(out_dir / 'preview.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
