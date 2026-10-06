"""Spike: cut Milo out of his design picture onto a plain grey square, so an
IP-Adapter reference carries the character and not the forest behind him.

    python spikes/cutout_milo.py SOURCE.png TARGET.png

Region grow from points on each part of the body over pale blue pixels, holes filled
(eyes, mouth), grown by a few pixels to keep the black outline.
Deterministic, no model."""
import sys
from collections import deque

import numpy as np
from PIL import Image

src, dst = sys.argv[1:3]
im = np.asarray(Image.open(src).convert("RGB")).astype(int)
H, W = im.shape[:2]
r, g, b = im[..., 0], im[..., 1], im[..., 2]
body = (b > 150) & (g > 140) & (b - r > 15)          # the pale blue-white of the ghost
mask = np.zeros((H, W), bool)
# belly, head, right hand, both feet: the ink lines cut the figure into parts
q = deque([(560, 500), (180, 520), (320, 710), (800, 390), (820, 610)])
while q:
    y, x = q.popleft()
    if 0 <= y < H and 0 <= x < W and not mask[y, x] and body[y, x]:
        mask[y, x] = True
        q.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
# fill holes: everything not reachable from the border without crossing the mask
outside = np.zeros((H, W), bool)
q = deque([(0, 0)])
while q:
    y, x = q.popleft()
    if 0 <= y < H and 0 <= x < W and not outside[y, x] and not mask[y, x]:
        outside[y, x] = True
        q.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
mask = ~outside
for _ in range(5):                                    # keep the ink outline
    mask = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
ys, xs = np.nonzero(mask)
print("mask px", int(mask.sum()), "bbox", xs.min(), ys.min(), xs.max(), ys.max())
out = np.full_like(im, 128)
out[mask] = im[mask]
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
side = int(max(y1 - y0, x1 - x0) * 1.12)
cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
canvas = np.full((side, side, 3), 128, np.uint8)
sy, sx = cy - side // 2, cx - side // 2
for yy in range(side):
    for_y = sy + yy
    if 0 <= for_y < H:
        lo, hi = max(sx, 0), min(sx + side, W)
        canvas[yy, lo - sx:hi - sx] = out[for_y, lo:hi]
Image.fromarray(canvas).resize((1024, 1024), Image.LANCZOS).save(dst)
