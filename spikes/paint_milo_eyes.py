"""Spike: paint turquoise irises into Milo's design (deterministic, no model).

    python spikes/paint_milo_eyes.py SOURCE.png TARGET.png COMPARISON.png

The eye positions (flood-fill seeds) are those of benchmarks/references/milo/milo_final_design.png."""
import sys
from collections import deque
import numpy as np
from PIL import Image
src, dst, cmp_ = sys.argv[1:4]
img = Image.open(src).convert("RGB")
im = np.asarray(img).astype(int)
dark = im.sum(axis=2) < 200
def blob(seed):
    m = np.zeros(dark.shape, bool); q = deque([seed])
    while q:
        y, x = q.popleft()
        if m[y, x] or not dark[y, x]: continue
        m[y, x] = True
        q.extend(((y+1,x),(y-1,x),(y,x+1),(y,x-1)))
    return m
def erode(m, r):
    for _ in range(r):
        m = m & np.roll(m,1,0) & np.roll(m,-1,0) & np.roll(m,1,1) & np.roll(m,-1,1)
    return m
out = im.copy()
yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
for seed in ((262, 497), (268, 556)):
    m = blob(seed); ys, xs = np.nonzero(m)
    cy, cx, h, w = ys.mean(), xs.mean(), ys.ptp()+1, xs.ptp()+1
    iris = erode(m, 3)
    out[iris] = (38, 206, 204)
    out[iris & (((yy-(cy+h*0.10))/(h*0.25))**2 + ((xx-cx)/(w*0.27))**2 <= 1)] = 0
    out[iris & (((yy-(cy-h*0.06))/(h*0.08))**2 + ((xx-(cx-w*0.10))/(w*0.10))**2 <= 1)] = (245, 250, 250)
res = Image.fromarray(out.astype(np.uint8)); res.save(dst)
crop = lambda a: Image.fromarray(np.asarray(a)[200:320, 440:620]).resize((540, 360), Image.NEAREST)
z = Image.new("RGB", (1080, 360)); z.paste(crop(img), (0, 0)); z.paste(crop(res), (540, 0)); z.save(cmp_)
