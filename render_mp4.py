"""
render_mp4.py — offline rendering of the drawing to a full-quality MP4 video.

Reproduces the logic of the p5.js visualization:
- X position from the pitch percentile within the piece
- "jellyfish" curves (Catmull-Rom with distant anchors, driven by Perlin noise)
- permanent low-opacity accumulation + fading flashes
- a mask that clips the drawing to the shape
- for Italy: drawing in a lying space, then rotation of the composition
  (low notes at the bottom, high notes at the top, as in the browser)

Note: p5.noise() and the Perlin noise used here are not bit-identical, so the
video is not pixel-identical to the browser preview; logic and look are the same.
"""

import io
import json
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

SR = 22050
FPS = 30
FLASH_DUR = 0.5
ACCUM_ALPHA = 26


# ---------- Perlin noise (equivalent, not identical to p5) ----------
_p = np.arange(256, dtype=int)
np.random.seed(42)
np.random.shuffle(_p)
_p = np.stack([_p, _p]).flatten()

def _fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
def _lerp(a, b, t): return a + t * (b - a)
def _grad(h, x, y):
    h = h & 3
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return (np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v))

def perlin(x, y):
    """Vectorized 2D Perlin noise, returns ~0..1 like p5.noise."""
    xi = np.floor(x).astype(int) & 255
    yi = np.floor(y).astype(int) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    u = _fade(xf); v = _fade(yf)
    aa = _p[_p[xi] + yi]; ab = _p[_p[xi] + yi + 1]
    ba = _p[_p[xi + 1] + yi]; bb = _p[_p[xi + 1] + yi + 1]
    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
    return (_lerp(x1, x2, v) + 1) / 2

def noise1(x):
    return float(perlin(np.array([x * 1.0]), np.array([0.0]))[0])

def noise2(x, y):
    return float(perlin(np.array([x * 1.0]), np.array([y * 1.0]))[0])


# ---------- HSL->RGB colour, as in the sketch ----------
def hsl_rgb(h, s, l):
    h = h / 360.0
    if s == 0:
        r = g = b = l
    else:
        def f(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = f(p, q, h + 1/3); g = f(p, q, h); b = f(p, q, h - 1/3)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------- Catmull-Rom, like p5's curve() ----------
def catmull_rom(p0, p1, p2, p3, seg=32):
    pts = []
    for i in range(seg + 1):
        t = i / seg; t2 = t * t; t3 = t2 * t
        x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        pts.append((x, y))
    return pts


def midi_percentile(m, sorted_midis):
    import bisect
    if not sorted_midis:
        return 0.5
    return bisect.bisect_right(sorted_midis, m) / len(sorted_midis)


def event_geometry(ev, idx, DW, DH, sorted_midis):
    if ev["freq"] is None:
        return None
    midi = 69 + 12 * math.log2(ev["freq"] / 440.0)
    px = midi_percentile(midi, sorted_midis)
    x = _lerp(DW * 0.08, DW * 0.92, px)
    y = DH * (0.2 + 0.6 * noise1(idx * 0.1))
    amp = ev.get("amplitude", 0.5)
    if not isinstance(amp, (int, float)):
        amp = 0.5
    length = _lerp(DH * 0.15, DH * 0.45, max(0, min(1, amp)))
    d = noise2(x * 0.004, y * 0.004) * 2 * math.pi
    x2 = x + math.cos(d) * length; y2 = y + math.sin(d) * length
    ad = _lerp(DH * 0.38, DH * 1.5, max(0, min(1, amp)))
    a1 = noise1(x * 0.005 + 10) * 2 * math.pi
    a2 = noise1(x2 * 0.005 + 20) * 2 * math.pi
    a1x = x + math.cos(a1) * ad; a1y = y + math.sin(a1) * ad
    a2x = x2 + math.cos(a2) * ad; a2y = y2 + math.sin(a2) * ad
    r, g, b = hsl_rgb(_lerp(220, 140, px), 0.72, 0.62)
    return dict(x=x, y=y, x2=x2, y2=y2, a1x=a1x, a1y=a1y, a2x=a2x, a2y=a2y,
                r=r, g=g, b=b, time=ev["time"])


def draw_curve(base, geo, rgba, width):
    """Draws the curve ON TOP of what is already there (alpha blending, like the
    browser canvas). ImageDraw on an RGBA image replaces pixels instead of
    blending them, so the curve is drawn on a separate layer the size of its
    bounding box and then composited onto the base image."""
    pts = catmull_rom((geo['a1x'], geo['a1y']), (geo['x'], geo['y']),
                      (geo['x2'], geo['y2']), (geo['a2x'], geo['a2y']))
    w = max(1, int(round(width)))
    pad = w + 2
    x0 = max(0, int(math.floor(min(p[0] for p in pts))) - pad)
    y0 = max(0, int(math.floor(min(p[1] for p in pts))) - pad)
    x1 = min(base.width, int(math.ceil(max(p[0] for p in pts))) + pad)
    y1 = min(base.height, int(math.ceil(max(p[1] for p in pts))) + pad)
    if x1 <= x0 or y1 <= y0:
        return  # curve entirely outside the drawing
    tmp = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).line([(px - x0, py - y0) for px, py in pts],
                             fill=rgba, width=w, joint='curve')
    base.alpha_composite(tmp, dest=(x0, y0))


def render(score, audio_path, mask_b64, mask_key, out_path,
           max_seconds=None, progress=None):
    events = sorted([e for e in score["events"] if e.get("freq")], key=lambda e: e["time"])
    if max_seconds:
        events = [e for e in events if e["time"] < max_seconds]
    duration = min(score["duration"], max_seconds) if max_seconds else score["duration"]
    sorted_midis = sorted(69 + 12 * math.log2(e["freq"] / 440.0) for e in events)

    # mask
    raw = mask_b64.split(",", 1)[1]
    import base64
    mimg = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")

    rotate = (mask_key == "italy")
    # upright canvas size
    aspect = mimg.width / mimg.height
    if aspect >= 1:
        W = 1200; H = int(1200 / aspect)
    else:
        H = 1000; W = int(1000 * aspect)
    mimg = mimg.resize((W, H), Image.LANCZOS)
    mask_alpha = mimg.split()[3]

    # drawing space (lying for Italy)
    DW, DH = (H, W) if rotate else (W, H)

    # precompute geometries
    geos = [event_geometry(e, i, DW, DH, sorted_midis) for i, e in enumerate(events)]
    geos = [g for g in geos if g]

    bg = (10, 13, 18)
    n_frames = int(duration * FPS)
    tmpdir = tempfile.mkdtemp()

    accum = Image.new("RGBA", (DW, DH), (0, 0, 0, 0))
    next_ev = 0

    for f in range(n_frames):
        t = f / FPS
        while next_ev < len(geos) and geos[next_ev]["time"] <= t:
            g = geos[next_ev]
            draw_curve(accum, g, (g["r"], g["g"], g["b"], ACCUM_ALPHA), 1.2)
            next_ev += 1

        flash = Image.new("RGBA", (DW, DH), (0, 0, 0, 0))
        for g in geos:
            el = t - g["time"]
            if 0 <= el <= FLASH_DUR:
                tt = el / FLASH_DUR
                a = int(_lerp(180, 0, tt)); w = _lerp(2.2, 0.6, tt)
                draw_curve(flash, g, (g["r"], g["g"], g["b"], a), w)

        # compose the drawing
        layer = Image.new("RGBA", (DW, DH), (0, 0, 0, 0))
        layer.alpha_composite(accum)
        layer.alpha_composite(flash)

        # rotation for Italy, like p.rotate(-HALF_PI) in the browser: in PIL a
        # positive angle rotates COUNTERCLOCKWISE, so +90 brings the right side of
        # the lying space (high notes) to the top and the left (low notes) down.
        if rotate:
            layer = layer.rotate(90, expand=True)  # from (DW,DH) to (DH,DW)=(W,H)

        # apply the mask: the drawing's alpha is multiplied by the mask,
        # so the drawing appears ONLY inside the shape
        larr = np.array(layer)
        marr = np.array(mask_alpha)
        larr[:, :, 3] = (larr[:, :, 3].astype(int) * marr // 255).astype(np.uint8)
        masked = Image.fromarray(larr, "RGBA")

        frame = Image.new("RGB", (W, H), bg)
        frame.paste(masked, (0, 0), masked)
        frame.save(os.path.join(tmpdir, f"f{f:05d}.png"))

        if progress and f % 15 == 0:
            progress(f / n_frames)

    # assemble with ffmpeg + audio
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS), "-i", os.path.join(tmpdir, "f%05d.png"),
        "-i", audio_path, "-t", str(duration),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest", out_path
    ], check=True)
    return out_path
