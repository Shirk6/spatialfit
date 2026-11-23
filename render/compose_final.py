"""Final demo cut: section titles, sparse captions, parameter panel, labels, synced A|B split screen, smooth transitions.

Inputs (all under build/):
  video/gen/g_####.png  + proj_0000.json   (blender_chair_shots.py, shot_gen.json)   parametric morph + exploded view
  video/end/g_####.png                     (blender_chair_shots.py, shot_end.json)   ending, A and B side by side
  video/fall_A/frame_####.png              (blender_render2.py, fall_A.npz, speed 1)  early fixture release
  acp_par/A_jam.mp4, acp_par/B.mp4         (blender_render2.py via acp_render_parallel.py)
  plan_A.json, *_hud.json, *_summary.json

usage: python render/compose_final.py [--out build/video/SpatialFit_demo.mp4] [--only s1,s2,s3,s4] [--preview]
"""
import argparse, json, subprocess, math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
FONT = ROOT / "render" / "fonts"
W, H, FPS = 1920, 1080, 30
AMBER = (255, 176, 64)
TEAL = (96, 206, 196)
WHITE = (245, 246, 248)
GREY = (168, 174, 184)
RED = (255, 98, 84)


def font(weight, size):
    return ImageFont.truetype(str(FONT / f"Inter-{weight}.ttf"), size)


F = dict(title=font("SemiBold", 70), sub=font("Light", 30), idx=font("Medium", 26), cap=font("Medium", 32),
         capsub=font("Regular", 24), label=font("Medium", 25), small=font("Regular", 21), smallb=font("SemiBold", 21),
         panel_h=font("SemiBold", 17), panel=font("Regular", 22), panel_v=font("Medium", 22), big=font("SemiBold", 44),
         step=font("SemiBold", 28), chip=font("Medium", 22), hero=font("SemiBold", 96), num=font("SemiBold", 40))


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1, fade=0.4):
    """0 -> 1 -> 0 envelope for an element visible on [t0, t1] (seconds) with fades."""
    if t < t0 or t > t1:
        return 0.0
    return min(smooth((t - t0) / fade), smooth((t1 - t) / fade))


# ------------------------------------------------------------------ sources
def load_png(p):
    return np.asarray(Image.open(p).convert("RGB"))


class PngSeq:
    def __init__(self, pattern, n, start=0):
        self.pattern, self.n, self.start, self.cache = pattern, n, start, {}

    def __call__(self, i):
        i = int(min(max(i, 0), self.n - 1))
        if i not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[i] = load_png(self.pattern.format(i + self.start))
        return self.cache[i]


class Video:
    """Forward-only random access into an mp4 (requests must be non-decreasing, repeats are free)."""

    def __init__(self, path):
        self.path = path
        self.r = imageio.get_reader(str(path)); self.i = -1; self.f = None
        self.n = self.r.count_frames()
        self.r = imageio.get_reader(str(path))

    def __call__(self, i):
        i = int(min(max(i, 0), self.n - 1))
        if i < self.i:
            self.r = imageio.get_reader(str(self.path)); self.i = -1
        while self.i < i:
            self.f = self.r.get_next_data(); self.i += 1
        return self.f


def blur_dim(img, amount, dim):
    """amount 0..1 -> gaussian blur radius up to 22 px; dim 0..1 darkening."""
    out = img
    if amount > 0.01:
        k = int(2 * round(22 * amount) + 1)
        out = cv2.GaussianBlur(img, (k, k), 0)
    if dim > 0.001:
        out = (out.astype(np.float32) * (1 - dim)).astype(np.uint8)
    return out


def vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    return np.clip(1.0 - 0.28 * np.clip(r - 0.55, 0, None) ** 1.6, 0, 1)[..., None]


VIG = vignette()


def studio_bg():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W * 0.5) / W) ** 2 + ((y - H * 0.42) / H) ** 2)
    c0, c1 = np.array([30, 33, 40], np.float32), np.array([9, 10, 12], np.float32)
    t = np.clip(r / 0.75, 0, 1)[..., None]
    return (c0 * (1 - t) + c1 * t).astype(np.uint8)


BG = studio_bg()


# ------------------------------------------------------------------ overlay drawing (RGBA layer composited on frame)
class Layer:
    def __init__(self):
        self.im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, f, fill, a=1.0, anchor="la", shadow=True):
        if a <= 0.003:
            return
        if shadow:
            sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(sh).text((xy[0] + 0, xy[1] + 2), s, font=f, fill=(0, 0, 0, int(150 * a)), anchor=anchor)
            sh = sh.filter(ImageFilter.GaussianBlur(4))
            self.im.alpha_composite(sh)
        self.d.text(xy, s, font=f, fill=(*fill, int(255 * a)), anchor=anchor)

    def rect(self, box, fill, a=1.0, r=0):
        if a <= 0.003:
            return
        col = (*fill[:3], int((fill[3] if len(fill) > 3 else 255) * a))
        if r:
            self.d.rounded_rectangle(box, r, fill=col)
        else:
            self.d.rectangle(box, fill=col)

    def glass(self, box, a=1.0, r=18):
        """Frosted card: darker translucent rounded rect with a thin light border."""
        if a <= 0.003:
            return
        self.d.rounded_rectangle(box, r, fill=(14, 16, 20, int(178 * a)), outline=(255, 255, 255, int(34 * a)), width=1)

    def line(self, pts, fill, a=1.0, w=2):
        if a > 0.003:
            self.d.line(pts, fill=(*fill, int(255 * a)), width=w)

    def dot(self, c, r, fill, a=1.0):
        if a > 0.003:
            self.d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(*fill, int(255 * a)))

    def over(self, frame):
        base = Image.fromarray(frame).convert("RGBA")
        base.alpha_composite(self.im)
        return np.asarray(base.convert("RGB"))


_GRAD = np.linspace(0, 1, 380, dtype=np.float32) ** 1.6


def bottom_shade(L, a, strength=165):
    """Soft dark gradient along the bottom edge so captions stay legible on bright floors."""
    if a <= 0.003:
        return
    al = (_GRAD * strength * a).astype(np.uint8)
    g = np.zeros((380, W, 4), np.uint8); g[..., 3] = al[:, None]
    L.im.alpha_composite(Image.fromarray(g, "RGBA"), (0, H - 380))


def caption(L, t, t0, t1, s, sub=None, x=96, y=944, shade=True):
    """Lower-third caption: amber bar + one line (+ optional small second line), slides in from the left."""
    a = ramp(t, t0, t1, 0.45)
    if a <= 0:
        return
    if shade:
        bottom_shade(L, a)
    dx = int((1 - smooth(min(1, (t - t0) / 0.45))) * -24)
    hgt = 44 if sub is None else 74
    L.rect([x + dx, y, x + dx + 5, y + hgt], (*AMBER, 255), a)
    L.text((x + dx + 22, y + 2), s, F["cap"], WHITE, a)
    if sub:
        L.text((x + dx + 22, y + 44), sub, F["capsub"], (212, 216, 224), a)


def section_card(L, t, dur, idx, title, sub):
    """Centered section title: index, title, growing accent rule, subtitle. t in [0, dur]."""
    a = ramp(t, 0.0, dur, 0.45)
    lift = int((1 - smooth(min(1, t / 0.6))) * 16)
    cy = H // 2 - 30 + lift
    L.text((W // 2, cy - 78), idx, F["idx"], AMBER, a, anchor="mm")
    L.text((W // 2, cy), title, F["title"], WHITE, a, anchor="mm")
    half = int(110 * smooth(min(1, max(0, (t - 0.15) / 0.7))))
    L.line([(W // 2 - half, cy + 56), (W // 2 + half, cy + 56)], AMBER, a, 3)
    L.text((W // 2, cy + 100), sub, F["sub"], GREY, a, anchor="mm")


def rounded_mask(tw, th, r=14):
    m = np.zeros((th, tw), np.uint8)
    cv2.rectangle(m, (r, 0), (tw - r - 1, th - 1), 255, -1); cv2.rectangle(m, (0, r), (tw - 1, th - r - 1), 255, -1)
    for cx, cy in ((r, r), (tw - r - 1, r), (r, th - r - 1), (tw - r - 1, th - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return (m.astype(np.float32) / 255)[..., None]


def cam_project(cam, lens, sensor, res, p):
    """Blender camera (loc, R columns = camera x, y, z axes in world) -> pixel coords."""
    loc, Rm = np.asarray(cam["loc"]), np.asarray(cam["R"])
    d = np.asarray(p, float) - loc
    x, y, z = d @ Rm[0], d @ Rm[1], d @ Rm[2]
    f = lens / sensor * res[0]
    return (res[0] / 2 + f * x / -z, res[1] / 2 - f * y / -z)


def pill(L, xy, text, f, fg, bg=(18, 20, 24, 225), a=1.0, pad=(14, 7), anchor="l", dot=None):
    if a <= 0.003:
        return
    tw = L.d.textlength(text, font=f); asc, desc = f.getmetrics(); th = asc + desc
    extra = 16 if dot else 0
    w, h = tw + 2 * pad[0] + extra, th + 2 * pad[1] - 4
    x, y = xy
    if anchor == "c":
        x -= w / 2
    elif anchor == "r":
        x -= w
    L.rect([x, y - h / 2, x + w, y + h / 2], bg, a, r=int(h / 2))
    if dot:
        L.dot((x + pad[0] + 4, y), 5, dot, a)
    L.text((x + pad[0] + extra, y - th / 2 + 1), text, f, fg, a, shadow=False)


def dashed(L, p0, p1, col, a, w=2, dash=9, gap=7):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    Ltot = np.linalg.norm(p1 - p0)
    if Ltot < 1:
        return
    u = (p1 - p0) / Ltot; s_ = 0.0
    while s_ < Ltot:
        e_ = min(Ltot, s_ + dash)
        L.line([tuple(p0 + u * s_), tuple(p0 + u * e_)], col, a, w)
        s_ += dash + gap


def arrow(L, tail, head, col, a, w=8, glow=True):
    """Thick 2D arrow with a soft glow; head is the arrow tip."""
    if a <= 0.003:
        return
    t, h = np.asarray(tail, float), np.asarray(head, float)
    v = h - t; ln = np.linalg.norm(v)
    if ln < 2:
        return
    u = v / ln; nrm = np.array([-u[1], u[0]])
    hl, hw = max(22, w * 3.2), max(16, w * 2.4)
    base = h - u * hl
    tri = [tuple(h), tuple(base + nrm * hw / 2), tuple(base - nrm * hw / 2)]
    if glow:
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
        gd.line([tuple(t), tuple(base)], fill=(*col, int(150 * a)), width=w + 10); gd.polygon(tri, fill=(*col, int(150 * a)))
        L.im.alpha_composite(g.filter(ImageFilter.GaussianBlur(9)))
    L.d.line([tuple(t), tuple(base + u * 2)], fill=(*col, int(255 * a)), width=w)
    L.d.polygon(tri, fill=(*col, int(255 * a)))


# ------------------------------------------------------------------ clips
class Clip:
    """A clip renders frame k (0..n-1) -> RGB uint8 (H, W, 3)."""

    def __init__(self, n, fn, name=""):
        self.n, self.fn, self.name = n, fn, name

    def __call__(self, k):
        return self.fn(k)


def card_clip(bg_fn, idx, title, sub, dur=None, unblur=0.7):
    dur = CUT["card_dur"] if dur is None else dur
    """Section card over a blurred, dimmed background that resolves into the shot during the last `unblur` s."""
    n = int(dur * FPS)

    def fn(k):
        t = k / FPS
        u = smooth((t - (dur - unblur)) / unblur) if t > dur - unblur else 0.0
        img = blur_dim(bg_fn(0), 1 - u, 0.55 * (1 - u))
        L = Layer()
        section_card(L, t, dur - 0.25, idx, title, sub)
        return L.over(img)
    return Clip(n, fn, f"card {idx}")


# ---- opening
def opening_clip(bg_fn, dur=4.2):
    n = int(dur * FPS)

    def fn(k):
        t = k / FPS
        img = blur_dim(bg_fn(0), 1.0, 0.62)
        img = (img.astype(np.float32) * smooth(t / 0.9)).astype(np.uint8)
        L = Layer()
        a = ramp(t, 0.5, dur, 0.6)
        L.text((W // 2, H // 2 - 40), "SpatialFit Chair", F["hero"], WHITE, a, anchor="mm")
        half = int(150 * smooth((t - 0.8) / 0.8))
        L.line([(W // 2 - half, H // 2 + 30), (W // 2 + half, H // 2 + 30)], AMBER, a, 3)
        L.text((W // 2, H // 2 + 78), "Robot-oriented modular design for personalized assembly", F["sub"], GREY, a, anchor="mm")
        L.text((W // 2, H // 2 + 128), "Group 4", F["small"], GREY, a * 0.85, anchor="mm")
        return L.over(img)
    return Clip(n, fn, "opening")


# ---- section 1: parametric generation
def gen_clip():
    J = json.load(open(BUILD / "video" / "shot_gen.json"))
    info = J["info"]
    proj = json.load(open(BUILD / "video" / "gen" / "proj_0000.json"))["proj"]
    src = PngSeq(str(BUILD / "video" / "gen" / "g_{:04d}.png"), J["n"])
    segs = {}
    for i, f in enumerate(info):
        segs.setdefault(f["seg"], [i, i])[1] = i
    s = lambda name: segs[name][0] / FPS
    e = lambda name: segs[name][1] / FPS
    rows = [("USER", None), ("Popliteal height", ("user", "popliteal_height")), ("Hip breadth", ("user", "hip_breadth")),
            ("Buttock-knee length", ("user", "buttock_knee_length")), ("Elbow above seat", ("user", "elbow_above_seat")),
            ("DESK", None), ("Top height", ("desk", "top_height")), ("Underside height", ("desk", "underside_height")),
            ("Clear width", ("desk", "clear_width")),
            ("CHAIR", None), ("Seat height  H", "H"), ("Seat width  W", "W"), ("Seat depth  D", "D"), ("Backrest  α", "alpha")]
    labels = [("LF", "Legs ×4", (-170, 40)), ("FR", "Width rails ×2", (-40, 150)),
              ("SR", "Side rails ×2", (60, 90)), ("SE", "Seat", (150, -50)), ("BA", "Backrest", (160, -40))]

    def fn(k):
        t = k / FPS
        f = info[k]
        L = Layer()
        # parameter panel
        a = ramp(t, 0.3, e("holdB") + 0.2, 0.5)
        if a > 0:
            x0, y0, w = 88, 214, 470
            L.glass([x0, y0, x0 + w, y0 + 652], a)
            morphing = f["seg"] == "morph"
            tag = "Chair A" if f["t"] < 0.5 else "Chair B"
            L.text((x0 + 28, y0 + 22), tag, F["big"], WHITE, a, shadow=False)
            if f["seg"] == "holdB":
                L.text((x0 + w - 28, y0 + 40), "✓ desk fit", F["smallb"], TEAL, a * ramp(t, s("holdB"), 99, 0.3),
                       anchor="ra", shadow=False)
            y = y0 + 96
            for lab, key in rows:
                if key is None:
                    L.text((x0 + 28, y + 8), lab, F["panel_h"], AMBER, a, shadow=False)
                    y += 36
                    continue
                v = f["inputs"][key[0]][key[1]] if isinstance(key, tuple) else f[key]
                unit = "°" if key == "alpha" else " mm"
                derived = not isinstance(key, tuple)
                col = AMBER if (morphing and derived) else WHITE
                L.text((x0 + 28, y), lab, F["panel"], GREY, a, shadow=False)
                L.text((x0 + w - 28, y), f"{v:.0f}{unit}", F["panel_v"], col, a, anchor="ra", shadow=False)
                y += 34
        caption(L, t, s("morph") + 0.2, e("holdB"), "Body & desk measurements \u2192 chair dimensions", x=1010)
        caption(L, t, s("explode") + 0.3, e("holdX") + 1, "10 modules \u00b7 fixed connector geometry", x=1010)
        # exploded-view labels
        la = ramp(t, s("holdX") + 0.1, e("holdX") + 1, 0.5)
        if la > 0:
            for part, text, (dx, dy) in labels:
                px, py = proj[k][part]
                tx, ty = px + dx, py + dy
                L.dot((px, py), 5, AMBER, la)
                L.line([(px, py), (tx, ty)], (230, 232, 236), la * 0.8, 2)
                anchor = "ls" if dx >= 0 else "rs"
                L.text((tx + (8 if dx >= 0 else -8), ty - 6), text, F["label"], WHITE, la, anchor=anchor)
        # live dimension callouts (H at the front-left corner, W along the front edge)
        da = ramp(t, 0.6, s("explode") + 0.1, 0.45)
        if da > 0:
            P = proj[k]; col = AMBER if f["seg"] == "morph" else (236, 238, 242)
            ht, hf, hr = P["dim:H_top"], P["dim:H_floor"], P["dim:H_ref"]
            dashed(L, hr, ht, (236, 238, 242), da * 0.7, 2)
            L.line([hf, ht], col, da, 3)
            for q in (ht, hf):
                L.line([(q[0] - 9, q[1]), (q[0] + 9, q[1])], col, da, 3)
            pill(L, (ht[0] - 16, (ht[1] + min(hf[1], H - 40)) / 2), f"H  {f['H']:.0f} mm", F["label"], col, a=da, anchor="r")
            wl, wr = P["dim:W_l"], P["dim:W_r"]
            dashed(L, P["dim:W_refl"], wl, (236, 238, 242), da * 0.7, 2); dashed(L, P["dim:W_refr"], wr, (236, 238, 242), da * 0.7, 2)
            L.line([wl, wr], col, da, 3)
            for q in (wl, wr):
                L.dot(q, 4, col, da)
            mid = ((wl[0] + wr[0]) / 2, (wl[1] + wr[1]) / 2 - 30)
            pill(L, mid, f"W  {f['W']:.0f} mm", F["label"], col, a=da, anchor="c")
        return L.over((src(k).astype(np.float32) * VIG).astype(np.uint8))
    return Clip(J["n"], fn, "s1 gen"), src


# ---- section 2: planning graphic + early-release fall
def plan_graphic_clip(bg_fn, dur=6.0):
    P = json.load(open(BUILD / "plan_A.json"))
    seq, fun = P["sequence"], P["funnel"]
    n = int(dur * FPS)
    stages = [(fun["all_orders"], "all orderings"), (fun["precedence"], "precedence-feasible"),
              (fun["geometric"], "collision-free insertion"), (fun["reachable"], "reachable (cuRobo IK)")]

    def fn(k):
        t = k / FPS
        img = blur_dim(bg_fn(0), 1.0, 0.6)
        L = Layer()
        A = ramp(t, 0.0, dur, 0.45)
        L.text((W // 2, 250), "Planned assembly sequence", F["chip"], GREY, A, anchor="mm")
        cw, gap = 118, 30
        x0 = W // 2 - (len(seq) * cw + (len(seq) - 1) * gap) // 2
        for i, pname in enumerate(seq):
            on = smooth((t - 0.4 - 0.22 * i) / 0.3)
            x = x0 + i * (cw + gap)
            fill = (int(40 + (AMBER[0] - 40) * on), int(44 + (AMBER[1] - 44) * on), int(52 + (AMBER[2] - 52) * on), 235)
            L.rect([x, 290, x + cw, 350], fill, A, r=12)
            L.text((x + cw // 2, 320), pname, F["step"], (20, 20, 22) if on > 0.5 else WHITE, A, anchor="mm", shadow=False)
            if i < len(seq) - 1:
                L.text((x + cw + gap // 2, 320), "›", F["step"], GREY, A * 0.8, anchor="mm", shadow=False)
        groups = [("legs", 0, 3), ("side rails", 4, 5), ("width rails", 6, 7), ("seat", 8, 8), ("backrest", 9, 9)]
        for gi, (gname, i0, i1) in enumerate(groups):
            ga = A * smooth((t - 0.6 - 0.22 * i1) / 0.4)
            xa_, xb_ = x0 + i0 * (cw + gap) + 6, x0 + i1 * (cw + gap) + cw - 6
            L.line([(xa_, 366), (xb_, 366)], GREY, ga * 0.9, 2)
            L.line([(xa_, 360), (xa_, 366)], GREY, ga * 0.9, 2); L.line([(xb_, 360), (xb_, 366)], GREY, ga * 0.9, 2)
            L.text(((xa_ + xb_) / 2, 392), gname, F["small"], (210, 214, 222), ga, anchor="mm", shadow=False)
        # funnel
        bw, bg_ = 330, 56
        fx0 = W // 2 - (4 * bw + 3 * bg_) // 2
        for i, (v, lab) in enumerate(stages):
            on = smooth((t - 2.2 - 0.45 * i) / 0.45)
            x = fx0 + i * (bw + bg_)
            L.glass([x, 470, x + bw, 640], A * on, r=16)
            shown = int(round(v if i == 0 else stages[0][0] + (v - stages[0][0]) * smooth((t - 2.2 - 0.45 * i) / 0.8)))
            col = TEAL if i == 3 else WHITE
            L.text((x + bw // 2, 535), f"{max(shown, v):,}", F["num"], col, A * on, anchor="mm", shadow=False)
            L.text((x + bw // 2, 595), lab, F["small"], GREY, A * on, anchor="mm", shadow=False)
            if i < 3:
                L.text((x + bw + bg_ // 2, 555), "→", F["num"], AMBER, A * on, anchor="mm", shadow=False)
        caption(L, t, 0.6, dur + 1, "Search: precedence → insertion clearance → reachability → stability")
        return L.over(img)
    return Clip(n, fn, "s2 plan")


def fall_clip():
    n_src = 196
    src = PngSeq(str(BUILD / "video" / "fall_A" / "frame_{:04d}.png"), n_src, start=1)
    hud = json.load(open(BUILD / "fall_A_hud.json"))
    disp = [e for e in hud["events"] if e["kind"] == "collapse"][0]["max_disp_mm"]
    # time remap: 1x hold, 0.5x slow-motion around the release + fall, 1x tail
    rel = 45                                     # log frame of the release (1.5 s)
    pos, p = [], 0.0
    while p < n_src - 1:
        pos.append(p)
        sp = 0.5 if rel - 4 <= p <= rel + 40 else 1.0
        p += sp
    pos += [n_src - 1] * 20
    n = len(pos)
    t_rel = pos.index(next(x for x in pos if x >= rel)) / FPS

    def fn(k):
        t = k / FPS
        L = Layer()
        caption(L, t, 0.2, t_rel + 0.1, "Stability check before each fixture release")
        caption(L, t, t_rel + 0.5, n / FPS + 1, f"Released after step 6 → the frame collapses ({disp:.0f} mm)",
                sub="Rejected — the fixture stays clamped until FR / BR close the frame (step 7)")
        if t_rel - 0.1 < t < t_rel + 1.6:              # brief red pulse badge
            a = ramp(t, t_rel - 0.1, t_rel + 1.6, 0.25)
            L.rect([W - 330, 70, W - 90, 118], (*RED, 230), a, r=24)
            L.text((W - 210, 94), "fixture released", F["chip"], WHITE, a, anchor="mm", shadow=False)
        return L.over(src(pos[k]))
    return Clip(n, fn, "s2 fall"), src


def stability_clip():
    """Step 6 vs step 7: fixture released + the same 3.6 N push on SL. Force arrow, support base, centre of mass."""
    Fr = json.load(open(BUILD / "video" / "fall2_frames.json"))
    lf = Fr["log_frames"]; nv = len(lf)
    hold = 54
    n = nv + hold
    ann = {c: json.load(open(BUILD / "video" / f"fall{c}_annot.json")) for c in (6, 7)}
    cam = {c: json.load(open(BUILD / "video" / f"fall{c}" / "camera.json")) for c in (6, 7)}
    src = {c: PngSeq(str(BUILD / "video" / f"fall{c}" / "frame_{:04d}.png"), nv, start=1) for c in (6, 7)}
    TW, TH, GAP, TY = 928, 522, 24, 200
    X = {6: (W - 2 * TW - GAP) // 2, 7: (W - 2 * TW - GAP) // 2 + TW + GAP}
    mask = rounded_mask(TW, TH)
    sc_ = TW / 1280.0
    t_rel_log = ann[6]["hold_t"]
    k_rel = next(k for k, li in enumerate(lf) if ann[6]["frames"][li]["t"] >= t_rel_log)
    push_k = [k for k, li in enumerate(lf) if ann[6]["frames"][li]["push"]]
    kp0, kp1 = push_k[0], push_k[-1]
    k_verdict = k_rel + int(2.6 * FPS)
    half_w = ann[6]["support_width_mm"] / 2
    com_z = ann[6]["frames"][0]["com"][2] * 1000
    tip = math.degrees(math.atan(half_w / com_z))
    title = {6: ("After step 6", "side frames not yet connected"), 7: ("After step 7", "FR joins the left and right frames")}
    verdict = {6: (f"\u00d7  Collapses \u00b7 {ann[6]['max_disp_mm']:.0f} mm  \u2192  release rejected", RED),
               7: (f"\u2713  Stable \u00b7 {ann[7]['max_disp_mm']:.1f} mm  \u2192  fixture can be released", TEAL)}

    def tile_pt(c, k, p):
        u, v = cam_project(cam[c]["frames"][min(k, nv - 1)], cam[c]["lens"], cam[c]["sensor_width"], cam[c]["res"], p)
        return (X[c] + u * sc_, TY + v * sc_)

    def fn(k):
        t = k / FPS
        kk = min(k, nv - 1)
        frame = BG.copy()
        for c in (6, 7):
            img = cv2.resize(src[c](kk), (TW, TH), interpolation=cv2.INTER_AREA)
            roi = frame[TY:TY + TH, X[c]:X[c] + TW].astype(np.float32)
            frame[TY:TY + TH, X[c]:X[c] + TW] = (img * mask + roi * (1 - mask)).astype(np.uint8)
        L = Layer()
        L.text((W // 2, 92), "Stability check: release the fixture, apply the same 3.6 N push", F["step"], WHITE, 1.0, anchor="mm")
        for c in (6, 7):
            A_ = ann[c]; fr = A_["frames"][lf[kk]]
            x = X[c]
            L.text((x + 4, TY - 46), title[c][0], F["step"], WHITE, 1.0)
            L.text((x + 4 + L.d.textlength(title[c][0], font=F["step"]) + 16, TY - 40), title[c][1], F["small"], GREY, 1.0)
            # support base on the floor
            base_a = ramp(t, 0.15, 99, 0.5)
            fill = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(fill)
            for poly, strong in ((A_["support"], True), (A_["support_other"], False)):
                if not poly:
                    continue
                pts = [tile_pt(c, kk, q) for q in poly]
                fd.polygon(pts, fill=(*TEAL, int((80 if strong else 34) * base_a)))
                fd.line(pts + [pts[0]], fill=(*TEAL, int((255 if strong else 110) * base_a)), width=3 if strong else 2)
            L.im.alpha_composite(fill)
            # centre of mass and its ground projection
            com = np.asarray(fr["com"]); lo, hi = np.asarray(A_["support"][0][:2]), np.asarray(A_["support"][2][:2])
            inside = bool(np.all(com[:2] >= lo - 1e-4) and np.all(com[:2] <= hi + 1e-4))
            ccol = TEAL if inside else RED
            pc, pg = tile_pt(c, kk, com), tile_pt(c, kk, [com[0], com[1], 0.001])
            dashed(L, pc, pg, ccol, base_a, 3)
            L.d.ellipse([pg[0] - 9, pg[1] - 5, pg[0] + 9, pg[1] + 5], outline=(*ccol, int(255 * base_a)), width=3)
            L.dot(pc, 9, (20, 22, 26), base_a); L.dot(pc, 6, (250, 250, 250), base_a)
            if k < k_rel:
                pill(L, (pc[0] + 18, pc[1] - 18), "centre of mass", F["small"], WHITE, a=base_a * ramp(t, 0.2, k_rel / FPS, 0.4))
            # force arrow while the push acts (held a moment longer so it can be read)
            fa = ramp(t, kp0 / FPS - 0.05, kp1 / FPS + 0.7, 0.15)
            if fa > 0:
                li_app = lf[min(max(k, kp0), kp1)]
                app = np.asarray(A_["frames"][li_app]["app"]); dvec = np.asarray(A_["push_N"]); dvec = dvec / np.linalg.norm(dvec)
                head, tail = tile_pt(c, kk, app - dvec * 0.02), tile_pt(c, kk, app - dvec * 0.26)
                arrow(L, tail, head, AMBER, fa, 8)
                pill(L, (tail[0] - 10, tail[1] - 4), "3.6 N", F["label"], AMBER, a=fa, anchor="r")
            # live displacement readout
            dmm = fr["disp_mm"]
            pill(L, (x + 18, TY + 34), f"max displacement  {dmm:5.1f} mm", F["small"], RED if dmm > 3 else WHITE, a=1.0)
            # status chip under the tile
            if k < k_rel:
                pill(L, (x, TY + TH + 38), "Fixture clamped", F["chip"], WHITE, dot=TEAL)
            elif k < k_verdict:
                pill(L, (x, TY + TH + 38), "Fixture released  +  push", F["chip"], WHITE, dot=AMBER)
            else:
                txt, col = verdict[c]
                va_ = smooth((k - k_verdict) / 12)
                pill(L, (x, TY + TH + 38), txt, F["chip"], (16, 16, 18), bg=(*col, 240), a=va_)
        caption(L, t, 0.3, k_rel / FPS + 0.2, "Before each fixture release the planner simulates the sub-assembly", y=868)
        caption(L, t, k_verdict / FPS + 0.3, n / FPS + 1,
                f"Open side frame stands on a {ann[6]['support_width_mm']:.0f} mm strip \u2192 tips over at ~{tip:.0f}\u00b0",
                sub="So the fixture stays clamped until step 7, when the frame is closed", y=854)
        return L.over(frame)
    cache = {}

    def first(i=0):
        if "f" not in cache:
            cache["f"] = fn(0)
        return cache["f"]
    return Clip(n, fn, "s2 stability"), first


# ---- section 3: synced split screen
SLOW = ("insert", "jam", "re-observe", "latch", "grasp", "fixture release", "stability")
PHASE = {"observe": "Observe", "transfer (cuRobo)": "Collision-free transfer · cuRobo", "approach": "Approach",
         "grasp": "Grasp", "lift + verify": "Lift & verify grasp", "descend to pre-insert": "Descend",
         "align": "Align to measured receivers", "insert (force control)": "Force-controlled insertion",
         "fixture clamp": "Fixture clamp", "release": "Release", "retreat": "Retreat", "return home": "Return",
         "jam -> retreat": "Jam detected → retreat", "latch": "Latch", "fixture release": "Release fixture",
         "stability check": "Stability check", "done": "Complete"}


def video_index(hud, speed=4, slow=1, fps=30):
    """Output-frame -> log-index list; must match blender_render2.py."""
    step = max(1, round(30 / fps))
    idx, i = [], 0
    while i < len(hud):
        idx.append(i)
        i += (slow if any(k in hud[i]["phase"] for k in SLOW) else speed) * step
    idx += [len(hud) - 1] * int(1.5 * fps)
    return idx


def remap_side(hud, idx, speed_fast, slow_phases, slow_speed=1.0):
    """Per step: list of output samples (video frame indices) after variable-speed retiming."""
    seg_of = []
    for vi, li in enumerate(idx):
        h = hud[li]
        seg_of.append(11 if (h["part"] == "" and vi > 10) else h["step"])
    steps = {}
    for vi, s in enumerate(seg_of):
        steps.setdefault(s, []).append(vi)
    out = {}
    for s, vis in steps.items():
        sp = np.array([slow_speed if any(k in hud[idx[v]]["phase"] for k in slow_phases) else speed_fast for v in vis])
        sp = np.convolve(np.pad(sp, 7, mode="edge"), np.ones(15) / 15, mode="valid")    # ease speed changes
        pos, p, j = [], 0.0, 0
        while p <= len(vis) - 1:
            pos.append(vis[int(p)])
            p += sp[min(int(p), len(sp) - 1)]
        out[s] = pos
    return out


CUT = dict(speed_fast=2.2, jam_speed=1.0, cap_dur=3.6, first_cap=(0.5, 4.6), end_frames=210, end_tail=1.2,
           end_labels=(0.9, 3.9), end_sum=4.0, card_dur=2.6, ins_cap=True, slow_end=False)
SHORT = dict(speed_fast=4.6, jam_speed=1.7, cap_dur=2.4, first_cap=(0.3, 2.6), end_frames=150, end_tail=0.8,
             end_labels=(0.5, 2.6), end_sum=2.6, card_dur=1.9, ins_cap=False, slow_end=True)
VIDS = {"A": BUILD / "acp_par" / "A_jam.mp4", "B": BUILD / "acp_par" / "B.mp4"}
TILES = (BUILD / "video" / "tiles.json").exists()     # locally rendered subset (scripts/video_split_frames.py)


class TileFrames:
    """Video-frame index -> PNG rendered by blender_render2.py --frames (only the frames the split uses)."""

    def __init__(self, side):
        T = json.load(open(BUILD / "video" / "tiles.json"))[side]
        self.k = {v: i for i, v in enumerate(T["video_frames"])}
        self.vs = T["video_frames"]
        self.seq = PngSeq(str(BUILD / "video" / f"tile_{side}" / "frame_{:04d}.png"), len(self.vs), start=1)

    def index(self, v):
        if v not in self.k:                                   # nearest rendered frame
            import bisect
            v = self.vs[min(bisect.bisect_left(self.vs, v), len(self.vs) - 1)]
        return self.k[v]

    def __call__(self, v):
        return self.seq(self.index(v))


def split_clip(max_frames=None):
    logs = {"A": "A_seed0_bias", "B": "B_seed0"}
    vids = VIDS
    huds = {k: json.load(open(BUILD / f"{v}_hud.json"))["hud"] for k, v in logs.items()}
    idxs = {k: video_index(huds[k]) for k in logs}
    summ = {k: json.load(open(BUILD / f"{v}_summary.json")) for k, v in logs.items()}
    endp = ("fixture release", "stability", "done") if CUT["slow_end"] else ()
    rem = {"A": remap_side(huds["A"], idxs["A"], CUT["speed_fast"], ("jam",) + endp, CUT["jam_speed"]),
           "B": remap_side(huds["B"], idxs["B"], CUT["speed_fast"], endp, CUT["jam_speed"])}
    # jam step on A plays near real time (1x of the rendered video, whose insert phases are already 1x log)
    timeline = []            # list of (vidx_A, vidx_B, step)
    for s in sorted(set(rem["A"]) | set(rem["B"])):
        a, b = rem["A"].get(s, []), rem["B"].get(s, [])
        Ln = max(len(a), len(b))
        for k in range(Ln):
            timeline.append((a[min(k, len(a) - 1)] if a else None, b[min(k, len(b) - 1)] if b else None, s))
    n = len(timeline) if max_frames is None else min(max_frames, len(timeline))
    srcs = {k: (TileFrames(k) if TILES else Video(v)) for k, v in vids.items()}
    TW, TH, GAP = 928, 522, 24
    X = {"A": (W - 2 * TW - GAP) // 2, "B": (W - 2 * TW - GAP) // 2 + TW + GAP}
    TY = 200
    ZOOM, ZOOM_DX = 1.08, -30
    CAP_Y = 868
    mask = np.zeros((TH, TW), np.uint8)
    cv2.rectangle(mask, (14, 0), (TW - 15, TH - 1), 255, -1); cv2.rectangle(mask, (0, 14), (TW - 1, TH - 15), 255, -1)
    for cx, cy in ((14, 14), (TW - 15, 14), (14, TH - 15), (TW - 15, TH - 15)):
        cv2.circle(mask, (cx, cy), 14, 255, -1)
    mask = (mask.astype(np.float32) / 255)[..., None]
    dims = {"A": "460 × 430 × 420 mm · 100°", "B": "500 × 460 × 470 mm · 110°"}
    # key moments (output-frame times) for captions
    def first(pred):
        for k, (va, vb, s) in enumerate(timeline):
            if pred(va, vb, s):
                return k / FPS
        return None
    ph = lambda side, v: huds[side][idxs[side][v]]["phase"] if v is not None else ""
    t_ins = first(lambda va, vb, s: "insert" in ph("A", va))
    t_jam = first(lambda va, vb, s: "jam" in ph("A", va))
    t_s7 = first(lambda va, vb, s: s == 7)
    t_ba = first(lambda va, vb, s: s == 10 and "insert" in ph("A", va))
    t_rel = first(lambda va, vb, s: s == 11 and "fixture release" in ph("A", va))
    T = n / FPS
    t_ins = first(lambda va, vb, s: s >= 5 and "insert" in ph("A", va)) or t_ins
    raw = [(*CUT["first_cap"], "Same planner, grasps and insertion primitives for both sizes", None)]
    if t_ins and CUT["ins_cap"]:
        raw.append((t_ins - 0.3, t_ins + CUT["cap_dur"], "Force-controlled insertion with wrist F/T feedback", None))
    if t_jam:
        t_s6 = first(lambda va, vb, s: s == 6) or t_jam
        raw.append((max(t_s6 + 0.2, t_jam - 1.4), (t_s7 or t_jam + 6) + (0.4 if CUT["slow_end"] else -0.2),
                    "5 mm perception error \u2192 jam detected \u2192 retreat \u2192 re-measure \u2192 reinsert", None))
    if t_ba:
        raw.append((t_ba - 0.3, t_ba + CUT["cap_dur"], "Backrest: same vertical insertion at 100\u00b0 and 110\u00b0", None))
    if t_rel:
        raw.append((t_rel, T + 1, "Fixture released \u2192 both chairs stand on their own",
                    f"max displacement  A {summ['A']['final_disp_mm']:.2f} mm  \u00b7  B {summ['B']['final_disp_mm']:.2f} mm"))
    caps, last = [], -1.0
    for t0, t1, txt, sub in sorted(raw):
        t0 = max(t0, last + 0.35)
        if t1 - t0 > 1.5:
            caps.append((t0, t1, txt, sub)); last = t1
    # force trace for A (log 30 Hz) during the jam step
    jam_step = huds["A"][idxs["A"][timeline[int(t_jam * FPS)][0]]]["step"] if t_jam else None
    fA = np.abs(np.array([h["force"] for h in huds["A"]]))
    tags = {k_: json.load(open(BUILD / "video" / f"tags_{k_}.json")) for k_ in ("A", "B")} if TILES else None
    run_start, run_end, cur = [0] * n, [0] * n, 0
    for k_ in range(1, n + 1):
        if k_ == n or timeline[k_][2] != timeline[cur][2]:
            for j in range(cur, k_):
                run_start[j], run_end[j] = cur, k_ - 1
            cur = k_

    def fn(k):
        t = k / FPS
        va, vb, s = timeline[k]
        frame = BG.copy()
        crop = {}
        for side, v in (("A", va), ("B", vb)):
            src_img = srcs[side](v)
            sh, sw = src_img.shape[:2]
            cw_, ch_ = int(sw / ZOOM), int(sh / ZOOM)               # gentle push-in on the work area
            cx0 = int((sw - cw_) / 2 + ZOOM_DX * sw / 1920); cy0 = int((sh - ch_) / 4)   # keep headroom for the arm
            img = cv2.resize(src_img[cy0:cy0 + ch_, cx0:cx0 + cw_], (TW, TH), interpolation=cv2.INTER_AREA)
            crop[side] = (cx0, cy0, cw_, ch_, sw)
            x = X[side]
            roi = frame[TY:TY + TH, x:x + TW].astype(np.float32)
            frame[TY:TY + TH, x:x + TW] = (img * mask + roi * (1 - mask)).astype(np.uint8)
        L = Layer()
        for side, v in (("A", va), ("B", vb)):
            x = X[side]
            L.text((x + 4, TY - 46), f"Chair {side}", F["step"], WHITE, 1.0)
            L.text((x + 128, TY - 40), dims[side], F["small"], GREY, 1.0)
            h = huds[side][idxs[side][v]]
            label = PHASE.get(h["phase"], h["phase"])
            jam = "jam" in h["phase"] or h.get("retry", 0) > 0
            col = AMBER if jam else TEAL
            tw = L.d.textlength(label, font=F["chip"])
            L.rect([x, TY + TH + 18, x + tw + 36, TY + TH + 58], (22, 25, 30, 235), 1.0, r=20)
            L.dot((x + 18, TY + TH + 38), 5, col)
            L.text((x + 30, TY + TH + 26), label, F["chip"], WHITE, 1.0, shadow=False)
        # live tag on the part being handled (fades across step changes)
        if tags is not None:
            ta = min(1.0, (k - run_start[k]) / 10.0, (run_end[k] - k) / 10.0)
            for side, v in (("A", va), ("B", vb)):
                tg = tags[side][srcs[side].index(v)]
                if not tg or ta <= 0:
                    continue
                cx0, cy0, cw_, ch_, sw = crop[side]
                sxy = 1280.0 / sw
                u = X[side] + (tg["uv"][0] / sxy - cx0) * TW / cw_
                w_ = TY + (tg["uv"][1] / sxy - cy0) * TH / ch_
                if not (X[side] + 20 < u < X[side] + TW - 200 and TY + 60 < w_ < TY + TH - 10):
                    continue
                a_ = smooth(ta)
                L.d.ellipse([u - 11, w_ - 11, u + 11, w_ + 11], outline=(*AMBER, int(230 * a_)), width=3)
                L.dot((u, w_), 4, AMBER, a_)
                L.line([(u + 8, w_ - 8), (u + 34, w_ - 34)], AMBER, a_, 2)
                pill(L, (u + 34, w_ - 34), f"{tg['part']} \u00b7 {tg['kind']}", F["small"], WHITE, a=a_)
        # step indicator (shared)
        stp = min(s, 10)
        part = huds["A"][idxs["A"][va]]["part"] if va is not None else ""
        L.text((W // 2, 92), f"Step {stp} / 10" + (f"  ·  {part}" if part else "  ·  complete"),
               F["step"], WHITE, 1.0, anchor="mm")
        segw, segg = 54, 8
        sx0 = W // 2 - (10 * segw + 9 * segg) // 2
        for i in range(10):
            done = i < stp - 1 or s == 11
            cur = i == stp - 1 and s != 11
            col = (*AMBER, 255) if cur else ((*TEAL, 255) if done else (60, 64, 72, 255))
            L.rect([sx0 + i * (segw + segg), 124, sx0 + i * (segw + segg) + segw, 130], col, 1.0, r=3)
        # captions (pre-scheduled, never overlapping)
        for c in caps:
            caption(L, t, c[0], c[1], c[2], sub=c[3], y=CAP_Y if c[3] is None else CAP_Y - 14)
        # force sparkline under tile A during the jam step
        if t_jam:
            a = ramp(t, t_jam - 0.6, (t_s7 or t_jam + 6) - 0.2, 0.4)
            if a > 0 and va is not None:
                li = idxs["A"][va]
                lo = max(0, li - 240)
                trace = fA[lo:li + 1]
                bx, by, bw, bh = X["A"] + TW - 330, TY + 18, 310, 120
                L.glass([bx, by, bx + bw, by + bh], a, r=14)
                L.text((bx + 16, by + 10), "insertion force", F["small"], GREY, a, shadow=False)
                L.text((bx + bw - 16, by + 10), f"{trace[-1]:.1f} N", F["smallb"], AMBER, a, anchor="ra", shadow=False)
                if len(trace) > 1:
                    fmax = 25.0
                    pts = [(bx + 16 + (bw - 32) * (j + 240 - len(trace) + 1) / 240,
                            by + bh - 14 - (bh - 50) * min(1, v_ / fmax)) for j, v_ in enumerate(trace)]
                    L.line(pts, AMBER, a, 3)
        return L.over(frame)
    cache = {}

    def first_frame(i=0):
        if "f" not in cache:
            cache["f"] = fn(0)
        return cache["f"]
    return Clip(n, fn, "s3 split"), first_frame


# ---- section 4: ending
def end_clip(dur_tail=None):
    dur_tail = CUT["end_tail"] if dur_tail is None else dur_tail
    n_src = CUT["end_frames"]
    src = PngSeq(str(BUILD / "video" / "end" / "g_{:04d}.png"), n_src)
    proj = json.load(open(BUILD / "video" / "end" / "proj_0000.json"))["proj"]
    n = n_src + int(dur_tail * FPS)
    lines = ["Parametric generation → 10 robot-friendly modules",
             "Automatic sequence planning with stability checks",
             "Closed-loop assembly · cuRobo transfer + force-controlled insertion"]

    def fn(k):
        t = k / FPS
        img = (src(min(k, n_src - 1)).astype(np.float32) * VIG).astype(np.uint8)
        L = Layer()
        a = ramp(t, 0.6, 99, 0.6)
        L.text((W // 2, 110), "Two personalized chairs \u00b7 one model \u00b7 one robot program", F["cap"], WHITE, a, anchor="mm")
        P = proj[min(k, n_src - 1)]
        la = ramp(t, *CUT["end_labels"], 0.5)
        for c, dims_ in (("A", "460 \u00d7 430 \u00d7 420 mm \u00b7 100\u00b0"), ("B", "500 \u00d7 460 \u00d7 470 mm \u00b7 110\u00b0")):
            px, py = P[f"lab:{c}"]
            L.rect([px - 150, py - 52, px + 150, py + 30], (16, 18, 22, 215), la, r=16)
            L.text((px, py - 26), f"Chair {c}", F["step"], WHITE, la, anchor="mm", shadow=False)
            L.text((px, py + 8), dims_, F["small"], (220, 224, 230), la, anchor="mm", shadow=False)
        bottom_shade(L, ramp(t, 0.9, 99, 0.6), 200)
        for i, s in enumerate(lines):
            ai = ramp(t, CUT["end_sum"] + 0.3 * i, 99, 0.5)
            L.dot((W // 2 - 430, 900 + 46 * i + 2), 5, AMBER, ai)
            L.text((W // 2 - 410, 900 + 46 * i - 15), s, F["capsub"], (226, 229, 235), ai)
        out = L.over(img)
        fo = smooth((t - (n / FPS - 1.2)) / 1.2)          # fade to black
        return (out.astype(np.float32) * (1 - fo)).astype(np.uint8)
    return Clip(n, fn, "s4 end"), src


# ------------------------------------------------------------------ timeline + encode
def crossfade(a, b, k, n):
    w = smooth(k / n)
    return (a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).astype(np.uint8)


def build(sections):
    clips = []                                   # (clip, xfade_frames_into_this_clip)
    gen, gen_src = gen_clip()
    if "s1" in sections:
        clips += [(opening_clip(gen_src), 0),
                  (card_clip(gen_src, "01", "Parametric Generation", "User & desk measurements → custom modules"), 18),
                  (gen, 0)]
    if "s2" in sections:
        stab, stab_first = stability_clip()
        clips += [(card_clip(stab_first, "02", "Assembly Sequence Planning", "Precedence \u00b7 collision \u00b7 reachability \u00b7 stability",
                             unblur=0.01), 24),
                  (plan_graphic_clip(stab_first), 0),
                  (Clip(int(0.8 * FPS), lambda k, s=stab_first: blur_dim(s(0), 1 - smooth(k / 24), 0.6 * (1 - smooth(k / 24))), "unblur"), 0),
                  (stab, 0)]
    if "s3" in sections:
        split, split_first = split_clip()
        clips += [(card_clip(lambda i: split_first(), "03", "Robotic Assembly", "Chair A and chair B · UR5e in MuJoCo"), 24),
                  (split, 0)]
    if "s4" in sections:
        end, end_src = end_clip()
        clips += [(card_clip(end_src, "04", "Result", "Same pipeline, two parameter sets"), 24), (end, 0)]
    return clips


def encode(clips, out, preview=False):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    size = (960, 540) if preview else (W, H)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "24" if preview else "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = 0
    tail = []                                    # last frames of the previous clip, blended into this clip's head
    for ci, (clip, xf) in enumerate(clips):
        print(f"[compose] {clip.name}: {clip.n} frames ({clip.n / FPS:.1f} s), xfade {xf}", flush=True)
        keep = clips[ci + 1][1] if ci + 1 < len(clips) else 0
        new_tail = []
        for k in range(clip.n):
            fr = clip(k)
            if k < len(tail):
                fr = crossfade(tail[k], fr, k + 1, len(tail) + 1)
            if k >= clip.n - keep:
                new_tail.append(fr)               # written as part of the next clip's crossfade
                continue
            if preview:
                fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
            p.stdin.write(fr.tobytes()); total += 1
        tail = new_tail
    for fr in tail:
        p.stdin.write((cv2.resize(fr, size, interpolation=cv2.INTER_AREA) if preview else fr).tobytes()); total += 1
    p.stdin.close(); p.wait()
    print(f"[compose] wrote {out} ({total} frames, {total / FPS:.1f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BUILD / "video" / "SpatialFit_demo.mp4"))
    ap.add_argument("--only", default="s1,s2,s3,s4")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--short", action="store_true", help="~30 s cut: robotic assembly + result only")
    a = ap.parse_args()
    if a.short:
        CUT.update(SHORT)
        if a.only == "s1,s2,s3,s4":
            a.only = "s3,s4"
    clips = build(a.only.split(","))
    if not a.only.startswith("s1"):                  # cut opens on a section card: fade in from black
        c0, x0 = clips[0]
        clips[0] = (Clip(c0.n, lambda k, f=c0.fn: (f(k).astype(np.float32) * smooth(k / 18)).astype(np.uint8), c0.name), x0)
    encode(clips, a.out, a.preview)
