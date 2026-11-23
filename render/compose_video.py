"""Overlay the HUD on Blender frames and encode an mp4.

usage: python render/compose_video.py FRAMES_DIR LOG.npz CONFIG OUT.mp4 [title]
"""
import json, sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatialfit import params

fdir, log, cfg, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
title = sys.argv[5] if len(sys.argv) > 5 else f"Chair {cfg}"
p = params.load(f"configs/{cfg}.yaml")
hud = json.load(open(log.replace(".npz", "_hud.json")))["hud"]
meta = json.load(open(fdir / "frames.json"))
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PHASE = {"observe": "observe part", "observe hosts": "measure receivers", "transfer (cuRobo)": "collision-free transfer (cuRobo)",
         "approach": "approach", "grasp": "grasp", "lift + verify": "lift + verify grip", "re-observe": "re-measure (closed loop)",
         "align": "align to measured receivers", "descend to pre-insert": "descend", "insert (force control)": "insert - force control",
         "latch": "latch", "fixture clamp": "fixture clamp", "release": "release", "retreat": "retreat", "return home": "return home",
         "jam -> retreat": "JAM detected -> retreat", "re-observe + replan": "re-measure + retry", "fixture release": "release fixture",
         "stability check": "stability check", "done": "chair complete", "home": "return home", "init": ""}
w = imageio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8)
img0 = Image.open(fdir / "frame_00000.png"); W, H = img0.size
s = W / 1280
font_t, font_b, font_s = ImageFont.truetype(FB, int(24 * s)), ImageFont.truetype(F, int(19 * s)), ImageFont.truetype(F, int(15 * s))
for k, i in enumerate(meta["indices"]):
    fp = fdir / f"frame_{k:05d}.png"
    if not fp.exists():
        break
    im = Image.open(fp).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    h = hud[i]
    dr.rectangle([0, 0, W, int(84 * s)], fill=(16, 18, 22, 200))
    dr.text((int(20 * s), int(12 * s)), title, font=font_t, fill=(255, 255, 255, 255))
    dr.text((int(20 * s), int(48 * s)), f"W {p.W:.0f}  D {p.D:.0f}  H {p.H:.0f} mm   backrest {p.alpha:.0f} deg",
            font=font_s, fill=(190, 200, 210, 255))
    phase = PHASE.get(h["phase"], h["phase"])
    jam = "jam" in h["phase"] or h["retry"] > 0
    if h["part"]:
        step = f"step {h['step']}/{h['total']}   {h['part']}"
        dr.text((int(560 * s), int(14 * s)), step, font=font_b, fill=(255, 255, 255, 255))
        # progress bar
        x0, y0, x1 = int(560 * s), int(50 * s), int(800 * s)
        dr.rectangle([x0, y0, x1, y0 + int(8 * s)], fill=(70, 74, 82, 255))
        dr.rectangle([x0, y0, x0 + (x1 - x0) * (h["step"] - 1) / max(1, h["total"]), y0 + int(8 * s)], fill=(80, 200, 140, 255))
    dr.text((int(830 * s), int(14 * s)), phase, font=font_b, fill=(255, 200, 70, 255) if jam else (150, 210, 255, 255))
    if h["retry"]:
        dr.text((int(830 * s), int(48 * s)), f"retry {h['retry']}", font=font_s, fill=(255, 200, 70, 255))
    if "insert" in h["phase"]:
        dr.text((int(980 * s), int(48 * s)), f"axial force {h['force']:5.1f} N", font=font_s, fill=(255, 200, 70, 255))
    spd = "x1" if any(t in h["phase"] for t in ("insert", "jam", "re-observe", "latch", "grasp", "fixture", "stability")) else f"x{meta['speed']}"
    dr.text((W - int(210 * s), H - int(30 * s)), f"simulation  {spd}", font=font_s, fill=(40, 40, 40, 255))
    w.append_data(np.array(Image.alpha_composite(im, ov).convert("RGB")))
w.close()
print("wrote", out, k + 1, "frames")
