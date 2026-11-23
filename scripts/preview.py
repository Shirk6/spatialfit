"""Quick MuJoCo render of a recorded build log (for checking; the final video is rendered in Blender).

usage: preview.py LOG.npz CONFIG OUT.mp4 [speed=3] [title]
"""
import sys, json, numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene

log, cfg, out = sys.argv[1], sys.argv[2], sys.argv[3]
speed = int(sys.argv[4]) if len(sys.argv) > 4 else 3
title = sys.argv[5] if len(sys.argv) > 5 else f"Chair {cfg}"
z = np.load(log); hud = json.load(open(log.replace('.npz', '_hud.json')))['hud']
p = params.load(f'configs/{cfg}.yaml')
wc = scene.Workcell(geometry.generate(p))
m, d = wc.model, wc.data
W, H = 1280, 720
r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.05, 0.12, 0.3]; cam.distance = 2.3; cam.azimuth = -125; cam.elevation = -38
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
except OSError:
    font = small = ImageFont.load_default()
w = imageio.get_writer(out, fps=30, codec='libx264', quality=7, macro_block_size=8)
for i in range(0, len(z['qpos']), speed):
    d.qpos[:] = z['qpos'][i]; d.mocap_pos[:] = z['mocap'][i]; mujoco.mj_forward(m, d)
    r.update_scene(d, cam)
    im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
    h = hud[i]
    dr.rectangle([0, 0, W, 78], fill=(20, 22, 26))
    dr.text((16, 10), f"{title}   W{p.W:.0f} x D{p.D:.0f} x H{p.H:.0f} mm, back {p.alpha:.0f} deg", font=font, fill=(255, 255, 255))
    step = f"step {h['step']}/{h['total']}  {h['part']}" if h['part'] else ""
    line = f"{step}   {h['phase']}"
    if h['retry']:
        line += f"   retry {h['retry']}"
    if h['note']:
        line += f"   [{h['note']}]"
    dr.text((16, 44), line, font=small, fill=(255, 205, 80) if (h['retry'] or 'jam' in h['phase']) else (190, 220, 255))
    if 'insert' in h['phase']:
        dr.text((W - 300, 44), f"insertion force {h['force']:5.1f} N", font=small, fill=(255, 205, 80))
    dr.text((W - 260, H - 30), f"sim t = {h['t']:6.1f} s   x{speed}", font=small, fill=(60, 60, 60))
    w.append_data(np.array(im))
w.close()
print("wrote", out)
