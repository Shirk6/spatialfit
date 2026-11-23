import sys, mujoco, numpy as np, imageio
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene
cfg, out = sys.argv[1], sys.argv[2]
parts = geometry.generate(params.load(f'configs/{cfg}.yaml'))
wc = scene.Workcell(parts); m, d = wc.model, wc.data
r = mujoco.Renderer(m, 540, 960)
imgs = []
for cam in ['wide', 'chair']:
    mujoco.mj_forward(m, d); r.update_scene(d, cam); imgs.append(r.render())
for n in wc.names:
    p, q = wc.target_pose(n); wc.set_part_pose(n, p, q)
mujoco.mj_forward(m, d)
for cam in ['wide', 'chair']:
    r.update_scene(d, cam); imgs.append(r.render())
top = np.concatenate(imgs[:2], 1); bot = np.concatenate(imgs[2:], 1)
imageio.imwrite(out, np.concatenate([top, bot], 0))
