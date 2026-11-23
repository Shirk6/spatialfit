"""Video shot for the planner's stability check: the frame after step N (4 legs + 2 side rails) with the fixture
released too early collapses under gravity -> why the planner keeps the clamps until FR/BR close the frame.

usage: python scripts/video_fall.py [CONFIG] [N_INSTALLED]  -> build/fall_<CONFIG>.npz (+ _hud.json)
"""
import sys, json
import numpy as np
import mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene
from spatialfit.executor import Recorder

cfg = sys.argv[1] if len(sys.argv) > 1 else 'A'
n_inst = int(sys.argv[2]) if len(sys.argv) > 2 else 6
seq = json.load(open(f'build/plan_{cfg}.json'))['sequence']
installed = seq[:n_inst]
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')))
m, d = wc.model, wc.data
wc.reset()
for _ in range(1500):                       # settle staged parts (same as run_build)
    d.ctrl[wc.arm_act] = scene.HOME_Q; mujoco.mj_step(m, d)
for n in installed:
    pos, q = wc.target_pose(n)
    wc.set_part_pose(n, pos, q)
mujoco.mj_forward(m, d)
for n in installed:
    for c in wc.parts[n].connectors:
        if c["host"] in installed:
            wc.set_weld(f"latch_{c['name']}", True)
    if wc.parts[n].kind == "leg":
        wc.set_weld(f"clamp_{n}", True, body1_pose=wc.target_pose(n))
wc.set_nests(True)
d.time = 0.0
rec = Recorder(wc)
rec.set(step=n_inst, total=10, part="", phase="fixture holding")


def run(sec, phase, push=None):
    rec.set(phase=phase)
    t0 = d.time
    for _ in range(int(sec / m.opt.timestep)):
        d.ctrl[wc.arm_act] = scene.HOME_Q
        if push is not None:   # 3.6 N disturbance (planner self-support check magnitude), pushed away from the camera, first 0.3 s
            d.xfrc_applied[wc.body[push], :3] = [-3.0, -2.0, 0.0] if d.time - t0 < 0.3 else 0.0
        mujoco.mj_step(m, d)
        rec.tick()
    d.xfrc_applied[:] = 0


run(1.5, "fixture holding")
p0 = {n: d.xpos[wc.body[n]].copy() for n in installed}
rec.event("fixture_release_early", installed=installed)
for n in installed:
    if wc.parts[n].kind == "leg":
        wc.set_weld(f"clamp_{n}", False)
wc.set_nests(False)
run(3.5, "fixture release", push=installed[-1])
disp = max(np.linalg.norm(d.xpos[wc.body[n]] - p0[n]) for n in installed)
rec.event("collapse", max_disp_mm=round(disp * 1000, 1))
rec.save(f'build/fall_{cfg}.npz')
print(json.dumps(dict(installed=installed, max_disp_mm=round(disp * 1000, 1), frames=len(rec.xpos))))
