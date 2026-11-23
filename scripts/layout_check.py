"""Check floor staging: no overlaps, parts settle stably, every pick pose reachable and collision-free."""
import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
cfg = sys.argv[1] if len(sys.argv) > 1 else 'A'
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml'))); m, d = wc.model, wc.data
mujoco.mj_forward(m, d)
pen = [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist * 1000, 1)) for c in d.contact[:d.ncon] if c.dist < -1e-4]
print('initial penetrations', pen[:8])
p0 = {n: wc.part_pose(n)[0] for n in wc.names}
for _ in range(2000):
    d.ctrl[wc.arm_act] = scene.HOME_Q; mujoco.mj_step(m, d)
print('settle displacement mm', {n: round(float(np.linalg.norm(wc.part_pose(n)[0] - p0[n]) * 1000), 1) for n in wc.names})
ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0))
for n in wc.names:
    s = wc.parts[n]; pos, rot = ex.observe(n); tcp, Rs = ex.grasp_world(pos, rot, s)
    q, Rg = ex.best_ik(tcp, Rs, 0.10, seed=scene.HOME_Q, check=(n, [0.05, 0.0]))
    print(f'{n}: pick {"OK" if q is not None else "BLOCKED"} tcp={np.round(tcp, 3)}')
