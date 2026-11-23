import sys, numpy as np
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
wc = scene.Workcell(geometry.generate(params.load('configs/A.yaml')))
for n in wc.names:   # empty world: move parts far away
    wc.set_part_pose(n, [5, 5, 1])
import mujoco; mujoco.mj_forward(wc.model, wc.data)
cp = motion.CuroboPlanner(wc)
qs = np.array([4.5, -1.22, 1.47, -1.82, -1.57, -3.35]); qg = np.array([2.42, -0.53, 1.08, -0.55, -0.71, -3.14])
for a, b in [(qs, qg), (qs, np.r_[qg[0], scene.HOME_Q[1:]]), (scene.HOME_Q, qg), (qs - [2*np.pi,0,0,0,0,0], qg)]:
    traj, st = cp.plan_js(a, b); print(np.round(a, 2), '->', np.round(b, 2), st, None if traj is None else traj.shape, 'free', cp.collision_free(a), cp.collision_free(b))
