import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
wc = scene.Workcell(geometry.generate(params.load('configs/A.yaml')))
for n in wc.names:
    if n not in ('SE', 'BA'):
        p, q = wc.target_pose(n); wc.set_part_pose(n, p, q)
mujoco.mj_forward(wc.model, wc.data)
cp = motion.CuroboPlanner(wc)
from spatialfit.executor import Executor, Recorder
ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
s_ = wc.parts['SE']; pos, rot = ex.observe('SE'); tcp, Rs = ex.grasp_world(pos, rot, s_)
cp.update_world(grasp_target='SE')
qs = np.array([4.5, -1.22, 1.47, -1.82, -1.57, -3.35]); ex.q_cmd = qs
q, Rg = ex.best_ik(tcp, Rs, 0.06, seed=qs, check=('SE', [0.03, 0.0]))
print('goal', None if q is None else np.round(q, 2))
okc = 0
for i in range(5):
    traj, st = cp.plan_js(qs, q); okc += traj is not None
print('direct success', okc, '/5')
