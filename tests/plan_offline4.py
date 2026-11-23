import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
t = dict(scene.STAGING); t['SE'] = (0.68, 0.0, np.pi); t['BA'] = (0.15, -0.62, np.pi)
wc = scene.Workcell(geometry.generate(params.load('configs/A.yaml')), staging=t)
for n in wc.names:
    if n not in ('SE', 'BA'):
        p, q = wc.target_pose(n); wc.set_part_pose(n, p, q)
mujoco.mj_forward(wc.model, wc.data)
cp = motion.CuroboPlanner(wc)
ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
QS = np.array([4.5, -1.22, 1.47, -1.82, -1.57, -3.35])
s = wc.parts['SE']; pos, rot = ex.observe('SE'); tcp, Rs = ex.grasp_world(pos, rot, s)
cp.update_world(grasp_target='SE')
cands = ex.best_ik(tcp, Rs, 0.06, seed=QS, check=('SE', [0.03, 0.0]), all_solutions=True)
print('n cands', len(cands))
full = cp._world_cfg(set(), grasp_target='SE')
worlds = {'full': full, 'seat only': cp.WorldConfig(cuboid=[c for c in full.cuboid if c.name.startswith('SE')] + [c for c in full.cuboid if c.name == 'floor']),
          'no floor': cp.WorldConfig(cuboid=[c for c in full.cuboid if c.name != 'floor']),
          'empty-ish': cp.WorldConfig(cuboid=[c for c in full.cuboid if c.name == 'static_pedestal'])}
for q, Rm in cands[:3]:
    for wn, w in worlds.items():
        cp.mg.update_world(w)
        r1 = cp.plan_js(QS, q)[1]; r2 = cp.plan_js(scene.HOME_Q, q)[1]
        print(np.round(q, 2), wn, 'from QS:', r1, ' from HOME:', r2, 'goal free', cp.collision_free(q))
