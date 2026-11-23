import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
wc = scene.Workcell(geometry.generate(params.load('configs/A.yaml')))
for n in wc.names:
    if n not in ('SE', 'BA'):
        p, q = wc.target_pose(n); wc.set_part_pose(n, p, q)
mujoco.mj_forward(wc.model, wc.data)
cp = motion.CuroboPlanner(wc)
qs = np.array([4.5, -1.22, 1.47, -1.82, -1.57, -3.35]); qg = np.array([2.42, -0.53, 1.08, -0.55, -0.71, -3.14])
full = cp._world_cfg(set(), grasp_target='SE')
names = sorted({c.name.rsplit('_', 1)[0] for c in full.cuboid})
cp.mg.update_world(full); print('full', cp.plan_js(qs, qg)[1])
for grp in names:
    w = cp.WorldConfig(cuboid=[c for c in full.cuboid if c.name.rsplit('_', 1)[0] != grp])
    cp.mg.update_world(w); tr, st = cp.plan_js(qs, qg)
    if tr is not None: print('removing', grp, '->', st)
