"""Search floor staging poses for the seat (pallet) and backrest (stand) so both A and B picks are collision-free."""
import sys, itertools, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder

def feasible(table, cfg, names, installed_rest=True):
    wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')), staging=table); m, d = wc.model, wc.data
    mujoco.mj_forward(m, d)
    if any(c.dist < -1e-4 for c in d.contact[:d.ncon]):
        return False
    if installed_rest:   # seat / backrest are picked late: everything else already sits in the chair
        for k in wc.names:
            if k not in names:
                p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
        mujoco.mj_forward(m, d)
    ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
    for n in names:
        s = wc.parts[n]; pos, rot = ex.observe(n); tcp, Rs = ex.grasp_world(pos, rot, s)
        q, _ = ex.best_ik(tcp, Rs, 0.10, seed=scene.HOME_Q, check=(n, [0.05, 0.0]))
        if q is None:
            return False
    return True

table = dict(scene.STAGING)
found = {}
for n, xs, ys in (("SE", [0.62, 0.68, 0.74], [0.0, -0.08, 0.10]), ("BA", [0.40, 0.48, 0.56], [-0.34, -0.42, -0.50])):
    for x, y, yaw in itertools.product(xs, ys, [np.pi, -np.pi / 2, np.pi / 2, 0.0]):
        t = dict(table); t[n] = (x, y, yaw)
        if all(feasible(t, c, [n]) for c in 'AB'):
            table[n] = (x, y, yaw); found[n] = table[n]; print('found', n, table[n]); break
    else:
        print('no feasible pose for', n)
print('all parts feasible (initial scene):', all(feasible(table, c, list(table), installed_rest=False) for c in 'AB'))
print('STAGING =', {k: tuple(round(float(v), 4) for v in vals) for k, vals in table.items()})
