import sys, itertools, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
cfg = sys.argv[1]
parts = geometry.generate(params.load(f'configs/{cfg}.yaml'))
cp = None
qs = np.array([4.5, -1.22, 1.47, -1.82, -1.57, -3.35])
for (x, y, yaw), bax in itertools.product([(0.62, -0.30, np.pi/2), (0.55, -0.30, np.pi/2), (0.62, 0.10, -np.pi/2), (0.55, 0.12, -np.pi/2), (0.70, -0.15, np.pi/2)], [(0.30, -0.40, 0.0), (-0.05, -0.62, 0.0)]):
    ph = 0.22; scene.PALLET_H = ph
    t = dict(scene.STAGING); t['SE'] = (x, y, yaw); t['BA'] = bax
    wc = scene.Workcell(parts, staging=t)
    for n in wc.names:
        if n not in ('SE', 'BA'):
            p, q = wc.target_pose(n); wc.set_part_pose(n, p, q)
    mujoco.mj_forward(wc.model, wc.data)
    if cp is None:
        cp = motion.CuroboPlanner(wc)
    cp.wc = wc
    ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
    pos, rot = ex.observe('SE'); tcp, Rs = ex.grasp_world(pos, rot, parts['SE'])
    cp.update_world(grasp_target='SE')
    q, Rg = ex.best_ik(tcp, Rs, 0.06, seed=qs, check=('SE', [0.03, 0.0]))
    okc = 0
    if q is not None:
        for st0 in (qs, scene.HOME_Q, qs):
            traj, st = cp.plan_js(st0, q); okc += traj is not None
    pen = [c for c in wc.data.contact[:wc.data.ncon] if c.dist < -1e-4]
    print(f'pen {len(pen)} BA {bax} SE x {x} y {y} yaw {yaw:.2f}: goal {None if q is None else np.round(q, 2)} plan {okc}/3', flush=True)
