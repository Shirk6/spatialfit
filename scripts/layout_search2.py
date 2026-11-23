"""Joint search of seat-pallet and backrest-stand floor poses: both late picks must be IK-feasible, collision-free
(MuJoCo + cuRobo) and plannable by cuRobo from the configuration after the frame is complete (A and B)."""
import sys, itertools, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
QS = np.array([4.5, -1.22, 1.47, -1.82, -1.57, -3.35])          # robot after placing BR
SE_C = [(0.62, 0.10, -np.pi / 2), (0.62, -0.05, np.pi), (0.68, 0.0, np.pi), (0.55, 0.12, -np.pi / 2)]
BA_C = [(0.40, -0.40, np.pi), (0.30, -0.48, np.pi), (-0.50, -0.42, 0.0), (0.45, -0.45, np.pi / 2), (0.15, -0.62, np.pi)]
parts = {c: geometry.generate(params.load(f'configs/{c}.yaml')) for c in 'AB'}
cp = None
def score(se, ba):
    global cp
    tot = 0
    for c in 'AB':
        t = dict(scene.STAGING); t['SE'] = se; t['BA'] = ba
        wc = scene.Workcell(parts[c], staging=t); m, d = wc.model, wc.data
        mujoco.mj_forward(m, d)
        if any(x.dist < -1e-4 for x in d.contact[:d.ncon]):
            return -1
        for k in wc.names:
            if k not in ('SE', 'BA'):
                p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
        mujoco.mj_forward(m, d)
        if cp is None:
            cp = motion.CuroboPlanner(wc)
        cp.wc = wc
        ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
        for n in ('SE', 'BA'):
            if n == 'BA':   # seat is installed when the backrest is picked
                p, q = wc.target_pose('SE'); wc.set_part_pose('SE', p, q); mujoco.mj_forward(m, d)
            s = parts[c][n]; pos, rot = ex.observe(n); tcp, Rs = ex.grasp_world(pos, rot, s)
            cp.update_world(grasp_target=n)
            off = 0.06 if np.array(s.grasp['R'])[2, 2] > -0.7 else 0.10
            cands = ex.best_ik(tcp, Rs, off, seed=QS, check=(n, [off / 2, 0.0]), all_solutions=True)
            okn = 0
            for q, Rm in cands[:3]:
                tr, st = cp.plan_js(QS, q)
                if tr is not None:
                    okn = 1; break
            tot += okn
    return tot
best = None
for se, ba in itertools.product(SE_C, BA_C):
    sc = score(se, ba)
    print('SE', np.round(se, 2), 'BA', np.round(ba, 2), 'score', sc, flush=True)
    if sc == 4:
        best = (se, ba); break
print('BEST', best)
