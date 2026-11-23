import sys, itertools, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
P = {c: geometry.generate(params.load(f'configs/{c}.yaml')) for c in 'AB'}
def pickable(wc, n):
    ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
    s = wc.parts[n]; pos, rot = ex.observe(n)
    for g in s.meta['grasps']:
        tcp, Rs = ex.grasp_world(pos, rot, s, g)
        off = 0.06 if Rs[0][2, 2] > -0.7 else 0.10
        sols = ex.best_ik(tcp, Rs, off, seed=scene.HOME_Q, check=(n, [off / 2, 0.0], g['width'] + 0.001), all_solutions=True)
        for q, Rm in sols:
            if ex.place_lookahead(n, tcp, Rm, pos, rot, g['width']):
                return True
    return False
for se, ba in itertools.product([-60, -75, -90, -105], [150, 165, 180, 195, 210]):
    ok = True
    for c in 'AB':
        t = dict(scene.STAGING); t['SE'] = ('orbit', np.radians(se)); t['BA'] = ('orbit', np.radians(ba))
        wc = scene.Workcell(P[c], staging=t); m, d = wc.model, wc.data; mujoco.mj_forward(m, d)
        if any(x.dist < -1e-4 for x in d.contact[:d.ncon]):
            ok = 'overlap'; break
        for n in ('SE', 'BA'):
            for k in ['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR'] + (['SE'] if n == 'BA' else []):
                p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
            mujoco.mj_forward(m, d)
            if not pickable(wc, n):
                ok = f'{c}{n} not pickable'; break
        if ok is not True: break
    print('SE', se, 'BA', ba, '->', ok, flush=True)
