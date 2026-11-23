import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
cfg, n = sys.argv[1], sys.argv[2]
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml'))); m = wc.model
if len(sys.argv) > 3:
    for k in ['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR'] + (['SE'] if n == 'BA' else []):
        p_, q_ = wc.target_pose(k); wc.set_part_pose(k, p_, q_)
mujoco.mj_forward(m, wc.data)
ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
s = wc.parts[n]; pos, rot = ex.observe(n)
for gi, g in enumerate(s.meta['grasps']):
    tcp, Rs = ex.grasp_world(pos, rot, s, g)
    for fi, Rm in enumerate(Rs):
        back = -Rm[:, 2]
        for pan in np.linspace(-np.pi, np.pi, 9):
            q, ok = ex.ik.solve(tcp + back * 0.06, Rm, np.r_[pan, scene.HOME_Q[1:]])
            if not ok: continue
            r = [ex.robot_clear(q, n)]
            for off in (0.03, 0.0):
                q, ok2 = ex.ik.solve(tcp + back * off, Rm, q); r.append((ok2, ex.robot_clear(q, n)))
            r.append(ex.robot_clear(q, n, finger_width=g['width'] + 0.001))
            q1, _ = ex.ik.solve(tcp + back * 0.06, Rm, np.r_[pan, scene.HOME_Q[1:]]); r.insert(0, ('open', ex.robot_clear(q1, n, finger_width=g['width'] + 0.016)))
            d2 = ex.ik.d
            who = sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist*1000,1)) for c in d2.contact[:d2.ncon] if c.dist < 0.001 and ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & wc.robot_bodies)})
            print(gi, fi, round(pan, 2), 'tcp', np.round(tcp, 3), r, who[:4]); break
print('--- best_ik direct')
for g in s.meta['grasps']:
    tcp, Rs = ex.grasp_world(pos, rot, s, g)
    sols = ex.best_ik(tcp, Rs, 0.06, seed=scene.HOME_Q, check=(n, [0.03, 0.0], g['width'] + 0.001), all_solutions=True)
    print('tool z', np.round(Rs[0][:, 2], 2), len(sols))
    Rm = Rs[0]; back = -Rm[:, 2]
    for sd in [scene.HOME_Q] + [np.r_[pan, scene.HOME_Q[1:]] for pan in np.linspace(-np.pi, np.pi, 9)]:
        q, ok = ex.ik.solve(tcp + back * 0.06, Rm, sd)
        if not ok: continue
        q = ex.wrap_near(q, scene.HOME_Q); qq = q.copy(); res = []
        for off in (0.03, 0.0):
            qq, ok2 = ex.ik.solve(tcp + back * off, Rm, qq); res.append((ok2, ex.robot_clear(qq, n, finger_width=g['width'] + 0.016)))
        res.append(ex.robot_clear(qq, n, finger_width=g['width'] + 0.001))
        print('   seed', round(sd[0], 2), res)
g = s.meta['grasps'][1]; tcp, Rs = ex.grasp_world(pos, rot, s, g); Rm = Rs[0]
q, ok = ex.ik.solve(tcp, Rm, np.r_[-3.14, scene.HOME_Q[1:]])
print('clear', ex.robot_clear(q, n, finger_width=g['width'] + 0.016)); d2 = ex.ik.d
print(sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist*1000,1)) for c in d2.contact[:d2.ncon] if c.dist < 0.001 and ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & wc.robot_bodies)}))
