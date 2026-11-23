import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
from scipy.spatial.transform import Rotation as R
cx, cy, n = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
wc = scene.Workcell(geometry.generate(params.load('configs/A.yaml')), chair_x=cx, chair_y=cy); m = wc.model
seq = ['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR', 'SE'][:['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR', 'SE', 'BA'].index(n)]
for k in wc.names:
    if k in seq: p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
    else: wc.set_part_pose(k, [5 + wc.names.index(k), 5, 1])
mujoco.mj_forward(m, wc.data)
ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0))
s = wc.parts[n]; tpos, _ = wc.target_pose(n)
for gi, g in enumerate(s.meta['grasps']):
    tcp, Rs = ex.grasp_world(tpos, R.identity(), s, g)
    for fi, Rm in enumerate(Rs):
        rt = R.from_matrix(Rm); ex.carry = (n, rt.inv().apply(tpos - tcp), rt.inv())
        for pan in np.linspace(-np.pi, np.pi, 9):
            q, ok = ex.ik.solve(tcp + [0, 0, 0.07], Rm, np.r_[pan, scene.HOME_Q[1:]])
            if not ok: continue
            clear = ex.scene_clear(q); d2 = ex.ik.d
            who = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, round(c.dist*1000,1)) for c in d2.contact[:d2.ncon] if c.dist < 0.001
                          and ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & (wc.robot_bodies | {wc.body[n]}))})
            q2, ok2 = ex.ik.solve(tcp, Rm, q)
            rc2 = ex.robot_clear(q2, n); who2 = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, round(c.dist*1000,1)) for c in d2.contact[:d2.ncon] if c.dist < 0.001 and ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & wc.robot_bodies)})
            print('   seated robot_clear', rc2, who2[:4])
            print('grasp', gi, 'flip', fi, 'pan', round(pan, 2), 'q', np.round(q, 2), 'clear', clear, 'seated', ok2, round(float(np.abs(q2 - q).max()), 2), who[:5])
            break
