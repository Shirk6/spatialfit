import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
wc = scene.Workcell(geometry.generate(params.load('configs/A.yaml'))); m, d = wc.model, wc.data
mujoco.mj_forward(m, d)
ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0))
for n in ['SE', 'BA']:
    s = wc.parts[n]; pos, rot = ex.observe(n); tcp, Rs = ex.grasp_world(pos, rot, s)
    for k, Rm in enumerate(Rs):
        back = -Rm[:, 2]
        for pan in np.linspace(-np.pi, np.pi, 9):
            q, ok = ex.ik.solve(tcp + back * 0.10, Rm, np.r_[pan, scene.HOME_Q[1:]])
            res = [ok, ok and ex.robot_clear(q, n)]
            qq = q
            for off in (0.05, 0.0):
                qq, ok2 = ex.ik.solve(tcp + back * off, Rm, qq)
                res += [ok2, ok2 and ex.robot_clear(qq, n)]
            print(n, k, round(pan, 2), res)
