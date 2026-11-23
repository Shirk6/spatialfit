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
        best = None
        for pan in np.linspace(-np.pi, np.pi, 13):
            q, ok = ex.ik.solve(tcp - Rm[:, 2] * 0.0, Rm, np.r_[pan, scene.HOME_Q[1:]])
            if not ok: continue
            d2 = ex.ik.d; d2.qpos[:] = d.qpos; d2.qpos[wc.arm_qadr] = q; mujoco.mj_forward(m, d2)
            bad = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, round(c.dist * 1000)) for c in d2.contact[:d2.ncon]
                   if (m.geom_bodyid[c.geom1] in wc.robot_bodies) != (m.geom_bodyid[c.geom2] in wc.robot_bodies) and c.dist < 0.001})
            print(n, 'flip', k, 'pan', round(pan, 2), 'q', np.round(q, 2), 'contacts', bad[:4])
