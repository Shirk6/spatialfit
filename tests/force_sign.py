import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
parts = geometry.generate(params.load('configs/A.yaml'))
wc = scene.Workcell(parts); ik = motion.DLSIK(wc)
ex = Executor(wc, ik, None, Recorder(wc), np.random.default_rng(0))
ex.gripper(0.0, 0.3)
p = np.array([-0.62, -0.25, scene.TABLE_H])   # free spot on left table
q, ok = ik.solve(p + [0, 0, 0.05], motion.tool_down(0), scene.HOME_Q); print('ik', ok)
ex.q_cmd = q; ex.wait(1.5)
print('free fz', round(ex._fz(), 2), 'tcp', np.round(wc.tcp_pose()[0], 4))
for dz in [0.0, -0.008, -0.010, -0.012, -0.014]:
    q, ok = ik.solve(p + [0, 0, dz + 0.008], motion.tool_down(0), ex.q_cmd)   # finger tips 8 mm below tcp
    ex.q_cmd = q; ex.wait(0.6)
    print(f'cmd tip depth {dz*1000:5.1f} mm  fz={ex._fz():7.2f}  ncon={wc.data.ncon}')
m, d = wc.model, wc.data
from collections import Counter
print(Counter(tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))) for c in d.contact[:d.ncon]))
