import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
from spatialfit.motion import tool_down
parts = geometry.generate(params.load('configs/B.yaml'))
wc = scene.Workcell(parts); ik = motion.DLSIK(wc)
ex = Executor(wc, ik, None, Recorder(wc), np.random.default_rng(0))
m = wc.model
for n in ['BR', 'FR', 'SL']:
    s = parts[n]; pos, rot = ex.observe(n); tcp, yaws = ex.grasp_world(pos, rot, s)
    for yaw in yaws:
        for sd in [scene.HOME_Q] + [np.r_[p, scene.HOME_Q[1:]] for p in np.linspace(-np.pi, np.pi, 9)]:
            q, ok = ik.solve(tcp + [0, 0, 0.1], tool_down(yaw), sd)
            if not ok: continue
            res = []
            for dz in [-0.05, -0.10]:
                q, ok2 = ik.solve(tcp + [0, 0, 0.1 + dz], tool_down(yaw), q)
                d2 = ex.ik.d; d2.qpos[:] = wc.data.qpos; d2.qpos[wc.arm_qadr] = q; mujoco.mj_forward(m, d2)
                bad = [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist*1000,1)) for c in d2.contact[:d2.ncon]
                       if (m.geom_bodyid[c.geom1] in wc.robot_bodies) != (m.geom_bodyid[c.geom2] in wc.robot_bodies) and c.dist < 0.001]
                res.append((ok2, bad[:3]))
            print(n, round(np.degrees(yaw)), round(sd[0], 2), res)
            break
