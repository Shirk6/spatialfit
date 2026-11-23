import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
parts = geometry.generate(params.load('configs/A.yaml'))
wc = scene.Workcell(parts); ik = motion.DLSIK(wc); cp = motion.CuroboPlanner(wc)
ex = Executor(wc, ik, cp, Recorder(wc), np.random.default_rng(0))
m, d = wc.model, wc.data
s = parts['BA']; pos, rot = ex.observe('BA'); tcp, yaws = ex.grasp_world(pos, rot, s)
print('tab tcp world', tcp, 'yaws', np.degrees(yaws))
q, yaw = ex.best_ik(tcp + [0, 0, 0.1], yaws); print('chosen yaw', np.degrees(yaw))
ok, why = ex.pick('BA'); print(ok, why)
print('tcp now', wc.tcp_pose()[0], 'fingers', d.qpos[wc.finger_qadr])
print({(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon] if 'finger' in m.geom(c.geom1).name + m.geom(c.geom2).name or 'palm' in m.geom(c.geom1).name + m.geom(c.geom2).name})
import imageio
r = mujoco.Renderer(m, 480, 640); r.update_scene(d, 'chair'); a = r.render()
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.5, 0.3, 0.4]; cam.distance = 1.2; cam.azimuth = 200; cam.elevation = -20
r.update_scene(d, cam); b = r.render()
imageio.imwrite(sys.argv[1], np.concatenate([a, b], 1))
print('q', np.round(ex.q_now(), 2), 'qcmd', np.round(ex.q_cmd, 2))
print('robot contacts', {(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon] if (m.geom_bodyid[c.geom1] in wc.robot_bodies) or (m.geom_bodyid[c.geom2] in wc.robot_bodies)})
