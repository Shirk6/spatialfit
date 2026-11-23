import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
parts = geometry.generate(params.load('configs/A.yaml'))
wc = scene.Workcell(parts); ik = motion.DLSIK(wc); cp = motion.CuroboPlanner(wc)
ex = Executor(wc, ik, cp, Recorder(wc), np.random.default_rng(0))
m, d = wc.model, wc.data
def status(tag):
    tp = wc.tcp_pose()[0]; cmdp = ik.fk(ex.q_cmd)[0]
    rob = [(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact[:d.ncon]
           if (m.geom_bodyid[c.geom1] in wc.robot_bodies) != (m.geom_bodyid[c.geom2] in wc.robot_bodies)]
    print(f'{tag:12s} tcp={np.round(tp,4)} cmd={np.round(cmdp,4)} |dq|={np.abs(ex.q_now()-ex.q_cmd).max():.4f} LF={np.round(wc.part_pose("LF")[0],4)} robot-contacts={set(rob)}')
ok, why = ex.pick('LF'); status(f'pick {why}')
tpos, trot = ex.target_from_hosts('LF'); print('target', tpos)
p_tcp, R_tcp = ex._tcp_target('LF', tpos, trot); print('tcp target', p_tcp)
q_pre, ok = ik.solve(p_tcp + [0, 0, 0.06], R_tcp, ex.q_cmd); print('ik ok', ok, 'fk', ik.fk(q_pre)[0])
ok, st = ex.transfer_to(q_pre, exclude=['LF']); status(f'transfer {st}')
ex.move_linear(p_tcp + [0, 0, 0.06], R_tcp, speed=0.04); status('align')
print('q_cmd', np.round(ex.q_cmd, 3)); print('q    ', np.round(ex.q_now(), 3))
print('act force', np.round(d.actuator_force[wc.arm_act], 1))
lf = wc.body['LF']
print('LF contacts', {(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon] if lf in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})
print('robot self contacts', {(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact[:d.ncon] if m.geom_bodyid[c.geom1] in wc.robot_bodies and m.geom_bodyid[c.geom2] in wc.robot_bodies})
print('active eq', [m.equality(i).name for i in range(m.neq) if d.eq_active[i]])
