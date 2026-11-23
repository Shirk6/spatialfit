import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
cfg, n = sys.argv[1], sys.argv[2]
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')))
if len(sys.argv) > 3:
    for _ in range(1500):
        wc.data.ctrl[wc.arm_act] = scene.HOME_Q; mujoco.mj_step(wc.model, wc.data)
    print('settled; LF pos', wc.part_pose('LF'))
cp = motion.CuroboPlanner(wc)
ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0))
from curobo.types.robot import JointState
s = wc.parts[n]; pos, rot = ex.observe(n); tcp, Rs = ex.grasp_world(pos, rot, s)
cp.update_world(grasp_target=n)
for k, Rm in enumerate(Rs):
    for pan in np.linspace(-np.pi, np.pi, 9):
        q, ok = ex.ik.solve(tcp - Rm[:, 2] * 0.10, Rm, np.r_[pan, scene.HOME_Q[1:]])
        if not ok: continue
        js = JointState.from_position(cp.t.to_device([list(q)]), joint_names=scene.ARM_JOINTS)
        st = cp.mg.check_constraints(js)
        print(k, round(pan, 2), np.round(q, 2), 'feasible', bool(st.feasible.item()), 'clear', ex.robot_clear(q, n),
              {kk: (v.item() if hasattr(v, 'item') and getattr(v, 'numel', lambda: 2)() == 1 else None) for kk, v in vars(st).items() if v is not None and kk in ('constraint', 'cost')})
        break
Rm = Rs[0]; q, ok = ex.ik.solve(tcp - Rm[:, 2] * 0.10, Rm, np.r_[-1.57, scene.HOME_Q[1:]])
for excl in [None] + wc.names:
    cp.mg.update_world(cp._world_cfg(set([excl]) if excl else set(), grasp_target=n if excl != n else None))
    print('exclude', excl, cp.collision_free(q))
cfgw = cp._world_cfg(set(), grasp_target=n)
for c in cfgw.cuboid:
    if 'static' in c.name or c.name == 'floor': print(c.name, np.round(c.pose, 3), np.round(c.dims, 3))
