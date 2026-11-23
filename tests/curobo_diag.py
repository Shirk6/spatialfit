"""Why does cuRobo fail to plan the seat transfer? Intercept the first carried transfer and run ablations."""
import sys, numpy as np, mujoco, torch
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
from spatialfit.scene import ARM_JOINTS
from curobo.types.robot import JointState
cfg = sys.argv[1] if len(sys.argv) > 1 else 'A'
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml'))); m, d = wc.model, wc.data
pre = ['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR']
for k in pre:
    p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
mujoco.mj_forward(m, d)
for k in pre:
    if wc.parts[k].kind == 'leg': wc.set_weld(f'clamp_{k}', True, body1_pose=wc.target_pose(k))
    for c in wc.parts[k].connectors:
        if c['host'] in pre: wc.set_weld(f"latch_{c['name']}", True)
for _ in range(500):
    d.ctrl[wc.arm_act] = scene.HOME_Q; mujoco.mj_step(m, d)
cp = motion.CuroboPlanner(wc)
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
cp_cfg_ign = load_yaml(join_path(get_robot_configs_path(), 'ur5e.yml'))['robot_cfg']['kinematics']['self_collision_ignore']
cp_cfg_ign.update({'wrist_3_link': cp_cfg_ign.get('wrist_3_link', []) + ['gripper_link'], 'wrist_2_link': cp_cfg_ign.get('wrist_2_link', []) + ['gripper_link'], 'wrist_1_link': cp_cfg_ign.get('wrist_1_link', []) + ['gripper_link'], 'gripper_link': ['tool0']})
ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0))
done = {'x': False}
orig = Executor.transfer_to

def js(q):
    return JointState.from_position(cp.t.to_device([list(q)]), joint_names=ARM_JOINTS)

def link_clearance(q):
    """min signed distance between attached-object spheres and every robot link's spheres (cuRobo model)."""
    kin = cp.mg.kinematics
    st = kin.get_state(cp.t.to_device([list(q)]))
    sph = st.link_spheres_tensor[0].cpu().numpy()          # (n, 4): x y z r
    names = kin.kinematics_config.link_name_to_idx_map
    idx_map = kin.kinematics_config.link_sphere_idx_map.cpu().numpy()
    inv = {v: k for k, v in names.items()}
    att = [i for i in range(len(sph)) if inv.get(int(idx_map[i])) == 'attached_object' and sph[i, 3] > 0]
    out = {}
    for i in range(len(sph)):
        ln = inv.get(int(idx_map[i]))
        if ln in ('attached_object', None) or sph[i, 3] <= 0:
            continue
        dmin = min(np.linalg.norm(sph[i, :3] - sph[j, :3]) - sph[i, 3] - sph[j, 3] for j in att) if att else 9
        out[ln] = min(out.get(ln, 9), dmin)
    return len(att), {k: round(float(v) * 1000, 1) for k, v in sorted(out.items(), key=lambda kv: kv[1])[:5]}

def transfer_diag(self, q_goal, exclude, grasp_target=None):
    if self.carry is not None and not done['x'] and len(sys.argv) > 2:
        done['x'] = True
        q0 = self.q_now(); g2 = q_goal.copy(); g2[4] -= 2 * np.pi * np.sign(g2[4])
        for wname, ex_ in (('full world + part', True), ('empty world + part', False)):
            if ex_: cp.update_world(exclude=exclude)
            else: cp.mg.update_world(cp.WorldConfig(cuboid=[cp.Cuboid(name='floor', pose=[0, 0, -1.0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]))
            for nm, st0 in (('start', q0), ('HOME', self.wrap_near(scene.HOME_Q, q0))):
                print('J', wname, 'from', nm, ': goal', np.round(q_goal[4], 2), cp.plan_js(st0, q_goal)[1], '| wrapped', np.round(g2[4], 2), cp.plan_js(st0, g2)[1])
        cp.update_world(exclude=exclude)
    if self.carry is not None and not done['x']:
        done['x'] = True
        q0 = self.q_now()
        cp.update_world(exclude=exclude)
        print('== carried part', self.carry[0])
        print('A full world:      start free', cp.collision_free(q0), ' goal free', cp.collision_free(q_goal), ' plan', cp.plan_js(q0, q_goal)[1])
        natt, clr = link_clearance(q_goal)
        print('   attached spheres', natt, ' closest robot links to attached part at goal (mm, <0 = overlap):', clr)
        natt0, clr0 = link_clearance(q0)
        print('   ... at start:', clr0)
        cp.mg.update_world(cp.WorldConfig(cuboid=[cp.Cuboid(name='floor', pose=[0, 0, -1.0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]))
        print('B empty world:     goal free', cp.collision_free(q_goal), ' plan', cp.plan_js(q0, q_goal)[1])
        cp.update_world(exclude=exclude)
        cp.detach()
        print('C part detached:   goal free', cp.collision_free(q_goal), ' plan', cp.plan_js(q0, q_goal)[1])
        from curobo.geom.sphere_fit import SphereFitType
        cub = cp._cuboids_of(self.carry[0], prefix='att2')
        cp.mg.attach_external_objects_to_robot(js(q0), cub, surface_sphere_radius=0.003, sphere_fit_type=SphereFitType.SAMPLE_SURFACE)
        print('D small spheres:   goal free', cp.collision_free(q_goal), ' plan', cp.plan_js(q0, q_goal)[1], link_clearance(q_goal)[1])
        cp.detach(); cp.attach(self.carry[0], q0)
        print('   start q', np.round(q0, 3))
        cp.mg.update_world(cp.WorldConfig(cuboid=[cp.Cuboid(name='floor', pose=[0, 0, -1.0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]))
        cp.detach()
        rng = np.random.default_rng(1)
        starts = [scene.HOME_Q] + [np.r_[pan, scene.HOME_Q[1:]] for pan in (-2.5, -0.8, 0.0, 0.8)]
        print('H empty world, no part, goal = seat placement pose:')
        for st0 in starts:
            print('     from', np.round(st0, 2), '->', cp.plan_js(st0, q_goal)[1])
        pg, Rg = self.ik.fk(q_goal)
        sols = []
        for k in range(300):
            sd = rng.uniform(-np.pi, np.pi, 6); sd[2] = rng.uniform(-2.8, 2.8)
            q, ok = self.ik.solve(pg, Rg, sd, iters=150)
            if ok and all(np.abs(self.wrap_near(q, o) - o).max() > 0.3 for o in sols):
                sols.append(q)
        print('I distinct IK branches for the same tool pose:', len(sols))
        for q in sols:
            print('     ', np.round(q, 2), 'mj_clear', self.scene_clear(q), 'cu_free', cp.collision_free(q), 'plan from HOME:', cp.plan_js(scene.HOME_Q, self.wrap_near(q, scene.HOME_Q))[1])
        cp.update_world(exclude=exclude); cp.attach(self.carry[0], q0)
        for j in (3, 4, 5):
            qg2 = q_goal.copy(); qg2[j] -= 2 * np.pi * np.sign(qg2[j])
            print(f'F goal with joint {j} wrapped by 2pi:', np.round(qg2, 2), cp.plan_js(q0, qg2)[1])
        # plan in two cuRobo segments: lift straight up first (Cartesian), then plan the rest
        up = self._lift_path(q0, 0.80)
        if up is not None:
            print('G cuRobo from lifted pose:', cp.plan_js(up[-1], q_goal)[1])
        lim = cp.mg.kinematics.get_joint_limits().position.cpu().numpy()
        print('E q_goal', np.round(q_goal, 3)); print('  cuRobo limits lo', np.round(lim[0], 2), ' hi', np.round(lim[1], 2))
        print('  outside limits:', [(j, round(float(q_goal[j]), 2)) for j in range(6) if not (lim[0][j] <= q_goal[j] <= lim[1][j])])
        print('  start outside limits:', [(j, round(float(q0[j]), 2)) for j in range(6) if not (lim[0][j] <= q0[j] <= lim[1][j])])
        # self-collision only: robot spheres vs robot spheres (cuRobo sphere model), excluding ignored pairs
        kin = cp.mg.kinematics
        stt = kin.get_state(cp.t.to_device([list(q_goal)]))
        sph = stt.link_spheres_tensor[0].cpu().numpy()
        names = kin.kinematics_config.link_name_to_idx_map; inv = {v: k for k, v in names.items()}
        idx_map = kin.kinematics_config.link_sphere_idx_map.cpu().numpy()
        lab = [inv.get(int(i)) for i in idx_map]
        ign = cp_cfg_ign
        pairs = {}
        for i in range(len(sph)):
            for j in range(i + 1, len(sph)):
                a, b = lab[i], lab[j]
                if a == b or sph[i, 3] <= 0 or sph[j, 3] <= 0 or 'attached_object' in (a, b):
                    continue
                if b in ign.get(a, []) or a in ign.get(b, []):
                    continue
                dd = np.linalg.norm(sph[i, :3] - sph[j, :3]) - sph[i, 3] - sph[j, 3]
                key = tuple(sorted((a, b)))
                pairs[key] = min(pairs.get(key, 9), dd)
        print('  closest robot self pairs (mm):', [(k, round(v * 1000, 1)) for k, v in sorted(pairs.items(), key=lambda kv: kv[1])[:5]])
        st = cp.mg.check_constraints(js(q_goal))
    return orig(self, q_goal, exclude, grasp_target)
Executor.transfer_to = transfer_diag
ex.installed = pre
ok = ex.build(['SE'])
print('RESULT', ok, ex.log)
