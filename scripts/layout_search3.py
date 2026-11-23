"""Pick->place feasibility for the late, large parts (seat, backrest): cuRobo must plan home->pre-grasp and
lifted->pre-insert (part attached) in both chair configurations. Staging yaw = installed yaw (no reorientation)."""
import sys, itertools, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
from scipy.spatial.transform import Rotation as R
parts = {c: geometry.generate(params.load(f'configs/{c}.yaml')) for c in 'AB'}
cp = None

def check(c, table, n):
    global cp
    wc = scene.Workcell(parts[c], staging=table); m, d = wc.model, wc.data
    mujoco.mj_forward(m, d)
    if any(x.dist < -1e-4 for x in d.contact[:d.ncon]):
        return 'overlap'
    order = wc.names
    for k in order:
        if k != n and not (n == 'SE' and k == 'BA'):
            p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
    mujoco.mj_forward(m, d)
    if cp is None:
        cp = motion.CuroboPlanner(wc)
    cp.wc = wc
    ex = Executor(wc, motion.DLSIK(wc), cp, Recorder(wc), np.random.default_rng(0)); ex.obs_sigma = 0
    s = parts[c][n]; pos, rot = ex.observe(n); tcp, Rs = ex.grasp_world(pos, rot, s)
    cp.update_world(grasp_target=n)
    off = 0.06 if np.array(s.grasp['R'])[2, 2] > -0.7 else 0.10
    cands = ex.best_ik(tcp, Rs, off, seed=scene.HOME_Q, check=(n, [off / 2, 0.0]), all_solutions=True)
    for q_pre, Rg in cands[:3]:
        if cp.plan_js(scene.HOME_Q, q_pre)[0] is None:
            continue
        q_g, _ = ex.ik.solve(tcp, Rg, q_pre)
        q_l, _ = ex.ik.solve(tcp + [0, 0, 0.12], Rg, q_g)
        # attach at grasp, then move the part kinematically with the lifted pose for planning
        cp.attach(n, q_g)
        tpos, trot = wc.target_pose(n)
        # tool pose at placement keeps the same hand-part transform
        r_part = R.from_quat([rot.as_quat()[0], rot.as_quat()[1], rot.as_quat()[2], rot.as_quat()[3]])
        T_tp = r_part.inv() * R.from_matrix(Rg)
        p_in_part = r_part.inv().apply(tcp - pos)
        R_place = (scene.rot_of(trot) * T_tp).as_matrix()
        p_place = tpos + scene.rot_of(trot).apply(p_in_part)
        wc.set_part_pose(n, [5, 5, 1]); mujoco.mj_forward(m, d)      # part travels with the hand
        cp.update_world(exclude=[n])
        ok = False
        for sd in [q_l] + [np.r_[pan, scene.HOME_Q[1:]] for pan in np.linspace(-np.pi, np.pi, 9)]:
            q_pi, okk = ex.ik.solve(p_place + [0, 0, 0.07], R_place, sd)
            if okk and cp.plan_js(q_l, ex.wrap_near(q_pi, q_l))[0] is not None:
                ok = True; break
        cp.detach()
        wc.set_part_pose(n, pos, scene.quat_wxyz(rot)); mujoco.mj_forward(m, d)
        if ok:
            return 'ok'
    return 'fail' if cands else 'no-ik'

if __name__ == '__main__':
    base = dict(scene.STAGING)
    base.update({"LF": (0.10, -0.46, 0.0), "RF": (0.24, -0.46, 0.0), "LR": (0.38, -0.46, 0.0), "RR": (0.52, -0.46, 0.0),
                 "FR": (0.36, 0.02, np.pi / 2), "BR": (0.44, 0.02, np.pi / 2), "SL": (0.52, 0.02, 0.0), "SR": (0.60, 0.02, 0.0)})
    for se in [(-0.62, 0.0, 0.0), (-0.58, 0.12, 0.0), (-0.66, -0.10, 0.0)]:
        t = dict(base); t['SE'] = se; t['BA'] = (-0.30, -0.55, 0.0)
        print('SE', se, [check(c, t, 'SE') for c in 'AB'], flush=True)
    for ba in [(-0.30, -0.55, 0.0), (-0.20, -0.60, 0.0), (-0.40, -0.48, 0.0), (-0.10, -0.62, 0.0)]:
        t = dict(base); t['SE'] = (-0.62, 0.0, 0.0); t['BA'] = ba
        print('BA', ba, [check(c, t, 'BA') for c in 'AB'], flush=True)
