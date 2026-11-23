import sys, time, numpy as np
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
parts = geometry.generate(params.load('configs/A.yaml'))
wc = scene.Workcell(parts)
ik = motion.DLSIK(wc)
t0 = time.time(); cp = motion.CuroboPlanner(wc); print('curobo init', round(time.time()-t0, 1), 's')
# reachability of every grasp at staging and at target (+pre poses)
for n in wc.names:
    s = parts[n]
    for tag, (pos, yaw) in (('stage', wc.staging[n]), ('target', (wc.target_pose(n)[0], 0.0))):
        from scipy.spatial.transform import Rotation as R
        rp = R.from_euler('z', yaw)
        tcp = pos + rp.apply(s.grasp['tcp'])
        ax = rp.apply([1, 0, 0] if s.grasp['axis'] == 'x' else [0, 1, 0])
        yaw_g = np.arctan2(ax[1], ax[0]) - np.pi / 2   # tool y along grasp axis
        res = []
        for yy in (yaw_g, yaw_g + np.pi):
            for dz in (0.0, 0.10):
                q, ok = ik.solve(tcp + [0, 0, dz], motion.tool_down(yy), scene.HOME_Q)
                res.append(ok)
        ok_cu, _ = cp.ik_batch([(tcp, motion.tool_down(yaw_g)), (tcp, motion.tool_down(yaw_g + np.pi))])
        print(f'{n:3s} {tag:6s} tcp={np.round(tcp,3)} DLS ok={res} cuRobo ok={ok_cu.tolist()}')
q_goal, ok = ik.solve(wc.staging['LF'][0] + parts['LF'].grasp['tcp'] + [0, 0, 0.1], motion.tool_down(0), scene.HOME_Q)
t0 = time.time(); traj, st = cp.plan_js(scene.HOME_Q, q_goal); print('plan', st, None if traj is None else traj.shape, round(time.time()-t0, 3), 's')
