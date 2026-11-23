import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
from scipy.spatial.transform import Rotation as R
import math
for cx, cy, yaw in [(0.0, 0.50, 0.0), (0.0, 0.56, 0.0), (0.20, 0.50, 0.0), (0.0, 0.52, math.pi), (0.20, 0.50, math.pi), (0.0, 0.58, math.pi), (0.52, 0.0, -math.pi/2), (0.52, 0.0, math.pi/2)]:
    out = []
    for cfg in 'AB':
        wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')), chair_x=cx, chair_y=cy, chair_yaw=yaw)
        for k in wc.names:
            p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
        mujoco.mj_forward(wc.model, wc.data)
        ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0))
        seq = ['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR', 'SE', 'BA']
        for i, n in enumerate(seq):
            for j, k in enumerate(seq):
                if j < i:
                    p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
                else:
                    wc.set_part_pose(k, [5 + j, 5, 1])
            s = wc.parts[n]; tpos, tq = wc.target_pose(n); trot = scene.rot_of(tq)
            mujoco.mj_forward(wc.model, wc.data)
            ok = False
            for g in s.meta['grasps']:
                tcp, Rs = ex.grasp_world(tpos, trot, s, g)
                for Rm in Rs:
                    if ex.place_lookahead(n, tcp, Rm, tpos, trot, g['width']):
                        ok = True; break
                if ok: break
            if not ok: out.append(cfg + n)
    print(cx, cy, round(yaw, 2), 'not placeable:', out, flush=True)
