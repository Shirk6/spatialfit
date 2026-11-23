"""Which (grasp, jaw flip) of the seat / backrest admit a self-collision-free, collision-free placement?"""
import sys, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
from scipy.spatial.transform import Rotation as R
for cfg in 'AB':
    wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')))
    for k in wc.names:
        if k not in ('SE', 'BA'):
            p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
    mujoco.mj_forward(wc.model, wc.data)
    ex = Executor(wc, motion.DLSIK(wc), None, Recorder(wc), np.random.default_rng(0))
    for n in ('SE', 'BA'):
        if n == 'BA':
            p, q = wc.target_pose('SE'); wc.set_part_pose('SE', p, q); mujoco.mj_forward(wc.model, wc.data)
        s = wc.parts[n]
        tpos, tq = wc.target_pose(n)
        for gi, g in enumerate(s.meta['grasps']):
            tcp, Rs = ex.grasp_world(tpos, R.identity(), s, g)
            for fi, Rm in enumerate(Rs):
                print(cfg, n, 'grasp', gi, 'flip', fi, 'place ok:', ex.place_lookahead(n, tcp, Rm, tpos, R.identity()))
