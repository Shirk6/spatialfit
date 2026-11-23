"""Fast harness: chair frame pre-installed (latched), run only the late parts (SE, BA) through the executor."""
import sys, json, numpy as np, mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion
from spatialfit.executor import Executor, Recorder
cfg = sys.argv[1] if len(sys.argv) > 1 else 'A'
parts_ = sys.argv[2].split(',') if len(sys.argv) > 2 else ['SE', 'BA']
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml'))); m, d = wc.model, wc.data
pre = [k for k in ['LF', 'LR', 'RR', 'RF', 'SR', 'SL', 'FR', 'BR', 'SE', 'BA'] if k not in parts_]
for k in pre:
    p, q = wc.target_pose(k); wc.set_part_pose(k, p, q)
mujoco.mj_forward(m, d)
for k in pre:
    if wc.parts[k].kind == 'leg':
        wc.set_weld(f'clamp_{k}', True, body1_pose=wc.target_pose(k))
    for c in wc.parts[k].connectors:
        if c['host'] in pre:
            wc.set_weld(f"latch_{c['name']}", True)
for _ in range(500):
    d.ctrl[wc.arm_act] = scene.HOME_Q; mujoco.mj_step(m, d)
cp = motion.CuroboPlanner(wc)
rec = Recorder(wc)
ex = Executor(wc, motion.DLSIK(wc), cp, rec, np.random.default_rng(0)); ex.debug = True
ex.installed = pre
ok = ex.build(parts_)
print('RESULT', ok, ex.log)
rec.save(f'build/late_{cfg}.npz')
