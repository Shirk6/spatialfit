"""One build trial in MuJoCo: perturbed workcell -> closed-loop (or ablated) executor -> summary dict.

Shared by scripts/run_build.py (single run) and experiments/run_trials.py (batches that reuse one cuRobo instance).
"""
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from . import params, geometry, scene, motion
from .executor import Executor, Recorder

ROOT = Path(__file__).resolve().parents[1]


def setup(config, seed, stage_sigma=0.002, nest_err=0.0, stage_yaw_deg=1.0, ba_push=None):
    """Workcell for `config` with seeded perturbations. Returns (wc, rng, nominal staging poses, nest offsets)."""
    rng = np.random.default_rng(seed)
    parts = geometry.generate(params.load(ROOT / f'configs/{config}.yaml'))
    staging = None
    if ba_push is not None:               # backrest rack distance override (default scene.BA_PUSH)
        staging = dict(scene.STAGING); staging['BA'] = ('orbit', np.radians(180), ba_push)
    wc = scene.Workcell(parts, staging=staging)
    wc.reset()
    nominal = {n: (wc.staging[n][0].copy(), np.array(wc.staging[n][1], float)) for n in wc.names}
    # staging perturbation (kitting errors) for parts lying on the floor; racked panels sit in their racks
    for n in wc.names:
        if n in wc.rack_parts:
            continue
        pos, q = wc.staging[n]
        pos = pos + np.r_[rng.normal(0, stage_sigma, 2), 0]
        q = scene.quat_wxyz(R.from_euler('z', rng.normal(0, np.radians(stage_yaw_deg))) * scene.rot_of(q))
        wc.staging[n] = (pos, q)
        wc.set_part_pose(n, pos, q)
    offsets = {}
    if nest_err > 0:          # fixture error: each leg nest off by U(-e, e) in x and y (separate stream: same staging)
        r2 = np.random.default_rng([seed, 1])
        offsets = {n: np.r_[r2.uniform(-nest_err, nest_err, 2), 0.0] for n in wc.nest_mocap}
        wc.set_nest_offsets(offsets)
    mujoco.mj_forward(wc.model, wc.data)
    # let the staged parts settle on the floor under gravity (robot holds home)
    for _ in range(1500):
        wc.data.ctrl[wc.arm_act] = scene.HOME_Q
        mujoco.mj_step(wc.model, wc.data)
    wc.data.time = 0.0
    return wc, rng, nominal, offsets


def run(config, seed, cp=None, seq=None, bias=None, stage_sigma=0.002, nest_err=0.0, perceive=True,
        force_ctrl=True, debug=False, stage_yaw_deg=1.0, ba_push=None):
    """Run one build. Pass an existing CuroboPlanner as `cp` to skip its ~18 s warm-up (it is re-pointed at the new
    workcell). Returns (summary, recorder, planner)."""
    wc, rng, nominal, offsets = setup(config, seed, stage_sigma, nest_err, stage_yaw_deg, ba_push)
    ik = motion.DLSIK(wc)
    if cp is None:
        t0 = time.time(); cp = motion.CuroboPlanner(wc); print('cuRobo ready', round(time.time() - t0, 1), 's', flush=True)
    else:
        cp.wc = wc
        cp.detach()
    rec = Recorder(wc)
    ex = Executor(wc, ik, cp, rec, rng, bias=bias, perceive=perceive, force_ctrl=force_ctrl, nominal=nominal)
    ex.debug = debug
    if seq is None:
        import json
        seq = json.load(open(ROOT / f'build/plan_{config}.json'))['sequence']
    t0 = time.time()
    ok = ex.build(seq)
    disp = None
    if ok and len(seq) == 10:
        ex.home(); disp = ex.final_release()
    fail = next((s for s in ex.log if not s['ok']), None)
    summary = dict(config=config, seed=seed, success=bool(ok), steps=ex.log, sim_time=round(float(wc.data.time), 2),
                   wall_time=round(time.time() - t0, 1), final_disp_mm=None if disp is None else round(disp * 1000, 2),
                   perceive=perceive, force_ctrl=force_ctrl, stage_sigma_mm=stage_sigma * 1000, stage_yaw_deg=stage_yaw_deg,
                   nest_err_mm=nest_err * 1000, ba_push_m=scene.BA_PUSH if ba_push is None else ba_push,
                   nest_offsets_mm={n: np.round(o[:2] * 1000, 3).tolist() for n, o in offsets.items()},
                   fail_part=None if fail is None else fail['part'], fail_cause=None if fail is None else fail['cause'],
                   n_parts_ok=sum(s['ok'] for s in ex.log),
                   jams=[e for e in rec.events if e['kind'] == 'jam'],
                   fallbacks=[e for e in rec.events if e['kind'] in ('fallback_path', 'plan_fail')])
    return summary, rec, cp
