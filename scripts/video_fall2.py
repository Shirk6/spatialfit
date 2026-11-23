"""Stability comparison shot: release the fixture after step 6 (side frames not yet connected) vs after step 7
(FR closes the frame) and apply the same 3.6 N disturbance to SL. Records at 120 Hz for smooth slow motion and
logs what the overlay needs: push vector + application point, centre of mass, support polygon, displacement.

usage: python scripts/video_fall2.py   -> build/fall6.npz, build/fall7.npz (+ _hud.json), build/video/fall{6,7}_annot.json
"""
import sys, json
import numpy as np
import mujoco
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene
from spatialfit.executor import Recorder

PUSH = np.array([-3.0, -2.0, 0.0])        # N, world; |F| = 3.6 N, same magnitude as the planner's self-support check
PUSH_T = 0.3                               # s
HOLD_T, AFTER_T = 1.5, 3.2
FPS = 120
FOOT = 0.016                               # half size of the 32 mm foot pad


def run_case(n_inst, cfg='A'):
    seq = json.load(open(f'build/plan_{cfg}.json'))['sequence']
    installed = seq[:n_inst]
    wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')))
    m, d = wc.model, wc.data
    wc.reset()
    for _ in range(1500):
        d.ctrl[wc.arm_act] = scene.HOME_Q; mujoco.mj_step(m, d)
    for n in installed:
        pos, q = wc.target_pose(n); wc.set_part_pose(n, pos, q)
    mujoco.mj_forward(m, d)
    for n in installed:
        for c in wc.parts[n].connectors:
            if c["host"] in installed:
                wc.set_weld(f"latch_{c['name']}", True)
        if wc.parts[n].kind == "leg":
            wc.set_weld(f"clamp_{n}", True, body1_pose=wc.target_pose(n))
    wc.set_nests(True)
    d.time = 0.0
    rec = Recorder(wc, fps=FPS)
    rec.set(step=n_inst, total=10, part="", phase="fixture holding")
    # the group whose stability we visualise: the pushed side frame (step 6) or the whole closed frame (step 7)
    group = ["LF", "LR", "SL"] if n_inst < 7 else installed
    legs = [n for n in group if wc.parts[n].kind == "leg"]
    annot = []
    p0 = None

    def com():
        ms = np.array([m.body_mass[wc.body[n]] for n in group])
        return (ms[:, None] * np.array([d.xipos[wc.body[n]] for n in group])).sum(0) / ms.sum()

    def tick(push_on):
        before = len(rec.xpos)
        rec.tick()
        if len(rec.xpos) > before:
            disp = 0.0 if p0 is None else max(np.linalg.norm(d.xpos[wc.body[n]] - p0[n]) for n in installed)
            annot.append(dict(t=round(float(d.time), 4), phase=rec.state["phase"], push=bool(push_on),
                              app=d.xipos[wc.body["SL"]].tolist(), com=com().tolist(), disp_mm=disp * 1000))

    def run(sec, phase, push=False):
        rec.set(phase=phase)
        t0 = d.time
        for _ in range(int(round(sec / m.opt.timestep))):
            d.ctrl[wc.arm_act] = scene.HOME_Q
            on = push and d.time - t0 < PUSH_T
            d.xfrc_applied[wc.body["SL"], :3] = PUSH if on else 0.0
            mujoco.mj_step(m, d)
            tick(on)
        d.xfrc_applied[:] = 0

    run(HOLD_T, "fixture holding")
    p0 = {n: d.xpos[wc.body[n]].copy() for n in installed}
    # support polygon of the group at release (feet are 32 mm pads on the floor)
    feet = np.array([d.xpos[wc.body[n]][:2] for n in legs])
    lo, hi = feet.min(0) - FOOT, feet.max(0) + FOOT
    support = [[lo[0], lo[1], 0.001], [hi[0], lo[1], 0.001], [hi[0], hi[1], 0.001], [lo[0], hi[1], 0.001]]
    other = []
    if n_inst < 7:                           # the second (unpushed) side frame, drawn faintly
        f2 = np.array([d.xpos[wc.body[n]][:2] for n in ("RF", "RR")])
        l2, h2 = f2.min(0) - FOOT, f2.max(0) + FOOT
        other = [[l2[0], l2[1], 0.001], [h2[0], l2[1], 0.001], [h2[0], h2[1], 0.001], [l2[0], h2[1], 0.001]]
    rec.event("fixture_release", installed=installed)
    for n in installed:
        if wc.parts[n].kind == "leg":
            wc.set_weld(f"clamp_{n}", False)
    wc.set_nests(False)
    run(AFTER_T, "fixture release", push=True)
    disp = max(np.linalg.norm(d.xpos[wc.body[n]] - p0[n]) for n in installed)
    rec.event("result", max_disp_mm=round(disp * 1000, 1))
    rec.save(f'build/fall{n_inst}.npz')
    json.dump(dict(n_inst=n_inst, installed=installed, group=group, fps=FPS, push_N=PUSH.tolist(), push_t=PUSH_T,
                   hold_t=HOLD_T, support=support, support_other=other, max_disp_mm=disp * 1000,
                   support_width_mm=float((hi - lo).min() * 1000), frames=annot),
              open(f'build/video/fall{n_inst}_annot.json', 'w'))
    print(json.dumps(dict(n_inst=n_inst, frames=len(annot), max_disp_mm=round(disp * 1000, 1),
                          support_mm=((hi - lo) * 1000).round(0).tolist())))


for k in (6, 7):
    run_case(k)
