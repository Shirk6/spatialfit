"""E1 single-joint coupon: insert the front rail FR into two fixed legs (no robot).

Compares the joint design under a leg-spacing error e (mm, random sign) along the rail axis:
  float      : one locating end + one floating end (+-1.5 mm extra travel), rail aligned to the locating end
  rigid_loc  : both ends tight (FLOAT = 0), rail aligned to the locating end
  rigid_mid  : both ends tight, rail centred between the two measured receivers (best case for a rigid rail)

The rail hangs from a mocap "wrist" through the same compliant weld as the robot grasp (solref 0.012) and is
inserted with the executor's admittance law (F_ref 25 N, F_max 90 N, v_max 20 mm/s), single attempt.
Success = both tongues within 1.2 mm of full depth.

usage: python experiments/e1_coupon.py [--trials 20] [--jobs 8] [--out experiments/results/E1/e1_trials.csv]
"""
import argparse, csv, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatialfit import params, geometry  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / 'configs/A.yaml'
OBS_SIGMA = 0.0003                  # receiver position noise (m), as in the executor
YAW_SIGMA = np.radians(0.05)        # yaw of a rail fitted to two receivers 0.4 m apart
ENGAGE_H = 0.042
V_MAX, F_REF, F_MAX, DT = 0.02, 25.0, 90.0, 0.01


def _add_part(body, s):
    for i, bx in enumerate(s.boxes):
        body.add_geom(name=f'{s.name}:{i}:{bx["role"]}', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[v / 2 for v in bx['size']], pos=bx['center'], quat=bx['quat'], density=700.0,
                      friction=[0.6, 0.01, 0.001], condim=4, solref=[0.004, 1.0], margin=0.0)


def build(design, e, sign):
    geometry.FLOAT = 1.5 if design == 'float' else 0.0
    parts = geometry.generate(params.load(CFG))
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 5
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for n in ('LF', 'RF'):                          # legs fixed (clamped in their nests)
        pos = parts[n].origin.copy()
        if n == 'RF':
            pos[0] += sign * e
        _add_part(spec.worldbody.add_body(name=n, pos=pos.tolist()), parts[n])
    s = parts['FR']
    spec.worldbody.add_body(name='drv', mocap=True, pos=s.origin.tolist())
    rail = spec.worldbody.add_body(name='FR', pos=s.origin.tolist(), gravcomp=1.0)
    rail.add_freejoint(name='FR_free')
    _add_part(rail, s)
    spec.add_equality(name='wrist', type=mujoco.mjtEq.mjEQ_WELD, name1='drv', name2='FR',
                      objtype=mujoco.mjtObj.mjOBJ_BODY, solref=[0.012, 1])
    m = spec.compile()
    m.eq_data[0, :11] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]   # anchor at the rail origin, identity relative pose
    return m, mujoco.MjData(m), s


def rail_force_z(m, d, rid):
    """Upward (resisting) contact force on the rail, N."""
    f6 = np.zeros(6)
    fz = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if rid not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]      # force on geom2, world frame
        fz += fw[2] if b2 == rid else -fw[2]
    return fz


def trial(args):
    design, e_mm, k = args
    rng = np.random.default_rng([int(e_mm * 1000), k])   # same perturbation for every design
    sign = rng.choice([-1.0, 1.0])
    e = e_mm / 1000
    noise_xy = rng.normal(0, OBS_SIGMA, 2)
    noise_yaw = rng.normal(0, YAW_SIGMA)
    m, d, s = build(design, e, sign)
    rid = m.body('FR').id
    qa = m.jnt_qposadr[m.joint('FR_free').id]
    target = s.origin.copy()
    if design == 'rigid_mid':
        target[0] += sign * e / 2
    target[:2] += noise_xy
    quat = R.from_euler('z', noise_yaw).as_quat()[[3, 0, 1, 2]]
    start = target + [0, 0, ENGAGE_H]
    d.mocap_pos[0], d.mocap_quat[0] = start, quat
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = start, quat
    mujoco.mj_forward(m, d)
    tips = np.array([c['pos'] for c in s.connectors])     # connector points in the rail frame
    z_seat = s.origin[2]

    def depth_err():
        r = R.from_quat(d.qpos[qa + 3:qa + 7][[1, 2, 3, 0]])
        return (d.qpos[qa:qa + 3] + r.apply(tips))[:, 2] - z_seat

    for _ in range(100):                                  # settle on the mocap
        mujoco.mj_step(m, d)
    z_cmd, z_end = start[2], target[2] - 0.004
    peak, hist, result, t_seat = 0.0, [], 'TIMEOUT', None
    lat_max = 0.0
    n_ctrl = int(round(DT / m.opt.timestep))
    while d.time < 8.0:
        F = rail_force_z(m, d, rid)
        peak = max(peak, F)
        v = np.clip(V_MAX * (F_REF - F) / F_REF, -V_MAX, V_MAX)
        z_cmd = max(z_cmd - v * DT, z_end)
        d.mocap_pos[0][2] = z_cmd
        for _ in range(n_ctrl):
            mujoco.mj_step(m, d)
        lat_max = max(lat_max, np.linalg.norm(d.qpos[qa:qa + 2] - d.mocap_pos[0][:2]))
        de = depth_err()
        hist.append(de.max())
        if de.max() < 0.0012 and abs(F) < F_MAX:
            for _ in range(150):
                mujoco.mj_step(m, d)
            if depth_err().max() < 0.0012:
                result, t_seat = 'SEATED', d.time
                break
        if F > F_MAX:
            result = 'JAM_FORCE'
            break
        if len(hist) > 80 and hist[-80] - hist[-1] < 0.0003 and de.max() > 0.002:
            result = 'JAM_NO_PROGRESS'
            break
    return dict(design=design, e_mm=e_mm, trial=k, sign=int(sign), success=int(result == 'SEATED'), result=result,
                peak_force_N=round(peak, 2), t_seat_s=None if t_seat is None else round(t_seat, 3),
                max_depth_err_mm=round(float(depth_err().max()) * 1000, 2), lateral_comp_mm=round(lat_max * 1000, 2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--errors', default='0,0.5,1,1.5,2,2.5,3,3.5,4')
    ap.add_argument('--designs', default='float,rigid_loc,rigid_mid')
    ap.add_argument('--out', default='experiments/results/E1/e1_trials.csv')
    a = ap.parse_args()
    jobs = [(dz, float(e), k) for dz in a.designs.split(',') for e in a.errors.split(',') for k in range(a.trials)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.jobs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(trial, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(jobs)}', flush=True)
    rows.sort(key=lambda r: (r['design'], r['e_mm'], r['trial']))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for dz in a.designs.split(','):
        print(dz, ' '.join(f"{e}:{np.mean([r['success'] for r in rows if r['design'] == dz and r['e_mm'] == float(e)]):.2f}"
                           for e in a.errors.split(',')))
