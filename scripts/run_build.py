"""Run one closed-loop build in MuJoCo and save the log (npz + HUD json)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, '.')
from spatialfit import trial

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='A'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--seq', default=None, help='default: build/plan_<config>.json')
ap.add_argument('--bias', default='', help='part:dx,dy,dz (mm) injected on first host observation')
ap.add_argument('--stage_sigma', type=float, default=0.002); ap.add_argument('--out', default=None)
ap.add_argument('--nest_err', type=float, default=0.0, help='leg nest position error U(-e, e) mm in x and y')
ap.add_argument('--no_perception', action='store_true', help='ablation: act on nominal poses only')
ap.add_argument('--no_force', action='store_true', help='ablation: position-only insertion, no jam retry')
ap.add_argument('--ba_push', type=float, default=None, help='backrest rack push (m); 0 = original layout')
a = ap.parse_args()
bias = {}
if a.bias:
    k, v = a.bias.split(':'); bias[k] = np.array([float(x) for x in v.split(',')]) / 1000
summary, rec, _ = trial.run(a.config, a.seed, seq=a.seq.split(',') if a.seq else None, bias=bias,
                            stage_sigma=a.stage_sigma, nest_err=a.nest_err / 1000, perceive=not a.no_perception,
                            force_ctrl=not a.no_force, debug=bool(os.environ.get('SF_DEBUG')),
                            ba_push=a.ba_push)
out = a.out or f'build/{a.config}_seed{a.seed}.npz'
rec.save(out)
json.dump(summary, open(out.replace('.npz', '_summary.json'), 'w'), indent=1)
print(json.dumps(summary))
