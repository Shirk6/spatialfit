"""Batch build trials (experiments E2 and E3) in one process, reusing a single cuRobo instance.

Conditions (E3 ablation; CL is also E2):
  CL   closed loop       : perception (host re-observation + measured in-hand pose) + admittance insertion with retry
  NOF  no force feedback : perception kept, position-only insertion to the nominal depth, no jam detection / retry
  NOP  no perception     : nominal poses only (no re-observation), admittance insertion + retry kept
  OL   open loop         : neither

Every trial of a given seed sees the same perturbations (staging error, leg-nest error); the run is resumable
(finished trials are skipped).

usage: python experiments/run_trials.py --plan "CL:A,B;NOF:A;NOP:A;OL:A" --seeds 0-19 --nest_err 1.0
"""
import argparse, gc, json, signal, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatialfit import trial  # noqa: E402

CONDS = dict(CL=(True, True), NOF=(True, False), NOP=(False, True), OL=(False, False))


class WallTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise WallTimeout()


def seeds_of(s):
    out = []
    for part in s.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='CL:A,B;NOF:A;NOP:A;OL:A')
    ap.add_argument('--seeds', default='0-19')
    ap.add_argument('--nest_err', type=float, default=1.0, help='mm')
    ap.add_argument('--stage_sigma', type=float, default=2.0, help='mm')
    ap.add_argument('--stage_yaw', type=float, default=1.0, help='deg (sigma of the staging yaw error)')
    ap.add_argument('--ba_push', type=float, default=None, help='backrest rack push away from the base (m); default scene.BA_PUSH')
    ap.add_argument('--out', default='experiments/results/E23')
    ap.add_argument('--timeout', type=int, default=900, help='wall-clock seconds per trial')
    ap.add_argument('--shard', default='0/1', help='k/N: run only jobs i with i %% N == k (parallel workers)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan = [(c, cfg) for item in a.plan.split(';') for c, cfgs in [item.split(':')] for cfg in cfgs.split(',')]
    jobs = [(c, cfg, s) for s in seeds_of(a.seeds) for c, cfg in plan]
    k, n_shards = map(int, a.shard.split('/'))
    jobs = jobs[k::n_shards]
    signal.signal(signal.SIGALRM, _alarm)
    cp = None
    t_start = time.time()
    for i, (cond, cfg, seed) in enumerate(jobs):
        stem = out / f'{cond}_{cfg}_seed{seed:02d}'
        if Path(f'{stem}_summary.json').exists():
            continue
        perceive, force = CONDS[cond]
        print(f'=== [{i + 1}/{len(jobs)}] {cond} {cfg} seed {seed}  (elapsed {time.time() - t_start:.0f} s)', flush=True)
        signal.alarm(a.timeout)
        try:
            summary, rec, cp = trial.run(cfg, seed, cp=cp, stage_sigma=a.stage_sigma / 1000, nest_err=a.nest_err / 1000,
                                         perceive=perceive, force_ctrl=force, stage_yaw_deg=a.stage_yaw,
                                         ba_push=a.ba_push)
            signal.alarm(0)
            summary['cond'] = cond
            with open(f'{stem}_hud.json', 'w') as f:
                json.dump(dict(hud=rec.hud, events=rec.events), f)
        except Exception as exc:          # record the failure and start the next trial with a fresh planner
            signal.alarm(0)
            summary = dict(config=cfg, seed=seed, cond=cond, success=False, steps=[], error=repr(exc),
                           fail_cause='WALL_TIMEOUT' if isinstance(exc, WallTimeout) else 'EXCEPTION',
                           traceback=traceback.format_exc()[-2000:])
            cp = None
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        json.dump(summary, open(f'{stem}_summary.json', 'w'), indent=1)
        print(f'--> {cond} {cfg} seed {seed}: success={summary["success"]} fail={summary.get("fail_part")}/'
              f'{summary.get("fail_cause")} parts_ok={summary.get("n_parts_ok")} disp={summary.get("final_disp_mm")} '
              f'wall={summary.get("wall_time")}', flush=True)
    print('all done', round(time.time() - t_start), 's')
