import sys, json
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene, motion, planner
cfg = sys.argv[1] if len(sys.argv) > 1 else 'A'
wc = scene.Workcell(geometry.generate(params.load(f'configs/{cfg}.yaml')))
sp = planner.SequencePlanner(wc, motion.DLSIK(wc))
out = sp.plan()
json.dump(out, open(f'build/plan_{cfg}.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('self_support', 'final_disp_mm', 'plan_time_s')}))
