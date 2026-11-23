"""Compile workcell, report contacts at reset (staging) and in the nominal assembled state (compatibility check)."""
import sys, mujoco, numpy as np
sys.path.insert(0, '.')
from spatialfit import params, geometry, scene
cfg = sys.argv[1] if len(sys.argv) > 1 else 'A'
p = params.load(f'configs/{cfg}.yaml'); parts = geometry.generate(p)
wc = scene.Workcell(parts)
m, d = wc.model, wc.data
print('nbody', m.nbody, 'ngeom', m.ngeom, 'neq', m.neq, 'nu', m.nu)
def report(tag):
    mujoco.mj_forward(m, d)
    bad = [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist*1000, 2)) for c in d.contact[:d.ncon] if c.dist < -1e-4]
    print(tag, 'contacts', d.ncon, 'penetrating', len(bad)); [print('   ', b) for b in bad[:15]]
report('staging:')
for n in wc.names:
    pos, q = wc.target_pose(n); wc.set_part_pose(n, pos, q)
report('assembled:')
# clearance: min distance between mating geoms
for n in wc.names:
    for c in parts[n].connectors:
        if c['host'] in parts:
            dmin = 1e9
            for g1 in wc.geoms_of[n]:
                for g2 in wc.geoms_of[c['host']]:
                    dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.01, None)
                    dmin = min(dmin, dist)
            print(f'  {c["name"]:8s} {c["type"]:9s} min gap {dmin*1000:6.2f} mm')
