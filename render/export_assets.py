"""Export the MuJoCo workcell's visual geometry for Blender.

Writes <out>/scene.json (bodies -> geoms with local transforms, colours, roles) and <out>/meshes/*.obj
(robot meshes taken from the compiled model). Box geoms are described analytically and built in Blender.

usage: python render/export_assets.py CONFIG OUTDIR
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatialfit import params, geometry, scene

cfg, out = sys.argv[1], Path(sys.argv[2])
(out / "meshes").mkdir(parents=True, exist_ok=True)
wc = scene.Workcell(geometry.generate(params.load(f"configs/{cfg}.yaml")))
m = wc.model
part_body = {wc.body[n]: n for n in wc.names}


def write_obj(mid, path):
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    V = m.mesh_vert[va:va + vn]
    F = m.mesh_face[fa:fa + fn]
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


bodies = []
written = set()
for b in range(1, m.nbody):
    geoms = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        gtype = int(m.geom_type[g])
        group = m.geom_group[g]
        is_robot = b in wc.robot_bodies
        if is_robot and gtype not in (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)):
            continue                                  # collision capsules/cylinders of the UR model
        if is_robot and gtype == int(mujoco.mjtGeom.mjGEOM_MESH) and group != 2:
            continue
        rgba = m.geom_rgba[g].tolist()
        if m.geom_matid[g] >= 0:
            rgba = m.mat_rgba[m.geom_matid[g]].tolist()
        name = m.geom(g).name
        entry = dict(name=name, pos=m.geom_pos[g].tolist(), quat=m.geom_quat[g].tolist(), rgba=rgba,
                     role=name.split(":")[-1] if ":" in name else ("robot" if is_robot else "fixture"))
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = m.geom_dataid[g]
            fn = f"meshes/mesh_{mid}.obj"
            if mid not in written:
                write_obj(mid, out / fn); written.add(mid)
            entry.update(type="mesh", file=fn)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            entry.update(type="box", size=(2 * m.geom_size[g]).tolist())
        else:
            continue
        geoms.append(entry)
    if geoms:
        bodies.append(dict(id=b, name=m.body(b).name, part=part_body.get(b), robot=b in wc.robot_bodies, geoms=geoms))

# static world geoms (body 0): pedestal etc. (floor is created in Blender)
world = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0 and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
        world.append(dict(name=m.geom(g).name, pos=m.geom_pos[g].tolist(), quat=m.geom_quat[g].tolist(),
                          size=(2 * m.geom_size[g]).tolist(), rgba=m.geom_rgba[g].tolist(), type="box", role="fixture"))
json.dump(dict(config=cfg, bodies=bodies, world=world, chair_origin=wc.chair_origin.tolist()),
          open(out / "scene.json", "w"), indent=1)
print("bodies", len(bodies), "meshes", len(written), "world boxes", len(world))
