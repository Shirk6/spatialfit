"""Render a recorded MuJoCo build log in Blender (Cycles, GPU).

usage:
  blender -b --factory-startup -P render/blender_render.py -- SCENE_DIR LOG.npz OUT_DIR
          [--speed 4] [--slow 1] [--res 1280x720] [--samples 48] [--cam wide|close|hero]
          [--frames i,j,k (log indices, stills)] [--max N]
Writes OUT_DIR/frame_XXXXX.png and OUT_DIR/frames.json (log index of every rendered frame).
"""
import bpy, sys, json, math, time
from pathlib import Path
import numpy as np
from mathutils import Vector, Quaternion

argv = sys.argv[sys.argv.index("--") + 1:]
scene_dir, log_path, out_dir = Path(argv[0]), Path(argv[1]), Path(argv[2])
opts = {argv[i][2:]: argv[i + 1] for i in range(3, len(argv) - 1, 2) if argv[i].startswith("--")}
speed = int(opts.get("speed", 4)); slow = int(opts.get("slow", 1))
rx, ry = (int(v) for v in opts.get("res", "1280x720").split("x"))
samples = int(opts.get("samples", 32)); cam_name = opts.get("cam", "wide")
out_dir.mkdir(parents=True, exist_ok=True)

S = json.load(open(scene_dir / "scene.json"))
L = np.load(log_path)
hud = json.load(open(str(log_path).replace(".npz", "_hud.json")))["hud"]
xpos, xquat = L["xpos"], L["xquat"]

# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for _t in ("OPTIX", "CUDA"):                                   # OptiX on the 5090 worker; CUDA under WSL2
    prefs.compute_device_type = _t; prefs.get_devices()
    if any(d.type == _t for d in prefs.devices):
        break
print("[render] device type", prefs.compute_device_type, [(d.name, d.type) for d in prefs.devices], flush=True)
for d in prefs.devices:
    d.use = d.type != "CPU"
sc.cycles.device = "GPU"
sc.cycles.samples = samples
sc.cycles.use_denoising = True
if hasattr(sc.cycles, "denoising_use_gpu"):
    sc.cycles.denoising_use_gpu = True
sc.cycles.max_bounces = 6
sc.render.use_persistent_data = True
sc.render.resolution_x, sc.render.resolution_y = rx, ry
sc.render.film_transparent = False
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Punchy"
sc.view_settings.exposure = -0.35

# ---------------------------------------------------------------- materials
_mats = {}


def mat(key, base, metal=0.0, rough=0.45, grain=False, coat=0.0):
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(key); m.use_nodes = True
    nt = m.node_tree; bs = nt.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*base, 1)
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Roughness"].default_value = rough
    if "Coat Weight" in bs.inputs:
        bs.inputs["Coat Weight"].default_value = coat
    if grain:     # subtle wood grain: stretched noise -> colour ramp
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (2.0, 40.0, 2.0)
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 6.0; nz.inputs["Detail"].default_value = 8.0
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (base[0] * 0.78, base[1] * 0.74, base[2] * 0.70, 1)
        cr.color_ramp.elements[1].color = (min(1, base[0] * 1.12), min(1, base[1] * 1.10), min(1, base[2] * 1.05), 1)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        nt.links.new(nz.outputs["Fac"], cr.inputs["Fac"]); nt.links.new(cr.outputs["Color"], bs.inputs["Base Color"])
    _mats[key] = m
    return m


def material_for(body, g):
    role, rgba = g["role"], g["rgba"]
    if body is not None and body.get("robot"):
        if g["type"] == "box":
            return mat("gripper", (0.035, 0.037, 0.042), 0.4, 0.38)
        c = tuple(rgba[:3])
        return mat(f"ur_{c}", c, 0.15 if sum(c) > 1.5 else 0.25, 0.32 if sum(c) > 1.5 else 0.45, coat=0.2)
    if role in ("panel", "foot", "cleat", "hinge"):
        return mat("wood", (0.55, 0.31, 0.13), 0.0, 0.45, grain=True, coat=0.15)
    if role in ("tongue", "post"):
        return mat("steel", (0.62, 0.64, 0.66), 0.85, 0.28)
    if role in ("head", "leg", "rail", "body"):
        return mat("frame", (0.015, 0.11, 0.13), 0.4, 0.34, coat=0.3)
    name = g["name"]
    if name.startswith("nest_"):
        return mat("nest", (0.90, 0.36, 0.04), 0.0, 0.45)
    if name.startswith("rack_"):
        return mat("rack", (0.16, 0.17, 0.19), 0.5, 0.45)
    if name == "pedestal":
        return mat("pedestal", (0.12, 0.13, 0.15), 0.3, 0.5)
    return mat(f"misc_{tuple(rgba[:3])}", tuple(rgba[:3]), 0.0, 0.5)


# ---------------------------------------------------------------- geometry
_cube = None


def box_obj(name, size, material, parent=None, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    global _cube
    if _cube is None:
        bpy.ops.mesh.primitive_cube_add(size=1.0); _cube = bpy.context.object.data
        bpy.data.objects.remove(bpy.context.object)
    me = _cube.copy()
    o = bpy.data.objects.new(name, me); sc.collection.objects.link(o)
    o.scale = size
    bev = min(size) * 0.18
    if bev > 0.0004:
        md = o.modifiers.new("bevel", "BEVEL"); md.width = min(bev, 0.003); md.segments = 2
        md.affect = "EDGES"; md.use_clamp_overlap = True
        o.modifiers.new("wn", "WEIGHTED_NORMAL")
    o.data.materials.append(material)
    o.parent = parent; o.location = pos
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = quat
    return o


_mesh_cache = {}


def mesh_obj(name, file, material, parent, pos, quat):
    if file not in _mesh_cache:
        V, F = [], []
        for line in open(scene_dir / file):
            if line.startswith("v "):
                V.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                F.append(tuple(int(x) - 1 for x in line.split()[1:4]))
        me = bpy.data.meshes.new(file); me.from_pydata(V, [], F); me.update()
        for p in me.polygons:
            p.use_smooth = True
        _mesh_cache[file] = me
    o = bpy.data.objects.new(name, _mesh_cache[file]); sc.collection.objects.link(o)
    if not o.data.materials:
        o.data.materials.append(material)
    o.material_slots[0].link = "OBJECT"; o.material_slots[0].material = material
    o.parent = parent; o.location = pos
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = quat
    return o


empties = {}
for b in S["bodies"]:
    e = bpy.data.objects.new(f"B_{b['name']}", None); sc.collection.objects.link(e)
    e.rotation_mode = "QUATERNION"; empties[b["id"]] = e
    for g in b["geoms"]:
        mt = material_for(b, g)
        if g["type"] == "box":
            box_obj(g["name"] or b["name"], g["size"], mt, e, g["pos"], g["quat"])
        else:
            mesh_obj(g["name"] or b["name"], g["file"], mt, e, g["pos"], g["quat"])
for g in S["world"]:
    box_obj(g["name"], g["size"], material_for(None, g), None, g["pos"], g["quat"])

# floor + studio backdrop
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0)); floor = bpy.context.object
floor.data.materials.append(mat("floor", (0.58, 0.59, 0.61), 0.0, 0.6))
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes["Background"]; bg.inputs["Color"].default_value = (0.74, 0.76, 0.79, 1); bg.inputs["Strength"].default_value = 0.6

# lights
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 2.8; sun.data.angle = math.radians(8)
sun.rotation_euler = (math.radians(38), math.radians(12), math.radians(-35))
key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA")); sc.collection.objects.link(key)
key.data.energy = 650; key.data.size = 2.5; key.location = (1.8, -1.6, 2.6)
key.rotation_euler = (math.radians(45), 0, math.radians(48))
fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "AREA")); sc.collection.objects.link(fill)
fill.data.energy = 220; fill.data.size = 3.0; fill.location = (-2.2, 1.5, 2.0); fill.rotation_euler = (math.radians(55), 0, math.radians(-125))

# camera (MuJoCo-style lookat / distance / azimuth / elevation)
CAMS = {"wide": ((-0.02, 0.10, 0.42), 2.75, -128, -27), "close": ((0.0, 0.42, 0.38), 1.15, -140, -28),
        "hero": ((0.0, 0.40, 0.45), 1.6, -150, -18), "front": ((0.0, 0.25, 0.35), 2.3, -90, -26)}
if cam_name.startswith("custom:"):
    v = [float(x) for x in cam_name[7:].split(",")]
    CAMS["custom"] = ((v[0], v[1], v[2]), v[3], v[4], v[5]); cam_name = "custom"
look, dist, az, el = CAMS[cam_name]
az, el = math.radians(az), math.radians(el)
fwd = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
cam.location = Vector(look) - dist * fwd
cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = (-fwd).to_track_quat("Z", "Y")
cam.data.lens = 35; cam.data.dof.use_dof = False
sc.camera = cam

# ---------------------------------------------------------------- frames
SLOW = ("insert", "jam", "re-observe", "latch", "grasp", "fixture release", "stability")
if "frames" in opts:
    idx = [int(v) for v in opts["frames"].split(",")]
else:
    idx, i = [], 0
    while i < len(xpos):
        idx.append(i)
        ph = hud[i]["phase"]
        i += slow if any(k in ph for k in SLOW) else speed
    idx += [len(xpos) - 1] * 45                      # hold the final frame 1.5 s
if "max" in opts:
    idx = idx[: int(opts["max"])]
json.dump(dict(indices=idx, speed=speed, slow=slow), open(out_dir / "frames.json", "w"))

t0 = time.time()
for k, i in enumerate(idx):
    for bid, e in empties.items():
        e.location = xpos[i, bid].tolist()
        e.rotation_quaternion = xquat[i, bid].tolist()
    sc.render.filepath = str(out_dir / f"frame_{k:05d}.png")
    bpy.ops.render.render(write_still=True)
    if k % 25 == 0:
        el_ = time.time() - t0
        print(f"[render] {k + 1}/{len(idx)}  {el_ / (k + 1):.2f} s/frame  eta {el_ / (k + 1) * (len(idx) - k - 1) / 60:.1f} min", flush=True)
print("done", len(idx), "frames in", round(time.time() - t0, 1), "s")
