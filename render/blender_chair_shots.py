"""Chair-only studio shots (parametric morph / exploded view / ending), same look as blender_render2.py.

usage:
  blender -b --factory-startup -P render/blender_chair_shots.py -- SHOT.json OUT_DIR/ [--samples 24] [--res 1920x1080]
          [--range a:b]
writes OUT_DIR/frame_####.png and OUT_DIR/proj.json (per-frame 2D screen positions of part / group centres)
"""
import bpy, sys, json, math
from pathlib import Path
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:]
shot, out = Path(argv[0]), Path(argv[1])
opts = {argv[i][2:]: argv[i + 1] for i in range(2, len(argv) - 1, 2) if argv[i].startswith("--")}
samples = int(opts.get("samples", 24))
rx, ry = (int(v) for v in opts.get("res", "1920x1080").split("x"))
J = json.load(open(shot))
n_all = J["n"]
a_, b_ = (int(v) for v in opts.get("range", f"0:{n_all}").split(":"))
frames = list(range(a_, min(b_, n_all)))
n = len(frames)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for _t in ("OPTIX", "CUDA"):
    prefs.compute_device_type = _t; prefs.get_devices()
    if any(d.type == _t for d in prefs.devices):
        break
for d in prefs.devices:
    d.use = d.type != "CPU"
sc.cycles.device = "GPU"; sc.cycles.samples = samples; sc.cycles.use_adaptive_sampling = True
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 4; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 0
sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
sc.render.use_persistent_data = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = rx, ry, 100
sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Punchy"; sc.view_settings.exposure = -0.2
sc.render.fps = J["fps"]

_m = {}


def mat(key, base, metal=0.0, rough=0.45, coat=0.0, grain=None, spec=0.5, aniso=0.0):
    if key in _m:
        return _m[key]
    m = bpy.data.materials.new(key); m.use_nodes = True
    nt = m.node_tree; bs = nt.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*base, 1)
    bs.inputs["Metallic"].default_value = metal; bs.inputs["Roughness"].default_value = rough
    for name, val in (("Coat Weight", coat), ("Coat Roughness", 0.08), ("Specular IOR Level", spec), ("Anisotropic", aniso)):
        if name in bs.inputs:
            bs.inputs[name].default_value = val
    if grain:
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = grain
        wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.inputs["Scale"].default_value = 3.0
        wv.inputs["Distortion"].default_value = 2.0; wv.inputs["Detail"].default_value = 2.0
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (base[0] * 0.72, base[1] * 0.66, base[2] * 0.60, 1)
        cr.color_ramp.elements[1].color = (min(1, base[0] * 1.10), min(1, base[1] * 1.08), min(1, base[2] * 1.04), 1)
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], wv.inputs["Vector"])
        nt.links.new(wv.outputs["Fac"], cr.inputs["Fac"]); nt.links.new(cr.outputs["Color"], bs.inputs["Base Color"])
    _m[key] = m
    return m


M = dict(
    oak=mat("oak", (0.58, 0.36, 0.18), 0.0, 0.42, coat=0.35, grain=(3.0, 60.0, 3.0)),
    frame=mat("frame", (0.02, 0.075, 0.09), 0.8, 0.3, coat=0.5),
    steel=mat("steel", (0.80, 0.81, 0.83), 1.0, 0.22, aniso=0.4),
    stage=mat("stage", (0.09, 0.093, 0.10), 0.0, 0.6, spec=0.3),
    floor=mat("floor", (0.035, 0.037, 0.042), 0.0, 0.22, spec=0.6),
)
ROLE = dict(panel="oak", foot="oak", cleat="oak", hinge="oak", tongue="steel", post="steel")

bpy.ops.mesh.primitive_cube_add(size=1.0); CUBE = bpy.context.object.data; bpy.data.objects.remove(bpy.context.object)


def box(name, size, material, pos=(0, 0, 0), quat=(1, 0, 0, 0), bevel=0.002):
    o = bpy.data.objects.new(name, CUBE.copy()); sc.collection.objects.link(o)
    o.scale = size
    md = o.modifiers.new("bevel", "BEVEL"); md.width = bevel; md.segments = 2; md.use_clamp_overlap = True
    o.modifiers.new("wn", "WEIGHTED_NORMAL")
    o.data.materials.append(material)
    o.location = pos; o.rotation_mode = "QUATERNION"; o.rotation_quaternion = quat
    return o


def keyframe(obj, path, values):
    if obj.animation_data is None:
        obj.animation_data_create()
    act = obj.animation_data.action or bpy.data.actions.new(obj.name + "_act")
    obj.animation_data.action = act
    fr = np.arange(1, n + 1, dtype=np.float32)
    for k in range(values.shape[1]):
        fc = act.fcurves.find(path, index=k) or act.fcurves.new(path, index=k)
        fc.keyframe_points.add(n)
        co = np.empty(2 * n, dtype=np.float32); co[0::2] = fr; co[1::2] = values[:, k]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", np.zeros(n, dtype=np.int32))
        fc.update()


fi = np.array(frames)
for key, s in J["series"].items():
    role = J["meta"][key]["role"]
    loc, size, quat = (np.asarray(s[a], dtype=np.float32)[fi] for a in ("loc", "size", "quat"))
    o = box(key, tuple(size[0]), M[ROLE.get(role, "frame")], tuple(loc[0]), tuple(quat[0]))
    if max(np.ptp(loc, axis=0).max(), np.ptp(size, axis=0).max(), np.ptp(quat, axis=0).max()) > 1e-7:
        # key all three: in Blender 4.5 an object keyed on a single path only never binds its action slot
        keyframe(o, "location", loc)
        keyframe(o, "scale", size)
        keyframe(o, "rotation_quaternion", quat)

# studio: round stage + glossy floor + gradient world + 4 area lights (same recipe as blender_render2.py)
bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=1.25, depth=0.02, location=(0, 0, -0.0101))
st = bpy.context.object; st.data.materials.append(M["stage"])
md = st.modifiers.new("bevel", "BEVEL"); md.width = 0.008; md.segments = 3
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.021)); bpy.context.object.data.materials.append(M["floor"])
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
nt = world.node_tree; bg = nt.nodes["Background"]
grad = nt.nodes.new("ShaderNodeTexGradient"); grad.gradient_type = "SPHERICAL"
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.02, 0.022, 0.026, 1); ramp.color_ramp.elements[1].color = (0.22, 0.23, 0.26, 1)
tc = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(tc.outputs["Generated"], grad.inputs["Vector"]); nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.35


def area(name, energy, size, loc, target, color=(1, 1, 1)):
    o = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA")); sc.collection.objects.link(o)
    o.data.energy = energy; o.data.size = size; o.data.color = color; o.location = loc
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y")


area("key", 420, 2.0, (-1.6, -1.9, 2.5), (0.0, 0.0, 0.4), (1.0, 0.94, 0.86))
area("fill", 110, 3.0, (2.2, -1.0, 1.6), (0.0, 0.0, 0.4), (0.85, 0.92, 1.0))
area("rim", 420, 1.6, (0.4, 2.3, 2.0), (0.0, 0.0, 0.6), (0.80, 0.90, 1.0))
area("top", 110, 3.0, (0.0, 0.0, 3.0), (0.0, 0.0, 0.0))

# camera: slow orbit; look/dist/el/az0 (deg), az_rate (deg/s)
C = J["cam"]
look, dist, el = Vector(C["look"]), C["dist"], math.radians(C["el"])
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
cam.data.lens = C["lens"]; cam.data.dof.use_dof = True; cam.data.dof.aperture_fstop = 4.0
tgt = bpy.data.objects.new("cam_target", None); sc.collection.objects.link(tgt); tgt.location = look
cam.data.dof.focus_object = tgt
cam.rotation_mode = "QUATERNION"
locs, rots = [], []
for f in frames:
    az = math.radians(C["az0"] + C["az_rate"] * f / J["fps"])
    v = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    cf = J["info"][f].get("camf", {})
    lk = Vector((look.x, look.y, cf.get("lookz", look.z))); ds = cf.get("dist", dist)
    locs.append(tuple(lk - ds * v)); rots.append(tuple((-v).to_track_quat("Z", "Y")))
keyframe(cam, "location", np.array(locs, dtype=np.float32))
keyframe(cam, "rotation_quaternion", np.array(rots, dtype=np.float32))
sc.camera = cam
sc.frame_start, sc.frame_end = 1, n

# 2D projections of part centres (for labels in compositing)
proj = []
for k, f in enumerate(frames):
    sc.frame_set(k + 1)
    pts = {}
    for name, c in J["info"][f]["centers"].items():
        p = world_to_camera_view(sc, cam, Vector(c))
        pts[name] = [p.x * rx, (1 - p.y) * ry]
    proj.append(pts)
out.mkdir(parents=True, exist_ok=True)
json.dump(dict(first=a_, proj=proj), open(out / f"proj_{a_:04d}.json", "w"))
if "proj-only" in opts:
    print("projections only", flush=True)
    sys.exit(0)

sc.render.image_settings.file_format = "PNG"
sc.render.filepath = str(out / "frame_")
sc.frame_start, sc.frame_end = 1, n
bpy.ops.render.render(animation=True)
for k in range(n, 0, -1):
    src = out / f"frame_{k:04d}.png"
    if src.exists():
        src.rename(out / f"g_{a_ + k - 1:04d}.png")
print(f"done {n} frames", flush=True)
