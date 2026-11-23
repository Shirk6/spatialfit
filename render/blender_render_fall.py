"""Variant of blender_render2.py for the stability-comparison shot (fixed camera + push-in, camera dump).\n\nFast, studio-look render of a recorded MuJoCo build (keyframed animation, one render call, direct mp4).

usage:
  blender -b --factory-startup -P render/blender_render2.py -- SCENE_DIR LOG.npz OUT(.mp4 | dir/)
          [--engine eevee|cycles] [--res 1920x1080] [--samples 16] [--speed 4] [--slow 1]
          [--max N] [--frames i,j (stills, log indices)] [--orbit -138:-112]
"""
import bpy, sys, json, math, time
from pathlib import Path
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
scene_dir, log_path, out = Path(argv[0]), Path(argv[1]), argv[2]
opts = {argv[i][2:]: argv[i + 1] for i in range(3, len(argv) - 1, 2) if argv[i].startswith("--")}
engine = opts.get("engine", "eevee"); speed = int(opts.get("speed", 4)); slow = int(opts.get("slow", 1))
fps = int(opts.get("fps", 30))                     # output frame rate; the log is recorded at 30 Hz
step = max(1, round(30 / fps))
rx, ry = (int(v) for v in opts.get("res", "1920x1080").split("x"))
samples = int(opts.get("samples", 16))
orb_start = float(opts.get("orbit-start", -136.0))       # deg
orb_rate = float(opts.get("orbit-rate", 10.0 / 60.0))     # deg per second of video (slow, continuous across chunks)

S = json.load(open(scene_dir / "scene.json"))
L = np.load(log_path)
hud = json.load(open(str(log_path).replace(".npz", "_hud.json")))["hud"]
xpos, xquat = L["xpos"], L["xquat"]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

# ------------------------------------------------------------------ engine
if engine == "eevee":
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    ee = sc.eevee
    ee.taa_render_samples = max(8, samples)
    for attr, val in (("use_raytracing", True), ("use_shadows", True), ("shadow_ray_count", 1), ("shadow_step_count", 6),
                      ("use_gtao", True), ("gtao_distance", 0.4), ("fast_gi_method", "GLOBAL_ILLUMINATION")):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except Exception:
                pass
    if hasattr(ee, "ray_tracing_options"):
        ee.ray_tracing_options.resolution_scale = "2"
else:
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
    if hasattr(sc.cycles, "denoising_use_gpu"):
        sc.cycles.denoising_use_gpu = True
    sc.cycles.max_bounces = 4; sc.cycles.glossy_bounces = 2; sc.cycles.transmission_bounces = 0
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.render.use_persistent_data = True
    print("[render] cycles device", prefs.compute_device_type, flush=True)
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = rx, ry, 100
sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Punchy"; sc.view_settings.exposure = -0.2
sc.render.fps = fps

# ------------------------------------------------------------------ materials
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
    if grain:   # oak grain: wave texture distorted by noise, along the board's long axis
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = grain
        wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.inputs["Scale"].default_value = 3.0
        wv.inputs["Distortion"].default_value = 2.0; wv.inputs["Detail"].default_value = 2.0
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (base[0] * 0.72, base[1] * 0.66, base[2] * 0.60, 1)
        cr.color_ramp.elements[1].color = (min(1, base[0] * 1.10), min(1, base[1] * 1.08), min(1, base[2] * 1.04), 1)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], wv.inputs["Vector"])
        nt.links.new(wv.outputs["Fac"], cr.inputs["Fac"]); nt.links.new(cr.outputs["Color"], bs.inputs["Base Color"])
    _m[key] = m
    return m


M = dict(
    oak=mat("oak", (0.58, 0.36, 0.18), 0.0, 0.42, coat=0.35, grain=(3.0, 60.0, 3.0)),
    frame=mat("frame", (0.02, 0.075, 0.09), 0.8, 0.3, coat=0.5),            # dark anodised teal
    steel=mat("steel", (0.80, 0.81, 0.83), 1.0, 0.22, aniso=0.4),
    alu=mat("alu", (0.70, 0.71, 0.73), 1.0, 0.35, aniso=0.6),
    accent=mat("accent", (0.95, 0.42, 0.05), 0.0, 0.35, coat=0.4),             # nest accent
    gripper=mat("gripper", (0.03, 0.032, 0.036), 0.5, 0.35, coat=0.3),
    plinth=mat("plinth", (0.06, 0.065, 0.075), 0.6, 0.28, coat=0.6),           # robot pedestal / fixtures
    rack=mat("rack", (0.10, 0.105, 0.115), 0.7, 0.3, coat=0.3),
    stage=mat("stage", (0.09, 0.093, 0.10), 0.0, 0.6, spec=0.3),             # work platform
    floor=mat("floor", (0.035, 0.037, 0.042), 0.0, 0.22, spec=0.6),            # glossy studio floor
)


def material_for(body, g):
    role, rgba, name = g["role"], g["rgba"], g["name"]
    if body is not None and body.get("robot"):
        if g["type"] == "box":
            return M["gripper"]
        c = tuple(rgba[:3]); light = sum(c) > 1.5
        return mat(f"ur_{c}", c if not light else (0.86, 0.87, 0.88), 0.0 if light else 0.4, 0.28 if light else 0.4, coat=0.5 if light else 0.2)
    if role in ("panel", "foot", "cleat", "hinge"):
        return M["oak"]
    if role in ("tongue", "post"):
        return M["steel"]
    if role in ("head", "leg", "rail", "body"):
        return M["frame"]
    if name.startswith("nest_"):
        return M["accent"]
    if name.startswith("rack_"):
        return M["rack"]
    if name == "pedestal":
        return M["plinth"]
    return mat(f"misc_{tuple(rgba[:3])}", tuple(rgba[:3]), 0.0, 0.5)


# ------------------------------------------------------------------ geometry helpers
_cube = None


def box(name, size, material, parent=None, pos=(0, 0, 0), quat=(1, 0, 0, 0), bevel=0.003):
    global _cube
    if _cube is None:
        bpy.ops.mesh.primitive_cube_add(size=1.0); _cube = bpy.context.object.data; bpy.data.objects.remove(bpy.context.object)
    o = bpy.data.objects.new(name, _cube.copy()); sc.collection.objects.link(o)
    o.scale = size
    bw = min(min(size) * 0.2, bevel)
    if bw > 0.0004:
        md = o.modifiers.new("bevel", "BEVEL"); md.width = bw; md.segments = 2; md.use_clamp_overlap = True
        o.modifiers.new("wn", "WEIGHTED_NORMAL")
    o.data.materials.append(material)
    o.parent = parent; o.location = pos; o.rotation_mode = "QUATERNION"; o.rotation_quaternion = quat
    return o


_mesh = {}


def mesh(name, file, material, parent, pos, quat):
    if file not in _mesh:
        V, F = [], []
        for line in open(scene_dir / file):
            if line.startswith("v "):
                V.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                F.append(tuple(int(x) - 1 for x in line.split()[1:4]))
        me = bpy.data.meshes.new(file); me.from_pydata(V, [], F); me.update()
        for p in me.polygons:
            p.use_smooth = True
        _mesh[file] = me
    o = bpy.data.objects.new(name, _mesh[file]); sc.collection.objects.link(o)
    if not o.data.materials:
        o.data.materials.append(material)
    o.material_slots[0].link = "OBJECT"; o.material_slots[0].material = material
    o.parent = parent; o.location = pos; o.rotation_mode = "QUATERNION"; o.rotation_quaternion = quat
    return o


def table(name, size, pos, quat, material, parent):
    """Replace a solid shelf block by a small table (30 mm top plate on four 40 mm legs), in the rack body's frame."""
    from mathutils import Quaternion
    sx, sy, sz = size; c = Vector(pos); q = Quaternion(quat)
    box(f"{name}_top", (sx, sy, 0.03), material, parent, tuple(c + q @ Vector((0, 0, sz / 2 - 0.015))), quat, bevel=0.006)
    for dx in (-1, 1):
        for dy in (-1, 1):
            off = Vector((dx * (sx / 2 - 0.03), dy * (sy / 2 - 0.03), -0.015))
            box(f"{name}_leg{dx}{dy}", (0.04, 0.04, sz - 0.03), material, parent, tuple(c + q @ off), quat, bevel=0.004)


# ------------------------------------------------------------------ build bodies
empties = {}
for b in S["bodies"]:
    e = bpy.data.objects.new(f"B_{b['name']}", None); sc.collection.objects.link(e)
    e.rotation_mode = "QUATERNION"; empties[b["id"]] = e
    for g in b["geoms"]:
        mt = material_for(b, g)
        if g["type"] == "box":
            if g["name"] == "rack_SE_base" and g["size"][2] > 0.12:
                table(g["name"], g["size"], g["pos"], g["quat"], M["rack"], e)     # seat shelf drawn as a table
            else:
                box(g["name"] or b["name"], g["size"], mt, e, g["pos"], g["quat"], bevel=0.002 if b.get("part") else 0.004)
        else:
            mesh(g["name"] or b["name"], g["file"], mt, e, g["pos"], g["quat"])
for g in S["world"]:
    box(g["name"], g["size"], material_for(None, g), None, g["pos"], g["quat"], bevel=0.01)

# work platform (low bevelled stage) + glossy floor + soft backdrop
box("stage", (2.3, 2.1, 0.02), M["stage"], None, (0.0, 0.08, -0.0101), bevel=0.008)
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
    return o


area("key", 520, 2.2, (1.9, -1.7, 2.7), (0.0, 0.2, 0.3), (1.0, 0.94, 0.86))     # warm key
area("fill", 120, 3.0, (-2.4, -0.8, 1.8), (0.0, 0.2, 0.3), (0.85, 0.92, 1.0))    # cool fill
area("rim", 450, 1.6, (-0.6, 2.4, 2.2), (0.0, 0.4, 0.5), (0.80, 0.90, 1.0))     # rim from behind
area("top", 120, 3.0, (0.0, 0.3, 3.2), (0.0, 0.3, 0.0))

# ------------------------------------------------------------------ frames to show
SLOW = ("insert", "jam", "re-observe", "latch", "grasp", "fixture release", "stability")
if "frames" in opts:
    idx = [int(v) for v in opts["frames"].split(",")]
else:
    idx, i = [], 0
    while i < len(xpos):
        idx.append(i)
        i += (slow if any(k in hud[i]["phase"] for k in SLOW) else speed) * step
    idx += [len(xpos) - 1] * int(1.5 * fps)
first = 0
if "range" in opts:                                  # render only output frames [a, b) of the full video
    a_, b_ = (int(v) for v in opts["range"].split(":"))
    first, idx = a_, idx[a_:b_]
if "max" in opts:
    idx = idx[: int(opts["max"])]
n = len(idx)
print(f"[render] output frames {first}..{first + n - 1}", flush=True)
sc.frame_start, sc.frame_end = 1, n


def keyframe(obj, path, values):
    """values: (n, k) array -> one fcurve per component, inserted in bulk."""
    if obj.animation_data is None:
        obj.animation_data_create()
    act = obj.animation_data.action or bpy.data.actions.new(obj.name + "_act")
    obj.animation_data.action = act
    frames = np.arange(1, n + 1, dtype=np.float32)
    for k in range(values.shape[1]):
        fc = act.fcurves.find(path, index=k) or act.fcurves.new(path, index=k)
        fc.keyframe_points.add(n)
        co = np.empty(2 * n, dtype=np.float32); co[0::2] = frames; co[1::2] = values[:, k]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", np.zeros(n, dtype=np.int32))   # CONSTANT
        fc.update()


ii = np.array(idx)
for bid, e in empties.items():
    keyframe(e, "location", xpos[ii, bid])
    keyframe(e, "rotation_quaternion", xquat[ii, bid])

# camera: fixed framing with a slow push-in (--cam lx,ly,lz,dist0,dist1,el,az0,az1,lens); dumps per-frame camera
cv = [float(v) for v in opts.get("cam", "0,0.58,0.24,2.0,1.7,-20,-104,-98,42").split(",")]
look, d0, d1, el, az0, az1, lens = Vector(cv[0:3]), cv[3], cv[4], math.radians(cv[5]), cv[6], cv[7], cv[8]
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
cam.data.lens = lens; cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = 36.0
cam.data.dof.use_dof = True; cam.data.dof.aperture_fstop = 4.5
tgt = bpy.data.objects.new("cam_target", None); sc.collection.objects.link(tgt); tgt.location = look
cam.data.dof.focus_object = tgt
cam.rotation_mode = "QUATERNION"
locs, rots, dump = [], [], []
for k in range(n):
    u = k / max(1, n - 1); u = u * u * (3 - 2 * u)
    az = math.radians(az0 + (az1 - az0) * u); dist = d0 + (d1 - d0) * u
    f = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    loc = look - dist * f; q = (-f).to_track_quat("Z", "Y")
    locs.append(tuple(loc)); rots.append(tuple(q))
    Rm = q.to_matrix()
    dump.append(dict(loc=list(loc), R=[list(Rm.col[0]), list(Rm.col[1]), list(Rm.col[2])]))
keyframe(cam, "location", np.array(locs, dtype=np.float32))
keyframe(cam, "rotation_quaternion", np.array(rots, dtype=np.float32))
sc.camera = cam
if not out.endswith(".mp4"):
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(lens=lens, sensor_width=36.0, res=[rx, ry], frames=dump, log_index=idx),
              open(Path(out) / "camera.json", "w"))

# ------------------------------------------------------------------ output
t0 = time.time()
if out.endswith(".mp4"):
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "HIGH"; sc.render.ffmpeg.ffmpeg_preset = "REALTIME"
    sc.render.filepath = out; sc.render.use_file_extension = False
    print(f"[render] {n} frames, engine {sc.render.engine}, {rx}x{ry}", flush=True)
    bpy.ops.render.render(animation=True)
else:
    Path(out).mkdir(parents=True, exist_ok=True)
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(Path(out) / "frame_")
    bpy.ops.render.render(animation=True)
el_ = time.time() - t0
print(f"done {n} frames in {el_:.1f} s ({el_ / n:.3f} s/frame)", flush=True)
