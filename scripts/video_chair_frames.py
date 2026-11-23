"""Per-frame box geometry for the chair-only video shots (rendered by render/blender_chair_shots.py).

shot "gen":  hold A -> parametric morph A->B (user + desk inputs interpolated) -> hold B -> explode -> hold
shot "end":  chairs A and B side by side (static; the camera orbits in Blender)

usage: python scripts/video_chair_frames.py   -> build/video/shot_gen.json, build/video/shot_end.json
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from spatialfit import params, geometry

FPS = 30
OUT = Path('build/video'); OUT.mkdir(parents=True, exist_ok=True)
A, B = params.load('configs/A.yaml'), params.load('configs/B.yaml')


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * x * (x * (6 * x - 15) + 10)          # smootherstep


def lerp_params(t):
    f = lambda a, b: a + (b - a) * t
    ia, ib = A.inputs, B.inputs
    inputs = {grp: {k: f(ia[grp][k], ib[grp][k]) for k in ia[grp]} for grp in ("user", "desk")}
    return params.ChairParams("morph", f(A.H, B.H), f(A.W, B.W), f(A.D, B.D), f(A.alpha, B.alpha), inputs)


# exploded-view offsets (m, chair frame: +x right, +y rear, +z up)
def explode_offset(name, part):
    o = part.origin
    sx, sy = np.sign(o[0]) or 0.0, np.sign(o[1]) or 0.0
    if part.kind == "leg":
        return np.array([sx * 0.16, sy * 0.16, -0.0])
    if part.kind == "rail_y":
        return np.array([sx * 0.30, 0.0, 0.10])
    if part.kind == "rail_x":
        return np.array([0.0, sy * 0.30, 0.20])
    if part.kind == "seat":
        return np.array([0.0, -0.06, 0.34])
    if part.kind == "backrest":
        return np.array([0.0, 0.30, 0.52])
    return np.zeros(3)


GROUPS = {"legs": ["LF", "RF", "LR", "RR"], "width rails": ["FR", "BR"], "side rails": ["SL", "SR"],
          "seat": ["SE"], "backrest": ["BA"]}


def frame_boxes(p, explode=0.0, shift=(0, 0, 0)):
    parts = geometry.generate(p)
    out, centers = [], {}
    for n, s in parts.items():
        off = explode_offset(n, s) * explode + np.asarray(shift)
        centers[n] = (s.origin + off + np.mean([b["center"] for b in s.boxes], axis=0)).tolist()
        for k, b in enumerate(s.boxes):
            out.append(dict(key=f"{n}_{k}", part=n, role=b["role"],
                            loc=(s.origin + off + np.asarray(b["center"])).tolist(), size=b["size"], quat=b["quat"]))
        if n == "SE":       # dimension-callout anchors: seat height H at the front-left corner, width W along the front edge
            pb = max((b for b in s.boxes if b["role"] == "panel"), key=lambda b: b["size"][0] * b["size"][1])
            c = s.origin + off + np.asarray(pb["center"]); sz = np.asarray(pb["size"])
            yf, zt, xl, xr = c[1] - sz[1] / 2, c[2] + sz[2] / 2, c[0] - sz[0] / 2, c[0] + sz[0] / 2
            lf = parts["LF"].origin + explode_offset("LF", parts["LF"]) * explode + np.asarray(shift)
            hx, hy = lf[0] - 0.075, lf[1]                     # beside the front-left leg
            centers["dim:H_top"] = [hx, hy, zt]; centers["dim:H_floor"] = [hx, hy, 0.0]
            centers["dim:H_ref"] = [xl, hy, zt]
            centers["dim:W_l"] = [xl, yf, zt + 0.04]; centers["dim:W_r"] = [xr, yf, zt + 0.04]
            centers["dim:W_refl"] = [xl, yf, zt]; centers["dim:W_refr"] = [xr, yf, zt]
    return out, centers


def shot_gen():
    seg = [("holdA", 2.2), ("morph", 5.0), ("holdB", 1.3), ("explode", 2.4), ("holdX", 3.6)]
    frames = []
    for name, dur in seg:
        n = int(round(dur * FPS))
        for i in range(n):
            u = i / max(1, n - 1)
            t = {"holdA": 0.0, "morph": ease(u)}.get(name, 1.0)
            ex = {"explode": ease(u), "holdX": 1.0}.get(name, 0.0)
            p = lerp_params(t)
            boxes, centers = frame_boxes(p, ex)
            for g, members in GROUPS.items():
                centers["grp:" + g] = np.mean([centers[mm] for mm in members], axis=0).tolist()
            frames.append(dict(seg=name, t=t, explode=ex, H=p.H, W=p.W, D=p.D, alpha=p.alpha, inputs=p.inputs,
                               boxes=boxes, centers=centers, camf=dict(dist=2.75 + 1.35 * ex, lookz=0.47 + 0.24 * ex)))
    return frames


def shot_end(dur=7.0):
    gap = 0.18
    xa = -(A.W / 2000 + gap / 2); xb = B.W / 2000 + gap / 2
    ba, ca = frame_boxes(A, 0.0, (xa, 0, 0))
    bb, cb = frame_boxes(B, 0.0, (xb, 0, 0))
    for b in ba:
        b["key"] = "A_" + b["key"]
    for b in bb:
        b["key"] = "B_" + b["key"]
    static = dict(seg="end", boxes=ba + bb, centers={"A": [xa, 0, 0.45], "B": [xb, 0, 0.45], "lab:A": [xa, -A.D / 2000 - 0.09, 0.0],
                           "lab:B": [xb, -B.D / 2000 - 0.09, 0.0],
                           "top:A": [xa, 0.22, 1.02], "top:B": [xb, 0.24, 1.08]},
                  A=dict(H=A.H, W=A.W, D=A.D, alpha=A.alpha), B=dict(H=B.H, W=B.W, D=B.D, alpha=B.alpha))
    return [static] * int(round(dur * FPS))


def dump(name, frames, cam):
    keys = [b["key"] for b in frames[0]["boxes"]]
    meta = {b["key"]: dict(part=b["part"], role=b["role"]) for b in frames[0]["boxes"]}
    series = {k: dict(loc=[], size=[], quat=[]) for k in keys}
    for f in frames:
        idx = {b["key"]: b for b in f["boxes"]}
        for k in keys:
            for a in ("loc", "size", "quat"):
                series[k][a].append(idx[k][a])
    info = [{k: v for k, v in f.items() if k != "boxes"} for f in frames]
    json.dump(dict(fps=FPS, n=len(frames), cam=cam, meta=meta, series=series, info=info, groups=GROUPS),
              open(OUT / f"shot_{name}.json", "w"))
    print(name, len(frames), "frames", len(keys), "boxes")


dump("gen", shot_gen(), dict(look=[0.0, 0.03, 0.47], dist=2.75, el=-19.0, az0=52.0, az_rate=2.2, lens=45))
dump("end", shot_end(), dict(look=[0.0, 0.0, 0.46], dist=3.25, el=-14.0, az0=70.0, az_rate=1.8, lens=45))
