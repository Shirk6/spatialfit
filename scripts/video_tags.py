"""Screen positions of the part being handled, for the live part tag in the A|B split screen.

Re-creates blender_render2.py's orbit camera for the tile renders (rendered with --frames, so the orbit
index is the position k in the frame list) and projects each part's geometric centre.
writes build/video/tags_{A,B}.json: per tile frame k -> {"part": name, "uv": [x, y] in 1280x720 px} or null
"""
import json, math, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '.')
from spatialfit import params, geometry

RX, RY, LENS, SENSOR = 1280, 720, 38.0, 36.0
LOOK, DIST, EL = np.array([-0.02, 0.12, 0.40]), 2.7, math.radians(-24)
FULL = {"LF": "leg", "RF": "leg", "LR": "leg", "RR": "leg", "FR": "width rail", "BR": "width rail",
        "SL": "side rail", "SR": "side rail", "SE": "seat", "BA": "backrest"}


def camera(k, fps=30):
    az = math.radians(-136.0 + (10.0 / 60.0) * k / fps)
    f = np.array([math.cos(EL) * math.cos(az), math.cos(EL) * math.sin(az), math.sin(EL)])
    loc = LOOK - DIST * f
    zc = -f / np.linalg.norm(f)                       # to_track_quat('Z', 'Y'): local Z along -f, local Y up
    up = np.array([0, 0, 1.0]); yc = up - zc * up.dot(zc); yc /= np.linalg.norm(yc)
    xc = np.cross(yc, zc)
    return loc, xc, yc, zc


def project(p, cam):
    loc, xc, yc, zc = cam
    d = np.asarray(p) - loc
    x, y, z = d.dot(xc), d.dot(yc), d.dot(zc)
    fpx = LENS / SENSOR * RX
    return [RX / 2 + fpx * x / -z, RY / 2 - fpx * y / -z]


if __name__ == "__main__":
    T = json.load(open('build/video/tiles.json'))
    for side, cfg in (("A", "A"), ("B", "B")):
        parts = geometry.generate(params.load(f'configs/{cfg}.yaml'))
        local = {n: np.mean([b["center"] for b in s.boxes], axis=0) for n, s in parts.items()}
        L = np.load(f'build/{T[side]["log"]}.npz')
        names = list(L["body_names"])
        hud = json.load(open(f'build/{T[side]["log"]}_hud.json'))["hud"]
        out = []
        for k, li in enumerate(T[side]["log_frames"]):
            n = hud[li]["part"]
            if not n:
                out.append(None); continue
            b = names.index(n)
            q = L["xquat"][li, b]
            c = L["xpos"][li, b] + R.from_quat([q[1], q[2], q[3], q[0]]).apply(local[n])
            out.append(dict(part=n, kind=FULL[n], uv=project(c, camera(k))))
        json.dump(out, open(f'build/video/tags_{side}.json', 'w'))
        print(side, len(out))
    # sanity: robot pedestal top-centre in tile A frame 0
    print("base(0,0,0.40) ->", np.round(project([0, 0, 0.40], camera(0)), 1))
