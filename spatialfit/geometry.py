"""Parametric generator: ChairParams -> 10 PartSpecs built only from boxes.

The same box list feeds MuJoCo (collision + visual), cuRobo (cuboid obstacles) and Blender.
Internal units are millimetres in the chair frame; exported specs are metres.
Chair frame: origin at floor centre, +x = width (right), +y = depth (rear), +z = up.
"""
from dataclasses import dataclass, field
import math
import numpy as np

# fixed connector dimensions (mm) shared by every configuration
RAIL_W, RAIL_H = 24, 30          # rail cross-section
TONGUE_T, TONGUE_W, TONGUE_H = 10, 16, 20  # along-rail thickness, across width, depth
TIP_H, TIP_T, TIP_W = 6, 6, 12   # stepped lead-in tip
CLEAR = 1.0                      # side clearance per side (simulation value)
FLOAT = 1.5                      # extra travel per side at floating ends
HEAD = 64                        # corner head size
POST = 16                        # backrest post section
DOWN = (0.0, 0.0, -1.0)
SEAT_OVERHANG = 50.0     # mm the seat panel extends beyond the front of the frame


def tool_R(z_dir, y_dir):
    """Tool orientation (columns = tool x, y, z) from approach axis z and finger-closing axis y."""
    z = np.asarray(z_dir, float); z /= np.linalg.norm(z)
    y = np.asarray(y_dir, float); y -= z * np.dot(y, z); y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z]).tolist()


@dataclass
class Box:
    center: tuple   # mm, chair frame (converted later)
    size: tuple     # full extents, mm
    role: str = "body"
    quat: tuple = (1.0, 0.0, 0.0, 0.0)


@dataclass
class PartSpec:
    name: str
    kind: str
    origin: np.ndarray                 # target position of part frame in chair frame (m)
    boxes: list                        # dicts with center/size in part frame (m)
    coarse: list                       # coarse boxes for planning obstacles (part frame, m)
    requires: list                     # host parts that must be installed first
    connectors: list                   # dicts: name, host, type, pos (part frame, m)
    grasp: dict                        # tcp (part frame, m), R (tool axes in part frame), width (m); no dedicated tabs
    insert_dir: tuple = (0.0, 0.0, -1.0)
    color: str = "frame"
    meta: dict = field(default_factory=dict)


def _decompose(solids, cuts):
    """Axis-aligned union(solids) minus union(cuts) -> list of boxes (lo, hi) via grid + greedy merge."""
    axes = []
    for k in range(3):
        pts = set()
        for lo, hi in solids + cuts:
            pts.add(round(lo[k], 4)); pts.add(round(hi[k], 4))
        axes.append(sorted(pts))
    nx, ny, nz = (len(a) - 1 for a in axes)
    filled = np.zeros((nx, ny, nz), bool)
    mids = [np.array([(a[i] + a[i + 1]) / 2 for i in range(len(a) - 1)]) for a in axes]

    def inside(boxes, p):
        return any(all(lo[k] < p[k] < hi[k] for k in range(3)) for lo, hi in boxes)
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                p = (mids[0][i], mids[1][j], mids[2][l])
                filled[i, j, l] = inside(solids, p) and not inside(cuts, p)
    used = np.zeros_like(filled)
    out = []
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                if not filled[i, j, l] or used[i, j, l]:
                    continue
                i2 = i
                while i2 + 1 < nx and filled[i2 + 1, j, l] and not used[i2 + 1, j, l]:
                    i2 += 1
                j2 = j
                while j2 + 1 < ny and np.all(filled[i:i2 + 1, j2 + 1, l] & ~used[i:i2 + 1, j2 + 1, l]):
                    j2 += 1
                l2 = l
                while l2 + 1 < nz and np.all(filled[i:i2 + 1, j:j2 + 1, l2 + 1] & ~used[i:i2 + 1, j:j2 + 1, l2 + 1]):
                    l2 += 1
                used[i:i2 + 1, j:j2 + 1, l:l2 + 1] = True
                out.append(((axes[0][i], axes[1][j], axes[2][l]), (axes[0][i2 + 1], axes[1][j2 + 1], axes[2][l2 + 1])))
    return out


def _lohi(cx, cy, cz, sx, sy, sz):
    return ((cx - sx / 2, cy - sy / 2, cz - sz / 2), (cx + sx / 2, cy + sy / 2, cz + sz / 2))


def _box_from_lohi(lo, hi, role="body"):
    return Box(tuple((a + b) / 2 for a, b in zip(lo, hi)), tuple(b - a for a, b in zip(lo, hi)), role)


def _quat_x(deg):
    t = math.radians(deg) / 2
    return (math.cos(t), math.sin(t), 0.0, 0.0)


def _rot_x(deg, v):
    t = math.radians(deg)
    x, y, z = v
    return (x, y * math.cos(t) - z * math.sin(t), y * math.sin(t) + z * math.cos(t))


def _quat_from_matrix(Rm):
    from scipy.spatial.transform import Rotation as _R
    x, y, z, w = _R.from_matrix(Rm).as_quat()
    return (w, x, y, z)


def tapered(center, size, taper_h, inset, role, t=3.0):
    """Box whose bottom `taper_h` tapers inward by `inset` on all four sides (lead-in), built from boxes only."""
    cx, cy, cz = center
    sx, sy, sz = size
    z0 = cz - sz / 2
    out = [Box((cx, cy, z0 + taper_h + (sz - taper_h) / 2), (sx, sy, sz - taper_h), role),
           Box((cx, cy, z0 + taper_h / 2), (sx - 2 * inset, sy - 2 * inset, taper_h), role)]
    L = math.hypot(inset, taper_h)
    for axis in (0, 1):
        for sgn in (-1, 1):
            half = (sx if axis == 0 else sy) / 2
            other = (sy if axis == 0 else sx) - 2 * inset
            # slope segment from top-outer (half, taper_h) to bottom-inner (half-inset, 0) in the (axis, z) plane
            top = np.zeros(3); bot = np.zeros(3)
            top[axis] = sgn * half; top[2] = taper_h
            bot[axis] = sgn * (half - inset); bot[2] = 0.0
            zax = (top - bot) / L
            xax = np.zeros(3); xax[axis] = -sgn * taper_h / L; xax[2] = inset / L  # inward normal (perp. to slope)
            if xax[2] < 0:
                xax = -xax
            xax = xax - zax * np.dot(xax, zax); xax /= np.linalg.norm(xax)
            if np.dot(xax, [-sgn if axis == 0 else 0, -sgn if axis == 1 else 0, 0]) < 0:
                xax = -xax
            yax = np.cross(zax, xax)
            mid = (top + bot) / 2 + xax * t / 2
            c = (cx + mid[0], cy + mid[1], z0 + mid[2])
            dims = (t, other, L)
            out.append(Box(c, dims, role, _quat_from_matrix(np.column_stack([xax, yax, zax]))))
    return out


def generate(p) -> dict:
    """Return dict name -> PartSpec for chair params p (H, W, D, alpha in mm/deg)."""
    H, W, D, alpha = p.H, p.W, p.D, p.alpha
    OH = SEAT_OVERHANG                 # seat front edge overhangs the frame (waterfall edge, also the grasp edge)
    a, b = (W - 50) / 2, (D - OH - 50) / 2
    zx = H - 48   # bearing plane of width rails (proposal Eq. 2)
    zy = H - 80   # bearing plane of side rails (proposal Eq. 3)
    head_lo = H - 130
    parts = {}

    def add(name, kind, origin_mm, boxes, coarse, requires, connectors, grasp, color, meta=None):
        o = np.array(origin_mm, float)
        gconv = lambda g: dict(g, tcp=((np.array(g["tcp"]) - o) / 1000).tolist(), width=g["width"] / 1000,
                               R=np.asarray(g["R"], float).tolist())
        conv = lambda bx: dict(center=((np.array(bx.center) - o) / 1000).tolist(),
                               size=(np.array(bx.size) / 1000).tolist(), role=bx.role, quat=list(bx.quat))
        parts[name] = PartSpec(
            name, kind, o / 1000, [conv(bx) for bx in boxes], [conv(bx) for bx in coarse], requires,
            [dict(c, pos=((np.array(c["pos"]) - o) / 1000).tolist()) for c in connectors],
            gconv(grasp[0] if isinstance(grasp, list) else grasp),
            color=color, meta=dict(meta or {}, grasps=[gconv(g) for g in (grasp if isinstance(grasp, list) else [grasp])]))

    # ---------------- legs ----------------
    legs = {"LF": (-1, -1), "RF": (1, -1), "LR": (-1, 1), "RR": (1, 1)}
    for name, (sx, sy) in legs.items():
        X, Y = sx * a, sy * b
        M = lambda cx, cy: (X + sx * cx, Y + sy * cy)     # canonical corner -> chair frame

        def mb(cx0, cx1, cy0, cy1, z0, z1):
            (x0, y0), (x1, y1) = M(cx0, cy0), M(cx1, cy1)
            return ((min(x0, x1), min(y0, y1), z0), (max(x0, x1), max(y0, y1), z1))
        solids = [mb(-HEAD / 2, HEAD / 2, -HEAD / 2, HEAD / 2, head_lo, zx)]
        rear = sy > 0
        if rear:
            solids.append(mb(-17, 17, HEAD / 2, 66, H - 100, zx))
        fx = TONGUE_T / 2 + CLEAR + (FLOAT if sx > 0 else 0)        # x-rail floating end on right legs
        fy = TONGUE_T / 2 + CLEAR + (FLOAT if sy > 0 else 0)        # side-rail floating end on rear legs
        hw = TONGUE_W / 2 + CLEAR
        cuts = [mb(-fx, fx, 20 - hw, 20 + hw, zx - TONGUE_H, zx + 1),          # width-rail receiver
                mb(-20 - RAIL_W / 2 - 2 * CLEAR, -20 + RAIL_W / 2 + 2 * CLEAR, -HEAD, 7, zy, zx + 1),  # side-rail lane (guide, 2 mm)
                mb(-20 - hw, -20 + hw, -fy, fy, zy - TONGUE_H, zy + 1)]        # side-rail receiver
        if rear:
            ps = POST / 2 + CLEAR + (FLOAT if sx > 0 else 0)
            cuts.append(mb(-ps, ps, 50 - POST / 2 - CLEAR, 50 + POST / 2 + CLEAR, zx - 30, zx + 1))
        boxes = [_box_from_lohi(lo, hi, "head") for lo, hi in _decompose(solids, cuts)]
        boxes += tapered((X, Y, head_lo / 2), (40, 40, head_lo), 10, 4, "leg")
        coarse = [Box((X, Y, head_lo / 2), (40, 40, head_lo)), _box_from_lohi(*solids[0])]
        if rear:
            coarse.append(_box_from_lohi(*solids[1]))
        conns = [dict(name=f"{name}.nest", host="fixture", type="locating", pos=(X, Y, 0))]
        add(name, "leg", (X, Y, 0), boxes, coarse, [], conns,
            dict(tcp=(X, Y, zx - 45), R=tool_R(DOWN, (1, 0, 0)), width=HEAD), "frame",   # pinch the corner head
            dict(receivers=dict(x=M(0, 20) + (zx,), y=M(-20, 0) + (zy,),
                                post=(M(0, 50) + (zx,)) if rear else None)))

    # ---------------- width rails (front / rear) ----------------
    for name, sy, hosts in (("FR", -1, ("LF", "RF")), ("BR", 1, ("LR", "RR"))):
        y = sy * (b + 20)
        L = 2 * a + 40
        boxes = [Box((0, y, zx + RAIL_H / 2), (L, RAIL_W, RAIL_H), "rail")]
        conns = []
        for sx, host in zip((-1, 1), hosts):
            x = sx * a
            boxes += tapered((x, y, zx - TONGUE_H / 2), (TONGUE_T, TONGUE_W, TONGUE_H), TIP_H, 3, "tongue")
            conns.append(dict(name=f"{name}.{host}", host=host, type="floating" if sx > 0 else "locating",
                              pos=(x, y, zx)))
        add(name, "rail_x", (0, y, zx), boxes, [boxes[0]], list(hosts), conns,
            dict(tcp=(0, y, zx + RAIL_H - 10), R=tool_R(DOWN, (0, 1, 0)), width=RAIL_W), "frame")

    # ---------------- side rails ----------------
    for name, sx, hosts in (("SL", -1, ("LF", "LR")), ("SR", 1, ("RF", "RR"))):
        x = sx * (a - 20)
        L = 2 * b + 12
        boxes = [Box((x, 0, zy + RAIL_H / 2), (RAIL_W, L, RAIL_H), "rail")]
        conns = []
        for sy, host in zip((-1, 1), hosts):
            y = sy * b
            boxes += tapered((x, y, zy - TONGUE_H / 2), (TONGUE_W, TONGUE_T, TONGUE_H), TIP_H, 3, "tongue")
            conns.append(dict(name=f"{name}.{host}", host=host, type="floating" if sy > 0 else "locating",
                              pos=(x, y, zy)))
        add(name, "rail_y", (x, 0, zy), boxes, [boxes[0]], list(hosts), conns,
            dict(tcp=(x, 0, zy + RAIL_H - 10), R=tool_R(DOWN, (1, 0, 0)), width=RAIL_W), "frame")

    # ---------------- seat cassette ----------------
    zs = H - 18
    boxes = [Box((0, -OH / 2, zs + 9), (W, D, 18), "panel")]          # panel: rear edge at b+25, front at -(b+25+OH)
    for sx in (-1, 1):  # feet on side rails + inner cleats locating against side rails
        boxes.append(Box((sx * (a - 20), 0, (zy + RAIL_H + zs) / 2), (20, 2 * b - 120, zs - zy - RAIL_H), "foot"))
        boxes.append(Box((sx * (a - 20 - RAIL_W / 2 - 2 * CLEAR - 5), 0, zs - 21), (10, 80, 42), "cleat"))
    for sy in (-1, 1):  # cleats locating against width rails
        boxes.append(Box((0, sy * (b + 20 - RAIL_W / 2 - 2 * CLEAR - 5), zs - 10), (80, 10, 20), "cleat"))
    conns = [dict(name=f"SE.{r}", host=r, type="support", pos=xy + (zs,))
             for r, xy in (("FR", (0, -(b + 20))), ("BR", (0, b + 20)), ("SL", (-(a - 20), 0)), ("SR", (a - 20, 0)))]
    # edge pinch on a side edge (side rails are inset 45 mm, so the lower finger fits under the panel); either side
    add("SE", "seat", (0, 0, zs), boxes, [boxes[0], boxes[1], boxes[3]], ["FR", "BR", "SL", "SR"], conns,
        [dict(tcp=(0, -(b + 25 + OH) + 17, zs + 9), R=tool_R((0, 1, 0), (0, 0, 1)), width=18)] +      # front edge (overhang)
        [dict(tcp=(sx * (W / 2 - 17), 0, zs + 9), R=tool_R((-sx, 0, 0), (0, 0, 1)), width=18)
         for sx in (1, -1)], "wood")   # or either side edge

    # ---------------- backrest cassette ----------------
    tilt = alpha - 90
    yb = b + 50
    zh = H + 30                      # hinge bar bottom
    post_lo = zx - 30
    boxes, coarse = [], []
    for sx in (-1, 1):
        boxes += tapered((sx * a, yb, (post_lo + zh) / 2), (POST, POST, zh - post_lo), 8, 4, "post")
    boxes.append(Box((0, yb, zh + 8), (2 * a + POST, POST, 16), "hinge"))
    ph, pt = 360, 18
    pivot = (0, yb, zh + 16)
    c_local = (0, 0, ph / 2)
    c = _rot_x(-tilt, c_local)
    boxes.append(Box((pivot[0] + c[0], pivot[1] + c[1], pivot[2] + c[2]), (W - 40, pt, ph), "panel", _quat_x(-tilt)))
    up = np.array(_rot_x(-tilt, (0, 0, 1))); normal = np.array(_rot_x(-tilt, (0, 1, 0)))
    conns = [dict(name="BA.LR", host="LR", type="locating", pos=(-a, yb, zx)),
             dict(name="BA.RR", host="RR", type="floating", pos=(a, yb, zx)),
             dict(name="BA.SE", host="SE", type="clearance", pos=(0, b, H))]
    add("BA", "backrest", (0, yb, zx), boxes, list(boxes), ["LR", "RR", "SE"], conns,
        [dict(tcp=tuple(np.array(pivot) + up * (ph - 17)), R=tool_R(-up, normal), width=pt)] +      # top edge of the panel
        [dict(tcp=(sx * a, yb, zx + 45),
              R=tool_R((-sx * math.cos(math.radians(pt_)), 0, -math.sin(math.radians(pt_))), (0, 1, 0)), width=POST)
         for sx in (-1, 1) for pt_ in (0, 30, 50)],   # or pinch a post from outside, horizontally or from above-outside
        "wood", dict(tilt=tilt))   # pinch the left or the right post from outside
    return parts


def summary(parts):
    return {n: dict(kind=s.kind, boxes=len(s.boxes), requires=s.requires,
                    origin_mm=(s.origin * 1000).round(1).tolist()) for n, s in parts.items()}
