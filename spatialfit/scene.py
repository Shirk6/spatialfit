"""MuJoCo workcell built with MjSpec: UR5e + parallel gripper + leg nests + staging tables + 10 chair parts."""
from pathlib import Path
import numpy as np
import warnings
import mujoco
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
UR5E = ROOT / "third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME_Q = np.array([-1.5708, -1.9, 1.9, -1.5708, -1.5708, 0.0])

# layout (m)
PEDESTAL_H = 0.40
CHAIR_X = 0.0                        # chair in front of the robot, its front (seat overhang) facing the robot
CHAIR_YAW = np.pi                    # chair rotated: its back faces the robot
CHAIR_Y = 0.58                       # chair centre in front of robot (+y), chair front faces robot
TCP_Z = 0.092                        # TCP offset from flange along tool z
PALM = (0.04, 0.09, 0.04)            # palm (x, closing-y, z)
FINGER = (0.022, 0.010, 0.060)
COLORS = {"frame": (0.05, 0.30, 0.32, 1), "wood": (0.78, 0.55, 0.32, 1), "tab": (0.08, 0.09, 0.10, 1),
          "tongue": (0.70, 0.72, 0.74, 1), "post": (0.70, 0.72, 0.74, 1)}
DENSITY = 700.0


def quat_wxyz(rot: R):
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def rot_of(q_wxyz):
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


RACK_H = 0.03                        # base plate of the floor racks for the two panels
BA_PUSH = 0.15                       # radial push of the backrest rack away from the robot base (m)
STAGING = {                          # (x, y, yaw[, beta]): pose = Rz(yaw) * Ry(beta); beta != 0 -> part stands in a rack
    "LF": (-0.38, 0.38, 0.0), "RF": (-0.50, 0.38, 0.0), "LR": (-0.38, 0.52, 0.0), "RR": (-0.50, 0.52, 0.0),
    "FR": (-0.36, 0.0, np.pi / 2), "BR": (-0.44, 0.0, np.pi / 2), "SL": (-0.52, 0.0, 0.0), "SR": (-0.60, 0.0, 0.0),
    "SE": ("orbit", np.radians(-90)),            # installed pose swung about the robot base
    # backrest: swung 180 deg about the base and pushed BA_PUSH further out, so the forearm has room to reach over
    # a tall, reclined backrest (chair B)
    "BA": ("orbit", np.radians(180), BA_PUSH),
}


def rack_height(entry):
    if entry[0] == "orbit" or len(entry) > 4:
        return entry[4] if len(entry) > 4 else 0.0
    return RACK_H if len(entry) > 3 and abs(entry[3]) > 1e-6 else 0.0


def staging_rot(entry):
    yaw = entry[2]
    beta = entry[3] if len(entry) > 3 else 0.0
    return R.from_euler("z", yaw) * R.from_euler("y", beta)


def part_corners(s, pos, rot):
    pts = []
    for bx in s.boxes:
        r = rot * rot_of(np.array(bx["quat"]))
        c = pos + rot.apply(bx["center"])
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.array(bx["size"]) / 2
        pts.append(c + r.apply(corners))
    return np.vstack(pts)


def staging_layout(parts, table=None):
    """Staging poses (pos, quat): small parts lie/stand on the floor; the panels stand in floor racks."""
    table = table or STAGING
    poses = {}
    for n, e in table.items():
        rot = staging_rot(e)
        z0 = part_corners(parts[n], np.zeros(3), rot)[:, 2].min()
        z = -z0 + 0.0003 + rack_height(e)
        poses[n] = (np.array([e[0], e[1], z]), quat_wxyz(rot))
    return poses


class Workcell:
    def __init__(self, parts, chair_y=CHAIR_Y, timestep=0.001, staging=None, chair_x=None, chair_yaw=None):
        self.parts = parts
        self.names = list(parts)
        self.chair_origin = np.array([CHAIR_X if chair_x is None else chair_x, chair_y, 0.0])
        self.chair_rot = R.from_euler("z", CHAIR_YAW if chair_yaw is None else chair_yaw)
        self.chair_quat = quat_wxyz(self.chair_rot)
        self.staging_table = dict(staging or STAGING)
        self.staging = staging_layout(parts, {k: v for k, v in self.staging_table.items() if v[0] != "orbit"})
        for n, e in self.staging_table.items():
            if e[0] == "orbit":          # installed pose rotated about the robot base axis: transfer is (almost) a pure base rotation
                th = e[1]
                dr = e[2] if len(e) > 2 else 0.0          # optional: push the staged part radially away from the base (m)
                rz = R.from_euler("z", th)
                tp = self.to_world(parts[n].origin)
                q = quat_wxyz(rz * self.chair_rot)
                xy = rz.apply(np.r_[tp[:2], 0.0])[:2]
                xy = xy * (1.0 + dr / max(np.linalg.norm(xy), 1e-6))
                pos = np.r_[xy, tp[2]]
                self.staging[n] = (pos, q)
                low = part_corners(parts[n], pos, rot_of(q))[:, 2].min()
                self.staging_table[n] = ("orbit", th, 0.0, 0.0, max(low - 0.0003, 0.0))
        self.nest_offset = {n: np.zeros(3) for n in ["LF", "RF", "LR", "RR"]}   # fixture error (m, world); default none
        spec = mujoco.MjSpec()
        spec.option.timestep = timestep
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 5
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        spec.visual.quality.shadowsize = 4096
        self._world(spec)
        self._robot(spec)
        self._nests(spec)
        self._parts(spec)
        self._racks(spec)
        self.spec = spec
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ids()
        self.reset()

    # ---------------- build ----------------
    def _world(self, spec):
        wb = spec.worldbody
        tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                               rgb1=[0.86, 0.87, 0.86], rgb2=[0.80, 0.81, 0.80], width=512, height=512)
        spec.add_material(name="floor", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.05)
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="floor")
        wb.add_light(pos=[0.3, -0.6, 2.5], dir=[-0.1, 0.3, -1], diffuse=[0.75, 0.75, 0.75], castshadow=True)
        wb.add_light(pos=[-1.5, 1.5, 2.0], dir=[0.6, -0.5, -0.7], diffuse=[0.35, 0.35, 0.35], castshadow=False)
        wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.12, 0.12, PEDESTAL_H / 2],
                    pos=[0, 0, PEDESTAL_H / 2], rgba=[0.25, 0.27, 0.29, 1])
        wb.add_camera(name="wide", pos=[1.55, -1.25, 1.35], xyaxes=[0.63, 0.78, 0, -0.38, 0.31, 0.87])
        wb.add_camera(name="front", pos=[0.0, -1.35, 1.05], xyaxes=[1, 0, 0, 0, 0.45, 0.89])
        wb.add_camera(name="chair", pos=[0.75, -0.15, 0.85], xyaxes=[0.55, 0.83, 0, -0.45, 0.3, 0.84])

    def _racks(self, spec):
        """Floor racks for parts staged on edge: a base plate under the lowest region + two low side walls."""
        self.rack_parts = []
        table = self.staging_table
        for n, (pos, q) in self.staging.items():
            hgt = rack_height(table[n])
            if hgt <= 0:
                continue                                  # plain floor staging
            self.rack_parts.append(n)
            pts = part_corners(self.parts[n], pos, rot_of(q))
            fwd = rot_of(q).apply([1.0, 0, 0]); yaw = np.arctan2(fwd[1], fwd[0])     # rack aligned with the part
            rz = R.from_euler("z", yaw)
            pts_l = rz.inv().apply(pts - np.r_[pos[:2], 0])
            low = pts_l[pts_l[:, 2] < pts_l[:, 2].min() + 0.03]
            lo, hi = low[:, :2].min(0), low[:, :2].max(0)
            body = spec.worldbody.add_body(name=f"rack_{n}", pos=[pos[0], pos[1], 0.0], quat=quat_wxyz(rz).tolist())
            thin = int(np.argmin(hi - lo)); long_ = 1 - thin
            c = (lo + hi) / 2; ext = hi - lo
            size = np.zeros(3); size[long_] = ext[long_] / 2; size[thin] = ext[thin] / 2 + 0.03; size[2] = hgt / 2
            body.add_geom(name=f"rack_{n}_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=size.tolist(),
                          pos=[c[0], c[1], hgt / 2], rgba=[0.42, 0.44, 0.47, 1])
            RACK_TOP = hgt
            # side walls hug the thin direction (1 mm clearance), 28 mm tall; a flat-lying panel needs none
            flat = self.parts[n].kind == "seat"
            for sgn in (() if flat else (-1, 1)):
                wc_ = c.copy(); wc_[thin] += sgn * (ext[thin] / 2 + 0.001 + 0.006)
                wsize = np.zeros(3); wsize[long_] = ext[long_] / 2 + 0.02; wsize[thin] = 0.006; wsize[2] = 0.014
                body.add_geom(name=f"rack_{n}_wall{sgn}", type=mujoco.mjtGeom.mjGEOM_BOX, size=wsize.tolist(),
                              pos=[wc_[0], wc_[1], RACK_TOP + 0.014], rgba=[0.42, 0.44, 0.47, 1])

    def _robot(self, spec):
        arm = mujoco.MjSpec.from_file(str(UR5E))
        for b in arm.bodies:
            b.gravcomp = 1.0
        wrist3 = arm.body("wrist_3_link")
        site = arm.site("attachment_site")
        palm = wrist3.add_body(name="palm", pos=site.pos, quat=site.quat, gravcomp=1.0)
        palm.add_site(name="ft_site", size=[0.005] * 3)
        palm.add_geom(name="palm_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in PALM],
                      pos=[0, 0, PALM[2] / 2], rgba=[0.15, 0.15, 0.17, 1], mass=0.6)
        palm.add_site(name="tcp", pos=[0, 0, TCP_Z], size=[0.004] * 3, rgba=[1, 0, 0, 1])
        for side, sgn in (("l", 1), ("r", -1)):
            f = palm.add_body(name=f"finger_{side}", pos=[0, 0, PALM[2]], gravcomp=1.0)
            f.add_joint(name=f"finger_{side}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, sgn, 0],
                        range=[0.0, 0.056], damping=20, armature=0.01)
            f.add_geom(name=f"finger_{side}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[FINGER[0] / 2, FINGER[1] / 2, FINGER[2] / 2],
                       pos=[0, sgn * FINGER[1] / 2, FINGER[2] / 2], rgba=[0.30, 0.31, 0.33, 1],
                       friction=[1.2, 0.02, 0.001], condim=4, mass=0.05)
            arm.add_actuator(name=f"finger_{side}", target=f"finger_{side}_joint", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                             gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                             gainprm=[800] + [0] * 9, biasprm=[0, -800, -40] + [0] * 7, ctrlrange=[0, 0.056],
                             forcerange=[-80, 80])
        arm.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft_site")
        arm.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft_site")
        mount = spec.worldbody.add_frame(pos=[0, 0, PEDESTAL_H])
        spec.attach(arm, frame=mount, prefix="")

    def _nests(self, spec):
        self.nest_names = []
        for n in ["LF", "RF", "LR", "RR"]:
            s = self.parts[n]
            pos = self.to_world(s.origin)
            b = spec.worldbody.add_body(name=f"nest_{n}", pos=pos.tolist(), quat=self.chair_quat.tolist(), mocap=True)
            inner, wall, h = 0.021, 0.010, 0.030
            for i, (dx, dy, sx, sy) in enumerate([(inner + wall / 2, 0, wall / 2, inner + wall),
                                                  (-inner - wall / 2, 0, wall / 2, inner + wall),
                                                  (0, inner + wall / 2, inner, wall / 2),
                                                  (0, -inner - wall / 2, inner, wall / 2)]):
                b.add_geom(name=f"nest_{n}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, h / 2],
                           pos=[dx, dy, h / 2], rgba=[0.95, 0.55, 0.12, 1])
            self.nest_names.append(f"nest_{n}")

    def _parts(self, spec):
        for n, s in self.parts.items():
            pos, q = self.staging[n]
            body = spec.worldbody.add_body(name=n, pos=pos.tolist(), quat=list(q))
            body.add_freejoint(name=f"{n}_free")
            for i, bx in enumerate(s.boxes):
                role = bx["role"]
                rgba = COLORS.get(role, COLORS[s.color])
                body.add_geom(name=f"{n}:{i}:{role}", type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=[v / 2 for v in bx["size"]], pos=bx["center"], quat=bx["quat"],
                              density=DENSITY, rgba=list(rgba), friction=[0.6, 0.01, 0.001], condim=4,
                              solref=[0.004, 1.0], margin=0.0)
            spec.add_equality(name=f"kit_{n}", type=mujoco.mjtEq.mjEQ_WELD, name1=n, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name2="world", active=False, solref=[0.01, 1])
            spec.add_equality(name=f"grasp_{n}", type=mujoco.mjtEq.mjEQ_WELD, name1="palm", name2=n,
                              objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=[0.012, 1])   # compliant wrist (RCC-like)
            if s.kind == "leg":   # fixture clamp: holds the leg in its nest until final release
                spec.add_equality(name=f"clamp_{n}", type=mujoco.mjtEq.mjEQ_WELD, name1=n, name2="world",
                                  objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=[0.005, 1])
            for c in s.connectors:
                if c["host"] in self.parts:
                    spec.add_equality(name=f"latch_{c['name']}", type=mujoco.mjtEq.mjEQ_WELD, name1=c["host"], name2=n,
                                      objtype=mujoco.mjtObj.mjOBJ_BODY, active=False, solref=[0.005, 1])

    # ---------------- ids / helpers ----------------
    def _ids(self):
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_act = np.array([m.actuator(n).id for n in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]])
        self.finger_act = np.array([m.actuator("finger_l").id, m.actuator("finger_r").id])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(f"finger_{s}_joint").id] for s in "lr"])
        self.tcp = m.site("tcp").id
        self.palm = m.body("palm").id
        self.base = m.body("base").id
        self.body = {n: m.body(n).id for n in self.names}
        self.qadr = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.names}
        self.eq = {m.equality(i).name: i for i in range(m.neq)}
        self.geoms_of = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body[n]] for n in self.names}
        self.robot_bodies = set(range(m.body("base").id, m.body("finger_r").id + 1))
        self.nest_mocap = {n: m.body_mocapid[m.body(f"nest_{n}").id] for n in ["LF", "RF", "LR", "RR"]}
        self.force_adr = m.sensor_adr[m.sensor("ft_force").id]

    def reset(self, q=HOME_Q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = q
        self.data.ctrl[self.arm_act] = q
        self.data.ctrl[self.finger_act] = 0.04
        self.data.qpos[self.finger_qadr] = 0.04
        for n, mid in self.nest_mocap.items():
            self.data.mocap_pos[mid] = self.datum_pose(n)[0]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        for n in self.rack_parts:            # rack clamp holds panels standing on edge until they are grasped
            self.set_weld(f"kit_{n}", True)

    def target_pose(self, n):
        """Nominal installed pose of part n in world."""
        return self.to_world(self.parts[n].origin), self.chair_quat.copy()

    def datum_pose(self, n):
        """Actual installed pose a leg is clamped to: nominal + the (unknown to the robot) nest position error."""
        p, q = self.target_pose(n)
        return p + self.nest_offset.get(n, np.zeros(3)), q

    def set_nest_offsets(self, offsets):
        """Shift the leg nests (fixture error, m, world frame); the clamps then seat the legs at the shifted datum."""
        for n, off in offsets.items():
            self.nest_offset[n] = np.asarray(off, float)
            self.data.mocap_pos[self.nest_mocap[n]] = self.datum_pose(n)[0]
        mujoco.mj_forward(self.model, self.data)

    def to_world(self, p_chair):
        """Chair-frame point -> world."""
        return self.chair_origin + self.chair_rot.apply(np.asarray(p_chair, float))

    def part_pose(self, n):
        a = self.qadr[n]
        return self.data.qpos[a:a + 3].copy(), self.data.qpos[a + 3:a + 7].copy()

    def set_part_pose(self, n, pos, quat=(1, 0, 0, 0)):
        a = self.qadr[n]
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[self.model.jnt_dofadr[self.model.joint(f"{n}_free").id]:][:6] = 0

    def set_weld(self, name, active, body1_pose=None):
        """Activate a weld locking the current relative pose, or (body1_pose given) driving body1 to that world pose."""
        m, d = self.model, self.data
        i = self.eq[name]
        if active:
            b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
            p1, p2 = d.xpos[b1], d.xpos[b2]
            r1, r2 = rot_of(d.xquat[b1]), rot_of(d.xquat[b2])
            if body1_pose is not None:
                p1, r1 = np.asarray(body1_pose[0]), rot_of(np.asarray(body1_pose[1]))
            rel_p = r1.inv().apply(p2 - p1)
            rel_q = quat_wxyz(r1.inv() * r2)
            m.eq_data[i, 0:3] = 0
            m.eq_data[i, 3:6] = rel_p
            m.eq_data[i, 6:10] = rel_q
            m.eq_data[i, 10] = 1.0
        d.eq_active[i] = 1 if active else 0

    def tcp_pose(self):
        d = self.data
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def ft_force(self):
        """Force at the flange in world frame (sensor reports in site frame)."""
        f = self.data.sensordata[self.force_adr:self.force_adr + 3]
        return self.data.site_xmat[self.model.site("ft_site").id].reshape(3, 3) @ f

    def set_nests(self, raised: bool):
        for n, mid in self.nest_mocap.items():
            p = self.datum_pose(n)[0]
            self.data.mocap_pos[mid] = p + (np.zeros(3) if raised else np.array([0, 0, -0.035]))

    def penetrations(self, n, ignore=(), thresh=-2e-4):
        """Contacts of part n with anything not in `ignore` deeper than thresh."""
        m, d = self.model, self.data
        bid = self.body[n]
        ign = {self.body[x] for x in ignore if x in self.body}
        out = []
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            if other in ign:
                continue
            if c.dist < thresh:
                out.append((m.body(other).name, float(c.dist), m.geom(c.geom1).name, m.geom(c.geom2).name))
        return out
