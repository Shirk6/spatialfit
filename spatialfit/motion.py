"""Motion layer: exact MuJoCo DLS IK for Cartesian segments + cuRobo for collision-free free-space transfer."""
import copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .scene import ARM_JOINTS, TCP_Z, quat_wxyz, rot_of


def tool_down(yaw):
    """Tool z pointing down; palm closing axis (tool y) rotated by yaw about world z."""
    return (R.from_euler("z", yaw) * R.from_euler("x", np.pi)).as_matrix()


class DLSIK:
    """Damped least squares on the MuJoCo model (proposal Eq. 6), scratch MjData so the sim is untouched."""

    def __init__(self, wc):
        self.wc = wc
        self.m = wc.model
        self.d = mujoco.MjData(self.m)
        self.jac_p = np.zeros((3, self.m.nv))
        self.jac_r = np.zeros((3, self.m.nv))
        lo = self.m.jnt_range[[self.m.joint(j).id for j in ARM_JOINTS]]
        self.lo, self.hi = lo[:, 0], lo[:, 1]

    def fk(self, q):
        self.d.qpos[self.wc.arm_qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.wc.tcp].copy(), self.d.site_xmat[self.wc.tcp].reshape(3, 3).copy()

    def solve(self, pos, Rm, q_seed, iters=200, tol=(1e-4, 1e-3), lam=0.02, q_ref=None, eta=0.0):
        q = np.array(q_seed, float)
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = (R.from_matrix(Rm) * R.from_matrix(Rc).inv()).as_rotvec()
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(er) < tol[1]:
                return q, True
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jac_p, self.jac_r, self.wc.tcp)
            J = np.vstack([self.jac_p[:, self.wc.arm_dadr] / 0.1, self.jac_r[:, self.wc.arm_dadr]])
            e = np.concatenate([ep / 0.1, er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            if q_ref is not None:
                dq -= eta * (q - q_ref)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, Rc = self.fk(q)
        ok = np.linalg.norm(pos - p) < 5 * tol[0]
        return q, ok


class CuroboPlanner:
    """cuRobo MotionGen with a UR5e + gripper sphere model; world = cuboids synced from MuJoCo."""

    def __init__(self, wc, cache_obb=160):
        import torch  # noqa
        from curobo.geom.types import WorldConfig, Cuboid
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        self.Cuboid, self.WorldConfig = Cuboid, WorldConfig
        self.wc = wc
        self.t = TensorDeviceType()
        cfg = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))["robot_cfg"]
        k = cfg["kinematics"]
        k["extra_links"] = {
            "gripper_link": dict(parent_link_name="tool0", link_name="gripper_link", fixed_transform=[0, 0, 0, 1, 0, 0, 0],
                                 joint_type="FIXED", joint_name="gripper_joint"),
            "tcp_link": dict(parent_link_name="tool0", link_name="tcp_link", fixed_transform=[0, 0, TCP_Z, 1, 0, 0, 0],
                             joint_type="FIXED", joint_name="tcp_joint"),
            "attached_object": dict(parent_link_name="tcp_link", link_name="attached_object",
                                    fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_type="FIXED", joint_name="attach_joint"),
        }
        k["ee_link"] = "tcp_link"
        k["collision_link_names"] = k["collision_link_names"] + ["gripper_link", "attached_object"]
        k["collision_spheres"]["gripper_link"] = (
            [{"center": [0, y, 0.02], "radius": 0.021} for y in (-0.034, -0.011, 0.011, 0.034)] +   # palm 40x90x40 mm
            [{"center": [0, s * 0.036, z], "radius": 0.012} for s in (-1, 1) for z in (0.055, 0.072, 0.087)])
        k["extra_collision_spheres"] = {"attached_object": 200}
        ign = k["self_collision_ignore"]
        ign["wrist_2_link"] = ign.get("wrist_2_link", []) + ["gripper_link", "attached_object"]
        ign["wrist_3_link"] = ign.get("wrist_3_link", []) + ["gripper_link", "attached_object"]
        ign["wrist_1_link"] = ign.get("wrist_1_link", []) + ["gripper_link"]
        ign["gripper_link"] = ["attached_object", "tool0"]
        ign["wrist_1_link"] = ign["wrist_1_link"] + ["attached_object"]
        ign["attached_object"] = ["wrist_1_link", "wrist_2_link", "wrist_3_link", "gripper_link", "tool0"]
        k["self_collision_buffer"]["gripper_link"] = 0.0
        k["self_collision_buffer"]["attached_object"] = 0.0
        robot = RobotConfig.from_dict(cfg, self.t)
        self.world = self._world_cfg(exclude=set())
        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot, self.world, self.t, interpolation_dt=0.01, collision_cache={"obb": cache_obb},
            use_cuda_graph=False, collision_activation_distance=0.012, num_trajopt_seeds=8, num_graph_seeds=4)
        self.mg = MotionGen(mg_cfg)
        self.mg.warmup(enable_graph=True)

    # ---- frames ----
    def _base(self):
        d = self.wc.data
        return d.xpos[self.wc.base].copy(), rot_of(d.xquat[self.wc.base])

    def to_base(self, p, rot: R):
        b, rb = self._base()
        return rb.inv().apply(p - b), rb.inv() * rot

    def _cuboids_of(self, n, prefix=None, skip_roles=()):
        pos, q = self.wc.part_pose(n)
        rp = rot_of(q)
        out = []
        for i, bx in enumerate(self.wc.parts[n].coarse):
            if bx["role"] in skip_roles:
                continue
            c = pos + rp.apply(bx["center"])
            r = rp * rot_of(np.array(bx["quat"]))
            pb, rb = self.to_base(c, r)
            out.append(self.Cuboid(name=f"{prefix or n}_{i}", pose=pb.tolist() + quat_wxyz(rb).tolist(),
                                   dims=list(bx["size"])))
        return out

    def _world_cfg(self, exclude, grasp_target=None):
        m, d = self.wc.model, self.wc.data
        cubs = []
        for g in range(m.ngeom):
            bid = m.geom_bodyid[g]
            name = m.geom(g).name
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or bid in self.wc.robot_bodies:
                continue
            if bid == 0 or name.startswith("nest_") or name.startswith("rack_"):
                pb, rb = self.to_base(d.geom_xpos[g].copy(), R.from_matrix(d.geom_xmat[g].reshape(3, 3)))
                cubs.append(self.Cuboid(name=f"static_{name}", pose=pb.tolist() + quat_wxyz(rb).tolist(),
                                        dims=(2 * m.geom_size[g]).tolist()))
        pb, rb = self.to_base(np.array([0, 0, -0.062]), R.identity())   # floor (top at -12 mm: fingers may skim it)
        cubs.append(self.Cuboid(name="floor", pose=pb.tolist() + quat_wxyz(rb).tolist(), dims=[4, 4, 0.1]))
        for n in self.wc.names:
            if n == grasp_target:
                cubs += self._cuboids_of(n, skip_roles=("tab",))   # keep the part as obstacle, except its handle
            elif n not in exclude:
                cubs += self._cuboids_of(n)
        return self.WorldConfig(cuboid=cubs)

    def update_world(self, exclude=(), grasp_target=None):
        self.mg.update_world(self._world_cfg(set(exclude), grasp_target))

    def attach(self, n, q):
        from curobo.types.robot import JointState
        js = JointState.from_position(self.t.to_device([list(q)]), joint_names=ARM_JOINTS)
        from curobo.geom.sphere_fit import SphereFitType
        return self.mg.attach_external_objects_to_robot(js, self._cuboids_of(n, prefix=f"att_{n}"),
                                                        surface_sphere_radius=0.012,
                                                        sphere_fit_type=SphereFitType.SAMPLE_SURFACE)

    def detach(self):
        self.mg.detach_object_from_robot()

    def plan_js(self, q0, q1, time_dilation=0.6):
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        s = JointState.from_position(self.t.to_device([list(q0)]), joint_names=ARM_JOINTS)
        g = JointState.from_position(self.t.to_device([list(q1)]), joint_names=ARM_JOINTS)
        res = self.mg.plan_single_js(s, g, MotionGenPlanConfig(max_attempts=10, enable_graph_attempt=1, timeout=8.0, time_dilation_factor=time_dilation))
        if not res.success.item():
            return None, str(res.status)
        traj = res.get_interpolated_plan().position.cpu().numpy()
        return traj, "ok"

    def ik_batch(self, poses_world):
        """poses_world: list of (pos, Rm). Returns success mask and solutions (cuRobo kinematics)."""
        from curobo.types.math import Pose
        ps, qs = [], []
        for p, Rm in poses_world:
            pb, rb = self.to_base(np.asarray(p), R.from_matrix(Rm))
            ps.append(pb); qs.append(quat_wxyz(rb))
        pose = Pose(self.t.to_device(np.array(ps)), self.t.to_device(np.array(qs)))
        r = self.mg.ik_solver.solve_batch(pose)
        return r.success.view(-1).cpu().numpy(), r.solution.view(len(ps), -1).cpu().numpy()

    def collision_free(self, q):
        from curobo.types.robot import JointState
        js = JointState.from_position(self.t.to_device([list(q)]), joint_names=ARM_JOINTS)
        st = self.mg.check_constraints(js)
        return bool(st.feasible.item())
