"""Closed-loop executor: observe -> plan -> grasp -> verify -> transfer -> align -> admittance insert -> latch -> release."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .scene import quat_wxyz, rot_of, HOME_Q
from .motion import tool_down  # noqa: F401

HIGH_H = 0.22   # transfers end this high above the insertion pose, then descend vertically


class Recorder:
    def __init__(self, wc, fps=30):
        self.wc, self.dt = wc, 1.0 / fps
        self.next_t = 0.0
        self.qpos, self.mocap, self.xpos, self.xquat, self.hud = [], [], [], [], []
        self.events = []
        self.state = dict(step=0, total=0, part="", phase="init", force=0.0, retry=0, note="")

    def set(self, **kw):
        self.state.update(kw)

    def event(self, kind, **kw):
        e = dict(t=round(float(self.wc.data.time), 3), kind=kind, **kw)
        self.events.append(e)
        print(f"[{e['t']:7.2f}s] {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    def tick(self, force=False):
        d = self.wc.data
        if d.time + 1e-9 < self.next_t and not force:
            return
        self.next_t += self.dt
        self.qpos.append(d.qpos.copy()); self.mocap.append(d.mocap_pos.copy())
        self.xpos.append(d.xpos.copy()); self.xquat.append(d.xquat.copy())
        self.hud.append(dict(self.state, t=round(float(d.time), 3)))

    def save(self, path):
        m = self.wc.model
        np.savez_compressed(path, qpos=np.array(self.qpos), mocap=np.array(self.mocap), xpos=np.array(self.xpos),
                            xquat=np.array(self.xquat), fps=1 / self.dt,
                            body_names=np.array([m.body(i).name for i in range(m.nbody)]))
        with open(str(path).replace(".npz", "_hud.json"), "w") as f:
            json.dump(dict(hud=self.hud, events=self.events), f)


class Executor:
    def __init__(self, wc, ik, planner, rec, rng, obs_sigma=0.0003, bias=None, max_retries=3,
                 perceive=True, force_ctrl=True, nominal=None):
        """perceive=False / force_ctrl=False are the open-loop ablations (experiment E3):
        no perception -> every pose is the nominal one (`nominal`: part -> (pos, quat) before any perturbation);
        no force control -> position-controlled insertion to the nominal depth, no jam detection or retry."""
        self.wc, self.ik, self.cp, self.rec, self.rng = wc, ik, planner, rec, rng
        self.m, self.d = wc.model, wc.data
        self.obs_sigma = obs_sigma
        self.perceive, self.force_ctrl = perceive, force_ctrl
        self.nominal = {n: (np.asarray(p, float), rot_of(q)) for n, (p, q) in (nominal or {}).items()}
        self.part_stats = {}                # per-part insertion stats: attempts, peak force
        self._peak = 0.0
        self.bias = dict(bias or {})       # part -> xyz bias (m) injected on its first host observation
        self.max_retries = max_retries
        self.q_cmd = self.d.qpos[wc.arm_qadr].copy()
        self._q_prev = self.q_cmd.copy()
        self._qd_f = np.zeros(6)
        self.installed = []
        self.log = []
        self.debug = False
        self.carry = None
        self.engage_h = 0.042   # fast approach stops here; admittance takes over

    # ---------- low level ----------
    def step(self, n=1):
        # servo is kp*(ctrl-q) - kd*qdot; adding tau*qdot_cmd (tau = kd/kp) gives PD tracking with velocity feed-forward
        tau = -self.m.actuator_biasprm[self.wc.arm_act, 2] / self.m.actuator_gainprm[self.wc.arm_act, 0]
        for _ in range(n):
            qd = (self.q_cmd - self._q_prev) / self.m.opt.timestep
            self._qd_f = 0.8 * self._qd_f + 0.2 * qd
            self._q_prev = self.q_cmd.copy()
            self.d.ctrl[self.wc.arm_act] = self.q_cmd + tau * self._qd_f
            mujoco.mj_step(self.m, self.d)
            self.rec.tick()

    def wait(self, sec):
        self.step(int(sec / self.m.opt.timestep))

    def q_now(self):
        return self.d.qpos[self.wc.arm_qadr].copy()

    def track(self, traj, dt=0.01):
        k = int(round(dt / self.m.opt.timestep))
        prev = self.q_cmd.copy()
        for q in traj:
            for i in range(k):
                self.q_cmd = prev + (q - prev) * (i + 1) / k
                self.step()
            prev = q
        self.q_cmd = traj[-1].copy()
        self.wait(0.15)

    def move_linear(self, pos, Rm, speed=0.06, settle=0.1):
        p0, R0 = self.ik.fk(self.q_cmd)
        dist = np.linalg.norm(pos - p0)
        n = max(2, int(dist / speed / 0.01))
        slerp_r = R.from_matrix([R0, Rm])
        from scipy.spatial.transform import Slerp
        sl = Slerp([0, 1], slerp_r)
        q = self.q_cmd.copy()
        traj = []
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            q, ok = self.ik.solve(p0 + s * (pos - p0), sl([s]).as_matrix()[0], q)
            traj.append(q.copy())
        self.track(np.array(traj))
        self.wait(settle)

    def gripper(self, width, sec=0.35):
        self.d.ctrl[self.wc.finger_act] = np.clip(width / 2, 0, 0.056)
        self.wait(sec)

    def flange_force_z(self):
        return float(self.wc.ft_force()[2])

    # ---------- perception (ground truth + noise, optional bias) ----------
    def observe(self, n, bias=None):
        if not self.perceive:               # open loop: believe the nominal pose
            p, r = self.nominal[n]
            return p.copy(), r
        p, q = self.wc.part_pose(n)
        p = p + self.rng.normal(0, self.obs_sigma, 3) * np.array([1, 1, 0.5])
        yaw_noise = self.rng.normal(0, np.radians(0.15))
        r = R.from_euler("z", yaw_noise) * rot_of(q)
        if bias is not None:
            p = p + bias
        return p, r

    def _planning_qpos(self, d2):
        """Scene the planner believes in: the live state, but without perception every part sits at its nominal pose."""
        d2.qpos[:] = self.d.qpos
        if not self.perceive:
            for n, (p, r) in self.nominal.items():
                if self.carry is not None and n == self.carry[0]:
                    continue                    # the carried part follows the hand (set by the caller)
                a = self.wc.qadr[n]
                d2.qpos[a:a + 3] = p
                d2.qpos[a + 3:a + 7] = quat_wxyz(r)

    # ---------- geometry helpers ----------
    def grasp_world(self, pos, rot, s, g=None):
        """World TCP position and the two symmetric tool orientations (jaw flip) for grasp g at part pose (pos, rot)."""
        g = g or s.grasp
        tcp = pos + rot.apply(g["tcp"])
        Rg = rot.as_matrix() @ np.array(g["R"])
        return tcp, [Rg, Rg @ np.diag([-1.0, -1.0, 1.0])]

    def scene_clear(self, q, margin=0.001):
        """MuJoCo check of robot + carried part at arm config q (part follows the measured in-hand transform)."""
        if self.carry is None:
            return self.robot_clear(q)
        n, p_in, r_in = self.carry[:3]
        d2 = self.ik.d
        self._planning_qpos(d2)
        d2.qpos[self.wc.arm_qadr] = q
        if len(self.carry) > 3:
            d2.qpos[self.wc.finger_qadr] = self.carry[3] / 2       # fingers closed on the part
        mujoco.mj_kinematics(self.m, d2)
        tp, tR = d2.site_xpos[self.wc.tcp].copy(), R.from_matrix(d2.site_xmat[self.wc.tcp].reshape(3, 3))
        a = self.wc.qadr[n]
        d2.qpos[a:a + 3] = tp + tR.apply(p_in)
        d2.qpos[a + 3:a + 7] = quat_wxyz(tR * r_in)
        mujoco.mj_forward(self.m, d2)
        rb, pb = self.wc.robot_bodies, self.wc.body[n]
        hand = {self.wc.palm, self.m.body("finger_l").id, self.m.body("finger_r").id}
        for c in d2.contact[:d2.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 in rb and b2 in rb:                     # robot self-collision
                if c.dist < 0.0:
                    return False
                continue
            if {b1, b2} & {pb} and ({b1, b2} - {pb}) <= hand:
                continue                                  # the hand holding the part
            if {b1, b2} & {pb} and ({b1, b2} - {pb}) & rb:
                if c.dist < margin:
                    return False                          # arm links hitting the carried part
                continue
            inv1, inv2 = (b1 in rb or b1 == pb), (b2 in rb or b2 == pb)
            if inv1 == inv2:            # scene-scene
                continue
            if c.dist < margin:
                return False
        return True

    def robot_clear(self, q, allow_part=None, carrying=False, finger_width=None):
        """MuJoCo check: robot at q (current scene) has no contact except fingers/palm touching `allow_part`."""
        d2 = self.ik.d
        self._planning_qpos(d2)
        d2.qpos[self.wc.arm_qadr] = q
        if finger_width is not None:
            d2.qpos[self.wc.finger_qadr] = finger_width / 2
        mujoco.mj_forward(self.m, d2)
        rb = self.wc.robot_bodies
        allow = self.wc.body.get(allow_part, -1)
        for c in d2.contact[:d2.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 in rb and b2 in rb:
                if c.dist < 0.0:
                    return False                          # robot self-collision
                continue
            if (b1 in rb) == (b2 in rb):
                continue
            other = b2 if b1 in rb else b1
            if other == allow and (carrying or c.dist > -0.002):
                continue
            if c.dist < 0.001:
                return False
        return True

    @staticmethod
    def wrap_near(q, ref):
        """Shift revolute joints with +-2pi range by 2pi so they stay close to ref (shorter, smoother motions)."""
        q = q.copy()
        for j in (0, 3, 4, 5):
            while q[j] - ref[j] > np.pi and q[j] - 2 * np.pi >= -2 * np.pi:
                q[j] -= 2 * np.pi
            while q[j] - ref[j] < -np.pi and q[j] + 2 * np.pi <= 2 * np.pi:
                q[j] += 2 * np.pi
        return q

    def best_ik(self, tcp, Rs, offset=0.10, seed=None, check=None, all_solutions=False):
        """IK for the pre-pose (tcp backed off `offset` along -tool z). check=(part, [offsets]) also requires a
        collision-free robot at those offsets (e.g. half-way and at the grasp). Returns (q_pre, R)."""
        seed = self.q_cmd if seed is None else seed
        best, sols = None, []
        seeds = [seed.copy()] + [np.r_[pan, HOME_Q[1:]] for pan in np.linspace(-np.pi, np.pi, 9)]
        for Rm in Rs:
            back = -Rm[:, 2]
            for sd in seeds:
                q, ok = self.ik.solve(tcp + back * offset, Rm, sd)
                if ok and check is not None:
                    part, offs = check[:2]
                    fw = check[2] if len(check) > 2 else None
                    qq = q.copy()
                    for off in offs:
                        qq, ok2 = self.ik.solve(tcp + back * off, Rm, qq)
                        if not (ok2 and self.robot_clear(qq, part, finger_width=None if fw is None else fw + 0.015)):
                            ok = False          # approach with the gripper opened as it will be
                            break
                    if ok and fw is not None and not self.robot_clear(qq, part, finger_width=fw):
                        ok = False          # closed fingers would collide with a feature of the part

                if ok:
                    q = self.wrap_near(q, seed)
                if ok:
                    cost = np.abs(q - seed).sum() + 2.0 * abs(q[5] - seed[5])
                    if self.cp is not None and not self.cp.collision_free(q):
                        cost += 100.0   # cuRobo sphere model flags it: keep only as a last resort
                    sols.append((cost, q, Rm))
                    if best is None or cost < best[0]:
                        best = (cost, q, Rm)
        if all_solutions:
            out, seen = [], []
            for c, q, Rm in sorted(sols, key=lambda x: x[0]):
                if all(np.abs(q - o).max() > 0.2 for o in seen):
                    out.append((q, Rm)); seen.append(q)
            return out
        return (None, None) if best is None else (best[1], best[2])

    def target_from_hosts(self, n, bias=None):
        """Closed-loop target: part pose implied by *measured* host poses (locating end sets position, floating end yaw)."""
        s = self.wc.parts[n]
        hosts = [c for c in s.connectors if c["host"] in self.wc.parts]
        if not hosts:                                   # legs: fixture nest is the datum
            tp, tq = self.wc.target_pose(n)
            return tp, rot_of(tq)
        pts_meas, pts_nom = [], []
        for c in hosts:
            h = self.wc.parts[c["host"]]
            hp, hr = self.observe(c["host"], bias)
            # nominal connector point in host frame
            c_world_nom = s.origin + np.array(c["pos"])
            c_in_host = c_world_nom - h.origin
            pts_meas.append(hp + hr.apply(c_in_host))
            pts_nom.append(self.wc.to_world(c_world_nom))
        pts_meas, pts_nom = np.array(pts_meas), np.array(pts_nom)
        # planar rigid fit (x, y, yaw) + mean z
        cm, cn = pts_meas[:, :2].mean(0), pts_nom[:, :2].mean(0)
        H = (pts_nom[:, :2] - cn).T @ (pts_meas[:, :2] - cm)
        yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]) if len(hosts) > 1 else 0.0
        rz = R.from_euler("z", yaw)
        o_nom = self.wc.to_world(s.origin)
        o_xy = cm + rz.apply(np.r_[o_nom[:2] - cn, 0])[:2]
        loc = [i for i, c in enumerate(hosts) if c["type"] == "locating"]
        if loc:   # locating end defines the position; floating end only contributes yaw
            i = loc[0]
            o_xy = pts_meas[i, :2] + rz.apply(np.r_[o_nom[:2] - pts_nom[i, :2], 0])[:2]
        dzs = pts_meas[:, 2] - pts_nom[:, 2]
        dz = dzs.max() if all(c["type"] == "support" for c in hosts) else dzs.mean()   # a panel rests on its highest support
        return np.r_[o_xy, o_nom[2] + dz], rz * self.wc.chair_rot

    # ---------- skills ----------
    @staticmethod
    def equivalents(q, lim=6.18):
        """Same tool pose, different joint values: revolute joints with +-2pi range shifted by 2pi."""
        out = [q.copy()]
        for j in (4, 3, 0, 5):
            for k in (-1, 1):
                q2 = q.copy(); q2[j] += k * 2 * np.pi
                if abs(q2[j]) <= lim:
                    out.append(q2)
        return out

    def transfer_to(self, q_goal, exclude, grasp_target=None):
        self.cp.update_world(exclude=exclude, grasp_target=grasp_target)
        q0 = self.q_now()
        goals = sorted(self.equivalents(q_goal), key=lambda g: np.abs(g - q0).sum())
        traj = None
        for g in goals[:4]:                  # equivalent joint solutions: the wrist may unwind the other way round
            traj, status = self.cp.plan_js(q0, g)
            if traj is not None:
                break
        if traj is None:                     # via home, again trying equivalent goal solutions
            home = self.wrap_near(HOME_Q, q0)
            t1, s1 = self.cp.plan_js(q0, home)
            if t1 is not None:
                for g in sorted(self.equivalents(q_goal), key=lambda g: np.abs(g - home).sum())[:4]:
                    t2, s2 = self.cp.plan_js(home, g)
                    if t2 is not None:
                        traj, status = np.vstack([t1, t2]), "ok(via home)"
                        break
        if traj is None:   # fall back: go through a safe intermediate configuration above the cell
            q0 = self.q_now()
            vias = [self.wrap_near(HOME_Q, q0), np.r_[q_goal[0], HOME_Q[1:4], q_goal[4:]],
                    self.wrap_near(np.r_[q_goal[0], HOME_Q[1:]], q0), np.r_[q0[0], HOME_Q[1:]]]
            for mid in vias:
                t1, s1 = self.cp.plan_js(q0, mid)
                if t1 is None:
                    continue
                t2, s2 = self.cp.plan_js(mid, q_goal)
                if t2 is not None:
                    traj, status = np.vstack([t1, t2]), "ok(via)"
                    break
        if traj is None:   # fallback A: Cartesian lift -> traverse high -> lower, every sample checked in MuJoCo
            traj = self._cartesian_detour(q_goal)
            if traj is not None:
                status = "ok(detour)"
                self.rec.event("fallback_path", method="cartesian detour")
        if traj is None:   # last resort: collision-checked joint-space interpolation (direct, or via home)
            q0 = self.q_now()
            for path in ([q0, q_goal], [q0, self.wrap_near(HOME_Q, q0), q_goal]):
                pts = []
                for a, b in zip(path[:-1], path[1:]):
                    n_s = max(2, int(np.abs(b - a).max() / 0.6 / 0.01))       # <= 0.6 rad/s
                    for i in range(1, n_s + 1):
                        s_ = 0.5 - 0.5 * np.cos(np.pi * i / n_s)
                        pts.append(a + s_ * (b - a))
                if self.debug:
                    bad = [i for i, q in enumerate(pts[::4]) if not self.scene_clear(q, margin=0.003)]
                    if bad:
                        self.scene_clear(pts[::4][bad[0]], margin=0.003)
                        d2 = self.ik.d; m = self.m
                        who = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d2.contact[:d2.ncon] if c.dist < 0.003})
                        print(f"   fallback blocked at {bad[0]}/{len(pts[::4])} by {who[:6]}")
                if all(self.scene_clear(q, margin=0.003) for q in pts[::4]):
                    traj, status = np.array(pts), "ok(joint-interp)"
                    self.rec.event("fallback_path", method="joint interpolation", waypoints=len(path))
                    break
        if traj is None:
            self.rec.event("plan_fail", status=status, start_free=self.cp.collision_free(self.q_now()),
                           goal_free=self.cp.collision_free(q_goal), q_goal=np.round(q_goal, 2).tolist(), q_start=np.round(self.q_now(), 2).tolist())
            return False, status
        self.rec.set(phase="transfer (cuRobo)")
        self.track(traj)
        return True, "ok"

    def place_lookahead(self, n, tcp, Rm, pos, rot, width=0.0):
        """Would grasping n at (tcp, Rm) allow a collision-free, self-collision-free placement? (nominal target)"""
        rt = R.from_matrix(Rm)
        p_in, r_in = rt.inv().apply(pos - tcp), rt.inv() * rot          # part in TCP frame
        tpos, trot = self.wc.target_pose(n)
        r_tcp = rot_of(trot) * r_in.inv()
        p_tcp = tpos - r_tcp.apply(p_in)
        saved = self.carry
        self.carry = (n, p_in, r_in, width)
        ok_any = False
        for sd in [self.q_cmd] + [np.r_[pan, HOME_Q[1:]] for pan in np.linspace(-np.pi, np.pi, 9)]:
            q, ok = self.ik.solve(p_tcp + [0, 0, 0.07], r_tcp.as_matrix(), sd)
            if ok and self.scene_clear(q):
                q2, ok2 = self.ik.solve(p_tcp, r_tcp.as_matrix(), q)
                if ok2 and np.abs(q2 - q).max() < 0.5 and self.robot_clear(q2, n, finger_width=width + 0.008):   # seated + slightly opened
                    ok_any = True
                    break
        self.carry = saved
        return ok_any

    def _lift_path(self, q_start, z_top):
        """Straight-up Cartesian path from q_start to height z_top (fixed orientation); None on IK failure/jumps."""
        p0, R0 = self.ik.fk(q_start)
        n_s = max(3, int(abs(z_top - p0[2]) / 0.004))
        q, pts = q_start.copy(), []
        for i in range(1, n_s + 1):
            q, ok = self.ik.solve(np.r_[p0[:2], p0[2] + (z_top - p0[2]) * i / n_s], R0, q, iters=60)
            if not ok or (pts and np.abs(q - pts[-1]).max() > 0.15):
                return None
            pts.append(q.copy())
        return pts

    def _cartesian_detour(self, q_goal, z_travel=0.80):
        """Lift straight up, swing across at a safe height (joint interpolation), lower straight down onto q_goal."""
        q0 = self.q_now()
        zt = max(z_travel, self.ik.fk(q0)[0][2], self.ik.fk(q_goal)[0][2])
        up = self._lift_path(q0, zt)
        down = self._lift_path(q_goal, zt)
        if up is None or down is None:
            if self.debug:
                print("   detour: lift IK failed", up is None, down is None)
            return None
        qa, qb = up[-1], self.wrap_near(down[-1], up[-1])   # same physical pose, joint values kept near qa
        shift = qb - down[-1]
        down = [q + shift for q in down]
        k = max(2, int(np.abs(qb - qa).max() / 0.006))
        mid = [qa + (qb - qa) * (0.5 - 0.5 * np.cos(np.pi * i / k)) for i in range(1, k + 1)]
        pts = [q0] + up + mid + down[::-1] + [q_goal + shift]
        bad = [i for i, q in enumerate(pts[::3]) if not self.scene_clear(q, margin=0.003)]
        if bad:
            if self.debug:
                self.scene_clear(pts[::3][bad[0]], margin=0.003); d2 = self.ik.d; m = self.m
                who = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d2.contact[:d2.ncon]
                              if c.dist < 0.003 and ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & (self.wc.robot_bodies | ({self.wc.body[self.carry[0]]} if self.carry else set())))})
                print(f"   detour: collision at {bad[0]}/{len(pts[::3])}", who[:6])
            return None
        out = [pts[0]]
        for q in pts[1:]:
            k = max(1, int(np.abs(q - out[-1]).max() / 0.006))
            base = out[-1]
            out += [base + (q - base) * (i / k) for i in range(1, k + 1)]
        return np.array(out)

    def pick(self, n):
        s = self.wc.parts[n]
        self.rec.set(phase="observe")
        pos, rot = self.observe(n)
        self.cp.update_world(grasp_target=n)
        cands = []
        for g in s.meta.get("grasps", [s.grasp]):        # every grasp option of the part, in its current pose
            tcp_g, Rs = self.grasp_world(pos, rot, s, g)
            side = Rs[0][2, 2] > -0.7                        # horizontal approach (edge / side grasp)
            off = 0.06 if side else 0.10
            sols = self.best_ik(tcp_g, Rs, off, check=(n, [off / 2, 0.0], g["width"] + 0.001), all_solutions=True)
            if self.debug:
                print(f"   pick {n}: grasp tool z {np.round(Rs[0][:, 2], 2)} -> {len(sols)} IK candidates")
            for q, Rm in sols:
                cands.append((q, Rm, tcp_g, g))
        la = {}
        for c in cands:                      # look ahead: prefer grasps that also admit a clean placement
            key = (id(c[3]), round(float(c[1][0, 1]), 3), round(float(c[1][1, 1]), 3), round(float(c[1][2, 1]), 3))
            if key not in la:
                la[key] = self.place_lookahead(n, c[2], c[1], pos, rot, c[3]["width"])
        cands.sort(key=lambda c: np.abs(c[0] - self.q_cmd).sum() +
                   (0.0 if la[(id(c[3]), round(float(c[1][0, 1]), 3), round(float(c[1][1, 1]), 3), round(float(c[1][2, 1]), 3))] else 1000.0))
        q_pre, Rg = (cands[0][0], cands[0][1]) if cands else (None, None)
        tcp = cands[0][2] if cands else None
        if q_pre is None and self.debug:
            tcp, Rs = self.grasp_world(pos, rot, s)
            for Rm in Rs:
                for pan in np.linspace(-np.pi, np.pi, 9):
                    q, ok = self.ik.solve(tcp - Rm[:, 2] * 0.10, Rm, np.r_[pan, HOME_Q[1:]])
                    if not ok:
                        continue
                    r = [self.robot_clear(q, n), self.cp.collision_free(q)]
                    qq = q
                    for off in (0.05, 0.0):
                        qq, ok2 = self.ik.solve(tcp - Rm[:, 2] * off, Rm, qq)
                        r.append((ok2, self.robot_clear(qq, n)))
                    print("   pick-dbg", n, round(pan, 2), r)
        if q_pre is None:
            return False, "UNREACHABLE"
        self.gripper(max(gg["width"] for gg in s.meta.get("grasps", [s.grasp])) + 0.016, 0.05)
        tries = []
        for q, Rm, tcp_g, g in cands[:4]:    # a few IK branches, each also with the base rotating the other way round
            tries.append((q, Rm, tcp_g, g))
            for k in (-1, 1):
                q2 = q.copy(); q2[0] += k * 2 * np.pi
                if abs(q2[0]) <= 2 * np.pi - 0.05:
                    tries.append((q2, Rm, tcp_g, g))
        ok = False
        for q_pre, Rg, tcp, g in tries[:8]:
            ok, st = self.transfer_to(q_pre, exclude=[], grasp_target=n)
            if ok:
                break
        if not ok:
            return False, "PLAN_FAIL"
        self.grasp_used = g
        rg = R.from_matrix(Rg)
        self.assumed_in_hand = (rg.inv().apply(pos - tcp), rg.inv() * rot)   # part in TCP frame as believed at grasp
        self.rec.set(phase="approach")
        self.move_linear(tcp, Rg, speed=0.05)
        self.rec.set(phase="grasp")
        self.gripper(max(g["width"] - 0.004, 0.0), 0.5)
        if self._finger_contacts(n) < 2:
            if self.debug:
                m = self.m
                print("   grasp miss: tcp err mm", np.round((self.wc.tcp_pose()[0] - tcp) * 1000, 1), "tool z", np.round(Rg[:, 2], 2),
                      "fingers", np.round(self.d.qpos[self.wc.finger_qadr], 4),
                      "finger contacts", sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in self.d.contact[:self.d.ncon] if 'finger' in m.geom(c.geom1).name + m.geom(c.geom2).name})[:6])
            return False, "GRASP_MISS"
        self.wc.set_weld(f"grasp_{n}", True)
        self.wc.set_weld(f"kit_{n}", False)             # rack clamp opens
        self.m.body_gravcomp[self.wc.body[n]] = 1.0      # payload feed-forward compensation
        rel0 = self._rel_palm(n)
        self.rec.set(phase="lift + verify")
        p_now, R_now = self.ik.fk(self.q_cmd)
        self.move_linear(p_now + [0, 0, 0.12], R_now, speed=0.06)
        drift = np.linalg.norm(self._rel_palm(n)[0] - rel0[0])
        if drift > 0.002 and self.perceive:     # needs an in-hand measurement
            return False, "SLIP"
        self.cp.attach(n, self.q_now())
        if self.perceive:
            tp, tR = self.wc.tcp_pose(); rt = R.from_matrix(tR)
            pp, pq = self.wc.part_pose(n)
            self.carry = (n, rt.inv().apply(pp - tp), rt.inv() * rot_of(pq), g["width"])
        else:
            self.carry = (n, *self.assumed_in_hand, g["width"])
        return True, "ok"

    def _finger_contacts(self, n):
        fl, fr = self.m.geom("finger_l_geom").id, self.m.geom("finger_r_geom").id
        geoms = set(self.wc.geoms_of[n])
        hit = set()
        for c in self.d.contact[:self.d.ncon]:
            for f in (fl, fr):
                if (c.geom1 == f and c.geom2 in geoms) or (c.geom2 == f and c.geom1 in geoms):
                    hit.add(f)
        return len(hit)

    def _rel_palm(self, n):
        d = self.d
        rp = rot_of(d.xquat[self.wc.palm])
        return rp.inv().apply(d.xpos[self.wc.body[n]] - d.xpos[self.wc.palm]), rp.inv() * rot_of(d.xquat[self.wc.body[n]])

    def _tcp_target(self, n, part_pos, part_rot):
        """TCP pose placing the part at (part_pos, part_rot), using the *measured* in-hand transform."""
        if self.perceive:
            mp, mr = self.observe(n)
            tp, tR = self.ik.fk(self.q_now())
            rt = R.from_matrix(tR)
            p_in_tcp = rt.inv().apply(mp - tp)
            r_in_tcp = rt.inv() * mr
        else:                               # open loop: the in-hand transform assumed when grasping
            p_in_tcp, r_in_tcp = self.assumed_in_hand
        r_tcp = part_rot * r_in_tcp.inv()
        p_tcp = part_pos - r_tcp.apply(p_in_tcp)
        return p_tcp, r_tcp.as_matrix()

    def place(self, n):
        s = self.wc.parts[n]
        first_bias = self.bias.pop(n, None)
        pre_h = 0.07
        self.rec.set(phase="observe hosts")
        tpos, trot = self.target_from_hosts(n, first_bias)
        p_tcp, R_tcp = self._tcp_target(n, tpos, trot)
        # choose an IK branch whose pre-insert configuration is collision-free in the cuRobo world (part attached)
        self.cp.update_world(exclude=[n])
        best = None
        seeds = [self.q_cmd.copy()] + [np.r_[pan, HOME_Q[1:]] for pan in np.linspace(-np.pi, np.pi, 9)]
        cands = []
        for sd in seeds:
            q, ok = self.ik.solve(p_tcp + [0, 0, pre_h], R_tcp, sd)
            if ok:
                q = self.wrap_near(q, self.q_cmd)
            if ok and self.scene_clear(q):
                q2, ok2 = self.ik.solve(p_tcp, R_tcp, q)   # seated pose reachable on the same branch
                if not (ok2 and np.abs(q2 - q).max() < 0.5):
                    continue
                for hh in (HIGH_H, 0.15, pre_h):                 # highest approach pose reachable on the same branch
                    qh, okh = self.ik.solve(p_tcp + [0, 0, hh], R_tcp, q)
                    if okh and np.abs(qh - q).max() < 0.8 and self.scene_clear(qh):
                        pen = 0.0 if self.cp.collision_free(qh) else 100.0
                        cands.append((np.abs(qh - self.q_cmd).sum() + pen + 3.0 * (HIGH_H - hh), qh))
                        best = min(cands, key=lambda c: c[0])
                        break
        if best is None:
            if self.debug:
                for sd in seeds:
                    q, ok = self.ik.solve(p_tcp + [0, 0, pre_h], R_tcp, sd)
                    clear = ok and self.robot_clear(q)
                    m = self.m; d2 = self.ik.d
                    bad = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d2.contact[:d2.ncon]
                                  if (m.geom_bodyid[c.geom1] in self.wc.robot_bodies) != (m.geom_bodyid[c.geom2] in self.wc.robot_bodies) and c.dist < 0.001})
                    q2, ok2 = self.ik.solve(p_tcp, R_tcp, q)
                    print("   seed", np.round(sd[0], 2), "ik", ok, "mj_clear", clear, bad[:3], "seated", ok2, np.abs(q2 - q).max().round(2), np.round(q, 2))
            return False, "UNREACHABLE"
        for _, q_high in sorted(cands, key=lambda c: c[0])[:3]:     # try the closest few branches
            ok, st = self.transfer_to(q_high, exclude=[n])       # free-space transfer to a pose high above the target
            if ok:
                break
        if not ok:
            return False, "PLAN_FAIL"
        self.rec.set(phase="descend to pre-insert")
        self.move_linear(p_tcp + [0, 0, pre_h], R_tcp, speed=0.06)
        stats = self.part_stats.setdefault(n, dict(attempts=0, peak_force=0.0))
        for attempt in range(self.max_retries + 1 if self.force_ctrl else 1):   # no force sensing -> no jam detection
            stats["attempts"] = attempt + 1
            self.rec.set(retry=attempt)
            if attempt > 0 or first_bias is None:     # closed loop: re-measure hosts and in-hand pose after the transfer
                self.rec.set(phase="re-observe" if attempt == 0 else "re-observe + replan")
                tpos, trot = self.target_from_hosts(n)
                p_tcp, R_tcp = self._tcp_target(n, tpos, trot)
                if attempt > 1:   # small search offset on later retries
                    ang = attempt * 2.1
                    p_tcp = p_tcp + 0.0006 * np.array([np.cos(ang), np.sin(ang), 0])
            self.rec.set(phase="align")
            self.move_linear(p_tcp + [0, 0, pre_h], R_tcp, speed=0.04, settle=0.2)
            if self.debug:
                qq, okq = self.ik.solve(p_tcp + [0, 0, pre_h], R_tcp, self.q_cmd)
                m = self.m
                rc = sorted({(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in self.d.contact[:self.d.ncon]
                             if ((m.geom_bodyid[c.geom1] in self.wc.robot_bodies) != (m.geom_bodyid[c.geom2] in self.wc.robot_bodies))})
                print("   align: ik ok", okq, "tcp err mm", np.round((self.wc.tcp_pose()[0] - (p_tcp + [0, 0, pre_h])) * 1000, 1),
                      "cmd err mm", np.round((self.ik.fk(self.q_cmd)[0] - (p_tcp + [0, 0, pre_h])) * 1000, 1), "robot contacts", rc[:5])
                print("   dq", np.round(self.q_cmd - self.q_now(), 3), "act F", np.round(self.d.actuator_force[self.wc.arm_act], 1),
                      "q", np.round(self.q_now(), 2), "gravcomp part", self.m.body_gravcomp[self.wc.body[n]])
                fa = self.m.jnt_dofadr[self.m.joint(f"{n}_free").id]
                print("   qfrc_constraint arm", np.round(self.d.qfrc_constraint[self.wc.arm_dadr], 1),
                      "part passive", np.round(self.d.qfrc_passive[fa:fa + 3], 1), "part bias", np.round(self.d.qfrc_bias[fa:fa + 3], 1),
                      "active eq", [self.m.equality(i).name for i in range(self.m.neq) if self.d.eq_active[i] and n in self.m.equality(i).name])
                dof = self.wc.arm_dadr[4]
                contrib = {}
                keys = {(int(self.d.efc_type[i]), int(self.d.efc_id[i])) for i in range(self.d.nefc)}
                res = np.zeros(self.m.nv)
                for key in keys:
                    f = np.array([self.d.efc_force[i] if (int(self.d.efc_type[i]), int(self.d.efc_id[i])) == key else 0.0
                                  for i in range(self.d.nefc)])
                    mujoco.mj_mulJacTVec(self.m, self.d, res, f)
                    contrib[key] = res[dof]
                top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
                print("   wrist_2 constraint torque by (type,id):", [(k, round(v, 1), self.m.equality(k[1]).name if k[0] == 0 else "") for k, v in top])
                bid = self.wc.body[n]
                print("   part contacts", sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist * 1000, 1)) for c in self.d.contact[:self.d.ncon]
                                               if bid in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})[:8])
            self.rec.set(phase="approach")
            self.move_linear(p_tcp + [0, 0, self.engage_h], R_tcp, speed=0.05, settle=0.1)
            if self.force_ctrl:
                result = self.insert(n, p_tcp, R_tcp, tpos)
            else:
                result = self.insert_position(n, p_tcp, R_tcp, tpos)
            stats["peak_force"] = max(stats["peak_force"], self._peak)
            if result == "SEATED":
                break
            self.rec.event("jam", part=n, attempt=attempt, cause=result)
            self.rec.set(phase="jam -> retreat", note="jam detected")
            p_now, _ = self.ik.fk(self.q_cmd)
            self.move_linear(p_now + [0, 0, 0.02], R_tcp, speed=0.03, settle=0.2)
            self.rec.set(note="")
        else:
            return False, "JAM"
        self.rec.set(phase="latch" if s.kind != "leg" else "fixture clamp")
        if s.kind == "leg":
            self.wc.set_weld(f"clamp_{n}", True, body1_pose=self.wc.datum_pose(n))    # clamp pushes leg onto its datum
        for c in s.connectors:
            if c["host"] in self.wc.parts:
                self.wc.set_weld(f"latch_{c['name']}", True)
        self.wait(0.1)
        self.rec.set(phase="release")
        self.wc.set_weld(f"grasp_{n}", False)
        self.m.body_gravcomp[self.wc.body[n]] = 0.0
        self.gripper(self.grasp_used["width"] + 0.008, 0.4)      # open just enough to slide off
        p_before = self.wc.part_pose(n)[0]
        self.wait(0.3)
        moved = np.linalg.norm(self.wc.part_pose(n)[0] - p_before)
        self.cp.detach()
        self.carry = None
        self.rec.set(phase="retreat")
        p_now, R_now = self.ik.fk(self.q_cmd)
        back = -R_now[:, 2]
        if back[2] < 0.7:   # side / edge grasp: back out along the tool axis first
            self.move_linear(p_now + back * 0.06, R_now, speed=0.06)
            p_now = p_now + back * 0.06
        self.move_linear(p_now + [0, 0, 0.08], R_now, speed=0.08)
        if moved > 0.001:
            return False, "RETENTION"
        return True, "ok"

    def insert(self, n, p_tcp, R_tcp, tpos, v_max=0.02, F_ref=25.0, F_max=90.0, dt=0.01):
        """Axial admittance (first-order form of Eq. 7): descent rate follows (F_ref - F_resist)."""
        self.rec.set(phase="insert (force control)")
        base = np.mean([self._fz() for _ in range(3)])
        z_cmd = self.ik.fk(self.q_cmd)[0][2]
        z_end = p_tcp[2] - 0.004
        hist = []
        t0 = self.d.time
        self._peak = 0.0
        while self.d.time - t0 < 8.0:
            F = self._fz() - base                    # resisting (upward) force on the tool, N
            self._peak = max(self._peak, F)
            self.rec.set(force=round(F, 2))
            v = np.clip(v_max * (F_ref - F) / F_ref, -v_max, v_max)
            z_cmd = max(z_cmd - v * dt, z_end)
            q, _ = self.ik.solve(np.r_[p_tcp[:2], z_cmd], R_tcp, self.q_cmd, iters=30)
            k = int(dt / self.m.opt.timestep)
            prev = self.q_cmd.copy()
            for i in range(k):
                self.q_cmd = prev + (q - prev) * (i + 1) / k
                self.step()
            pz = self.wc.part_pose(n)[0][2]
            hist.append(pz)
            if self.debug and len(hist) % 25 == 1:
                pp = self.wc.part_pose(n)[0]
                print(f"    ins t={self.d.time - t0:4.2f} F={F:6.1f} z_cmd-tcp={z_cmd - p_tcp[2]:+.4f} part-target={np.round((pp - tpos) * 1000, 2)} "
                      f"true_err={np.round((pp - self.wc.to_world(self.wc.parts[n].origin)) * 1000, 2)}")
            depth_err = pz - tpos[2]
            if depth_err < 0.0012 and abs(F) < F_max:
                self.wait(0.15)
                if self.wc.part_pose(n)[0][2] - tpos[2] < 0.0012:
                    self.rec.set(force=0.0)
                    return "SEATED"
            if F > F_max:
                return "JAM_FORCE"
            if len(hist) > 80 and hist[-80] - hist[-1] < 0.0003 and depth_err > 0.002:
                if self.debug:
                    bid = self.wc.body[n]
                    for c in self.d.contact[:self.d.ncon]:
                        g1, g2 = c.geom1, c.geom2
                        if bid in (self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]):
                            print("      contact", self.m.geom(g1).name, self.m.geom(g2).name, round(c.dist * 1000, 2), np.round(c.pos, 4))
                return "JAM_NO_PROGRESS"
        return "TIMEOUT"

    def insert_position(self, n, p_tcp, R_tcp, tpos, v=0.02, dt=0.01, preload=0.001):
        """Ablation without force feedback: drive the TCP straight to the nominal seated height (+1 mm preload) at
        the admittance law's top speed, then check the depth. The F/T reading is only logged, never used."""
        self.rec.set(phase="insert (position only)")
        base = np.mean([self._fz() for _ in range(3)])
        z_cmd = self.ik.fk(self.q_cmd)[0][2]
        z_end = p_tcp[2] - preload
        self._peak = 0.0
        k = int(dt / self.m.opt.timestep)
        settle = int(0.3 / dt)
        while z_cmd > z_end + 1e-9 or settle > 0:
            if z_cmd <= z_end + 1e-9:
                settle -= 1
            z_cmd = max(z_cmd - v * dt, z_end)
            q, _ = self.ik.solve(np.r_[p_tcp[:2], z_cmd], R_tcp, self.q_cmd, iters=30)
            prev = self.q_cmd.copy()
            for i in range(k):
                self.q_cmd = prev + (q - prev) * (i + 1) / k
                self.step()
            F = self._fz() - base
            self._peak = max(self._peak, F)
            self.rec.set(force=round(F, 2))
        self.rec.set(force=0.0)
        return "SEATED" if self.wc.part_pose(n)[0][2] - tpos[2] < 0.0012 else "JAM_POSITION"

    def _fz(self):
        f = self.wc.ft_force()
        return float(-f[2])   # sensor reports force of child on parent frame; sign set so contact push-back > 0

    # ---------- top level ----------
    def build(self, sequence):
        self.rec.set(total=len(sequence))
        for k, n in enumerate(sequence, 1):
            self.rec.set(step=k, part=n, retry=0)
            self.rec.event("start", part=n, step=k)
            ok, why = self.pick(n)
            if ok:
                ok, why = self.place(n)
            st = self.part_stats.get(n, {})
            self.log.append(dict(part=n, ok=ok, cause=why, t=round(float(self.d.time), 2),
                                 attempts=st.get("attempts", 0), peak_force=round(float(st.get("peak_force", 0.0)), 2)))
            self.rec.event("done" if ok else "fail", part=n, cause=why)
            if not ok:
                return False
            self.installed.append(n)
            tp, tq = self.wc.target_pose(n)
            self.nominal[n] = (tp, rot_of(tq))           # open loop believes the part sits exactly at its nominal pose
            self.home()
        return True

    def home(self, keep_part=True):
        """Return to the home configuration between parts (cuRobo, with the current scene as obstacles)."""
        self.rec.set(phase="return home")
        self.cp.update_world()
        for goal in (HOME_Q, self.wrap_near(HOME_Q, self.q_now())):
            traj, st = self.cp.plan_js(self.q_now(), goal)
            if traj is not None:
                self.track(traj)
                return True
        return False

    def final_release(self):
        self.rec.set(phase="fixture release", part="")
        for n in self.wc.names:
            if self.wc.parts[n].kind == "leg":
                self.wc.set_weld(f"clamp_{n}", False)
        self.wait(0.3)
        start = {n: self.wc.part_pose(n)[0] for n in self.wc.names}
        for i in range(60):
            for n, mid in self.wc.nest_mocap.items():
                p = self.wc.datum_pose(n)[0]
                self.d.mocap_pos[mid] = p + np.array([0, 0, -0.035 * (i + 1) / 60])
            self.wait(1 / 60)
        self.rec.set(phase="stability check")
        self.wait(2.0)
        disp = max(np.linalg.norm(self.wc.part_pose(n)[0] - start[n]) for n in self.wc.names)
        self.rec.event("final_stability", max_disp_mm=round(disp * 1000, 2))
        self.rec.set(phase="done", note=f"max displacement {disp*1000:.1f} mm")
        self.wait(1.0)
        return disp
