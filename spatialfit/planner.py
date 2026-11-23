"""Assembly sequence planner: precedence (from connectors) -> insertion-sweep clearance -> reachability -> stability.

Also counts how many of the 10! orders survive each filter (the 'funnel').
"""
import itertools, math, time
from functools import lru_cache
import numpy as np
import mujoco

SWEEP_HEIGHTS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.042, 0.05, 0.062, 0.075, 0.09, 0.11]
ENGAGE_H = 0.042
CLEAR_FREE = 0.005     # required clearance to installed parts while above the engage height
CLEAR_NEAR = 0.0012    # required clearance to non-host parts during the final insertion


class SequencePlanner:
    def __init__(self, wc, ik):
        self.wc, self.ik = wc, ik
        self.m, self.d = wc.model, mujoco.MjData(wc.model)
        self.names = wc.names
        self.parts = wc.parts
        self.far = {n: np.array([5.0 + 2 * i, 5.0, 1.0]) for i, n in enumerate(self.names)}
        self.clash = {}
        self.reasons = {}

    # ---------- pairwise insertion-sweep check ----------
    def _place(self, n, dz=0.0):
        a = self.wc.qadr[n]
        p = self.wc.to_world(self.parts[n].origin)
        self.d.qpos[a:a + 3] = p + [0, 0, dz]
        self.d.qpos[a + 3:a + 7] = self.wc.chair_quat

    def _min_dist(self, n, other, cutoff=0.02):
        m, d = self.m, self.d
        best = cutoff
        for g1 in self.wc.geoms_of[n]:
            for g2 in self.wc.geoms_of[other]:
                if np.linalg.norm(d.geom_xpos[g1] - d.geom_xpos[g2]) - m.geom_rbound[g1] - m.geom_rbound[g2] > best:
                    continue
                best = min(best, mujoco.mj_geomDistance(m, d, g1, g2, best, None))
        return best

    def sweep_clash(self, n, other):
        """True if inserting n vertically is blocked by an already-installed `other`."""
        key = (n, other)
        if key in self.clash:
            return self.clash[key]
        for k in self.names:
            a = self.wc.qadr[k]; self.d.qpos[a:a + 3] = self.far[k]; self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self._place(other)
        host = other in self.parts[n].requires
        res = False
        for h in SWEEP_HEIGHTS:
            self._place(n, h)
            mujoco.mj_kinematics(self.m, self.d)
            dist = self._min_dist(n, other)
            need = (0.001 if h > ENGAGE_H else 0.0) if host else (CLEAR_FREE if h > ENGAGE_H else CLEAR_NEAR)
            if dist < need - 1e-6:
                res = (h, dist)
                break
        self.clash[key] = res
        return res

    # ---------- reachability ----------
    def reachable(self, n):
        """Installed grasp pose and its pre-insert pose have IK solutions (either jaw flip)."""
        s = self.parts[n]
        from .scene import HOME_Q
        opts = []
        for g in s.meta.get("grasps", [s.grasp]):
            p = self.wc.to_world(s.origin + np.array(g["tcp"])); Rg = self.wc.chair_rot.as_matrix() @ np.array(g["R"])
            opts += [(p, Rg), (p, Rg @ np.diag([-1.0, -1.0, 1.0]))]
        for p, Rm in opts:
            for pan in np.linspace(-np.pi, np.pi, 9):
                q = np.r_[pan, HOME_Q[1:]]
                q, ok1 = self.ik.solve(p + [0, 0, 0.07], Rm, q)
                q, ok2 = self.ik.solve(p, Rm, q)
                if ok1 and ok2:
                    return True
        return False

    # ---------- stability of a sub-assembly (MuJoCo, latched joints; optional 3.6 N push on one part) ----------
    def stable(self, installed, fixture=True, sec=1.5, push=None):
        m = self.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.d.qpos
        for k in self.names:
            a = self.wc.qadr[k]
            if k in installed:
                d.qpos[a:a + 3] = self.wc.to_world(self.parts[k].origin); d.qpos[a + 3:a + 7] = self.wc.chair_quat
            else:
                d.qpos[a:a + 3] = self.far[k]; d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        d.qpos[self.wc.arm_qadr] = [0, -2.6, 2.4, -1.5, -1.57, 0]     # robot parked out of the way
        d.ctrl[self.wc.arm_act] = d.qpos[self.wc.arm_qadr]
        mujoco.mj_forward(m, d)
        eq_save = m.eq_data.copy()
        d.eq_active[:] = 0
        for k in self.names:
            if k not in installed:
                self._weld(m, d, f"kit_{k}")            # hold unused parts far away
        for k in installed:
            for c in self.parts[k].connectors:
                if c["host"] in installed:
                    self._weld(m, d, f"latch_{c['name']}")
            if fixture and self.parts[k].kind == "leg":
                self._weld(m, d, f"clamp_{k}")
        if not fixture:
            for mid in self.wc.nest_mocap.values():
                d.mocap_pos[mid] += [0, 0, -0.035]
        p0 = {k: d.xpos[self.wc.body[k]].copy() for k in installed}
        for i in range(int(sec / m.opt.timestep)):
            # small lateral disturbance (comparable to insertion contact forces) for the first 0.3 s
            if push is not None:
                d.xfrc_applied[self.wc.body[push], :3] = [3.0, 2.0, 0.0] if i * m.opt.timestep < 0.3 else 0.0
            mujoco.mj_step(m, d)
        d.xfrc_applied[:] = 0
        m.eq_data[:] = eq_save
        disp = max(np.linalg.norm(d.xpos[self.wc.body[k]] - p0[k]) for k in installed) if installed else 0.0
        return disp < 0.003, disp, d

    def _weld(self, m, d, name):
        from .scene import rot_of, quat_wxyz
        i = self.wc.eq[name]
        b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
        r1 = rot_of(d.xquat[b1])
        m.eq_data[i, 0:3] = 0
        m.eq_data[i, 3:6] = r1.inv().apply(d.xpos[b2] - d.xpos[b1])
        m.eq_data[i, 6:10] = quat_wxyz(r1.inv() * rot_of(d.xquat[b2]))
        m.eq_data[i, 10] = 1
        d.eq_active[i] = 1

    # ---------- search ----------
    def plan(self, verbose=True):
        t0 = time.time()
        names = self.names
        idx = {n: i for i, n in enumerate(names)}
        req = {n: [r for r in self.parts[n].requires] for n in names}
        reach = {n: self.reachable(n) for n in names}
        for n, o in itertools.permutations(names, 2):
            self.sweep_clash(n, o)
        full = (1 << len(names)) - 1

        def prec_ok(mask, n):
            return all(mask >> idx[r] & 1 for r in req[n])

        def geom_ok(mask, n):
            return not any(mask >> idx[o] & 1 and self.clash[(n, o)] for o in names if o != n)

        @lru_cache(None)
        def count(mask, level):
            if mask == full:
                return 1
            tot = 0
            for n in names:
                if mask >> idx[n] & 1:
                    continue
                if not prec_ok(mask, n):
                    continue
                if level >= 1 and not geom_ok(mask, n):
                    continue
                if level >= 2 and not reach[n]:
                    continue
                tot += count(mask | 1 << idx[n], level)
            return tot
        funnel = dict(all_orders=math.factorial(len(names)), precedence=count(0, 0),
                      geometric=count(0, 1), reachable=count(0, 2))

        # greedy-with-backtracking search ranked by travel between consecutive targets
        def tgt(n):
            return self.wc.to_world(self.parts[n].origin)

        def dfs(mask, seq):
            if mask == full:
                return seq
            cands = [n for n in names if not mask >> idx[n] & 1 and prec_ok(mask, n) and geom_ok(mask, n) and reach[n]]
            last = tgt(seq[-1]) if seq else np.array([0, 0, 0.0])
            cands.sort(key=lambda n: (len(req[n]) > 0, np.linalg.norm(tgt(n) - last)))
            for n in cands:
                r = dfs(mask | 1 << idx[n], seq + [n])
                if r:
                    return r
            return None
        seq = dfs(0, [])
        final_ok, final_disp, _ = self.stable(set(names), fixture=False, sec=2.0)
        funnel["final_stable_without_fixture"] = bool(final_ok)
        def descendants(n):
            out, stack = set(), [n]
            while stack:
                x = stack.pop()
                for y in names:
                    if x in req[y] and y not in out:
                        out.add(y); stack.append(y)
            return out
        rejected = []
        for (n, o), r in self.clash.items():
            if r and o not in descendants(n) and o not in req[n]:
                rejected.append(dict(part=n, blocked_by=o, height_mm=round(r[0] * 1000, 1), clearance_mm=round(r[1] * 1000, 2)))
        # self-support of each intermediate state of the chosen sequence without the fixture
        self_support = []
        for k in range(1, len(seq) + 1):
            ok, disp, _ = self.stable(set(seq[:k]), fixture=False, sec=1.5, push=seq[k - 1])
            self_support.append(dict(step=k, part=seq[k - 1], stable_without_fixture=bool(ok), max_disp_mm=round(disp * 1000, 1)))
        out = dict(sequence=seq, funnel=funnel, geometric_rejections=rejected, reachable=reach,
                   self_support=self_support, final_disp_mm=round(final_disp * 1000, 2), plan_time_s=round(time.time() - t0, 1))
        if verbose:
            print("funnel:", funnel); print("sequence:", " -> ".join(seq)); print("rejections:", rejected)
        return out
