"""Kinematics (spec §4.4): forward kinematics, couplings, loop closure, roles and mobility.

Joint values are native units at the API (deg for revolute, mm for prismatic). A part transform
T maps home-world -> current-world and is built by the product of exponentials over the tree,

    T_child(q) = T_parent(q) · M_j(q_j − home_j),

with M_j a rotation about joint j's home-world line or a translation along its axis. Couplings
act on deltas from home and are linear, so every joint value is an affine function of the
independent (non-coupled) ones: q = h + D·(u − h) + o. Loop closure solves the pin residuals
for the non-coupled, non-driven loop joints with a trust-region least-squares corrector and an
analytic Jacobian (screw-theory twists chained through D).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .assembly import Assembly, Joint, ModelError, _suggest
from .geom import rot_about_line, transform_points, translation

__all__ = ["Pose", "Kinematics", "RESIDUAL_TOL"]

RESIDUAL_TOL = 1e-4  # mm: a pose is ok when every residual row is within this
RANK_RTOL = 1e-9  # SVD rank tolerance relative to the largest singular value
_SOLVER_TOL = 1e-12  # least_squares xtol = ftol = gtol
_STEP_DEG = 10.0  # continuation: largest revolute driver step per corrector solve
_STEP_FRAC = 0.05  # continuation: largest prismatic driver step, as a fraction of L
_MAX_STEPS = 720
_PREDICT_RCOND = 1e-6  # pinv cutoff for the tangent predictor (skips near-singular directions)
_N_PERTURB, _PERTURB_DEG, _PERTURB_FRAC, _SEED = 3, 10.0, 0.05, 0  # generic-rank samples


@dataclass
class Pose:
    q: dict[str, float]  # every non-fixed joint, native units (couplings applied)
    transforms: dict[str, np.ndarray]  # every part: home-world -> current-world
    residual: float  # max |residual row| in mm after solve
    ok: bool  # residual <= 1e-4 mm
    limit_violations: list[tuple[str, float, tuple[float, float]]] = field(default_factory=list)  # coupled/passive


class Kinematics:
    """Kinematic model of a validated ``Assembly``."""

    def __init__(self, asm: Assembly):
        errors = asm.validate()
        if errors:
            raise ModelError(errors)
        self.asm = asm
        joints = asm.joints
        self.fixed: set[str] = {n for n, j in joints.items() if j.kind == "fixed"}
        self.coupled: set[str] = {c.driven for c in asm.couplings}
        self.loops: list[list[str]] = [asm._tree_path(p.a, p.b) for p in asm.pins]
        self.L: float = asm._char_length()

        self._names = [n for n, j in joints.items() if j.kind != "fixed"]
        self._idx = {n: i for i, n in enumerate(self._names)}
        self._rev = np.array([joints[n].kind == "revolute" for n in self._names], dtype=bool)
        self._home = np.array([joints[n].home for n in self._names], dtype=float)
        self._to_internal = np.where(self._rev, math.pi / 180.0, 1.0)  # native -> rad | mm
        # Column scale for rank/pinv: a revolute column per degree becomes per unit of arc length
        # L·θ, so revolute and prismatic columns are comparable (both mm per mm).
        self._col_scale = np.where(self._rev, self.L * math.pi / 180.0, 1.0)
        self._D, self._offset = self._coupling_map()

        self._parent: dict[str, Joint] = {j.child: j for j in joints.values()}
        self._chains: dict[str, list[Joint]] = {p: [joints[n] for n in asm._chain(p)] for p in asm.parts}
        self._loop_joints: set[str] = {n for loop in self.loops for n in loop}
        self._coupling_drivers: set[str] = {c.driver for c in asm.couplings}
        self._pin_parts = list(dict.fromkeys(x for p in asm.pins for x in (p.a, p.b)))
        self._pin_rows: list[slice] = []
        r = 0
        for p in asm.pins:
            k = 3 if p.axis is None else 6
            self._pin_rows.append(slice(r, r + k))
            r += k
        self._n_rows = r
        self._sample_jacs: list[np.ndarray] | None = None

    # ------------------------------------------------------------------------ joint values

    def _coupling_map(self) -> tuple[np.ndarray, np.ndarray]:
        """(D, o) with q = h + D·(u − h) + o; rows of coupled joints replace their own entry."""
        n = len(self._names)
        D, off = np.eye(n), np.zeros(n)
        by_driven = {c.driven: c for c in self.asm.couplings}
        done: set[str] = set()

        def resolve(name: str) -> None:  # drivers first; validate() guarantees no cycles
            if name in done or name not in by_driven:
                return
            c = by_driven[name]
            resolve(c.driver)
            j, k = self._idx[c.driven], self._idx[c.driver]
            D[j] = c.ratio * D[k]
            off[j] = c.ratio * off[k] + c.offset
            done.add(name)

        for name in by_driven:
            resolve(name)
        return D, off

    def _u_from(self, values: Mapping[str, float]) -> np.ndarray:
        """Home values overlaid with ``values`` (fixed joints ignored, unknown names -> KeyError)."""
        u = self._home.copy()
        for name, v in values.items():
            i = self._idx.get(name)
            if i is None:
                if name in self.fixed:
                    continue
                raise KeyError(f"unknown joint '{name}'{_suggest(name, self._names)}")
            u[i] = float(v)
        return u

    def _q(self, u: np.ndarray) -> np.ndarray:
        return self._home + self._D @ (u - self._home) + self._offset

    def expand(self, q_free: Mapping[str, float]) -> dict[str, float]:
        """All non-fixed joint values: missing = home, coupled joints recomputed from their drivers."""
        return dict(zip(self._names, self._q(self._u_from(q_free)).tolist()))

    # ------------------------------------------------------------------------ forward kinematics

    def _motion(self, j: Joint, q: np.ndarray) -> np.ndarray:
        if j.kind == "fixed":
            return np.eye(4)
        delta = q[self._idx[j.name]] - j.home
        if j.kind == "revolute":
            return rot_about_line(j.origin, j.axis, delta)
        return translation(delta * j.axis)

    def _transforms(self, q: np.ndarray, parts: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """Transforms of ``parts`` (default all) *and all their ancestors*, from full joint vector q."""
        T: dict[str, np.ndarray] = {}

        def get(p: str) -> np.ndarray:
            t = T.get(p)
            if t is None:
                j = self._parent.get(p)
                t = np.eye(4) if j is None else get(j.parent) @ self._motion(j, q)
                T[p] = t
            return t

        for p in self.asm.parts if parts is None else parts:
            get(p)
        return T

    def fk(self, q: Mapping[str, float]) -> dict[str, np.ndarray]:
        """Part transforms for joint values ``q`` (missing joints = home; couplings applied)."""
        T = self._transforms(self._q(self._u_from(q)))
        return {p: T[p] for p in self.asm.parts}

    def point(self, pose: Pose, part: str, p) -> np.ndarray:
        """World position at ``pose`` of the home-world point(s) ``p`` fixed to ``part``."""
        return transform_points(pose.transforms[part], p)

    # ------------------------------------------------------------------------ loop residual

    def _residual(self, q: np.ndarray) -> np.ndarray:
        """Per pin: T_a·p − T_b·p (mm), plus L·(R_a n − R_b n) for hinge pins."""
        r = np.zeros(self._n_rows)
        if not self._n_rows:
            return r
        T = self._transforms(q, self._pin_parts)
        for pin, rows in zip(self.asm.pins, self._pin_rows):
            Ta, Tb = T[pin.a], T[pin.b]
            dR = Ta[:3, :3] - Tb[:3, :3]
            r[rows.start:rows.start + 3] = dR @ pin.point + Ta[:3, 3] - Tb[:3, 3]
            if pin.axis is not None:
                r[rows.start + 3:rows.stop] = self.L * (dR @ pin.axis)
        return r

    def _jac_joint(self, q: np.ndarray) -> np.ndarray:
        """∂residual/∂q_j for every non-fixed joint as if independent, per rad | mm.

        A revolute joint moves a point x of its subtree at ω × (x − o) and a direction m at ω × m
        (ω, o = the joint's current world axis and origin); a prismatic joint moves points along
        its current world axis. Part a's chain enters with +, part b's with −.
        """
        J = np.zeros((self._n_rows, len(self._names)))
        if not self._n_rows:
            return J
        T = self._transforms(q, self._pin_parts)
        for pin, rows in zip(self.asm.pins, self._pin_rows):
            r0 = rows.start
            for sign, part in ((1.0, pin.a), (-1.0, pin.b)):
                Tp = T[part]
                x = Tp[:3, :3] @ pin.point + Tp[:3, 3]
                m = None if pin.axis is None else Tp[:3, :3] @ pin.axis
                for j in self._chains[part]:
                    if j.kind == "fixed":
                        continue
                    i = self._idx[j.name]
                    Tpar = T[j.parent]
                    w = Tpar[:3, :3] @ j.axis
                    if j.kind == "prismatic":
                        J[r0:r0 + 3, i] += sign * w
                        continue
                    o = Tpar[:3, :3] @ j.origin + Tpar[:3, 3]
                    J[r0:r0 + 3, i] += sign * np.cross(w, x - o)
                    if m is not None:
                        J[r0 + 3:r0 + 6, i] += sign * self.L * np.cross(w, m)
        return J

    def _jac(self, q: np.ndarray) -> np.ndarray:
        """∂residual/∂u for every independent joint, native units (per deg | per mm), couplings chained."""
        return (self._jac_joint(q) * self._to_internal) @ self._D

    def unknowns(self, driven: Iterable[str]) -> list[str]:
        """Loop joints solved for when ``driven`` are prescribed: non-coupled loop joints not driven."""
        d = set(driven)
        return [n for n in self._names if n in self._loop_joints and n not in self.coupled and n not in d]

    def jacobians(self, pose: Pose, driven: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """(J_p, J_d): residual Jacobian w.r.t. ``unknowns(driven)`` and w.r.t. each of ``driven``.

        Native units: columns are per degree (revolute) or per mm (prismatic); rows are the mm
        residual rows. So dq_p/dq_d = −pinv(J_p)·J_d is in deg|mm per deg|mm. Fixed and coupled
        joints in ``driven`` get zero columns.
        """
        driven = list(driven)
        for name in driven:
            if name not in self._idx and name not in self.fixed:
                raise KeyError(f"unknown joint '{name}'{_suggest(name, self._names)}")
        J = self._jac(self._q(self._u_from(pose.q)))
        Jp = J[:, [self._idx[n] for n in self.unknowns(driven)]]
        Jd = np.zeros((self._n_rows, len(driven)))
        for k, name in enumerate(driven):
            if name in self._idx:
                Jd[:, k] = J[:, self._idx[name]]
        return Jp, Jd

    # ------------------------------------------------------------------------ solving

    def solve(self, drive: Mapping[str, float], guess: Mapping[str, float] | None = None) -> Pose:
        """Pose with ``drive`` prescribed and every loop closed; never raises (failure -> ok=False).

        Starts from ``guess`` (e.g. the previous frame's ``pose.q``; default home) and walks the
        drivers to their targets in steps of ≤ 10° / ≤ 0.05·L, each step a tangent predictor
        followed by a least-squares corrector, so the solution stays on the branch of the start
        pose. Entries of ``drive`` naming unknown, fixed or coupled joints are ignored
        (``motion.check_study`` rejects those). Revolute unknowns end within guess ± 180°.
        """
        ref = self._home.copy()
        drv: dict[str, float] = {}
        try:
            for name, v in (guess or {}).items():
                if name in self._idx and math.isfinite(float(v)):
                    ref[self._idx[name]] = float(v)
            drv = {n: float(v) for n, v in drive.items() if n in self._idx and n not in self.coupled}
            unknown = self.unknowns(drv)
            didx = np.array([self._idx[n] for n in drv], dtype=int)
            target = np.array(list(drv.values()), dtype=float)
            if not np.all(np.isfinite(target)):
                return self._pose(ref, unknown, residual=math.inf)
            uidx = np.array([self._idx[n] for n in unknown], dtype=int)
            if uidx.size == 0:
                u = ref.copy()
                u[didx] = target
            else:
                u = self._track(ref, uidx, didx, target)
            return self._pose(u, unknown)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            # malformed input or a numerical breakdown: report an open pose instead of raising
            return self._pose(ref, self.unknowns(drv), residual=math.inf)

    def _track(self, ref: np.ndarray, uidx: np.ndarray, didx: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Continuation from ``ref`` to the driver ``target`` values; returns the full u vector."""
        start = ref[didx]
        n_steps = 1
        if didx.size:
            max_step = np.where(self._rev[didx], _STEP_DEG, _STEP_FRAC * self.L)
            n_steps = int(min(_MAX_STEPS, max(1, math.ceil(float(np.max(np.abs(target - start) / max_step))))))
        u = ref.copy()
        for k in range(1, n_steps + 1):
            u = self._step(u, uidx, didx, start + (target - start) * (k / n_steps))
        for i in uidx:
            # a full turn of a revolute is the same pose — unless it drives a coupling
            if self._rev[i] and self._names[i] not in self._coupling_drivers:
                u[i] = ref[i] + (u[i] - ref[i] + 180.0) % 360.0 - 180.0
        return u

    def _step(self, u: np.ndarray, uidx: np.ndarray, didx: np.ndarray, d_next: np.ndarray) -> np.ndarray:
        """Move the drivers to ``d_next`` and re-close the loops (predictor, then plain warm start)."""
        plain = u.copy()
        plain[didx] = d_next
        starts = [plain]
        delta = d_next - u[didx]
        if np.any(delta != 0):
            J = self._jac(self._q(u))
            scale = self._col_scale[uidx]
            dy = -np.linalg.pinv(J[:, uidx] / scale, rcond=_PREDICT_RCOND) @ (J[:, didx] @ delta)
            if np.max(np.abs(dy), initial=0.0) <= self.L:  # ignore wild steps near singularities
                predicted = plain.copy()
                predicted[uidx] += dy / scale
                starts.insert(0, predicted)
        best: tuple[np.ndarray, float] | None = None
        for u0 in starts:
            u1, res = self._correct(u0, uidx)
            if res <= RESIDUAL_TOL:
                return u1
            if best is None or res < best[1]:
                best = (u1, res)
        return best[0]

    def _correct(self, u0: np.ndarray, uidx: np.ndarray) -> tuple[np.ndarray, float]:
        """Least-squares closure over the unknowns (revolutes in radians); returns (u, max |r|)."""
        s = self._to_internal[uidx]

        def q_of(x: np.ndarray) -> np.ndarray:
            u = u0.copy()
            u[uidx] = x / s
            return self._q(u)

        sol = least_squares(
            lambda x: self._residual(q_of(x)),
            u0[uidx] * s,
            jac=lambda x: self._jac(q_of(x))[:, uidx] / s,
            method="trf",
            x_scale="jac",
            xtol=_SOLVER_TOL,
            ftol=_SOLVER_TOL,
            gtol=_SOLVER_TOL,
        )
        u = u0.copy()
        u[uidx] = sol.x / s
        return u, float(np.max(np.abs(sol.fun), initial=0.0))

    def _pose(self, u: np.ndarray, unknown: list[str], residual: float | None = None) -> Pose:
        q = self._q(u)
        T = self._transforms(q)
        if residual is None:
            residual = float(np.max(np.abs(self._residual(q)), initial=0.0))
        values = dict(zip(self._names, q.tolist()))
        watched = self.coupled | set(unknown)
        violations = []
        for name in self._names:
            lim = self.asm.joints[name].limits
            if name in watched and lim is not None and not lim[0] - 1e-9 <= values[name] <= lim[1] + 1e-9:
                violations.append((name, values[name], lim))
        return Pose(values, {p: T[p] for p in self.asm.parts}, residual, residual <= RESIDUAL_TOL, violations)

    # ------------------------------------------------------------------------ structure: rank, mobility, roles

    def _rank(self, J: np.ndarray, cols: list[int]) -> int:
        """SVD rank of J[:, cols] with columns scaled to mm-equivalent (rtol 1e-9)."""
        if J.shape[0] == 0 or not cols:
            return 0
        sv = np.linalg.svd(J[:, cols] / self._col_scale[cols], compute_uv=False)
        return int(np.count_nonzero(sv > RANK_RTOL * sv[0])) if sv[0] > 1e-12 else 0

    def _samples(self) -> list[np.ndarray]:
        """Jacobians at home and at 3 seeded random perturbations of it (for generic ranks)."""
        if self._sample_jacs is None:
            rng = np.random.default_rng(_SEED)
            amp = np.where(self._rev, _PERTURB_DEG, _PERTURB_FRAC * self.L)
            us = [self._home] + [self._home + amp * rng.uniform(-1.0, 1.0, amp.size) for _ in range(_N_PERTURB)]
            self._sample_jacs = [self._jac(self._q(u)) for u in us]
        return self._sample_jacs

    def mobility(self, driven: Iterable[str]) -> int:
        """Remaining DOF of the loops with ``driven`` prescribed: #unknowns − generic rank(J_p)."""
        cols = [self._idx[n] for n in self.unknowns(driven)]
        if not cols:
            return 0
        return len(cols) - max(self._rank(J, cols) for J in self._samples())

    def idle_unknowns(self, driven: Iterable[str]) -> list[str]:
        """Unknowns of the loops that no joint in ``driven`` acts on, directly or via couplings.

        A study driving ``driven`` leaves those loops at home: their leftover DOF are not the
        study's concern (``mobility(driven + idle)`` counts only the loops it drives). Loops that
        share a joint or are tied by a coupling act as one; a loop merely carried by a driven
        joint (its common ancestor) moves rigidly and stays at home internally.
        """
        root: dict[str, str] = {}

        def find(x: str) -> str:
            root.setdefault(x, x)
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        for loop in self.loops:
            for name in loop[1:]:
                root[find(name)] = find(loop[0])
        for c in self.asm.couplings:
            root[find(c.driven)] = find(c.driver)
        acted = {find(n) for n in driven}
        return [n for n in self.unknowns(driven) if find(n) not in acted]

    def singular_at_home(self, driven: Iterable[str]) -> bool:
        """True when J_p loses rank at the home pose although its generic rank is higher."""
        cols = [self._idx[n] for n in self.unknowns(driven)]
        if not cols:
            return False
        ranks = [self._rank(J, cols) for J in self._samples()]
        return ranks[0] < max(ranks)

    def overconstrained(self, driven: Iterable[str]) -> list[str]:
        """Messages when the drivers over-drive a loop: generic rank([J_p J_d]) > rank(J_p)."""
        drivers = [n for n in dict.fromkeys(driven) if n in self._idx and n not in self.coupled]
        p_cols = [self._idx[n] for n in self.unknowns(drivers)]
        d_cols = [self._idx[n] for n in drivers]
        Js = self._samples()
        if not d_cols or max(self._rank(J, p_cols + d_cols) for J in Js) <= max(self._rank(J, p_cols) for J in Js):
            return []

        def acts(J: np.ndarray, rows: slice, col: int) -> bool:
            return float(np.max(np.abs(J[rows, col]), initial=0.0)) / self._col_scale[col] > 1e-9

        every = slice(0, self._n_rows)
        acting = [n for n in drivers if any(acts(J, every, self._idx[n]) for J in Js)]
        involved = [i for i, rows in enumerate(self._pin_rows)
                    if any(acts(J, rows, c) for J in Js for c in d_cols)]
        # DOF of the involved loops with nothing driven
        loop_cols = sorted({self._idx[n] for i in involved for n in self.loops[i] if n not in self.coupled})
        rows = np.concatenate([np.arange(self._n_rows)[self._pin_rows[i]] for i in involved])
        dof = len(loop_cols) - max(self._rank(J[rows], loop_cols) for J in Js)
        pins = ", ".join(f"'{self.asm.pins[i].name}'" for i in involved)
        return [f"overconstrained drivers: {', '.join(acting)} all act on the loop closed by pin {pins}, "
                f"which has {dof} degree(s) of freedom — drive at most {dof} of them"]

    def roles_for(self, studies: Iterable) -> dict[str, str]:
        """Joint -> fixed | coupled | driver (in any study) | passive (on a loop) | free."""
        driven = {n for s in studies for n in (s.drive or {})}
        roles = {}
        for name, j in self.asm.joints.items():
            if j.kind == "fixed":
                roles[name] = "fixed"
            elif name in self.coupled:
                roles[name] = "coupled"
            elif name in driven:
                roles[name] = "driver"
            elif name in self._loop_joints:
                roles[name] = "passive"
            else:
                roles[name] = "free"
        return roles
