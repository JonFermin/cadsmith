"""Kinematics (spec §4.4): forward kinematics, couplings, loop closure, roles and mobility.

Joint values are native units at the API (deg for revolute, mm for prismatic). A part transform
T maps home-world -> current-world and is built by the product of exponentials over the tree,

    T_child(q) = T_parent(q) · M_j(q_j − home_j),

with M_j a rotation about joint j's home-world line or a translation along its axis. Couplings
act on deltas from home and are linear, so every joint value is an affine function of the
independent (non-coupled) ones: q = h + D·(u − h) + o. Loop closure solves the pin residuals
for the non-coupled, non-driven loop joints with a trust-region least-squares corrector and an
analytic Jacobian (screw-theory twists chained through D).

Continuation keeps the solution on one assembly branch: each sub-step starts from a secant
extrapolation of the previous solutions (falling back to the tangent predictor and a plain warm
start) and keeps the converged solution closest to that extrapolation, so a frame that lands
exactly on a change point (where J_p is singular and both branches meet) does not flip branch.
Generic ranks (mobility, over-driven loops) are taken at random poses *on* the constraint
manifold — perturbed and re-closed — so special-geometry mobile linkages (a redundant parallel
link, a spherical 4R) are not mistaken for overconstrained ones.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .assembly import Assembly, Joint, ModelError, _suggest
from .geom import rot_about_line, transform_points, translation

__all__ = ["Pose", "Kinematics", "RESIDUAL_TOL", "COND_MAX", "SINGULAR_COND", "driven_block"]

RESIDUAL_TOL = 1e-4  # mm: a pose is ok when every residual row is within this
RANK_RTOL = 1e-9  # SVD rank tolerance relative to the largest singular value
COND_MAX = 1e8  # a (scaled) loop Jacobian worse conditioned than this gets no holding load
# A frame this badly conditioned sits on a change/dead point: a double root there is only solved
# to ~1e-6°, where cond is ~1e7, so the branch-ambiguity test uses a looser bound than COND_MAX.
SINGULAR_COND = 1e6
_SOLVER_TOL = 1e-12  # least_squares xtol = ftol = gtol
_STEP_DEG = 10.0  # continuation: largest revolute driver step per corrector solve
_STEP_FRAC = 0.05  # continuation: largest prismatic driver step, as a fraction of L
_MAX_STEPS = 720
_PREDICT_RCOND = 1e-6  # pinv cutoff for the tangent predictor (skips near-singular directions)
_N_PERTURB, _PERTURB_DEG, _PERTURB_FRAC, _SEED = 3, 10.0, 0.05, 0  # generic-rank samples
_ZERO = 1e-12  # relative magnitude below which a Jacobian entry is structurally zero
_IDLE_COL = 1e-12  # relative (scaled) column norm below which an unknown doesn't move the residual


def driven_block(Jp: np.ndarray, Jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, passive columns) of the loop equations the drivers act on, as boolean masks.

    The closure equations decouple into blocks; starting from the rows the drivers touch, add
    every passive joint in those rows and every row those joints touch, until closed. Loops a
    study doesn't drive stay at home and don't affect its loads, conditioning or singularities.
    """
    scale = max(float(np.max(np.abs(Jp), initial=0.0)), float(np.max(np.abs(Jd), initial=0.0)), 1.0)
    A, B = np.abs(Jp) > _ZERO * scale, np.abs(Jd) > _ZERO * scale
    rows = B.any(axis=1)
    while True:
        cols = A[rows].any(axis=0)
        grown = rows | A[:, cols].any(axis=1)
        if np.array_equal(grown, rows):
            return rows, cols
        rows = grown


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

    def solve(self, drive: Mapping[str, float], guess: Mapping[str, float] | None = None, *,
              prev: Mapping[str, float] | None = None, step_scale: float = 1.0) -> Pose:
        """Pose with ``drive`` prescribed and every loop closed; never raises (failure -> ok=False).

        Starts from ``guess`` (e.g. the previous frame's ``pose.q``; default home) and walks the
        drivers to their targets in steps of ≤ 10° / ≤ 0.05·L (times ``step_scale``), each step
        a predictor followed by a least-squares corrector, so the solution stays on the branch of
        the start pose. ``prev`` — the pose before ``guess`` (e.g. frame k−2) — seeds the secant
        predictor that carries the branch through singular poses. Entries of ``drive`` naming
        unknown, fixed or coupled joints are ignored (``motion.check_study`` rejects those).
        Revolute unknowns are unwrapped step by step (each within ±180° of the previous step), so
        a passive joint that turns further than 180° in one call keeps its true value.
        """
        ref = self._home.copy()
        drv: dict[str, float] = {}
        try:
            for name, v in (guess or {}).items():
                if name in self._idx and math.isfinite(float(v)):
                    ref[self._idx[name]] = float(v)
            before = None
            if prev is not None:
                before = self._home.copy()
                for name, v in prev.items():
                    if name in self._idx and math.isfinite(float(v)):
                        before[self._idx[name]] = float(v)
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
                u = self._track(ref, uidx, didx, target, before, float(step_scale))
            return self._pose(u, unknown)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            # malformed input or a numerical breakdown: report an open pose instead of raising
            return self._pose(ref, self.unknowns(drv), residual=math.inf)

    def _track(self, ref: np.ndarray, uidx: np.ndarray, didx: np.ndarray, target: np.ndarray,
               before: np.ndarray | None = None, step_scale: float = 1.0) -> np.ndarray:
        """Continuation from ``ref`` to the driver ``target`` values; returns the full u vector.

        ``before`` (the solution preceding ``ref``, if any) and every sub-step's solution feed the
        secant predictor of the next sub-step.
        """
        start = ref[didx]
        n_steps = 1
        if didx.size:
            max_step = np.where(self._rev[didx], _STEP_DEG, _STEP_FRAC * self.L) * max(step_scale, 1e-3)
            n_steps = int(min(_MAX_STEPS, max(1, math.ceil(float(np.max(np.abs(target - start) / max_step))))))
        wrap = [i for i in uidx if self._rev[i] and self._names[i] not in self._coupling_drivers]
        history = ([before] if before is not None else []) + [ref]
        u = ref.copy()
        for k in range(1, n_steps + 1):
            d_next = start + (target - start) * (k / n_steps)
            u_new = self._step(u, uidx, didx, d_next, self._secant(history, uidx, didx, d_next))
            for i in wrap:
                # a full turn of a revolute is the same pose (unless it drives a coupling): keep
                # each sub-step within ±180° of the last, which unwraps fast passive joints
                u_new[i] = u[i] + (u_new[i] - u[i] + 180.0) % 360.0 - 180.0
            history = [history[-1], u_new]
            u = u_new
        return u

    def _secant(self, history: list[np.ndarray], uidx: np.ndarray, didx: np.ndarray,
                d_next: np.ndarray) -> np.ndarray | None:
        """Linear extrapolation of the unknowns from the last two solutions to drivers ``d_next``
        (None without two solutions that differ in their drivers)."""
        if len(history) < 2 or not didx.size:
            return None
        u0, u1 = history[-2], history[-1]
        scale = self._col_scale[didx]
        dd_prev = (u1[didx] - u0[didx]) * scale
        dd = (d_next - u1[didx]) * scale
        denom = float(dd_prev @ dd_prev)
        if denom <= 1e-24 * max(1.0, float(dd @ dd)):
            return None
        r = float(dd @ dd_prev) / denom
        hint = u1.copy()
        hint[didx] = d_next
        hint[uidx] = u1[uidx] + r * (u1[uidx] - u0[uidx])
        return hint

    def _step(self, u: np.ndarray, uidx: np.ndarray, didx: np.ndarray, d_next: np.ndarray,
              hint: np.ndarray | None = None) -> np.ndarray:
        """Move the drivers to ``d_next`` and re-close the loops.

        Starts: the secant ``hint`` (when given), the tangent predictor, a plain warm start. Without
        a hint the first converged solution wins; with one, a converged solution within half the
        hinted step of the hint is taken at once, else the converged solution closest to the hint
        (the branch the previous steps were on — at a change point both branches converge).
        """
        plain = u.copy()
        plain[didx] = d_next
        starts = [plain]
        delta = d_next - u[didx]
        scale = self._col_scale[uidx]
        if np.any(delta != 0):
            J = self._jac(self._q(u))
            dy = -np.linalg.pinv(J[:, uidx] / scale, rcond=_PREDICT_RCOND) @ (J[:, didx] @ delta)
            if np.max(np.abs(dy), initial=0.0) <= self.L:  # ignore wild steps near singularities
                predicted = plain.copy()
                predicted[uidx] += dy / scale
                starts.insert(0, predicted)
        if hint is not None:
            starts.insert(0, hint)
            accept = 0.5 * float(np.linalg.norm((hint[uidx] - u[uidx]) * scale)) + 1e-9 * self.L
            turns = self._rev[uidx] & np.array([self._names[i] not in self._coupling_drivers for i in uidx])
        best: tuple[np.ndarray, float] | None = None
        closest: tuple[np.ndarray, float] | None = None
        for u0 in starts:
            u1, res = self._correct(u0, uidx)
            if res <= RESIDUAL_TOL:
                if hint is None:
                    return u1
                diff = u1[uidx] - hint[uidx]
                diff = np.where(turns, (diff + 180.0) % 360.0 - 180.0, diff)  # a full turn is no offset
                off = float(np.linalg.norm(diff * scale))
                if off <= accept:
                    return u1
                if closest is None or off < closest[1]:
                    closest = (u1, off)
            if best is None or res < best[1]:
                best = (u1, res)
        return closest[0] if closest is not None else best[0]

    def _correct(self, u0: np.ndarray, uidx: np.ndarray) -> tuple[np.ndarray, float]:
        """Least-squares closure over the unknowns (revolutes in radians); returns (u, max |r|).

        Unknowns the residual does not depend on at ``u0`` (a zero Jacobian column — e.g. the spin
        of a rod about its own axis between two ball joints) are held at their start values
        first: least squares would leave them there anyway, but the null direction slows its
        convergence ~10×. If that reduced solve does not close the loops, all unknowns are solved.
        """
        if uidx.size > 1:
            norms = np.linalg.norm(self._jac(self._q(u0))[:, uidx] * self._col_scale[uidx], axis=0)
            idle = norms <= _IDLE_COL * float(np.max(norms, initial=0.0))
            if idle.any() and not idle.all():
                u, res = self._lsq(u0, uidx[~idle])
                if res <= RESIDUAL_TOL:
                    return u, res
        return self._lsq(u0, uidx)

    def _lsq(self, u0: np.ndarray, uidx: np.ndarray) -> tuple[np.ndarray, float]:
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
        """Jacobians at home and at 3 seeded random poses near it (for generic ranks).

        The random poses lie on the constraint manifold: every joint is perturbed, then the loop
        joints are re-closed by least squares. A perturbation that leaves the loops open would
        break a special geometry (parallel links, intersecting axes) and show constraints that
        are redundant on every real pose. When home itself doesn't close (a model whose loops
        can't assemble) or no perturbation re-closes, the raw perturbations are used.
        """
        if self._sample_jacs is None:
            rng = np.random.default_rng(_SEED)
            amp = np.where(self._rev, _PERTURB_DEG, _PERTURB_FRAC * self.L)
            raw = [self._home + amp * rng.uniform(-1.0, 1.0, amp.size) for _ in range(_N_PERTURB)]
            loop = np.array([self._idx[n] for n in self._names if n in self._loop_joints and n not in self.coupled],
                            dtype=int)
            closed = []
            home_closed = float(np.max(np.abs(self._residual(self._q(self._home))), initial=0.0)) <= RESIDUAL_TOL
            if loop.size and home_closed:
                for u in raw:
                    try:
                        u1, res = self._correct(u, loop)
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if res <= RESIDUAL_TOL:
                        closed.append(u1)
            us = [self._home] + (closed or raw)
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

    def singular(self, pose: Pose, driven: Iterable[str]) -> bool:
        """Does the pose sit on a singularity of the loops ``driven`` acts on — a change point or
        dead point where J_p loses its generic rank (cond > 1e6), so the assembly branch beyond it
        is ambiguous? False for open poses and for loops left with free DOF (underconstrained)."""
        driven = list(driven)
        if not pose.ok:
            return False
        Jp, Jd = self.jacobians(pose, driven)
        if Jp.shape[1] == 0:
            return False
        rows, cols = driven_block(Jp, Jd)
        if not cols.any():
            return False
        unknown = self.unknowns(driven)
        idx = [self._idx[unknown[i]] for i in np.flatnonzero(cols)]
        if len(idx) - max(self._rank(J[rows], idx) for J in self._samples()) > 0:
            return False  # underconstrained block: its rank deficiency is mobility, not a singularity
        sv = np.linalg.svd(Jp[np.ix_(rows, cols)] / self._col_scale[idx], compute_uv=False)
        return bool(sv.size < len(idx) or sv[-1] <= 0 or sv[0] / sv[-1] > SINGULAR_COND)

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
