"""Quasi-static gravity holding loads by virtual work (spec §4.7).

The potential energy of the moving parts is V(q) = Σ m_i · (−g)·(T_i·com_i) · 1e-3 J (g in m/s²,
positions in mm). The load an actuator on driver d must supply to hold a pose is dV/dq_d
(positive pushes +q): frictionless and backdrivable, so every other joint carries no load.

The derivative is total: coupled joints follow their drivers (``Kinematics.fk`` re-expands the
couplings) and passive loop joints follow the loop closure, dq_p/dq_d = −pinv(J_p)·J_d. The
partials ∂V/∂u_j of each independent joint come from central differences of forward kinematics
(no re-solve). Loads are converted from J per native unit to N·m (×180/π, per degree) or N
(×1000, per mm). A coupled joint moved by the study also gets the load it would carry if it
were the driver, reflected by power balance: F_c = load_d / (dq_c/dq_d in SI units).
"""

from __future__ import annotations

import math

import numpy as np

from .assembly import Assembly
from .kinematics import Kinematics, Pose
from .massprops import MassProps

__all__ = ["gravity_loads", "COND_MAX"]

COND_MAX = 1e8  # frames whose (scaled) loop Jacobian is worse conditioned than this get no load
_FD_STEP = 1e-3  # central-difference step, deg | mm
_SI = {"revolute": math.pi / 180.0, "prismatic": 1e-3}  # native unit -> rad | m
_UNIT = {"revolute": "N·m", "prismatic": "N"}
_ZERO = 1e-12  # |dq_c/dq_d| (SI) below this: the coupled joint doesn't move with the driver
_MAX_RTOL = 1e-6  # loads within this relative round-off of the maximum count as reaching it


def _driven_block(Jp: np.ndarray, Jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, passive columns) of the loop equations the drivers act on, as boolean masks.

    The closure equations decouple into blocks; starting from the rows the drivers touch, add
    every passive joint in those rows and every row those joints touch, until closed. Loops the
    study doesn't drive stay at home and don't affect its loads (nor its conditioning).
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


def _loop_sensitivity(Jp: np.ndarray, Jd: np.ndarray, col_scale: np.ndarray) -> np.ndarray | None:
    """dq_p/dq_d (native units, passive × drivers), or None when the driven loops are singular.

    Columns are scaled to comparable units (revolute per unit arc length L·θ) before the
    condition number and the pseudo-inverse, as in ``Kinematics``. A block with more passive
    joints than independent equations (leftover mobility) has cond = ∞: gravity would move it,
    so no holding load exists.
    """
    S = np.zeros((Jp.shape[1], Jd.shape[1]))
    rows, cols = _driven_block(Jp, Jd)
    if not cols.any():
        return S
    Js = Jp[np.ix_(rows, cols)] / col_scale[cols]
    sv = np.linalg.svd(Js, compute_uv=False)
    if sv.size < Js.shape[1] or sv[-1] <= 0 or sv[0] / sv[-1] > COND_MAX:
        return None
    S[cols] = -(np.linalg.pinv(Js) @ Jd[rows]) / col_scale[cols][:, None]
    return S


def gravity_loads(asm: Assembly, kin: Kinematics, result, props: dict[str, MassProps]) -> dict[str, dict]:
    """Gravity holding loads of a ``StudyResult``, keyed by joint.

    Each entry is ``{"unit": "N·m" | "N", "series": [float | None per frame], "max_abs",
    "frame", "reflected_from"}``: first the study's drivers (``reflected_from`` None), then the
    coupled joints the study moves (``reflected_from`` = the driver whose load is reflected —
    with several drivers, the one the joint follows most strongly). Frames with open loops or
    cond(J_p) > 1e8 are None; ``max_abs``/``frame`` are None when every frame is.
    """
    joints = asm.joints
    drivers = list(result.drive)
    passive = kin.unknowns(drivers)
    coupled = [n for n in joints if n in kin.coupled]
    col_scale = np.array([kin.L * _SI["revolute"] if joints[n].kind == "revolute" else 1.0 for n in passive])
    masses = [(name, mp.mass_kg, np.asarray(mp.com, dtype=float)) for name, mp in props.items()
              if mp.mass_kg > 0 and name in asm.parts and not asm.parts[name].ground]
    weight = -np.asarray(asm.gravity, dtype=float)  # V = Σ m (−g)·x

    # couplings are affine: D[c][j] = ∂q_c/∂u_j, exact from one unit step through expand()
    base = kin.expand({})
    D = {}
    for j in drivers + passive:
        stepped = kin.expand({j: base[j] + 1.0})
        D[j] = {c: stepped[c] - base[c] for c in coupled}

    def dV(q: dict[str, float], j: str) -> float:
        """∂V/∂u_j in J per native unit (central difference of FK; couplings re-expanded)."""
        Tp = kin.fk({**q, j: q[j] + _FD_STEP})
        Tm = kin.fk({**q, j: q[j] - _FD_STEP})
        dx = sum(m * (weight @ ((Tp[n][:3, :3] - Tm[n][:3, :3]) @ c + Tp[n][:3, 3] - Tm[n][:3, 3]))
                 for n, m, c in masses)
        return float(dx) * 1e-3 / (2.0 * _FD_STEP)

    loads: dict[str, list[float | None]] = {d: [] for d in drivers}
    ratios: dict[str, list[np.ndarray | None]] = {c: [] for c in coupled}  # per frame: dq_c/dq_d (SI) per driver
    for pose in result.poses:
        S = _frame_sensitivity(kin, pose, drivers, col_scale)
        if S is None:
            for d in drivers:
                loads[d].append(None)
            for c in coupled:
                ratios[c].append(None)
            continue
        grad_p = np.array([dV(pose.q, p) for p in passive])
        for i, d in enumerate(drivers):
            total = dV(pose.q, d) + float(grad_p @ S[:, i])  # J per deg | mm
            loads[d].append(total / _SI[joints[d].kind])
        for c in coupled:
            r = np.array([D[d][c] + sum(D[p][c] * S[k, i] for k, p in enumerate(passive))
                          for i, d in enumerate(drivers)])
            ratios[c].append(r * _SI[joints[c].kind] / np.array([_SI[joints[d].kind] for d in drivers]))

    out = {d: _entry(joints[d].kind, series, None) for d, series in loads.items()}
    for c in coupled:
        moving = [r for r in ratios[c] if r is not None]
        strength = np.max(np.abs(moving), axis=0) if moving else np.zeros(len(drivers))
        if not drivers or float(np.max(strength, initial=0.0)) <= _ZERO:
            continue
        i = int(np.argmax(strength))
        d = drivers[i]
        series = [None if r is None or abs(r[i]) <= _ZERO or load is None else load / float(r[i])
                  for r, load in zip(ratios[c], loads[d])]
        out[c] = _entry(joints[c].kind, series, d)
    return out


def _frame_sensitivity(kin: Kinematics, pose: Pose, drivers: list[str], col_scale: np.ndarray) -> np.ndarray | None:
    """dq_p/dq_d at one frame, or None when the frame gets no load (open loop, singular)."""
    if not pose.ok:
        return None
    Jp, Jd = kin.jacobians(pose, drivers)
    return _loop_sensitivity(Jp, Jd, col_scale)


def _entry(kind: str, series: list[float | None], reflected_from: str | None) -> dict:
    values = [(abs(v), k) for k, v in enumerate(series) if v is not None]
    max_abs, frame = max(values, key=lambda item: item[0]) if values else (None, None)
    if values:
        # the first frame reaching the maximum up to finite-difference round-off, so a constant
        # load reports f0 rather than whichever frame's last digits happened to be largest
        frame = next(k for a, k in values if a >= max_abs * (1 - _MAX_RTOL))
    return {"unit": _UNIT[kind], "series": [None if v is None else float(v) for v in series],
            "max_abs": None if max_abs is None else float(max_abs), "frame": frame,
            "reflected_from": reflected_from}
