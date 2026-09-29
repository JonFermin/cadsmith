"""Motion studies (spec §4.5): drive sampling, default studies, study checks and study runs.

A study prescribes driver joints as functions of a normalized parameter u ∈ [0, 1] and samples
it per §1: ``once`` gives u_k = k/(N−1) at t_k = duration·k/(N−1); ``pingpong`` gives
u_k = 1 − |1 − 2k/N| at t_k = duration·k/N (k = 0..N−1), so the last frame wraps seamlessly to
the first. ``run_study`` solves every frame with the previous frame's pose as the warm start, so
the passive loop joints stay on the assembly branch they start on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .assembly import Assembly, Joint, Study, _suggest
from .kinematics import Kinematics, Pose

__all__ = [
    "StudyResult",
    "sample_drive",
    "default_studies",
    "check_study",
    "run_study",
    "JUMP_DEG",
    "JUMP_MM",
]

JUMP_DEG = 20.0  # a passive revolute changing more than this between frames is a branch-jump candidate
JUMP_MM = 5.0  # same for a passive prismatic joint
_FRAMES_JOINT = 40  # default study of a single non-loop joint (§2.2)
_FRAMES_LOOP = 60  # default study of a loop group (§2.2)
_FULL_TURN = (0.0, 360.0)  # default sweep of a revolute joint without limits
_LIMIT_TOL = 1e-9  # joint-limit slack for driver values (deg | mm)


@dataclass
class StudyResult:
    study: Study
    t: np.ndarray  # (frames,) s
    drive: dict[str, np.ndarray]  # driver joint -> (frames,) values, deg | mm
    poses: list[Pose]
    probes: dict[str, np.ndarray]  # probe -> (frames, 3) current world, mm
    branch_jumps: list[tuple[int, str, float]]  # (frame, passive joint, |change| deg | mm)


# ------------------------------------------------------------------------------ sampling


def _frame_params(frames: int, loop: str, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """(u, t) per §1 frame sampling; ``ValueError`` for frames < 2 or an unknown loop mode."""
    if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames < 2:
        raise ValueError(f"frames must be an integer >= 2 (got {frames!r})")
    n = int(frames)
    k = np.arange(n, dtype=float)
    if loop == "once":
        return k / (n - 1), float(duration) * k / (n - 1)
    if loop == "pingpong":
        return 1.0 - np.abs(1.0 - 2.0 * k / n), float(duration) * k / n
    raise ValueError(f"loop must be 'once' or 'pingpong' (got {loop!r})")


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _keyframes(spec) -> list[tuple[float, float]] | None:
    """``[(u, v), ...]`` as float pairs, or None when ``spec`` is not a keyframe list."""
    try:
        items = list(spec)
    except TypeError:
        return None
    pairs = []
    for item in items:
        try:
            u, v = item
        except (TypeError, ValueError):
            return None
        if not (_is_number(u) and _is_number(v)):
            return None
        pairs.append((float(u), float(v)))
    return pairs or None


def _drive_values(spec, u: np.ndarray) -> np.ndarray:
    """Evaluate one joint's drive spec at the parameters ``u``.

    ``(start, end)`` interpolates linearly, ``[(u, v), ...]`` piecewise-linearly between
    keyframes sorted by u (held constant outside them), ``callable(u)`` is called per frame and a
    plain number holds the joint at that value. ``ValueError`` for anything else or non-finite values.
    """
    if callable(spec):
        vals = np.array([float(spec(float(x))) for x in u])
    elif _is_number(spec):
        vals = np.full(u.shape, float(spec))
    else:
        items = list(spec) if isinstance(spec, (list, tuple, np.ndarray)) else None
        if items is not None and len(items) == 2 and all(_is_number(x) for x in items):
            a, b = float(items[0]), float(items[1])
            vals = (1.0 - u) * a + u * b  # exact end points at u = 0 and u = 1
        else:
            keys = _keyframes(spec) if items is not None else None
            if keys is None:
                raise ValueError(f"expected (start, end), [(u, value), ...], a number or callable(u) -> value, "
                                 f"got {spec!r}")
            keys.sort(key=lambda kv: kv[0])
            vals = np.interp(u, [k[0] for k in keys], [k[1] for k in keys])
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"drive produced non-finite values ({spec!r})")
    return vals


def sample_drive(study: Study) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Frame times (s) and each driven joint's values (deg | mm) per §1 sampling.

    ``ValueError`` for an unsamplable study (frames < 2, unknown loop mode, malformed drive).
    """
    u, t = _frame_params(study.frames, study.loop, study.duration)
    return t, {str(name): _drive_values(spec, u) for name, spec in study.drive.items()}


# ------------------------------------------------------------------------------ default studies


def _sweep_range(joint: Joint) -> tuple[float, float] | None:
    """Default sweep: the limits, or a full turn for a revolute without limits (None: can't sweep)."""
    if joint.limits is not None:
        return (float(joint.limits[0]), float(joint.limits[1]))
    return _FULL_TURN if joint.kind == "revolute" else None


def _loop_groups(loops: list[list[str]]) -> list[set[str]]:
    """Mobility groups: loops that share a joint move together."""
    groups: list[set[str]] = []
    for loop in loops:
        merged = set(loop)
        rest = []
        for g in groups:
            if g & merged:
                merged |= g
            else:
                rest.append(g)
        groups = rest + [merged]
    return groups


def _loop_drivers(asm: Assembly, kin: Kinematics, group: set[str]) -> list[str]:
    """Drivers for one loop group in §2.2 priority order: actuated joints, then joints with
    limits, then revolutes (declaration order within each tier).

    A candidate is taken only if it removes a degree of freedom (a joint whose motion is already
    determined would over-drive the loop), so a 1-DOF loop gets exactly one driver and a
    multi-DOF group gets one per DOF where sweepable candidates exist.
    """
    free = [n for n in asm.joints if n in group and n not in kin.coupled]
    tiers = ([n for n in asm.actuators if n in free],
             [n for n in free if asm.joints[n].limits is not None],
             [n for n in free if asm.joints[n].kind == "revolute"])
    chosen: list[str] = []
    mobility = kin.mobility(chosen)
    for name in dict.fromkeys(n for tier in tiers for n in tier):
        if _sweep_range(asm.joints[name]) is None:
            continue
        m = kin.mobility(chosen + [name])
        if m < mobility:
            chosen.append(name)
            mobility = m
    return chosen


def default_studies(asm: Assembly, kin: Kinematics) -> list[Study]:
    """§2.2 default studies, one per mobility group, ordered by their first driver's declaration.

    Every non-loop, non-coupled joint is swept over its limits (revolute without limits 0→360°)
    once in 40 frames; a prismatic joint without limits has no natural range and is left at home.
    Each group of loops sharing joints is driven through its preferred driver(s) (see
    ``_loop_drivers``) over the same ranges in 60 frames. The runner uses these when the script
    declares no study.
    """
    order = {name: i for i, name in enumerate(asm.joints)}
    loop_joints = {n for loop in kin.loops for n in loop}
    studies: list[tuple[int, Study]] = []
    for name, joint in asm.joints.items():
        if joint.kind == "fixed" or name in kin.coupled or name in loop_joints:
            continue
        rng = _sweep_range(joint)
        if rng is not None:
            studies.append((order[name], Study(f"sweep_{name}", {name: rng}, frames=_FRAMES_JOINT)))
    for group in _loop_groups(kin.loops):
        drivers = _loop_drivers(asm, kin, group)
        if drivers:
            drive = {d: _sweep_range(asm.joints[d]) for d in drivers}
            studies.append((order[drivers[0]], Study("sweep_" + "+".join(drivers), drive, frames=_FRAMES_LOOP)))
    return [s for _, s in sorted(studies, key=lambda item: item[0])]


# ------------------------------------------------------------------------------ checks


def _unit(joint: Joint) -> str:
    return "°" if joint.kind == "revolute" else " mm"


def check_study(kin: Kinematics, study: Study) -> list[str]:
    """Every problem that stops ``study`` from running, as messages (empty list = runnable).

    Unknown, coupled or fixed joints in ``drive``; malformed drive specs; keyframes with u outside
    [0, 1]; driver values outside the joint's limits; frames < 2; bad loop mode or duration;
    drivers that over-drive a loop.
    """
    asm = kin.asm
    where = f"study '{study.name}'"
    errors: list[str] = []
    u: np.ndarray | None = None
    try:
        u, _ = _frame_params(study.frames, study.loop, 1.0)  # u doesn't depend on the duration
    except ValueError as exc:
        errors.append(f"{where}: {exc}")
    if not (_is_number(study.duration) and math.isfinite(study.duration) and study.duration > 0):
        errors.append(f"{where}: duration must be > 0 s (got {study.duration!r})")
    if not isinstance(study.drive, Mapping) or not study.drive:
        errors.append(f"{where}: drive must be a non-empty dict {{joint: values}}")
        return errors

    movable = [n for n, j in asm.joints.items() if j.kind != "fixed"]
    drivers: list[str] = []
    for name, spec in study.drive.items():
        joint = asm.joints.get(name)
        if joint is None:
            errors.append(f"{where}: unknown joint '{name}'{_suggest(name, movable)}")
            continue
        if joint.kind == "fixed":
            errors.append(f"{where}: joint '{name}' is fixed and can't be driven")
            continue
        if name in kin.coupled:
            src = next(c for c in asm.couplings if c.driven == name)
            errors.append(f"{where}: joint '{name}' is driven by {src.kind} '{src.name}' — "
                          f"drive its driver '{src.driver}' instead")
            continue
        drivers.append(name)
        keys = None if callable(spec) or _is_number(spec) else _keyframes(spec)
        bad = [k for k, _ in keys or [] if not 0.0 <= k <= 1.0]
        if bad:
            errors.append(f"{where}: keyframes for '{name}' need u in [0, 1] (got u = "
                          f"{', '.join(f'{b:g}' for b in bad)})")
        if u is None:
            continue
        try:
            vals = _drive_values(spec, u)
        except Exception as exc:  # user callables may raise anything; report, never propagate
            errors.append(f"{where}: drive for '{name}' failed: {type(exc).__name__}: {exc}")
            continue
        if joint.limits is not None:
            lo, hi = joint.limits
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmin < lo - _LIMIT_TOL or vmax > hi + _LIMIT_TOL:
                errors.append(f"{where}: '{name}' is driven over {vmin:g}…{vmax:g}{_unit(joint)}, outside its "
                              f"limits [{lo:g}, {hi:g}]{_unit(joint)}")
    errors.extend(f"{where}: {msg}" for msg in kin.overconstrained(drivers))
    return errors


# ------------------------------------------------------------------------------ running


def _passive_step(kin: Kinematics, drivers: list[str], pose: Pose) -> np.ndarray | None:
    """dq_p/dq_d (native units) at ``pose`` for the passive joints, or None when unavailable."""
    if not pose.ok:
        return None
    Jp, Jd = kin.jacobians(pose, drivers)
    if Jp.shape[1] == 0:
        return np.zeros((0, len(drivers)))
    return -np.linalg.pinv(Jp) @ Jd


def _branch_jumps(kin: Kinematics, drivers: list[str], poses: list[Pose]) -> list[tuple[int, str, float]]:
    """Passive joints that jump more than 20° / 5 mm between consecutive frames.

    A change that big is only a jump if the drivers' motion doesn't explain it: when both frames
    are closed, the change predicted by the trapezoidal rule on dq_p/dq_d = −pinv(J_p)·J_d must
    also miss by more than the threshold. That keeps coarse studies of fast-moving passive joints
    from being flagged, while a switch of assembly branch (which the tangent never predicts) is.
    Open frames are judged on the raw change alone.
    """
    passive = kin.unknowns(drivers)
    if not passive:
        return []
    limit = np.array([JUMP_DEG if kin.asm.joints[n].kind == "revolute" else JUMP_MM for n in passive])
    slopes: dict[int, np.ndarray | None] = {}

    def slope(k: int) -> np.ndarray | None:
        if k not in slopes:
            slopes[k] = _passive_step(kin, drivers, poses[k])
        return slopes[k]

    jumps = []
    for k in range(1, len(poses)):
        prev, cur = poses[k - 1], poses[k]
        dq = np.array([cur.q[n] - prev.q[n] for n in passive])
        big = np.abs(dq) > limit
        if not big.any():
            continue
        s0, s1 = slope(k - 1), slope(k)
        if s0 is not None and s1 is not None:
            dd = np.array([cur.q[d] - prev.q[d] for d in drivers])
            big &= np.abs(dq - 0.5 * (s0 + s1) @ dd) > limit
        jumps.extend((k, passive[i], float(abs(dq[i]))) for i in np.flatnonzero(big))
    return jumps


def run_study(asm: Assembly, kin: Kinematics, study: Study) -> StudyResult:
    """Solve every frame of ``study`` (frame k warm-starts from frame k−1) and track the probes.

    Assumes ``check_study`` passed; the first frame is reached from the home pose.
    """
    t, drive = sample_drive(study)
    poses: list[Pose] = []
    guess: dict[str, float] | None = None
    for k in range(len(t)):
        pose = kin.solve({name: float(v[k]) for name, v in drive.items()}, guess)
        poses.append(pose)
        guess = pose.q
    probes = {p.name: np.array([kin.point(pose, p.part, p.point) for pose in poses]) for p in asm.probes}
    return StudyResult(study, t, drive, poses, probes, _branch_jumps(kin, list(drive), poses))
