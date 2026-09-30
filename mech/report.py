"""Issues, report.json and the compact text summary (spec §4.9).

``build_report`` turns the analysis results into a JSON-ready dict: counts, roles, mass, per-study
ranges / probes / clearance / loads (ranges and probe stats over the closed frames only), and a
flat, worst-first list of ``Issue`` dicts whose messages carry actionable detail (overlap extent +
home-world location, every driver's value at the frame, the driver sub-range where a loop opens,
which limit a passive joint breaks and where). ``attach_targets`` folds the target results in
afterwards (the runner evaluates targets on the built report).

``format_summary`` renders a report as ≤ 15 lines in a fixed order —
header · FAIL · target_miss · WARN · INFO · loops · clearance · loads · targets · ranges · Δprev ·
view — deduplicating issues across studies and dropping the least important lines first when over
budget. A partial run (``--study``/``--frames``) says so in the header, and Δprev compares only
the studies both runs sampled alike. Every float in the report goes through ``geom.fnum`` (6
significant figures, None for non-finite) so the JSON never contains NaN/Infinity; the summary
prints 3 significant figures.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from .geom import fnum, slug
from .kinematics import RESIDUAL_TOL
from .massprops import assembly_props
from .targets import joint_series, path_length, probe_series

if TYPE_CHECKING:  # the analysis modules are only needed for type hints here
    from .assembly import Assembly
    from .clearance import PairResult, SweepResult
    from .kinematics import Kinematics
    from .massprops import MassProps
    from .motion import StudyResult

__all__ = ["Issue", "build_report", "build_check_report", "invalid_report", "attach_targets", "format_summary",
           "format_check", "jsonable", "issue_key", "sig", "decl", "vec", "extent", "SUMMARY_LINES"]

Severity = Literal["FAIL", "WARN", "INFO"]

SUMMARY_LINES = 15  # default line budget of format_summary
_SEVERITY_RANK = {"FAIL": 0, "WARN": 1, "INFO": 2}
# Report order within a severity (most actionable first).
_CODE_ORDER = ["invalid_model", "interference", "static_interference", "loop_open", "over_capacity",
               "tight_clearance", "joint_limit", "branch_jump", "singular_pose", "target_miss", "underconstrained",
               "near_planar", "joint_off_part", "gear_mesh", "contact", "held_at_home"]
_CODE_RANK = {c: i for i, c in enumerate(_CODE_ORDER)}
# Codes whose issue value measures how bad it is (bigger = worse, except a gap); others keep their order.
_RANKED_BY_VALUE = {"interference", "static_interference", "loop_open", "over_capacity", "tight_clearance",
                    "joint_limit", "branch_jump", "singular_pose", "underconstrained", "contact"}
_SMALLER_IS_WORSE = {"tight_clearance"}
_SHORT = {"tight_clearance": "tight"}  # display names in the summary
_SF_WARN = 1.5  # over_capacity WARN below this safety factor
_STILL = 1e-9  # a joint range narrower than this counts as "did not move"
MINUS = "−"


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str
    study: str | None = None
    frame: int | None = None
    parts: list[str] = field(default_factory=list)
    value: float | None = None
    location: list | None = None
    extent: list | None = None

    def to_dict(self) -> dict:
        return jsonable(asdict(self))


# ================================================================================================
# number formatting
# ================================================================================================


def jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-native types: numpy -> Python, floats via ``fnum`` (no NaN/inf)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return fnum(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, Issue):
        return obj.to_dict()
    return str(obj)


def sig(x, n: int = 3) -> str:
    """``n`` significant figures for a measured value: fixed notation for 1e-3 ≤ |x| < 1e5
    (trailing zeros kept: ``12.0``), else a compact exponent (``2.1e-9``); U+2212 minus sign."""
    if x is None or not math.isfinite(float(x)):
        return "n/a"
    x = float(x)
    if x == 0:
        return "0"
    r = float(f"{x:.{n - 1}e}")  # rounded first, so 9.996 -> 10.0 gets the right exponent
    e = math.floor(math.log10(abs(r)))
    if -3 <= e < 5:
        s = f"{abs(r):.{max(0, n - 1 - e)}f}"
    else:
        mant, ex = f"{abs(r):.{n - 1}e}".split("e")
        s = f"{mant.rstrip('0').rstrip('.')}e{int(ex)}"
    return (MINUS if r < 0 else "") + s


def decl(x) -> str:
    """A value the user declared (limit, capacity, target bound): shortest form, e.g. ``0.5``."""
    if x is None or not math.isfinite(float(x)):
        return "n/a"
    s = f"{float(x):g}"
    return MINUS + s[1:] if s.startswith("-") else s


def _comps(v) -> list[str]:
    """Vector components with one shared decimal count (3 sig figs of the largest component)."""
    a = [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]
    if not all(math.isfinite(x) for x in a):
        return ["n/a"] * len(a)
    m = max((abs(x) for x in a), default=0.0)
    if m == 0:
        return ["0"] * len(a)
    e = math.floor(math.log10(float(f"{m:.2e}")))
    if not -3 <= e < 5:
        return [sig(x) for x in a]
    d = max(0, 2 - e)
    return ["0" if round(x, d) == 0 else (MINUS if x < 0 else "") + f"{abs(round(x, d)):.{d}f}" for x in a]


def vec(v) -> str:
    """A point or direction: ``(48.1, 12.0, 1.9)``."""
    return f"({', '.join(_comps(v))})"


def extent(v) -> str:
    """A box size: ``0.4×2.1×4.0``."""
    return "×".join(_comps(v))


def _unit(kind: str | None) -> str:
    return " mm" if kind == "prismatic" else "°"


def _jv(value, kind: str | None) -> str:
    """Joint value with its unit (``80.0°`` / ``12.5 mm``)."""
    return f"{sig(value)}{_unit(kind)}"


def _jrange(lo, hi, kind: str | None) -> str:
    return f"{sig(lo)}…{sig(hi)}{_unit(kind)}"


def _frames(ks: list[int]) -> str:
    """``f47`` or ``f47–f60``."""
    return f"f{ks[0]}" if ks[0] == ks[-1] else f"f{ks[0]}–f{ks[-1]}"


def _runs(ks: Iterable[int]) -> list[list[int]]:
    """Split sorted frame indices into runs of consecutive frames."""
    runs: list[list[int]] = []
    for k in sorted(ks):
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


# ================================================================================================
# report building
# ================================================================================================


class _StudyCtx:
    """Per-study lookups shared by the issue builders.

    An issue's pose is given by the values of the study drivers that move its parts (every driver
    when the parts aren't known), with the coupled outputs they drive in parentheses — e.g.
    ``j_pan_motor=−502°, j_tilt=84.0° (j_pan=−167°)`` — so the pose can be set up again; drivers
    that move neither part (another arm driven in the same study) are left out.
    """

    def __init__(self, asm: Assembly, name: str, result: StudyResult, kin: Kinematics | None = None):
        self.asm = asm
        self.name = name
        self.result = result
        self.drivers = [d for d in result.study.drive if d in result.drive]
        self.primary = self.drivers[0] if self.drivers else None
        self.kinds = {n: j.kind for n, j in asm.joints.items()}
        self.kind = self.kinds.get(self.primary)
        # joints the drivers move through couplings (e.g. a gear's output): shown next to them
        self.coupled = [c.driven for c in asm.couplings
                        if c.driven in asm.joints and _coupling_root(asm, c.driven) in self.drivers]
        # motion groups: joints tied by a loop or a coupling move together
        self._root: dict[str, str] = {}
        for loop in (kin.loops if kin is not None else []):
            for j in loop[1:]:
                self._union(j, loop[0])
        for c in asm.couplings:
            self._union(c.driven, c.driver)

    def _find(self, x: str) -> str:
        self._root.setdefault(x, x)
        while self._root[x] != x:
            self._root[x] = self._root[self._root[x]]
            x = self._root[x]
        return x

    def _union(self, a: str, b: str) -> None:
        self._root[self._find(a)] = self._find(b)

    def movers(self, parts: Iterable[str] | None = None) -> list[str]:
        """The drivers (declaration order) that move any of ``parts``: on a part's joint chain to
        the ground, or acting on a loop / coupling that joint belongs to. All drivers without parts."""
        if not parts:
            return list(self.drivers)
        chain = set()
        for p in parts:
            seen = set()
            while p not in seen and (j := self.asm.parent_joint(p)) is not None:
                seen.add(p)
                chain.add(self._find(j.name))
                p = j.parent
        movers = [d for d in self.drivers if self._find(d) in chain]
        return movers or list(self.drivers)

    def values(self, k: int, parts: Iterable[str] | None = None, frac: float | None = None) -> dict[str, float]:
        """Driver values at frame k (those that move ``parts``), then the coupled outputs they drive.
        ``frac`` (e.g. 1.5, a hit found between frames 1 and 2) interpolates between its two frames,
        as the sweep's sub-frame poses do."""
        movers = self.movers(parts)
        poses, n = self.result.poses, len(self.result.poses)
        k0, s = k, 0.0
        if frac is not None and 0 <= math.floor(frac) < n - 1:
            k0, s = int(math.floor(frac)), float(frac) - math.floor(frac)
        k1 = min(k0 + 1, n - 1)

        def lerp(a, b) -> float:
            return float(a) + s * (float(b) - float(a))

        out = {d: lerp(self.result.drive[d][k0], self.result.drive[d][k1]) for d in movers}
        q0 = poses[k0].q if k0 < n else {}
        q1 = poses[k1].q if k1 < n else q0
        out.update({c: lerp(q0[c], q1.get(c, q0[c])) for c in self.coupled
                    if c in q0 and _coupling_root(self.asm, c) in movers})
        return out

    def at(self, k: int, parts: Iterable[str] | None = None, frac: float | None = None,
           flat: bool = False) -> str:
        """``j_crank=80.0°`` — the drivers' values at frame k (coupled outputs in parentheses, or
        just listed after them with ``flat``, for text that is already parenthesized)."""
        if self.primary is None:
            return f"f{k}"
        vals = self.values(k, parts, frac)
        head = ", ".join(f"{d}={_jv(v, self.kinds.get(d))}" for d, v in vals.items() if d in self.drivers)
        tail = [f"{c}={_jv(v, self.kinds.get(c))}" for c, v in vals.items() if c not in self.drivers]
        if not tail:
            return head
        return f"{head}, {', '.join(tail)}" if flat else f"{head} ({', '.join(tail)})"

    def drive_range(self, ks: list[int], joint: str | None = None) -> str:
        joint = joint or self.primary
        vals = [float(self.result.drive[joint][k]) for k in ks]
        return _jrange(min(vals), max(vals), self.kinds.get(joint))

    def span(self, ks: list[int], parts: Iterable[str] | None = None) -> str:
        """The drive over frames ``ks``: ``125…235°`` for one driver, ``j_a 0…10°, j_b 5…9°`` for several."""
        movers = self.movers(parts)
        if len(movers) == 1 and len(self.drivers) == 1:
            return self.drive_range(ks)
        return ", ".join(f"{d} {self.drive_range(ks, d)}" for d in movers)


def _home_point(T: np.ndarray, p) -> np.ndarray:
    """Current-world point -> home world of the part whose transform is T."""
    R, t = T[:3, :3], T[:3, 3]
    return R.T @ (np.asarray(p, dtype=float) - t)


def _pin_residual(L: float, pin, T: dict[str, np.ndarray]) -> float:
    """Max |residual row| (mm) of one pin: position rows plus L-scaled axis rows (as in §4.4)."""
    Ta, Tb = T[pin.a], T[pin.b]
    dR = Ta[:3, :3] - Tb[:3, :3]
    rows = np.abs(dR @ pin.point + Ta[:3, 3] - Tb[:3, 3])
    worst = float(np.max(rows))
    if pin.axis is not None:
        worst = max(worst, float(np.max(np.abs(L * (dR @ pin.axis)))))
    return worst


def _loop_issues(asm: Assembly, kin: Kinematics, ctx: _StudyCtx) -> list[Issue]:
    """``loop_open`` FAIL per pin: which driver sub-range fails, how badly, and what closes."""
    poses = ctx.result.poses
    failing = [k for k, p in enumerate(poses) if not p.ok]
    if not failing:
        return []
    if not asm.pins:
        return [Issue("FAIL", "loop_open", f"study '{ctx.name}': pose failed to solve in {len(failing)} frame(s) "
                      f"({_frames(failing)})", ctx.name, failing[0])]
    per_pin = {pin.name: {k: _pin_residual(kin.L, pin, poses[k].transforms) for k in failing} for pin in asm.pins}
    blamed = {name: [k for k in failing if r[k] > RESIDUAL_TOL] for name, r in per_pin.items()}
    orphans = set(failing) - {k for ks in blamed.values() for k in ks}
    ok_frames = [k for k, p in enumerate(poses) if p.ok]
    issues = []
    for pin in asm.pins:
        ks = sorted(set(blamed[pin.name]) | orphans)
        if not ks:
            continue
        res = [poses[k].residual if k in orphans else per_pin[pin.name][k] for k in ks]
        finite = [r for r in res if math.isfinite(r)]
        worst_k = ks[int(np.argmax([r if math.isfinite(r) else -1.0 for r in res]))]
        runs = _runs(ks)
        pin_parts = [pin.a, pin.b]
        single = len(ctx.drivers) == 1
        sep = ", " if single else "; "
        if ctx.primary is None:
            where = _frames(ks)
        else:
            shown = [f"{ctx.span(r, pin_parts)} ({_frames(r)})" for r in runs[:3]]
            more = f" +{len(runs) - 3} more ranges" if len(runs) > 3 else ""
            where = (f"{ctx.primary} " if single else "") + sep.join(shown) + more
        worst = f"max residual {sig(max(finite))} mm" if finite else "solver failed"
        if not ok_frames:
            closes = "never closes — check the pin point and joint origins/axes"
        elif ctx.primary is None:
            closes = f"closes {_frames(ok_frames)}"
        else:
            closes = "closes " + sep.join(ctx.span(r, pin_parts) for r in _runs(ok_frames)[:3])
        issues.append(Issue("FAIL", "loop_open",
                            f"loop {pin.name} open for {where}, {worst}; {closes} — out of reach: shorten the drive "
                            f"or change link lengths",
                            ctx.name, worst_k, [pin.a, pin.b], max(finite) if finite else None,
                            pin.point.tolist()))
    return issues


def _excess(row: tuple[int, float, tuple[float, float]]) -> float:
    """How far a (frame, value, (lo, hi)) limit violation lies outside the limits."""
    _, v, (lo, hi) = row
    return lo - v if v < lo else v - hi


def _limit_issues(asm: Assembly, ctx: _StudyCtx) -> list[Issue]:
    """``joint_limit`` WARN per coupled/passive joint that leaves its limits."""
    hits: dict[str, list[tuple[int, float, tuple[float, float]]]] = {}
    for k, pose in enumerate(ctx.result.poses):
        for name, value, lim in pose.limit_violations:
            hits.setdefault(name, []).append((k, float(value), (float(lim[0]), float(lim[1]))))
    issues = []
    for name, rows in hits.items():
        k, v, (lo, hi) = max(rows, key=_excess)
        j = asm.joints[name]
        rel = f"< {decl(lo)}" if v < lo else f"> {decl(hi)}"
        ks = sorted(r[0] for r in rows)
        issues.append(Issue("WARN", "joint_limit",
                            f"{name} {_jv(v, j.kind)} {rel} limit @ {ctx.at(k, [j.child])} (f{k}; {len(ks)} frame(s) "
                            f"{_frames(ks)})", ctx.name, k, [j.child], _excess((k, v, (lo, hi)))))
    return issues


def _jump_issues(asm: Assembly, ctx: _StudyCtx) -> list[Issue]:
    """``branch_jump`` WARN per passive joint that flips assembly mode between two closed frames.

    A jump into or out of an open frame is the loop failing to close (already a ``loop_open``
    FAIL): the open frame's joint values are a least-squares guess, not an assembly mode.
    """
    poses = ctx.result.poses
    by_joint: dict[str, list[tuple[int, float]]] = {}
    for k, name, jump in ctx.result.branch_jumps:
        k = int(k)
        if not (poses[k].ok and (k == 0 or poses[k - 1].ok)):
            continue
        by_joint.setdefault(name, []).append((k, float(jump)))
    issues = []
    for name, rows in by_joint.items():
        k, jump = max(rows, key=lambda r: abs(r[1]))
        j = asm.joints.get(name)
        kind = j.kind if j else None
        n = f" ({len(rows)} jumps)" if len(rows) > 1 else ""
        issues.append(Issue("WARN", "branch_jump",
                            f"{name} jumps {_jv(abs(jump), kind)} at f{k} @ {ctx.at(k, [j.child] if j else None)}{n} — "
                            f"dead point or assembly-mode flip; add frames or keep the drive away from the toggle",
                            ctx.name, k, [j.child] if j else [], abs(jump)))
    return issues


def _singular_issues(ctx: _StudyCtx, home_singular: bool) -> list[Issue]:
    """``singular_pose`` WARN per study whose closed frames sit on a change/dead point
    (``StudyResult.singular``). A frame 0 at a singular home pose is left to the structural INFO."""
    poses = ctx.result.poses
    ks = [int(k) for k in getattr(ctx.result, "singular", None) or [] if poses[int(k)].ok]
    if home_singular and ks and ks[0] == 0 and all(
            abs(float(v[0]) - ctx.asm.joints[d].home) <= 1e-9 for d, v in ctx.result.drive.items()):
        ks = ks[1:]
    if not ks:
        return []
    k, n = ks[0], len(ks)
    runs = _runs(ks)
    shown = ", ".join(_frames(r) for r in runs[:3]) + (f" +{len(runs) - 3} more" if len(runs) > 3 else "")
    return [Issue("WARN", "singular_pose",
                  f"study '{ctx.name}' passes a change/dead point at f{k} @ {ctx.at(k)} ({n} frame{'s' if n != 1 else ''} "
                  f"{shown}) — the loop Jacobian is singular there, so the real mechanism can take either assembly "
                  f"branch; mech kept the branch of the frames before", ctx.name, k, [], float(n))]


def _pair_frames(sweep: SweepResult) -> dict[frozenset, list[tuple[int, PairResult]]]:
    pairs: dict[frozenset, list[tuple[int, PairResult]]] = {}
    for k, frame in enumerate(sweep.per_frame):
        for r in frame:
            pairs.setdefault(frozenset((r.a, r.b)), []).append((k, r))
    return pairs


def _home_location(r: PairResult, T_a: np.ndarray, T_b: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The pair's overlap/closest-point midpoint in the home world of part a and of part b
    (``r.location`` / ``r.location_b`` when provided, else mapped back through T_a / T_b)."""
    mid = (np.asarray(r.pa, dtype=float) + np.asarray(r.pb, dtype=float)) / 2
    loc = np.asarray(r.location, dtype=float) if r.location is not None else _home_point(T_a, mid)
    loc_b = getattr(r, "location_b", None)
    if loc_b is not None:
        loc_b = np.asarray(loc_b, dtype=float)
    else:
        loc_b = _home_point(T_a if T_b is None else T_b, mid)
    return loc, loc_b


def _spot(r: PairResult, loc: np.ndarray, loc_b: np.ndarray, direction: str = "") -> str:
    """``(x, y, z)`` when both parts' home geometry puts the spot at the same point (at home, or
    parts that moved together), else ``a (x, y, z)<direction>, b (x, y, z)`` — where to look on each."""
    if vec(loc) == vec(loc_b):
        return f"{vec(loc)}{direction}"
    return f"{r.a} {vec(loc)}{direction}, {r.b} {vec(loc_b)}"


def _interference_msg(asm: Assembly, r: PairResult, loc: np.ndarray, loc_b: np.ndarray, where: str) -> str:
    ext = f" {extent(r.extent)} mm" if r.extent is not None else ""
    note = " — meshing pair: check center distance/backlash" if r.meshing else ""
    max_depth = asm.allow_depth.get(frozenset((r.a, r.b)))
    if r.allowed and max_depth is not None:
        depth = getattr(r, "depth", None)
        note = (f" — allowed contact deeper than max_depth {decl(max_depth)} mm (mean depth {sig(depth)}): fix the "
                f"fit, or raise max_depth / max_depth=None if the overlap is intended (belt teeth)")
    return f"{r.a}/{r.b} {sig(r.volume)} mm³ overlap{ext} @ {_spot(r, loc, loc_b)} · {where}{note}"


def _tight_msg(r: PairResult, loc: np.ndarray, loc_b: np.ndarray, T_a: np.ndarray, clearance: float,
               where: str) -> str:
    # closest-point direction a -> b, rotated into the same home frame as a's location
    d = T_a[:3, :3].T @ (np.asarray(r.pb, dtype=float) - np.asarray(r.pa, dtype=float))
    n = float(np.linalg.norm(d))
    direction = f" dir {vec(d / n)}" if n > 1e-12 else ""
    return f"{r.a}/{r.b} gap {sig(r.distance)} mm < {decl(clearance)} @ {_spot(r, loc, loc_b, direction)} · {where}"


def _clearance_issues(asm: Assembly, ctx: _StudyCtx, sweep: SweepResult) -> list[Issue]:
    """Worst ``interference`` (FAIL) or ``tight_clearance`` (WARN) per pair over the study."""
    issues = []
    for key, rows in _pair_frames(sweep).items():
        bad = [(k, r) for k, r in rows if r.status == "interference"]
        code, sev = "interference", "FAIL"
        if not bad:
            bad = [(k, r) for k, r in rows if r.status == "tight"]
            code, sev = "tight_clearance", "WARN"
        if not bad:
            continue
        if code == "interference":
            k, r = max(bad, key=lambda kr: kr[1].volume)
        else:
            k, r = min(bad, key=lambda kr: kr[1].distance if kr[1].distance is not None else math.inf)
        ks = sorted({kk for kk, _ in bad})  # distinct frames (a pair can report several spots per frame)
        frac = getattr(r, "at", None)  # a hit the sweep found between two frames (filed at the nearer one)
        if frac is not None and abs(frac - round(frac)) > 1e-9:
            k0 = int(math.floor(frac))
            where = f"between f{k0}–f{k0 + 1} {ctx.at(k, [r.a, r.b], frac)}"
        else:
            where = f"f{k} {ctx.at(k, [r.a, r.b])}"
        where += f" ({len(ks)} frames {_frames(ks)})" if len(ks) > 1 else ""
        T = ctx.result.poses[k].transforms
        T_a = T[r.a]
        loc, loc_b = _home_location(r, T_a, T.get(r.b))
        if code == "interference":
            msg, value = _interference_msg(asm, r, loc, loc_b, where), r.volume
        else:
            msg, value = _tight_msg(r, loc, loc_b, T_a, asm.clearance, where), r.distance
        issues.append(Issue(sev, code, msg, ctx.name, k, [r.a, r.b], value, loc.tolist(),
                            None if r.extent is None else list(r.extent)))
    return issues


def _fix_joint(asm: Assembly, a: str, b: str) -> str | None:
    for j in asm.joints.values():
        if j.kind == "fixed" and {j.parent, j.child} == {a, b}:
            return j.name
    return None


def _home_issues(asm: Assembly, home: list[PairResult], swept: bool) -> list[Issue]:
    """Home-pose findings: ``static_interference`` inside rigid groups (the sweeps skip those
    pairs), plus interference/tight of moving pairs when no study was swept (``swept`` False)."""
    groups = asm.rigid_groups()
    eye = np.eye(4)
    issues = []
    for r in home:
        loc, loc_b = _home_location(r, eye, eye)
        ext = None if r.extent is None else list(r.extent)
        rigid = groups.get(r.a) is not None and groups.get(r.a) == groups.get(r.b)
        if r.status == "interference" and rigid:
            fix = _fix_joint(asm, r.a, r.b)
            how = f"fix-attached by '{fix}'" if fix else "in the same rigid group"
            issues.append(Issue("WARN" if fix else "FAIL", "static_interference",
                                _interference_msg(asm, r, loc, loc_b, "at home") + f" — parts are {how}; union "
                                f"them into one part or leave a gap", None, None, [r.a, r.b], r.volume, loc.tolist(),
                                ext))
        elif rigid or swept:
            continue
        elif r.status == "interference":
            issues.append(Issue("FAIL", "interference", _interference_msg(asm, r, loc, loc_b, "at home"), None, None,
                                [r.a, r.b], r.volume, loc.tolist(), ext))
        elif r.status == "tight":
            issues.append(Issue("WARN", "tight_clearance", _tight_msg(r, loc, loc_b, eye, asm.clearance, "at home"),
                                None, None, [r.a, r.b], r.distance, loc.tolist()))
    return issues


def _contact_issues(asm: Assembly, sweeps: dict[str, SweepResult], home: list[PairResult]) -> list[Issue]:
    """``contact`` INFO per allow_contact pair that overlaps within its max_depth: the deepest
    overlap over every study (and the home pose), one line per pair."""
    deepest: dict[frozenset, tuple[PairResult, int | None, str | None]] = {}

    def note(r: PairResult, k: int | None, study: str | None) -> None:
        depth = getattr(r, "depth", None)
        if not (r.allowed and r.status == "contact" and depth is not None and math.isfinite(depth)):
            return
        key = frozenset((r.a, r.b))
        if key not in deepest or depth > deepest[key][0].depth:
            deepest[key] = (r, k, study)

    for name, sweep in sweeps.items():
        for k, frame in enumerate(sweep.per_frame):
            for r in frame:
                note(r, k, name)
    for r in home:
        note(r, None, None)
    issues = []
    for key, (r, k, study) in sorted(deepest.items(), key=lambda kv: sorted(kv[0])):
        where = "at home" if k is None else f"@f{k}"
        allowed = asm.allow_depth.get(key)
        issues.append(Issue("INFO", "contact",
                            f"{r.a}/{r.b} mean depth {sig(r.depth)} mm, {sig(r.volume)} mm³ {where} (allowed ≤ "
                            f"{decl(allowed)} mm)", study, k, [r.a, r.b], r.depth))
    return issues


def _undefined_load_reason(result: StudyResult, mobility: int) -> str:
    """Why a study's holding loads are undefined (statics gives None for every frame)."""
    if not any(p.ok for p in result.poses):
        return "no closed frame"
    if mobility > 0:
        return f"{mobility} DOF undetermined"
    return "singular at every closed frame"


def _load_entries(asm: Assembly, loads: dict[str, dict], why: str | None = None) -> dict[str, dict]:
    """report.json load entries; ``why`` is recorded for loads that came out undefined."""
    out = {}
    for jname, ld in loads.items():
        act = asm.actuators.get(jname)
        cap = act.capacity if act is not None else None
        mx = ld.get("max_abs")
        sf = cap / mx if cap is not None and mx is not None and mx > 0 else None
        out[jname] = {"unit": ld.get("unit"), "max_abs": fnum(mx), "frame": ld.get("frame"),
                      "capacity": fnum(cap), "sf": fnum(sf), "reflected_from": ld.get("reflected_from")}
        if mx is None:
            out[jname]["why"] = why or "no closed frame"
    return out


def _capacity_issues(asm: Assembly, ctx: _StudyCtx, entries: dict[str, dict]) -> list[Issue]:
    issues = []
    for jname, e in entries.items():
        cap, mx, k = e["capacity"], e["max_abs"], e["frame"]
        if cap is None or mx is None:
            continue
        j = asm.joints.get(jname)
        at = f" @f{k} ({ctx.at(int(k), [j.child] if j else None, flat=True)})" if k is not None else ""
        refl = f" (reflected from {e['reflected_from']})" if e.get("reflected_from") else ""
        load = f"{jname} holding load {sig(mx)} {e['unit']}{at}{refl}"
        if mx > cap:
            issues.append(Issue("FAIL", "over_capacity", f"{load} exceeds capacity {decl(cap)} (SF {sig(cap / mx)}) "
                                f"— bigger actuator, gearing, or counterbalance", ctx.name, k,
                                [j.child] if j else [], mx))
        elif e["sf"] is not None and e["sf"] < _SF_WARN:
            issues.append(Issue("WARN", "over_capacity", f"{load} near capacity {decl(cap)} (SF {sig(e['sf'])} "
                                f"< {_SF_WARN:g})", ctx.name, k, [j.child] if j else [], mx))
    return issues


def _structure_issues(kin: Kinematics, study) -> tuple[int, list[Issue]]:
    """(mobility, [underconstrained WARN | singular-home INFO]) for one study's drivers.

    Only the loops the drivers act on count: a loop the study never touches stays at home (and
    is reported by ``held_at_home`` if no study moves it).
    """
    driven = list(study.drive)
    driven += kin.idle_unknowns(driven)  # untouched loops: held at home, not undetermined
    m = kin.mobility(driven)
    if m > 0:
        free = ", ".join(kin.unknowns(driven))
        return m, [Issue("WARN", "underconstrained",
                         f"study '{study.name}': {m} DOF left undetermined among {free} — drive another loop "
                         f"joint, couple it, or add a pin", study.name, None, [], float(m))]
    if kin.singular_at_home(driven):
        return m, [Issue("INFO", "underconstrained",
                         f"study '{study.name}': home pose is singular for driver(s) {', '.join(study.drive)} "
                         f"(dead point / toggle); solving starts there", study.name)]
    return m, []


def _sort_issues(issues: list[Issue]) -> list[Issue]:
    """Worst first: severity, then code order, then value (bigger worse, except gaps)."""
    def key(i: Issue):
        if i.code not in _RANKED_BY_VALUE:
            badness = 0.0
        elif i.value is None or not math.isfinite(i.value):
            badness = math.inf  # undefined values (e.g. solver failures) sort first
        else:
            badness = -i.value if i.code in _SMALLER_IS_WORSE else i.value
        return (_SEVERITY_RANK[i.severity], _CODE_RANK.get(i.code, len(_CODE_RANK)), -badness)

    return sorted(issues, key=key)


def _coupling_root(asm: Assembly, joint: str) -> str:
    """The joint at the top of ``joint``'s coupling chain (the one that ultimately drives it)."""
    driver_of = {c.driven: c.driver for c in asm.couplings}
    seen = set()
    while joint in driver_of and joint not in seen:
        seen.add(joint)
        joint = driver_of[joint]
    return joint


def _held_at_home(asm: Assembly, kin: Kinematics, roles: dict[str, str], drivers: Iterable[str],
                  partial: bool = False) -> list[Issue]:
    """Joints the ``drivers`` (of the studies run, or of every study for ``mech check``) never move.

    Motion is judged structurally, not from the sampled values (a 2-frame 0→360° study samples
    one pose, but its loop is still driven): a loop moves when some driver acts on it, a coupled
    joint when the head of its coupling chain moves. Free joints are always held; in a partial
    run (``--study``) so are the drivers of the studies that were skipped.
    """
    drivers = set(drivers)
    idle = set(kin.idle_unknowns(sorted(drivers)))

    def moves(name: str) -> bool:
        role = roles.get(name)
        if role == "coupled":
            root = _coupling_root(asm, name)
            return root != name and moves(root)
        if role == "driver":
            return name in drivers
        return role == "passive" and name not in idle

    held = [asm.joints[n] for n, role in roles.items() if role != "fixed" and not moves(n)]
    if not held:
        return []
    what = ", ".join(f"{j.name} {_jv(j.home, j.kind)}" for j in held)
    by = "the selected studies" if partial else "any study"
    return [Issue("INFO", "held_at_home", f"{what} — not driven by {by}", None, None, [j.child for j in held])]


def _pair_counts(asm: Assembly) -> dict:
    """What the clearance sweep covers: the part pairs that move relative to each other (not
    ignored, not in one rigid group) and how many of them may touch (allow_contact / meshing)."""
    groups, meshing, names = asm.rigid_groups(), asm.meshing_pairs(), list(asm.parts)
    checked = allowed = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = frozenset((a, b))
            if key in asm.ignored or groups[a] == groups[b]:
                continue
            checked += 1
            allowed += key in asm.allowed or key in meshing
    return {"required": asm.clearance, "pairs_checked": checked, "allowed_contact": allowed}


def build_report(asm: Assembly, kin: Kinematics, props: dict[str, MassProps], results: dict[str, StudyResult],
                 sweeps: dict[str, SweepResult], loads: dict[str, dict[str, dict]], home: list[PairResult],
                 *, roles: dict[str, str], viewer_url: str | None, prev: dict | None = None,
                 partial: dict | None = None, out_dir: str | None = None) -> dict:
    """The report dict (report.json) for a valid, analyzed assembly; see module docstring.

    ``viewer_url`` is the viewer base URL for this mechanism (``…/mech.html?m=<slug>``); the report
    stores the *targeted* URL (first FAIL/WARN issue, else a ghosted quad view). ``prev`` is the
    previous report, used for the Δprev comparison. ``partial`` describes a narrowed run
    (``{"studies": run, "skipped": not run, "frames": override | None}``, None for a full run);
    ``out_dir`` is the export root when it is not the ``<repo>/output`` the viewer serves.
    """
    issues: list[Issue] = [Issue("WARN", code, msg) for code, msg in asm.validate_warnings()]
    mass = assembly_props(asm, props)
    studies_out, mobility = [], {}
    for name, res in results.items():
        ctx = _StudyCtx(asm, name, res, kin)
        m, structural = _structure_issues(kin, res.study)
        mobility[name] = m
        issues += structural
        issues += _loop_issues(asm, kin, ctx)
        issues += _limit_issues(asm, ctx)
        issues += _jump_issues(asm, ctx)
        issues += _singular_issues(ctx, any(i.severity == "INFO" for i in structural))
        sweep = sweeps.get(name)
        if sweep is not None:
            issues += _clearance_issues(asm, ctx, sweep)
        entries = _load_entries(asm, loads.get(name, {}), _undefined_load_reason(res, m))
        issues += _capacity_issues(asm, ctx, entries)
        studies_out.append(_study_entry(asm, ctx, sweep, entries))
    issues += _home_issues(asm, home, swept=bool(sweeps))
    issues += _contact_issues(asm, sweeps, home)
    issues += [Issue("INFO", "gear_mesh", f"{'/'.join(sorted(p))} meshing (gear/rack coupling): contact allowed, "
                     f"interference still checked", parts=sorted(p)) for p in sorted(asm.meshing_pairs(), key=sorted)]
    if results:
        drivers = {d for res in results.values() for d in res.study.drive}
    else:  # mech check: judged over every study's drivers
        drivers = {n for n, role in roles.items() if role == "driver"}
    issues += _held_at_home(asm, kin, roles, drivers, partial=bool(partial))

    counts = {r: 0 for r in ("driver", "coupled", "passive", "free", "fixed")}
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1
    report = {
        "name": asm.name,
        "status": "PASS",
        "params": dict(asm.params),
        "partial": partial,
        "parts": len(asm.parts),
        "joints": counts,
        "roles": dict(roles),
        "joint_kinds": {n: j.kind for n, j in asm.joints.items()},
        "pins": [p.name for p in asm.pins],
        "mobility": mobility,
        "mass": {"total_g": mass.mass_kg * 1000.0, "com_mm": mass.com.tolist(),
                 "parts": {n: mp.mass_kg * 1000.0 for n, mp in props.items()}},
        "issues": _sort_issues(issues),
        "targets": [],
        "studies": studies_out,
        "clearance": _pair_counts(asm),
        "home_min_clearance": _home_min_clearance(asm, home),
        "viewer_url": viewer_url,
        "out_dir": out_dir,
    }
    _finalize(report, prev)
    return report


def build_check_report(asm: Assembly, kin: Kinematics, props: dict[str, MassProps], studies: list,
                       home: list[PairResult], *, roles: dict[str, str]) -> dict:
    """Report for ``mech check``: no studies are run, but each study's mobility and structural
    issues (underconstrained / singular home) are included next to the home-pose findings."""
    report = build_report(asm, kin, props, {}, {}, {}, home, roles=roles, viewer_url=None)
    extra: list[Issue] = []
    for study in studies:
        m, issues = _structure_issues(kin, study)
        report["mobility"][study.name] = m
        extra += issues
    report["issues"] += [i.to_dict() for i in extra]
    _finalize(report, None)
    return report


def _home_min_clearance(asm: Assembly, home: list[PairResult]) -> dict | None:
    """The home pose's signed clearance, as a sweep's ``min_clearance``: −(mean overlap depth) of the
    deepest interference between parts that move relative to each other (any pair: joined,
    allowed and meshing included), else the smallest gap among the pairs clearance applies to
    (not joined/allowed/meshing)."""
    groups = asm.rigid_groups()
    hits = [r for r in home if r.status == "interference" and groups.get(r.a) != groups.get(r.b)]
    if hits:
        def depth(r: PairResult) -> float:
            d = getattr(r, "depth", None)
            return d if d is not None and math.isfinite(d) else 0.0

        r = max(hits, key=depth)
        return {"parts": [r.a, r.b], "value": -depth(r) if depth(r) > 0 else 0.0}
    rows = [r for r in home if r.distance is not None and not (r.joined or r.allowed or r.meshing)]
    if not rows:
        return None
    r = min(rows, key=lambda r: r.distance)
    return {"parts": [r.a, r.b], "value": r.distance}


def _study_entry(asm: Assembly, ctx: _StudyCtx, sweep: SweepResult | None, loads: dict[str, dict]) -> dict:
    """report.json study entry. Joint ranges and probe stats cover the closed frames only (an open
    frame is no configuration of the mechanism), with loop unknowns re-wrapped across open gaps
    — the same series the targets use (``targets.joint_series`` / ``path_length``)."""
    res = ctx.result
    ranges = {}
    for j, joint in asm.joints.items():
        if joint.kind == "fixed" or not any(j in p.q for p in res.poses):
            continue
        ks, vals = joint_series(asm, res, j)
        if ks:
            ranges[j] = [float(vals.min()), float(vals.max())]
    probes = {}
    for pname in res.probes:
        ks, P = probe_series(res, pname)
        if not ks:
            continue
        probes[pname] = {"min": P.min(axis=0).tolist(), "max": P.max(axis=0).tolist(), "start": P[0].tolist(),
                         "end": P[-1].tolist(), "path_mm": path_length(res, pname)}
    mc = None
    if sweep is not None and sweep.min_clearance is not None:
        a, b, value, frame = sweep.min_clearance
        frac = _subframe(sweep, a, b, value, frame)
        mc = {"parts": [a, b], "value": value, "frame": frame, "at": ctx.values(frame, [a, b], frac)}
        if frac is not None:
            mc["subframe"] = frac
    residuals = [p.residual for p in res.poses]
    return {"name": ctx.name, "frames": len(res.poses), "joint_ranges": ranges, "probes": probes,
            "min_clearance": mc, "loads": loads, "max_residual": max(residuals) if residuals else 0.0}


def _subframe(sweep: SweepResult, a: str, b: str, value: float, frame: int) -> float | None:
    """The fractional frame of the sweep result behind a min clearance, when it was found between
    two frames (filed at the nearer frame); None for a frame's own measurement."""
    if not 0 <= frame < len(sweep.per_frame):
        return None
    for r in sweep.per_frame[frame]:
        frac = getattr(r, "at", None)
        if {r.a, r.b} != {a, b} or frac is None or abs(frac - round(frac)) <= 1e-9:
            continue
        depth = getattr(r, "depth", None)
        measured = -depth if r.status == "interference" and depth is not None else r.distance
        if measured is not None and math.isclose(measured, value, rel_tol=1e-9, abs_tol=1e-12):
            return float(frac)
    return None


def invalid_report(name, errors: list[str], params: dict, partial: dict | None = None) -> dict:
    """Report for a model that could not be analyzed (status INVALID, one invalid_model FAIL per error)."""
    issues = [Issue("FAIL", "invalid_model", str(e)) for e in errors]
    return jsonable({
        "name": str(name), "status": "INVALID", "params": dict(params or {}), "partial": partial, "parts": None,
        "joints": {}, "roles": {}, "joint_kinds": {}, "pins": [], "mobility": {}, "mass": None, "issues": issues,
        "targets": [], "studies": [], "clearance": None, "home_min_clearance": None, "viewer_url": None,
        "out_dir": None, "delta_prev": None,
    })


def _target_miss_message(t: dict) -> str:
    """``label: metric value < min X (margin −d)`` — the violated bound and the signed margin
    are always named (the value is rounded, the margin says by how much it misses)."""
    v = t.get("value")
    if v is None:
        why = t.get("error") or "undefined"
        return f"{t['label']}: {t['metric']} n/a ({why})"
    lo, hi = t.get("min"), t.get("max")
    if lo is None and hi is None:
        return f"{t['label']}: {t['metric']} {sig(v)}"
    below = lo is not None and (hi is None or abs(v - lo) <= abs(v - hi))  # outside: the nearer bound is crossed
    bound = f"< min {decl(lo)}" if below else f"> max {decl(hi)}"
    margin = f" (margin {sig(t['margin'])})" if t.get("margin") is not None else ""
    return f"{t['label']}: {t['metric']} {sig(v)} {bound}{margin}"


def attach_targets(report: dict, targets: list[dict], prev: dict | None = None) -> dict:
    """Store target results in ``report``, add a ``target_miss`` issue (with the target's severity)
    for each missed one, then recompute status, targeted viewer URL and Δprev. Returns ``report``."""
    report["targets"] = jsonable(targets)
    kept = [i for i in report["issues"] if i["code"] != "target_miss"]
    misses = [Issue(t["severity"], "target_miss", _target_miss_message(t), t.get("worst_study") or t.get("study"),
                    value=t.get("value")).to_dict() for t in report["targets"] if t.get("met") is False]
    report["issues"] = kept + misses
    _finalize(report, prev)
    return report


def _finalize(report: dict, prev: dict | None) -> None:
    """Sort issues, derive status, targeted viewer URL and Δprev; make everything JSON-native."""
    issues = [Issue(**i) if isinstance(i, dict) else i for i in report["issues"]]
    issues = _sort_issues(issues)
    report["issues"] = [i.to_dict() for i in issues]
    sevs = {i.severity for i in issues}
    report["status"] = "FAIL" if "FAIL" in sevs else "WARN" if "WARN" in sevs else "PASS"
    if report.get("viewer_url"):
        report["viewer_url"] = _targeted_url(report["viewer_url"], report["issues"])
    for k, v in jsonable({k: v for k, v in report.items() if k != "issues"}).items():
        report[k] = v
    report["delta_prev"] = _delta_prev(report, prev)


def _targeted_url(url: str, issues: list[dict]) -> str:
    """Viewer URL aimed at the first FAIL/WARN issue that shows parts (``&issue=i&ui=0``), else
    a ghosted quad overview (``&ghost=6&layout=quad&ui=0``). Idempotent on its own output."""
    parts = urlsplit(url)
    keep = [(k, v) for k, v in parse_qsl(parts.query) if k not in ("issue", "ghost", "layout", "ui")]
    bad = [n for n, i in enumerate(issues) if i["severity"] in ("FAIL", "WARN")]
    if bad:
        idx = next((n for n in bad if issues[n]["parts"]), bad[0])
        extra = [("issue", str(idx)), ("ui", "0")]
    else:
        extra = [("ghost", "6"), ("layout", "quad"), ("ui", "0")]
    query = urlencode(keep + extra, safe=",:")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


# ================================================================================================
# Δprev
# ================================================================================================


_JOINT_SUBJECT = {"over_capacity", "joint_limit", "branch_jump"}  # issue lines that start with the joint
_UNSET = "(unset)"  # a build() param one of the two runs did not have


def issue_key(issue: dict) -> tuple:
    """Identity of an issue across studies and runs: code + the subject its line names — the
    joint of over_capacity / joint_limit / branch_jump, the pin of loop_open, else the part pair
    (partless: the text before ':', or the first quoted name of a validate warning)."""
    code, message = issue["code"], issue.get("message", "")
    if code in _JOINT_SUBJECT and message:
        return code, message.split(" ", 1)[0]
    if code == "loop_open":
        pin = re.match(r"loop (\S+) ", message)
        if pin:
            return code, pin.group(1)
    if issue.get("parts"):
        return code, tuple(sorted(issue["parts"]))
    head, colon, _ = message.partition(":")
    if not colon:  # validate_warnings messages: the first quoted name is the subject (joint / pin)
        quoted = re.search(r"'([^']+)'", message)
        head = quoted.group(1) if quoted else message
    return code, head


def _key_label(key: tuple) -> str:
    code, what = key
    what = "/".join(what) if isinstance(what, tuple) else what
    return f"{_SHORT.get(code, code)} {what}".strip()


def _scope(report: dict) -> dict[str, Any]:
    """study name -> frame count: what a run sampled (two runs compare only on shared entries)."""
    return {s["name"]: s.get("frames") for s in report.get("studies") or []}


def _not_compared(old: dict[str, Any], now: dict[str, Any]) -> dict[str, str]:
    """Studies Δprev leaves out: run in only one of the two runs, or at another frame count."""
    out = {name: "not run" if name not in now else f"{n}→{now[name]} frames"
           for name, n in old.items() if now.get(name, _UNSET) != n}
    out.update({name: "new study" for name in now if name not in old})
    return out


def _overall_min_clearance(report: dict, names: set[str] | None = None) -> float | None:
    """Min clearance over the studies (only ``names`` when given); the home pose's when no study ran."""
    studies = [s for s in report.get("studies") or [] if names is None or s["name"] in names]
    vals = [s["min_clearance"]["value"] for s in studies
            if s.get("min_clearance") and s["min_clearance"].get("value") is not None]
    if not vals and names is None and report.get("home_min_clearance"):
        vals = [report["home_min_clearance"].get("value")]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _param_changes(old: dict, now: dict) -> dict[str, list]:
    keys = list(old) + [k for k in now if k not in old]
    return {k: [old.get(k, _UNSET), now.get(k, _UNSET)] for k in keys if old.get(k, _UNSET) != now.get(k, _UNSET)}


def _delta_prev(report: dict, prev: dict | None) -> dict:
    """Changes since the previous report (only what changed at 3 significant figures).

    Only like is compared with like: studies run in both reports at the same frame count
    (others are listed under ``not_compared``) plus study-independent findings, and targets
    evaluated over the same studies. So a ``--study``/``--frames`` run never calls an issue of a
    study it skipped or sampled differently "fixed", and the status is compared only when both
    runs sampled the same studies.
    """
    if not prev:
        return {"first_run": True}
    try:
        if prev.get("status") == "INVALID":
            return {"prev_status": "INVALID"}
        out: dict[str, Any] = {}
        skipped = _not_compared(_scope(prev), _scope(report))
        same = not skipped
        common = {name for name in _scope(report) if name not in skipped}
        if same and prev.get("status") != report["status"]:
            out["status"] = [prev.get("status"), report["status"]]
        params = _param_changes(prev.get("params") or {}, report.get("params") or {})
        if params:
            out["params"] = params
        old_t = {t["label"]: t for t in prev.get("targets") or []}
        now_t = {t["label"]: t for t in report.get("targets") or []}
        comparable = [label for label, t in now_t.items() if label in old_t
                      and (same or t.get("metric") == "mass_g" or (t.get("study") is not None and t["study"] in common))]

        def keys(r: dict) -> set[tuple]:
            found = set()
            for i in r.get("issues") or []:
                if i.get("severity") not in ("FAIL", "WARN"):
                    continue
                if i["code"] == "target_miss":
                    if issue_key(i)[1] not in comparable:
                        continue  # added/removed, or evaluated over other studies: reported apart
                elif i.get("study") is not None and i["study"] not in common:
                    continue
                found.add(issue_key(i))
            return found

        old, new = keys(prev), keys(report)
        if old - new:
            out["fixed"] = sorted(_key_label(k) for k in old - new)
        if new - old:
            out["new"] = sorted(_key_label(k) for k in new - old)
        added = [label for label in now_t if label not in old_t]
        removed = [label for label in old_t if label not in now_t]
        if added:
            out["targets_added"] = added
        if removed:
            out["targets_removed"] = removed
        pairs = {"mass_g": ((prev.get("mass") or {}).get("total_g"), (report.get("mass") or {}).get("total_g"))}
        if same or common:
            names = None if same else common
            pairs = {"min_clearance": (_overall_min_clearance(prev, names), _overall_min_clearance(report, names)),
                     **pairs}
        for name, (a, b) in pairs.items():
            if sig(a) != sig(b):
                out[name] = [a, b]
        changed = {label: [old_t[label].get("value"), now_t[label].get("value")] for label in comparable
                   if sig(old_t[label].get("value")) != sig(now_t[label].get("value"))}
        if changed:
            out["targets"] = changed
        if skipped:
            out["not_compared"] = skipped
        return jsonable(out)
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"unreadable": True}


def _pval(v) -> str:
    """A build() param value as the user would type it."""
    if v == _UNSET:
        return "unset"
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return decl(v)
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _listed(items: list[str], verbose: bool, n: int = 3) -> str:
    if verbose or len(items) <= n:
        return ", ".join(items)
    return f"{', '.join(items[:n])} (+{len(items) - n} more — --verbose)"


def _format_delta(d: dict | None, verbose: bool = False) -> str:
    if d is None or d.get("first_run"):
        return "Δprev: first run"
    if d.get("unreadable"):
        return "Δprev: previous report.json unreadable"
    if d.get("prev_status") == "INVALID":
        return "Δprev: previous run was INVALID"
    bits = []
    if "status" in d:
        bits.append(f"status {d['status'][0]}→{d['status'][1]}")
    if "params" in d:
        bits.append("params " + _listed([f"{k} {_pval(a)}→{_pval(b)}" for k, (a, b) in d["params"].items()],
                                         verbose))
    for what in ("fixed", "new"):
        if what in d:
            bits.append(f"{what} {_listed(d[what], verbose)}")
    if "targets_added" in d:
        bits.append(f"added target {_listed(d['targets_added'], verbose)}")
    if "targets_removed" in d:
        bits.append(f"removed target {_listed(d['targets_removed'], verbose)}")
    if "min_clearance" in d:
        a, b = d["min_clearance"]
        bits.append(f"min clearance {sig(a)}→{sig(b)} mm")
    if "mass_g" in d:
        a, b = d["mass_g"]
        bits.append(f"mass {sig(a)}→{sig(b)} g")
    for label, (a, b) in (d.get("targets") or {}).items():
        bits.append(f"{label} {sig(a)}→{sig(b)}")
    skipped = d.get("not_compared")
    note = "not compared: " + ", ".join(f"{name} ({why})" for name, why in skipped.items()) if skipped else ""
    if not bits:
        return "Δprev: no change" + (f" ({note})" if note else "")
    return "Δprev: " + " · ".join(bits + ([note] if note else []))


# ================================================================================================
# summary
# ================================================================================================


@dataclass
class _Line:
    text: str
    tally: str  # what the overflow line counts it as, e.g. "WARN tight", "INFO", "load"


def _dedup(report: dict) -> list[tuple[dict, list[str]]]:
    """Issues deduplicated on (code, part pair) across studies: (worst issue, study names)."""
    groups: dict[tuple, tuple[dict, list[str]]] = {}
    for i in report["issues"]:  # already worst-first, so the first of a group is its worst
        key = issue_key(i)
        if key not in groups:
            groups[key] = (i, [])
        if i.get("study") and i["study"] not in groups[key][1]:
            groups[key][1].append(i["study"])
    return list(groups.values())


def _issue_line(issue: dict, studies: list[str], multi: bool) -> _Line:
    short = _SHORT.get(issue["code"], issue["code"])
    tag = f" [{', '.join(studies)}]" if multi and studies else ""
    tally = "INFO" if issue["severity"] == "INFO" else f"{issue['severity']} {short}"
    return _Line(f"{issue['severity']:<4} {short} {issue['message']}{tag}", tally)


def _header(report: dict, title: str = "mech") -> str:
    j = report.get("joints") or {}
    total = sum(j.values())
    detail = ", ".join(f"{n} {role}" for role, n in j.items() if n)
    bits = [f"{report.get('parts')} parts", f"{total} joint{'s' if total != 1 else ''}" + (f" ({detail})" if detail else "")]
    n_loops = len(report.get("pins") or [])
    if n_loops:
        bits.append(f"{n_loops} loop{'s' if n_loops > 1 else ''}")
    mass = report.get("mass")
    if mass:
        bits.append(f"{sig(mass['total_g'])} g · CoG {vec(mass['com_mm'])}")
    return f"{title} {report['name']} — {report['status']}{_partial_tag(report)}   " + " · ".join(bits)


def _partial_tag(report: dict) -> str:
    """`` (partial run: --study back, skipped sweep; --frames 12)`` — what a narrowed run left out."""
    p = report.get("partial")
    if not p:
        return ""
    bits = []
    if p.get("skipped") or p.get("frames") is None:
        run = f"--study {', '.join(p.get('studies') or [])}"
        bits.append(run + (f", skipped {', '.join(p['skipped'])}" if p.get("skipped") else ""))
    if p.get("frames") is not None:
        bits.append(f"--frames {p['frames']}")
    return f" (partial run: {'; '.join(bits)})"


def _loops_line(report: dict) -> str:
    codes = {i["code"] for i in report["issues"]}
    pins = report.get("pins") or []
    mob = report.get("mobility") or {}
    bad = bool({"loop_open", "branch_jump", "singular_pose"} & codes) or any(v > 0 for v in mob.values())
    bits = []
    if pins:
        res = [s["max_residual"] for s in report.get("studies") or []]
        name = f"loop {pins[0]}" if len(pins) == 1 else f"loops {', '.join(pins)}"
        if not res:
            bits.append(f"{name} (no studies run)")
        else:
            worst = None if None in res else max(res)  # None = a residual was not finite
            bits.append(f"{name} {'OPEN' if 'loop_open' in codes else 'closed'} (max {sig(worst)} mm)")
        jumps = sum(1 for i in report["issues"] if i["code"] == "branch_jump")
        bits.append(f"{jumps} branch jump{'s' if jumps != 1 else ''}" if jumps else "no branch jumps")
        singular = sum(int(i.get("value") or 0) for i in report["issues"] if i["code"] == "singular_pose")
        if singular:
            bits.append(f"{singular} singular frame{'s' if singular != 1 else ''}")
    else:
        bits.append("open chain (no loops)")
    if mob:
        vals = set(mob.values())
        bits.append(f"mobility {vals.pop()}" if len(vals) == 1 else
                    "mobility " + ", ".join(f"{s} {m}" for s, m in mob.items()))
    return f"{'!!' if bad else 'OK':<4} " + " · ".join(bits)


def _clearance_line(report: dict) -> str:
    """``clearance min 2.06 mm (a/b @f38 j=25.1°) · required 0.3 · 21 pairs checked · 2 allowed-contact
    pairs`` — the smallest gap among the pairs the clearance rule applies to, always shown."""
    kinds = report.get("joint_kinds") or {}
    studies = report.get("studies") or []
    best = None
    for s in studies:
        mc = s.get("min_clearance")
        if mc and mc.get("value") is not None and (best is None or mc["value"] < best[0]["value"]):
            best = (mc, s["name"])
    if best:
        mc, study = best
        at = ", ".join(f"{j}={_jv(v, kinds.get(j))}" for j, v in (mc.get("at") or {}).items())
        sub = mc.get("subframe")
        where = f"between f{math.floor(sub)}–f{math.floor(sub) + 1}" if sub is not None else f"@f{mc['frame']}"
        where += (f" {at}" if at else "") + (f" [{study}]" if len(studies) > 1 else "")
        where += "; negative = overlap depth" if mc["value"] < 0 else ""  # the parts interpenetrate
        bits = [f"clearance min {sig(mc['value'])} mm ({'/'.join(mc['parts'])} {where})"]
    elif not studies and report.get("home_min_clearance"):
        hc = report["home_min_clearance"]
        neg = "; negative = overlap depth" if hc["value"] is not None and hc["value"] < 0 else ""
        bits = [f"clearance min {sig(hc['value'])} mm ({'/'.join(hc['parts'])} at home{neg})"]
    else:
        bits = ["clearance min n/a (no unjoined, non-allowed pair to measure)"]
    c = report.get("clearance") or {}
    if c.get("required") is not None:
        bits.append(f"required {decl(c['required'])}")
    if c.get("pairs_checked") is not None:
        n = c["pairs_checked"]
        bits.append(f"{n} pair{'s' if n != 1 else ''} checked")
    if c.get("allowed_contact"):
        n = c["allowed_contact"]
        bits.append(f"{n} allowed-contact pair{'s' if n != 1 else ''}")
    return " · ".join(bits)


def _load_lines(report: dict) -> list[_Line]:
    best: dict[str, tuple[dict, str]] = {}
    for s in report.get("studies") or []:
        for j, e in (s.get("loads") or {}).items():
            cur = best.get(j)
            mx = e.get("max_abs")
            if cur is None or (mx is not None and (cur[0].get("max_abs") is None or mx > cur[0]["max_abs"])):
                best[j] = (e, s["name"])
    multi = len(report.get("studies") or []) > 1
    lines, unloaded = [], []
    for j, (e, study) in best.items():
        if e.get("max_abs") is None:
            lines.append(_Line(f"load {j} n/a ({e.get('why') or 'no closed frame'})", "load"))
            continue
        if e["max_abs"] == 0:  # gravity along the joint axis (or balanced): nothing to hold
            if e.get("capacity") is not None:
                unloaded.append(j)  # rated actuators are named once, without an arbitrary frame
            continue
        where = f" @f{e['frame']}" if e.get("frame") is not None else ""
        where += f" ({study})" if multi else ""
        refl = f" (reflected from {e['reflected_from']})" if e.get("reflected_from") else ""
        if e.get("capacity") is None:
            tail = " · no actuator capacity"
        else:
            tail = f" · capacity {decl(e['capacity'])} → SF {sig(e['sf']) if e.get('sf') is not None else '∞'}"
        lines.append(_Line(f"load {j} max {sig(e['max_abs'])} {e.get('unit') or ''}{where}{refl}{tail}", "load"))
    if unloaded:
        axes = "axis" if len(unloaded) == 1 else "axes"
        lines.append(_Line(f"no gravity load on {', '.join(unloaded)} ({axes} ∥ g or balanced)", "load"))
    return lines


def _bounds(t: dict) -> str:
    lo, hi, v = t.get("min"), t.get("max"), sig(t.get("value"))
    if lo is not None and hi is not None:
        return f"{decl(lo)} ≤ {v} ≤ {decl(hi)}"
    if lo is not None:
        return f"{v} ≥ {decl(lo)}"
    if hi is not None:
        return f"{v} ≤ {decl(hi)}"
    return v


def _targets_line(report: dict) -> str | None:
    ts = report.get("targets") or []
    if not ts:
        return None
    met = [t for t in ts if t.get("met") is True]
    unevaluated = [t["label"] for t in ts if t.get("met") is None]
    bits = [f"targets {len(met)}/{len(ts) - len(unevaluated)}"] + [f"{t['label']} {t['metric']} {_bounds(t)}"
                                                                   for t in met]
    if unevaluated:
        bits.append(f"not evaluated: {', '.join(unevaluated)}")
    return " · ".join(bits)


def _ranges_line(report: dict) -> str | None:
    kinds = report.get("joint_kinds") or {}
    joints: dict[str, list[float]] = {}
    probes: dict[str, dict] = {}
    for s in report.get("studies") or []:
        for j, (lo, hi) in (s.get("joint_ranges") or {}).items():
            cur = joints.setdefault(j, [lo, hi])
            cur[0], cur[1] = min(cur[0], lo), max(cur[1], hi)
        for p, e in (s.get("probes") or {}).items():
            cur = probes.setdefault(p, {"min": list(e["min"]), "max": list(e["max"]), "path": e["path_mm"]})
            cur["min"] = [min(a, b) for a, b in zip(cur["min"], e["min"])]
            cur["max"] = [max(a, b) for a, b in zip(cur["max"], e["max"])]
            cur["path"] = max(cur["path"], e["path_mm"])
    bits = [f"{j} {_jrange(lo, hi, kinds.get(j))}" for j, (lo, hi) in joints.items() if hi - lo > _STILL]
    bits += [f"probe {p} Δ{vec(np.subtract(e['max'], e['min']))} path {sig(e['path'])} mm" for p, e in probes.items()]
    return "ranges " + " · ".join(bits) if bits else None


def _view_line(report: dict) -> str:
    name = slug(report["name"])
    if report.get("partial"):
        return (f"view: not exported — partial run; {name}.mech keeps the last full run "
                f"(drop --study/--frames to update it)")
    if not report.get("viewer_url"):
        return f"view: not exported (drop --no-export to write output/{name}.mech)"
    if report.get("out_dir"):  # not the <repo>/output the viewer and a bare `mech shot` read
        d = report["out_dir"]
        return (f"view: exported to {d} (the viewer serves <repo>/output only) · "
                f"shot: uv run mech shot {name} --output-dir {d}")
    return f"view {report['viewer_url']} · shot: uv run mech shot {name}"


# Over budget, lines are kept in this priority: (tier, how many — None = all remaining).
_SUMMARY_PRIORITY = [("FAIL", 1), ("target_miss", 1), ("WARN", 1), ("loops", 1), ("clearance", 1), ("targets", 1),
                     ("dprev", 1),
                     ("FAIL", None), ("target_miss", None), ("WARN", None), ("loads", 1), ("ranges", 1),
                     ("loads", None), ("INFO", None)]
_CHECK_PRIORITY = [("FAIL", 1), ("WARN", 1), ("mobility", 1), ("home", 1), ("FAIL", None), ("WARN", None),
                   ("roles", 1), ("INFO", None)]


def _fit(tiers: dict[str, list[_Line]], order: list[str], priority: list[tuple[str, int | None]],
         fixed_top: list[str], fixed_bottom: list[str], budget: int | None) -> list[str]:
    """Lay out tiers in display ``order``; over budget, keep each tier's first lines by ``priority``
    (dropping from the bottom of each tier) and add a ``(+N more: …)`` tally line."""
    total = len(fixed_top) + len(fixed_bottom) + sum(len(v) for v in tiers.values())
    keep = {name: len(lines) for name, lines in tiers.items()}
    dropped: list[_Line] = []
    if budget is not None and total > budget:
        room = budget - len(fixed_top) - len(fixed_bottom) - 1  # one line for the tally
        keep = dict.fromkeys(tiers, 0)
        for name, n in priority:
            if name not in tiers:
                continue
            want = len(tiers[name]) if n is None else min(n, len(tiers[name]))
            add = max(0, min(want - keep[name], room))
            keep[name] += add
            room -= add
        for name, lines in tiers.items():
            dropped += lines[keep[name]:]
    out = list(fixed_top)
    for name in order:
        out += [line.text for line in tiers.get(name, [])[:keep.get(name, 0)]]
    if dropped:
        counts: dict[str, int] = {}
        for line in dropped:
            counts[line.tally] = counts.get(line.tally, 0) + 1
        what = ", ".join(f"{n} {t}" for t, n in counts.items())
        out.append(f"(+{len(dropped)} more: {what} — --verbose)")
    return out + list(fixed_bottom)


def format_summary(report: dict, verbose: bool = False) -> str:
    """The compact text summary (≤ 15 lines unless ``verbose``); see the module docstring."""
    budget = None if verbose else SUMMARY_LINES
    multi = len(report.get("studies") or []) > 1
    tiers: dict[str, list[_Line]] = {"FAIL": [], "target_miss": [], "WARN": [], "INFO": []}
    for issue, studies in _dedup(report):
        tier = "target_miss" if issue["code"] == "target_miss" else issue["severity"]
        tiers[tier].append(_issue_line(issue, studies, multi))
    if report["status"] == "INVALID":
        n = len(report["issues"])
        header = f"mech {report['name']} — INVALID   {n} error{'s' if n != 1 else ''}"
        footer = "nothing analyzed — fix the errors above and re-run"
        return "\n".join(_fit(tiers, ["FAIL"], [("FAIL", None)], [header], [footer], budget))
    tiers["loops"] = [_Line(_loops_line(report), "loop status")]
    tiers["clearance"] = [_Line(_clearance_line(report), "clearance")]
    tiers["loads"] = _load_lines(report)
    for name, tally, text in (("targets", "targets", _targets_line(report)), ("ranges", "ranges", _ranges_line(report)),
                              ("dprev", "Δprev", _format_delta(report.get("delta_prev"), verbose))):
        tiers[name] = [_Line(text, tally)] if text else []
    order = ["FAIL", "target_miss", "WARN", "INFO", "loops", "clearance", "loads", "targets", "ranges", "dprev"]
    return "\n".join(_fit(tiers, order, _SUMMARY_PRIORITY, [_header(report)], [_view_line(report)], budget))


def format_check(report: dict, verbose: bool = False) -> str:
    """Summary for ``mech check``: header, roles, mobility per study, home clearance, issues."""
    if report["status"] == "INVALID":
        return format_summary(report, verbose)
    budget = None if verbose else SUMMARY_LINES
    tiers: dict[str, list[_Line]] = {"FAIL": [], "WARN": [], "INFO": []}
    for issue, studies in _dedup(report):
        tiers[issue["severity"]].append(_issue_line(issue, studies, len(report.get("mobility") or {}) > 1))
    roles = report.get("roles") or {}
    tiers["roles"] = [_Line("roles " + " · ".join(f"{j} {r}" for j, r in roles.items()), "roles")] if roles else []
    mob = report.get("mobility") or {}
    tiers["mobility"] = [_Line("studies " + " · ".join(f"{s} mobility {m}" for s, m in mob.items())
                               if mob else "studies none (nothing to drive)", "mobility")]
    hc = report.get("home_min_clearance")
    home = (f"home min clearance {sig(hc['value'])} mm ({'/'.join(hc['parts'])}"
            f"{'; negative = overlap depth' if (hc.get('value') or 0) < 0 else ''})" if hc
            else "home min clearance: no unjoined pairs within reach")
    tiers["home"] = [_Line(home, "home")]
    order = ["FAIL", "WARN", "INFO", "roles", "mobility", "home"]
    footer = "next: uv run mech run <script> to run the studies"
    return "\n".join(_fit(tiers, order, _CHECK_PRIORITY, [_header(report, "mech check")], [footer], budget))
