"""Design-target evaluation (spec §2.3, §4.8).

A target states design intent as a metric that must land in [min, max] (bounds compared with a
1e-9·max(1, |bound|) float-noise allowance). String metrics are evaluated per study on the
study's *closed* frames (``pose.ok``) — a pose the solver could not close is not a
configuration of the mechanism, and its loop_open issue is reported separately — and with
``study=None`` the worst value over all run studies counts. A revolute loop unknown is
re-wrapped by whole turns across each run of open frames (its value there is a least-squares
guess that may have drifted by 360°). For the motion-extent metrics (``span``, ``path``,
``delta``, ``rot``, ``angle``) a study in which the joint/probe/part does not move at all says
nothing about how far it moves, so such studies are skipped while another study moves it (e.g.
``span:j_out`` is not 0 just because a second study swings an unrelated arm). ``mass_g`` and
callables ``f(report_dict) -> float`` are study-independent; a callable's dict carries each
study's per-frame ``series`` (see ``callable_view``).

Partial runs (``report["partial"]``: a ``--study`` filter skipped studies) never guess about the
studies that did not run: a target of a skipped study, or a ``study=None`` target that no run
study defines (or moves), or a callable that fails on the partial report, is *not evaluated*
(``met`` None, left out of the status). A miss on the studies that ran is a real miss — more
studies can only make the worst value worse.

Result rows: ``{"label", "metric", "value", "min", "max", "margin", "met", "severity", "study",
"worst_study", "error"}`` — ``study`` is the target's declared study, ``worst_study`` where the
counted value came from, ``margin`` the signed distance to the nearest bound (negative = outside),
``met`` None when not evaluated, and ``error`` says why ``value`` is undefined or not evaluated.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .geom import fnum

if TYPE_CHECKING:
    from .assembly import Assembly, Target
    from .motion import StudyResult

__all__ = ["evaluate_targets", "metric_name", "callable_view", "closed_runs", "joint_series", "probe_series", "path_length",
           "rotation_deg", "BOUND_RTOL"]

BOUND_RTOL = 1e-9  # float-noise allowance on target bounds, relative to max(1, |bound|)
_STILL = 1e-9  # an extent below this counts as "did not move"
_AXES = {"x": 0, "y": 1, "z": 2}
_EXTENT_METRICS = ("span", "path", "delta", "rot", "angle")  # 0 in a study that doesn't move it


class _Undefined(Exception):
    """The metric has no value for this study (message says why)."""


def metric_name(metric) -> str:
    """Display form of a metric: the string itself, or ``name()`` for a callable."""
    return metric if isinstance(metric, str) else f"{getattr(metric, '__name__', 'callable')}()"


# ================================================================================================
# closed-frame series (shared with report.py so ranges, probe stats and targets agree)
# ================================================================================================


def closed_runs(res: StudyResult) -> list[list[int]]:
    """Runs of consecutive closed frames (``pose.ok``), in frame order."""
    runs: list[list[int]] = []
    for k, pose in enumerate(res.poses):
        if not pose.ok:
            continue
        if runs and runs[-1][-1] == k - 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


def _loop_unknown(asm: Assembly, res: StudyResult, joint: str) -> bool:
    """A revolute joint the study solves for (not driven, not coupled): its value can wrap."""
    j = asm.joints[joint]
    coupled = {c.driven for c in asm.couplings}
    return j.kind == "revolute" and joint not in (res.study.drive or {}) and joint not in coupled


def joint_series(asm: Assembly, res: StudyResult, joint: str) -> tuple[list[int], np.ndarray]:
    """(closed frame indices, joint values there).

    A revolute loop unknown is re-wrapped by whole turns at each gap of open frames so the next
    run of closed frames continues from the previous one (the solver's guess drifts while the
    loop is open); driven and coupled values are prescribed and never shifted. A fixed joint is 0.
    """
    runs = closed_runs(res)
    ks = [k for run in runs for k in run]
    if asm.joints[joint].kind == "fixed":
        return ks, np.zeros(len(ks))
    values = []
    wrap = _loop_unknown(asm, res, joint)
    shift, last = 0.0, None
    for run in runs:
        v = np.array([float(res.poses[k].q[joint]) for k in run])
        if wrap and last is not None:
            shift = 360.0 * round((last - (v[0] + shift)) / 360.0) + shift
        v = v + shift
        last = float(v[-1])
        values.append(v)
    return ks, (np.concatenate(values) if values else np.zeros(0))


def probe_series(res: StudyResult, probe: str) -> tuple[list[int], np.ndarray]:
    """(closed frame indices, probe positions there as an (n, 3) array)."""
    ks = [k for run in closed_runs(res) for k in run]
    P = np.asarray(res.probes[probe], dtype=float).reshape(-1, 3)
    return ks, P[ks]


def path_length(res: StudyResult, probe: str) -> float:
    """Path length (mm) of a probe along the study's motion: chords within each run of closed
    frames (never bridging open frames); a pingpong study adds the wrap chord (last frame →
    frame 0), so its value is the full there-and-back cycle."""
    P = np.asarray(res.probes[probe], dtype=float).reshape(-1, 3)
    runs = closed_runs(res)
    total = sum(float(np.linalg.norm(np.diff(P[run], axis=0), axis=1).sum()) for run in runs)
    n = len(res.poses)
    if getattr(res.study, "loop", "once") == "pingpong" and n > 1 and res.poses[0].ok and res.poses[-1].ok:
        total += float(np.linalg.norm(P[-1] - P[0]))
    return total


def rotation_deg(R: np.ndarray) -> float:
    """Rotation angle (deg, 0…180) of a rotation matrix; accurate for small and large angles."""
    R = np.asarray(R, dtype=float)[:3, :3]
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(0.5 * float(np.linalg.norm(v)), 0.5 * (float(np.trace(R)) - 1.0)))


# ================================================================================================
# metric values
# ================================================================================================


def _ok_frames(res: StudyResult) -> list[int]:
    ks = [k for k, p in enumerate(res.poses) if p.ok]
    if not ks:
        raise _Undefined(f"no closed frame in study '{res.study.name}'")
    return ks


def _study_entry(report: dict, name: str) -> dict:
    return next(s for s in report["studies"] if s["name"] == name)


def _clearance(report: dict, res: StudyResult) -> float:
    """Min clearance of the checked (non-joined, non-allowed, non-meshing) pairs over the study's
    closed frames — the sweep measures it exactly, however far apart the parts are."""
    mc = _study_entry(report, res.study.name).get("min_clearance")
    if mc is None or mc.get("value") is None:
        raise _Undefined("no unjoined part pair was measured (none exist, or no frame closed)")
    return float(mc["value"])


def _load(report: dict, res: StudyResult, joint: str, key: str) -> float:
    entry = (_study_entry(report, res.study.name).get("loads") or {}).get(joint)
    if entry is None:
        raise _Undefined(f"study '{res.study.name}' computes no load for '{joint}'")
    mx = entry.get("max_abs")
    if mx is None:
        raise _Undefined(f"load of '{joint}' undefined ({entry.get('why') or 'no closed frame'})")
    if key == "load":
        return float(mx)
    if entry.get("capacity") is None:
        raise _Undefined(f"'{joint}' has no actuator capacity")
    return math.inf if mx == 0 else float(entry["capacity"]) / float(mx)


def _max_rotation(res: StudyResult, a: str, b: str | None) -> float:
    """Largest rotation (deg) of part a from home, or of b relative to a, over the closed frames."""
    angles, eye = [], np.eye(4)
    for k in _ok_frames(res):
        T = res.poses[k].transforms  # a part without a transform is at home
        R = np.asarray(T.get(a, eye), dtype=float)[:3, :3]
        if b is not None:
            R = R.T @ np.asarray(T.get(b, eye), dtype=float)[:3, :3]
        angles.append(rotation_deg(R))
    return max(angles)


def _study_value(asm: Assembly, report: dict, res: StudyResult, metric: str) -> float:
    """Value of a per-study string metric (§2.3 vocabulary; validate() checked the grammar)."""
    kind, _, arg = metric.partition(":")
    kind, arg = kind.strip(), arg.strip()
    if kind in ("span", "min", "max"):
        _ok_frames(res)
        _, v = joint_series(asm, res, arg)
        return float({"span": np.ptp, "min": np.min, "max": np.max}[kind](v))
    if kind in ("min_dist", "max_dist"):
        _ok_frames(res)
        a, b = (x.strip() for x in arg.split(","))
        d = np.linalg.norm(probe_series(res, a)[1] - probe_series(res, b)[1], axis=1)
        return float(d.min() if kind == "min_dist" else d.max())
    if kind == "path":
        _ok_frames(res)
        return path_length(res, arg)
    if kind == "delta":
        _ok_frames(res)
        name, _, ax = arg.rpartition(".")
        return float(np.ptp(probe_series(res, name.strip())[1][:, _AXES[ax.strip()]]))
    if kind == "rot":
        return _max_rotation(res, arg, None)
    if kind == "angle":
        a, b = (x.strip() for x in arg.split(","))
        return _max_rotation(res, a, b)
    if kind == "clearance":
        return _clearance(report, res)
    if kind in ("load", "sf"):
        return _load(report, res, arg, kind)
    raise _Undefined(f"unknown metric '{metric}'")


# ================================================================================================
# bounds and rows
# ================================================================================================


def _slack(bound: float) -> float:
    return BOUND_RTOL * max(1.0, abs(bound))


def _violation(v: float, t: Target) -> float:
    """How far v lies outside [min, max] (0 inside)."""
    lo = t.min - v if t.min is not None else -math.inf
    hi = v - t.max if t.max is not None else -math.inf
    return max(lo, hi, 0.0)


def _margin(v: float, t: Target) -> float:
    """Signed distance to the nearest bound (inf without bounds): negative outside [min, max]."""
    return min(v - t.min if t.min is not None else math.inf, t.max - v if t.max is not None else math.inf)


def _worst(values: list[tuple[str, float]], t: Target) -> tuple[str, float]:
    """The (study, value) that counts: largest violation, else smallest margin (first on ties)."""
    return max(values, key=lambda sv: (_violation(sv[1], t), -_margin(sv[1], t)))


def _met(v: float, t: Target) -> bool:
    """Inside [min, max] up to float noise (a value that lands on a bound by construction meets it)."""
    return ((t.min is None or v >= t.min - _slack(t.min))
            and (t.max is None or v <= t.max + _slack(t.max)))


def _row(t: Target, value: float | None, met: bool | None, worst_study: str | None,
         error: str | None) -> dict[str, Any]:
    if value is not None and not math.isfinite(value) and error is None:
        error = "unbounded (∞)"  # e.g. SF at zero load; JSON can only say null
    margin = None if value is None else _margin(value, t)
    return {"label": t.label, "metric": metric_name(t.metric), "value": fnum(value), "min": fnum(t.min),
            "max": fnum(t.max), "margin": fnum(margin), "met": met, "severity": t.severity, "study": t.study,
            "worst_study": worst_study, "error": error}


# ================================================================================================
# evaluation
# ================================================================================================


def callable_view(asm: Assembly, report: dict, results: dict[str, StudyResult]) -> dict:
    """The dict a callable metric receives: the report (targets not yet filled in) with a
    per-frame ``series`` on every study entry — ``ok`` [bool], ``joints`` {j: [value]},
    ``probes`` {p: [[x, y, z]]} and ``transforms`` {part: [4×4 row-major nested lists]} (home
    world → current world), one element per frame."""
    view = dict(report)
    eye = np.eye(4)
    studies = []
    for entry in report.get("studies") or []:
        res = results.get(entry["name"])
        entry = dict(entry)
        if res is not None:
            entry["series"] = {
                "ok": [bool(p.ok) for p in res.poses],
                "joints": {j: [float(p.q.get(j, 0.0)) for p in res.poses] for j in asm.joints},
                "probes": {name: np.asarray(pts, dtype=float).tolist() for name, pts in res.probes.items()},
                "transforms": {part: [np.asarray(p.transforms.get(part, eye), dtype=float).tolist()
                                      for p in res.poses] for part in asm.parts},  # missing = at home
            }
        studies.append(entry)
    view["studies"] = studies
    return view


def _evaluate(asm: Assembly, report: dict, results: dict[str, StudyResult], t: Target,
              view: dict | None) -> dict[str, Any]:
    skipped = list((report.get("partial") or {}).get("skipped") or [])
    not_run = f"studies not run: {', '.join(skipped)}"
    if callable(t.metric):
        try:
            v = float(t.metric(view))
        except Exception as exc:  # user code: report the failure on the target, never crash the run
            why = f"{type(exc).__name__}: {exc}"
            return _row(t, None, None, None, f"not evaluated ({not_run}): {why}") if skipped else \
                _row(t, None, False, None, why)
        if math.isnan(v):
            return _row(t, None, None if skipped else False, None, "callable returned NaN")
        return _row(t, v, _met(v, t), None, None)
    kind = t.metric.partition(":")[0].strip()
    if kind == "mass_g":
        v = float((report.get("mass") or {}).get("total_g") or 0.0)
        return _row(t, v, _met(v, t), None, None)
    if t.study is not None and t.study not in results:
        return _row(t, None, None, None, f"study '{t.study}' was not run")
    names = [t.study] if t.study is not None else list(results)
    values, errors = [], []
    for name in names:
        try:
            values.append((name, _study_value(asm, report, results[name], t.metric)))
        except _Undefined as exc:
            errors.append(str(exc))
    partial = t.study is None and bool(skipped)  # the value also depends on studies that did not run
    if not values:
        why = "; ".join(errors) or "no study was run"
        return _row(t, None, None, None, f"not evaluated ({not_run}): {why}") if partial else \
            _row(t, None, False, None, why)
    if kind in _EXTENT_METRICS:
        moving = [(name, v) for name, v in values if abs(v) > _STILL]
        if moving:
            values = moving
        elif partial:
            return _row(t, None, None, None, f"not evaluated: nothing moves it in the studies run ({not_run})")
    study, v = _worst(values, t)
    return _row(t, v, _met(v, t), study, None)


def evaluate_targets(asm: Assembly, report: dict, results: dict[str, StudyResult]) -> list[dict]:
    """One result row per ``asm.targets`` entry (declaration order); see the module docstring.

    ``value`` is None (JSON null) when undefined or infinite (e.g. an SF at zero load, which
    meets any ``min``).
    """
    view = callable_view(asm, report, results) if any(callable(t.metric) for t in asm.targets) else None
    return [_evaluate(asm, report, results, t, view) for t in asm.targets]
