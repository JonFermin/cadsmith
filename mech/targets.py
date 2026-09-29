"""Design-target evaluation (spec §2.3, §4.8).

A target states design intent as a metric that must land in [min, max]. String metrics are
evaluated per study on the study's *closed* frames (``pose.ok``) — a pose the solver could not
close is not a configuration of the mechanism, and its loop_open issue is reported separately —
and with ``study=None`` the worst value over all run studies counts. For the motion-extent
metrics (``span``, ``path``, ``delta``) a study in which the joint/probe does not move at all
says nothing about how far it moves, so such studies are skipped while another study moves it
(e.g. ``span:j_out`` is not 0 just because a second study swings an unrelated arm). ``mass_g``
and callables ``f(report_dict) -> float`` are study-independent.

Result rows: ``{"label", "metric", "value", "min", "max", "met", "severity", "study",
"worst_study", "error"}`` — ``study`` is the target's declared study, ``worst_study`` where the
counted value came from, ``met`` is None when the target's study was not run (``--study``
filter), and ``error`` says why ``value`` is undefined.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .geom import fnum

if TYPE_CHECKING:
    from .assembly import Assembly, Target
    from .motion import StudyResult

__all__ = ["evaluate_targets", "metric_name"]

_AXES = {"x": 0, "y": 1, "z": 2}
_EXTENT_METRICS = ("span", "path", "delta")  # 0 in a study that doesn't move the joint/probe


class _Undefined(Exception):
    """The metric has no value for this study (message says why)."""


def metric_name(metric) -> str:
    """Display form of a metric: the string itself, or ``name()`` for a callable."""
    return metric if isinstance(metric, str) else f"{getattr(metric, '__name__', 'callable')}()"


def _ok_frames(res: StudyResult) -> list[int]:
    ks = [k for k, p in enumerate(res.poses) if p.ok]
    if not ks:
        raise _Undefined(f"no closed frame in study '{res.study.name}'")
    return ks


def _joint_values(asm: Assembly, res: StudyResult, joint: str) -> np.ndarray:
    ks = _ok_frames(res)
    if asm.joints[joint].kind == "fixed":
        return np.zeros(len(ks))  # a fixed joint has no coordinate; it never moves
    return np.array([res.poses[k].q[joint] for k in ks], dtype=float)


def _probe_points(res: StudyResult, probe: str) -> np.ndarray:
    return np.asarray(res.probes[probe], dtype=float)[_ok_frames(res)]


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
        raise _Undefined(f"load of '{joint}' undefined (no closed frame)")
    if key == "load":
        return float(mx)
    if entry.get("capacity") is None:
        raise _Undefined(f"'{joint}' has no actuator capacity")
    return math.inf if mx == 0 else float(entry["capacity"]) / float(mx)


def _study_value(asm: Assembly, report: dict, res: StudyResult, metric: str) -> float:
    """Value of a per-study string metric (§2.3 vocabulary; validate() checked the grammar)."""
    kind, _, arg = metric.partition(":")
    kind, arg = kind.strip(), arg.strip()
    if kind in ("span", "min", "max"):
        v = _joint_values(asm, res, arg)
        return float({"span": np.ptp, "min": np.min, "max": np.max}[kind](v))
    if kind in ("min_dist", "max_dist"):
        a, b = (x.strip() for x in arg.split(","))
        d = np.linalg.norm(_probe_points(res, a) - _probe_points(res, b), axis=1)
        return float(d.min() if kind == "min_dist" else d.max())
    if kind == "path":
        P = _probe_points(res, arg)
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if kind == "delta":
        name, _, ax = arg.rpartition(".")
        return float(np.ptp(_probe_points(res, name)[:, _AXES[ax]]))
    if kind == "clearance":
        return _clearance(report, res)
    if kind in ("load", "sf"):
        return _load(report, res, arg, kind)
    raise _Undefined(f"unknown metric '{metric}'")


def _violation(v: float, t: Target) -> float:
    """How far v lies outside [min, max] (0 inside)."""
    lo = t.min - v if t.min is not None else -math.inf
    hi = v - t.max if t.max is not None else -math.inf
    return max(lo, hi, 0.0)


def _margin(v: float, t: Target) -> float:
    """Distance to the nearest bound (inf without bounds); smaller = closer to failing."""
    return min(v - t.min if t.min is not None else math.inf, t.max - v if t.max is not None else math.inf)


def _worst(values: list[tuple[str, float]], t: Target) -> tuple[str, float]:
    """The (study, value) that counts: largest violation, else smallest margin (first on ties)."""
    return max(values, key=lambda sv: (_violation(sv[1], t), -_margin(sv[1], t)))


def _row(t: Target, value: float | None, met: bool | None, worst_study: str | None,
         error: str | None) -> dict[str, Any]:
    if value is not None and not math.isfinite(value) and error is None:
        error = "unbounded (∞)"  # e.g. SF at zero load; JSON can only say null
    return {"label": t.label, "metric": metric_name(t.metric), "value": fnum(value), "min": fnum(t.min),
            "max": fnum(t.max), "met": met, "severity": t.severity, "study": t.study,
            "worst_study": worst_study, "error": error}


def _met(v: float, t: Target) -> bool:
    return _violation(v, t) == 0.0


def _evaluate(asm: Assembly, report: dict, results: dict[str, StudyResult], t: Target) -> dict[str, Any]:
    if callable(t.metric):
        try:
            v = float(t.metric(report))
        except Exception as exc:  # user code: report the failure on the target, never crash the run
            return _row(t, None, False, None, f"{type(exc).__name__}: {exc}")
        if math.isnan(v):
            return _row(t, None, False, None, "callable returned NaN")
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
    if not values:
        return _row(t, None, False, None, "; ".join(errors) or "no study was run")
    if kind in _EXTENT_METRICS and any(v != 0.0 for _, v in values):
        values = [(name, v) for name, v in values if v != 0.0]
    study, v = _worst(values, t)
    return _row(t, v, _met(v, t), study, None)


def evaluate_targets(asm: Assembly, report: dict, results: dict[str, StudyResult]) -> list[dict]:
    """One result row per ``asm.targets`` entry (declaration order); see the module docstring.

    ``value`` is None (JSON null) when undefined or infinite (e.g. an SF at zero load, which
    meets any ``min``).
    """
    return [_evaluate(asm, report, results, t) for t in asm.targets]
