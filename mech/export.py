"""Scene export for the viewer (spec §4.10): ``output/<slug>.mech/``.

Layout::

    <slug>.mech/
      parts/<part-slug>.stl   binary STL per part, world mm, Z up, at the home pose
      scene.json              parts, joints, pins, probes, per-study frames (+ the report)
      report.json             the report alone (read back as ``prev`` by the next run)
      series.json             full-precision per-frame series (t, ok, joints, probes, loads, residual)
      assembly.step           optional, all parts at home
      report.partial.json     the last partial (``--study``/``--frames``) run's report

scene.json/report.json/series.json always describe the last *full* run (every study at its
declared frame count): a partial run writes only ``report.partial.json``, and an INVALID full
run writes its report.json and removes the scene, series, STLs and STEP of the model it no
longer matches. ``virtual`` parts (the internal bodies of a spherical joint) are not exported:
scene.json lists each ``ball()`` under ``balls`` (name, parent, child, home center, its three
revolutes), so the viewer draws the ball instead of axes whose knuckles have no transforms.

File names come from ``geom.slug`` (ASCII, Windows-reserved-safe, de-duplicated) and the viewer
only uses ``parts[].mesh``, never ids, to find them. JSON files are written atomically with
``allow_nan=False``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from build123d import Compound, export_step, export_stl

from .geom import fnum, slug, to_json16, vec3
from .report import jsonable, real_parts

if TYPE_CHECKING:
    from .assembly import Assembly
    from .clearance import SweepResult
    from .kinematics import Kinematics
    from .massprops import MassProps
    from .motion import StudyResult

__all__ = ["export_scene", "mech_dir", "read_report", "write_report", "write_json", "clear_scene", "series_data",
           "ExportError", "REPORT", "PARTIAL_REPORT", "SERIES"]

ANGULAR_TOLERANCE = 0.2  # rad, STL tessellation
_IDENTITY_TOL = 1e-9  # a transform this close to I in every frame is omitted from scene.json
_SCENE_ISSUE_STATUSES = ("interference", "tight")  # what the viewer draws
REPORT, PARTIAL_REPORT, SERIES = "report.json", "report.partial.json", "series.json"


def mech_dir(out_root: Path | str, name: str) -> Path:
    """``out_root/<slug(name)>.mech`` — the export directory of an assembly."""
    return Path(out_root) / f"{slug(name)}.mech"


def write_json(path: Path, obj: Any, *, indent: int | None = None) -> None:
    """Write JSON-native ``obj`` atomically (temp file + replace), UTF-8, no NaN/Infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    separators = None if indent is not None else (",", ":")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, allow_nan=False, ensure_ascii=False, indent=indent, separators=separators)
        f.write("\n")
    os.replace(tmp, path)


def read_report(out_root: Path | str, name: str, *, partial: bool = False) -> dict | None:
    """The previous ``report.json`` of ``name`` (``report.partial.json`` with ``partial``); None if
    missing or unreadable."""
    path = mech_dir(out_root, name) / (PARTIAL_REPORT if partial else REPORT)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_report(report: dict, out_root: Path | str, *, partial: bool = False) -> Path:
    """Write the report alone: ``report.partial.json`` for a partial run, else ``report.json``
    (an INVALID full run, which has no scene: see ``clear_scene``)."""
    path = mech_dir(out_root, report["name"]) / (PARTIAL_REPORT if partial else REPORT)
    write_json(path, jsonable(report), indent=1)
    return path


def clear_scene(out_root: Path | str, name: str) -> None:
    """Remove the scene, part STLs and STEP of ``name`` (after an INVALID full run they would show
    a model that no longer exists) and the now-stale ``report.partial.json``."""
    out = mech_dir(out_root, name)
    for f in ("scene.json", SERIES, "assembly.step", PARTIAL_REPORT):
        (out / f).unlink(missing_ok=True)
    for stl in (out / "parts").glob("*.stl"):
        stl.unlink()


class ExportError(Exception):
    """A part could not be exported (its STL was not written); the message names the part."""


def _write_stl(shape, path: Path, tolerance: float, part: str | None = None) -> None:
    """Binary STL; verifies the file really is 84 + 50·n bytes (export_stl can report success on
    a path Windows silently mangled). Any failure raises ``ExportError`` and leaves no
    ``*.tmp.stl`` behind."""
    tmp = path.with_name(path.stem + ".tmp.stl")
    what = f"part '{part}'" if part is not None else path.name
    try:
        try:
            ok = export_stl(shape, str(tmp), tolerance=tolerance, angular_tolerance=ANGULAR_TOLERANCE,
                            ascii_format=False)
        except Exception as exc:  # OCC tessellation / writer failure
            raise ExportError(f"could not write the STL of {what} ({type(exc).__name__}: {exc})") from None
        if not ok:
            raise ExportError(f"could not write the STL of {what} (export_stl failed)")
        try:
            size = tmp.stat().st_size
            with tmp.open("rb") as f:
                f.seek(80)
                head = f.read(4)
        except OSError as exc:
            raise ExportError(f"the STL of {what} was not written ({exc})") from None
        n = struct.unpack("<I", head)[0] if len(head) == 4 else -1
        if n < 0 or size != 84 + 50 * n:
            raise ExportError(f"the STL of {what} is broken: {size} bytes is not a binary STL of {n} triangles")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _bbox(shape) -> list[list[float]]:
    bb = shape.bounding_box()
    return [[fnum(x) for x in vec3(bb.min)], [fnum(x) for x in vec3(bb.max)]]


def _is_identity(T: np.ndarray) -> bool:
    return bool(np.max(np.abs(np.asarray(T, dtype=float) - np.eye(4))) <= _IDENTITY_TOL)


def _pt(p) -> list[float | None]:
    return [fnum(x) for x in np.asarray(p, dtype=float).reshape(-1)]


def _scene_issue(r, frame: int | None) -> dict:
    return {"frame": frame, "a": r.a, "b": r.b, "status": r.status, "distance": fnum(r.distance),
            "volume": fnum(r.volume), "pa": _pt(r.pa), "pb": _pt(r.pb)}


def _full(x) -> float | None:
    """A float at full precision for series.json (None when not finite: JSON has no NaN)."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def series_data(asm: Assembly, results: dict[str, StudyResult], loads: dict[str, dict[str, dict]]) -> dict:
    """series.json: every study's per-frame series at full precision — ``t``, ``ok``, joint values
    (every non-fixed joint), probe positions, the raw load series per joint (with its unit) and
    the loop residual. What the summary rounds to 3 and report.json to 6 significant figures."""
    joints = [n for n, j in asm.joints.items() if j.kind != "fixed"]
    studies = []
    for name, res in results.items():
        poses = res.poses
        studies.append({
            "name": name, "frames": len(poses), "loop": getattr(res.study, "loop", "once"),
            "t": [_full(t) for t in res.t],
            "ok": [bool(p.ok) for p in poses],
            "joints": {j: [_full(p.q.get(j)) for p in poses] for j in joints},
            "probes": {pname: [[_full(x) for x in np.asarray(pt, dtype=float).reshape(-1)] for pt in pts]
                       for pname, pts in res.probes.items()},
            "loads": {j: {"unit": ld.get("unit"), "reflected_from": ld.get("reflected_from"),
                          "series": [_full(v) for v in ld.get("series", [])]}
                      for j, ld in (loads.get(name) or {}).items()},
            "residual": [_full(p.residual) for p in poses],
        })
    return {"version": 1, "name": asm.name, "units": "mm", "angles": "deg", "studies": studies}


def _study_scene(asm: Assembly, res: StudyResult, sweep: SweepResult | None, loads: dict[str, dict],
                 home_issues: list[dict]) -> dict:
    poses = res.poses
    joints = [n for n, j in asm.joints.items() if j.kind != "fixed"]
    transforms = {}
    for part in real_parts(asm):
        Ts = [p.transforms[part] for p in poses]
        if not all(_is_identity(T) for T in Ts):
            transforms[part] = [to_json16(T) for T in Ts]
    issues = list(home_issues)
    if sweep is not None:
        for k, frame in enumerate(sweep.per_frame):
            issues += [_scene_issue(r, k) for r in frame if r.status in _SCENE_ISSUE_STATUSES]
    return {
        "name": res.study.name,
        "frames": len(poses),
        "duration": fnum(res.study.duration),
        "loop": res.study.loop,
        "t": [fnum(t) for t in res.t],
        "joints": {j: [fnum(p.q.get(j)) for p in poses] for j in joints},
        "transforms": transforms,
        "probes": {name: [_pt(p) for p in pts] for name, pts in res.probes.items()},
        "issues": issues,
        "loads": {j: [fnum(v) for v in ld.get("series", [])] for j, ld in loads.items()},
        "residual": [fnum(p.residual) for p in poses],
    }


def export_scene(asm: Assembly, kin: Kinematics, props: dict[str, MassProps], results: dict[str, StudyResult],
                 sweeps: dict[str, SweepResult], loads: dict[str, dict[str, dict]], report: dict, out_root: Path, *,
                 roles: dict[str, str], tolerance: float = 0.05, step: bool = False) -> Path:
    """Write the viewer scene for an analyzed assembly; returns ``out_root/<slug>.mech/``.

    STLs of parts that no longer exist are removed from ``parts/`` so the folder mirrors the
    model. ``tolerance`` is the linear STL deflection (mm).
    """
    out = mech_dir(out_root, asm.name)
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    real = real_parts(asm)  # virtual joint bodies are not exported

    taken: set[str] = set()
    meshes = {name: f"parts/{slug(name, taken)}.stl" for name in real}
    for name in real:
        _write_stl(asm.parts[name].shape, out / meshes[name], tolerance, name)
    wanted = {Path(m).name for m in meshes.values()}
    for stale in parts_dir.glob("*.stl"):  # incl. *.tmp.stl left by an interrupted run
        if stale.name not in wanted:
            stale.unlink()

    if step:
        shapes = []
        for name in real:
            part = asm.parts[name]
            s = copy.copy(part.shape)  # shallow: shares the geometry, owns the label
            s.label = name
            shapes.append(s)
        assembly = Compound(children=shapes)
        assembly.label = asm.name
        export_step(assembly, str(out / "assembly.step"))
    else:  # a STEP from an earlier --step run no longer matches this geometry
        (out / "assembly.step").unlink(missing_ok=True)

    # interference found at the home pose (static_interference) is drawn with frame null
    home_issues = [
        {"frame": None, "a": i["parts"][0], "b": i["parts"][1], "status": "interference", "distance": 0.0,
         "volume": i["value"], "pa": i["location"], "pb": i["location"]}
        for i in report["issues"] if i["code"] == "static_interference" and len(i["parts"]) == 2
    ]
    scene = {
        "version": 1,
        "name": asm.name,
        "units": "mm",
        "up": [0, 0, 1],
        "clearance": fnum(asm.clearance),
        "parts": [
            {"id": name, "color": part.color, "opacity": fnum(part.opacity), "mesh": meshes[name],
             "ground": part.ground, "mass_g": fnum(props[name].mass_kg * 1000.0) if name in props else None,
             "material": part.material.name, "bom": part.bom, "bbox": _bbox(part.shape)}
            for name, part in asm.parts.items() if name in meshes
        ],
        "joints": [
            {"name": j.name, "kind": j.kind, "parent": j.parent, "child": j.child, "origin": _pt(j.origin),
             "axis": _pt(j.axis), "limits": None if j.limits is None else [fnum(x) for x in j.limits],
             "home": fnum(j.home), "role": roles.get(j.name)}
            for j in asm.joints.values()
        ],
        "pins": [{"name": p.name, "a": p.a, "b": p.b, "point": _pt(p.point),
                  "axis": None if p.axis is None else _pt(p.axis)} for p in asm.pins],
        # ball() joints: their three revolutes chain through virtual knuckles that have no mesh or
        # transforms here, so the viewer draws the ball at `center` (home world, moves with `parent`)
        "balls": [{"name": n, "parent": p, "child": c, "center": _pt(center),
                   "joints": [f"{n}_{i}" for i in (1, 2, 3)]} for n, (p, c, center) in asm.balls.items()],
        "probes": [{"name": p.name, "part": p.part} for p in asm.probes],
        "studies": [_study_scene(asm, res, sweeps.get(name), loads.get(name, {}), home_issues)
                    for name, res in results.items()],
        "report": report,
    }
    write_json(out / "scene.json", jsonable(scene))
    write_json(out / REPORT, jsonable(report), indent=1)
    write_json(out / SERIES, series_data(asm, results, loads))  # full precision: not through jsonable/fnum
    (out / PARTIAL_REPORT).unlink(missing_ok=True)  # older than this full run: Δprev uses report.json
    return out
