"""Scene export for the viewer (spec §4.10): ``output/<slug>.mech/``.

Layout::

    <slug>.mech/
      parts/<part-slug>.stl   binary STL per part, world mm, Z up, at the home pose
      scene.json              parts, joints, pins, probes, per-study frames (+ the report)
      report.json             the report alone (read back as ``prev`` by the next run)
      assembly.step           optional, all parts at home

File names come from ``geom.slug`` (ASCII, Windows-reserved-safe, de-duplicated) and the viewer
only uses ``parts[].mesh``, never ids, to find them. JSON files are written atomically with
``allow_nan=False``.
"""

from __future__ import annotations

import copy
import json
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from build123d import Compound, export_step, export_stl

from .geom import fnum, slug, to_json16, vec3
from .report import jsonable

if TYPE_CHECKING:
    from .assembly import Assembly
    from .clearance import SweepResult
    from .kinematics import Kinematics
    from .massprops import MassProps
    from .motion import StudyResult

__all__ = ["export_scene", "mech_dir", "read_report", "write_report", "write_json"]

ANGULAR_TOLERANCE = 0.2  # rad, STL tessellation
_IDENTITY_TOL = 1e-9  # a transform this close to I in every frame is omitted from scene.json
_SCENE_ISSUE_STATUSES = ("interference", "tight")  # what the viewer draws


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


def read_report(out_root: Path | str, name: str) -> dict | None:
    """The previous ``report.json`` of ``name`` (None if missing or unreadable)."""
    path = mech_dir(out_root, name) / "report.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_report(report: dict, out_root: Path | str) -> Path:
    """Write ``report.json`` only (used for INVALID runs, which have no scene)."""
    path = mech_dir(out_root, report["name"]) / "report.json"
    write_json(path, jsonable(report), indent=1)
    return path


def _write_stl(shape, path: Path, tolerance: float) -> None:
    """Binary STL; verifies the file really is 84 + 50·n bytes (export_stl can report success on
    a path Windows silently mangled)."""
    tmp = path.with_name(path.stem + ".tmp.stl")
    if not export_stl(shape, str(tmp), tolerance=tolerance, angular_tolerance=ANGULAR_TOLERANCE, ascii_format=False):
        raise RuntimeError(f"export_stl failed for {path.name}")
    try:
        size = tmp.stat().st_size
        with tmp.open("rb") as f:
            f.seek(80)
            head = f.read(4)
    except OSError as exc:
        raise RuntimeError(f"STL {path.name} was not written ({exc})") from exc
    n = struct.unpack("<I", head)[0] if len(head) == 4 else -1
    if n < 0 or size != 84 + 50 * n:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"STL {path.name}: {size} bytes is not a binary STL of {n} triangles")
    os.replace(tmp, path)


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


def _study_scene(asm: Assembly, res: StudyResult, sweep: SweepResult | None, loads: dict[str, dict],
                 home_issues: list[dict]) -> dict:
    poses = res.poses
    joints = [n for n, j in asm.joints.items() if j.kind != "fixed"]
    transforms = {}
    for part in asm.parts:
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

    taken: set[str] = set()
    meshes = {name: f"parts/{slug(name, taken)}.stl" for name in asm.parts}
    for name, part in asm.parts.items():
        _write_stl(part.shape, out / meshes[name], tolerance)
    wanted = {Path(m).name for m in meshes.values()}
    for stale in parts_dir.glob("*.stl"):
        if stale.name not in wanted:
            stale.unlink()

    if step:
        shapes = []
        for name, part in asm.parts.items():
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
            for name, part in asm.parts.items()
        ],
        "joints": [
            {"name": j.name, "kind": j.kind, "parent": j.parent, "child": j.child, "origin": _pt(j.origin),
             "axis": _pt(j.axis), "limits": None if j.limits is None else [fnum(x) for x in j.limits],
             "home": fnum(j.home), "role": roles.get(j.name)}
            for j in asm.joints.values()
        ],
        "pins": [{"name": p.name, "a": p.a, "b": p.b, "point": _pt(p.point),
                  "axis": None if p.axis is None else _pt(p.axis)} for p in asm.pins],
        "probes": [{"name": p.name, "part": p.part} for p in asm.probes],
        "studies": [_study_scene(asm, res, sweeps.get(name), loads.get(name, {}), home_issues)
                    for name, res in results.items()],
        "report": report,
    }
    write_json(out / "scene.json", jsonable(scene))
    write_json(out / "report.json", jsonable(report), indent=1)
    return out
