"""Write a hand-built mech viewer fixture: output/demo_fixture.mech/.

Independent of the `mech` package (which may not exist yet): geometry comes straight from
build123d and every number in the fixture is computed, not invented, so the viewer can be
developed and screenshot-tested against a scene that obeys the MECH_SPEC §4.10 schema.

The mechanism is a butt hinge whose flap swings about +X over an adjustable end stop:

* ``base``  – ground leaf with two outer knuckles.
* ``flap``  – moving leaf, revolute ``j_flap`` about the hinge axis (driven 0 → 138°).
* ``stop``  – end-stop block on a prismatic ``j_stop`` (held at home). It is a separate body
  so the flap/stop pair is *not* joined, which is what lets it be ``tight`` (frame 27) and
  ``interference`` (frames 28–29) under the spec's status rules.

Run from the repo root:  ``uv run python previewer/fixtures/make_fixture.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from build123d import Box, Compound, Cylinder, Location, Pos, Rot, Shape, Vector, export_stl

NAME = "demo_fixture"
CLEARANCE = 1.0  # mm; generous so the approach to the stop shows up as `tight`
FRAMES = 30
DURATION = 3.0
SWEEP = (0.0, 138.0)  # deg; beyond ~139° the flap would start to bite the base leaf
PLA_DENSITY = 1.24  # g/cm³
G = 9.80665

AXIS_ORIGIN = np.array([0.0, 0.0, 1.5])  # hinge axis runs along +X through the leaves' mid-plane
AXIS_DIR = np.array([1.0, 0.0, 0.0])
STOP_AXIS = np.array([0.0, 1.0, 0.0])

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / f"{NAME}.mech"


# ---------------------------------------------------------------- numbers / transforms

def fnum(x: float) -> float | None:
    """6 significant figures, None for non-finite (spec §1 JSON rules)."""
    x = float(x)
    return float(f"{x:.6g}") if math.isfinite(x) else None


def fvec(v) -> list[float | None]:
    return [fnum(c) for c in v]


def vnp(v: Vector) -> np.ndarray:
    return np.array([v.X, v.Y, v.Z])


def rot_about_line(origin: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """4×4 rotation by `deg` about the line (origin, axis), right-hand rule (Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    th = math.radians(deg)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    r = np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * (kx @ kx)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = origin - r @ origin
    return t


def to_json16(t: np.ndarray) -> list[float | None]:
    """Column-major flattening, i.e. three.js Matrix4.fromArray order."""
    return [fnum(x) for x in t.T.ravel()]


def to_location(t: np.ndarray) -> Location:
    """Rigid 4×4 → build123d Location (rotations here are about +X only)."""
    angle = math.degrees(math.atan2(t[2, 1], t[1, 1]))
    return Pos(*t[:3, 3]) * Rot(angle, 0, 0)


# ---------------------------------------------------------------- geometry (world mm, Z up)

def xcyl(radius: float, x0: float, x1: float) -> Shape:
    """Cylinder along X from x0 to x1 on the hinge axis."""
    cyl = Cylinder(radius, x1 - x0, rotation=(0, 90, 0))
    return Pos((x0 + x1) / 2, AXIS_ORIGIN[1], AXIS_ORIGIN[2]) * cyl


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Shape:
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def build_parts() -> dict[str, Shape]:
    r_knuckle = 3.5
    base = (
        box(-30, 30, -44, -4, 0, 3)
        + box(-30, -10.5, -4, 0, 0, 3)
        + box(10.5, 30, -4, 0, 0, 3)
        + xcyl(r_knuckle, -30, -10.5)
        + xcyl(r_knuckle, 10.5, 30)
    )
    flap = box(-30, 30, 4, 44, 0, 3) + box(-10, 10, 0, 4, 0, 3) + xcyl(r_knuckle, -10, 10)
    stop = box(-20, 20, -24, -20, 3, 23)
    return {"base": base, "flap": flap, "stop": stop}


PART_META = {
    "base": {"color": "#6f7d8f", "ground": True, "bom": None},
    "flap": {"color": "#3d8bd9", "ground": False, "bom": None},
    "stop": {"color": "#46b58c", "ground": False, "bom": "adjustable end stop"},
}


# ---------------------------------------------------------------- analysis

def pair_state(a: Shape, b: Shape, ta: np.ndarray, tb: np.ndarray, joined: bool) -> dict | None:
    """Clearance status of one pair at one pose, or None when ok (spec §4.6 rules)."""
    sa, sb = to_location(ta) * a, to_location(tb) * b
    dist, pa, pb = sa.distance_to_with_closest_points(sb)
    volume, common = 0.0, None
    if dist <= 1e-6:
        solids = sa.intersect(sb) or []  # ShapeList (possibly empty) or None
        common = Compound(list(solids)) if solids else None
        volume = common.volume if common is not None else 0.0
        if volume > 1e-3:
            bb = common.bounding_box()
            pa = pb = bb.center()
        dist = 0.0
    if volume > 1e-3:
        status = "interference"
    elif dist == 0.0:
        status = "contact"
    elif dist < CLEARANCE and not joined:
        status = "tight"
    else:
        return None
    return {"status": status, "distance": dist, "volume": volume,
            "pa": vnp(pa), "pb": vnp(pb), "common": common}


def main() -> None:
    shapes = build_parts()
    (OUT / "parts").mkdir(parents=True, exist_ok=True)

    # mass properties (PLA)
    mass_g = {n: s.volume * PLA_DENSITY * 1e-3 for n, s in shapes.items()}
    flap_com = vnp(shapes["flap"].center())

    # motion: `once` sampling, u_k = k/(N-1)
    k = np.arange(FRAMES)
    u = k / (FRAMES - 1)
    t = DURATION * u
    q_flap = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * u
    t_flap = [rot_about_line(AXIS_ORIGIN, AXIS_DIR, q) for q in q_flap]

    tip_home = np.array([0.0, 44.0, 1.5])
    tip = np.array([(tf @ np.append(tip_home, 1.0))[:3] for tf in t_flap])

    # gravity holding torque on j_flap: dV/dθ with V = m·g·z_com (J), N·m per rad == N·m
    rel = flap_com - AXIS_ORIGIN
    th = np.radians(q_flap)
    dz = rel[1] * np.cos(th) - rel[2] * np.sin(th)  # d z_com / dθ, mm/rad
    load = mass_g["flap"] * 1e-3 * G * dz * 1e-3

    # clearance sweep over the three pairs
    eye = np.eye(4)
    pairs = [("base", "flap", True), ("base", "stop", True), ("flap", "stop", False)]
    scene_issues, worst = [], {}
    for f, tf in enumerate(t_flap):
        pose = {"base": eye, "flap": tf, "stop": eye}
        for a, b, joined in pairs:
            st = pair_state(shapes[a], shapes[b], pose[a], pose[b], joined)
            if st is None:
                continue
            scene_issues.append({"frame": f, "a": a, "b": b, "status": st["status"],
                                 "distance": fnum(st["distance"]), "volume": fnum(st["volume"]),
                                 "pa": fvec(st["pa"]), "pb": fvec(st["pb"])})
            if st["status"] in ("interference", "tight"):
                key = (a, b, st["status"])
                score = st["volume"] if st["status"] == "interference" else -st["distance"]
                if key not in worst or score > worst[key][0]:
                    worst[key] = (score, f, st, pose[a])

    issues = []
    for (a, b, status), (_, f, st, ta) in sorted(worst.items(), key=lambda kv: kv[0][2] != "interference"):
        mid = (st["pa"] + st["pb"]) / 2
        home_mid = (np.linalg.inv(ta) @ np.append(mid, 1.0))[:3]
        if status == "interference":
            # extent/location live in part a's HOME frame (spec §4.6): pull the overlap back first
            bb = (to_location(np.linalg.inv(ta)) * st["common"]).bounding_box()
            ext = vnp(bb.size)
            home_mid = vnp(bb.center())
            msg = (f"{a}/{b} {st['volume']:.3g} mm³ overlap {ext[0]:.2g}×{ext[1]:.2g}×{ext[2]:.2g} mm "
                   f"@ ({home_mid[0]:.3g}, {home_mid[1]:.3g}, {home_mid[2]:.3g}) · j_flap={q_flap[f]:.1f}°")
            issues.append({"severity": "FAIL", "code": "interference", "message": msg, "study": "fold",
                           "frame": f, "parts": [a, b], "value": fnum(st["volume"]),
                           "location": fvec(home_mid), "extent": fvec(ext)})
        else:
            d = st["pb"] - st["pa"]
            d = d / np.linalg.norm(d)
            msg = (f"{a}/{b} gap {st['distance']:.3g} mm < {CLEARANCE} @ ({home_mid[0]:.3g}, "
                   f"{home_mid[1]:.3g}, {home_mid[2]:.3g}) dir ({d[0]:.2g}, {d[1]:.2g}, {d[2]:.2g}) · "
                   f"j_flap={q_flap[f]:.1f}°")
            issues.append({"severity": "WARN", "code": "tight_clearance", "message": msg, "study": "fold",
                           "frame": f, "parts": [a, b], "value": fnum(st["distance"]),
                           "location": fvec(home_mid), "extent": None})
    issues.append({"severity": "INFO", "code": "held_at_home", "message": "j_stop held at home (0 mm)",
                   "study": None, "frame": None, "parts": ["stop"], "value": None,
                   "location": None, "extent": None})

    # STL export (binary, verified size)
    parts_json = []
    for name, shape in shapes.items():
        path = OUT / "parts" / f"{name}.stl"
        export_stl(shape, path, tolerance=0.05, angular_tolerance=0.2)
        raw = path.read_bytes()
        n_tri = int.from_bytes(raw[80:84], "little")
        if len(raw) != 84 + 50 * n_tri:
            raise RuntimeError(f"{path} is not a valid binary STL")
        bb = shape.bounding_box()
        parts_json.append({"id": name, "color": PART_META[name]["color"], "opacity": 1.0,
                           "mesh": f"parts/{name}.stl", "ground": PART_META[name]["ground"],
                           "mass_g": fnum(mass_g[name]), "material": "PLA", "bom": PART_META[name]["bom"],
                           "bbox": [fvec(vnp(bb.min)), fvec(vnp(bb.max))]})

    capacity = 0.01
    i_max = int(np.argmax(np.abs(load)))
    max_abs = float(abs(load[i_max]))
    interf_frames = [i["frame"] for i in scene_issues if i["status"] == "interference" and i["b"] == "stop"]
    total_g = sum(mass_g.values())
    com = sum(mass_g[n] * vnp(shapes[n].center()) for n in shapes) / total_g
    probe_path = float(np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1)))
    roles = {"j_flap": "driver", "j_stop": "free"}

    report = {
        "name": NAME, "status": "FAIL", "params": {}, "parts": len(shapes),
        "joints": {"driver": 1, "coupled": 0, "passive": 0, "free": 1, "fixed": 0},
        "roles": roles, "mobility": {"fold": 0},
        "mass": {"total_g": fnum(total_g), "com_mm": fvec(com), "parts": {n: fnum(m) for n, m in mass_g.items()}},
        "issues": issues,
        "targets": [{"label": "fold angle", "metric": "span:j_flap", "value": fnum(SWEEP[1] - SWEEP[0]),
                     "min": 120.0, "max": None, "met": True, "severity": "FAIL", "study": None},
                    {"label": "tip lift", "metric": "delta:tip.z", "value": fnum(np.ptp(tip[:, 2])),
                     "min": 50.0, "max": None, "met": False, "severity": "WARN", "study": "fold"}],
        "studies": [{
            "name": "fold", "frames": FRAMES,
            "joint_ranges": {"j_flap": fvec(SWEEP), "j_stop": [0.0, 0.0]},
            "probes": {"tip": {"min": fvec(tip.min(axis=0)), "max": fvec(tip.max(axis=0)),
                               "start": fvec(tip[0]), "end": fvec(tip[-1]), "path_mm": fnum(probe_path)}},
            "min_clearance": {"parts": ["flap", "stop"], "value": 0.0,
                              "frame": interf_frames[0] if interf_frames else None},
            "loads": {"j_flap": {"unit": "N·m", "max_abs": fnum(max_abs), "frame": i_max,
                                 "capacity": capacity, "sf": fnum(capacity / max_abs), "reflected_from": None}},
            "max_residual": 0.0,
        }],
        "viewer_url": f"http://localhost:3000/mech.html?m={NAME}&issue=0&ui=0",
    }

    scene = {
        "version": 1, "name": NAME, "units": "mm", "up": [0, 0, 1], "clearance": CLEARANCE,
        "parts": parts_json,
        "joints": [
            {"name": "j_flap", "kind": "revolute", "parent": "base", "child": "flap",
             "origin": fvec(AXIS_ORIGIN), "axis": fvec(AXIS_DIR), "limits": [0.0, 180.0], "home": 0.0,
             "role": roles["j_flap"]},
            {"name": "j_stop", "kind": "prismatic", "parent": "base", "child": "stop",
             "origin": [0.0, -22.0, 3.0], "axis": fvec(STOP_AXIS), "limits": [-10.0, 10.0], "home": 0.0,
             "role": roles["j_stop"]},
        ],
        "pins": [],
        "probes": [{"name": "tip", "part": "flap"}],
        "studies": [{
            "name": "fold", "frames": FRAMES, "duration": DURATION, "loop": "once",
            "t": fvec(t), "joints": {"j_flap": fvec(q_flap), "j_stop": [0.0] * FRAMES},
            "transforms": {"flap": [to_json16(tf) for tf in t_flap]},
            "probes": {"tip": [fvec(p) for p in tip]},
            "issues": scene_issues,
            "loads": {"j_flap": fvec(load)},
            "residual": [0.0] * FRAMES,
        }],
        "report": report,
    }

    for fname, data in (("scene.json", scene), ("report.json", report)):
        (OUT / fname).write_text(json.dumps(data, allow_nan=False, ensure_ascii=False, indent=1), encoding="utf-8")

    counts = {s: sum(1 for i in scene_issues if i["status"] == s) for s in ("interference", "tight", "contact")}
    print(f"wrote {OUT} · flap {mass_g['flap']:.3g} g · issues {counts} · load max {max_abs:.3g} N·m @f{i_max}")


if __name__ == "__main__":
    main()
