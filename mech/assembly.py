"""Assembly model: parts, joints, couplings, loop-closing pins and design intent (spec §2, §4.1).

Everything is declared in *home-world* coordinates: parts are modeled in place, joints carry a
world origin + axis at the drawn pose, and ``home`` is each joint's value in that drawing.
Builder methods record problems instead of raising (except duplicate names), and
``validate()`` reports them all at once together with the structural checks.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from build123d import Axis, Compound, Edge, Location
from build123d.topology import Shape
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.Extrema import Extrema_ExtFlag_MIN

from .geom import from_location, inside, unit, vec3
from .materials import Material, get_material

__all__ = [
    "ModelError",
    "Part",
    "Joint",
    "Coupling",
    "Pin",
    "Probe",
    "Actuator",
    "Study",
    "Target",
    "Assembly",
]

_PALETTE = ("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f")
_GROUND_COLOR = "#8a8f98"
_SEVERITIES = ("FAIL", "WARN", "INFO")
_LOOP_MODES = ("once", "pingpong")
# Target metric vocabulary (§2.3): name -> what the argument after ':' refers to.
_METRICS = {
    "span": "joint", "min": "joint", "max": "joint", "load": "joint", "sf": "joint",
    "min_dist": "probe_pair", "max_dist": "probe_pair", "path": "probe", "delta": "probe_axis",
    "rot": "part", "angle": "part_pair", "clearance": None, "mass_g": None,
}
# Coupling kinds with a required (driver kind, driven kind).
_COUPLING_KINDS = {"gear": ("revolute", "revolute"), "belt": ("revolute", "revolute"),
                   "screw": ("revolute", "prismatic"), "rack": ("revolute", "prismatic")}
_PARALLEL_TOL = 1e-3  # rad: screw and belt axes must be parallel within this
_PLANAR_LOOSE, _PLANAR_TIGHT = 1e-3, 1e-9  # rad: near_planar warning band
_TOUCH = 1e-6  # mm: parts this close at home touch (a bearing on its shaft, a bushing on its rod)
_MIN_VOLUME = 1e-9  # mm³: a part with less solid volume than this is not a solid
ALLOW_DEPTH = 0.1  # mm: default max mean overlap depth 2V/A an allow_contact pair may reach
_TOOTHED = ("gear", "rack")  # LibPart kinds whose teeth mesh through gear/rack couplings


class ModelError(Exception):
    """The model is invalid; ``errors`` lists every problem ``Assembly.validate()`` found."""

    def __init__(self, errors: Iterable[str] | str):
        self.errors: list[str] = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("invalid model:\n" + "\n".join(f"  - {e}" for e in self.errors))


@dataclass
class Part:
    name: str
    shape: Shape
    material: Material
    color: str
    ground: bool
    opacity: float = 1.0
    mass_g: float | None = None
    bom: str | None = None
    kind: str | None = None  # LibPart kind: "gear" | "rack" | "pulley" | None


@dataclass
class Joint:
    name: str
    kind: Literal["revolute", "prismatic", "fixed"]
    parent: str
    child: str
    origin: np.ndarray  # home world, mm
    axis: np.ndarray  # home world, unit
    limits: tuple[float, float] | None = None  # deg | mm, same coordinate as home
    home: float = 0.0


@dataclass
class Coupling:
    name: str
    driver: str
    driven: str
    ratio: float  # native units: deg/deg, mm/deg, deg/mm or mm/mm
    offset: float = 0.0
    kind: str = "couple"  # gear | belt | screw | rack | couple


@dataclass
class Pin:
    name: str
    a: str
    b: str
    point: np.ndarray  # home world
    axis: np.ndarray | None = None  # hinge axis (unit) or None for a ball joint


@dataclass
class Probe:
    name: str
    part: str
    point: np.ndarray  # home world


@dataclass
class Actuator:
    joint: str
    capacity: float  # N·m (revolute) | N (prismatic)


@dataclass
class Study:
    name: str
    drive: dict
    frames: int = 60
    loop: str = "once"
    duration: float = 3.0


@dataclass
class Target:
    label: str
    metric: str | Callable
    min: float | None
    max: float | None
    study: str | None
    severity: str


def _suggest(name: str, options: Iterable[str]) -> str:
    """' (did you mean 'x'?)' for the closest option, else a short list of what exists."""
    opts = [str(o) for o in options]
    close = difflib.get_close_matches(str(name), opts, n=1, cutoff=0.6)
    if close:
        return f" (did you mean '{close[0]}'?)"
    if opts and len(opts) <= 8:
        return f" (known: {', '.join(opts)})"
    return ""


def _is_shape(obj: Any) -> bool:
    return isinstance(obj, Shape) and obj.wrapped is not None


def _angle_between_lines(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) between two unit directions treated as undirected lines."""
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), abs(float(a @ b)))


def _boxes_overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], pad: float) -> bool:
    """Do two (min, max) axis-aligned boxes, each inflated by ``pad``, overlap with positive volume?"""
    return bool(np.all(a[0] - pad < b[1] + pad) and np.all(b[0] - pad < a[1] + pad))


class Assembly:
    """A mechanism: parts modeled in place + joints/couplings/pins + studies and targets."""

    def __init__(self, name: str, *, clearance: float = 0.3, interference_tol: float = 0.5,
                 pin_tol: float = 5.0, gravity=(0, 0, -9.80665)):
        self.name = str(name)
        self.clearance = float(clearance)
        self.interference_tol = float(interference_tol)
        self.pin_tol = float(pin_tol)
        self.gravity = vec3(gravity)
        self.parts: dict[str, Part] = {}
        self.joints: dict[str, Joint] = {}
        self.couplings: list[Coupling] = []
        self.pins: list[Pin] = []
        self.probes: list[Probe] = []
        self.actuators: dict[str, Actuator] = {}
        self.studies: list[Study] = []
        self.targets: list[Target] = []
        self.allowed: set[frozenset] = set()
        self.allow_depth: dict[frozenset, float | None] = {}  # allowed pair -> max mean overlap depth
        self.ignored: set[frozenset] = set()
        self.checked: set[frozenset] = set()  # check_clearance pairs: joined, but clearance still applies
        self.params: dict = {}  # filled by the runner with the build() keyword values
        self._errors: list[str] = []  # problems found while building; reported by validate()
        # screw/rack ratios depend on joint geometry, so they are (re)computed in validate()
        self._ratio_rules: list[tuple[Coupling, Callable[[], float | None]]] = []
        self._bbox_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._dist_cache: dict[tuple, float] = {}
        self._touch_cache: dict[tuple, bool] = {}
        self._solid_cache: dict[int, tuple[object, int, float]] = {}  # id(shape) -> (shape, #solids, volume)
        self._joined_memo: tuple[tuple, set[frozenset]] | None = None

    def __repr__(self) -> str:
        return f"Assembly({self.name!r}: {len(self.parts)} parts, {len(self.joints)} joints, {len(self.pins)} pins)"

    # ---------------------------------------------------------------------------- builders

    def part(self, name: str, shape_or_libpart, *, material="PLA", color: str | None = None,
             ground: bool = False, mass_g: float | None = None, density: float | None = None,
             opacity: float = 1.0, bom: str | None = None) -> str:
        """Add a part modeled in place (home-world coordinates). Duplicate name -> ValueError."""
        name = str(name)
        if name in self.parts:
            raise ValueError(f"duplicate part name '{name}'")
        where = f"part '{name}'"
        shape = shape_or_libpart
        kind = None
        if not isinstance(shape, Shape) and hasattr(shape, "shape"):  # LibPart: take its defaults
            bom = getattr(shape, "bom", None) if bom is None else bom
            mass_g = getattr(shape, "mass_g", None) if mass_g is None else mass_g
            kind = getattr(shape, "kind", None)
            shape = shape.shape
        if isinstance(shape, (list, tuple)) and shape and all(isinstance(s, Shape) for s in shape):
            shape = Compound(list(shape))
        if not _is_shape(shape):
            self._errors.append(f"{where}: expected a build123d Shape/Part/Compound, got {type(shape).__name__}")
        try:
            mat = get_material(material, density)
        except (KeyError, ValueError) as exc:
            self._errors.append(f"{where}: {exc.args[0]}")
            mat = Material(str(material), float("nan"))
        if mass_g is not None and not (math.isfinite(float(mass_g)) and float(mass_g) > 0):
            self._errors.append(f"{where}: mass_g must be > 0 (got {mass_g})")
        if not 0.0 <= float(opacity) <= 1.0:
            self._errors.append(f"{where}: opacity must be in [0, 1] (got {opacity})")
        if color is None:
            n_moving = sum(1 for p in self.parts.values() if not p.ground)
            color = _GROUND_COLOR if ground else _PALETTE[n_moving % len(_PALETTE)]
        self.parts[name] = Part(name, shape, mat, str(color), bool(ground), float(opacity),
                                None if mass_g is None else float(mass_g), bom, kind)
        return name

    def revolute(self, name: str, parent: str, child: str, origin=None, axis=None, *, at=None,
                 limits=None, home: float = 0.0) -> str:
        """Hinge about the world line (origin, axis) at home; values in degrees."""
        return self._motion_joint("revolute", name, parent, child, origin, axis, at, limits, home)

    def prismatic(self, name: str, parent: str, child: str, origin=None, axis=None, *, at=None,
                  limits=None, home: float = 0.0) -> str:
        """Slide along ``axis``; values in mm."""
        return self._motion_joint("prismatic", name, parent, child, origin, axis, at, limits, home)

    def fix(self, child: str, parent: str, name: str | None = None) -> str:
        """Rigidly attach ``child`` to ``parent``."""
        name = str(name) if name is not None else f"fix_{child}"
        self._add_joint(Joint(name, "fixed", str(parent), str(child), np.zeros(3), np.array([0.0, 0.0, 1.0])))
        return name

    def gear(self, driver: str, driven: str, ratio: float, *, name: str | None = None) -> str:
        """Revolute→revolute; ``ratio`` assumes co-directional axes (external mesh: negative)."""
        return self._add_coupling("gear", driver, driven, float(ratio), 0.0, name)

    def belt(self, driver: str, driven: str, t_driver: float, t_driven: float, *, name: str | None = None) -> str:
        """Belt/chain drive between parallel revolute joints: both pulleys turn the same way, ratio
        = s·t_driver/t_driven (teeth or pitch diameters), s = sign(axis·axis). Never meshing."""
        cname = self._add_coupling("belt", driver, driven, None, 0.0, name)
        try:
            td, tn = float(t_driver), float(t_driven)
        except (TypeError, ValueError):
            td = tn = math.nan
        if not (math.isfinite(td) and math.isfinite(tn) and td > 0 and tn > 0):
            self._errors.append(f"belt '{cname}': t_driver and t_driven must be > 0 (got {t_driver!r}, {t_driven!r})")
            td = tn = 1.0
        self._add_ratio_rule(self.couplings[-1], lambda: self._belt_ratio(driver, driven, td / tn))
        return cname

    def screw(self, driver: str, driven: str, lead: float, *, hand: str = "right", name: str | None = None) -> str:
        """Revolute screw → prismatic nut: ratio = ∓s·lead/360 (right/left hand), s = sign(axis·axis)."""
        cname = self._add_coupling("screw", driver, driven, None, 0.0, name)
        where = f"screw '{cname}'"
        lead = float(lead)
        if not (math.isfinite(lead) and lead > 0):
            self._errors.append(f"{where}: lead must be > 0 mm/rev (got {lead})")
        if hand not in ("right", "left"):
            self._errors.append(f"{where}: hand must be 'right' or 'left' (got {hand!r})")
        self._add_ratio_rule(self.couplings[-1], lambda: self._screw_ratio(driver, driven, lead, hand))
        return cname

    def rack(self, driver: str, driven: str, pitch_radius: float, *, sign: int | None = None,
             name: str | None = None) -> str:
        """Pinion (revolute) → rack (prismatic): ratio = sign·π·r/180 mm/deg, sign from geometry by default."""
        cname = self._add_coupling("rack", driver, driven, None, 0.0, name)
        where = f"rack '{cname}'"
        r = float(pitch_radius)
        if not (math.isfinite(r) and r > 0):
            self._errors.append(f"{where}: pitch_radius must be > 0 (got {pitch_radius})")
        if sign is not None and sign not in (1, -1):
            self._errors.append(f"{where}: sign must be +1, -1 or None (got {sign!r})")
        self._add_ratio_rule(self.couplings[-1], lambda: self._rack_ratio(driver, driven, r, sign))
        return cname

    def couple(self, driver: str, driven: str, ratio: float = 1.0, offset: float = 0.0, *,
               name: str | None = None) -> str:
        """Generic linear coupling on deltas from home, native units."""
        return self._add_coupling("couple", driver, driven, float(ratio), float(offset), name)

    def pin(self, name: str, a: str, b: str, point, axis=None) -> str:
        """Close a loop: ``point`` (home world) stays coincident on parts a and b; ``axis`` = hinge."""
        name = str(name)
        if any(p.name == name for p in self.pins):
            raise ValueError(f"duplicate pin name '{name}'")
        pt = self._vec(point, f"pin '{name}' point")
        ax = None if axis is None else self._dir(axis, f"pin '{name}' axis")
        self.pins.append(Pin(name, str(a), str(b), pt, ax))
        return name

    def probe(self, name: str, part: str, point) -> str:
        """Track the trajectory of a point (home world) fixed to ``part``."""
        name = str(name)
        if any(p.name == name for p in self.probes):
            raise ValueError(f"duplicate probe name '{name}'")
        self.probes.append(Probe(name, str(part), self._vec(point, f"probe '{name}' point")))
        return name

    def actuator(self, joint: str, capacity: float) -> str:
        """Declare an actuator (N·m or N); also marks the preferred driver of a loop."""
        joint = str(joint)
        if joint in self.actuators:
            raise ValueError(f"duplicate actuator on joint '{joint}'")
        capacity = float(capacity)
        if not (math.isfinite(capacity) and capacity > 0):
            self._errors.append(f"actuator on '{joint}': capacity must be > 0 (got {capacity})")
        self.actuators[joint] = Actuator(joint, capacity)
        return joint

    def allow_contact(self, a: str, b: str, *, max_depth: float | None = ALLOW_DEPTH) -> None:
        """Parts a and b may touch and overlap slightly (a nominal fit, a belt on its pulley).

        Contact and overlaps whose mean depth 2V/A stays within ``max_depth`` mm are ``contact``;
        a deeper overlap still interferes. ``max_depth=None`` allows any overlap and skips the
        overlap boolean for the pair (fast; its depth is then never checked).
        """
        self._add_pair(self.allowed, "allow_contact", a, b)
        where = f"allow_contact('{a}', '{b}')"
        if max_depth is not None:
            try:
                max_depth = float(max_depth)
            except (TypeError, ValueError):
                self._errors.append(f"{where}: max_depth must be a number >= 0 mm or None (got {max_depth!r})")
                max_depth = ALLOW_DEPTH
            if not (math.isfinite(max_depth) and max_depth >= 0):
                self._errors.append(f"{where}: max_depth must be a number >= 0 mm or None (got {max_depth})")
        self.allow_depth[frozenset((str(a), str(b)))] = max_depth

    def ignore(self, a: str, b: str) -> None:
        """Skip the pair a, b in clearance checks entirely."""
        self._add_pair(self.ignored, "ignore", a, b)

    def check_clearance(self, a: str, b: str) -> None:
        """Hold a joined pair to ``clearance`` anyway (e.g. an arm swinging close to the frame it is
        hinged to, far from the hinge): the pair is checked like any unjoined pair."""
        self._add_pair(self.checked, "check_clearance", a, b)

    def study(self, name: str, drive: dict, *, frames: int = 60, loop: str = "once", duration: float = 3.0) -> str:
        """Motion study: ``drive`` maps joint -> (start, end) | [(u, v), ...] | callable(u)."""
        name = str(name)
        if any(s.name == name for s in self.studies):
            raise ValueError(f"duplicate study name '{name}'")
        self.studies.append(Study(name, drive, int(frames), str(loop), float(duration)))
        return name

    def target(self, label: str, metric: str | Callable, *, min: float | None = None, max: float | None = None,
               study: str | None = None, severity: str = "FAIL") -> str:
        """Design intent: ``metric`` must land in [min, max] (see §2.3 for the vocabulary)."""
        self.targets.append(Target(str(label), metric, None if min is None else float(min),
                                   None if max is None else float(max), study, str(severity)))
        return str(label)

    # ---------------------------------------------------------------------------- builder internals

    def _vec(self, v, where: str) -> np.ndarray:
        try:
            return vec3(v)
        except ValueError as exc:
            self._errors.append(f"{where}: {exc}")
            return np.zeros(3)

    def _dir(self, v, where: str) -> np.ndarray:
        try:
            return unit(v)
        except ValueError as exc:
            self._errors.append(f"{where}: {exc}")
            return np.array([0.0, 0.0, 1.0])

    def _add_joint(self, joint: Joint) -> None:
        if joint.name in self.joints:
            raise ValueError(f"duplicate joint name '{joint.name}'")
        self.joints[joint.name] = joint

    def _motion_joint(self, kind, name, parent, child, origin, axis, at, limits, home) -> str:
        name = str(name)
        where = f"{kind} '{name}'"
        o, n = np.zeros(3), np.array([0.0, 0.0, 1.0])
        if at is not None:
            if origin is not None or axis is not None:
                self._errors.append(f"{where}: give either at= or origin=/axis=, not both")
            if isinstance(at, Location):
                T = from_location(at)
                o, n = T[:3, 3].copy(), self._dir(T[:3, 2], where)
            elif isinstance(at, Axis):
                o, n = vec3(at.position), self._dir(at.direction, where)
            else:
                self._errors.append(f"{where}: at= must be a build123d Location or Axis (got {type(at).__name__})")
        elif origin is None or axis is None:
            self._errors.append(f"{where}: needs origin= and axis= (or at=)")
        else:
            o, n = self._vec(origin, f"{where} origin"), self._dir(axis, f"{where} axis")
        lim = None
        if limits is not None:
            try:
                lo, hi = (float(x) for x in limits)
            except (TypeError, ValueError):
                self._errors.append(f"{where}: limits must be (min, max) (got {limits!r})")
            else:
                if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                    self._errors.append(f"{where}: limits must be finite with min < max (got {limits!r})")
                lim = (lo, hi)
        home = float(home)
        if not math.isfinite(home):
            self._errors.append(f"{where}: home must be finite (got {home})")
        self._add_joint(Joint(name, kind, str(parent), str(child), o, n, lim, home))
        return name

    def _add_coupling(self, kind, driver, driven, ratio, offset, name) -> str:
        """Append a coupling; ``ratio=None`` means a screw/rack rule computes it from geometry."""
        cname = str(name) if name is not None else f"{kind}_{driven}"
        if (ratio is not None and not math.isfinite(ratio)) or not math.isfinite(offset):
            self._errors.append(f"{kind} '{cname}': ratio and offset must be finite (got {ratio}, {offset})")
        self.couplings.append(Coupling(cname, str(driver), str(driven),
                                       float("nan") if ratio is None else ratio, offset, kind))
        return cname

    def _add_ratio_rule(self, coupling: Coupling, rule: Callable[[], float | None]) -> None:
        self._ratio_rules.append((coupling, rule))
        try:  # joints usually exist already; otherwise validate() resolves it
            ratio = rule()
        except ValueError:
            return
        if ratio is not None:
            coupling.ratio = ratio

    def _add_pair(self, bucket: set, what: str, a: str, b: str) -> None:
        if str(a) == str(b):
            self._errors.append(f"{what}('{a}', '{b}'): needs two different parts")
        bucket.add(frozenset((str(a), str(b))))

    def _screw_ratio(self, driver: str, driven: str, lead: float, hand: str) -> float | None:
        jd, jn = self.joints.get(driver), self.joints.get(driven)
        if jd is None or jn is None or (jd.kind, jn.kind) != ("revolute", "prismatic"):
            return None  # reported by the name/kind checks
        if _angle_between_lines(jd.axis, jn.axis) > _PARALLEL_TOL:
            raise ValueError(f"screw axis of '{driver}' {jd.axis.round(4).tolist()} is not parallel to "
                             f"the travel axis of '{driven}' {jn.axis.round(4).tolist()}")
        s = 1.0 if jd.axis @ jn.axis > 0 else -1.0
        return (-1.0 if hand == "right" else 1.0) * s * lead / 360.0

    def _belt_ratio(self, driver: str, driven: str, ratio: float) -> float | None:
        jd, jn = self.joints.get(driver), self.joints.get(driven)
        if jd is None or jn is None or (jd.kind, jn.kind) != ("revolute", "revolute"):
            return None  # reported by the name/kind checks
        if _angle_between_lines(jd.axis, jn.axis) > _PARALLEL_TOL:
            raise ValueError(f"pulley axis of '{driver}' {jd.axis.round(4).tolist()} is not parallel to the "
                             f"pulley axis of '{driven}' {jn.axis.round(4).tolist()}")
        return ratio if jd.axis @ jn.axis > 0 else -ratio

    def _rack_ratio(self, driver: str, driven: str, r: float, sign: int | None) -> float | None:
        jd, jn = self.joints.get(driver), self.joints.get(driven)
        if jd is None or jn is None or (jd.kind, jn.kind) != ("revolute", "prismatic"):
            return None
        if sign is None:
            rack_part = self.parts.get(jn.child)
            if rack_part is None or not _is_shape(rack_part.shape):
                return None
            lo, hi = self._bbox(rack_part)
            # the pitch point moves along axis_p × (c − origin_p) for a positive pinion rotation
            v = float(np.cross(jd.axis, (lo + hi) / 2 - jd.origin) @ jn.axis)
            if abs(v) < 1e-9 * self._char_length():
                raise ValueError(f"can't infer the rack direction from geometry (rack '{jn.child}' center "
                                 f"is on the pinion axis or travels along it); pass sign=+1 or -1")
            sign = 1 if v > 0 else -1
        return sign * math.pi * r / 180.0

    # ---------------------------------------------------------------------------- geometry queries

    def _bbox(self, part: Part) -> tuple[np.ndarray, np.ndarray]:
        key = id(part.shape)
        if key not in self._bbox_cache:
            bb = part.shape.bounding_box()
            self._bbox_cache[key] = (vec3(bb.min), vec3(bb.max))
        return self._bbox_cache[key]

    def _extent(self) -> tuple[np.ndarray, np.ndarray]:
        """Home-pose bounding box of all (valid) parts."""
        boxes = [self._bbox(p) for p in self.parts.values() if _is_shape(p.shape)]
        if not boxes:
            return np.zeros(3), np.zeros(3)
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)

    def _char_length(self) -> float:
        """Characteristic length L (mm): home bounding-box diagonal of the whole assembly."""
        lo, hi = self._extent()
        return max(float(np.linalg.norm(hi - lo)), 1.0)

    def _distance(self, part: Part, p: np.ndarray, n: np.ndarray | None) -> float:
        """Distance (mm) from ``part`` to the point ``p`` (n None) or to the line through p along n.

        Points buried inside a solid count as distance 0 (OCC's distance only sees the boundary;
        containment is classified per solid, see ``geom.inside``).
        """
        lo, hi = self._extent()
        half = self._char_length() + float(np.linalg.norm(p - (lo + hi) / 2))  # line spans the whole assembly
        key = (part.name, id(part.shape), tuple(np.round(p, 9)), None if n is None else tuple(np.round(n, 12)),
               round(half, 6))
        if key not in self._dist_cache:
            shape = part.shape
            if n is not None:
                d = shape.distance_to(Edge.make_line(tuple(p - half * n), tuple(p + half * n)))
                if d > 0 and self._line_pierces(part, p, n):
                    d = 0.0
            elif inside(shape, p):
                d = 0.0
            else:
                d = shape.distance_to(tuple(float(x) for x in p))
            self._dist_cache[key] = float(d)
        return self._dist_cache[key]

    def _line_pierces(self, part: Part, p: np.ndarray, n: np.ndarray, samples: int = 9) -> bool:
        """Does the line (p, n) pass through the interior of ``part``?

        OCC's line/face extremum can miss a crossing whose face-classification ray grazes a vertex —
        e.g. a spur gear's own axis through its tooth-symmetric faces reports the root radius — so
        sample ``is_inside`` along the line's chord through the part's bounding box.
        """
        lo, hi = self._bbox(part)
        t0, t1 = -math.inf, math.inf
        for k in range(3):  # slab clipping of p + t·n against the box
            if abs(n[k]) < 1e-12:
                if not lo[k] <= p[k] <= hi[k]:
                    return False
                continue
            a, b = (lo[k] - p[k]) / n[k], (hi[k] - p[k]) / n[k]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
        return any(inside(part.shape, p + t * n) for t in np.linspace(t0, t1, samples + 2)[1:-1])

    # ---------------------------------------------------------------------------- tree queries

    def parent_joint(self, part: str) -> Joint | None:
        """The joint whose child is ``part`` (None for ground/floating parts)."""
        return next((j for j in self.joints.values() if j.child == part), None)

    def _chain(self, part: str) -> list[str]:
        """Joint names from the ground down to ``part`` (includes fixed joints)."""
        chain, seen, p = [], set(), part
        while p not in seen and (j := self.parent_joint(p)) is not None:
            seen.add(p)
            chain.append(j.name)
            p = j.parent
        return chain[::-1]

    def _tree_path(self, a: str, b: str) -> list[str]:
        """Non-fixed joints on the tree path a → LCA → b (the loop a pin between a and b closes)."""
        ca, cb = self._chain(a), self._chain(b)
        k = 0
        while k < min(len(ca), len(cb)) and ca[k] == cb[k]:
            k += 1
        return [j for j in ca[k:][::-1] + cb[k:] if self.joints[j].kind != "fixed"]

    def rigid_groups(self) -> dict[str, int]:
        """Part -> rigid-group id: union-find over fixed joints; all ground parts share one group."""
        root = {p: p for p in self.parts}

        def find(x: str) -> str:
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        def union(x: str, y: str) -> None:
            root[find(x)] = find(y)

        for j in self.joints.values():
            if j.kind == "fixed" and j.parent in root and j.child in root:
                union(j.child, j.parent)
        grounds = [p for p, part in self.parts.items() if part.ground]
        for g in grounds[1:]:
            union(g, grounds[0])
        ids: dict[str, int] = {}
        return {p: ids.setdefault(find(p), len(ids)) for p in self.parts}

    def is_joined(self, a: str, b: str) -> bool:
        """Is the pair exempt from ``clearance`` (only overlap counts)? True for parts of one rigid
        group, and for the ``joined_pairs()`` a joint or pin connects directly."""
        g = self.rigid_groups()
        if a not in g or b not in g:
            return False
        return g[a] == g[b] or frozenset((a, b)) in self.joined_pairs()

    def _links(self) -> list[tuple[str, str, np.ndarray, np.ndarray | None]]:
        """(part a, part b, point, axis | None) of every moving joint and every pin: the joint's
        parent/child with its origin + axis line, the pin's parts with its point (+ hinge axis)."""
        links = [(j.parent, j.child, j.origin, j.axis) for j in self.joints.values() if j.kind != "fixed"]
        return links + [(p.a, p.b, p.point, p.axis) for p in self.pins]

    def _joined_key(self) -> tuple:
        return (tuple((n, id(p.shape), p.ground) for n, p in self.parts.items()),
                tuple((j.name, j.kind, j.parent, j.child, j.origin.tobytes(), j.axis.tobytes())
                      for j in self.joints.values()),
                tuple((p.name, p.a, p.b, p.point.tobytes(), None if p.axis is None else p.axis.tobytes())
                      for p in self.pins),
                frozenset(self.checked), self.pin_tol)

    def joined_pairs(self) -> set[frozenset]:
        """Part pairs of two different rigid bodies whose closeness a joint or pin explains.

        A joint (or pin) connects two rigid bodies, but only the parts that carry it bear on each
        other: the two parts it names (parent/child, pin a/b); on both sides, the parts within
        ``pin_tol`` of its axis line (ball pin: its point) — shafts, bearings, bushings; and any
        cross pair that touches at home (a bushing on its guide rod, a large bearing on its
        shaft). Every other part of the two bodies — posts, brackets and motors on the ground
        body, anything fixed to a moving link away from its pivot — keeps ``clearance``.
        ``check_clearance`` pairs are removed. Pairs within one rigid group are not listed
        (``is_joined`` covers them).
        """
        key = self._joined_key()
        if self._joined_memo is not None and self._joined_memo[0] == key:
            return self._joined_memo[1]
        groups = self.rigid_groups()
        bodies: dict[int, list[str]] = {}
        for name, g in groups.items():
            if _is_shape(self.parts[name].shape):
                bodies.setdefault(g, []).append(name)
        pairs: set[frozenset] = set()
        for a, b, point, axis in self._links():
            if a not in groups or b not in groups or groups[a] == groups[b]:
                continue
            pairs.add(frozenset((a, b)))
            side_a, side_b = bodies.get(groups[a], []), bodies.get(groups[b], [])
            carry_a = [x for x in side_a if self._carries(x, point, axis)]
            carry_b = [y for y in side_b if self._carries(y, point, axis)]
            pairs.update(frozenset((x, y)) for x in carry_a for y in carry_b)
            pairs.update(frozenset((x, y)) for x in side_a for y in side_b
                         if frozenset((x, y)) not in pairs and self._touching(x, y))
        pairs -= self.checked
        self._joined_memo = (key, pairs)
        return pairs

    def _carries(self, name: str, point: np.ndarray, axis: np.ndarray | None) -> bool:
        """Does part ``name`` come within ``pin_tol`` of a joint's axis line (or a ball pin's point)?"""
        try:
            return self._distance(self.parts[name], point, axis) <= self.pin_tol
        except Exception:  # OCC kernel failure: not provably on the axis
            return False

    def _touching(self, a: str, b: str) -> bool:
        """Do parts a and b touch (or overlap) at home?"""
        pa, pb = self.parts[a], self.parts[b]
        key = (a, id(pa.shape), b, id(pb.shape))
        if key not in self._touch_cache:
            (lo_a, hi_a), (lo_b, hi_b) = self._bbox(pa), self._bbox(pb)
            gap = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo_b - hi_a, lo_a - hi_b))))
            touching = False
            if gap <= _TOUCH:
                try:
                    ext = BRepExtrema_DistShapeShape(pa.shape.wrapped, pb.shape.wrapped, Extrema_ExtFlag_MIN)
                    touching = bool(ext.IsDone() and ext.NbSolution() > 0 and ext.Value() <= _TOUCH)
                except Exception:  # OCC kernel failure: not provably touching
                    touching = False
            self._touch_cache[key] = touching
        return self._touch_cache[key]

    def meshing_pairs(self) -> set[frozenset]:
        """Part pairs whose teeth mesh through a gear/rack coupling.

        The two coupled joints move two rigid bodies (their children's rigid groups). On each
        side the tooth-bearing candidates are the parts tagged as gears/racks (library
        ``spur_gear``/``gear_pair``/``rack``); an untagged side offers its joint child when that
        engages the other side (a gear drawn as the joint's child), else all its parts (a gear
        fixed to a shaft that is the child). Meshing pairs are the candidate cross pairs whose home
        bounding boxes (inflated by ``clearance``) overlap — none when nothing engages (a
        ``gear_mesh`` WARN from ``validate_warnings``). Everything else — a crank fixed to a
        pinion, the pulleys of a ``belt`` — is clearance-checked normally.
        """
        groups = self.rigid_groups()
        pairs: set[frozenset] = set()
        for c in self.couplings:
            pairs |= self._coupling_mesh(c, groups) or set()
        return pairs

    def _coupling_mesh(self, c: Coupling, groups: dict[str, int]) -> set[frozenset] | None:
        """The meshing pairs of one gear/rack coupling (see ``meshing_pairs``); None when the
        coupling has no teeth to mesh (another kind, unknown joints, or one rigid body)."""
        jd, jn = self.joints.get(c.driver), self.joints.get(c.driven)
        if c.kind not in ("gear", "rack") or jd is None or jn is None:
            return None
        gd, gn = groups.get(jd.child), groups.get(jn.child)
        if gd is None or gn is None or gd == gn:
            return None
        body_d = [p for p, g in groups.items() if g == gd and _is_shape(self.parts[p].shape)]
        body_n = [p for p, g in groups.items() if g == gn and _is_shape(self.parts[p].shape)]
        toothed_d = [p for p in body_d if self.parts[p].kind in _TOOTHED]
        toothed_n = [p for p in body_n if self.parts[p].kind in _TOOTHED]

        def engage(a: str, b: str) -> bool:
            return _boxes_overlap(self._bbox(self.parts[a]), self._bbox(self.parts[b]), self.clearance)

        def candidates(body: list[str], toothed: list[str], child: str, other: list[str]) -> list[str]:
            if toothed:
                return toothed
            return [child] if child in body and any(engage(child, o) for o in other) else body

        cand_d = candidates(body_d, toothed_d, jd.child, toothed_n or body_n)
        cand_n = candidates(body_n, toothed_n, jn.child, toothed_d or body_d)
        return {frozenset((a, b)) for a in cand_d for b in cand_n if engage(a, b)}

    # ---------------------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Every modeling error, in one pass. Never raises; an empty list means valid."""
        errors = list(self._errors)
        if not self.parts:
            errors.append("assembly has no parts")
        elif not any(p.ground for p in self.parts.values()):
            errors.append("no ground part: mark at least one part ground=True")
        self._check_parts(errors)
        self._check_tree(errors)
        self._check_couplings(errors)
        self._check_pins(errors)
        self._check_references(errors)
        return errors

    def _check_parts(self, errors: list[str]) -> None:
        """Every part must be solid material: clearance, mass properties and STL export need it."""
        for p in self.parts.values():
            if not _is_shape(p.shape):
                continue
            hit = self._solid_cache.get(id(p.shape))
            if hit is None or hit[0] is not p.shape:
                try:
                    solids = p.shape.solids()
                    hit = (p.shape, len(solids), sum(abs(float(s.volume)) for s in solids))
                except Exception as exc:  # OCC kernel failure: report, never raise
                    errors.append(f"part '{p.name}': could not compute volume ({exc})")
                    continue
                self._solid_cache[id(p.shape)] = hit
            _, n_solids, vol = hit
            if vol <= _MIN_VOLUME:
                what = "contains no solid" if not n_solids else "has zero volume"
                mass = " (mass_g is set, but mass properties need a solid)" if p.mass_g is not None else ""
                errors.append(f"part '{p.name}' is not a closed solid: the shape {what}, zero volume{mass} "
                              f"— did a boolean remove everything, or is it a Face/Shell/Wire?")

    def _check_tree(self, errors: list[str]) -> None:
        parts = list(self.parts)
        parents: dict[str, list[Joint]] = {}
        for j in self.joints.values():
            where = f"{j.kind} '{j.name}'"
            ok = True
            for role, pname in (("parent", j.parent), ("child", j.child)):
                if pname not in self.parts:
                    errors.append(f"{where}: unknown {role} part '{pname}'{_suggest(pname, parts)}")
                    ok = False
            if not ok:
                continue
            if j.parent == j.child:
                errors.append(f"{where}: parent and child are the same part '{j.child}'")
                continue
            parents.setdefault(j.child, []).append(j)
            if j.kind != "fixed" and j.limits is not None:
                lo, hi = j.limits
                if lo < hi and not lo - 1e-9 <= j.home <= hi + 1e-9:
                    errors.append(f"{where}: home {j.home:g} is outside limits [{lo:g}, {hi:g}]")
        claimed = {j.child for j in self.joints.values()}  # incl. joints with an unknown parent (reported above)
        for pname, part in self.parts.items():
            js = parents.get(pname, [])
            if part.ground and js:
                errors.append(f"ground part '{pname}' cannot be the child of joint '{js[0].name}'")
            elif not part.ground and pname not in claimed:
                errors.append(f"part '{pname}' is floating: it is not ground=True and no joint has it as child "
                              f"(add a joint, or fix('{pname}', parent))")
            if len(js) > 1:
                errors.append(f"part '{pname}' is the child of {len(js)} joints ({', '.join(j.name for j in js)}); "
                              f"each part has one parent joint — close loops with pin()")
        reported: set[frozenset] = set()
        for start in self.parts:
            seen: dict[str, int] = {}
            path: list[str] = []
            p = start
            while p in parents and not self.parts[p].ground:
                if p in seen:
                    cycle = path[seen[p]:]
                    if frozenset(cycle) not in reported:
                        reported.add(frozenset(cycle))
                        errors.append(f"joint cycle: {' → '.join(cycle + [p])} never reaches a ground part")
                    break
                seen[p] = len(path)
                path.append(p)
                p = parents[p][0].parent

    def _check_couplings(self, errors: list[str]) -> None:
        for coupling, rule in self._ratio_rules:
            try:
                ratio = rule()
            except ValueError as exc:
                errors.append(f"{coupling.kind} '{coupling.name}': {exc}")
            else:
                if ratio is not None:
                    coupling.ratio = ratio
        joints = list(self.joints)
        driven_by: dict[str, list[str]] = {}
        edges: dict[str, list[str]] = {}
        for c in self.couplings:
            where = f"{c.kind} '{c.name}'"
            known = True
            for role, jname in (("driver", c.driver), ("driven", c.driven)):
                if jname not in self.joints:
                    errors.append(f"{where}: unknown {role} joint '{jname}'{_suggest(jname, joints)}")
                    known = False
            if not known:
                continue
            if c.driver == c.driven:
                errors.append(f"{where}: a joint can't drive itself ('{c.driver}')")
                continue
            kd, kn = self.joints[c.driver].kind, self.joints[c.driven].kind
            need = _COUPLING_KINDS.get(c.kind)
            if need is not None and (kd, kn) != need:
                errors.append(f"{where}: needs a {need[0]} driver and a {need[1]} driven joint, got "
                              f"'{c.driver}' ({kd}) → '{c.driven}' ({kn})")
                continue
            if "fixed" in (kd, kn):
                errors.append(f"{where}: fixed joints can't be coupled ('{c.driver}' {kd} → '{c.driven}' {kn})")
                continue
            driven_by.setdefault(c.driven, []).append(c.name)
            edges.setdefault(c.driver, []).append(c.driven)
        for jname, names in driven_by.items():
            if len(names) > 1:
                errors.append(f"joint '{jname}' is driven by {len(names)} couplings ({', '.join(names)})")
        # coupling cycle: DFS over driver -> driven edges
        state: dict[str, int] = {}  # 1 = on stack, 2 = done
        stack: list[str] = []

        def visit(j: str) -> None:
            state[j] = 1
            stack.append(j)
            for k in edges.get(j, []):
                if state.get(k) == 1:
                    cyc = stack[stack.index(k):] + [k]
                    errors.append(f"coupling cycle: {' → '.join(cyc)}")
                elif k not in state:
                    visit(k)
            stack.pop()
            state[j] = 2

        for j in list(edges):
            if j not in state:
                visit(j)

    def _check_pins(self, errors: list[str]) -> None:
        parts = list(self.parts)
        for pin in self.pins:
            where = f"pin '{pin.name}'"
            known = [x for x in (pin.a, pin.b) if x in self.parts]
            for x in (pin.a, pin.b):
                if x not in self.parts:
                    errors.append(f"{where}: unknown part '{x}'{_suggest(x, parts)}")
            if pin.a == pin.b:
                errors.append(f"{where}: needs two different parts (got '{pin.a}' twice)")
                continue
            for x in known:
                part = self.parts[x]
                if not _is_shape(part.shape):
                    continue
                try:
                    d = self._distance(part, pin.point, pin.axis)
                except Exception as exc:  # OCC kernel failure: report, never raise
                    errors.append(f"{where}: could not measure distance to part '{x}' ({exc})")
                    continue
                if d > self.pin_tol:
                    what = "axis line" if pin.axis is not None else "point"
                    errors.append(f"pin_off_part: pin '{pin.name}' {what} is {d:.3g} mm from part '{x}' "
                                  f"(pin_tol {self.pin_tol:g} mm)")

    def _check_references(self, errors: list[str]) -> None:
        parts, joints, probes = list(self.parts), list(self.joints), [p.name for p in self.probes]
        for pr in self.probes:
            if pr.part not in self.parts:
                errors.append(f"probe '{pr.name}': unknown part '{pr.part}'{_suggest(pr.part, parts)}")
        for a in self.actuators.values():
            if a.joint not in self.joints:
                errors.append(f"actuator: unknown joint '{a.joint}'{_suggest(a.joint, joints)}")
            elif self.joints[a.joint].kind == "fixed":
                errors.append(f"actuator: joint '{a.joint}' is fixed")
        for what, bucket in (("allow_contact", self.allowed), ("ignore", self.ignored),
                             ("check_clearance", self.checked)):
            for pair in bucket:
                for x in sorted(pair):
                    if x not in self.parts:
                        errors.append(f"{what}: unknown part '{x}'{_suggest(x, parts)}")
        for pair in sorted(self.checked & (self.allowed | self.ignored), key=sorted):
            other = "allow_contact" if pair in self.allowed else "ignore"
            a, b = sorted(pair)
            errors.append(f"check_clearance('{a}', '{b}') contradicts {other} for the same pair")
        for s in self.studies:
            where = f"study '{s.name}'"
            if not isinstance(s.drive, dict) or not s.drive:
                errors.append(f"{where}: drive must be a non-empty dict {{joint: values}}")
            else:
                for jname in s.drive:
                    if jname not in self.joints:
                        errors.append(f"{where}: unknown joint '{jname}'{_suggest(jname, joints)}")
            if s.loop not in _LOOP_MODES:
                errors.append(f"{where}: loop must be one of {', '.join(_LOOP_MODES)} (got '{s.loop}')")
            if not (math.isfinite(s.duration) and s.duration > 0):
                errors.append(f"{where}: duration must be > 0 s (got {s.duration})")
        for t in self.targets:
            self._check_target(t, joints, probes, errors)

    def _check_target(self, t: Target, joints: list[str], probes: list[str], errors: list[str]) -> None:
        where = f"target '{t.label}'"
        if t.severity not in _SEVERITIES:
            errors.append(f"{where}: severity must be one of {', '.join(_SEVERITIES)} (got '{t.severity}')")
        if t.min is not None and t.max is not None and t.min > t.max:
            errors.append(f"{where}: min {t.min:g} > max {t.max:g}")
        if callable(t.metric):
            return
        if not isinstance(t.metric, str):
            errors.append(f"{where}: metric must be a string or a callable (got {type(t.metric).__name__})")
            return
        kind, sep, arg = t.metric.partition(":")
        kind = kind.strip()
        if kind not in _METRICS or bool(sep) != (_METRICS[kind] is not None):
            errors.append(f"{where}: unknown metric '{t.metric}'{_suggest(kind, _METRICS)}")
            return
        expects = _METRICS[kind]

        def need(name: str, pool: list[str], what: str) -> None:
            if name not in pool:
                errors.append(f"{where}: unknown {what} '{name}' in '{t.metric}'{_suggest(name, pool)}")

        if expects == "joint":
            need(arg.strip(), joints, "joint")
        elif expects == "part":
            need(arg.strip(), list(self.parts), "part")
        elif expects == "part_pair":
            names = [x.strip() for x in arg.split(",")]
            if len(names) != 2:
                errors.append(f"{where}: '{t.metric}' needs two parts: {kind}:<partA>,<partB>")
            else:
                for x in names:
                    need(x, list(self.parts), "part")
        elif expects == "probe":
            need(arg.strip(), probes, "probe")
        elif expects == "probe_pair":
            names = [x.strip() for x in arg.split(",")]
            if len(names) != 2:
                errors.append(f"{where}: '{t.metric}' needs two probes: {kind}:<probeA>,<probeB>")
            else:
                for x in names:
                    need(x, probes, "probe")
        elif expects == "probe_axis":
            name, dot, ax = arg.strip().rpartition(".")
            if not dot or ax not in ("x", "y", "z"):
                errors.append(f"{where}: '{t.metric}' must look like delta:<probe>.x|y|z")
            else:
                need(name, probes, "probe")

    def validate_warnings(self) -> list[tuple[str, str]]:
        """Non-fatal findings as (code, message): ``near_planar``, ``joint_off_part`` and
        ``gear_mesh`` (a gear/rack coupling whose bodies' teeth do not engage at home).

        Returns [] for an invalid model (errors take precedence).
        """
        if self.validate():
            return []
        warnings: list[tuple[str, str]] = []
        groups = self.rigid_groups()
        for c in self.couplings:
            if self._coupling_mesh(c, groups) == set():
                warnings.append(("gear_mesh",
                                 f"{c.kind} '{c.name}': nothing moved by '{c.driver}' engages anything moved by "
                                 f"'{c.driven}' at home (no gear/rack within clearance) — move the gears into mesh, "
                                 f"or use belt() for a belt/chain drive, couple() for a ratio without modelled teeth"))
        for pin in self.pins:
            loop = [self.joints[j] for j in self._tree_path(pin.a, pin.b)]
            rev_axes = [j.axis for j in loop if j.kind == "revolute"] + ([pin.axis] if pin.axis is not None else [])
            if len(rev_axes) < 2:
                continue
            worst = max(_angle_between_lines(a, b) for i, a in enumerate(rev_axes) for b in rev_axes[i + 1:])
            # prismatic travel in a planar loop must be perpendicular to the hinge axes
            worst_perp = max((abs(math.pi / 2 - _angle_between_lines(j.axis, a))
                              for j in loop if j.kind == "prismatic" for a in rev_axes), default=0.0)
            if worst <= _PLANAR_LOOSE and (worst > _PLANAR_TIGHT or _PLANAR_TIGHT < worst_perp <= _PLANAR_LOOSE):
                dev = max(worst, worst_perp)
                warnings.append(("near_planar",
                                 f"loop closed by pin '{pin.name}' is nearly planar: axes deviate by {dev:.3g} rad "
                                 f"(> 1e-9, ≤ 1e-3) — make hinge axes exactly parallel (sliders exactly "
                                 f"perpendicular) or the loop may lock"))
        # A joint must touch its parent and child *bodies*: the named part or anything rigidly
        # attached to it (e.g. a shaft whose revolute parent is a plate but which runs in a
        # bearing fixed to that plate). A prismatic joint's axis line only gives the direction
        # of travel — where it runs has no kinematic effect — so it is not checked.
        for j in self.joints.values():
            if j.kind != "revolute":
                continue
            for role, pname in (("parent", j.parent), ("child", j.child)):
                body = [pname] + [p for p, g in groups.items() if g == groups[pname] and p != pname]
                d = math.inf
                for name in body:
                    part = self.parts[name]
                    if not _is_shape(part.shape):
                        continue
                    try:
                        d = min(d, self._distance(part, j.origin, j.axis))
                    except Exception:  # OCC kernel failure: skip the (non-fatal) check
                        continue
                    if d <= self.pin_tol:
                        break
                if self.pin_tol < d < math.inf:
                    attached = " or anything fixed to it" if len(body) > 1 else ""
                    warnings.append(("joint_off_part",
                                     f"{j.kind} '{j.name}' axis line is {d:.3g} mm from its {role} '{pname}'"
                                     f"{attached} (pin_tol {self.pin_tol:g} mm) — check origin/axis"))
        return warnings
