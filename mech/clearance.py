"""Clearance and interference between parts (spec §4.6).

Every unordered pair of parts that isn't ``ignore``d (nor involves a ``virtual`` part) is
classified per pose:

* **interference** — the parts overlap by more than ``interference_tol`` mm³, or the overlap's
  mean depth 2V/A (A = surface area of the common solid) exceeds 0.02 mm; an ``allow_contact``
  pair interferes only when that mean depth exceeds its ``max_depth`` (default 0.1 mm);
* **contact** — touching or overlapping within those tolerances, for joined / allowed / meshing
  pairs (other pairs: **tight**);
* **tight** — 0 < gap < ``clearance`` for pairs that are not joined, allowed or meshing (with the
  targets' float-noise allowance: a gap that lands on ``clearance`` by construction meets it);
* **ok** — everything else.

Parts are measured as solid material: the solids of a part are fused when they overlap (a
list-of-shapes part, see ``massprops.solid_body``) and containment is classified solid by solid,
so multi-solid parts (library motors, bearings) behave like one body.

Pipeline per pair and pose (part a at home, part b moved by M = inv(T_a)·T_b): lower bounds from
the two home boxes (world-AABB gap and a separating-axis test along the boxes' face normals,
vectorised over all pairs) and, for parts that only ever turn about a common hinge line (a
revolute or hinge pin between them, or one from each to a common body along one line), the gap
between their (height, radius) profiles about that line, which holds at every pose (the profiles
are boundaries: a part buried in the other is ruled out once, at the home relative pose, since
containment cannot change while turning keeps the boundaries apart); for parts a
prismatic joint links (a piston in its barrel's cooling fins), the gap between their radius
bands about the slide line (``tess.radial_band``), which holds however far they slide →
``BRepExtrema_ShapeProximity`` on cached tessellations at τ plus the meshes' error margin (τ = the
distance that decides the pair's status: touching for joined / allowed / meshing pairs,
``clearance`` — or the running minimum — for the rest), which proves "farther apart than τ" or
names the faces that may come closer → when those faces all lie on planes, cylinders or spheres,
the distance between their infinite surfaces (``tess.surface_gap``: a ball in its socket ring, a
pin in a bore) may prove τ without more → else the exact ``BRepExtrema_DistShapeShape`` on those
faces only (every other face is provably farther than τ) → containment, when the boundaries are
apart, from one boundary point per solid (the winding number on the other part's mesh when the
point is clear of that mesh's error band, else OCC's per-solid classifier) → a boolean common
only for touching / overlapping pairs, on the solids whose boundaries meet (plus any buried
solid); ``allow_contact(max_depth=None)`` pairs skip it. Meshing gear/rack pairs are
interference-only: the boolean runs when their tessellations intersect.

An overlap that would be reported as an interference is validated first: OCC's common of
coincident surfaces (a shaft in a bore of its own diameter) can return a whole solid at an
unlucky rotation. Every cell of the common must hold a point, deep inside the cell, that lies in
both parts (classified against the parts themselves); a cell with none is a boolean artefact and
is dropped — then the boolean is repeated with its arguments swapped and the larger validated
overlap kept.

Every measurement is cached per pair by the relative pose rounded to 1e-6, after mapping it to a
canonical representative when a part is a body of revolution (``tess.revolution_axis``): a pin
turning in its eye or a shaft in its bearing is measured once. Boolean commons are cached the
same way per pair of solid subsets, so a motor whose round shaft solid turns in a pulley's bore
repeats too. Closest points and overlap locations are stored in a's home frame and mapped
through T_a.

A sweep also guards the gaps *between* frames: for every pair held to ``clearance`` — and, for
overlap only, every joined / allowed pair proven at least ``clearance`` apart at both
ends of an interval (a joined stop the slider meets at its dead centre between two frames) — it
bounds how far the two parts move relative to each other from one closed frame to the next (the support
points of each part's convex hull, vectorised over all pairs): the chord between the two poses
plus a quarter of the path's bend — its second difference, from the neighbouring frames (or the
pose halfway) — since a motion that reverses between two frames (a slider's dead centre) has a
chord near 0. When that could close the gap,
the gaps at the two frames are first proven large enough where needed (one proximity query
each); a pair turning about its hinge line is settled when the faces within reach of each other
keep their (height, radius) profiles apart; only then is the interval bisected, with intermediate
poses solved by the kinematics, until the pair is shown clear or the collision is found
(reported at the nearest frame, ``PairResult.at`` = the fractional frame). Intervals given up on
unproven are counted (``stats["sub_unresolved"]``, per pair in ``stats["unresolved_pairs"]``).

``check_clearance`` pairs that a joint or pin links (both parts within ``pin_tol`` of its axis
line, a ball pin: of its point, or one running in the other's bore of any size) are held to
``clearance`` everywhere but the carried bore region —
the material within (the larger part's distance to that axis + clearance) of it — while the whole
parts are still checked for overlap (a joined twin of the pair).

``sweep`` reports per-frame progress and ``SweepResult.stats`` (module ``SweepResult``).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.Extrema import Extrema_ExtFlag_MIN
from OCP.gp import gp_Ax2, gp_Dir, gp_Lin, gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape

from . import tess
from .assembly import ALLOW_DEPTH, Assembly
from .geom import to_location, transform_aabb, transform_points
from .massprops import solid_body

__all__ = ["PairResult", "SweepResult", "ClearanceChecker", "TOUCH", "DEPTH_TOL", "ALLOW_DEPTH", "CLEARANCE_RTOL"]

TOUCH = 1e-6  # mm: a gap at or below this is touching
CLEARANCE_RTOL = 1e-9  # float-noise allowance below `clearance` (targets.BOUND_RTOL's): 0.3 − 3e-15 is not tight
DEPTH_TOL = 0.02  # mm: an overlap whose mean depth 2V/A exceeds this interferes whatever its volume
_VOLUME_EPS = 1e-9  # mm³: a boolean common below this is a touch, not an overlap
_KEY_DECIMALS = 6  # relative-pose cache key rounding (1e-6 in R, 1e-6 mm in t)
_MESH_DEFLECTION = 0.01  # mm: chordal deflection of the cached tessellations
_MESH_ANGLE = 0.2  # rad: angular deflection of the same
# A tessellation sits within its deflection of the true surface; twice that per part (or 3x the
# deflection BRepMesh reports, if larger) covers BRepMesh's slack.
_MESH_MARGIN = 2 * _MESH_DEFLECTION
_RANK = {"interference": 3, "tight": 2, "contact": 1, "ok": 0}
# Between-frame guard: bisection depth (1/32 of a frame), the relative rotation above which the
# chord-based motion bound is not trusted (the interval is bisected regardless), and a safety
# factor on that bound for paths that are not exact screw motions.
_SUB_DEPTH = 5
_SUB_MAX_TURN = math.pi / 2
_SUB_SAFETY = 1.25
_SUB_BUDGET = 200  # sub-frame measurements per pair and sweep
_CELL_RAYS = 24  # boolean validation: chords cast through a suspicious cell
_CELL_DEPTH = 1e-4  # mm (or 1e-3 x the cell size, if larger): how deep a validation point must be
_SLOWEST = 5  # stats["slowest"]: the pairs that took longest

Status = Literal["interference", "tight", "contact", "ok"]


@dataclass
class PairResult:
    a: str
    b: str
    distance: float | None  # mm; 0 when touching/overlapping/contained; None for meshing pairs
    volume: float  # overlap mm³ (0 unless distance == 0)
    pa: np.ndarray  # closest point on a (current world); overlap: common-bbox center
    pb: np.ndarray  # closest point on b (current world); overlap: common-bbox center
    status: Status
    joined: bool
    allowed: bool
    meshing: bool
    extent: np.ndarray | None = None  # overlap common-solid bbox size (home frame of a), mm
    location: np.ndarray | None = None  # overlap / closest-point midpoint in a's HOME frame (via inv(T_a))
    depth: float | None = None  # overlap mean depth 2V/A, mm (None unless the parts overlap)
    location_b: np.ndarray | None = None  # the same midpoint in b's HOME frame (via inv(T_b))
    at: float | None = None  # fractional frame when found between two frames (sub-frame check)


@dataclass
class SweepResult:
    per_frame: list[list[PairResult]]  # non-"ok" results per frame (empty for frames whose loops are open)
    worst: dict[frozenset, tuple[int, PairResult]]  # pair -> (frame, worst non-"ok" result)
    # (a, b, signed clearance, frame): −(mean overlap depth) of the deepest interference of any
    # pair when one occurs, else the minimum gap among non-joined/allowed/meshing pairs
    min_clearance: tuple[str, str, float, int] | None
    # seconds, pairs (moving pairs swept), exact (exact distance queries), booleans, proximity
    # (tessellation proximity queries), subframe_poses, slowest ([a, b, seconds] ×≤5), plus
    # pairs_checked, cache_hits, sub_poses (= subframe_poses), sub_measured, frames, ... and, when
    # the between-frame guard gave up on some (sub-)intervals without proving them clear (bisection
    # depth or budget spent, a pose in between not closing), sub_unresolved (how many) and
    # unresolved_pairs ([a, b, n] per pair, most first)
    stats: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Pair:
    a: str
    b: str
    joined: bool
    allowed: bool
    meshing: bool
    rigid: bool  # same rigid group: the relative pose never changes
    max_depth: float | None = ALLOW_DEPTH  # allowed pairs: deepest mean overlap still "contact"
    ka: str = ""  # body measured for a ("" = a's own material; else a check_clearance trimmed copy)
    kb: str = ""

    @property
    def excused(self) -> bool:
        """Closeness is expected (joined, allowed or meshing): only touching/overlap matters."""
        return self.joined or self.allowed or self.meshing

    @property
    def body_a(self) -> str:
        return self.ka or self.a

    @property
    def body_b(self) -> str:
        return self.kb or self.b


@dataclass(frozen=True)
class _Measure:
    """One pair at one relative pose, in a's home frame (a cache entry)."""

    distance: float | None
    volume: float
    depth: float  # mean overlap depth 2V/A, mm
    pa: np.ndarray
    pb: np.ndarray
    extent: np.ndarray | None
    touching: bool  # boundaries touch or overlap


@dataclass
class _Body:
    """Material measured for a part: its own, or a check_clearance pair's trimmed copy."""

    part: str  # the part whose transform moves it
    shape: TopoDS_Shape | None  # solid material (the raw shape when it has no solid)
    box: tuple[np.ndarray, np.ndarray]  # home AABB
    mesh: object = None  # tess.Mesh, False when it can't be meshed, None = not built yet
    axis: object = False  # revolution axis or None (False = not computed yet)
    solids: list | None = None  # its solids (the mesh's when meshed)
    probes: np.ndarray | None = None  # one boundary point per solid
    classifiers: list | None = None
    solid_axes: dict = field(default_factory=dict)
    surfaces: list | None = None  # per mesh face: its plane / cylinder / sphere (tess.face_surfaces)


@dataclass
class _Cell:
    shape: TopoDS_Shape
    volume: float
    area: float
    lo: np.ndarray
    hi: np.ndarray


def _rigid_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _rel(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """inv(Ta) @ Tb for stacks (n, 4, 4) of rigid transforms."""
    Ra_t = np.swapaxes(Ta[:, :3, :3], 1, 2)
    M = np.zeros(Ta.shape)
    M[:, :3, :3] = Ra_t @ Tb[:, :3, :3]
    M[:, :3, 3] = np.einsum("nij,nj->ni", Ra_t, Tb[:, :3, 3] - Ta[:, :3, 3])
    M[:, 3, 3] = 1.0
    return M


def _corners(box: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = box
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _pnt(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def _key(M: np.ndarray) -> bytes:
    return (np.round(M, _KEY_DECIMALS) + 0.0).tobytes()  # + 0.0 folds -0.0 into 0.0


def _explore(shape, kind):
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        yield exp.Current()
        exp.Next()


def _compound(shapes) -> TopoDS_Compound:
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for s in shapes:
        builder.Add(comp, s)
    return comp


def _shape_box(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return _pnt(box.CornerMin()), _pnt(box.CornerMax())


def _distance(sa: TopoDS_Shape, sb: TopoDS_Shape) -> BRepExtrema_DistShapeShape:
    ext = BRepExtrema_DistShapeShape()
    ext.LoadS1(sa)
    ext.LoadS2(sb)
    ext.SetFlag(Extrema_ExtFlag_MIN)
    ext.SetMultiThread(True)
    ext.Perform()
    return ext


def _common_cells(sa: TopoDS_Shape, sb: TopoDS_Shape) -> list[_Cell] | None:
    """The solids of the boolean common (volume, area, bbox); None if OCC failed."""
    op = BRepAlgoAPI_Common(sa, sb)
    if not op.IsDone():
        return None
    shape = op.Shape()
    out = []
    for s in list(_explore(shape, TopAbs_SOLID)) or [shape]:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, props)
        volume = abs(props.Mass())
        if volume <= _VOLUME_EPS:
            continue
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(s, props)
        lo, hi = _shape_box(s)
        out.append(_Cell(s, volume, props.Mass(), lo, hi))
    return out


def _aggregate(cells: list[_Cell]) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """(volume, area, bbox lo, bbox hi) of the cells; None when there are none."""
    if not cells:
        return None
    return (sum(c.volume for c in cells), sum(c.area for c in cells),
            np.min([c.lo for c in cells], axis=0), np.max([c.hi for c in cells], axis=0))


def _map_common(raw, L: np.ndarray | None):
    """A common's (volume, area, lo, hi) taken at a canonical pose, moved by the rotation L."""
    if L is None or raw is None:
        return raw
    volume, area, lo, hi = raw
    corners = _corners((lo, hi)) @ L[:3, :3].T + L[:3, 3]
    return volume, area, corners.min(axis=0), corners.max(axis=0)


def _overlap_measure(common: tuple[float, float, np.ndarray, np.ndarray], distance: float | None) -> _Measure:
    volume, area, lo, hi = common
    center = (lo + hi) / 2
    depth = 2.0 * volume / area if area > 0 else math.inf
    return _Measure(distance, volume, depth, center, center.copy(), hi - lo, True)


def _map_measure(m: _Measure, L: np.ndarray | None) -> _Measure:
    """A measure taken at a canonical pose, moved by the rotation L about a's axis (an overlap's
    box is boxed again: its extent may grow)."""
    if L is None:
        return m
    extent = m.extent
    if extent is not None:
        c = (m.pa + m.pb) / 2
        corners = _corners((c - extent / 2, c + extent / 2)) @ L[:3, :3].T
        extent = corners.max(axis=0) - corners.min(axis=0)
    return _Measure(m.distance, m.volume, m.depth, transform_points(L, m.pa), transform_points(L, m.pb), extent,
                    m.touching)


def _deep_points(cell: TopoDS_Shape, lo: np.ndarray, hi: np.ndarray):
    """Points inside the solid ``cell`` at least ``_CELL_DEPTH`` (relative) from its boundary:
    its centre of mass when that qualifies, then midpoints of chords through the cell along the
    three axes through that centre and along the normals of a sample of its faces."""
    size = float(np.linalg.norm(hi - lo))
    depth = max(_CELL_DEPTH, 1e-3 * size)
    cls = BRepClass3d_SolidClassifier(cell)
    faces = list(_explore(cell, TopAbs_FACE))
    boundary = _compound(faces)  # faces, not the solid: OCC's distance to a solid is 0 inside it

    def deep(p: np.ndarray) -> bool:
        pnt = gp_Pnt(*(float(x) for x in p))
        cls.Perform(pnt, 1e-9)
        if cls.State() != TopAbs_IN:
            return False
        ext = _distance(BRepBuilderAPI_MakeVertex(pnt).Vertex(), boundary)
        return ext.IsDone() and ext.NbSolution() > 0 and ext.Value() >= depth

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(cell, props)
    com = _pnt(props.CentreOfMass())
    if deep(com):
        yield com
    lines = [(com, np.eye(3)[i]) for i in range(3)]
    BRepMesh_IncrementalMesh(cell, max(1e-3, 0.02 * size), False, 0.5, True)
    step = max(1, len(faces) // _CELL_RAYS)
    for f in faces[::step]:
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(TopoDS.Face(f), loc)
        if tri is None or tri.NbTriangles() == 0:
            continue
        trsf = loc.Transformation()
        best = None
        for i in range(1, min(tri.NbTriangles(), 64) + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            p = np.array([_pnt(tri.Node(j).Transformed(trsf)) for j in (n1, n2, n3)])
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            area = float(np.linalg.norm(normal))
            if area > 1e-12 and (best is None or area > best[0]):
                best = (area, p.mean(axis=0), normal / area)
        if best is not None:
            lines.append((best[1], best[2]))
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(cell, 1e-9)
    reach = 2.0 * size + 1.0
    for origin, direction in lines:
        inter.Perform(gp_Lin(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, direction))), -reach, reach)
        if not inter.IsDone():
            continue
        w = sorted({round(inter.WParameter(i), 12) for i in range(1, inter.NbPnt() + 1)})
        for w0, w1 in zip(w, w[1:]):
            p = origin + (w0 + w1) / 2 * direction
            if deep(p):
                yield p


def _same_line(a, b, scale: float) -> bool:
    (p0, u0), (p1, u1) = a, b
    if np.linalg.norm(np.cross(u0, u1)) > 1e-9:
        return False
    w = p1 - p0
    return float(np.linalg.norm(w - np.dot(w, u0) * u0)) <= 1e-7 * max(1.0, scale)


class ClearanceChecker:
    """Classifies part pairs per pose; ``sweep`` runs it over a study with a relative-pose cache."""

    def __init__(self, asm: Assembly, clearance: float | None = None):
        self.asm = asm
        self.clearance = asm.clearance if clearance is None else float(clearance)
        self.interference_tol = asm.interference_tol
        self._bodies: dict[str, _Body] = {}
        for n, p in asm.parts.items():
            if getattr(p, "virtual", False):
                continue  # a ball joint's internal knuckle: no material
            body = solid_body(p.shape) if hasattr(p.shape, "solids") else None
            shape = p.shape if body is None else body
            self._bodies[n] = _Body(n, shape.wrapped, asm._bbox(p))
        names = list(self._bodies)
        groups = asm.rigid_groups()
        meshing = asm.meshing_pairs()
        checked = getattr(asm, "checked", set())
        self._pairs: list[_Pair] = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                key = frozenset((a, b))
                if key in asm.ignored:
                    continue
                pair = _Pair(a, b, asm.is_joined(a, b), key in asm.allowed, key in meshing,
                             groups[a] == groups[b], asm.allow_depth.get(key, ALLOW_DEPTH))
                self._pairs.extend(self._exempted(pair, groups) if key in checked else [pair])
        # (bodies, pair kind, rounded canonical M) -> exact measure, or a float lower bound on the gap
        self._cache: dict[tuple, _Measure | float] = {}
        # (body a, solid ids, body b, solid ids, rounded canonical M) -> boolean common entry
        self._commons: dict[tuple, dict] = {}
        self._kin = None  # Kinematics, built on the first sub-frame check
        self._index: dict[int, tuple] = {}
        self._pair_time: Counter = Counter()
        self._lines: dict[frozenset, tuple | None] | None = None  # rigid-group pair -> common hinge line
        self._static: dict[_Pair, float] = {}  # pair -> lower bound on its gap at every pose
        self._slides: dict[frozenset, tuple] | None = None  # rigid-group pair -> prismatic slide line
        self._slide_static: dict[_Pair, float] = {}  # pair -> radial-band gap about its slide line
        self._sweep_pairs: list[_Pair] | None = None

    # ------------------------------------------------------------------------ public API

    def check_pose(self, transforms: dict[str, np.ndarray], *, include_rigid: bool = True) -> list[PairResult]:
        """Results for one pose (parts missing from ``transforms`` stay at home).

        Lists every pair measured exactly: all pairs that are touching, overlapping or (when not
        joined/allowed/meshing) closer than ``clearance``, plus the pairs the minimum-clearance
        search measured — so the smallest ``distance`` among non-joined/allowed/meshing pairs
        is the exact minimum clearance of the pose. Pairs shown to be farther than that are
        left out. ``include_rigid=False`` skips pairs within one rigid group (their relative
        pose never changes, so the home check covers them).
        """
        pairs = self._pairs if include_rigid else [p for p in self._pairs if not p.rigid]
        results, _, _ = self._check(transforms, pairs, math.inf, Counter())
        if not include_rigid:
            self._index.pop(id(pairs), None)  # a one-off list: drop its batch index
        return results

    def sweep(self, result, progress: Callable[[str, int, int], None] | None = None) -> SweepResult:
        """Check every closed frame of a ``StudyResult``; intra-rigid-group pairs are skipped, and
        so are ``allow_contact(max_depth=None)`` pairs (nothing they do is ever reported).

        Frames whose loops failed to close (``pose.ok`` False) are not checked — their part
        placement is a least-squares best fit, and the runner reports them as ``loop_open``.
        Between consecutive closed frames the pairs held to ``clearance`` get the sub-frame check
        (module docstring). ``min_clearance`` is signed: an interference anywhere makes it
        −(mean overlap depth) of the deepest one. ``progress("clearance", done, total)`` is
        called once per frame, when the frame and the interval before it are done.
        """
        t0 = time.perf_counter()
        self._pair_time = Counter()
        stats: Counter = Counter()
        if self._sweep_pairs is None:
            self._sweep_pairs = [p for p in self._pairs if not p.rigid and not (p.allowed and p.max_depth is None)]
        pairs = self._sweep_pairs
        per_frame: list[list[PairResult]] = []
        worst: dict[frozenset, tuple[int, PairResult]] = {}
        lowers: list[dict[_Pair, float] | None] = []
        state = {"best": math.inf, "min": None, "deepest": None}

        def note(k: int, r: PairResult) -> None:
            """Fold one non-"ok" result into worst / the deepest interference."""
            key = frozenset((r.a, r.b))
            if key not in worst or _worse(r, worst[key][1]):
                worst[key] = (k, r)
            if r.status == "interference":
                depth = r.depth if r.depth is not None and math.isfinite(r.depth) else 0.0
                if state["deepest"] is None or depth > state["deepest"][2]:
                    state["deepest"] = (r.a, r.b, depth, k)

        guard = _Guard(self, result, pairs, per_frame, lowers, state, note, stats)
        total = len(result.poses)
        for k, pose in enumerate(result.poses):
            if not pose.ok:
                per_frame.append([])
                lowers.append(None)
                stats["frames_skipped"] += 1
            else:
                results, closest, lower = self._check(pose.transforms, pairs, state["best"], stats,
                                                      floor=guard.carried(k))
                if closest is not None:
                    state["best"] = closest.distance
                    state["min"] = (closest.a, closest.b, closest.distance, k)
                issues = [r for r in results if r.status != "ok"]
                per_frame.append(issues)
                lowers.append(lower)
                for r in issues:
                    note(k, r)
            if k > 0:
                guard.interval(k - 1)
            if progress is not None:
                progress("clearance", k + 1, total)
        stats["frames"] = total
        stats["seconds"] = time.perf_counter() - t0
        min_clearance = state["min"]
        if state["deepest"] is not None:
            a, b, depth, k = state["deepest"]
            min_clearance = (a, b, -depth if depth > 0 else 0.0, k)
        counts = {key: n for key, n in stats.items() if isinstance(key, str)}  # drop per-pair budgets
        counts["sub_poses"] = counts.get("subframe_poses", 0)
        unresolved = sorted(((key[1], key[2], n) for key, n in stats.items()
                             if isinstance(key, tuple) and key[0] == "unresolved"), key=lambda r: -r[2])
        if unresolved:
            counts["unresolved_pairs"] = [list(r) for r in unresolved]
        counts["pairs"] = len(pairs)
        counts["slowest"] = [[a, b, round(s, 3)] for (a, b), s in self._pair_time.most_common(_SLOWEST)]
        return SweepResult(per_frame, worst, min_clearance,
                           {"pairs_checked": 0, "cache_hits": 0, "exact": 0, "booleans": 0, "proximity": 0,
                            "subframe_poses": 0, **counts})

    # ------------------------------------------------------------------------ one pose

    def _batch(self, pairs: list[_Pair], transforms: dict) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """(part transforms, M = inv(T_a)·T_b per pair, a lower bound on each pair's gap — the
        largest of the world-AABB gap, the separating-axis gap along the six home-box face normals,
        the pair's ``_static_gap`` and its ``_slide_gap`` — and an upper bound on any point pair's
        distance from the world AABBs)."""
        eye = np.eye(4)
        T = {n: np.asarray(transforms.get(n, eye), dtype=float) for n in self.asm.parts}
        idx = self._index.get(id(pairs))
        if idx is None or idx[0] is not pairs:
            keys = list(self._bodies)
            pos = {k: i for i, k in enumerate(keys)}
            ia = np.array([pos[p.body_a] for p in pairs], dtype=np.int64)
            ib = np.array([pos[p.body_b] for p in pairs], dtype=np.int64)
            lo = np.array([self._bodies[k].box[0] for k in keys]).reshape(-1, 3)
            hi = np.array([self._bodies[k].box[1] for k in keys]).reshape(-1, 3)
            static = np.array([self._static_gap(p) for p in pairs], dtype=float).reshape(-1)
            lines = self._line_arrays(pairs)
            slide = np.array([self._slide_gap(p) for p in pairs], dtype=float).reshape(-1)
            slides = self._line_arrays(pairs, self._slide_line)
            idx = (pairs, keys, ia, ib, (lo + hi) / 2, (hi - lo) / 2, (static, *lines), (slide, *slides))
            self._index[id(pairs)] = idx
        _, keys, ia, ib, c, h, (static, lp, lu), (slide, sp, su) = idx
        if not pairs:
            return T, np.zeros((0, 4, 4)), np.zeros(0), np.zeros(0)
        TB = np.array([T[self._bodies[k].part] for k in keys])
        R, t = TB[:, :3, :3], TB[:, :3, 3]
        wc = np.einsum("nij,nj->ni", R, c) + t  # world AABBs of the bodies
        wh = np.einsum("nij,nj->ni", np.abs(R), h)
        aabb = np.linalg.norm(np.maximum(0.0, np.maximum((wc[ib] - wh[ib]) - (wc[ia] + wh[ia]),
                                                         (wc[ia] - wh[ia]) - (wc[ib] + wh[ib]))), axis=1)
        span = np.linalg.norm(np.maximum((wc[ib] + wh[ib]) - (wc[ia] - wh[ia]), (wc[ia] + wh[ia]) - (wc[ib] - wh[ib])),
                              axis=1)
        M = _rel(TB[ia], TB[ib])
        Rm = M[:, :3, :3]
        d = np.einsum("nij,nj->ni", Rm, c[ib]) + M[:, :3, 3] - c[ia]  # b's box centre in a's frame
        absR = np.abs(Rm)
        sep_a = np.abs(d) - h[ia] - np.einsum("nij,nj->ni", absR, h[ib])  # along a's box axes
        sep_b = np.abs(np.einsum("nji,nj->ni", Rm, d)) - np.einsum("nji,nj->ni", absR, h[ia]) - h[ib]  # b's
        obb = np.maximum(0.0, np.maximum(sep_a.max(axis=1), sep_b.max(axis=1)))
        static = np.where(_on_line(M, lp, lu), static, 0.0)  # only while the pair turns about its line
        slide = np.where(_on_line(M, sp, su, slide=True), slide, 0.0)  # … or slides along / turns about this one
        return T, M, np.maximum(np.maximum(aabb, obb), np.maximum(static, slide)), span

    def _line_arrays(self, pairs: list[_Pair], which=None) -> tuple[np.ndarray, np.ndarray]:
        """(point, axis) arrays of the pairs' hinge lines — or, with ``which=self._slide_line``,
        slide lines (NaN where a pair has none)."""
        which = self._hinge_line if which is None else which
        lp, lu = np.full((len(pairs), 3), np.nan), np.full((len(pairs), 3), np.nan)
        for i, p in enumerate(pairs):
            line = None if p.rigid else which(p.a, p.b)
            if line is not None:
                lp[i], lu[i] = line
        return lp, lu

    def _check(self, transforms: dict[str, np.ndarray], pairs: list[_Pair], best: float, stats: Counter, *,
               floor: dict | None = None) -> tuple[list[PairResult], PairResult | None, dict[_Pair, float]]:
        """Results of one pose, its closest non-excused pair if that beats the gap ``best``, and a
        lower bound on the gap of every pair (exact for the non-excused pairs measured; for an
        excused pair what its contact test proved, 0 when touching).

        Status pass: each pair is resolved down to the distance that decides its status —
        touching for excused pairs, ``clearance`` for the rest. Minimum-clearance pass: the
        non-excused pairs left unmeasured keep a lower bound on their gap (box gap, or the τ
        they were shown to exceed); they are re-tested at a radius that doubles until nothing
        unmeasured can beat the closest pair found (here or in earlier frames, ``best``).
        ``floor`` (pair -> gap) adds lower bounds known from elsewhere: a sweep carries each
        frame's bounds to the next, less how far the parts can have moved in between.
        """
        T, Ms, gaps, spans = self._batch(pairs, transforms)
        if floor:  # lower bounds known from elsewhere (the previous frame's, carried over)
            gaps = np.maximum(gaps, [floor.get(p, 0.0) for p in pairs])
        results: list[PairResult] = []
        closest: PairResult | None = None
        excused = np.array([p.excused for p in pairs], dtype=bool)
        lower = gaps.copy()  # per pair: a lower bound on its gap (exact once measured)

        def record(i: int, m: _Measure) -> None:
            nonlocal best, closest
            pair = pairs[i]
            r = self._result(pair, m, T[pair.a], T[pair.b])
            results.append(r)
            if not pair.excused:
                lower[i] = 0.0 if r.distance is None else r.distance
                if r.distance < best:
                    best, closest = r.distance, r

        for i in np.nonzero(excused & (gaps <= TOUCH))[0]:  # joined/allowed/meshing: contact only
            m, lb = self._lookup(pairs[i], Ms[i], TOUCH, stats)
            if m is not None:
                record(i, m)
                lower[i] = 0.0 if m.distance is None else m.distance
            else:
                lower[i] = max(lower[i], lb)
        # a non-excused pair matters below `clearance` (status) and below `best` (minimum)
        status_tau = max(self.clearance, TOUCH)
        reach = max(status_tau, best) if math.isfinite(best) else status_tau
        measured = np.zeros(len(pairs), dtype=bool)
        for i in np.nonzero(~excused & (gaps <= reach))[0]:
            if gaps[i] > status_tau and not _beats(float(gaps[i]), best):
                continue
            tau = max(status_tau, best if math.isfinite(best) else 0.0)
            m, lb = self._lookup(pairs[i], Ms[i], tau, stats)
            if m is None:
                lower[i] = max(lb, gaps[i])
            else:
                record(i, m)
                measured[i] = True
        # [lower bound on the gap, pair index] of unmeasured non-excused pairs that might be closest
        pending = [[float(lower[i]), int(i)] for i in np.nonzero(~excused & ~measured)[0]
                   if _beats(float(lower[i]), best)]
        radius = status_tau
        while pending:
            pending = [p for p in pending if _beats(p[0], best)]
            if not pending:
                break
            # with a closest pair in hand, one test at its gap settles the rest; until then widen
            radius = best if math.isfinite(best) else max(2.0 * radius, 2.0 * min(p[0] for p in pending))
            for p in sorted(pending, key=lambda item: item[0]):
                tau = min(best, radius)
                if p[0] >= tau:
                    continue
                i = p[1]
                if tau >= spans[i]:  # every point pair is within tau: a proximity test can't prune
                    tau = math.inf
                m, lb = self._lookup(pairs[i], Ms[i], tau, stats)
                if m is None:
                    p[0] = max(p[0], lb)
                    lower[i] = max(lower[i], lb)
                else:
                    record(i, m)
                    p[0] = math.inf
        return results, closest, {p: float(lower[i]) for i, p in enumerate(pairs)}

    def _lookup(self, pair: _Pair, M: np.ndarray, tau: float, stats: Counter, *,
                exact: bool = True) -> tuple[_Measure | None, float]:
        """(exact measure of ``pair`` at relative pose M, its gap), or (None, a lower bound ≥ tau
        on the gap) when the parts are provably farther apart than tau. ``exact=False`` (a
        non-excused pair): no exact query — the bound may then fall short of tau."""
        t0 = time.perf_counter()
        ka, kb = pair.body_a, pair.body_b
        Mc, L = self._canonical(ka, kb, M)
        key = (ka, kb, pair.excused, pair.meshing, pair.allowed and pair.max_depth is None, _key(Mc))
        stats["pairs_checked"] += 1
        hit = self._cache.get(key)
        if isinstance(hit, _Measure) or (hit is not None and hit >= tau):
            stats["cache_hits"] += 1
            if isinstance(hit, _Measure):
                return _map_measure(hit, L), hit.distance or 0.0
            return None, hit
        out = self._narrow(pair, Mc, tau, stats, exact)
        if isinstance(out, _Measure):
            self._cache[key] = out
            found = _map_measure(out, L), out.distance or 0.0
        else:
            bound = max(out, hit) if hit is not None else out
            self._cache[key] = bound
            found = None, bound
        self._pair_time[(pair.a, pair.b)] += time.perf_counter() - t0
        return found

    def _result(self, pair: _Pair, m: _Measure, Ta: np.ndarray, Tb: np.ndarray, at: float | None = None) -> PairResult:
        mid = (m.pa + m.pb) / 2  # a's home frame
        mid_b = transform_points(_rigid_inv(Tb) @ Ta, mid)  # the same point in b's home frame
        return PairResult(
            pair.a, pair.b, None if pair.meshing else m.distance, m.volume,
            transform_points(Ta, m.pa), transform_points(Ta, m.pb),
            self._status(pair, m), pair.joined, pair.allowed, pair.meshing,
            extent=None if m.extent is None else m.extent.copy(),
            location=mid,
            depth=m.depth if m.volume > 0 else None,
            location_b=mid_b,
            at=at,
        )

    def _status(self, pair: _Pair, m: _Measure) -> Status:
        if m.volume > 0:
            if pair.allowed:  # a fit / belt may overlap as deep as its max_depth, never deeper
                deep = pair.max_depth is not None and m.depth > pair.max_depth
                return "interference" if deep else "contact"
            if m.volume > self.interference_tol or m.depth > DEPTH_TOL:
                return "interference"
        if m.touching:
            return "contact" if pair.excused else "tight"
        if not pair.excused and m.distance is not None and _tight(m.distance, self.clearance):
            return "tight"
        return "ok"

    # ------------------------------------------------------------------------ bodies

    def _hinge_line(self, a: str, b: str):
        """(point, unit axis) of a line the relative motion of parts a and b is always a rotation
        about — a revolute joint or hinge pin between their rigid bodies, or one from each to a
        common third body along the same line — else None."""
        groups = self.asm.rigid_groups()
        if self._lines is None:
            self._lines = {}
            links: dict[frozenset, list] = {}
            for j in self.asm.joints.values():
                if j.kind == "revolute" and j.parent in groups and j.child in groups:
                    links.setdefault(frozenset((groups[j.parent], groups[j.child])), []).append((j.origin, j.axis))
            for p in self.asm.pins:
                if p.axis is not None and p.a in groups and p.b in groups:
                    links.setdefault(frozenset((groups[p.a], groups[p.b])), []).append((p.point, p.axis))
            self._links = {k: [(np.asarray(o, dtype=float), np.asarray(u, dtype=float) / np.linalg.norm(u))
                               for o, u in v] for k, v in links.items() if len(k) == 2}
        ga, gb = groups.get(a), groups.get(b)
        if ga is None or gb is None or ga == gb:
            return None
        key = frozenset((ga, gb))
        if key not in self._lines:
            line = None
            direct = self._links.get(key)
            if direct:
                line = direct[0]
            else:
                scale = 1.0 + max(float(np.linalg.norm(bx)) for body in self._bodies.values() for bx in body.box)
                for k, lines_a in self._links.items():
                    if ga not in k or line is not None:
                        continue
                    (g,) = k - {ga}
                    for la in lines_a:
                        for lb in self._links.get(frozenset((gb, g)), ()):
                            if _same_line(la, lb, scale) or _same_line(la, (lb[0], -lb[1]), scale):
                                line = la
                                break
                        if line is not None:
                            break
            self._lines[key] = line
        return self._lines[key]

    def _static_gap(self, pair: _Pair) -> float:
        """A lower bound on the pair's gap at EVERY pose (0 when unknown): for parts that only ever
        turn about a common hinge line, the distance between their (height, radius) profiles about
        that line (``tess.revolution_gap``) less the mesh margins. Resolved to just above touching
        for excused pairs (only contact matters) and to a few clearances for the rest.

        The profiles are the parts' *boundaries*: a part buried in the other's material (a pin in
        a plate that was never bored, a gear in a housing never hollowed out) has a profile apart
        from the outer part's, yet overlaps it. With the boundaries apart at every turn about the
        line, containment cannot change along it, so one test at the home relative pose (identity,
        on the line) settles it for every pose: a buried solid makes the bound 0."""
        if pair in self._static:
            return self._static[pair]
        bound = 0.0
        line = None if pair.rigid else self._hinge_line(pair.a, pair.b)
        A, B = (self._mesh(pair.body_a), self._mesh(pair.body_b)) if line is not None else (None, None)
        if A is not None and B is not None and A.shells and B.shells:
            margin = A.margin + B.margin
            cap = (4 * margin if pair.excused else max(2.0, 4 * self.clearance)) + margin
            gap = tess.revolution_gap(np.concatenate(A.shells), np.concatenate(B.shells), line[0], line[1], cap)
            bound = max(0.0, gap - margin) if gap < cap else cap - margin
            # (each probe is a mesh node ≥ gap from the other mesh: clear of its error band when
            # gap > 2 × its margin, so the winding number decides; else OCC's classifier)
            if bound > 0.0 and self._buried(pair, np.eye(4), Counter(), far=gap > 2 * max(A.margin, B.margin)):
                bound = 0.0
        self._static[pair] = bound
        return bound

    def _slide_line(self, a: str, b: str):
        """(point, unit axis) of a prismatic joint between the rigid bodies of parts a and b (their
        relative motion slides along it), else None."""
        groups = self.asm.rigid_groups()
        if self._slides is None:
            self._slides = {}
            for j in self.asm.joints.values():
                if (j.kind == "prismatic" and j.parent in groups and j.child in groups
                        and groups[j.parent] != groups[j.child]):
                    u = np.asarray(j.axis, dtype=float)
                    self._slides.setdefault(frozenset((groups[j.parent], groups[j.child])),
                                            (np.asarray(j.origin, dtype=float), u / np.linalg.norm(u)))
        ga, gb = groups.get(a), groups.get(b)
        if ga is None or gb is None or ga == gb:
            return None
        return self._slides.get(frozenset((ga, gb)))

    def _slide_gap(self, pair: _Pair) -> float:
        """A lower bound on the pair's gap at every pose that slides along (or turns about) its
        slide line (0 when there is none): the gap between the two parts' radius bands about the
        line (``tess.radial_band``) less the mesh margins — a piston can never reach the cooling
        fins around its barrel, however far it strokes."""
        if pair in self._slide_static:
            return self._slide_static[pair]
        bound = 0.0
        line = None if pair.rigid else self._slide_line(pair.a, pair.b)
        A, B = (self._mesh(pair.body_a), self._mesh(pair.body_b)) if line is not None else (None, None)
        if A is not None and B is not None and A.face_tris and B.face_tris:
            a_lo, a_hi = tess.radial_band(np.concatenate(A.face_tris), *line)
            b_lo, b_hi = tess.radial_band(np.concatenate(B.face_tris), *line)
            bound = max(0.0, max(a_lo - b_hi, b_lo - a_hi) - (A.margin + B.margin))
        self._slide_static[pair] = bound
        return bound

    def _mesh(self, key: str):
        """The body's cached tessellation (``tess.Mesh``), or None when some face has no
        triangles (a proximity test would not see it: such a body always gets exact queries)."""
        body = self._bodies[key]
        if body.mesh is None:
            mesh = None if body.shape is None else \
                tess.build_mesh(body.shape, _MESH_DEFLECTION, _MESH_ANGLE, margin_floor=_MESH_MARGIN)
            body.mesh = mesh if mesh is not None else False
        return body.mesh or None

    def _surfaces(self, key: str) -> list:
        """The body's faces' analytic surfaces (``tess.face_surfaces``, per mesh face id), cached."""
        body = self._bodies[key]
        if body.surfaces is None:
            mesh = self._mesh(key)
            body.surfaces = tess.face_surfaces(mesh) if mesh is not None else []
        return body.surfaces

    def _solids(self, key: str) -> list:
        body = self._bodies[key]
        if body.solids is None:
            mesh = self._mesh(key)
            if mesh is not None:
                body.solids = list(mesh.solids)
            else:
                shape = body.shape
                body.solids = [] if shape is None else [TopoDS.Solid(s) for s in _explore(shape, TopAbs_SOLID)]
        return body.solids

    def _axis(self, key: str):
        """The body's axis of rotational symmetry (``tess.revolution_axis``), or None."""
        body = self._bodies[key]
        if body.axis is False:
            lo, hi = body.box
            body.axis = tess.revolution_axis(body.shape, scale=float(np.linalg.norm(hi - lo))) \
                if body.shape is not None and self._solids(key) else None
        return body.axis

    def _solid_axis(self, key: str, ids: tuple[int, ...]):
        """Common revolution axis of the body's solids ``ids`` (() = the whole body), or None."""
        if not ids:
            return self._axis(key)
        body = self._bodies[key]
        if ids not in body.solid_axes:
            solids = self._solids(key)
            lo, hi = body.box
            scale = float(np.linalg.norm(hi - lo))
            axis = None
            for n, i in enumerate(ids):
                ax = tess.revolution_axis(solids[i], scale=scale)
                if ax is None or (n and not _same_line(axis, ax, scale)):
                    axis = None
                    break
                axis = axis if n else ax
            body.solid_axes[ids] = axis
        return body.solid_axes[ids]

    def _support(self, key: str) -> tuple[np.ndarray, float]:
        """(points whose convex hull contains the body's mesh, how far the body may reach beyond it)."""
        mesh = self._mesh(key)
        if mesh is not None:
            return mesh.support, mesh.margin
        return _corners(self._bodies[key].box), 0.0

    def _canonical(self, ka: str, kb: str, M: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        ax_a, ax_b = self._axis(ka), self._axis(kb)
        if ax_a is None and ax_b is None:
            return M, None
        return tess.canonical_pose(M, ax_a, ax_b, self._support(kb)[0])

    def _probes(self, key: str) -> np.ndarray:
        """One boundary point of each solid of the body (home world), shape (n, 3)."""
        body = self._bodies[key]
        if body.probes is None:
            mesh = self._mesh(key)
            if mesh is not None and all(len(shell) for shell in mesh.shells):
                pts = [shell[0, 0] for shell in mesh.shells]  # a mesh node: on the true surface
            else:
                pts = []
                for s in self._solids(key):
                    vs = list(_explore(s, TopAbs_VERTEX))
                    if vs:
                        pts.append(_pnt(BRep_Tool.Pnt_s(TopoDS.Vertex(vs[0]))))
            body.probes = np.array(pts, dtype=float).reshape(-1, 3)
        return body.probes

    def _inside(self, key: str, p: np.ndarray, far: bool, ids=None) -> bool:
        """Is the body-home point ``p`` inside (or on) one of the body's solids (``ids``: only
        those)? ``far``: p is clear of the mesh's error band, so the mesh's winding number decides."""
        mesh = self._mesh(key)
        if far and mesh is not None:
            for i, shell in enumerate(mesh.shells):
                if (ids is not None and i not in ids) or not len(shell):
                    continue
                lo, hi = mesh.shell_box[i]
                if np.all(p >= lo) and np.all(p <= hi) and tess.winding(shell, p) > 0.5:
                    return True
            return False
        body = self._bodies[key]
        if body.classifiers is None:
            body.classifiers = []
            for solid in self._solids(key):
                c = BRepClass3d_SolidClassifier()
                c.Load(solid)
                body.classifiers.append(c)
        pnt = gp_Pnt(*(float(x) for x in p))
        for i, c in enumerate(body.classifiers):
            if ids is not None and i not in ids:
                continue
            c.Perform(pnt, TOUCH)
            if c.State() == TopAbs_IN or c.IsOnAFace():
                return True
        return False

    # ------------------------------------------------------------------------ narrowphase (a's home frame)

    def _narrow(self, pair: _Pair, M: np.ndarray, tau: float, stats: Counter,
                exact: bool = True) -> _Measure | float:
        """Exact measure of the pair at M, or a lower bound ≥ tau on its gap (pipeline in the
        module docstring); with ``exact`` False, no exact query: 0 when proximity can't prove tau.
        Excused pairs are asked at tau = TOUCH: only contact matters."""
        out = self._narrow_any(pair, M, tau, stats, exact)
        if pair.meshing and not isinstance(out, _Measure):  # a gear pair in reach is always listed
            center = self._box_overlap_center(pair, M)
            return _Measure(None, 0.0, 0.0, center, center.copy(), None, False)
        return out

    def _narrow_any(self, pair: _Pair, M: np.ndarray, tau: float, stats: Counter,
                    exact: bool = True) -> _Measure | float:
        A, B = self._mesh(pair.body_a), self._mesh(pair.body_b)
        near = None
        margin = (A.margin if A is not None else 0.0) + (B.margin if B is not None else 0.0)
        if math.isfinite(tau) and A is not None and B is not None:
            stats["proximity"] += 1
            if pair.meshing:
                # interference-only (spec §4.6): teeth in mesh sit a backlash apart, inside the
                # meshes' error band, so only intersecting tessellations get the boolean
                near = tess.near_faces(A, B, M, 0.0)
                if near is None:
                    return self._buried_measure(pair, M, stats, far=False) or tau
                return self._touch_or_overlap(pair, M, near, None, None, stats)
            near = tess.near_faces(A, B, M, tau + margin)
            if near is None:  # the meshes are farther apart than tau + margin: the parts than tau
                return self._buried_measure(pair, M, stats, far=True) or tau
        if near is not None:  # the near faces' planes / cylinders / spheres may prove it outright
            gap = tess.surface_gap(self._surfaces(pair.body_a), self._surfaces(pair.body_b), near, M)
            if gap > tau:
                stats["surface_bounds"] += 1
                # each part's probe points are ≥ gap from the other's surface, so clear of that
                # mesh's error band when gap > 2 × its margin: the winding number decides
                far = gap > 2 * max(A.margin, B.margin)
                return self._buried_measure(pair, M, stats, far=far) or tau
        if not exact:
            return 0.0  # (proximity could not prove tau; nothing more was asked for)
        stats["exact"] += 1
        d, pa, pb = self._exact(pair, M, near, stats)
        if d > tau:  # (only with near faces) every other face is farther than tau as well
            return self._buried_measure(pair, M, stats, far=d > 2 * margin) or tau
        if d > TOUCH:
            buried = self._buried_measure(pair, M, stats, far=d > 2 * margin)
            return buried or _Measure(float(d), 0.0, 0.0, pa, pb, None, False)
        return self._touch_or_overlap(pair, M, near, pa, pb, stats)

    def _touch_or_overlap(self, pair: _Pair, M: np.ndarray, near, pa, pb, stats: Counter) -> _Measure:
        """Measure of a pair whose boundaries touch or cross (``near``: the faces that may meet):
        the boolean on the solids whose boundaries meet (all of them when another solid is buried)
        decides between an overlap and a touch."""
        if pa is None:
            pa = self._box_overlap_center(pair, M)
            pb = pa.copy()
        if pair.allowed and pair.max_depth is None:
            return _Measure(0.0, 0.0, 0.0, pa, pb, None, True)
        A, B = self._mesh(pair.body_a), self._mesh(pair.body_b)
        ids_a = ids_b = None
        if near is not None and A is not None and B is not None:
            ids_a = tuple(sorted({int(A.face_solid[f]) for f in near[0]} - {-1}))
            ids_b = tuple(sorted({int(B.face_solid[f]) for f in near[1]} - {-1}))
            rest_a = set(range(len(A.solids))) - set(ids_a)
            rest_b = set(range(len(B.solids))) - set(ids_b)
            # (their faces are farther than the proximity tolerance from the other mesh: clear of
            # its error band, except for meshing pairs, tested at tolerance 0)
            if (rest_a or rest_b) and self._buried(pair, M, stats, far=not pair.meshing, only_a=rest_a,
                                                   only_b=rest_b):
                ids_a = ids_b = None
        common = self._common(pair, M, ids_a, ids_b, stats)
        if common is None:  # boundaries touch, no volume in common
            return _Measure(0.0, 0.0, 0.0, pa, pb, None, True)
        return _overlap_measure(common, 0.0)

    def _exact(self, pair: _Pair, M: np.ndarray, near, stats: Counter) -> tuple[float, np.ndarray, np.ndarray]:
        """Exact boundary distance and closest points of a (home) and b (moved by M), over the
        ``near`` faces only when given."""
        ka, kb = pair.body_a, pair.body_b
        loc = to_location(M).wrapped
        attempts = []
        if near is not None:
            A, B = self._mesh(ka), self._mesh(kb)
            attempts.append((_compound([A.faces[i] for i in near[0]]), _compound([B.faces[i] for i in near[1]])))
        attempts.append((self._bodies[ka].shape, self._bodies[kb].shape))
        for sa, sb in attempts:
            ext = _distance(sa, sb.Moved(loc))
            if ext.IsDone() and ext.NbSolution() > 0:
                return float(ext.Value()), _pnt(ext.PointOnShape1(1)), _pnt(ext.PointOnShape2(1))
            stats["distance_failures"] += 1
        center = self._box_overlap_center(pair, M)  # OCC gave no answer: the boolean settles overlap
        return 0.0, center, center.copy()

    def _buried(self, pair: _Pair, M: np.ndarray, stats: Counter, *, far: bool, only_a=None, only_b=None) -> bool:
        """Is a solid of one body buried in the other? One boundary point per solid (``only_a`` /
        ``only_b``: of just those solids), tested both ways: with the boundaries apart, a solid
        with a point inside the other body lies inside it entirely."""
        ka, kb = pair.body_a, pair.body_b
        for inner, outer, to_outer, only in ((kb, ka, M, only_b), (ka, kb, _rigid_inv(M), only_a)):
            if not self._solids(outer):
                continue
            lo, hi = self._bodies[outer].box
            for i, p in enumerate(transform_points(to_outer, self._probes(inner))):
                if only is not None and i not in only:
                    continue
                if np.all(p >= lo - TOUCH) and np.all(p <= hi + TOUCH):
                    stats["inside_tests"] += 1
                    if self._inside(outer, p, far):
                        return True
        return False

    def _buried_measure(self, pair: _Pair, M: np.ndarray, stats: Counter, *, far: bool) -> _Measure | None:
        """The overlap of a pair whose boundaries are apart, when one body is buried in the other."""
        if not self._buried(pair, M, stats, far=far):
            return None
        center = self._box_overlap_center(pair, M)
        common = None if pair.allowed and pair.max_depth is None else self._common(pair, M, None, None, stats)
        if common is None:
            return _Measure(0.0, 0.0, 0.0, center, center.copy(), None, True)
        return _overlap_measure(common, 0.0)

    def _common(self, pair: _Pair, M: np.ndarray, ids_a, ids_b, stats: Counter):
        """(volume, area, lo, hi) in a's home frame of the material common to a's solids ``ids_a``
        (None: all) and b's ``ids_b`` moved by M; None when they share no volume.

        Cached per solid subsets and canonical pose (the subsets' own rotational symmetry). A
        common that would make the pair interfere is validated first (module docstring).
        """
        ka, kb = pair.body_a, pair.body_b
        sub_a = () if ids_a is None or len(ids_a) == len(self._solids(ka)) else tuple(ids_a)
        sub_b = () if ids_b is None or len(ids_b) == len(self._solids(kb)) else tuple(ids_b)
        ax_a, ax_b = self._solid_axis(ka, sub_a), self._solid_axis(kb, sub_b)
        Mc, L = (M, None) if ax_a is None and ax_b is None else \
            tess.canonical_pose(M, ax_a, ax_b, self._support(kb)[0])
        key = (ka, sub_a, kb, sub_b, _key(Mc))
        entry = self._commons.get(key)
        if entry is None:
            sa = self._bodies[ka].shape if not sub_a else _compound([self._solids(ka)[i] for i in sub_a])
            sb = self._bodies[kb].shape if not sub_b else _compound([self._solids(kb)[i] for i in sub_b])
            sb = sb.Moved(to_location(Mc).wrapped)
            stats["booleans"] += 1
            cells = _common_cells(sa, sb)
            if cells is None:  # OCC failed: try with the arguments swapped
                stats["booleans"] += 1
                cells = _common_cells(sb, sa)
            if cells is None:
                stats["boolean_failures"] += 1
                cells = []
            entry = {"cells": cells, "raw": _aggregate(cells), "validated": False, "sa": sa, "sb": sb,
                     "Mc": Mc, "ids_a": None if not sub_a else set(sub_a), "ids_b": None if not sub_b else set(sub_b)}
            self._commons[key] = entry
        raw = entry["raw"]
        interferes = raw is not None and self._status(pair, _overlap_measure(raw, 0.0)) == "interference"
        if interferes and not entry["validated"]:
            self._validate(pair, entry, stats)
            raw = entry["raw"]
        return _map_common(raw, L)

    def _validate(self, pair: _Pair, entry: dict, stats: Counter) -> None:
        """Drop the boolean cells that hold no point of both parts (module docstring); if any
        went, repeat the boolean with its arguments swapped and keep the larger valid overlap."""
        ka, kb = pair.body_a, pair.body_b
        Minv = _rigid_inv(entry["Mc"])

        def genuine(cell: _Cell) -> bool:
            sampled = False
            for p in _deep_points(cell.shape, cell.lo, cell.hi):
                sampled = True
                if self._inside(ka, p, False, entry["ids_a"]) and \
                        self._inside(kb, transform_points(Minv, p), False, entry["ids_b"]):
                    return True
            return not sampled  # nothing deep enough to test: keep it

        kept = [c for c in entry["cells"] if genuine(c)]
        if len(kept) < len(entry["cells"]):
            stats["spurious_cells"] += len(entry["cells"]) - len(kept)
            stats["booleans"] += 1
            swapped = _common_cells(entry["sb"], entry["sa"]) or []
            alt = [c for c in swapped if genuine(c)]
            if sum(c.volume for c in alt) > sum(c.volume for c in kept):
                kept = alt
        entry["cells"], entry["raw"], entry["validated"] = kept, _aggregate(kept), True

    def _box_overlap_center(self, pair: _Pair, M: np.ndarray) -> np.ndarray:
        """Center of the overlap of a's home AABB and b's moved AABB (a's home frame)."""
        a_lo, a_hi = self._bodies[pair.body_a].box
        b_lo, b_hi = transform_aabb(*self._bodies[pair.body_b].box, M)
        lo, hi = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
        return (lo + np.maximum(lo, hi)) / 2

    # ------------------------------------------------------------------------ check_clearance exemption

    def _exempted(self, pair: _Pair, groups: dict[str, int]) -> list[_Pair]:
        """A ``check_clearance`` pair linked by a joint or pin: a joined twin on the whole parts
        (overlap still interferes) plus the pair held to clearance on copies without the carried
        bore region (module docstring). Unlinked pairs, or when trimming fails, stay as they are."""
        if pair.excused or pair.rigid:
            return [pair]
        zones = self._zones(pair.a, pair.b, groups)
        if not zones:
            return [pair]
        keys = []
        for name in (pair.a, pair.b):
            body = self._bodies[name]
            trimmed = _trim(body.shape, zones)
            if trimmed is False:
                return [pair]
            key = f"{name}\x00{pair.a}\x00{pair.b}"
            self._bodies[key] = _Body(name, trimmed, body.box if trimmed is None else _shape_box(trimmed))
            keys.append(key)
        out = [_Pair(pair.a, pair.b, True, pair.allowed, pair.meshing, pair.rigid, pair.max_depth)]
        if all(self._bodies[k].shape is not None for k in keys):
            out.append(_Pair(pair.a, pair.b, False, pair.allowed, pair.meshing, pair.rigid, pair.max_depth,
                             keys[0], keys[1]))
        return out

    def _zones(self, a: str, b: str, groups: dict[str, int]) -> list[tuple]:
        """(point, axis | None, radius, half length) of the joints and pins linking a's and b's
        rigid bodies that both parts carry: within ``pin_tol`` of the axis line / ball point, or
        one running in the other's bore of any size (``Assembly._inside_bore``: a ø12 pin in a
        ø12.8 link eye, a ball in its socket ring) — the zone then reaches the bore wall."""
        ga, gb = groups[a], groups[b]
        links = [(j.origin, j.axis) for j in self.asm.joints.values()
                 if j.kind != "fixed" and {groups.get(j.parent), groups.get(j.child)} == {ga, gb}]
        links += [(p.point, p.axis) for p in self.asm.pins if {groups.get(p.a), groups.get(p.b)} == {ga, gb}]
        boxes = [self._bodies[n].box for n in (a, b)]
        half = float(np.linalg.norm(np.max([bx[1] for bx in boxes], axis=0) - np.min([bx[0] for bx in boxes], axis=0)))
        half += max(float(np.linalg.norm(np.asarray(p, dtype=float) - (bx[0] + bx[1]) / 2))
                    for p, _ in links for bx in boxes) if links else 0.0
        zones = []
        for point, axis in links:
            point = np.asarray(point, dtype=float)
            axis = None if axis is None else np.asarray(axis, dtype=float)
            dist = [self._axis_gap(n, point, axis, half) for n in (a, b)]
            if max(dist) > self.asm.pin_tol and not self._in_bore(a, b, dist, point, axis):
                continue
            zones.append((point, axis, max(dist) + self.clearance + 2 * _MESH_MARGIN, half))
        return zones

    def _in_bore(self, a: str, b: str, dist: list[float], point: np.ndarray, axis: np.ndarray | None) -> bool:
        """Does the part farther from the link's axis (ball pin: point) surround it with the other
        running in its bore (``Assembly._inside_bore``; for a ball pin, about the socket's own
        axis, ``Assembly._around_point``)?"""
        asm = self.asm
        ring, inner = (a, b) if dist[0] >= dist[1] else (b, a)
        if not math.isfinite(max(dist)):
            return False
        try:
            if axis is None:
                n = asm._around_point(ring, point)
                return n is not None and asm._inside_bore(ring, inner, point, n)
            return bool(len(asm._cover(ring, point, axis)[1])) and asm._inside_bore(ring, inner, point, axis)
        except Exception:  # OCC kernel failure: not provably in the bore
            return False

    def _axis_gap(self, name: str, point: np.ndarray, axis: np.ndarray | None, half: float) -> float:
        """Distance from a part's material to a joint's axis line (half length ``half`` either
        side of ``point``), or to a ball pin's point (0 inside)."""
        shape = self._bodies[name].shape
        if shape is None:
            return math.inf
        if axis is None:
            if self._inside(name, point, False):
                return 0.0
            tool = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
        else:
            p0, p1 = point - half * axis, point + half * axis
            tool = BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, p0)), gp_Pnt(*map(float, p1))).Edge()
        ext = _distance(shape, tool)
        return float(ext.Value()) if ext.IsDone() and ext.NbSolution() > 0 else math.inf


def _trim(shape: TopoDS_Shape | None, zones: list[tuple]) -> TopoDS_Shape | None | bool:
    """``shape`` minus the zones' cylinders (spheres for ball pins): its remaining solids, None
    when nothing remains, False when OCC failed."""
    if shape is None:
        return None
    out = shape
    for point, axis, radius, half in zones:
        if axis is None:
            tool = BRepPrimAPI_MakeSphere(gp_Pnt(*map(float, point)), radius).Shape()
        else:
            p0 = point - half * axis
            tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, axis))),
                                            radius, 2 * half).Shape()
        op = BRepAlgoAPI_Cut(out, tool)
        if not op.IsDone():
            return False
        out = op.Shape()
    solids = list(_explore(out, TopAbs_SOLID))
    return _compound(solids) if solids else None


class _Guard:
    """Between-frame check of the pairs held to clearance, and of the excused pairs apart at both
    ends of an interval for overlap (module docstring)."""

    def __init__(self, checker: ClearanceChecker, result, pairs, per_frame, lowers, state, note, stats):
        self.c = checker
        self.poses = result.poses
        self.drive = getattr(result, "drive", None)
        # pairs held to clearance, and joined / allowed pairs for overlap only (where they are
        # apart, ``settled``); meshing pairs are measured only where their meshes intersect, so a
        # sub-frame pose could not tell how far apart they are
        self.guarded = [p for p in pairs if not p.meshing]
        self.active = (bool(self.guarded) and getattr(result, "study", None) is not None and bool(self.drive)
                       and len(self.poses) >= 2)
        self.per_frame, self.lowers, self.state, self.note, self.stats = per_frame, lowers, state, note, stats
        self._poses: dict[tuple[int, float], object] = {}
        if not self.guarded:
            return
        names = [p.a for p in self.guarded], [p.b for p in self.guarded]
        self.parts = names
        sup = [[checker._support(p.body_a) for p in self.guarded],
               [checker._support(p.body_b) for p in self.guarded]]
        self.sup = [np.concatenate([s for s, _ in side]) for side in sup]
        self.counts = [np.array([len(s) for s, _ in side]) for side in sup]
        self.off = [np.concatenate([[0], np.cumsum(c)]) for c in self.counts]
        self.margin = [np.array([m for _, m in side]) for side in sup]
        # a pair that only ever turns about a common hinge line keeps this gap along any path
        self.static = np.array([checker._static_gap(p) for p in self.guarded])
        self.lines = checker._line_arrays(self.guarded)
        # … and a pair a prismatic joint links keeps its radius-band gap however far it slides
        self.slide = np.array([checker._slide_gap(p) for p in self.guarded])
        self.slides = checker._line_arrays(self.guarded, checker._slide_line)
        self.excused = np.array([p.excused for p in self.guarded], dtype=bool)

    def _M(self, transforms: dict, idx: np.ndarray) -> np.ndarray:
        eye = np.eye(4)
        Ta = np.array([np.asarray(transforms.get(self.parts[0][i], eye), dtype=float) for i in idx]).reshape(-1, 4, 4)
        Tb = np.array([np.asarray(transforms.get(self.parts[1][i], eye), dtype=float) for i in idx]).reshape(-1, 4, 4)
        return _rel(Ta, Tb)

    def _rows(self, side: int, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(support rows, owning position in idx, segment starts) of the guarded pairs ``idx``."""
        counts = self.counts[side][idx]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        owner = np.repeat(np.arange(len(idx)), counts)
        rows = np.arange(int(counts.sum())) - np.repeat(starts, counts) + np.repeat(self.off[side][idx], counts)
        return rows, owner, starts

    def _spread(self, side: int, idx: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Per guarded pair ``idx``: the largest |D·p| over the support points p of one view (side
        1: b's points in a's frame, 0: a's in b's), D an (n, 4, 4) difference of relative poses.
        |D·p| is convex in p, so its largest value over the convex hull sits at a support point."""
        rows, owner, starts = self._rows(side, idx)
        pts = self.sup[side][rows]
        disp = np.linalg.norm(np.einsum("nij,nj->ni", D[owner, :3, :3], pts) + D[owner, :3, 3], axis=1)
        return np.maximum.reduceat(disp, starts)

    def chord(self, idx: np.ndarray, M0: np.ndarray, M1: np.ndarray, bend: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(chord, θ, ΔR) of the guarded pairs ``idx`` from relative pose M0 to M1: the largest
        straight-line displacement of either part's material in the other's frame (its support
        points', plus the mesh margin turned by θ; the smaller of the two views), the relative
        rotation angle and matrix. Any gap changes by at most the chord between the two poses.

        With ``bend`` (``(2, n)``, per view, from ``bend``) the chord bounds the *path* between
        them: a support point strays at most bend/8 from the straight line between its two end
        positions, so it travels at most chord + bend/4 away from either end — also when the
        motion reverses between the two poses and the chord alone is near 0 (a slider at its dead
        centre midway between two frames)."""
        R0, R1 = M0[:, :3, :3], M1[:, :3, :3]
        dR = R1 @ np.swapaxes(R0, 1, 2)
        theta = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
        chords = []
        eye = np.broadcast_to(np.eye(4), M1.shape)
        for view, (side, D) in enumerate(((1, M1 - M0), (0, _rel(M1, eye) - _rel(M0, eye)))):
            c = self._spread(side, idx, D) + 2 * np.sin(theta / 2) * self.margin[side][idx]
            if bend is not None:
                c = c + bend[view] / 4
            chords.append(c)
        return np.minimum(chords[0], chords[1]), theta, dR

    def bend(self, idx: np.ndarray, Ma: np.ndarray, Mb: np.ndarray, Mc: np.ndarray,
             wa: float = 1.0, wc: float = 1.0) -> np.ndarray:
        """``(2, n)``: per view, the largest second difference wa·Ma − (wa + wc)·Mb + wc·Mc of a
        support point over three relative poses (plus the mesh margin's share) — the curvature
        of its path, at the spacing of the interval it is applied to (``wa = wc = 1``: equal
        steps; else the weights of the three-point formula for unequal ones)."""
        eye = np.broadcast_to(np.eye(4), Mb.shape)
        out = []
        for side, (A, B, C) in ((1, (Ma, Mb, Mc)), (0, (_rel(Ma, eye), _rel(Mb, eye), _rel(Mc, eye)))):
            D = wa * A - (wa + wc) * B + wc * C
            out.append(self._spread(side, idx, D) + np.linalg.norm(D[:, :3, :3], axis=(1, 2)) * self.margin[side][idx])
        return np.stack(out)  # view 0: b's points (side 1), view 1: a's (side 0) — as in chord

    def _bend_at(self, k: int, idx: np.ndarray, M0: np.ndarray, M1: np.ndarray) -> np.ndarray:
        """Path curvature (``bend``) of the guarded pairs over frame interval [k, k+1]: from the
        neighbouring closed frames where the drive carries on in the same direction (the guard's
        path model interpolates the drivers linearly, so a drive that turns at a frame bends the
        path there, not inside the interval), the larger of the two estimates; with neither, from
        the pose solved halfway (0 when that pose does not close)."""
        drive = np.array([np.asarray(v, dtype=float) for v in self.drive.values()]).reshape(len(self.drive), -1)
        n = len(self.poses)
        step = drive[:, k + 1] - drive[:, k]
        h = float(np.linalg.norm(step))
        if h == 0.0:  # the drivers stand still: so does every part
            return np.zeros((2, len(idx)))
        found = []
        for a in (k - 1, k):  # the triples (k-1, k, k+1) and (k, k+1, k+2)
            b, c = a + 1, a + 2
            if a < 0 or c >= n or not all(self.poses[x].ok for x in (a, b, c)):
                continue
            s1, s2 = drive[:, b] - drive[:, a], drive[:, c] - drive[:, b]
            h1, h2 = float(np.linalg.norm(s1)), float(np.linalg.norm(s2))
            if h1 == 0.0 or h2 == 0.0 or float(s1 @ s2) < 0.999 * h1 * h2 or not 0.25 <= h1 / h2 <= 4.0:
                continue  # the drive turns (or dwells) at the middle frame: no estimate across it
            # p'' ≈ 2[(p_c − p_b)/h2 − (p_b − p_a)/h1]/(h1 + h2); bend = p''·h² at this interval's step h
            w = 2.0 * h * h / (h1 + h2)
            Ms = [M0 if x == k else M1 if x == k + 1 else self._M(self.poses[x].transforms, idx) for x in (a, b, c)]
            found.append(self.bend(idx, *Ms, wa=w / h1, wc=w / h2))
        if found:
            return np.maximum.reduce(found)
        pm = self._pose(k, 0.5)
        if pm is None:
            return np.zeros((2, len(idx)))
        return 4.0 * self.bend(idx, M0, self._M(pm.transforms, idx), M1)

    def carried(self, k: int) -> dict | None:
        """Lower bounds on the guarded pairs' gaps at frame k carried over from frame k − 1: its
        bounds less the chord between the two frames (a gap can't shrink by more); None when
        frame k − 1 was not checked."""
        if not self.guarded or k == 0 or self.lowers[k - 1] is None or not self.poses[k].ok:
            return None
        idx = np.arange(len(self.guarded))
        M0, M1 = self._M(self.poses[k - 1].transforms, idx), self._M(self.poses[k].transforms, idx)
        chord, _, _ = self.chord(idx, M0, M1)
        prev = np.array([self.lowers[k - 1].get(p, 0.0) for p in self.guarded])
        return dict(zip(self.guarded, (prev - chord).tolist()))

    def settled(self, idx: np.ndarray, M0: np.ndarray, M1: np.ndarray, g0: np.ndarray,
                g1: np.ndarray, bend: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(settled, motion bound δ, need) for the guarded pairs ``idx`` between two poses where their
        gaps are ≥ g0, g1: can no collision — nor, when neither end is tight, a gap below
        ``clearance`` — hide between them?

        Three tests: the gap can't drop by more than δ (a point of the path is ≥ g0 − d1 and
        ≥ g1 − d2 with d1 + d2 ≤ δ); or the parts are separated along the relative rotation axis
        at both poses — rotation never moves points along its own axis, so a disc spinning over
        a plate stays clear however fast it turns; or the pair only ever turns about a common
        hinge line and its profiles about that line are farther apart than ``need`` (a gap held
        at every pose, ``_static_gap``) — or slides along a prismatic joint's line with radius
        bands about it farther apart than ``need`` (``_slide_gap``; every pose the kinematics
        solves in between keeps the joint). δ bounds how far either part's support points move in
        the other's frame along the path (``chord`` with the path's ``bend``: the chord between
        the two poses alone says nothing about a motion that reverses between them), stretched
        to the arc of the relative rotation; inf beyond a quarter turn.
        """
        clearance = self.c.clearance
        need = np.where(_tight(np.minimum(g0, g1), clearance), 0.0, clearance)
        # an excused pair (joined / allowed) only ever interferes: guarded for overlap where both
        # ends are proven at least `clearance` apart (a joined stop the slider meets at its dead
        # centre between two frames); touching or near at an end (a pin in its bore), its contact
        # is the joint's business and the frames' check covers it
        excused = self.excused[idx]
        need = np.where(excused, 0.0, need)
        chord, theta, dR = self.chord(idx, M0, M1, bend)
        with np.errstate(divide="ignore", invalid="ignore"):
            arc = np.where(theta < 1e-9, 1.0, (theta / 2) / np.sin(np.maximum(theta, 1e-12) / 2))
        delta = _SUB_SAFETY * arc * chord
        delta = np.where(theta > _SUB_MAX_TURN, np.inf, delta)
        lp, lu = self.lines[0][idx], self.lines[1][idx]
        turning = _on_line(M0, lp, lu) & _on_line(M1, lp, lu)
        sp, su = self.slides[0][idx], self.slides[1][idx]
        sliding = _on_line(M0, sp, su, slide=True) & _on_line(M1, sp, su, slide=True)
        ok = ((delta == 0.0) | ((g0 + g1 - delta) / 2 >= need) | (turning & (self.static[idx] >= need + TOUCH))
              | (sliding & (self.slide[idx] >= need + TOUCH)) | (excused & _tight(np.minimum(g0, g1), clearance)))
        # the boxes of support points separated along the relative rotation axis at both poses
        e = np.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]], axis=1)
        n = np.linalg.norm(e, axis=1)
        axial = (~ok) & (n >= 1e-9) & (theta <= _SUB_MAX_TURN)
        if axial.any():
            eu = e / np.where(n > 0, n, 1.0)[:, None]
            rows_a, own_a, st_a = self._rows(0, idx)
            pa = np.einsum("ni,ni->n", self.sup[0][rows_a], eu[own_a])
            a_lo, a_hi = np.minimum.reduceat(pa, st_a), np.maximum.reduceat(pa, st_a)
            rows_b, own_b, st_b = self._rows(1, idx)
            slack = need + TOUCH + self.margin[0][idx] + self.margin[1][idx]
            for M in (M0, M1):
                pb = np.einsum("ni,ni->n", np.einsum("nij,nj->ni", M[own_b, :3, :3], self.sup[1][rows_b])
                               + M[own_b, :3, 3], eu[own_b])
                b_lo, b_hi = np.minimum.reduceat(pb, st_b), np.maximum.reduceat(pb, st_b)
                axial &= np.maximum(b_lo - a_hi, a_lo - b_hi) >= slack
            ok |= axial
        return ok, delta, need

    def interval(self, k: int) -> None:
        """Sub-frame check of interval [k, k+1] for every guarded pair."""
        if not self.active or self.lowers[k] is None or self.lowers[k + 1] is None:
            return
        flagged = {frozenset((r.a, r.b)) for r in self.per_frame[k] + self.per_frame[k + 1]
                   if r.status == "interference"}
        idx = np.arange(len(self.guarded))
        T0, T1 = self.poses[k].transforms, self.poses[k + 1].transforms
        M0, M1 = self._M(T0, idx), self._M(T1, idx)
        g0 = np.array([self.lowers[k].get(p, 0.0) for p in self.guarded])
        g1 = np.array([self.lowers[k + 1].get(p, 0.0) for p in self.guarded])
        bend = self._bend_at(k, idx, M0, M1)
        ok, delta, need = self.settled(idx, M0, M1, g0, g1, bend)
        for i in np.nonzero(~ok)[0]:
            pair = self.guarded[i]
            if frozenset((pair.a, pair.b)) in flagged:
                continue  # already reported at a frame
            a, b = float(g0[i]), float(g1[i])
            if math.isfinite(delta[i]):  # prove the frame gaps large enough for the motion first
                want = float(need[i] + delta[i] / 2)
                if a < want:
                    a = max(a, self._gap(pair, M0[i], want))
                if b < want:
                    b = max(b, self._gap(pair, M1[i], want))
                if self.settled(idx[i:i + 1], M0[i:i + 1], M1[i:i + 1], np.array([a]), np.array([b]),
                                bend[:, i:i + 1])[0][0]:
                    continue
                if self._turning_clear(pair, M0[i], float(delta[i]), float(need[i])):
                    continue
            self._split(pair, i, k, 0.0, a, 1.0, b, _SUB_DEPTH, bend[:, i:i + 1])

    def _turning_clear(self, pair: _Pair, M0: np.ndarray, delta: float, need: float) -> bool:
        """For a pair that only ever turns about a common hinge line: can it come within ``need``
        while moving by at most ``delta`` from relative pose M0? Only faces within need + δ of
        each other at M0 can (the rest are too far for the motion); turning about the line keeps
        every point's height along it and radius from it, so those faces' (height, radius)
        profiles farther apart than need settle the interval for any angle of turn."""
        c = self.c
        line = c._hinge_line(pair.a, pair.b)
        A, B = c._mesh(pair.body_a), c._mesh(pair.body_b)
        if line is None or A is None or B is None or not _on_line(M0[None], line[0][None], line[1][None])[0]:
            return False
        margin = A.margin + B.margin
        self.stats["proximity"] += 1
        near = tess.near_faces(A, B, M0, need + delta + 2 * margin)
        if near is None:
            return True
        ta = [A.face_tris[f] for f in near[0]]
        tb = [B.face_tris[f] for f in near[1]]
        if not ta or not tb:
            return False
        cap = need + 4 * margin + TOUCH
        gap = tess.revolution_gap(np.concatenate(ta), np.concatenate(tb), line[0], line[1], cap)
        return gap - margin >= need + TOUCH

    def _gap(self, pair: _Pair, M: np.ndarray, want: float) -> float:
        """A lower bound on the pair's gap at relative pose M, aiming at ``want``: what a proximity
        query proves (no exact query — bisecting is cheaper than chasing a near miss)."""
        m, lb = self.c._lookup(pair, M, want, self.stats, exact=False)
        return lb if m is None else (m.distance or 0.0)

    def _pose(self, k: int, s: float):
        """Pose a fraction s of the way from frame k to k+1 (drivers interpolated), or None."""
        if s == 0.0:
            return self.poses[k]
        if s == 1.0:
            return self.poses[k + 1]
        key = (k, s)
        if key not in self._poses:
            if self.c._kin is None:
                from .kinematics import Kinematics  # deferred: kinematics is heavier to import
                self.c._kin = Kinematics(self.c.asm)
            target = {n: float(v[k] + s * (v[k + 1] - v[k])) for n, v in self.drive.items()}
            prev = self.poses[k - 1].q if k > 0 and self.poses[k - 1].ok else None
            self.stats["subframe_poses"] += 1
            pose = self.c._kin.solve(target, self.poses[k].q, prev=prev)
            self._poses[key] = pose if pose.ok else None
        return self._poses[key]

    def _unresolved(self, pair: _Pair) -> None:
        """Count a (sub-)interval the guard gave up on without proving it clear."""
        self.stats["sub_unresolved"] += 1
        self.stats[("unresolved", pair.a, pair.b)] += 1

    def _split(self, pair: _Pair, i: int, k: int, s0: float, g0: float, s1: float, g1: float,
               depth: int, bend: np.ndarray) -> bool:
        """Bisect [s0, s1] of frame interval k for ``pair`` while a collision can hide inside.
        Returns True once an interference was found (the search stops there). ``bend`` (``(2, 1)``)
        is the path's curvature over [s0, s1] at its own length: each half gets the larger of a
        quarter of it and the second difference of the half's three poses.

        A pair already tight at both ends (reported) is only followed while it closes in: a
        midpoint no closer than both ends means a persistent close pass (sliding teeth, a guide
        at a small gap), not a hidden collision. A sweep measures at most ``_SUB_BUDGET``
        sub-frame poses per pair; beyond that the interval counts as unresolved.
        """
        stats, c = self.stats, self.c
        p0, p1 = self._pose(k, s0), self._pose(k, s1)
        if p0 is None or p1 is None:
            self._unresolved(pair)  # (a pose in between the solver could not close)
            return False
        idx = np.array([i])
        M0, M1 = self._M(p0.transforms, idx), self._M(p1.transforms, idx)
        ok, delta, need = self.settled(idx, M0, M1, np.array([g0]), np.array([g1]), bend)
        if ok[0]:
            return False
        if math.isfinite(delta[0]) and self._turning_clear(pair, M0[0], float(delta[0]), float(need[0])):
            return False
        if depth == 0 or stats[("sub_pair", pair.a, pair.b)] >= _SUB_BUDGET:
            self._unresolved(pair)
            return False
        sm = (s0 + s1) / 2
        pm = self._pose(k, sm)
        if pm is None:
            self._unresolved(pair)
            return False
        Ta, Tb = pm.transforms[pair.a], pm.transforms[pair.b]
        half = np.maximum(bend / 4, self.bend(idx, M0, self._M(pm.transforms, idx), M1))
        # a gap beyond this settles both halves at once (given g0, g1 ≥ 0)
        tau = (float(delta[0]) / 2 if math.isfinite(delta[0]) else math.inf) + c.clearance
        m, gm = c._lookup(pair, _rigid_inv(Ta) @ Tb, tau, stats)
        if m is not None:
            stats["sub_measured"] += 1
            stats[("sub_pair", pair.a, pair.b)] += 1
            near = k if sm < 0.5 else k + 1
            r = c._result(pair, m, Ta, Tb, at=k + sm)
            # reported at the nearest frame: pa/pb ride on part a's placement there
            T_near = self._pose(near, 0.0).transforms[pair.a]
            r.pa, r.pb = transform_points(T_near, m.pa), transform_points(T_near, m.pb)
            gm = 0.0 if r.distance is None else r.distance
            new = r.status == "interference" or (not pair.excused and r.status != "ok" and gm < min(g0, g1))
            if new:  # a collision, or a pass closer than either frame shows
                self.per_frame[near].append(r)
                self.note(near, r)
                if r.status == "interference":
                    return True
            if not pair.excused and r.distance is not None and r.distance < self.state["best"]:
                self.state["best"] = r.distance
                self.state["min"] = (r.a, r.b, r.distance, near)
            if _tight(max(g0, g1), c.clearance) and gm >= min(g0, g1) - TOUCH:
                return False  # tight at both ends and not closing in
        return (self._split(pair, i, k, s0, g0, sm, gm, depth - 1, half)
                or self._split(pair, i, k, sm, gm, s1, g1, depth - 1, half))


def _on_line(M: np.ndarray, lp: np.ndarray, lu: np.ndarray, tol: float = 1e-6, *,
             slide: bool = False) -> np.ndarray:
    """Is each relative pose M (n, 4, 4) a rotation about its line (point lp, unit axis lu; NaN:
    no line → False) — the line's point and direction kept, to within tol (loop residuals)? With
    ``slide``, a shift along the line is allowed too (the line maps onto itself)."""
    with np.errstate(invalid="ignore"):
        moved = np.einsum("nij,nj->ni", M[:, :3, :3], lp) + M[:, :3, 3] - lp
        if slide:  # only the part of the shift across the line matters
            moved = moved - np.einsum("ni,ni->n", moved, lu)[:, None] * lu
        turned = np.einsum("nij,nj->ni", M[:, :3, :3], lu) - lu
        scale = 1.0 + np.linalg.norm(lp, axis=1)
        ok = (np.linalg.norm(moved, axis=1) <= tol * scale) & (np.linalg.norm(turned, axis=1) <= tol)
    return np.where(np.isnan(lp[:, 0]), False, ok)


def _tight(d, clearance: float):
    """Is the gap ``d`` (a float or an array) below ``clearance`` beyond float noise?"""
    return d < clearance - CLEARANCE_RTOL * max(1.0, clearance)


def _beats(bound: float, best: float) -> bool:
    """Could a pair whose gap is ≥ ``bound`` be closer than ``best``? (ties within 1e-9 can't)"""
    return bound < best - 1e-9 * max(1.0, best) if math.isfinite(best) else True


def _worse(r: PairResult, than: PairResult) -> bool:
    """Is ``r`` a worse finding than ``than`` for the same pair?"""
    if _RANK[r.status] != _RANK[than.status]:
        return _RANK[r.status] > _RANK[than.status]
    if r.status == "interference":
        return r.volume > than.volume
    if r.status == "tight":
        return r.distance is not None and than.distance is not None and r.distance < than.distance
    return False
