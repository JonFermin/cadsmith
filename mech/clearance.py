"""Clearance and interference between parts (spec §4.6).

Every unordered pair of parts that isn't ``ignore``d is classified per pose:

* **interference** — the parts overlap by more than ``interference_tol`` mm³, or the overlap's
  mean depth 2V/A (A = surface area of the common solid) exceeds 0.02 mm; an ``allow_contact``
  pair interferes only when that mean depth exceeds its ``max_depth`` (default 0.1 mm);
* **contact** — touching or overlapping within those tolerances, for joined / allowed / meshing
  pairs (other pairs: **tight**);
* **tight** — 0 < gap < ``clearance`` for pairs that are not joined, allowed or meshing;
* **ok** — everything else.

Parts are measured as solid material: the solids of a part are fused when they overlap (a
list-of-shapes part, see ``massprops.solid_body``) and containment is classified solid by solid,
so multi-solid parts (library motors, bearings) behave like one body.

Pipeline per pair and pose: transformed home AABBs (broadphase) → a conservative
``BRepExtrema_ShapeProximity`` test on cached tessellations, which proves "farther than τ"
cheaply (τ = the distance that decides the pair's status) → for pairs that may touch (joined /
allowed) only the boolean common: an overlap is measured, anything else is touching (no distance
query) → for the rest, exact ``BRepExtrema_DistShapeShape`` → containment by per-solid
classification when the boundaries are apart (OCC's distance only sees boundaries) → boolean
common only for touching / overlapping pairs. Meshing gear/rack pairs are interference-only: proximity at zero tolerance, then the
boolean on candidates. OCC's distance and booleans run multi-threaded.

Every pair is measured with part a at home and part b moved by M = inv(T_a)·T_b, and cached by
M rounded to 1e-6: a pair that moves rigidly together (or repeats a relative pose) is measured
once. Closest points and overlap locations are stored in a's home frame and mapped through T_a.

A sweep also guards the gaps *between* frames: for every pair held to ``clearance`` it bounds how
far the two parts move relative to each other from one closed frame to the next; when that could
close the gap (a thin blade passing a post between two samples), the interval is bisected with
intermediate poses solved by the kinematics until the pair is shown clear or the collision is
found (reported at the nearest frame, ``PairResult.at`` = the fractional frame).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ShapeProximity
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Extrema import Extrema_ExtFlag_MIN
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from .assembly import ALLOW_DEPTH, Assembly
from .geom import to_location, transform_aabb, transform_points, vec3
from .massprops import solid_body

__all__ = ["PairResult", "SweepResult", "ClearanceChecker", "TOUCH", "DEPTH_TOL", "ALLOW_DEPTH"]

TOUCH = 1e-6  # mm: a gap at or below this is touching
DEPTH_TOL = 0.02  # mm: an overlap whose mean depth 2V/A exceeds this interferes whatever its volume
_VOLUME_EPS = 1e-9  # mm³: a boolean common below this is a touch, not an overlap
_KEY_DECIMALS = 6  # relative-pose cache key rounding (1e-6 in R, 1e-6 mm in t)
_MESH_DEFLECTION = 0.01  # mm: chordal deflection of the cached tessellations
_MESH_ANGLE = 0.2  # rad: angular deflection of the same
# The tessellations sit within the deflection of the true surfaces, so meshes farther apart than
# τ + 2·deflection prove the parts are farther apart than τ; twice that again for BRepMesh slack.
_PROXIMITY_MARGIN = 4 * _MESH_DEFLECTION
_RANK = {"interference": 3, "tight": 2, "contact": 1, "ok": 0}
# Between-frame guard: bisection depth (1/32 of a frame), the relative rotation above which the
# chord-based motion bound is not trusted (the interval is bisected regardless), and a safety
# factor on that bound for paths that are not exact screw motions.
_SUB_DEPTH = 5
_SUB_MAX_TURN = math.pi / 2
_SUB_SAFETY = 1.25
_SUB_BUDGET = 200  # sub-frame measurements per pair and sweep

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
    stats: dict = field(default_factory=dict)  # pairs_checked, cache_hits, seconds, ...


@dataclass(frozen=True)
class _Pair:
    a: str
    b: str
    joined: bool
    allowed: bool
    meshing: bool
    rigid: bool  # same rigid group: the relative pose never changes
    max_depth: float | None = ALLOW_DEPTH  # allowed pairs: deepest mean overlap still "contact"

    @property
    def excused(self) -> bool:
        """Closeness is expected (joined, allowed or meshing): only touching/overlap matters."""
        return self.joined or self.allowed or self.meshing


@dataclass(frozen=True)
class _Measure:
    """One pair at one relative pose, in a's home frame (a cache entry)."""

    distance: float | None
    volume: float
    depth: float  # mean overlap depth 2V/A, mm
    pa: np.ndarray
    pb: np.ndarray
    extent: np.ndarray | None
    touching: bool  # boundaries touch or overlap (meshing: the tessellations do)


def _rigid_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _box_gap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Euclidean distance between two AABBs (0 when they overlap): a lower bound on the part gap."""
    d = np.maximum(0.0, np.maximum(b[0] - a[1], a[0] - b[1]))
    return float(np.linalg.norm(d))


def _corners(box: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = box
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _pnt(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def _distance(sa: TopoDS_Shape, sb: TopoDS_Shape) -> BRepExtrema_DistShapeShape:
    ext = BRepExtrema_DistShapeShape()
    ext.LoadS1(sa)
    ext.LoadS2(sb)
    ext.SetFlag(Extrema_ExtFlag_MIN)
    ext.SetMultiThread(True)
    ext.Perform()
    return ext


def _common(sa: TopoDS_Shape, sb: TopoDS_Shape) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """(volume mm³, area mm², bbox lo, bbox hi) of the boolean common; None if empty or failed."""
    op = BRepAlgoAPI_Common(sa, sb)
    op.SetRunParallel(True)
    if not op.IsDone():
        return None
    shape = op.Shape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = abs(props.Mass())
    if volume <= _VOLUME_EPS:
        return None
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return volume, props.Mass(), _pnt(box.CornerMin()), _pnt(box.CornerMax())


def _overlap_measure(common: tuple[float, float, np.ndarray, np.ndarray], distance: float | None) -> _Measure:
    volume, area, lo, hi = common
    center = (lo + hi) / 2
    depth = 2.0 * volume / area if area > 0 else math.inf
    return _Measure(distance, volume, depth, center, center.copy(), hi - lo, True)


def _turn(R: np.ndarray) -> float:
    """Rotation angle (rad) of a 3x3 rotation matrix."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))


class ClearanceChecker:
    """Classifies part pairs per pose; ``sweep`` runs it over a study with a relative-pose cache."""

    def __init__(self, asm: Assembly, clearance: float | None = None):
        self.asm = asm
        self.clearance = asm.clearance if clearance is None else float(clearance)
        self.interference_tol = asm.interference_tol
        names = list(asm.parts)
        groups = asm.rigid_groups()
        meshing = asm.meshing_pairs()
        self._pairs: list[_Pair] = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                key = frozenset((a, b))
                if key in asm.ignored:
                    continue
                self._pairs.append(_Pair(a, b, asm.is_joined(a, b), key in asm.allowed, key in meshing,
                                         groups[a] == groups[b], asm.allow_depth.get(key, ALLOW_DEPTH)))
        # solid material per part: overlapping solids fused, faces/shells dropped
        self._shapes = {}
        for n, p in asm.parts.items():
            body = solid_body(p.shape) if hasattr(p.shape, "solids") else None
            self._shapes[n] = p.shape if body is None else body
        self._home_box = {n: asm._bbox(p) for n, p in asm.parts.items()}
        # (a, b, rounded M) -> exact measure, or a float lower bound: "farther apart than this"
        self._cache: dict[tuple[str, str, bytes], _Measure | float] = {}
        self._probe_points: dict[str, np.ndarray] = {}
        self._classifiers: dict[str, list[BRepClass3d_SolidClassifier]] = {}
        self._tessellated: dict[str, TopoDS_Shape | None] = {}
        self._kin = None  # Kinematics, built on the first sub-frame check

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
        return results

    def sweep(self, result) -> SweepResult:
        """Check every closed frame of a ``StudyResult``; intra-rigid-group pairs are skipped.

        Frames whose loops failed to close (``pose.ok`` False) are not checked — their part
        placement is a least-squares best fit, and the runner reports them as ``loop_open``.
        Between consecutive closed frames the pairs held to ``clearance`` get the sub-frame check
        (module docstring). ``min_clearance`` is signed: an interference anywhere makes it
        −(mean overlap depth) of the deepest one.
        """
        t0 = time.perf_counter()
        stats: Counter = Counter()
        pairs = [p for p in self._pairs if not p.rigid]
        per_frame: list[list[PairResult]] = []
        worst: dict[frozenset, tuple[int, PairResult]] = {}
        bounds: list[dict[_Pair, float] | None] = []
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

        for k, pose in enumerate(result.poses):
            if not pose.ok:
                per_frame.append([])
                bounds.append(None)
                stats["frames_skipped"] += 1
                continue
            results, closest, lower = self._check(pose.transforms, pairs, state["best"], stats)
            if closest is not None:
                state["best"] = closest.distance
                state["min"] = (closest.a, closest.b, closest.distance, k)
            issues = [r for r in results if r.status != "ok"]
            per_frame.append(issues)
            bounds.append(lower)
            for r in issues:
                note(k, r)
        self._between_frames(result, pairs, per_frame, bounds, state, note, stats)
        stats["frames"] = len(result.poses)
        stats["seconds"] = time.perf_counter() - t0
        min_clearance = state["min"]
        if state["deepest"] is not None:
            a, b, depth, k = state["deepest"]
            min_clearance = (a, b, -depth if depth > 0 else 0.0, k)
        counts = {key: n for key, n in stats.items() if isinstance(key, str)}  # drop per-pair budgets
        return SweepResult(per_frame, worst, min_clearance, {"pairs_checked": 0, "cache_hits": 0, **counts})

    # ------------------------------------------------------------------------ one pose

    def _check(self, transforms: dict[str, np.ndarray], pairs: list[_Pair], best: float,
               stats: Counter) -> tuple[list[PairResult], PairResult | None, dict[_Pair, float]]:
        """Results of one pose, its closest non-excused pair if that beats the gap ``best``, and a
        lower bound on the gap of every non-excused pair (exact for the pairs measured).

        Status pass: each pair is resolved down to the distance that decides its status —
        touching for excused pairs, ``clearance`` for the rest. Minimum-clearance pass: the
        non-excused pairs left unmeasured keep a lower bound on their gap (box gap, or the τ
        they were shown to exceed); they are re-tested at a radius that doubles until nothing
        unmeasured can beat the closest pair found (here or in earlier frames, ``best``).
        """
        eye = np.eye(4)
        T = {n: np.asarray(transforms.get(n, eye), dtype=float) for n in self._shapes}
        boxes = {n: transform_aabb(*self._home_box[n], T[n]) for n in self._shapes}
        results: list[PairResult] = []
        closest: PairResult | None = None
        lower: dict[_Pair, float] = {}

        def record(pair: _Pair, m: _Measure, Ta: np.ndarray, Tb: np.ndarray) -> None:
            nonlocal best, closest
            r = self._result(pair, m, Ta, Tb)
            results.append(r)
            if not pair.excused:
                lower[pair] = 0.0 if r.distance is None else r.distance
                if r.distance < best:
                    best, closest = r.distance, r

        # [lower bound on the gap, pair, T_a, M, upper bound on the gap, T_b] of unmeasured non-excused pairs
        pending: list[list] = []
        status_tau = max(self.clearance, TOUCH)
        for pair in pairs:
            Ta, Tb = T[pair.a], T[pair.b]
            M = _rigid_inv(Ta) @ Tb
            box_a, box_b = boxes[pair.a], boxes[pair.b]
            gap = max(_box_gap(box_a, box_b), self._obb_gap(pair, M))
            # a non-excused pair matters below `clearance` (status) and below `best` (minimum)
            tau = TOUCH if pair.excused else max(status_tau, best if math.isfinite(best) else 0.0)
            want = gap <= TOUCH if pair.excused else (gap <= status_tau or _beats(gap, best))
            m = self._lookup(pair, M, tau, stats) if want else None
            if m is not None:
                record(pair, m, Ta, Tb)
            elif not pair.excused:
                span = float(np.linalg.norm(np.maximum(box_b[1] - box_a[0], box_a[1] - box_b[0])))
                pending.append([max(gap, tau) if gap <= tau else gap, pair, Ta, M, span, Tb])
        unmeasured = list(pending)

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
                if tau >= p[4]:  # every point pair is within tau: a proximity test can't prune
                    tau = math.inf
                m = self._lookup(p[1], p[3], tau, stats)
                if m is None:
                    p[0] = tau
                else:
                    record(p[1], m, p[2], p[5])
                    p[0] = math.inf
        for p in unmeasured:
            lower.setdefault(p[1], p[0])
        return results, closest, lower

    def _lookup(self, pair: _Pair, M: np.ndarray, tau: float, stats: Counter) -> _Measure | None:
        """Exact measure of ``pair`` at relative pose M, or None when it is provably farther than tau."""
        key = (pair.a, pair.b, (np.round(M, _KEY_DECIMALS) + 0.0).tobytes())  # + 0.0 folds -0.0 into 0.0
        stats["pairs_checked"] += 1
        hit = self._cache.get(key)
        if isinstance(hit, _Measure) or (hit is not None and hit >= tau):
            stats["cache_hits"] += 1
            return hit if isinstance(hit, _Measure) else None
        if pair.meshing:
            m = self._measure_meshing(pair, M, stats)
        elif self._apart(pair, M, tau, stats, contained=False if hit is not None else None):
            self._cache[key] = tau
            return None
        else:
            m = self._measure(pair, M, stats)
        self._cache[key] = m
        return m

    def _result(self, pair: _Pair, m: _Measure, Ta: np.ndarray, Tb: np.ndarray, at: float | None = None) -> PairResult:
        mid = (m.pa + m.pb) / 2  # a's home frame
        mid_b = transform_points(_rigid_inv(Tb) @ Ta, mid)  # the same point in b's home frame
        return PairResult(
            pair.a, pair.b, m.distance, m.volume,
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
        if not pair.excused and m.distance is not None and m.distance < self.clearance:
            return "tight"
        return "ok"

    # ------------------------------------------------------------------------ narrowphase (a's home frame)

    def _apart(self, pair: _Pair, M: np.ndarray, tau: float, stats: Counter, *, contained: bool | None) -> bool:
        """Provably farther apart than ``tau``: tessellations beyond tau + margin and no containment.

        ``contained`` is the containment answer when already known for this pose (a pose once
        shown apart can't be contained), None to test it.
        """
        if not math.isfinite(tau) or self._meshes_within(pair, M, tau + _PROXIMITY_MARGIN, stats) is not False:
            return False
        return not (self._contained(pair, M, stats) if contained is None else contained)

    def _meshes_within(self, pair: _Pair, M: np.ndarray, tol: float, stats: Counter) -> bool | None:
        """Do the cached tessellations come within ``tol`` (conservative)? None: no usable tessellation.

        ``BRepExtrema_ShapeProximity`` reports face pairs whose triangles are within ``tol``; its
        ``IsDone()`` only says whether any were found, so the overlap map is the answer.
        """
        mesh_a, mesh_b = self._mesh(pair.a), self._mesh(pair.b)
        if mesh_a is None or mesh_b is None:
            return None
        stats["proximity"] += 1
        prox = BRepExtrema_ShapeProximity(mesh_a, mesh_b.Moved(to_location(M).wrapped), tol)
        prox.Perform()
        return prox.OverlapSubShapes1().Size() > 0

    def _measure(self, pair: _Pair, M: np.ndarray, stats: Counter) -> _Measure:
        """Exact measure of a pair the proximity prefilter could not prove apart.

        Excused pairs (joined/allowed) only care about overlap: the boolean alone decides — an
        overlap is measured, anything else within the prefilter's reach counts as touching
        (``allow_contact(max_depth=None)`` pairs skip even the boolean). Other pairs get the exact
        distance, containment when the boundaries are apart, and the boolean when they touch.
        """
        sa = self._shapes[pair.a].wrapped
        sb = self._shapes[pair.b].moved(to_location(M)).wrapped
        stats["exact"] += 1
        tessellated = self._mesh(pair.a) is not None and self._mesh(pair.b) is not None
        if pair.excused:
            if not (pair.allowed and pair.max_depth is None):
                stats["booleans"] += 1
                common = _common(sa, sb)
                if common is not None:
                    return _overlap_measure(common, 0.0)
            if tessellated:  # within TOUCH + the mesh margin, no volume in common: touching
                center = self._box_overlap_center(pair, M)
                return _Measure(0.0, 0.0, 0.0, center, center.copy(), None, True)
            stats["exact_after_boolean"] += 1  # no prefilter proved the pair close: measure it
        ext = _distance(sa, sb)
        if ext.IsDone() and ext.NbSolution() > 0:
            d = ext.Value()
            pa, pb = _pnt(ext.PointOnShape1(1)), _pnt(ext.PointOnShape2(1))
        else:  # OCC gave no answer: the boolean below settles overlap either way
            stats["distance_failures"] += 1
            d = 0.0
            pa = pb = self._box_overlap_center(pair, M)
        if d > TOUCH and self._contained(pair, M, stats):
            d = 0.0
            pa = pb = self._box_overlap_center(pair, M)
        if d > TOUCH:
            return _Measure(float(d), 0.0, 0.0, pa, pb, None, False)
        if pair.excused:  # the boolean above found no volume (or is skipped)
            return _Measure(0.0, 0.0, 0.0, pa, pb, None, True)
        stats["booleans"] += 1
        common = _common(sa, sb)
        if common is None:  # boundaries touch, no volume in common
            return _Measure(0.0, 0.0, 0.0, pa, pb, None, True)
        return _overlap_measure(common, 0.0)

    def _measure_meshing(self, pair: _Pair, M: np.ndarray, stats: Counter) -> _Measure:
        """Interference-only check for gear/rack pairs: tessellation prefilter, then the boolean."""
        center = self._box_overlap_center(pair, M)
        if self._meshes_within(pair, M, 0.0, stats) is False:
            return _Measure(None, 0.0, 0.0, center, center.copy(), None, False)
        stats["booleans"] += 1
        common = _common(self._shapes[pair.a].wrapped, self._shapes[pair.b].moved(to_location(M)).wrapped)
        if common is None:
            return _Measure(None, 0.0, 0.0, center, center.copy(), None, True)
        return _overlap_measure(common, None)

    def _contained(self, pair: _Pair, M: np.ndarray, stats: Counter) -> bool:
        """Is one part buried inside the other? One boundary vertex per solid, tested both ways.

        With the boundaries apart, a solid whose vertex lies inside the other part lies inside
        it entirely. A vertex outside the other part's AABB can't be inside it, so the
        classifiers (one per solid of the outer part) only run when that box test passes.
        """
        Minv = _rigid_inv(M)
        for inner, outer, to_outer in ((pair.b, pair.a, M), (pair.a, pair.b, Minv)):
            if not self._probes(outer).size:  # no solid to be inside of
                continue
            lo, hi = self._home_box[outer]
            for p in transform_points(to_outer, self._probes(inner)):
                if np.all(p >= lo - TOUCH) and np.all(p <= hi + TOUCH):
                    stats["inside_tests"] += 1
                    if self._inside(outer, p):
                        return True
        return False

    def _inside(self, part: str, p: np.ndarray) -> bool:
        """Is the home-world point ``p`` inside (or on) some solid of ``part``? (per-solid classifiers)"""
        classifiers = self._classifiers.get(part)
        if classifiers is None:
            classifiers = []
            for solid in self._shapes[part].solids():
                c = BRepClass3d_SolidClassifier()
                c.Load(solid.wrapped)
                classifiers.append(c)
            self._classifiers[part] = classifiers
        pnt = gp_Pnt(*(float(x) for x in p))
        for c in classifiers:
            c.Perform(pnt, TOUCH)
            if c.State() == TopAbs_IN or c.IsOnAFace():
                return True
        return False

    def _probes(self, part: str) -> np.ndarray:
        """One boundary vertex of each solid of ``part`` (home world), shape (n, 3)."""
        pts = self._probe_points.get(part)
        if pts is None:
            vs = [s.vertices() for s in self._shapes[part].solids()]
            pts = np.array([vec3(v[0]) for v in vs if v]).reshape(-1, 3)
            self._probe_points[part] = pts
        return pts

    def _mesh(self, part: str) -> TopoDS_Shape | None:
        """Tessellated copy of a part's home shape (the model's own shape is left untouched).

        None when some face has no triangles (no faces at all, or BRepMesh failed on one): a
        proximity test would not see that face, so such a part always gets the exact query.
        """
        if part not in self._tessellated:
            shape = BRepBuilderAPI_Copy(self._shapes[part].wrapped, True, False).Shape()
            BRepMesh_IncrementalMesh(shape, _MESH_DEFLECTION, False, _MESH_ANGLE, True)
            complete, n_faces = True, 0
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            while exp.More() and complete:
                tri = BRep_Tool.Triangulation_s(TopoDS.Face(exp.Current()), TopLoc_Location())
                complete = tri is not None and tri.NbTriangles() > 0
                n_faces += 1
                exp.Next()
            self._tessellated[part] = shape if complete and n_faces else None
        return self._tessellated[part]

    def _obb_gap(self, pair: _Pair, M: np.ndarray) -> float:
        """Lower bound on the gap: separation of a's home box and b's home box moved by M, along
        the six face normals of the two boxes (a separating-axis test; 0 if none separates).
        Tighter than world AABBs for rotated parts — e.g. stacked links, a layer apart."""
        lo_a, hi_a = self._home_box[pair.a]
        lo_b, hi_b = self._home_box[pair.b]
        R = M[:3, :3]
        ca, ha = (lo_a + hi_a) / 2, (hi_a - lo_a) / 2
        cb, hb = R @ ((lo_b + hi_b) / 2) + M[:3, 3], (hi_b - lo_b) / 2
        axes = np.vstack([np.eye(3), R.T])  # a's box axes, then b's box axes (columns of R)
        sep = np.abs(axes @ (cb - ca)) - np.abs(axes) @ ha - np.abs(axes @ R) @ hb
        return max(0.0, float(np.max(sep)))

    def _box_overlap_center(self, pair: _Pair, M: np.ndarray) -> np.ndarray:
        """Center of the overlap of a's home AABB and b's moved AABB (a's home frame)."""
        a_lo, a_hi = self._home_box[pair.a]
        b_lo, b_hi = transform_aabb(*self._home_box[pair.b], M)
        lo, hi = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
        return (lo + np.maximum(lo, hi)) / 2

    # ------------------------------------------------------------------------ between frames

    def _motion_bound(self, pair: _Pair, T0: dict, T1: dict) -> float:
        """Upper bound (mm) on how far any point of one part moves relative to the other between
        two poses: the larger chord of the parts' home-box corners (the smaller of the two views),
        stretched to the arc of the relative rotation; inf beyond a quarter turn."""
        M0 = _rigid_inv(T0[pair.a]) @ T0[pair.b]
        M1 = _rigid_inv(T1[pair.a]) @ T1[pair.b]
        theta = _turn(M1[:3, :3] @ M0[:3, :3].T)
        if theta > _SUB_MAX_TURN:
            return math.inf
        dM, dN = M1 - M0, _rigid_inv(M1) - _rigid_inv(M0)
        chord_b = np.max(np.linalg.norm(_corners(self._home_box[pair.b]) @ dM[:3, :3].T + dM[:3, 3], axis=1))
        chord_a = np.max(np.linalg.norm(_corners(self._home_box[pair.a]) @ dN[:3, :3].T + dN[:3, 3], axis=1))
        arc = 1.0 if theta < 1e-9 else (theta / 2) / math.sin(theta / 2)
        return _SUB_SAFETY * arc * float(min(chord_a, chord_b))

    def _settled(self, pair: _Pair, T0: dict, T1: dict, g0: float, g1: float) -> bool:
        """Can no collision — nor, when neither end is tight, a gap below ``clearance`` — hide
        between two poses where the pair's gaps are ≥ g0, g1?

        Two tests: the gap can't drop by more than the motion bound δ (a point of the path is
        ≥ g0 − d1 and ≥ g1 − d2 with d1 + d2 ≤ δ); or the parts' boxes are separated along the
        relative rotation axis at both poses — rotation never moves points along its own axis,
        so a disc spinning over a plate stays clear however fast it turns.
        """
        need = self.clearance if min(g0, g1) >= self.clearance else 0.0
        delta = self._motion_bound(pair, T0, T1)
        if delta == 0.0 or (g0 + g1 - delta) / 2 >= need:
            return True
        M0 = _rigid_inv(T0[pair.a]) @ T0[pair.b]
        M1 = _rigid_inv(T1[pair.a]) @ T1[pair.b]
        dR = M1[:3, :3] @ M0[:3, :3].T
        e = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        n = float(np.linalg.norm(e))
        if n < 1e-9 or _turn(dR) > _SUB_MAX_TURN:
            return False
        e /= n
        a = _corners(self._home_box[pair.a]) @ e
        corners_b = _corners(self._home_box[pair.b])
        for M in (M0, M1):
            b = (corners_b @ M[:3, :3].T + M[:3, 3]) @ e
            if max(b.min() - a.max(), a.min() - b.max()) < need + TOUCH:
                return False
        return True

    def _between_frames(self, result, pairs, per_frame, bounds, state, note, stats) -> None:
        """Sub-frame check of the non-excused pairs between consecutive closed frames (docstring)."""
        study = getattr(result, "study", None)
        drive = getattr(result, "drive", None)
        poses = result.poses
        guarded = [p for p in pairs if not p.excused]
        if not guarded or study is None or not drive or len(poses) < 2:
            return
        cache: dict[tuple[int, float], object] = {}

        def pose_at(k: int, s: float):
            """Pose a fraction s of the way from frame k to k+1 (drivers interpolated), or None."""
            if s == 0.0:
                return poses[k]
            if s == 1.0:
                return poses[k + 1]
            key = (k, s)
            if key not in cache:
                if self._kin is None:
                    from .kinematics import Kinematics  # deferred: kinematics is heavier to import
                    self._kin = Kinematics(self.asm)
                target = {n: float(v[k] + s * (v[k + 1] - v[k])) for n, v in drive.items()}
                prev = poses[k - 1].q if k > 0 and poses[k - 1].ok else None
                stats["sub_poses"] += 1
                pose = self._kin.solve(target, poses[k].q, prev=prev)
                cache[key] = pose if pose.ok else None
            return cache[key]

        for k in range(len(poses) - 1):
            if bounds[k] is None or bounds[k + 1] is None:
                continue
            flagged = {frozenset((r.a, r.b)) for r in per_frame[k] + per_frame[k + 1]
                       if r.status == "interference"}
            for pair in guarded:
                if frozenset((pair.a, pair.b)) in flagged:
                    continue  # already reported at a frame
                g0, g1 = bounds[k].get(pair, 0.0), bounds[k + 1].get(pair, 0.0)
                self._split(pair, k, 0.0, g0, 1.0, g1, _SUB_DEPTH, pose_at, per_frame, state, note, stats)

    def _split(self, pair, k, s0, g0, s1, g1, depth, pose_at, per_frame, state, note, stats) -> bool:
        """Bisect [s0, s1] of frame interval k for ``pair`` while a collision can hide inside.
        Returns True once an interference was found (the search stops there).

        A pair already tight at both ends (reported) is only followed while it closes in: a
        midpoint no closer than both ends means a persistent close pass (sliding teeth, a guide
        at a small gap), not a hidden collision. A sweep measures at most ``_SUB_BUDGET``
        sub-frame poses per pair; beyond that the interval counts as unresolved.
        """
        p0, p1 = pose_at(k, s0), pose_at(k, s1)
        if p0 is None or p1 is None:
            return False
        if self._settled(pair, p0.transforms, p1.transforms, g0, g1):
            return False
        if depth == 0 or stats[("sub_pair", pair.a, pair.b)] >= _SUB_BUDGET:
            stats["sub_unresolved"] += 1
            return False
        delta = self._motion_bound(pair, p0.transforms, p1.transforms)
        sm = (s0 + s1) / 2
        pm = pose_at(k, sm)
        if pm is None:
            return False
        Ta, Tb = pm.transforms[pair.a], pm.transforms[pair.b]
        # a gap beyond this settles both halves at once (given g0, g1 ≥ 0)
        tau = (delta / 2 if math.isfinite(delta) else math.inf) + self.clearance
        m = self._lookup(pair, _rigid_inv(Ta) @ Tb, tau, stats)
        if m is None:
            gm = tau
        else:
            stats["sub_measured"] += 1
            stats[("sub_pair", pair.a, pair.b)] += 1
            near = k if sm < 0.5 else k + 1
            r = self._result(pair, m, Ta, Tb, at=k + sm)
            # reported at the nearest frame: pa/pb ride on part a's placement there
            T_near = pose_at(near, 0.0).transforms[pair.a]
            r.pa, r.pb = transform_points(T_near, m.pa), transform_points(T_near, m.pb)
            gm = 0.0 if r.distance is None else r.distance
            new = r.status == "interference" or (r.status != "ok" and gm < min(g0, g1))
            if new:  # a collision, or a pass closer than either frame shows
                per_frame[near].append(r)
                note(near, r)
                if r.status == "interference":
                    return True
            if r.distance is not None and r.distance < state["best"]:
                state["best"] = r.distance
                state["min"] = (r.a, r.b, r.distance, near)
            if max(g0, g1) < self.clearance and gm >= min(g0, g1) - TOUCH:
                return False  # tight at both ends and not closing in
        return (self._split(pair, k, s0, g0, sm, gm, depth - 1, pose_at, per_frame, state, note, stats)
                or self._split(pair, k, sm, gm, s1, g1, depth - 1, pose_at, per_frame, state, note, stats))


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
