"""Clearance and interference between parts (spec §4.6).

Every unordered pair of parts that isn't ``ignore``d is classified per pose:

* **interference** — the parts overlap by more than ``interference_tol`` mm³, or the overlap's
  mean depth 2V/A (A = surface area of the common solid) exceeds 0.02 mm; an ``allow_contact``
  pair reports **contact** instead;
* **contact** — touching or overlapping within those tolerances, for joined / allowed / meshing
  pairs (other pairs: **tight**);
* **tight** — 0 < gap < ``clearance`` for pairs that are not joined, allowed or meshing;
* **ok** — everything else.

Pipeline per pair and pose: transformed home AABBs (broadphase) → a conservative
``BRepExtrema_ShapeProximity`` test on cached tessellations, which proves "farther than τ"
cheaply (τ = the distance that decides the pair's status) → exact ``BRepExtrema_DistShapeShape``
→ containment by ``is_inside`` when the boundaries are apart (OCC's distance only sees
boundaries) → boolean common only for touching / overlapping pairs. Meshing gear/rack pairs are
interference-only: proximity at zero tolerance, then the boolean on candidates.

Every pair is measured with part a at home and part b moved by M = inv(T_a)·T_b, and cached by
M rounded to 1e-6: a pair that moves rigidly together (or repeats a relative pose) is measured
once. Closest points and overlap locations are stored in a's home frame and mapped through T_a.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from build123d import Vector
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_ShapeProximity
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Extrema import Extrema_ExtFlag_MIN
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from .assembly import Assembly
from .geom import to_location, transform_aabb, transform_points, vec3

__all__ = ["PairResult", "SweepResult", "ClearanceChecker", "TOUCH", "DEPTH_TOL"]

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
    location: np.ndarray | None = None  # overlap / closest-point midpoint in HOME world (via inv(T_a))


@dataclass
class SweepResult:
    per_frame: list[list[PairResult]]  # non-"ok" results per frame (empty for frames whose loops are open)
    worst: dict[frozenset, tuple[int, PairResult]]  # pair -> (frame, worst non-"ok" result)
    min_clearance: tuple[str, str, float, int] | None  # (a, b, gap, frame) over non-joined/allowed/meshing pairs
    stats: dict = field(default_factory=dict)  # pairs_checked, cache_hits, seconds, ...


@dataclass(frozen=True)
class _Pair:
    a: str
    b: str
    joined: bool
    allowed: bool
    meshing: bool
    rigid: bool  # same rigid group: the relative pose never changes

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


def _pnt(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def _common(sa: TopoDS_Shape, sb: TopoDS_Shape) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """(volume mm³, area mm², bbox lo, bbox hi) of the boolean common; None if empty or failed."""
    op = BRepAlgoAPI_Common(sa, sb)
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
                                         groups[a] == groups[b]))
        self._shapes = {n: p.shape for n, p in asm.parts.items()}
        self._home_box = {n: asm._bbox(p) for n, p in asm.parts.items()}
        # (a, b, rounded M) -> exact measure, or a float lower bound: "farther apart than this"
        self._cache: dict[tuple[str, str, bytes], _Measure | float] = {}
        self._probe_points: dict[str, np.ndarray] = {}
        self._tessellated: dict[str, TopoDS_Shape | None] = {}

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
        results, _ = self._check(transforms, pairs, math.inf, Counter())
        return results

    def sweep(self, result) -> SweepResult:
        """Check every closed frame of a ``StudyResult``; intra-rigid-group pairs are skipped.

        Frames whose loops failed to close (``pose.ok`` False) are not checked — their part
        placement is a least-squares best fit, and the runner reports them as ``loop_open``.
        """
        t0 = time.perf_counter()
        stats: Counter = Counter()
        pairs = [p for p in self._pairs if not p.rigid]
        per_frame: list[list[PairResult]] = []
        worst: dict[frozenset, tuple[int, PairResult]] = {}
        best, min_clearance = math.inf, None
        for k, pose in enumerate(result.poses):
            if not pose.ok:
                per_frame.append([])
                stats["frames_skipped"] += 1
                continue
            results, closest = self._check(pose.transforms, pairs, best, stats)
            if closest is not None:
                best = closest.distance
                min_clearance = (closest.a, closest.b, closest.distance, k)
            issues = [r for r in results if r.status != "ok"]
            per_frame.append(issues)
            for r in issues:
                key = frozenset((r.a, r.b))
                if key not in worst or _worse(r, worst[key][1]):
                    worst[key] = (k, r)
        stats["frames"] = len(result.poses)
        stats["seconds"] = time.perf_counter() - t0
        return SweepResult(per_frame, worst, min_clearance, {"pairs_checked": 0, "cache_hits": 0, **stats})

    # ------------------------------------------------------------------------ one pose

    def _check(self, transforms: dict[str, np.ndarray], pairs: list[_Pair], best: float,
               stats: Counter) -> tuple[list[PairResult], PairResult | None]:
        """Results of one pose, and its closest non-excused pair if that beats the gap ``best``.

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

        def record(pair: _Pair, m: _Measure, Ta: np.ndarray) -> None:
            nonlocal best, closest
            r = self._result(pair, m, Ta)
            results.append(r)
            if not pair.excused and r.distance < best:
                best, closest = r.distance, r

        # [lower bound on the gap, pair, T_a, M, upper bound on the gap] of unmeasured non-excused pairs
        pending: list[list] = []
        status_tau = max(self.clearance, TOUCH)
        for pair in pairs:
            Ta = T[pair.a]
            M = _rigid_inv(Ta) @ T[pair.b]
            box_a, box_b = boxes[pair.a], boxes[pair.b]
            gap = _box_gap(box_a, box_b)
            # a non-excused pair matters below `clearance` (status) and below `best` (minimum)
            tau = TOUCH if pair.excused else max(status_tau, best if math.isfinite(best) else 0.0)
            m = self._lookup(pair, M, tau, stats) if gap <= tau else None
            if m is not None:
                record(pair, m, Ta)
            elif not pair.excused:
                span = float(np.linalg.norm(np.maximum(box_b[1] - box_a[0], box_a[1] - box_b[0])))
                pending.append([max(gap, tau) if gap <= tau else gap, pair, Ta, M, span])

        radius = status_tau
        while pending:
            pending = [p for p in pending if p[0] < best]
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
                    record(p[1], m, p[2])
                    p[0] = math.inf
        return results, closest

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

    def _result(self, pair: _Pair, m: _Measure, Ta: np.ndarray) -> PairResult:
        return PairResult(
            pair.a, pair.b, m.distance, m.volume,
            transform_points(Ta, m.pa), transform_points(Ta, m.pb),
            self._status(pair, m), pair.joined, pair.allowed, pair.meshing,
            extent=None if m.extent is None else m.extent.copy(),
            location=(m.pa + m.pb) / 2,
        )

    def _status(self, pair: _Pair, m: _Measure) -> Status:
        if m.volume > 0 and (m.volume > self.interference_tol or m.depth > DEPTH_TOL):
            return "contact" if pair.allowed else "interference"
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
        sa = self._shapes[pair.a].wrapped
        sb = self._shapes[pair.b].moved(to_location(M)).wrapped
        stats["exact"] += 1
        ext = BRepExtrema_DistShapeShape(sa, sb, Extrema_ExtFlag_MIN)
        if ext.IsDone() and ext.NbSolution() > 0:
            d = ext.Value()
            pa, pb = _pnt(ext.PointOnShape1(1)), _pnt(ext.PointOnShape2(1))
        else:  # OCC gave no answer: the boolean below settles overlap either way
            stats["distance_failures"] += 1
            d = 0.0
            pa = pb = self._box_overlap_center(pair, M)
        if d > TOUCH and self._contained(pair, M, stats):
            d = 0.0
        if d > TOUCH:
            return _Measure(float(d), 0.0, 0.0, pa, pb, None, False)
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
        it entirely. A vertex outside the other part's AABB can't be inside it, so the (0.1 ms)
        classifier only runs when that box test passes.
        """
        Minv = _rigid_inv(M)
        for inner, outer, to_outer in ((pair.b, pair.a, M), (pair.a, pair.b, Minv)):
            if not self._probes(outer).size:  # no solid to be inside of
                continue
            lo, hi = self._home_box[outer]
            for p in transform_points(to_outer, self._probes(inner)):
                if np.all(p >= lo - TOUCH) and np.all(p <= hi + TOUCH):
                    stats["inside_tests"] += 1
                    if self._shapes[outer].is_inside(Vector(*p)):
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

    def _box_overlap_center(self, pair: _Pair, M: np.ndarray) -> np.ndarray:
        """Center of the overlap of a's home AABB and b's moved AABB (a's home frame)."""
        a_lo, a_hi = self._home_box[pair.a]
        b_lo, b_hi = transform_aabb(*self._home_box[pair.b], M)
        lo, hi = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
        return (lo + np.maximum(lo, hi)) / 2


def _worse(r: PairResult, than: PairResult) -> bool:
    """Is ``r`` a worse finding than ``than`` for the same pair?"""
    if _RANK[r.status] != _RANK[than.status]:
        return _RANK[r.status] > _RANK[than.status]
    if r.status == "interference":
        return r.volume > than.volume
    if r.status == "tight":
        return r.distance is not None and than.distance is not None and r.distance < than.distance
    return False
