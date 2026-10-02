"""Shared geometry helpers (spec §4.0).

Every mech module uses these instead of re-implementing transforms, JSON number formatting or
file-name slugs. Transforms are 4x4 homogeneous numpy arrays in millimetres; angles are degrees
at every API boundary. The modeling helpers at the bottom (`link`, `circle_intersect`) are meant
for model scripts: they build parts *in place* in home-world coordinates.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from build123d import Circle, Location, Part, Plane, Polygon, Pos, SlotCenterToCenter, Vector, Vertex, extrude
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import (GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_SurfaceOfRevolution,
                         GeomAbs_Torus)
from OCP.gp import gp_Pnt, gp_Trsf
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

__all__ = [
    "vec3",
    "unit",
    "to_location",
    "from_location",
    "rot_about_line",
    "translation",
    "transform_points",
    "transform_aabb",
    "inside",
    "Mesh",
    "tessellate",
    "axis_cover",
    "near_cover",
    "section_radii",
    "section_reach",
    "surface_axes",
    "to_json16",
    "fnum",
    "slug",
    "link",
    "circle_intersect",
    "polygon",
]

_EPS_LEN = 1e-12  # below this a direction vector is treated as zero
_JSON_ZERO = 1e-9  # to_json16 snaps |entry| below this to 0 (round-off in R and in mm alike)
MESH_DEFLECTION = 0.01  # mm: chordal deflection of `tessellate` meshes (the clearance checker's too)
MESH_ANGLE = 0.2  # rad: angular deflection of the same
_COVER_BINS = 720  # axis_cover: angular resolution (0.5°) of the "material all around" test
_COVER_LEVELS = 96  # axis_cover: most section levels tested per mesh
_COVER_REFINE = 4  # axis_cover: subdivision rounds for triangles only partly within the radius

# Device names Windows refuses as file base names regardless of extension.
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(10)} | {f"lpt{i}" for i in range(10)}
)
_SLUG_MAX = 64


# --------------------------------------------------------------------------------------------
# vectors and transforms
# --------------------------------------------------------------------------------------------


def vec3(v) -> np.ndarray:
    """Tuple / list / ndarray / build123d ``Vector`` or ``Vertex`` -> new float array of shape (3,).

    Raises ``ValueError`` for anything that is not exactly three finite numbers.
    """
    if isinstance(v, (Vector, Vertex)):
        a = np.array([v.X, v.Y, v.Z], dtype=float)
    else:
        try:
            a = np.array(v, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expected a 3D point/vector, got {v!r}") from exc
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"expected a 3D point/vector of finite numbers, got {v!r}")
    return a


def unit(v) -> np.ndarray:
    """Normalized copy of ``v``; ``ValueError`` on a zero-length vector."""
    a = vec3(v)
    n = float(np.linalg.norm(a))
    if n < _EPS_LEN:
        raise ValueError(f"zero-length direction {v!r}")
    return a / n


def to_location(T) -> Location:
    """4x4 transform -> build123d ``Location``.

    The rotation block is re-orthonormalized with an SVD (nearest rotation in the Frobenius
    norm), so slightly drifted or uniformly scaled matrices are accepted; reflections and
    singular matrices raise ``ValueError``. (``Location(ndarray)`` itself raises ``TypeError``.)
    """
    M = np.asarray(T, dtype=float)
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        raise ValueError(f"expected a finite 4x4 transform, got shape {M.shape}")
    U, s, Vt = np.linalg.svd(M[:3, :3])
    if s[-1] <= 1e-9 * max(s[0], _EPS_LEN):
        raise ValueError("transform rotation block is singular")
    R = U @ Vt
    if np.linalg.det(R) <= 0:
        raise ValueError("transform contains a reflection (det < 0); only proper rotations allowed")
    trsf = gp_Trsf()
    trsf.SetValues(*np.hstack([R, M[:3, 3:4]]).ravel().tolist())
    return Location(trsf)


def from_location(loc: Location) -> np.ndarray:
    """build123d ``Location`` -> 4x4 numpy transform."""
    if not isinstance(loc, Location):
        raise TypeError(f"expected a build123d Location, got {type(loc).__name__}")
    trsf = loc.wrapped.Transformation()
    T = np.eye(4)
    for i in range(3):
        for j in range(4):
            T[i, j] = trsf.Value(i + 1, j + 1)
    return T


def rot_about_line(origin, axis, deg: float) -> np.ndarray:
    """4x4 rotation by ``deg`` degrees about the line (``origin``, ``axis``), right-hand rule."""
    n = unit(axis)
    o = vec3(origin)
    th = math.radians(float(deg))
    K = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    # Rodrigues in the I + sin·K + (1 − cos)·K² form keeps an exact 1 on the axis for
    # coordinate axes, so planar mechanisms stay exactly planar.
    R = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = o - R @ o
    return T


def translation(v) -> np.ndarray:
    """4x4 pure translation by ``v`` (mm)."""
    T = np.eye(4)
    T[:3, 3] = vec3(v)
    return T


def transform_points(T, pts) -> np.ndarray:
    """Apply a 4x4 transform to one point (shape (3,)) or many (shape (N, 3))."""
    M = np.asarray(T, dtype=float)
    P = vec3(pts) if isinstance(pts, (Vector, Vertex)) else np.asarray(pts, dtype=float)
    if P.ndim == 1:
        return M[:3, :3] @ P + M[:3, 3]
    return P @ M[:3, :3].T + M[:3, 3]


def transform_aabb(bmin, bmax, T) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box of the transformed box ``[bmin, bmax]`` (exact for the 8 corners)."""
    M = np.asarray(T, dtype=float)
    lo, hi = vec3(bmin), vec3(bmax)
    c = M[:3, :3] @ ((lo + hi) / 2) + M[:3, 3]
    e = np.abs(M[:3, :3]) @ ((hi - lo) / 2)
    return c - e, c + e


def inside(shape, p, tolerance: float = 1e-6) -> bool:
    """Is the point ``p`` inside (or on the boundary of) any solid of ``shape``?

    Each solid is classified on its own: OCC's classifier run on a multi-solid compound (a
    list-of-shapes part, a library motor or bearing) calls many interior points outside.
    Shapes without solids (faces, shells, wires) contain nothing.
    """
    pnt = gp_Pnt(*(float(x) for x in vec3(p)))
    for solid in shape.solids() if hasattr(shape, "solids") else []:
        classifier = BRepClass3d_SolidClassifier(solid.wrapped)
        classifier.Perform(pnt, tolerance)
        if classifier.State() == TopAbs_IN or classifier.IsOnAFace():
            return True
    return False


# --------------------------------------------------------------------------------------------
# tessellations
# --------------------------------------------------------------------------------------------


@dataclass
class Mesh:
    """A shape's tessellation: ``vertices`` (N, 3) mm and ``triangles`` (M, 3) vertex indices, plus
    ``shape`` — the meshed OCC copy (for ``BRepExtrema_ShapeProximity``), or None when some face got
    no triangles (a proximity test would not see that face). Every point of the true surface lies
    within the chordal ``deflection`` of the mesh (planar faces exactly)."""

    vertices: np.ndarray
    triangles: np.ndarray
    shape: object | None
    deflection: float
    faces: int = 0  # faces of the meshed shape (the sub-shape indices of BRepExtrema_ShapeProximity)


def tessellate(shape, deflection: float = MESH_DEFLECTION, angle: float = MESH_ANGLE, *,
               parallel: bool = False) -> Mesh:
    """Tessellate a copy of ``shape`` (the shape itself is left untouched) into numpy arrays.

    ``parallel`` meshes the faces on several threads — slower for parts made of many small planar
    faces (gears), so it is off by default.
    """
    copy = BRepBuilderAPI_Copy(shape.wrapped, True, False).Shape()
    BRepMesh_IncrementalMesh(copy, float(deflection), False, float(angle), bool(parallel))
    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    base, complete, n_faces = 0, True, 0
    exp = TopExp_Explorer(copy, TopAbs_FACE)
    while exp.More():
        n_faces += 1
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(TopoDS.Face(exp.Current()), loc)
        exp.Next()
        if tri is None or tri.NbTriangles() == 0:
            complete = False
            continue
        pts = np.array([(p.X(), p.Y(), p.Z()) for p in (tri.Node(i) for i in range(1, tri.NbNodes() + 1))])
        if not loc.IsIdentity():
            pts = transform_points(_trsf_matrix(loc.Transformation()), pts)
        idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        verts.append(pts)
        tris.append(idx + base)
        base += len(pts)
    V = np.vstack(verts) if verts else np.zeros((0, 3))
    F = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    return Mesh(V, F, copy if complete and n_faces else None, float(deflection), n_faces)


def _trsf_matrix(trsf: gp_Trsf) -> np.ndarray:
    T = np.eye(4)
    for i in range(3):
        for j in range(4):
            T[i, j] = trsf.Value(i + 1, j + 1)
    return T


def axis_cover(mesh: Mesh, origin, axis, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Where along the line (origin, axis) a part's material is at the line — ``(near, around)``.

    Both are sorted, disjoint ``(k, 2)`` arrays of axial intervals ``[t0, t1]`` (mm from ``origin``
    along ``axis``). ``near``: the material comes within ``radius`` of the line (a shaft on it, a
    plate it pierces, a bore of radius ≤ ``radius``). ``around``: the plane section ⟂ the line
    encloses it — every direction from the line meets material, as in a bore of any size the line
    passes through (a bearing ring, a link eye, a housing). Computed on the tessellation: near is
    exact for the mesh up to a sub-``radius``/8 slice of partly-near triangles; around is exact for
    the mesh between consecutive vertex levels (at most 128 levels tested; gaps under 0.5° count
    as closed).
    """
    if mesh.triangles.size == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    T, X, Y = _axis_frame(mesh, origin, axis)
    return _near_intervals(T, X, Y, float(radius)), _around_intervals(T, X, Y)


def near_cover(mesh: Mesh, origin, axis, radius: float) -> np.ndarray:
    """Only the ``near`` intervals of ``axis_cover`` (where the mesh comes within ``radius`` of the line)."""
    if mesh.triangles.size == 0:
        return np.zeros((0, 2))
    return _near_intervals(*_axis_frame(mesh, origin, axis), float(radius))


def section_radii(mesh: Mesh, origin, axis, t: float) -> tuple[float, float] | None:
    """(nearest, farthest) distance from the line (origin, axis) to the boundary of the part's
    plane section ⟂ the line at axial level ``t`` (mm from ``origin``) — for a ring around the
    line, its bore and outer radius; None when the plane misses the part."""
    if mesh.triangles.size == 0:
        return None
    T, X, Y = _axis_frame(mesh, origin, axis)
    seg = _section(T, X, Y, T.min(axis=1), T.max(axis=1), float(t))
    if seg is None:
        return None
    (x0, y0), (x1, y1) = seg
    ex, ey = x1 - x0, y1 - y0
    ee = ex * ex + ey * ey
    s = np.clip(np.where(ee > 0, -(x0 * ex + y0 * ey) / np.where(ee > 0, ee, 1.0), 0.0), 0.0, 1.0)
    nearest = float(np.min(np.hypot(x0 + s * ex, y0 + s * ey)))
    farthest = float(max(np.max(np.hypot(x0, y0)), np.max(np.hypot(x1, y1))))
    return nearest, farthest


def section_reach(mesh: Mesh, origin, axis, t: float, bins: int = _COVER_BINS) -> float | None:
    """How far the part's plane section ⟂ the line (origin, axis) at axial level ``t`` reaches
    from the line in EVERY direction: the smallest, over the directions about the line (``bins``
    sectors), of the farthest section boundary point in that direction — 0 when some direction
    meets no material, None when the plane misses the part. A pin or shaft of radius r reaches r
    all round (a journal filling its bore); an arm sticking out of a hub reaches the hub's radius
    only, whatever its length."""
    if mesh.triangles.size == 0:
        return None
    T, X, Y = _axis_frame(mesh, origin, axis)
    seg = _section(T, X, Y, T.min(axis=1), T.max(axis=1), float(t))
    if seg is None:
        return None
    (x0, y0), (x1, y1) = seg
    width = 2 * math.pi / bins
    best = np.zeros(bins)
    r0, r1 = np.hypot(x0, y0), np.hypot(x1, y1)
    a0, a1 = np.arctan2(y0, x0) % (2 * math.pi), np.arctan2(y1, x1) % (2 * math.pi)
    for a, r in ((a0, r0), (a1, r1)):  # the segments' ends …
        np.maximum.at(best, np.minimum(np.floor(a / width).astype(np.int64), bins - 1), r)
    # … and where each segment crosses a sector's boundary ray (its radius along a segment is
    # convex, so a sector's farthest point is at one of those)
    span = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    start = np.where(span >= 0, a0, a1)
    k0 = np.floor(start / width).astype(np.int64) + 1
    k1 = np.floor((start + np.abs(span)) / width).astype(np.int64)
    counts = np.maximum(k1 - k0 + 1, 0)
    if counts.any():
        idx = np.repeat(np.arange(len(x0)), counts)
        k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts) + k0[idx]
        phi = k * width
        ex, ey = (x1 - x0)[idx], (y1 - y0)[idx]
        den = np.cos(phi) * ey - np.sin(phi) * ex
        num = x0[idx] * ey - y0[idx] * ex
        far = np.maximum(r0, r1)[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(np.abs(den) > 1e-15, num / np.where(np.abs(den) > 1e-15, den, 1.0), far)
        rho = np.clip(rho, 0.0, far)
        kk = k % bins
        np.maximum.at(best, kk, rho)
        np.maximum.at(best, (kk - 1) % bins, rho)
    return float(best.min())


def surface_axes(shape, point, radius: float) -> list[np.ndarray]:
    """Unit directions about which ``shape`` might surround ``point``: the axes of its surfaces of
    revolution (cylinders, cones, tori, revolved faces) whose axis line passes within ``radius``
    of the point, the polar axis of its spheres centred within ``radius`` of it (a spherical
    socket of any size), and the normals of its planar faces within ``radius`` of it — a socket
    ring's axis for a ball sitting in it. Parallel directions are listed once (sign ignored)."""
    p = vec3(point)
    out: list[np.ndarray] = []

    def add(d) -> None:
        v = np.array([d.X(), d.Y(), d.Z()], dtype=float)
        norm = float(np.linalg.norm(v))
        if norm < _EPS_LEN:
            return
        v /= norm
        if not any(abs(float(v @ e)) > 1 - 1e-9 for e in out):
            out.append(v)

    exp = TopExp_Explorer(shape.wrapped, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        exp.Next()
        try:
            s = BRepAdaptor_Surface(face, True)
            kind = s.GetType()
            if kind == GeomAbs_Plane:
                ax = s.Plane().Axis()
            elif kind == GeomAbs_Cylinder:
                ax = s.Cylinder().Axis()
            elif kind == GeomAbs_Cone:
                ax = s.Cone().Axis()
            elif kind == GeomAbs_Torus:
                ax = s.Torus().Axis()
            elif kind == GeomAbs_Sphere:
                sph = s.Sphere()
                c = sph.Location()
                if float(np.linalg.norm(p - np.array([c.X(), c.Y(), c.Z()]))) <= radius:
                    add(sph.Position().Direction())  # every line through the centre is an axis
                continue
            elif kind == GeomAbs_SurfaceOfRevolution:
                ax = s.AxeOfRevolution()
            else:
                continue
            loc, d = ax.Location(), ax.Direction()
            o = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
            n = np.array([d.X(), d.Y(), d.Z()], dtype=float)
            r = p - o
            off = abs(float(r @ n)) if kind == GeomAbs_Plane else float(np.linalg.norm(r - (r @ n) * n))
            if off <= radius:
                add(d)
        except Exception:  # a face OCC can't adapt: no candidate from it
            continue
    return out


def _axis_frame(mesh: Mesh, origin, axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-triangle vertex coordinates (M, 3) in the line's frame: axial t and in-plane x, y."""
    n = unit(axis)
    u = _any_perpendicular(n)
    w = np.cross(n, u)
    R = mesh.vertices - vec3(origin)
    F = mesh.triangles
    return (R @ n)[F], (R @ u)[F], (R @ w)[F]


def _origin_tri_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Distance from (0, 0) to each 2-D triangle (rows of X, Y); 0 inside a non-degenerate one."""
    X1, Y1 = np.roll(X, -1, axis=1), np.roll(Y, -1, axis=1)
    ex, ey = X1 - X, Y1 - Y
    side = ey * X - ex * Y  # sign of the origin w.r.t. each directed edge
    area2 = np.abs(side.sum(axis=1))
    inside = (area2 > 1e-18) & (np.all(side >= 0, axis=1) | np.all(side <= 0, axis=1))
    ee = ex * ex + ey * ey
    s = np.clip(np.where(ee > 0, -(X * ex + Y * ey) / np.where(ee > 0, ee, 1.0), 0.0), 0.0, 1.0)
    d = np.hypot(X + s * ex, Y + s * ey).min(axis=1)
    return np.where(inside, 0.0, d)


def _subdivide(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split each triangle (rows of per-vertex arrays) into its four midpoint children."""
    out = []
    for A in arrays:
        a, b, c = A[:, 0], A[:, 1], A[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        out.append(np.vstack([np.column_stack(v) for v in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]))
    return tuple(out)


def _near_intervals(T: np.ndarray, X: np.ndarray, Y: np.ndarray, r: float) -> np.ndarray:
    found = []
    for round_ in range(_COVER_REFINE + 1):
        hit = _origin_tri_distance(X, Y) <= r
        T, X, Y = T[hit], X[hit], Y[hit]
        if not len(T):
            break
        within = np.hypot(X, Y).max(axis=1) <= r
        short = T.max(axis=1) - T.min(axis=1) <= r / 8
        done = within | short | (round_ == _COVER_REFINE)
        found.append(np.column_stack([T[done].min(axis=1), T[done].max(axis=1)]))
        T, X, Y = T[~done], X[~done], Y[~done]
        if not len(T):
            break
        T, X, Y = _subdivide(T, X, Y)
    return _merge_intervals(np.vstack(found) if found else np.zeros((0, 2)))


def _around_intervals(T: np.ndarray, X: np.ndarray, Y: np.ndarray, budget: int = 128) -> np.ndarray:
    levels = np.unique(T)
    # a section can only surround the line if the part's projection does
    if len(levels) < 2 or not (X.min() < 0 < X.max() and Y.min() < 0 < Y.max()):
        return np.zeros((0, 2))
    if len(levels) - 1 > budget:
        levels = np.linspace(levels[0], levels[-1], budget + 1)
    lo, hi = levels[:-1], levels[1:]
    tmin, tmax = T.min(axis=1), T.max(axis=1)
    keep = [k for k, tau in enumerate((lo + hi) / 2) if _encloses(T, X, Y, tmin, tmax, float(tau))]
    return _merge_intervals(np.column_stack([lo[keep], hi[keep]]))


def _section(T, X, Y, tmin, tmax, tau: float):
    """The mesh's plane section at axial level ``tau`` as segments ((x0, y0), (x1, y1)) — one per
    crossing triangle — or None. Triangles with a vertex exactly on the level are skipped."""
    m = (tmin < tau) & (tmax > tau)
    if not m.any():
        return None
    t, x, y = T[m] - tau, X[m], Y[m]
    t1, x1, y1 = np.roll(t, -1, axis=1), np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = t * t1 < 0  # edges that cross the level
    s = np.where(cross, t / np.where(cross, t - t1, 1.0), 0.0)
    px, py = x + s * (x1 - x), y + s * (y1 - y)
    ok = cross.sum(axis=1) == 2
    if not ok.any():
        return None
    order = np.argsort(~cross[ok], axis=1, kind="stable")[:, :2]
    rows = np.arange(order.shape[0])[:, None]
    ax, ay = px[ok][rows, order], py[ok][rows, order]
    return (ax[:, 0], ay[:, 0]), (ax[:, 1], ay[:, 1])


def _encloses(T, X, Y, tmin, tmax, tau: float) -> bool:
    """Does the mesh section at axial level ``tau`` surround the line (every direction covered)?"""
    m = (tmin < tau) & (tmax > tau)
    if np.count_nonzero(m) < 3:
        return False
    xs, ys = X[m], Y[m]
    if not (xs.min() < 0 < xs.max() and ys.min() < 0 < ys.max()):  # the section can't surround the line
        return False
    seg = _section(T, X, Y, tmin, tmax, tau)
    if seg is None:
        return False
    (x0, y0), (x1, y1) = seg
    a0, a1 = np.arctan2(y0, x0), np.arctan2(y1, x1)
    span = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    start = np.where(span >= 0, a0, a1) % (2 * math.pi)
    start = np.where(start >= 2 * math.pi, 0.0, start)  # (−1e-17) % 2π rounds to exactly 2π
    width = 2 * math.pi / _COVER_BINS
    b0 = np.minimum(np.floor(start / width).astype(np.int64), _COVER_BINS - 1)
    b1 = np.floor((start + np.abs(span)) / width).astype(np.int64)  # may pass 2π: wraps below
    diff = np.zeros(_COVER_BINS + 1, dtype=np.int64)
    flat = b1 < _COVER_BINS
    np.add.at(diff, b0[flat], 1)
    np.add.at(diff, b1[flat] + 1, -1)
    wrap = ~flat
    np.add.at(diff, b0[wrap], 1)
    diff[_COVER_BINS] -= int(np.count_nonzero(wrap))
    diff[0] += int(np.count_nonzero(wrap))
    np.add.at(diff, np.minimum(b1[wrap] - _COVER_BINS + 1, _COVER_BINS), -1)
    return bool(np.all(np.cumsum(diff[:-1]) > 0))


def _merge_intervals(iv: np.ndarray, join: float = 1e-9) -> np.ndarray:
    """Sorted union of (k, 2) intervals; intervals within ``join`` of each other merge."""
    if not len(iv):
        return np.zeros((0, 2))
    iv = iv[np.argsort(iv[:, 0])]
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1] + join:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------------------------
# JSON / file-name formatting
# --------------------------------------------------------------------------------------------


def fnum(x) -> float | None:
    """Round to 6 significant figures for JSON; ``None`` for None / NaN / ±inf."""
    if x is None:
        return None
    f = float(x)
    if not math.isfinite(f):
        return None
    return float(f"{f:.6g}")


def to_json16(T) -> list[float]:
    """4x4 transform -> 16 floats, column-major (three.js ``Matrix4.fromArray`` order).

    Entries are rounded to 6 significant figures; round-off below 1e-9 (e.g. ``cos 90°``) is
    written as 0 so scene files stay compact.
    """
    out = []
    for x in np.asarray(T, dtype=float).T.ravel():
        v = fnum(x)
        out.append(0.0 if v is not None and abs(v) < _JSON_ZERO else v)
    return out


def slug(name, taken: set[str] | None = None) -> str:
    """File-system-safe identifier: ASCII ``[a-z0-9_-]``, never a Windows device name.

    Accents are transliterated (``ü`` -> ``u``), every other run of characters becomes ``_``,
    and the result is at most 64 characters (``"part"`` if nothing survives). With ``taken``,
    the slug is de-duplicated case-insensitively against it (``name``, ``name_2``, …) and then
    added to ``taken`` so a loop of calls yields unique names.
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s).strip("_-")
    s = s[:_SLUG_MAX].rstrip("_-") or "part"
    if s in _WINDOWS_RESERVED:
        s += "_"
    if taken is not None:
        used = {t.lower() for t in taken}
        base, k = s, 2
        while s in used:
            s = f"{base}_{k}"
            k += 1
        taken.add(s)
    return s


# --------------------------------------------------------------------------------------------
# modeling helpers for scripts
# --------------------------------------------------------------------------------------------


def _frame(origin: np.ndarray, x_dir: np.ndarray, z_dir: np.ndarray) -> np.ndarray:
    """4x4 whose columns are the right-handed frame (x, z × x, z) placed at ``origin``."""
    T = np.eye(4)
    T[:3, 0] = x_dir
    T[:3, 1] = np.cross(z_dir, x_dir)
    T[:3, 2] = z_dir
    T[:3, 3] = origin
    return T


def _any_perpendicular(n: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to unit ``n``."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p = np.cross(n, helper)
    return p / np.linalg.norm(p)


def link(p0, p1, width: float, thickness: float, *, z: float | None = None, hole: float | None = None,
         normal=(0, 0, 1)) -> Part:
    """Rounded bar (slot) between two world points, modeled in place.

    The bar lies in the plane perpendicular to ``normal``: its bottom face sits at offset ``z``
    along ``normal`` (default: ``p0``'s offset) and it extends ``thickness`` along ``normal``.
    ``p0`` and ``p1`` are projected onto that plane; the end arcs are centered on them. Holes of
    diameter ``hole`` (default ``0.4·width``; 0 for none) go through both ends. Coincident
    points give a disc.
    """
    width, thickness = float(width), float(thickness)
    if width <= 0 or thickness <= 0:
        raise ValueError(f"link width and thickness must be > 0 (got {width}, {thickness})")
    hole_d = 0.4 * width if hole is None else float(hole)
    if not 0 <= hole_d < width:
        raise ValueError(f"link hole diameter must be in [0, width) (got {hole_d} for width {width})")
    n = unit(normal)
    a, b = vec3(p0), vec3(p1)
    z0 = float(a @ n) if z is None else float(z)
    a_p = a - (a @ n - z0) * n
    b_p = b - (b @ n - z0) * n
    d = b_p - a_p
    length = float(np.linalg.norm(d))

    if length < 1e-9:
        profile = Circle(width / 2)
        hole_centers = [(0.0, 0.0)]
        x_dir = _any_perpendicular(n)
    else:
        profile = Pos(length / 2, 0) * SlotCenterToCenter(length, width)
        hole_centers = [(0.0, 0.0), (length, 0.0)]
        x_dir = d / length
    if hole_d > 0:
        for cx, cy in hole_centers:
            profile = profile - Pos(cx, cy) * Circle(hole_d / 2)
    bar = extrude(profile, amount=thickness)
    return to_location(_frame(a_p, x_dir, n)) * bar


def circle_intersect(c0, r0: float, c1, r1: float, side: int = +1, normal=(0, 0, 1)) -> tuple[float, float, float]:
    """Intersection of two coplanar circles (centers ``c0``, ``c1``; radii ``r0``, ``r1``).

    The circles lie in the plane through ``c0`` perpendicular to ``normal`` (``c1`` is projected
    onto it). ``side`` (+1 / −1) picks the solution p by the sign of ((c1 − c0) × (p − c0))·normal,
    i.e. +1 is to the left of c0→c1 when looking down ``normal``. ``ValueError`` if the circles
    don't meet (tangency within 1e-9 relative is accepted).
    """
    if side not in (1, -1):
        raise ValueError(f"side must be +1 or -1 (got {side!r})")
    r0, r1 = float(r0), float(r1)
    if r0 <= 0 or r1 <= 0:
        raise ValueError(f"circle radii must be > 0 (got {r0}, {r1})")
    n = unit(normal)
    a = vec3(c0)
    dv = vec3(c1) - a
    dv -= (dv @ n) * n
    d = float(np.linalg.norm(dv))
    if d < _EPS_LEN:
        raise ValueError("circle_intersect: concentric circles have no unique intersection")
    e = dv / d
    x = (r0 * r0 - r1 * r1 + d * d) / (2 * d)  # distance from c0 along c0→c1
    h2 = r0 * r0 - x * x
    if h2 < -1e-9 * max(r0, r1) ** 2:
        raise ValueError(
            f"circle_intersect: circles don't meet (center distance {d:.6g}, radii {r0:.6g} and {r1:.6g}; "
            f"need |r0 − r1| ≤ d ≤ r0 + r1)"
        )
    h = math.sqrt(max(h2, 0.0))
    p = a + x * e + side * h * np.cross(n, e)
    return (float(p[0]), float(p[1]), float(p[2]))


def polygon(points, *, plane: Plane = Plane.XY):
    """Closed polygon through 2-D ``points`` (x, y on ``plane``), always counter-clockwise.

    The winding is fixed so the face normal is ``plane.z_dir`` whatever order the points come in:
    a clockwise build123d ``Polygon`` gets a −Z face that extrudes the wrong way and fails to fuse
    with circles. The points are used as given (no re-alignment, unlike ``Polygon``'s default); a
    repeated closing point is dropped. ``ValueError`` for fewer than 3 distinct points, zero area,
    or self-intersecting edges. Returns a build123d sketch (``Polygon``) placed on ``plane``.
    """
    pts: list[tuple[float, float]] = []
    for p in points:
        try:
            q = [float(c) for c in p]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"polygon: expected (x, y) points, got {p!r}") from exc
        if len(q) not in (2, 3) or (len(q) == 3 and q[2] != 0.0) or not all(math.isfinite(c) for c in q):
            raise ValueError(f"polygon: points are finite 2-D (x, y) on the plane, got {p!r}")
        pts.append((q[0], q[1]))
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        raise ValueError(f"polygon: needs at least 3 distinct points (got {len(pts)})")
    P = np.array(pts)
    area2 = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    size = float(np.max(np.ptp(P, axis=0)))
    if abs(area2) <= 1e-12 * max(size, 1e-9) ** 2:
        raise ValueError("polygon: the points enclose no area (collinear or repeated)")
    bad = _self_intersection(P)
    if bad is not None:
        i, j = bad
        raise ValueError(f"polygon: edges {i}–{(i + 1) % len(P)} and {j}–{(j + 1) % len(P)} cross "
                         f"(self-intersecting outline)")
    if area2 < 0:
        pts.reverse()
    return plane * Polygon(*pts, align=None)


def _self_intersection(P: np.ndarray) -> tuple[int, int] | None:
    """First pair of non-adjacent edges of the closed polyline P that cross or touch, else None."""
    n = len(P)
    A, B = P, np.roll(P, -1, axis=0)
    D = B - A
    tol = 1e-12 * max(float(np.max(np.ptp(P, axis=0))), 1e-9) ** 2  # cross products are areas

    def sgn(v: np.ndarray) -> np.ndarray:
        return np.where(np.abs(v) <= tol, 0, np.sign(v))

    for i in range(n - 2):
        j = np.arange(i + 2, n if i > 0 else n - 1)  # skip edge i's neighbours
        if not j.size:
            continue
        e, f = D[i], D[j]
        s1 = sgn(e[0] * (A[j, 1] - A[i, 1]) - e[1] * (A[j, 0] - A[i, 0]))
        s2 = sgn(e[0] * (B[j, 1] - A[i, 1]) - e[1] * (B[j, 0] - A[i, 0]))
        s3 = sgn(f[:, 0] * (A[i, 1] - A[j, 1]) - f[:, 1] * (A[i, 0] - A[j, 0]))
        s4 = sgn(f[:, 0] * (B[i, 1] - A[j, 1]) - f[:, 1] * (B[i, 0] - A[j, 0]))
        hit = (s1 * s2 <= 0) & (s3 * s4 <= 0)
        collinear = (s1 == 0) & (s2 == 0)
        if collinear.any():  # on one line: they meet only if their spans along edge i overlap
            ee = float(e @ e)
            u0 = (A[j] - A[i]) @ e / ee
            u1 = (B[j] - A[i]) @ e / ee
            hit &= ~collinear | (np.maximum(np.minimum(u0, u1), 0.0) <= np.minimum(np.maximum(u0, u1), 1.0))
        if hit.any():
            return i, int(j[np.argmax(hit)])
    return None
