"""Tessellation and symmetry helpers for the clearance checker (spec §4.6).

* ``Mesh`` — a part's solid material meshed once in its home frame (a BRepMesh copy, chordal
  deflection ``deflection``): the copy itself (OCC proximity queries run on it), its faces with
  their triangles and solids, outward-oriented triangle shells per solid (``winding`` numbers: a
  point farther than ``margin`` from the mesh is inside the solid exactly when its winding number
  is 1), a bound ``margin`` on how far the mesh may sit from the true surface, and support points
  whose convex hull contains the part (motion bounds).
* ``near_faces`` — the faces of two meshed parts (one moved) that may come within a tolerance
  (``BRepExtrema_ShapeProximity``, conservative): every other face is provably farther apart, so
  an exact distance query only needs these.
* ``revolution_axis`` — the axis a solid is exactly invariant under rotation about (every face a
  coaxial surface of revolution, every edge a full coaxial circle or a seam), and
  ``canonical_pose``: one representative relative pose per class of poses that differ only by
  such rotations — a shaft turning in its bore is one measurement.
* ``revolution_gap`` — a lower bound on the distance between two triangle sets however far one
  turns about a line: the gap between their (height, radius) profiles about it.
* ``radial_band`` — a set's radius range about a line: disjoint bands bound the gap of two parts
  however far one turns about the line *or slides along it*.
* ``face_surfaces`` / ``surface_gap`` — the plane / cylinder / sphere a face lies on, and a lower
  bound on the distance between two such faces from their infinite surfaces (a ball in a socket
  ring, a pin in a bore): proves a small gap that the proximity test, conservative on long thin
  triangles, cannot — without an exact distance query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepExtrema import BRepExtrema_ShapeProximity
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import (GeomAbs_Circle, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                         GeomAbs_SurfaceOfRevolution, GeomAbs_Torus)
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull

from .geom import to_location

__all__ = ["Mesh", "build_mesh", "near_faces", "winding", "radial_band", "revolution_gap", "revolution_axis",
           "canonical_pose", "rot_about", "face_surfaces", "surface_gap"]

_MAX_SUPPORT = 64  # convex-hull vertices kept as support points (else a 32-vertex prism around them)
_AXIS_TOL = 1e-9  # rad: coaxial direction tolerance of revolution_axis
_AXIS_DIST = 1e-7  # mm (× part size, at least 1): coaxial position tolerance of revolution_axis
_PARALLEL = 1e-9  # rad: two planes / axes count as parallel for surface_gap below this


@dataclass
class Mesh:
    """A part's material meshed in its home frame (see the module docstring)."""

    shape: TopoDS_Shape  # the meshed copy (same geometry as the part)
    faces: list  # TopoDS_Face per face id (faces of ``shape``)
    face_solid: np.ndarray  # solid id per face id (-1: a face of no solid)
    solids: list  # TopoDS_Solid per solid id
    shells: list  # per solid id: (m, 3, 3) outward-oriented triangles
    shell_box: list  # per solid id: (lo, hi) of its shell's vertices
    face_tris: list  # per face id: (m, 3, 3) triangles
    margin: float  # mm: the mesh lies within this of the true surface (and vice versa)
    support: np.ndarray  # (k, 3) points whose convex hull contains the part
    lo: np.ndarray  # mesh AABB
    hi: np.ndarray
    n_explored: int  # faces a TopExp_Explorer visits on ``shape`` (OCC proximity face indices)
    ids: dict = field(repr=False)  # hash(face) -> [(face, id)]

    def face_id(self, f) -> int | None:
        """Id of the face IsSame as ``f`` (a face of ``shape``), or None."""
        for g, i in self.ids.get(hash(f), ()):
            if f.IsSame(g):
                return i
        return None


def _explore(shape, kind):
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        yield exp.Current()
        exp.Next()


def build_mesh(shape: TopoDS_Shape, deflection: float, angle: float, *, margin_floor: float = 0.0) -> Mesh | None:
    """Mesh of ``shape`` (its solids' faces, or its faces when it has no solid); None when some
    face gets no triangles (a proximity test would not see it) or there are no faces.

    The shape itself is left untouched (a copy is meshed). ``margin`` = max(margin_floor,
    3 × the largest deflection BRepMesh reports for a face).
    """
    copy = BRepBuilderAPI_Copy(shape, True, False).Shape()
    BRepMesh_IncrementalMesh(copy, deflection, False, angle, True)
    ids: dict[int, list] = {}  # IsSame dedup: solids of a compound may share a face
    faces, face_solid, solids = [], [], []

    def add(f, si: int) -> None:
        bucket = ids.setdefault(hash(f), [])
        if any(f.IsSame(g) for g, _ in bucket):
            return
        bucket.append((f, len(faces)))
        faces.append(TopoDS.Face(f))
        face_solid.append(si)

    for si, solid in enumerate(_explore(copy, TopAbs_SOLID)):
        solids.append(TopoDS.Solid(solid))
        for f in _explore(solid, TopAbs_FACE):
            add(f, si)
    if not solids:
        for f in _explore(copy, TopAbs_FACE):
            add(f, -1)
    if not faces:
        return None
    shells: list[list] = [[] for _ in solids]
    face_tris, nodes_all = [], []
    deflection_seen = 0.0
    for fi, face in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None or tri.NbTriangles() == 0:
            return None
        deflection_seen = max(deflection_seen, float(tri.Deflection()))
        n = tri.NbNodes()
        if loc.IsIdentity():
            pts = [tri.Node(i) for i in range(1, n + 1)]
        else:
            trsf = loc.Transformation()
            pts = [tri.Node(i).Transformed(trsf) for i in range(1, n + 1)]
        nodes = np.array([(p.X(), p.Y(), p.Z()) for p in pts], dtype=float).reshape(-1, 3)
        nodes_all.append(nodes)
        idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        if face.Orientation() == TopAbs_REVERSED:
            idx = idx[:, [0, 2, 1]]
        face_tris.append(nodes[idx])
        if face_solid[fi] >= 0:
            shells[face_solid[fi]].append(face_tris[-1])
    nodes = np.concatenate(nodes_all)
    shells = [np.concatenate(s) if s else np.zeros((0, 3, 3)) for s in shells]
    empty = (np.full(3, np.inf), np.full(3, -np.inf))
    boxes = [(s.reshape(-1, 3).min(axis=0), s.reshape(-1, 3).max(axis=0)) if len(s) else empty for s in shells]
    return Mesh(shape=copy, faces=faces, face_solid=np.asarray(face_solid, dtype=np.int64), solids=solids,
                shells=shells, shell_box=boxes, face_tris=face_tris,
                margin=max(margin_floor, 3.0 * deflection_seen), support=_support(nodes), lo=nodes.min(axis=0),
                hi=nodes.max(axis=0), n_explored=sum(1 for _ in _explore(copy, TopAbs_FACE)), ids=ids)


def _support(nodes: np.ndarray) -> np.ndarray:
    """Points whose convex hull contains every node: the hull vertices, or (when there are many)
    the 32 vertices of a 16-gon prism around them (along the PCA axis that gives the smallest one)."""
    pts = np.unique(np.round(nodes, 9), axis=0)
    try:
        hull = pts[ConvexHull(pts).vertices] if len(pts) >= 4 else pts
    except Exception:  # flat / degenerate point sets: the prism below contains them
        hull = pts
    if len(hull) <= _MAX_SUPPORT:
        return hull.copy()
    c = hull.mean(axis=0)
    _, _, vt = np.linalg.svd(hull - c, full_matrices=False)
    m = 16
    ang = 2 * math.pi * np.arange(m) / m
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    j1 = (np.arange(m) + 1) % m
    A = np.stack([dirs, dirs[j1]], axis=1)  # vertex j: dirs[j]·x = s[j] and dirs[j+1]·x = s[j+1]
    best = None
    for k in range(3):
        axis, u, v = vt[k], vt[(k + 1) % 3], vt[(k + 2) % 3]
        local = hull - c
        h = local @ axis
        uv = np.stack([local @ u, local @ v], axis=1)
        s = (uv @ dirs.T).max(axis=0)  # support of the cross-section along each direction
        poly = np.linalg.solve(A, np.stack([s, s[j1]], axis=1)[..., None])[..., 0]
        area = 0.5 * abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1))))
        vol = area * float(h.max() - h.min())
        if best is None or vol < best[0]:
            ring = c + poly[:, :1] * u + poly[:, 1:] * v
            best = (vol, np.concatenate([ring + h.min() * axis, ring + h.max() * axis]))
    return best[1]


def winding(shell: np.ndarray, p: np.ndarray) -> float:
    """Generalised winding number of the point p about the outward-oriented closed triangle shell
    (1 inside, 0 outside; exact for a point off the shell)."""
    a, b, c = shell[:, 0] - p, shell[:, 1] - p, shell[:, 2] - p
    la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la
           + np.einsum("ij,ij->i", c, a) * lb)
    return float(np.sum(np.arctan2(det, den))) / (2 * math.pi)


def near_faces(A: Mesh, B: Mesh, M: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Face ids of A and of B (moved by the rigid M into A's frame) whose triangles may come
    within ``tol`` of the other mesh; None when none do (the meshes are farther apart than tol).

    ``BRepExtrema_ShapeProximity`` is conservative: it may list faces that are farther, never
    misses one that is closer.
    """
    loc = to_location(M).wrapped
    prox = BRepExtrema_ShapeProximity(A.shape, B.shape.Moved(loc), tol)
    prox.Perform()
    m1, m2 = prox.OverlapSubShapes1(), prox.OverlapSubShapes2()
    if m1.Size() == 0:
        return None
    back = loc.Inverted()  # B's faces come back moved: undo it to match them to B's own
    fa = [A.face_id(prox.GetSubShape1(i)) for i in range(A.n_explored) if m1.IsBound(i)]
    fb = [B.face_id(prox.GetSubShape2(i).Moved(back)) for i in range(B.n_explored) if m2.IsBound(i)]
    if None in fa or None in fb:  # an unmatched face: every face of that side may be near
        fa = range(len(A.faces)) if None in fa else fa
        fb = range(len(B.faces)) if None in fb else fb
    return np.unique(np.array(list(fa), dtype=np.int64)), np.unique(np.array(list(fb), dtype=np.int64))


def _profile_boxes(tris: np.ndarray, c: np.ndarray, u: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """(n, 4) boxes [h_lo, h_hi, r_lo, r_hi] holding each triangle's image in the (height along
    the line, radius from the line) half-plane: h is linear, r convex (its max sits at a vertex,
    its min is the 2D distance from the axis to the triangle projected along it)."""
    d = tris - c
    h = d @ u
    x, y = d @ e1, d @ e2
    r = np.hypot(x, y)
    P = np.stack([x, y], axis=2)  # (n, 3, 2)
    best = np.full(len(tris), np.inf)
    sign = np.zeros((len(tris), 3))
    for i in range(3):
        a, b = P[:, i], P[:, (i + 1) % 3]
        ab = b - a
        L2 = np.einsum("ij,ij->i", ab, ab)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(L2 > 0, np.clip(-np.einsum("ij,ij->i", a, ab) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0), 0.0)
        best = np.minimum(best, np.linalg.norm(a + t[:, None] * ab, axis=1))
        sign[:, i] = ab[:, 0] * (-a[:, 1]) - ab[:, 1] * (-a[:, 0])  # which side of edge i the axis lies
    inside = np.all(sign >= 0, axis=1) | np.all(sign <= 0, axis=1)
    return np.stack([h.min(axis=1), h.max(axis=1), np.where(inside, 0.0, best), r.max(axis=1)], axis=1)


def radial_band(tris: np.ndarray, point, direction) -> tuple[float, float]:
    """(nearest, farthest) distance from the line (point, direction) of a triangle set.

    Any two points are at least |r_a − r_b| apart (r = distance from the line), and every point of
    a solid lies within its boundary's radius range (the regions nearer the line and farther from
    it than the whole boundary are connected and unbounded along the line, so outside the solid).
    So two parts whose bands are disjoint keep that gap however far one turns about the line or
    slides along it — a piston inside the cooling fins around its barrel.
    """
    if not len(tris):
        return math.inf, -math.inf
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    e1 = _perp(u)
    boxes = _profile_boxes(tris, np.asarray(point, dtype=float), u, e1, np.cross(u, e1))
    return float(boxes[:, 2].min()), float(boxes[:, 3].max())


def revolution_gap(tris_a: np.ndarray, tris_b: np.ndarray, point, direction, cap: float) -> float:
    """Lower bound (≤ cap) on the distance between triangle set A and triangle set B turned by
    ANY angle about the line (point, direction) — the distance between the two sets' images in
    the (height, radius) half-plane, which rotation about the line leaves unchanged.

    Both images are rasterised (cells ≤ cap/32) from per-triangle boxes; the bound is the
    distance between occupied cells less a cell diagonal. Returns cap when they are farther.

    For two solids' boundary triangles this bounds the distance between their *boundaries* only:
    one solid buried in the other has a disjoint image too, so a caller must rule containment
    out separately (it cannot change while turning about the line keeps the boundaries apart).
    """
    if not len(tris_a) or not len(tris_b) or cap <= 0:
        return 0.0
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    c = np.asarray(point, dtype=float)
    e1 = _perp(u)
    e2 = np.cross(u, e1)
    A = _profile_boxes(tris_a, c, u, e1, e2)
    B = _profile_boxes(tris_b, c, u, e1, e2)
    lo = np.maximum(A[:, [0, 2]].min(axis=0), B[:, [0, 2]].min(axis=0)) - cap  # window: where both
    hi = np.minimum(A[:, [1, 3]].max(axis=0), B[:, [1, 3]].max(axis=0)) + cap  # images may be near
    if np.any(hi <= lo):
        return cap
    s = max(cap / 32.0, float(np.max(hi - lo)) / 1500.0)
    shape = tuple(int(n) for n in np.ceil((hi - lo) / s).astype(np.int64) + 1)

    def raster(boxes: np.ndarray) -> np.ndarray:
        boxes = boxes[(boxes[:, 1] >= lo[0]) & (boxes[:, 0] <= hi[0]) & (boxes[:, 3] >= lo[1]) & (boxes[:, 2] <= hi[1])]
        grid = np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.int64)
        if len(boxes):
            i0 = np.clip(np.floor((boxes[:, 0] - lo[0]) / s).astype(np.int64), 0, shape[0] - 1)
            i1 = np.clip(np.floor((boxes[:, 1] - lo[0]) / s).astype(np.int64), 0, shape[0] - 1)
            j0 = np.clip(np.floor((boxes[:, 2] - lo[1]) / s).astype(np.int64), 0, shape[1] - 1)
            j1 = np.clip(np.floor((boxes[:, 3] - lo[1]) / s).astype(np.int64), 0, shape[1] - 1)
            np.add.at(grid, (i0, j0), 1)
            np.add.at(grid, (i0, j1 + 1), -1)
            np.add.at(grid, (i1 + 1, j0), -1)
            np.add.at(grid, (i1 + 1, j1 + 1), 1)
        return grid.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0

    ga, gb = raster(A), raster(B)
    if not ga.any() or not gb.any():
        return cap
    if np.any(ga & gb):
        return 0.0
    dist = distance_transform_edt(~ga)[gb].min() * s
    return float(min(cap, max(0.0, dist - s * math.sqrt(2.0))))


# ------------------------------------------------------------------------------ rotational symmetry


def revolution_axis(shape: TopoDS_Shape, *, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
    """(point, unit direction) of an axis ``shape`` is invariant under any rotation about, or None.

    Exact and conservative: every face must be a plane perpendicular to the axis or a cylinder,
    cone, torus, sphere (centre on the axis) or surface of revolution coaxial with it, and every
    edge a full circle coaxial with it, a seam or degenerate. Anything else (a flat, a key, a
    tooth, a spline face) → None. ``scale`` (mm, the part size) sets the position tolerance.
    """
    axis: list = []  # [point, direction]
    normals, centres = [], []
    tol = _AXIS_DIST * max(1.0, scale)

    def same_axis(p, u) -> bool:
        p, u = np.array([p.X(), p.Y(), p.Z()]), np.array([u.X(), u.Y(), u.Z()])
        u = u / np.linalg.norm(u)
        if not axis:
            axis.extend([p, u])
            return True
        p0, u0 = axis
        if np.linalg.norm(np.cross(u, u0)) > _AXIS_TOL:
            return False
        w = p - p0
        return float(np.linalg.norm(w - np.dot(w, u0) * u0)) <= tol

    n_faces = 0
    for f in _explore(shape, TopAbs_FACE):
        face = TopoDS.Face(f)
        n_faces += 1
        s = BRepAdaptor_Surface(face, True)
        kind = s.GetType()
        if kind == GeomAbs_Plane:
            d = s.Plane().Axis().Direction()
            normals.append(np.array([d.X(), d.Y(), d.Z()]))
        elif kind == GeomAbs_Cylinder:
            ax = s.Cylinder().Axis()
            if not same_axis(ax.Location(), ax.Direction()):
                return None
        elif kind == GeomAbs_Cone:
            ax = s.Cone().Axis()
            if not same_axis(ax.Location(), ax.Direction()):
                return None
        elif kind == GeomAbs_Torus:
            ax = s.Torus().Axis()
            if not same_axis(ax.Location(), ax.Direction()):
                return None
        elif kind == GeomAbs_Sphere:
            c = s.Sphere().Location()
            centres.append(np.array([c.X(), c.Y(), c.Z()]))
        elif kind == GeomAbs_SurfaceOfRevolution:
            ax = s.AxeOfRevolution()
            if not same_axis(ax.Location(), ax.Direction()):
                return None
        else:
            return None
        for e in _explore(face, TopAbs_EDGE):
            edge = TopoDS.Edge(e)
            if BRep_Tool.Degenerated_s(edge) or BRep_Tool.IsClosed_s(edge, face):
                continue  # pole of a sphere/cone, seam of a periodic face
            c = BRepAdaptor_Curve(edge)
            if c.GetType() != GeomAbs_Circle:
                return None
            if abs(c.LastParameter() - c.FirstParameter() - 2 * math.pi) > 1e-9:
                return None  # an arc: the face is not a full revolution
            circ = c.Circle()
            if not same_axis(circ.Location(), circ.Axis().Direction()):
                return None
    if not n_faces or not axis:
        return None
    p0, u0 = axis
    for n in normals:
        if np.linalg.norm(np.cross(n / np.linalg.norm(n), u0)) > _AXIS_TOL:
            return None
    for c in centres:
        w = c - p0
        if np.linalg.norm(w - np.dot(w, u0) * u0) > tol:
            return None
    return p0.copy(), u0.copy()


def rot_about(c: np.ndarray, u: np.ndarray, phi: float) -> np.ndarray:
    """4x4 rotation by phi (rad) about the line through c along the unit u."""
    x, y, z = u
    cs, sn = math.cos(phi), math.sin(phi)
    C = 1.0 - cs
    R = np.array([[cs + x * x * C, x * y * C - z * sn, x * z * C + y * sn],
                  [y * x * C + z * sn, cs + y * y * C, y * z * C - x * sn],
                  [z * x * C - y * sn, z * y * C + x * sn, cs + z * z * C]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c
    return T


def _perp(u: np.ndarray) -> np.ndarray:
    """A deterministic unit vector perpendicular to the unit u."""
    e = np.zeros(3)
    e[int(np.argmin(np.abs(u)))] = 1.0
    w = e - np.dot(e, u) * u
    return w / np.linalg.norm(w)


def canonical_pose(M: np.ndarray, axis_a, axis_b, refs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(Mc, L) with M = L·Mc·R for a rotation L about a's axis and R about b's axis (each only
    when that part has one): Mc is the same for every pose in that class, so a pair measured at
    Mc is known at M — its points (a's frame) mapped by L (None = identity).

    ``refs_b`` (points fixed in b's frame) pick L when b has no axis of its own: the first one
    off a's axis is turned into a fixed half-plane.
    """
    L = None
    if axis_a is not None:
        ca, ua = axis_a
        pts = [axis_b[0], axis_b[0] + axis_b[1]] if axis_b is not None else refs_b
        w = None
        for p in pts:
            q = M[:3, :3] @ p + M[:3, 3] - ca
            q = q - np.dot(q, ua) * ua
            if np.linalg.norm(q) > 1e-6:
                w = q
                break
        if w is not None:
            ea = _perp(ua)
            fa = np.cross(ua, ea)
            psi = math.atan2(float(np.dot(w, fa)), float(np.dot(w, ea)))
            L = rot_about(ca, ua, psi)
            M = rot_about(ca, ua, -psi) @ M
    if axis_b is not None:
        cb, ub = axis_b
        v = M[:3, :3] @ ub
        e = _perp(v)
        wb = _perp(ub)
        x0 = M[:3, :3] @ wb
        y0 = np.cross(v, x0)
        phi = math.atan2(float(np.dot(e, y0)), float(np.dot(e, x0)))
        M = M @ rot_about(cb, ub, phi)
    return M, L


# ------------------------------------------------------------------------------ analytic face bounds


def face_surfaces(mesh: Mesh) -> list[tuple | None]:
    """Per face id of ``mesh``: the infinite surface the face lies on — ``("plane", point, unit
    normal)``, ``("cylinder", axis point, unit axis, radius)``, ``("sphere", centre, radius)`` —
    or None for any other surface (home frame of the mesh's part)."""
    out: list[tuple | None] = []
    for face in mesh.faces:
        try:
            s = BRepAdaptor_Surface(face, True)
            kind = s.GetType()
            if kind == GeomAbs_Plane:
                ax = s.Plane().Axis()
                out.append(("plane", _vec(ax.Location()), _vec(ax.Direction())))
            elif kind == GeomAbs_Cylinder:
                c = s.Cylinder()
                out.append(("cylinder", _vec(c.Axis().Location()), _vec(c.Axis().Direction()), float(c.Radius())))
            elif kind == GeomAbs_Sphere:
                sp = s.Sphere()
                out.append(("sphere", _vec(sp.Location()), float(sp.Radius())))
            else:
                out.append(None)
        except Exception:  # a face OCC can't adapt: no bound from it
            out.append(None)
    return out


def _vec(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def _moved(surface: tuple, M: np.ndarray) -> tuple:
    R, t = M[:3, :3], M[:3, 3]
    kind = surface[0]
    if kind == "plane":
        return kind, R @ surface[1] + t, R @ surface[2]
    if kind == "cylinder":
        return kind, R @ surface[1] + t, R @ surface[2], surface[3]
    return kind, R @ surface[1] + t, surface[2]


def _line_dist(p: np.ndarray, q: np.ndarray, u: np.ndarray) -> float:
    """Distance from point p to the line (q, unit u)."""
    d = p - q
    return float(np.linalg.norm(d - (d @ u) * u))


def _pair_gap(a: tuple, b: tuple) -> float:
    """Lower bound on the distance between two infinite surfaces (0 when they may meet)."""
    if a[0] > b[0]:  # canonical order: cylinder < plane < sphere
        a, b = b, a
    ka, kb = a[0], b[0]
    if ka == "sphere":  # sphere / sphere
        d = float(np.linalg.norm(a[1] - b[1]))
        return max(0.0, d - a[2] - b[2], abs(a[2] - b[2]) - d)
    if kb == "sphere":  # cylinder or plane / sphere: |centre to surface| − r
        c, r = b[1], b[2]
        if ka == "plane":
            return max(0.0, abs(float((c - a[1]) @ a[2])) - r)
        return max(0.0, abs(_line_dist(c, a[1], a[2]) - a[3]) - r)
    if ka == "plane":  # plane / plane: parallel ones are their offset apart
        if float(np.linalg.norm(np.cross(a[2], b[2]))) > _PARALLEL:
            return 0.0
        return abs(float((b[1] - a[1]) @ a[2]))
    if kb == "plane":  # cylinder / plane: an axis parallel to the plane keeps it (offset − R) away
        if abs(float(a[2] @ b[2])) > _PARALLEL:
            return 0.0
        return max(0.0, abs(float((a[1] - b[1]) @ b[2])) - a[3])
    # cylinder / cylinder: parallel axes e apart — outside each other e − R1 − R2, nested |R1 − R2| − e
    if float(np.linalg.norm(np.cross(a[2], b[2]))) > _PARALLEL:
        return 0.0
    e = _line_dist(b[1], a[1], a[2])
    return max(0.0, e - a[3] - b[3], abs(a[3] - b[3]) - e)


def surface_gap(sa: list, sb: list, near: tuple[np.ndarray, np.ndarray], M: np.ndarray, *,
                limit: int = 64) -> float:
    """Lower bound on the distance between A's ``near[0]`` faces and B's ``near[1]`` faces (B
    moved by the rigid M into A's frame) from the infinite surfaces they lie on (``sa``/``sb``
    from ``face_surfaces``): a face is a subset of its surface, so two faces are at least as far
    apart as their surfaces. 0 when any face is of another kind or there are more than ``limit``
    face pairs."""
    fa, fb = list(near[0]), list(near[1])
    if not fa or not fb or len(fa) * len(fb) > limit:
        return 0.0
    if any(sa[i] is None for i in fa) or any(sb[j] is None for j in fb):
        return 0.0
    moved = {j: _moved(sb[j], M) for j in fb}
    best = math.inf
    for i in fa:
        for j in fb:
            best = min(best, _pair_gap(sa[i], moved[j]))
            if best <= 0.0:
                return 0.0
    return best
