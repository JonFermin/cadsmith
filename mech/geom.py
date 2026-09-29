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

import numpy as np
from build123d import Circle, Location, Part, Pos, SlotCenterToCenter, Vector, Vertex, extrude
from OCP.gp import gp_Trsf

__all__ = [
    "vec3",
    "unit",
    "to_location",
    "from_location",
    "rot_about_line",
    "translation",
    "transform_points",
    "transform_aabb",
    "to_json16",
    "fnum",
    "slug",
    "link",
    "circle_intersect",
]

_EPS_LEN = 1e-12  # below this a direction vector is treated as zero
_JSON_ZERO = 1e-9  # to_json16 snaps |entry| below this to 0 (round-off in R and in mm alike)

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
