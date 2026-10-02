"""Mass properties of parts and assemblies (spec §4.3).

Units: mass kg, volume mm³, COM mm (home world), inertia kg·mm² about the COM in world axes.
Density ρ in g/cm³ becomes kg/mm³ as ρ·1e-6.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from build123d import Compound
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .geom import vec3

__all__ = ["MassProps", "part_props", "assembly_props", "solid_body"]

_OVERLAP_RTOL = 1e-3  # fuse when Σ solid volumes exceeds the fused volume by more than 0.1 %
_BODY_CACHE_SIZE = 256  # solid_body results kept (mass props and clearance share them)
_body_cache: OrderedDict[int, tuple[object, object]] = OrderedDict()


@dataclass
class MassProps:
    mass_kg: float
    volume_mm3: float
    com: np.ndarray  # mm, world
    inertia: np.ndarray  # kg·mm² about the COM, world axes (3x3)


def _boxes_overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


def solid_body(shape):
    """The solid material of ``shape``: overlapping solids fused, non-solids (faces, shells) dropped.

    ``None`` when the shape has no solids (zero volume). OCC integrates a compound solid by
    solid and its booleans treat a compound's solids as separate arguments, so overlapping
    solids (a list-of-shapes part, overlapping library sub-bodies) would be counted twice in
    volumes and break interference booleans; they are fused first (only when some pair of solid
    bounding boxes actually overlaps, to skip the boolean in the common case). Results are
    cached per shape object, so mass properties and the clearance checker fuse a part once.
    """
    key = id(shape)
    hit = _body_cache.get(key)
    if hit is not None and hit[0] is shape:
        _body_cache.move_to_end(key)
        return hit[1]
    body = _fuse_overlapping(shape)
    _body_cache[key] = (shape, body)
    while len(_body_cache) > _BODY_CACHE_SIZE:
        _body_cache.popitem(last=False)
    return body


def _fuse_overlapping(shape):
    solids = shape.solids() if hasattr(shape, "solids") else []
    if not solids:
        return None
    if len(solids) == 1:
        return solids[0]
    body = Compound(list(solids))
    boxes = [(vec3(bb.min), vec3(bb.max)) for bb in (s.bounding_box() for s in solids)]
    if not any(_boxes_overlap(boxes[i], boxes[k]) for i in range(len(boxes)) for k in range(i + 1, len(boxes))):
        return body
    fused = solids[0].fuse(*solids[1:])
    if sum(s.volume for s in solids) > fused.volume * (1 + _OVERLAP_RTOL):
        return fused
    return body


def _unit_density_props(shape) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume mm³, centroid mm, inertia about centroid mm⁵ at unit density) of a solid body."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props)
    vol = props.Mass()
    c = props.CentreOfMass()
    M = props.MatrixOfInertia()
    J = np.array([[M.Value(i, k) for k in (1, 2, 3)] for i in (1, 2, 3)])
    if vol < 0:  # reversed orientation integrates to negative volume and inertia
        vol, J = -vol, -J
    return float(vol), np.array([c.X(), c.Y(), c.Z()]), (J + J.T) / 2


def part_props(part) -> MassProps:
    """Mass properties of a ``Part`` at home; ``part.mass_g`` rescales mass and inertia.

    A ``virtual`` part (the internal knuckle of a ball joint) is massless: no mass, volume or inertia.
    """
    if getattr(part, "virtual", False):
        bb = part.shape.bounding_box()
        return MassProps(0.0, 0.0, (vec3(bb.min) + vec3(bb.max)) / 2, np.zeros((3, 3)))
    body = solid_body(part.shape)
    vol, com, J = (0.0, None, np.zeros((3, 3))) if body is None else _unit_density_props(body)
    if vol <= 0:  # no solid material: a massless (or, with mass_g, point-mass) marker at the bbox center
        bb = part.shape.bounding_box()
        vol, com, J = 0.0, (vec3(bb.min) + vec3(bb.max)) / 2, np.zeros((3, 3))
    if part.mass_g is not None:
        mass = part.mass_g / 1000.0
        # uniform density mass/volume; a zero-volume part becomes a point mass
        inertia = J * (mass / vol) if vol > 0 else np.zeros((3, 3))
    else:
        rho = part.material.density * 1e-6  # g/cm³ -> kg/mm³
        mass, inertia = rho * vol, rho * J
    return MassProps(float(mass), vol, com, inertia)


def assembly_props(asm, props: dict[str, MassProps] | None = None,
                   transforms: dict[str, np.ndarray] | None = None) -> MassProps:
    """Combined mass properties of ``props`` (default: every non-virtual part at home) moved by
    ``transforms``.

    I = Σ [R_i I_i R_iᵀ + m_i (‖d_i‖² E − d_i d_iᵀ)],  d_i = R_i c_i + t_i − c_total.
    """
    if props is None:
        props = {name: part_props(p) for name, p in asm.parts.items() if not getattr(p, "virtual", False)}
    transforms = transforms or {}
    items = []
    for name, mp in props.items():
        T = transforms.get(name)
        R, t = (np.eye(3), np.zeros(3)) if T is None else (np.asarray(T)[:3, :3], np.asarray(T)[:3, 3])
        items.append((mp.mass_kg, mp.volume_mm3, R @ mp.com + t, R @ mp.inertia @ R.T))
    mass = sum(m for m, _, _, _ in items)
    volume = sum(v for _, v, _, _ in items)
    if mass <= 0:
        return MassProps(0.0, volume, np.zeros(3), np.zeros((3, 3)))
    com = sum(m * c for m, _, c, _ in items) / mass
    inertia = np.zeros((3, 3))
    for m, _, c, I in items:
        d = c - com
        inertia += I + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return MassProps(float(mass), float(volume), com, inertia)
