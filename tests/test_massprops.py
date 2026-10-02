"""mech.massprops + mech.materials: analytic box inertia, rotations, parallel axis, overrides."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, Compound, Plane, Pos, Rectangle, Rot

from mech import MATERIALS, Assembly, Material, assembly_props, part_props
from mech.geom import rot_about_line, translation
from mech.materials import get_material

STEEL = 7.85e-6  # kg/mm³
A, B, C = 10.0, 20.0, 30.0  # box edges along x, y, z (mm)
M_BOX = STEEL * A * B * C  # 0.0471 kg


def box_inertia(m, a, b, c) -> np.ndarray:
    """Solid cuboid about its center, edges a, b, c along x, y, z (kg·mm²)."""
    return m / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def steel_box_asm(*placements) -> Assembly:
    asm = Assembly("boxes")
    for k, loc in enumerate(placements):
        asm.part(f"box{k}", loc * Box(A, B, C), material="steel", ground=True)
    return asm


def test_steel_box_mass_com_inertia():
    asm = steel_box_asm(Pos(5, 6, 7))
    mp = part_props(asm.parts["box0"])
    assert mp.mass_kg == pytest.approx(0.0471, rel=1e-12)
    assert mp.volume_mm3 == pytest.approx(6000.0, rel=1e-12)
    np.testing.assert_allclose(mp.com, [5, 6, 7], atol=1e-9)
    np.testing.assert_allclose(mp.inertia, box_inertia(M_BOX, A, B, C), rtol=1e-9, atol=1e-12)
    assert mp.inertia[0, 0] == pytest.approx(5.1025, rel=1e-9)  # 0.0471·(20² + 30²)/12


def test_rotated_part_inertia_has_correct_products():
    """A box modeled rotated 30° about Z: I = R I₀ Rᵀ (checks the sign of OCC's products of inertia)."""
    asm = steel_box_asm(Pos(1, 2, 3) * Rot(0, 0, 30))
    mp = part_props(asm.parts["box0"])
    R = rot_about_line((0, 0, 0), (0, 0, 1), 30)[:3, :3]
    want = R @ box_inertia(M_BOX, A, B, C) @ R.T
    assert want[0, 1] > 0  # I_xy = −Σ m x y with the long y-edge tilted toward −x: positive
    np.testing.assert_allclose(mp.inertia, want, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mp.com, [1, 2, 3], atol=1e-9)


def test_assembly_props_with_rotated_transform():
    asm = steel_box_asm(Pos(5, 6, 7))
    T = translation((100, -20, 40)) @ rot_about_line((3, 1, 0), (1, 2, 2), 40)
    total = assembly_props(asm, transforms={"box0": T})
    R, t = T[:3, :3], T[:3, 3]
    assert total.mass_kg == pytest.approx(M_BOX, rel=1e-12)
    np.testing.assert_allclose(total.com, R @ [5, 6, 7] + t, atol=1e-9)
    np.testing.assert_allclose(total.inertia, R @ box_inertia(M_BOX, A, B, C) @ R.T, rtol=1e-9, atol=1e-9)


def test_assembly_props_parallel_axis():
    """Two boxes 40 mm apart along x (+ one rotated by a transform): parallel-axis theorem by hand."""
    asm = steel_box_asm(Pos(0, 0, 0), Pos(40, 0, 0))
    total = assembly_props(asm)
    I0 = box_inertia(M_BOX, A, B, C)
    np.testing.assert_allclose(total.com, [20, 0, 0], atol=1e-9)
    d2 = M_BOX * 20.0**2
    np.testing.assert_allclose(total.inertia, 2 * I0 + np.diag([0, 2 * d2, 2 * d2]), rtol=1e-9, atol=1e-9)

    # rotate box1 by 90° about its own vertical axis: its x/y moments swap, COMs unchanged
    T = rot_about_line((40, 0, 0), (0, 0, 1), 90)
    total = assembly_props(asm, transforms={"box1": T})
    I1 = box_inertia(M_BOX, B, A, C)
    np.testing.assert_allclose(total.inertia, I0 + I1 + np.diag([0, 2 * d2, 2 * d2]), rtol=1e-9, atol=1e-9)
    assert total.volume_mm3 == pytest.approx(12000.0)


def test_mass_override_scales_mass_and_inertia():
    asm = Assembly("m")
    asm.part("b", Box(A, B, C), material="steel", mass_g=100.0, ground=True)
    mp = part_props(asm.parts["b"])
    assert mp.mass_kg == pytest.approx(0.1)
    np.testing.assert_allclose(mp.inertia, box_inertia(0.1, A, B, C), rtol=1e-9)


def test_overlapping_solids_are_fused_disjoint_are_not():
    asm = Assembly("c")
    overlap = Compound([Box(10, 10, 10), Pos(5, 0, 0) * Box(10, 10, 10)])  # union = 15×10×10
    disjoint = Compound([Box(10, 10, 10), Pos(30, 0, 0) * Box(10, 10, 10)])
    asm.part("o", overlap, material="steel", ground=True)
    asm.part("d", disjoint, material="steel", ground=True)
    o, d = part_props(asm.parts["o"]), part_props(asm.parts["d"])
    assert o.volume_mm3 == pytest.approx(1500.0, rel=1e-9)
    assert o.mass_kg == pytest.approx(STEEL * 1500.0, rel=1e-9)
    np.testing.assert_allclose(o.com, [2.5, 0, 0], atol=1e-9)
    np.testing.assert_allclose(o.inertia, box_inertia(STEEL * 1500, 15, 10, 10), rtol=1e-9, atol=1e-9)
    assert d.volume_mm3 == pytest.approx(2000.0, rel=1e-9)
    np.testing.assert_allclose(d.com, [15, 0, 0], atol=1e-9)


def test_zero_volume_part_is_massless():
    asm = Assembly("z")
    asm.part("sheet", Plane.XY * Rectangle(10, 10), ground=True)
    mp = part_props(asm.parts["sheet"])
    assert mp.mass_kg == 0.0 and mp.volume_mm3 == 0.0
    np.testing.assert_allclose(mp.inertia, 0.0)
    np.testing.assert_allclose(mp.com, [0, 0, 0], atol=1e-9)


def test_materials_table_and_lookup():
    assert MATERIALS["PLA"].density == 1.24 and MATERIALS["steel"].density == 7.85
    assert get_material("Aluminum_6061") is MATERIALS["aluminum_6061"]
    assert get_material("carbon fiber") is MATERIALS["carbon_fiber"]
    assert get_material("aluminium").name == "aluminum_6061"
    assert get_material("steel", density=7.9) == Material("steel", 7.9)
    assert get_material("walnut", density=0.6) == Material("walnut", 0.6)
    m = Material("custom", 2.0)
    assert get_material(m) is m
    with pytest.raises(KeyError, match="did you mean 'steel'"):
        get_material("steal")
    with pytest.raises(ValueError):
        get_material("PLA", density=-1)
    expected = {"PLA": 1.24, "PETG": 1.27, "ABS": 1.04, "ASA": 1.07, "TPU": 1.21, "nylon": 1.01, "PA12": 1.01,
                "resin": 1.15, "POM": 1.41, "acrylic": 1.18, "aluminum_6061": 2.70, "steel": 7.85,
                "stainless_304": 8.00, "brass": 8.50, "copper": 8.96, "plywood": 0.68, "MDF": 0.75,
                "carbon_fiber": 1.60, "rubber": 1.10}
    assert {k: v.density for k, v in MATERIALS.items()} == expected


def test_default_material_is_pla():
    asm = Assembly("pla")
    asm.part("b", Box(10, 10, 10), ground=True)
    assert part_props(asm.parts["b"]).mass_kg == pytest.approx(1.24e-6 * 1000)
    assert math.isclose(assembly_props(asm).mass_kg, 1.24e-3)


def test_solid_body_is_fused_cached_and_drops_non_solids():
    """The clearance checker measures the same fused body the mass properties use."""
    from mech.massprops import solid_body

    overlap = Compound([Box(10, 10, 10), Pos(5, 0, 0) * Box(10, 10, 10)])
    body = solid_body(overlap)
    assert body.volume == pytest.approx(1500.0, rel=1e-9) and len(body.solids()) == 1
    assert solid_body(overlap) is body  # cached per shape object
    touching = Compound([Box(10, 10, 10), Pos(10, 0, 0) * Box(10, 10, 10)])
    assert sum(s.volume for s in solid_body(touching).solids()) == pytest.approx(2000.0, rel=1e-9)
    assert solid_body(Plane.XY * Rectangle(10, 10)) is None
    mixed = Compound([Box(10, 10, 10), Pos(0, 0, 20) * (Plane.XY * Rectangle(10, 10)).face()])
    assert solid_body(mixed).volume == pytest.approx(1000.0, rel=1e-9) and len(solid_body(mixed).faces()) == 6


def test_virtual_parts_are_massless_and_left_out():
    """The knuckles of a ball() joint weigh nothing and don't enter the assembly's properties."""
    asm = steel_box_asm(Pos(0, 0, 0))
    asm.part("arm", Pos(0, 0, 40) * Box(A, B, C), material="steel")
    asm.ball("b", "box0", "arm", (0, 0, 20))
    knuckle = asm.parts["b_k1"]
    assert knuckle.virtual
    mp = part_props(knuckle)
    assert (mp.mass_kg, mp.volume_mm3) == (0.0, 0.0) and np.all(mp.inertia == 0)
    np.testing.assert_allclose(mp.com, (0, 0, 20), atol=1e-9)
    total = assembly_props(asm)
    assert total.mass_kg == pytest.approx(2 * M_BOX, rel=1e-9) and total.volume_mm3 == pytest.approx(2 * A * B * C)
