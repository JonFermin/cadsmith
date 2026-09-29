"""mech.assembly: builders, validation (all errors, did-you-mean), warnings, groups, couplings."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Axis, Box, Location, Plane, Pos, Rectangle

from conftest import make_disc, make_four_bar
from mech import Assembly, Kinematics, ModelError
from mech.geom import link


def names_in(errors: list[str], *fragments: str) -> bool:
    """Some single error message contains every fragment."""
    return any(all(f in e for f in fragments) for e in errors)


# ------------------------------------------------------------------------------ builders


def test_duplicate_names_raise_immediately():
    asm = Assembly("dup")
    asm.part("frame", Box(10, 10, 10), ground=True)
    with pytest.raises(ValueError, match="duplicate part name 'frame'"):
        asm.part("frame", Box(1, 1, 1))
    asm.part("arm", Box(1, 1, 1))
    asm.revolute("j", "frame", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    with pytest.raises(ValueError, match="duplicate joint"):
        asm.prismatic("j", "frame", "arm", origin=(0, 0, 0), axis=(1, 0, 0))


def test_part_defaults_libpart_and_materials():
    class FakeLibPart:  # anything with .shape (+ optional .bom / .mass_g)
        shape = Box(10, 10, 10)
        bom = "608 bearing"
        mass_g = 12.0

    asm = Assembly("p")
    asm.part("base", Box(50, 50, 5), ground=True)
    asm.part("brg", FakeLibPart(), material="steel")
    asm.part("brg2", FakeLibPart(), mass_g=3.0, bom="other")
    asm.part("wood", Box(5, 5, 5), material="walnut", density=0.6)
    b = asm.parts["brg"]
    assert (b.bom, b.mass_g, b.material.name) == ("608 bearing", 12.0, "steel")
    assert (asm.parts["brg2"].bom, asm.parts["brg2"].mass_g) == ("other", 3.0)
    assert asm.parts["wood"].material.density == 0.6
    assert asm.parts["base"].color.startswith("#") and asm.parts["brg"].color != asm.parts["brg2"].color


def test_joint_at_location_and_axis():
    asm = Assembly("at")
    asm.part("frame", Box(40, 40, 40), ground=True)
    asm.part("a", Box(5, 5, 5))
    asm.part("b", Box(5, 5, 5))
    asm.revolute("ja", "frame", "a", at=Location((10, 0, 0), (0, 90, 0)))  # local Z -> world +X
    asm.prismatic("jb", "frame", "b", at=Axis((1, 2, 3), (0, 1, 0)))
    np.testing.assert_allclose(asm.joints["ja"].origin, [10, 0, 0], atol=1e-12)
    np.testing.assert_allclose(asm.joints["ja"].axis, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(asm.joints["jb"].origin, [1, 2, 3])
    np.testing.assert_allclose(asm.joints["jb"].axis, [0, 1, 0])
    assert asm.validate() == []


def test_screw_ratio_resolved_even_when_declared_before_joints():
    asm = Assembly("s")
    asm.part("frame", Box(40, 40, 5), ground=True)
    asm.part("screw", Pos(0, 0, 30) * Box(4, 4, 50))
    asm.part("nut", Pos(0, 0, 20) * Box(12, 12, 6))
    asm.screw("j_rot", "j_nut", lead=8)  # joints don't exist yet
    assert math.isnan(asm.couplings[0].ratio)
    asm.revolute("j_rot", "frame", "screw", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.prismatic("j_nut", "frame", "nut", origin=(0, 0, 0), axis=(0, 0, 1))
    assert asm.validate() == []
    assert asm.couplings[0].ratio == pytest.approx(-8 / 360)


# ------------------------------------------------------------------------------ validation


def broken_assembly() -> Assembly:
    asm = Assembly("broken")
    asm.part("frame", Box(100, 20, 5), ground=True)
    asm.part("crank", link((0, 0, 0), (40, 0, 0), 10, 5), material="alumnum")
    asm.part("rocker", link((100, 0, 0), (80, 60, 0), 10, 5))
    asm.part("slider", Box(10, 10, 10))
    asm.part("orphan", Box(3, 3, 3))  # floating
    asm.part("sheet", Plane.XY * Rectangle(10, 10), mass_g=5.0)  # zero volume with mass_g
    asm.part("loop_a", Box(3, 3, 3))
    asm.part("loop_b", Box(3, 3, 3))
    asm.revolute("j_crank", "fram", "crank", origin=(0, 0, 0), axis=(0, 0, 1))  # typo parent
    asm.revolute("j_rocker", "frame", "rocker", origin=(100, 0, 0), axis=(0, 0, 1), limits=(-10, 10), home=20)
    asm.prismatic("j_slide", "frame", "slider", origin=(0, 0, 0), axis=(1, 0, 0))
    asm.revolute("j_slide2", "frame", "slider", origin=(0, 0, 0), axis=(0, 0, 1))  # 2nd parent joint
    asm.revolute("j_sheet", "frame", "sheet", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_ab", "loop_b", "loop_a", origin=(0, 0, 0), axis=(0, 0, 1))  # a <- b <- a: cycle
    asm.revolute("j_ba", "loop_a", "loop_b", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.gear("j_crnk", "j_rocker", -1.0)  # unknown driver
    asm.gear("j_rocker", "j_slide", 2.0)  # revolute -> prismatic gear: kind mismatch
    asm.couple("j_crank", "j_rocker", 0.5, name="c2")
    asm.couple("j_slide", "j_rocker", 1.0, name="c3")  # j_rocker driven twice
    asm.couple("j_ab", "j_ba", 1.0)
    asm.couple("j_ba", "j_ab", 1.0)  # coupling cycle
    asm.pin("p_far", "rocker", "crank", point=(80, 60, 0))  # ball pin 45+ mm from the crank
    asm.pin("p_x", "rocker", "crankk", point=(80, 60, 0))
    asm.probe("tip", "rockr", (80, 60, 0))
    asm.study("s", {"j_crnak": (0, 90)})
    asm.target("t", "span:j_rockr", min=1)
    asm.target("t2", "spam:j_rocker")
    return asm


def test_validate_collects_every_error_with_suggestions():
    errors = broken_assembly().validate()  # must not raise
    assert names_in(errors, "unknown parent part 'fram'", "did you mean 'frame'")
    assert names_in(errors, "unknown material 'alumnum'", "did you mean 'aluminum_6061'")
    assert names_in(errors, "'orphan' is floating")
    assert not names_in(errors, "'crank' is floating")  # its joint exists; the error is the unknown parent
    assert names_in(errors, "'slider' is the child of 2 joints", "j_slide", "j_slide2")
    assert names_in(errors, "joint cycle", "loop_a", "loop_b")
    assert names_in(errors, "j_rocker", "home 20 is outside limits [-10, 10]")
    assert names_in(errors, "'sheet'", "mass_g", "zero volume")
    assert names_in(errors, "unknown driver joint 'j_crnk'", "did you mean 'j_crank'")
    assert names_in(errors, "gear 'gear_j_slide'", "revolute driver and a revolute driven", "prismatic")
    assert names_in(errors, "'j_rocker' is driven by 2 couplings", "c2", "c3")
    assert names_in(errors, "coupling cycle", "j_ab", "j_ba")
    assert names_in(errors, "pin_off_part", "'p_far'", "from part 'crank'", "mm")
    assert names_in(errors, "pin 'p_x'", "unknown part 'crankk'", "did you mean 'crank'")
    assert names_in(errors, "probe 'tip'", "did you mean 'rocker'")
    assert names_in(errors, "study 's'", "unknown joint 'j_crnak'", "did you mean 'j_crank'")
    assert names_in(errors, "target 't'", "unknown joint 'j_rockr'", "did you mean 'j_rocker'")
    assert names_in(errors, "target 't2'", "unknown metric", "did you mean 'span'")
    assert len(errors) == 17  # each problem reported exactly once
    with pytest.raises(ModelError) as exc:
        Kinematics(broken_assembly())
    assert exc.value.errors == errors


def test_pin_off_part_distance_value_and_hinge_line_vs_point():
    asm = make_four_bar()
    assert asm.validate() == []
    B = asm.pins[0].point
    # A hinge pin far up its own axis is still fine (the axis line threads both holes) ...
    asm.pin("p_hinge_high", "coupler", "rocker", point=B + [0, 0, 100], axis=(0, 0, 1))
    # ... a ball pin at that point is 84 mm above the coupler (top at z = 10.5, hole radius 2).
    asm.pin("p_ball_high", "coupler", "rocker", point=B + [0, 0, 100])
    errors = asm.validate()
    assert not names_in(errors, "p_hinge_high")
    d_coupler = math.hypot(100 - 10.5, 2.0)
    assert names_in(errors, "pin_off_part", "'p_ball_high'", f"{d_coupler:.3g} mm from part 'coupler'")
    assert len([e for e in errors if "p_ball_high" in e]) == 2  # coupler and rocker


def test_ball_pin_buried_inside_a_thick_part_is_on_part():
    asm = Assembly("buried")
    asm.part("block", Box(40, 40, 40), ground=True)
    asm.part("arm", Pos(0, 0, 25) * Box(10, 10, 10))
    asm.revolute("j", "block", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.pin("p", "block", "arm", point=(0, 0, 18))  # 2 mm inside the block top, 2 mm below the arm
    assert asm.validate() == []


def test_structural_error_kinds():
    asm = Assembly("x")
    asm.part("g", Box(10, 10, 10), ground=True)
    asm.part("g2", Box(10, 10, 10), ground=True)
    asm.part("a", Box(5, 5, 5))
    asm.revolute("j_g2", "g", "g2", origin=(0, 0, 0), axis=(0, 0, 1))  # ground can't be a child
    asm.revolute("j_a", "g", "a", origin=(0, 0, 0), axis=(0, 0, 0))  # zero axis
    asm.revolute("j_b", "g", "a", origin=(0, 0, 0))  # no axis (and second parent)
    asm.screw("j_a", "j_g2", lead=2)  # revolute -> revolute screw
    asm.rack("j_a", "j_b", pitch_radius=5, sign=2)
    errors = asm.validate()
    assert names_in(errors, "ground part 'g2' cannot be the child of joint 'j_g2'")
    assert names_in(errors, "revolute 'j_a' axis", "zero-length")
    assert names_in(errors, "revolute 'j_b'", "needs origin= and axis=")
    assert names_in(errors, "screw 'screw_j_g2'", "prismatic driven")
    assert names_in(errors, "rack 'rack_j_b'", "sign must be")
    assert Assembly("empty").validate() == ["assembly has no parts"]


def test_rack_sign_needs_geometry():
    def rack_on_axis(**kw) -> Assembly:
        asm = Assembly("r")
        asm.part("frame", Box(100, 10, 2), ground=True)
        asm.part("pinion", make_disc((0, 0, 0)))
        asm.part("rack", Pos(0, 0, 2) * Box(60, 4, 4))  # center on the pinion axis: direction undefined
        asm.revolute("j_p", "frame", "pinion", origin=(0, 0, 0), axis=(0, 0, 1))
        asm.prismatic("j_r", "frame", "rack", origin=(0, 0, 0), axis=(1, 0, 0))
        asm.rack("j_p", "j_r", pitch_radius=10, **kw)
        return asm

    assert names_in(rack_on_axis().validate(), "rack 'rack_j_r'", "pass sign=")
    explicit = rack_on_axis(sign=-1)
    assert explicit.validate() == []
    assert explicit.couplings[0].ratio == pytest.approx(-math.pi * 10 / 180)


# ------------------------------------------------------------------------------ warnings


def test_validate_warnings_near_planar_and_joint_off_part():
    asm = make_four_bar()
    assert asm.validate_warnings() == []
    tilt = 1e-4  # rocker axis tilted by 0.1 mrad: near planar
    asm.joints["j_rocker"].axis = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    codes = [c for c, _ in asm.validate_warnings()]
    assert codes == ["near_planar"]

    asm = make_four_bar()
    asm.part("lever", link((0, 40, 0), (30, 40, 0), 8, 4, z=-2))  # contains its own axis line
    asm.revolute("j_lever", "frame", "lever", origin=(0, 40, 0), axis=(1, 0, 0))  # horizontal axis at y = 40
    warns = asm.validate_warnings()
    assert [c for c, _ in warns] == ["joint_off_part"]
    assert "'j_lever'" in warns[0][1] and "parent 'frame'" in warns[0][1]


def test_joint_off_part_measures_the_rigid_body():
    """A shaft whose revolute parent is a plate with a big hole, carried by a bushing fixed to it."""
    def shaft_in_plate(with_bushing: bool) -> Assembly:
        asm = Assembly("shaft")
        asm.part("plate", Box(60, 60, 5) - Box(20, 20, 10), ground=True)  # 10 mm from the axis
        if with_bushing:
            asm.part("bushing", Box(20, 20, 5) - Box(6, 6, 10))
            asm.fix("bushing", "plate")
        asm.part("shaft", Pos(0, 0, 5) * Box(6, 6, 20))
        asm.revolute("j", "plate", "shaft", origin=(0, 0, 0), axis=(0, 0, 1))
        return asm

    assert shaft_in_plate(with_bushing=True).validate_warnings() == []
    (warn,) = shaft_in_plate(with_bushing=False).validate_warnings()
    assert warn[0] == "joint_off_part" and "10 mm from its parent 'plate'" in warn[1]


def test_axis_through_a_bore_less_gear_is_on_the_gear():
    """OCC's line/solid distance reports the root radius for a spur gear's own axis (its face
    classifier misses the centre of the tooth-symmetric faces); the line must still count as on-part."""
    from mech.parts import spur_gear

    asm = Assembly("gear")
    asm.part("base", Pos(0, 0, -5) * Box(40, 40, 4), ground=True)
    asm.part("gear", spur_gear(1.0, 20, 6))
    asm.revolute("j", "base", "gear", origin=(0, 0, 0), axis=(0, 0, 1))
    assert asm.validate() == [] and asm.validate_warnings() == []
    gear = asm.parts["gear"]
    assert asm._distance(gear, np.array([0.0, 0, 3]), np.array([0.0, 0, 1])) == 0.0
    assert asm._distance(gear, np.array([30.0, 0, 3]), np.array([0.0, 0, 1])) == pytest.approx(19.0, abs=0.01)


def test_validate_warnings_empty_for_invalid_model():
    asm = make_four_bar()
    asm.part("floating", Box(1, 1, 1))
    assert asm.validate() and asm.validate_warnings() == []


# ------------------------------------------------------------------------------ groups


def grouped_assembly() -> Assembly:
    asm = Assembly("groups")
    asm.part("frame", Box(100, 100, 5), ground=True)
    asm.part("base", Pos(80, 0, 10) * Box(10, 10, 10), ground=True)
    asm.part("bracket", Pos(0, 0, 5) * Box(10, 10, 5))
    asm.part("arm", Pos(20, 0, 10) * Box(40, 5, 5))
    asm.part("tool", Pos(40, 0, 15) * Box(4, 4, 4))
    asm.part("gear_a", make_disc((-30, 0, 3)))
    asm.part("gear_b", make_disc((-10, 0, 3)))
    asm.part("link1", Pos(0, 30, 5) * Box(10, 4, 4))
    asm.part("link2", Pos(10, 30, 5) * Box(10, 4, 4))
    asm.fix("bracket", "frame")
    asm.revolute("j_arm", "bracket", "arm", origin=(0, 0, 10), axis=(0, 1, 0))
    asm.fix("tool", "arm")
    asm.revolute("j_ga", "frame", "gear_a", origin=(-30, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_gb", "frame", "gear_b", origin=(-10, 0, 0), axis=(0, 0, 1))
    asm.gear("j_ga", "j_gb", -1.0)
    asm.revolute("j_l1", "frame", "link1", origin=(0, 30, 0), axis=(0, 0, 1))
    asm.revolute("j_l2", "frame", "link2", origin=(10, 30, 0), axis=(0, 0, 1))
    asm.pin("p_l", "link1", "link2", point=(5, 30, 5))
    return asm


def test_rigid_groups_is_joined_meshing_pairs():
    asm = grouped_assembly()
    assert asm.validate() == []
    g = asm.rigid_groups()
    assert g["frame"] == g["base"] == g["bracket"]  # all ground + fix
    assert g["arm"] == g["tool"] != g["frame"]
    assert len(set(g.values())) == 6
    assert asm.is_joined("frame", "base")  # same (ground) group
    assert asm.is_joined("tool", "base")  # tool~arm group is jointed to the ground group
    assert asm.is_joined("link1", "link2")  # pin
    assert not asm.is_joined("tool", "gear_a") and not asm.is_joined("gear_a", "gear_b")
    assert asm.meshing_pairs() == {frozenset({"gear_a", "gear_b"})}
    assert asm.parent_joint("tool").name == "fix_tool" and asm.parent_joint("frame") is None


def test_meshing_pairs_follow_a_gear_fixed_to_its_shaft():
    """The driven joint's child is a shaft; the gear fixed to it is what meshes with the pinion."""
    asm = Assembly("train")
    asm.part("frame", Pos(0, 0, -5) * Box(100, 40, 4), ground=True)
    asm.part("pinion", make_disc((0, 0, 0), r=8.0))
    asm.part("shaft", Pos(23, 0, 15) * Box(4, 4, 30))
    asm.part("gear", make_disc((23, 0, 0), r=15.5))
    asm.revolute("j_in", "frame", "pinion", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_out", "frame", "shaft", origin=(23, 0, 0), axis=(0, 0, 1))
    asm.fix("gear", "shaft")
    asm.gear("j_in", "j_out", -0.5)
    assert asm.validate() == []
    assert asm.meshing_pairs() == {frozenset({"pinion", "gear"})}
