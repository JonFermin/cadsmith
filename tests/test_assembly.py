"""mech.assembly: builders, validation (all errors, did-you-mean), warnings, groups, couplings."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Axis, Box, Cylinder, Location, Plane, Pos, Rectangle, Rot, Sphere

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
    # A hinge pin whose point is in the open, far up its own axis, is a reference-plane offset: the
    # pin starts at the nearest material (the rocker, 84 mm down) and threads the coupler below it ...
    asm.pin("p_hinge_high", "coupler", "rocker", point=B + [0, 0, 100], axis=(0, 0, 1))
    # ... a ball pin at that point is 84 mm above the coupler (top at z = 10.5, hole radius 2).
    asm.pin("p_ball_high", "coupler", "rocker", point=B + [0, 0, 100])
    errors = asm.validate()
    assert not names_in(errors, "p_hinge_high")
    d_coupler = math.hypot(100 - 10.5, 2.0)
    assert names_in(errors, "pin_off_part", "'p_ball_high'", f"{d_coupler:.3g} mm from part 'coupler'")
    assert len([e for e in errors if "p_ball_high" in e]) == 2  # coupler and rocker
    assert len(errors) == 2


def test_ball_pin_buried_inside_a_thick_part_is_on_part():
    asm = Assembly("buried")
    asm.part("block", Box(40, 40, 40), ground=True)
    asm.part("arm", Pos(0, 0, 25) * Box(10, 10, 10))
    asm.revolute("j", "block", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.pin("p", "block", "arm", point=(0, 0, 18))  # 2 mm inside the block top, 2 mm below the arm
    assert asm.validate() == []


def socket_model(*, dx: float = 0.0, cut: bool = False, name: str = "stud") -> Assembly:
    """A rod-end socket (a ring of bore radius 6 around a ø10 ball stud fixed to a base) closed by
    a ball pin at the ball center; the ring is ``dx`` mm off the ball, or cut open (a C)."""
    asm = Assembly("socket", pin_tol=5.0)
    asm.part("base", Pos(0, 0, -20) * Box(90, 20, 6), ground=True)
    asm.part("stud", Sphere(5) + Pos(0, 0, -11) * Cylinder(1.5, 12))  # ball + neck to the base top
    asm.fix("stud", "base")
    eye = Cylinder(8, 6) - Cylinder(6, 8) + Pos(25, 0, 0) * Box(36, 6, 6)  # ring around Z + arm
    if cut:
        eye -= Pos(-7, 0, 0) * Box(4, 1, 10)
    asm.part("rod", Pos(dx, 0, 0) * eye)
    asm.revolute("j", "base", "rod", origin=(40 + dx, 0, 0), axis=(0, 1, 0))
    asm.pin("p", "rod", name, point=(0, 0, 0))
    return asm


def test_ball_pin_socket_ring_holds_the_point_whatever_its_bore():
    """A socket ring around the ball holds a ball pin's point although its bore (radius 6) is
    beyond pin_tol 5 (geom.surface_axes finds the ring's axis, axis_cover that it surrounds the
    point); a ring cut open, or pulled off the ball, does not. Naming a part fixed to the ball but
    away from the point hints at the part that is there."""
    assert socket_model().validate() == []
    errors = socket_model(cut=True).validate()
    assert names_in(errors, "pin_off_part", "'p'", "6 mm from part 'rod'") and len(errors) == 1
    errors = socket_model(dx=20.0).validate()
    assert names_in(errors, "pin_off_part", "'p'", "from part 'rod'") and len(errors) == 1
    errors = socket_model(name="base").validate()  # the base top is 17 mm below the ball center
    assert names_in(errors, "pin_off_part", "17 mm from part 'base'", "'stud', fixed to it, is at the point")


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
    """A shaft whose revolute parent is a plate with a big open notch, carried by a bushing fixed to it."""
    def shaft_in_plate(with_bushing: bool) -> Assembly:
        asm = Assembly("shaft")
        asm.part("plate", Box(60, 60, 5) - Pos(0, 20, 0) * Box(20, 60, 10), ground=True)  # 10 mm from the axis
        if with_bushing:
            asm.part("bushing", Box(20, 20, 5) - Box(6, 6, 10))
            asm.fix("bushing", "plate")
        asm.part("shaft", Pos(0, 0, 5) * Box(6, 6, 20))
        asm.revolute("j", "plate", "shaft", origin=(0, 0, 0), axis=(0, 0, 1))
        return asm

    assert shaft_in_plate(with_bushing=True).validate_warnings() == []
    (warn,) = shaft_in_plate(with_bushing=False).validate_warnings()
    assert warn[0] == "joint_off_part" and "10 mm from its parent 'plate'" in warn[1]


def test_a_part_around_the_axis_is_on_it_whatever_the_bore():
    """A3: a plate whose closed hole the shaft passes through carries it, however big the hole
    (the open notch above does not); likewise a link eye bigger than pin_tol on its pin."""
    asm = Assembly("bore")
    asm.part("plate", Box(60, 60, 5) - Box(20, 20, 10), ground=True)  # closed hole, 10 mm from the axis
    asm.part("shaft", Pos(0, 0, 5) * Box(6, 6, 20))
    asm.revolute("j", "plate", "shaft", origin=(0, 0, 0), axis=(0, 0, 1))
    assert asm.validate() == [] and asm.validate_warnings() == []

    eye = Assembly("eye", pin_tol=5.0)
    eye.part("post", Pos(0, 0, -10) * Box(30, 30, 8) + Pos(0, 0, 10) * Cylinder(6.0, 36), ground=True)  # ø12 pin
    eye.part("link", Pos(25, 0, 10) * Box(70, 24, 6) - Pos(0, 0, 10) * Cylinder(6.4, 10))  # ø12.8 eye
    eye.part("arm", Pos(25, 0, 20) * Box(70, 24, 6) - Pos(0, 0, 20) * Cylinder(6.4, 10))
    eye.revolute("j_link", "post", "link", origin=(0, 0, 10), axis=(0, 0, 1))
    eye.revolute("j_arm", "post", "arm", origin=(0, 0, 20), axis=(0, 0, 1))
    eye.pin("p", "link", "arm", point=(0, 0, 15), axis=(0, 0, 1))  # 6.4 mm from the axis line
    assert eye.validate() == [] and eye.validate_warnings() == []
    # the eyes carry the pin: they run on it 0.4 mm apart, so the pair is joined (clearance exempt)
    assert eye.is_joined("link", "post") and eye.is_joined("arm", "post")
    # an eye cut open (a C) is not around the pin: 6.4 mm off it, beyond pin_tol
    eye.parts["arm"].shape = eye.parts["arm"].shape - Pos(-8, 0, 20) * Box(6, 2, 10)
    errors = eye.validate()
    assert names_in(errors, "pin_off_part", "'p'", "6.4 mm from part 'arm'") and len(errors) == 1


def test_hinge_attachment_is_local_to_the_hinge():
    """A2 (strandbeest): a pivot stub per station on one axis line. A link hinged at a station whose
    stub is missing is 'on' the infinite line only through the other station's stub, 40 mm along it."""
    def stations(stub_here: bool) -> Assembly:
        asm = Assembly("stubs")
        asm.part("frame", [Pos(0, 0, -20) * Box(20, 120, 6), Pos(30, 0, -8) * Box(6, 20, 26)], ground=True)
        stubs = [Pos(0, 40, 0) * Rot(90, 0, 0) * Cylinder(2, 12)]  # the other station, y 34…46
        if stub_here:
            stubs.append(Pos(0, 0, 0) * Rot(90, 0, 0) * Cylinder(2, 12))  # this station, y −6…6
        asm.part("pivots", stubs, ground=True)
        asm.part("link", Pos(15, 0, 0) * Box(40, 3, 8) - Rot(90, 0, 0) * Cylinder(2.2, 10))
        asm.part("rod", Pos(15, -3.5, 0) * Box(40, 3, 8) - Pos(0, -3.5, 0) * Rot(90, 0, 0) * Cylinder(2.2, 10))
        asm.revolute("j_link", "pivots", "link", origin=(0, 0, 0), axis=(0, 1, 0))
        asm.revolute("j_rod", "frame", "rod", origin=(30, -3.5, 0), axis=(0, 1, 0))
        asm.pin("p", "pivots", "rod", point=(0, -3.5, 0), axis=(0, 1, 0))
        return asm

    good = stations(stub_here=True)
    assert good.validate() == [] and good.validate_warnings() == []
    bad = stations(stub_here=False)
    errors = bad.validate()
    # rod (3 mm) at y −5…−2, the other station's stub at y 34…46: 36 mm of axis between them
    assert names_in(errors, "pin_off_part", "'p'", "parts 'pivots' and 'rod' meet its axis line only 36 mm apart",
                    "at most 12 mm")
    assert len(errors) == 1
    bad.pins.clear()
    (warn,) = bad.validate_warnings()
    assert warn[0] == "joint_off_part" and "'j_link'" in warn[1] and "parent 'pivots'" in warn[1]
    assert "meet its axis line only 32.5 mm apart" in warn[1]
    # the infinite axis line alone would have passed both (the old rule)
    assert bad._distance(bad.parts["pivots"], np.array([0.0, 0, 0]), np.array([0.0, 1, 0])) == 0.0


def test_hinge_sides_may_sit_away_from_the_drawn_point_but_must_meet():
    """A hinge's point may sit in the drawing plane under its stacked links (four-bar: rocker 2t+1
    up, frame 1.5t down, nothing between — at any t), and a clevis may straddle a lug: the two sides
    meet along the axis. A rocker lifted 100 mm off the frame and coupler does not."""
    for t in (5.0, 15.0):
        asm = make_four_bar(t=t)
        assert asm.validate() == [] and asm.validate_warnings() == [], t
    clevis = Assembly("clevis")
    clevis.part("lug", Pos(0, 0, -30) * Box(20, 60, 50) - Rot(90, 0, 0) * Cylinder(3, 70), ground=True)  # y ±30
    plates = [Pos(15, y, 0) * Box(50, 6, 16) - Pos(0, y, 0) * Rot(90, 0, 0) * Cylinder(3, 8) for y in (-34, 34)]
    clevis.part("fork", plates)  # plates 31…37 mm either side of the point, 1 mm off the lug
    clevis.revolute("j", "lug", "fork", origin=(0, 0, 0), axis=(0, 1, 0))
    assert clevis.validate() == [] and clevis.validate_warnings() == []
    lifted = make_four_bar()
    lifted.parts["rocker"].shape = Pos(0, 0, 100) * lifted.parts["rocker"].shape  # z 111…116
    errors = lifted.validate()
    assert names_in(errors, "pin_off_part", "'p_B'", "'coupler' and 'rocker'", "100 mm apart", "at most 20 mm")
    lifted.pins.clear()
    warns = [m for c, m in lifted.validate_warnings() if c == "joint_off_part"]
    assert len(warns) == 1 and "'j_rocker'" in warns[0] and "118 mm apart" in warns[0]
    clevis = Assembly("clevis")
    clevis.part("lug", Pos(0, 0, -30) * Box(20, 60, 50) - Rot(90, 0, 0) * Cylinder(3, 70), ground=True)  # y ±30
    plates = [Pos(15, y, 0) * Box(50, 6, 16) - Pos(0, y, 0) * Rot(90, 0, 0) * Cylinder(3, 8) for y in (-34, 34)]
    clevis.part("fork", plates)  # plates 31…37 mm either side of the point, 1 mm off the lug
    clevis.revolute("j", "lug", "fork", origin=(0, 0, 0), axis=(0, 1, 0))
    assert clevis.validate() == [] and clevis.validate_warnings() == []


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
    assert asm.is_joined("arm", "bracket")  # the parts j_arm names
    # only the parts that carry j_arm are joined across the two bodies: the tool (fixed to the arm
    # 40 mm from the hinge), the distant ground block and the frame (7.5 mm below the axis) keep clearance
    assert not asm.is_joined("tool", "base") and not asm.is_joined("tool", "bracket")
    assert not asm.is_joined("arm", "frame") and not asm.is_joined("arm", "base")
    assert asm.is_joined("link1", "link2")  # pin
    assert not asm.is_joined("tool", "gear_a") and not asm.is_joined("gear_a", "gear_b")
    assert asm.meshing_pairs() == {frozenset({"gear_a", "gear_b"})}
    assert asm.parent_joint("tool").name == "fix_tool" and asm.parent_joint("frame") is None


def blade_and_post(post: str = "ground", gap: float = 0.05) -> Assembly:
    """A blade hinged to a ground base swings past a post (review p2_ground_joined)."""
    asm = Assembly("groundjoined", clearance=0.3)
    asm.part("base", Pos(0, 0, -10) * Box(120, 120, 4), ground=True)
    asm.part("post", Pos(40, 2.0 + gap, -0.95) * Box(2, 2, 13.9), ground=post == "ground")
    if post == "fixed":
        asm.fix("post", "base")  # bolted to the base: same rigid body as the ground
    asm.part("blade", Pos(25, 0, 2.5) * Box(50, 2, 5))
    asm.revolute("j", "base", "blade", origin=(0, 0, 0), axis=(0, 0, 1), limits=(-30, 0))
    return asm


@pytest.mark.parametrize("post", ["ground", "fixed"])
def test_ground_membership_does_not_spread_the_joint_exemption(post):
    asm = blade_and_post(post)
    assert asm.validate() == []
    assert asm.is_joined("blade", "base") and not asm.is_joined("blade", "post")
    assert asm.joined_pairs() == {frozenset({"base", "blade"})}
    # the named pair can be opted back into the clearance check
    asm.check_clearance("blade", "base")
    assert not asm.is_joined("blade", "base") and asm.joined_pairs() == set()


def test_joint_carrying_parts_stay_joined():
    """Bearings/shafts on the joint axis and guide pairs touching at home stay exempt."""
    asm = Assembly("carry", clearance=0.3)
    asm.part("plate", Pos(0, 0, 2.5) * Box(80, 40, 5) - Pos(-20, 0, 0) * Box(20, 20, 20), ground=True)
    asm.part("bearing", Pos(-20, 0, 2.5) * (Box(20, 20, 5) - Box(5.1, 5.1, 10)), ground=True)  # 0.05 running gap
    asm.part("shaft", Pos(-20, 0, 10) * Box(5, 5, 30))
    asm.part("hub", Pos(-20, 0, 20) * Box(12, 12, 4))  # on the shaft, above everything
    asm.part("rail", Pos(20, 0, 15) * Box(4, 4, 20), ground=True)  # a guide rod ...
    asm.part("slider", Pos(20, 0, 15) * (Box(10, 10, 6) - Box(4, 4, 10)))  # ... its bushing touches it
    asm.part("stop", Pos(35, 0, 15) * Box(4, 4, 20), ground=True)  # beside the slider, 3 mm away
    asm.fix("hub", "shaft")
    asm.revolute("j_shaft", "plate", "shaft", origin=(-20, 0, 0), axis=(0, 0, 1))
    asm.prismatic("j_slide", "plate", "slider", origin=(20, 0, 15), axis=(0, 0, 1), limits=(-5, 5))
    assert asm.validate() == []
    assert asm.is_joined("shaft", "plate")  # named
    assert asm.is_joined("bearing", "shaft") and asm.is_joined("bearing", "hub")  # all on the axis line
    assert not asm.is_joined("plate", "hub")  # the plate's hole keeps it 10 mm from the axis
    assert asm.is_joined("rail", "slider")  # touching at home: the guide
    assert not asm.is_joined("stop", "slider") and not asm.is_joined("rail", "shaft")


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


@pytest.mark.parametrize("tagged", [True, False])
def test_meshing_pairs_leave_non_gear_parts_checked(tagged):
    """A crank disc fixed to the pinion overhangs the output gear (review p12_meshing_overreach):
    only the gears mesh, the disc/gear pair is clearance-checked."""
    from build123d import Align, Cylinder

    from mech.parts import gear_pair

    gp = gear_pair(1.0, 15, 30, 6.0, bore1=5, bore2=5)
    asm = Assembly("meshreach", clearance=0.3)
    asm.part("plate", Pos(gp.center_distance / 2, 0, -3) * Box(gp.center_distance + 40, 40, 4), ground=True)
    asm.part("pinion", gp.g1 if tagged else gp.g1.shape)
    asm.part("gear_out", gp.g2 if tagged else gp.g2.shape)
    asm.part("crank", Pos(0, 0, 6.05) * Cylinder(14, 3, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    asm.fix("crank", "pinion")
    asm.revolute("j_in", "plate", "pinion", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j_out", "plate", "gear_out", origin=(gp.center_distance, 0, 0), axis=(0, 0, 1))
    asm.gear("j_in", "j_out", gp.ratio)
    assert asm.validate() == []
    assert (asm.parts["pinion"].kind == "gear") is tagged
    assert asm.meshing_pairs() == {frozenset({"pinion", "gear_out"})}


def test_belt_drive_pulleys_are_not_meshing():
    """Two pulleys coupled with gear() (a belt drive) are far apart: no meshing pair at all."""
    from mech.parts import gt2_pulley

    asm = Assembly("belt")
    asm.part("plate", Pos(30, 0, -3) * Box(100, 40, 4), ground=True)
    asm.part("p20", gt2_pulley(20, 6, 5))
    asm.part("p60", Pos(60, 0, 0) * gt2_pulley(60, 6, 5))
    asm.revolute("j1", "plate", "p20", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j2", "plate", "p60", origin=(60, 0, 0), axis=(0, 0, 1))
    asm.gear("j1", "j2", 20 / 60)
    assert asm.validate() == [] and asm.parts["p20"].kind == "pulley"
    assert asm.meshing_pairs() == set()
    # a gear coupling whose parts never engage says so (it is a belt, or the gears are out of mesh)
    (warning,) = asm.validate_warnings()
    assert warning[0] == "gear_mesh" and "gear 'gear_j2'" in warning[1] and "belt()" in warning[1]


@pytest.mark.parametrize("flip", [False, True])
def test_belt_coupling_turns_both_pulleys_the_same_way(flip):
    """belt(driver, driven, t_driver, t_driven): ratio +t_d/t_n for co-directional axes (−… when one
    axis is flipped, so the pulleys still turn the same way in space); never meshing, no WARN."""
    from mech.parts import gt2_pulley

    asm = Assembly("belt")
    asm.part("plate", Pos(30, 0, -3) * Box(100, 40, 4), ground=True)
    asm.part("p20", gt2_pulley(20, 6, 5))
    asm.part("p60", Pos(60, 0, 0) * gt2_pulley(60, 6, 5))
    asm.revolute("j1", "plate", "p20", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("j2", "plate", "p60", origin=(60, 0, 0), axis=(0, 0, -1 if flip else 1))
    name = asm.belt("j1", "j2", 20, 60)
    assert asm.validate() == [] and asm.validate_warnings() == [] and asm.meshing_pairs() == set()
    c = asm.couplings[0]
    assert (name, c.kind) == ("belt_j2", "belt") and c.ratio == pytest.approx(-1 / 3 if flip else 1 / 3)
    q = Kinematics(asm).expand({"j1": 90.0})
    assert q["j2"] == pytest.approx(-30.0 if flip else 30.0)


def test_belt_coupling_validation():
    asm = Assembly("belt")
    asm.part("plate", Box(100, 40, 4), ground=True)
    asm.part("a", Pos(0, 0, 5) * Box(10, 10, 4))
    asm.part("b", Pos(40, 0, 5) * Box(10, 10, 4))
    asm.part("s", Pos(-40, 0, 5) * Box(10, 10, 4))
    asm.revolute("ja", "plate", "a", origin=(0, 0, 0), axis=(0, 0, 1))
    asm.revolute("jb", "plate", "b", origin=(40, 0, 0), axis=(1, 0, 0))
    asm.prismatic("js", "plate", "s", origin=(-40, 0, 0), axis=(1, 0, 0))
    asm.belt("ja", "jb", 20, 40, name="skew")
    asm.belt("ja", "js", 20, 40, name="slide")
    asm.belt("jb", "ja", 0, 40, name="zero")
    errors = asm.validate()
    assert names_in(errors, "belt 'skew'", "pulley axis of 'ja'", "not parallel")
    assert names_in(errors, "belt 'slide'", "needs a revolute driver and a revolute driven joint")
    assert names_in(errors, "belt 'zero'", "t_driver and t_driven must be > 0")


def test_rot_and_angle_targets_name_parts():
    asm = make_four_bar()
    asm.target("stays", "rot:coupler", max=90)
    asm.target("relative", "angle:crank,rocker", max=180)
    assert asm.validate() == []
    asm.target("typo", "rot:couplr", max=1)
    asm.target("one", "angle:crank", max=1)
    asm.target("joint", "rot:j_crank", max=1)
    errors = asm.validate()
    assert names_in(errors, "target 'typo'", "unknown part 'couplr'", "did you mean 'coupler'")
    assert names_in(errors, "target 'one'", "'angle:crank' needs two parts: angle:<partA>,<partB>")
    assert names_in(errors, "target 'joint'", "unknown part 'j_crank'")
    assert len(errors) == 3


# ------------------------------------------------------------------------------ solids, allow_contact, check_clearance


def test_parts_without_solid_volume_are_invalid():
    """A face part, or a lid a boolean cut away entirely (review p15_nonsolid / rv_empty)."""
    asm = Assembly("empty")
    asm.part("body", Pos(0, 0, 10) * Box(40, 40, 20), ground=True)
    asm.part("lid", Pos(0, 0, 21) * Box(40, 40, 2) - Pos(0, 0, 21) * Box(60, 60, 4))  # nothing left
    asm.part("sheet", Pos(0, 0, 30) * (Plane.XY * Rectangle(10, 10)))
    asm.revolute("j_lid", "body", "lid", origin=(0, 20, 20), axis=(1, 0, 0))
    asm.revolute("j_sheet", "body", "sheet", origin=(0, 0, 30), axis=(0, 0, 1))
    errors = asm.validate()
    assert names_in(errors, "part 'lid' is not a closed solid", "boolean")
    assert names_in(errors, "part 'sheet' is not a closed solid", "no solid", "Face")
    with pytest.raises(ModelError):
        Kinematics(asm)


def test_allow_contact_depth_and_check_clearance_validation():
    asm = make_four_bar()
    asm.allow_contact("crank", "coupler")
    asm.allow_contact("coupler", "rocker", max_depth=None)
    asm.allow_contact("frame", "rocker", max_depth=-1)
    asm.check_clearance("crank", "frame")
    asm.check_clearance("coupler", "crank")  # also allowed: contradictory
    asm.check_clearance("crank", "frme")
    assert asm.allow_depth[frozenset({"crank", "coupler"})] == pytest.approx(0.1)
    assert asm.allow_depth[frozenset({"coupler", "rocker"})] is None
    errors = asm.validate()
    assert names_in(errors, "allow_contact('frame', 'rocker')", "max_depth must be a number >= 0")
    assert names_in(errors, "check_clearance('coupler', 'crank') contradicts allow_contact")
    assert names_in(errors, "check_clearance: unknown part 'frme'", "did you mean 'frame'")
    assert len(errors) == 3


def test_prismatic_axis_line_position_is_not_linted():
    """A prismatic joint's axis line may run anywhere (review lift_prism_origin): no joint_off_part."""
    asm = Assembly("slide")
    asm.part("base", Pos(0, 0, -5) * Box(100, 20, 10), ground=True)
    asm.part("slider", Pos(0, 20, 10) * Box(10, 10, 10))
    asm.prismatic("j", "base", "slider", origin=(0, 0, -5), axis=(1, 0, 0), limits=(-20, 20))
    assert asm.validate() == [] and asm.validate_warnings() == []
    asm.joints["j"].kind = "revolute"  # the same line as a hinge axis is 15 mm off the slider
    assert [c for c, _ in asm.validate_warnings()] == ["joint_off_part"]


# ------------------------------------------------------------------------------ mesh / joined / fasten / ball


def ring_and_planet() -> Assembly:
    """A planet gear on a carrier inside a FIXED internal ring: no gear() coupling can name the
    ring/planet mesh (the ring is ground, not a revolute child) — mesh() declares it."""
    from mech.parts import internal_gear, spur_gear

    asm = Assembly("ringmesh")
    asm.part("ring", internal_gear(1.0, 48, 6, backlash=0.1, pinion=18), ground=True)
    asm.part("carrier", Pos(0, 0, -4) * Box(40, 8, 3))
    asm.part("planet", Pos(15, 0, 0) * spur_gear(1.0, 18, 6, bore=4, backlash=0.1))
    asm.revolute("j_c", "ring", "carrier", origin=(0, 0, -3), axis=(0, 0, 1))
    asm.revolute("j_p", "carrier", "planet", origin=(15, 0, 0), axis=(0, 0, 1))
    return asm


def test_mesh_declares_a_meshing_pair_the_couplings_cannot():
    asm = ring_and_planet()
    assert asm.meshing_pairs() == set()
    asm.mesh("ring", "planet")
    assert asm.validate() == [] and asm.validate_warnings() == []
    assert asm.meshing_pairs() == {frozenset({"ring", "planet"})}
    assert not asm.is_joined("ring", "planet")  # meshing, not joined: interference still checked
    # misspelt, self and contradictory declarations are errors
    asm.mesh("rng", "planet")
    asm.mesh("planet", "planet")
    asm.check_clearance("ring", "planet")
    errors = asm.validate()
    assert names_in(errors, "mesh: unknown part 'rng'", "did you mean 'ring'")
    assert names_in(errors, "mesh('planet', 'planet'): needs two different parts")
    assert names_in(errors, "check_clearance('planet', 'ring') contradicts mesh")
    assert len(errors) == 3


def test_mesh_pair_apart_at_home_warns():
    asm = ring_and_planet()
    asm.part("far", Pos(200, 0, 0) * Box(10, 10, 6))
    asm.fix("far", "carrier")
    asm.mesh("ring", "far")
    (warn,) = asm.validate_warnings()
    assert warn[0] == "gear_mesh" and "mesh('far', 'ring')" in warn[1] and "apart at home" in warn[1]


def test_joined_and_fasten_flow_through_joined_pairs():
    asm = make_four_bar()
    asm.part("screw", Pos(100, 0, -8) * Cylinder(1.5, 12), material="steel")  # into the frame's far end
    asm.fix("screw", "rocker")
    assert not asm.is_joined("crank", "rocker")
    asm.joined("coupler", "frame")
    asm.fasten("screw", "frame")
    assert asm.validate() == []
    pairs = asm.joined_pairs()
    assert {frozenset({"coupler", "frame"}), frozenset({"frame", "screw"})} <= pairs
    assert asm.is_joined("frame", "coupler") and asm.is_joined("screw", "frame")
    key = frozenset({"frame", "screw"})
    assert key in asm.allowed and asm.allow_depth[key] is None  # thread overlap: never interference
    # check_clearance undoes a joined pair, which is a contradiction in one model
    asm.check_clearance("frame", "coupler")
    assert names_in(asm.validate(), "check_clearance('coupler', 'frame') contradicts joined")


def test_joined_pair_far_apart_warns_and_unknown_names_suggest():
    asm = make_four_bar()
    asm.joined("crank", "rocker")  # the crank is 60 mm from the rocker at home: nothing joins them
    warns = [m for c, m in asm.validate_warnings() if c == "joint_off_part"]
    assert len(warns) == 1 and "joined('crank', 'rocker')" in warns[0] and "nothing joins them" in warns[0]
    asm.joined("crank", "rokcer")
    asm.fasten("scerw", "frame")
    errors = asm.validate()
    assert names_in(errors, "joined: unknown part 'rokcer'", "did you mean 'rocker'")
    assert names_in(errors, "joined: unknown part 'scerw'")
    assert names_in(errors, "allow_contact: unknown part 'scerw'")


def test_touching_uses_the_tessellations_and_stays_exact(monkeypatch):
    """B3: joined_pairs' touch test on toothed parts runs the exact distance only on the faces the
    mesh proximity finds close — never on two whole gears (seconds each) — and still finds touches."""
    import mech.assembly as A
    from mech.parts import gear_pair
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    def faces(s) -> int:
        n, exp = 0, TopExp_Explorer(s, TopAbs_FACE)
        while exp.More():
            n += 1
            exp.Next()
        return n

    sizes = []
    real = A.BRepExtrema_DistShapeShape

    def counting(sa, sb, *args):
        sizes.append((faces(sa), faces(sb)))
        return real(sa, sb, *args)

    monkeypatch.setattr(A, "BRepExtrema_DistShapeShape", counting)
    gp = gear_pair(1.0, 20, 30, 6, backlash=0.1)
    asm = Assembly("touch")
    asm.part("g1", gp.g1, ground=True)
    asm.part("g2", gp.g2, ground=True)
    asm.part("g3", Pos(0, 0, 6) * gp.g1)  # stacked on g1: touching face to face
    assert not asm._touching("g1", "g2")  # 0.05·cos 20° apart at the flanks
    assert asm._touching("g1", "g3")
    n_faces = faces(gp.g1.shape.wrapped)
    assert sizes and all(a < n_faces and b < n_faces for a, b in sizes)


def ball_four_bar() -> tuple[Assembly, dict]:
    """Spatial RSSR: crank about Z, rocker about X, coupler with a ball() at the crank and a ball pin
    at the rocker. Closed form: |O2 + r2·(0, cos φ, sin φ) − A(θ)| = L."""
    from build123d import Solid, Sphere

    def bar(p, q, r=2.0):
        p, q = np.asarray(p, float), np.asarray(q, float)
        cyl = Solid.make_cylinder(r, float(np.linalg.norm(q - p)), Plane(origin=tuple(p), z_dir=tuple(q - p)))
        return cyl.fuse(Pos(*p) * Sphere(1.6 * r)).fuse(Pos(*q) * Sphere(1.6 * r))

    g = dict(r1=20.0, r2=40.0, L=78.0, O2=np.array([50.0, 0.0, 40.0]))
    A0 = np.array([g["r1"], 0.0, 0.0])
    D = g["O2"] - A0
    K = (g["L"] ** 2 - D @ D - g["r2"] ** 2) / (2 * g["r2"])
    g["phi0"] = math.atan2(D[2], D[1]) + math.acos(K / math.hypot(D[1], D[2]))
    B0 = g["O2"] + g["r2"] * np.array([0.0, math.cos(g["phi0"]), math.sin(g["phi0"])])
    asm = Assembly("rssr")
    asm.part("frame", [Pos(0, 0, -8) * Box(20, 20, 6), Pos(*g["O2"]) * Box(10, 12, 12)], ground=True)
    asm.part("crank", bar((0, 0, 0), A0))
    asm.part("coupler", bar(A0, B0))
    asm.part("rocker", bar(g["O2"], B0))
    asm.revolute("j_crank", "frame", "crank", origin=(0, 0, 0), axis=(0, 0, 1))
    assert asm.ball("s_A", "crank", "coupler", A0) == "s_A"
    asm.revolute("j_rocker", "frame", "rocker", origin=g["O2"], axis=(1, 0, 0))
    asm.pin("s_B", "coupler", "rocker", B0)
    return asm, g


def test_ball_joint_is_an_exact_spherical_joint():
    """E4: ball() = three revolutes through the center via two virtual knuckles; the spatial RSSR
    follows its closed form exactly over a full crank turn."""
    from mech import part_props
    from mech.assembly import Study
    from mech.motion import run_study

    asm, g = ball_four_bar()
    assert asm.validate() == [] and asm.validate_warnings() == []
    assert [n for n in asm.joints if n.startswith("s_A")] == ["s_A_1", "s_A_2", "s_A_3"]
    knuckles = [n for n, p in asm.parts.items() if p.virtual]
    assert knuckles == ["s_A_k1", "s_A_k2"]
    assert all(part_props(asm.parts[k]).mass_kg == 0.0 for k in knuckles)
    axes = np.array([asm.joints[f"s_A_{i}"].axis for i in (1, 2, 3)])
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)  # orthonormal
    np.testing.assert_allclose(axes[2], (asm.pins[0].point - asm.joints["s_A_1"].origin) / g["L"], atol=1e-9)
    assert asm.is_joined("crank", "coupler")  # the ball's two sides carry each other
    kin = Kinematics(asm)
    res = run_study(asm, kin, Study("turn", {"j_crank": (0, 360)}, frames=25))
    assert all(p.ok for p in res.poses) and max(p.residual for p in res.poses) < 1e-9
    prev = g["phi0"]
    for theta, pose in zip(res.drive["j_crank"], res.poses):
        A = g["r1"] * np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta)), 0.0])
        D = g["O2"] - A
        K = (g["L"] ** 2 - D @ D - g["r2"] ** 2) / (2 * g["r2"])
        base, w = math.atan2(D[2], D[1]), math.acos(K / math.hypot(D[1], D[2]))
        phi = min((base + w, base - w), key=lambda c: abs((c - prev + math.pi) % (2 * math.pi) - math.pi))
        prev = phi
        got = g["phi0"] + math.radians(pose.q["j_rocker"])
        assert abs((got - phi + math.pi) % (2 * math.pi) - math.pi) < math.radians(1e-6)


def test_ball_center_off_its_parts_warns():
    asm, _ = ball_four_bar()
    parent, child, _c = asm.balls["s_A"]
    asm.balls["s_A"] = (parent, child, np.array([20.0, 0.0, 30.0]))  # 30 mm above the crank end
    warns = [m for c, m in asm.validate_warnings() if c == "joint_off_part"]
    assert any("ball 's_A' center" in m and "from its parent 'crank'" in m for m in warns)


# ------------------------------------------------------------------------------ review round 3


def ball_in_socket(rb: float = 10.0, play: float = 0.2, dx: float = 0.0) -> Assembly:
    """A ball-and-socket through ball(): a ground socket (a block minus a sphere of radius
    rb + play, open below) around an arm's ball of radius rb, the socket ``dx`` mm off it."""
    asm = Assembly("ball_socket", clearance=0.3)
    sock = Pos(0, 0, 6) * Box(40, 40, 30) - Sphere(rb + play) - Pos(0, 0, -12) * Cylinder(rb * 0.8, 20)
    asm.part("socket", Pos(dx, 0, 0) * sock, ground=True)
    asm.part("arm", Sphere(rb) + Pos(0, 0, -50) * Cylinder(3, 100), material="steel")
    asm.ball("b", "socket", "arm", center=(0, 0, 0))
    return asm


@pytest.mark.parametrize("rb, play", [(10.0, 0.2), (5.0, 0.2), (10.0, 3.0)])
def test_a_socket_around_the_ball_holds_its_center(rb, play):
    """ball()'s center check follows the ball-pin rule: a part surrounding the center holds it,
    whatever the socket's radius (here beyond pin_tol 5) — also a spherical cavity whose other
    faces are all far from the center (geom.surface_axes lists the sphere's axis)."""
    asm = ball_in_socket(rb, play)
    assert asm.validate() == [] and asm.validate_warnings() == []
    warns = [m for c, m in ball_in_socket(rb, play, dx=40.0).validate_warnings() if c == "joint_off_part"]
    assert any("ball 'b' center" in m and "from its parent 'socket'" in m for m in warns)


def test_mesh_pair_whose_teeth_miss_warns_though_the_boxes_overlap():
    """A planet drawn 4 mm inside its mesh radius: its box lies inside the ring's (as every
    planet's does), but its teeth are 2.24 mm from the ring's — a gear_mesh WARN with the gap."""
    from mech.parts import internal_gear, spur_gear

    def model(shift: float) -> Assembly:
        a = 15.0 - shift
        asm = Assembly("ring_apart", clearance=0.3)
        asm.part("ring", internal_gear(1.0, 48, 6, pinion=18), ground=True)
        asm.part("arm", Pos(a / 2, 0, 8) * Box(a + 10, 8, 3))
        asm.revolute("j_arm", "ring", "arm", origin=(0, 0, 0), axis=(0, 0, 1))
        asm.part("planet", Pos(a, 0, 0) * spur_gear(1.0, 18, 6, bore=4))
        asm.revolute("j_p", "arm", "planet", origin=(a, 0, 0), axis=(0, 0, 1))
        asm.couple("j_arm", "j_p", -30 / 18)
        asm.mesh("ring", "planet")
        return asm

    assert [c for c, _ in model(0.0).validate_warnings()] == []
    (warn,) = model(4.0).validate_warnings()
    assert warn[0] == "gear_mesh" and "mesh('planet', 'ring')" in warn[1] and "2.24 mm apart at home" in warn[1]


def test_fasten_across_bodies_that_move_relative_to_each_other_warns():
    """fasten() = joined + allow_contact(max_depth=None): the pair is never checked at all. A
    screw on the ground fastened to an arm hinged to it is a contradiction (the screw would turn
    with what it threads into) — joint_off_part names the joint; fastened to its own body, fine."""
    from mech.parts import socket_head_screw

    def model(screw_ground: bool) -> Assembly:
        asm = Assembly("fasten_moving", clearance=0.3)
        asm.part("base", Pos(0, 0, -5) * Box(120, 30, 10), ground=True)
        asm.part("arm", Pos(25, 0, 3) * Box(70, 12, 6) - Pos(40, 0, 3) * Cylinder(1.25, 8))
        asm.revolute("j", "base", "arm", origin=(0, 0, 3), axis=(0, 0, 1))
        asm.part("screw", Pos(40, 0, 6) * socket_head_screw("M3", 6), ground=screw_ground)
        if not screw_ground:
            asm.fix("screw", "arm")
        asm.fasten("screw", "arm")
        return asm

    warns = [m for c, m in model(screw_ground=True).validate_warnings() if c == "joint_off_part"]
    assert len(warns) == 1 and "fasten('arm', 'screw')" in warns[0] and "through 'j'" in warns[0]
    assert "fix() the fastener" in warns[0]
    assert model(screw_ground=False).validate_warnings() == []
