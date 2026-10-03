# mech API cheat-sheet

Full contract: `docs/MECH_SPEC.md`. Templates: `examples/*.py` (all PASS except `hinged_box`,
whose default `gap=0.2` is a deliberate `tight` WARN): `four_bar`, `slider_crank`, `gear_train`,
`leadscrew_stage`, `pendulum_arm`, `hinged_box`, `parallel_gripper` (SG90 + gear pair +
parallelogram jaws, callable target on per-frame transforms), `scissor_lift` (NEMA17 + T8 screw,
three loops, payload, motor SF). Larger, hand-checked designs in `examples/showcase/` (all PASS,
150–270 lines; read the one closest to the task before writing anything big): `strandbeest`
(18 Jansen loops, multi-layer links, `fasten` screws), `radial_engine` (master + articulated rods,
pistons in barrels), `planetary` (`internal_gear` + `mesh`, couplings, reflected loads),
`desktop_arm` (belt drives, three parallelograms, `polygon` plates, `joined`), `excavator` (four
hydraulic cylinders = barrel + rod on a prismatic, bucket four-bar), `delta_robot` (spatial
loops, ball pins in socket rings, IK-driven studies, callable targets).

## Canonical script (`examples/four_bar.py`)

```python
"""Crank-rocker four-bar linkage."""
from build123d import *
from mech import *
from mech.geom import link, circle_intersect

def build(ground=100.0, crank=40.0, coupler=90.0, rocker=80.0, t=5.0) -> Assembly:
    asm = Assembly("four_bar", clearance=0.3)
    O2, O4 = (0, 0, 0), (ground, 0, 0)
    A = (crank, 0, 0)                                          # crank drawn at 0°
    B = circle_intersect(A, coupler, O4, rocker, side=+1)      # closes the loop in the drawing
    asm.part("frame", Pos(ground / 2, 0, -2 * t) * Box(ground + 20, 16, t), ground=True, color="#666666")
    asm.part("crank", link(O2, A, width=10, thickness=t, z=0), material="aluminum_6061")
    asm.part("coupler", link(A, B, width=10, thickness=t, z=t + 0.5), material="aluminum_6061")
    asm.part("rocker", link(O4, B, width=10, thickness=t, z=2 * t + 1), material="aluminum_6061")
    asm.revolute("j_crank", "frame", "crank", origin=O2, axis=(0, 0, 1))          # no limits: full turn
    asm.revolute("j_coupler", "crank", "coupler", origin=A, axis=(0, 0, 1))
    asm.revolute("j_rocker", "frame", "rocker", origin=O4, axis=(0, 0, 1))
    asm.pin("p_B", "coupler", "rocker", point=B, axis=(0, 0, 1))                 # closes the loop
    asm.probe("mid", part="coupler", point=[(a + b) / 2 for a, b in zip(A, B)])
    asm.actuator("j_crank", capacity=0.5)
    asm.study("turn", drive={"j_crank": (0, 360)}, frames=72)
    asm.target("rocker swing", "span:j_rocker", min=40)
    return asm

if __name__ == "__main__":
    run(build())
```

## Conventions

| Quantity | Unit | Notes |
|---|---|---|
| length | mm | Z is up; gravity `(0, 0, −9.80665)` m/s² |
| revolute value | **degrees** | +θ = right-hand rule about `axis` |
| prismatic value | mm | + = along `axis` |
| mass | g | density g/cm³ |
| torque / force | N·m / N | loads = what the actuator must supply to hold (+ pushes +q) |
| time | s | study `duration` |

- **Parts are modeled in place**: world coordinates at the **home pose** (the drawing). Joints
  are declared in world coordinates at home (`origin` point + `axis`). No local frames.
- **home** = the joint value of the drawn pose (default 0). Limits use the same coordinate
  (a slider drawn at x=120 with its value = x needs `home=120`).
- Transform per part maps home-world → current-world; ground parts never move.
- Couplings act on deltas from home: `q_driven − home = ratio·(q_driver − home) + offset`.

## Assembly calls (points/axes: tuple, list, ndarray or build123d `Vector`)

| Call | Meaning |
|---|---|
| `Assembly(name, *, clearance=0.3, interference_tol=0.5, pin_tol=5.0, gravity=(0,0,-9.80665))` | `clearance` = min gap (mm) between parts that aren't joined; `interference_tol` mm³ |
| `part(name, shape, *, material="PLA", color=None, ground=False, mass_g=None, density=None, opacity=1.0, bom=None)` | shape: build123d Shape, list of Shapes, or `LibPart` (its bom/mass_g become defaults). `mass_g` overrides mass (inertia scaled) |
| `revolute(name, parent, child, origin=None, axis=None, *, at=None, limits=None, home=0.0)` | hinge; `at=` a `Location`/`Axis` (e.g. `motor.frames["shaft"]`) gives origin + Z axis |
| `prismatic(name, parent, child, origin=None, axis=None, *, at=None, limits=None, home=0.0)` | slide along `axis` |
| `fix(child, parent, name=None)` | rigid attach (default name `fix_<child>`) |
| `gear(driver, driven, ratio)` | rev→rev meshing gears; ratio for co-directional axes (external mesh negative: `gp.ratio`) |
| `belt(driver, driven, t_driver, t_driven)` | rev→rev belt/chain between parallel pulleys: both turn the same way, ratio `t_driver / t_driven` (teeth or pitch ø; sign from the axes); never meshing |
| `screw(driver, driven, lead, *, hand="right")` | rev→prismatic; right hand, co-directional axes: **+360° → −lead** |
| `rack(driver, driven, pitch_radius, *, sign=None)` | rev→prismatic, ratio ±π·r/180 mm/deg (sign from geometry) |
| `couple(driver, driven, ratio=1.0, offset=0.0)` | generic linear coupling, native units |
| `pin(name, a, b, point, axis=None)` | loop closure: `point` stays coincident on a and b; `axis` = hinge pin (axes stay aligned), none = ball |
| `probe(name, part, point)` | track a point (home world) fixed to `part` |
| `actuator(joint, capacity)` | N·m (revolute) / N (prismatic); also the preferred driver of a loop |
| `allow_contact(a, b, *, max_depth=0.1)` / `ignore(a, b)` | may touch and overlap as deep as max_depth mm (mean depth 2V/A) → `contact`; deeper → `interference`; `max_depth=None`: any overlap, never checked (a modelled belt around toothed pulleys) / skip the pair |
| `joined(a, b)` | a and b carry each other across a joint the model doesn't name (an arm and the rod beyond a stud's two revolutes, a hub on a shaft two hinges away): exempt from `clearance`, overlap still interferes |
| `mesh(a, b)` | a and b mesh (a planet in a **fixed** ring, which `gear()` can't name): contact allowed, any overlap interferes |
| `fasten(screw, part)` | a screw in a tapped hole drawn at tap-drill size, a press-fit pin: `joined` + `allow_contact(max_depth=None)` — the pair is never checked, so the two must stay rigid together (`fix()` the screw to the part it threads into; on bodies that move relative to each other it is a `joint_off_part` WARN) |
| `ball(name, parent, child, center, *, axis=None, limits=None)` | spherical joint = revolutes `<name>_1.._3` through `center` via two virtual knuckles (no mass/clearance/export); drive or pin them like any revolutes; `axis` = the child's spin axis (default center → child's bbox center; gimbal lock only 90° off it); a socket around the ball holds the center whatever its radius |
| `check_clearance(a, b)` | hold a joined pair (an arm and the frame it is hinged to) to `clearance` anyway; on a hinged/pinned pair the bore region around the pin stays exempt (an eye of any bore on its pin), the faces next to it are checked |
| `study(name, drive, *, frames=60, loop="once", duration=3.0)` | `drive = {joint: (start, end) \| [(u, v), ...] \| callable(u) \| number}`, u ∈ [0,1]; `loop="pingpong"` goes there and back |
| `target(label, metric, *, min=None, max=None, study=None, severity="FAIL")` | requirement; `severity="WARN"` for soft goals |

Rules: every part is `ground=True` or the child of exactly one joint; close loops with `pin`;
any non-coupled, non-fixed joint may be driven, including a loop joint (the rest of the loop is
solved); never drive a coupled joint; don't drive more joints of a loop than it has DOF (INVALID
"overconstrained"). A part is *on* a hinge pin/revolute when its material comes within `pin_tol`
of the axis line **or surrounds it** (a bearing, an eye — any bore size; never raise `pin_tol`
for a big bore), and the two sides must meet along the line near each other (≤ 2·pin_tol, or 4×
the thinner part, apart) — material of a part far along the same line doesn't count. A ball pin's
part holds the point, buries it, or surrounds it (a socket ring around the ball); name the ball
stud, not the platform 10 mm behind it (the error says which part is at the point).

**Default studies** (no `study` declared): each serial joint sweeps its limits over 40 frames
(revolute without limits 0→360°, prismatic without limits: no study); each loop is driven at its
actuator (else a joint with limits, else its first revolute) over 60 frames; a driver's range is
narrowed to the limits of the joints it drives through couplings (a lead screw turns exactly its
slide's stroke). Joints no study moves get one `held_at_home` INFO.

Frame sampling: `once` → u = k/(N−1) (ends included; `frames=73` over 0…360° gives 5° steps);
`pingpong` → N frames there and back (seamless loop in the viewer). A study whose frames all land on
one pose (2 frames over 0…360°) is INVALID.

A study is **one path** u ∈ [0, 1]: clearance is only checked along it (there is no grid/Cartesian
study). For several axes (pan + tilt, X + Y) declare one study per axis over its full range, plus a
combined raster when their interaction matters, e.g. `{"j_pan": lambda u: -170 + 340 * u,
"j_tilt": lambda u: -30 + 120 * abs((6 * u) % 2 - 1)}` (3 tilt sweeps across the pan): pick
`frames` so each tilt sweep gets ≥ 8 samples and its turnarounds land on frames (`frames = 6k + 1`);
never shrink it with `--frames` — a periodic drive aliases.

## geom helpers (`from mech.geom import ...`)

- `link(p0, p1, width, thickness, *, z=None, hole=None, normal=(0,0,1)) -> Part` — rounded bar
  between two points, bottom face at offset `z` along `normal`, end holes ø`hole` (default
  0.4·width, 0 = none). Stack links in layers (`z=0`, `t+0.5`, `2t+1`) so they can pass.
- `circle_intersect(c0, r0, c1, r1, side=+1, normal=(0,0,1)) -> (x, y, z)` — closes a linkage in
  the drawing; `side=+1` = left of c0→c1 looking down `normal`. `ValueError` if unreachable.
- `polygon(points, *, plane=Plane.XY) -> Sketch` — always counter-clockwise (normal along
  `plane.z_dir`, so it fuses with circles and extrudes the right way, whatever the point order);
  `ValueError` on < 3 points, zero area or self-intersection. Use it instead of `Polygon(...)`.
- Rarely needed: `vec3, unit, rot_about_line(origin, axis, deg), translation(v),
  transform_points(T, pts), to_location(T), from_location(loc)`.

## Parts library (`from mech.parts import ...`)

Each returns a `LibPart(shape, bom, mass_g, frames)`; place with `Pos(...) * Rot(...) * part`
or `part.mate("frame", target_location)`; `frames` are world `Location`s (Z = feature axis) usable
as `at=`. Real standard dimensions, simplified geometry (no threads). `mass_g=None` → material.

| Call | Geometry / frames |
|---|---|
| `nema17(length=40)`, `nema23(length=56)` | face z=0, body −Z, shaft +Z at origin; `shaft`, `hole_1..4`; catalog mass |
| `sg90()`, `mg996r()`, `n20_gearmotor()` | shaft axis on origin (+Z), servo body toward −X; `shaft`, `hole_*`, servos also `horn`. `frames["shaft"]` sits on the **mounting face** (servo: ear underside, z=0); `frames["horn"]` is the spline tip where a horn seats (sg90 **z=14.0**, mg996r **16.4**; body tops 6.8 / 10.0): `servo.frames["horn"] * horn_shape`. Either frame is a valid joint axis (`at=`). Stall torque (for `actuator`): SG90 ≈ 0.176 N·m, MG996R ≈ 1.08 N·m |
| `bearing("608")` (623–626, 608, 688, 6000–6005, 6200–6202, suffixes ok), `bearing("LM8UU"/"LM10UU")` | centered on origin, axis Z; `center`, `top`, `bottom` |
| `spur_gear(module, teeth, width, *, bore=0, backlash=0.05, pressure_angle=20)` | axis +Z, bottom z=0, tooth on +X; `axis` |
| `gear_pair(m, z1, z2, width, *, backlash=0.05, bore1=0, bore2=0) -> GearPair(g1, g2, ratio, center_distance)` | g1 at origin, g2 at (a, 0, 0) already meshed; `asm.gear(j1, j2, gp.ratio)` with both axes +Z |
| `internal_gear(module, teeth, width, *, rim=2.5·m, backlash=0.05, pressure_angle=20, pinion=None)` | ring gear, axis +Z, bottom z=0, a tooth **space** on +X, OD m·teeth + 2.5m + 2·rim; conjugate of `spur_gear`; `pinion=z` trims its tips against involute interference (always pass it). Pinion at `Pos(m·(teeth−z)/2, 0, 0) * spur_gear(m, z, …)` meshes; fixed ring + planets: `asm.mesh(ring, planet)`. teeth ≥ 34 at 20° |
| `rack(module, length, width, height)` | along X, teeth +Y, pitch line y=0; `Pos(0, r, 0) * Rot(0, 0, -90) * spur_gear(...)` meshes; `pitch` |
| `gt2_pulley(teeth, width, bore)` | pitch ø 2·teeth/π; `axis`, `belt`. Belt drive: `asm.belt(j_motor, j_out, t_motor, t_out)`. Model the belt, if at all, as a ground part with `allow_contact(belt, pulley, max_depth=None)` to each pulley (a band at the pitch line overlaps the teeth ~0.3 mm) |
| `rod(d, length)`, `t8_leadscrew(length)` (`T8_LEAD` = 8) | along +Z from z=0; `bottom`, `top`; steel |
| `t8_nut()` | flange z=0..3.5, body to z=−11.5; `center`, `hole_1..4`; bore ø8 = screw → `allow_contact` |
| `extrusion_2020(length)`, `extrusion_2040(length)` | centered on Z, z=0..length; catalog mass |
| `socket_head_screw(size, length)`, `button_head_screw`, `hex_nut(size)`, `washer(size)` (M2–M8) | screw head underside z=0, shank −Z; `head`, `tip`; nut/washer on z=0 extending +Z |
| `fit(nominal, "H7/g6")` → `{"hole", "shaft", "clearance"}` (min, max) mm | ISO 286; also `tolerance_zone(d, "H7")`, `it_grade` |
| `fdm_hole(d, clearance=0.2)`, `clearance_hole("M3", fit="normal")`, `tap_drill("M3")`, `heat_set_insert("M3")` | hole sizes |

Every part must be a closed solid (INVALID otherwise). Model nominal fits at nominal size (LM8UU
on rod(8), a ø5 pulley bore on the ø5 motor shaft, a 608 in a ø22 seat) — no 0.1 mm "safety"
gaps: coincident faces never read as interference, and a pair the joint carries (the parts it
names, parts on its axis line or around it running in its bore, parts touching at home) is
checked for overlap only. Declare `joined` for a carried pair no single joint names, `fasten` for
screws in tapped holes, `allow_contact` for any other intended touch.

## Materials (g/cm³)

PLA 1.24, PETG 1.27, ABS 1.04, ASA 1.07, TPU 1.21, nylon/PA12 1.01, resin 1.15, POM 1.41,
acrylic 1.18, aluminum_6061 2.70 (`aluminum`), steel 7.85, stainless_304 8.00 (`stainless`),
brass 8.50, copper 8.96, plywood 0.68, MDF 0.75, carbon_fiber 1.60, rubber 1.10. Case-insensitive;
`density=` makes a custom material.

## Target metrics (`target(label, metric, min=, max=)`)

Evaluated per study; with `study=None` the studies combine by what the metric measures:
`span`/`delta`/`path` → the **largest** study (what the mechanism can do: `span:j_yaw ≥ 270` is
met when any study sweeps 270°); `min`/`min_dist` → the minimum and `max`/`max_dist` → the maximum
over every frame of every study; `clearance`, `load`, `sf`, `rot`, `angle` → the **worst** study
(limits that must hold everywhere). Studies that don't move the joint/probe/part are skipped for
span/delta/path/rot/angle. Pin `study=` when a requirement belongs to one motion.

| Metric | Value |
|---|---|
| `span:<joint>` / `min:<joint>` / `max:<joint>` | joint range (deg / mm) |
| `min_dist:<pA>,<pB>` / `max_dist:<pA>,<pB>` | distance between two probes over the study |
| `path:<probe>` | path length (mm); a `pingpong` study counts the full there-and-back cycle |
| `delta:<probe>.x\|y\|z` | max − min along an axis |
| `rot:<part>` | largest rotation of the part from its home orientation (deg, 0…180); "jaw stays parallel" = `rot:jaw max=0.1` |
| `angle:<partA>,<partB>` | largest rotation of B relative to A (deg); 0 = they keep their relative orientation |
| `clearance` | signed: min gap among checked (non-joined, non-allowed) pairs; negative = mean overlap depth of the deepest interference of any pair |
| `mass_g` | total mass |
| `load:<joint>` / `sf:<joint>` | max \|holding load\| / capacity ÷ max load |
| `callable(report_dict) -> float` | anything else (schema below) |

All per-study metrics use the study's **closed** frames only. Bounds allow float noise
(1e-9·max(1, |bound|)): a value on its bound by construction meets it. A miss prints the bound and
the signed margin: `opening: max_dist:pad_l,pad_r 29.2 < min 30 (margin −0.800)`.

A callable gets the report dict (as in report.json, before targets are filled in):
`params`, `mass {total_g, com_mm, parts}`, `studies[i] {name, frames, joint_ranges {j: [min, max]},
probes {p: {min, max, start, end, path_mm}}, min_clearance {parts, value, frame, at}, loads {j: {unit,
max_abs, frame, capacity, sf}}, max_residual, series}` where `series` is per frame: `ok` [bool],
`joints` {j: [value]}, `probes` {p: [[x, y, z]]}, `transforms` {part: [4×4 row-major nested list]}
(home world → current world). Example (`examples/parallel_gripper.py`):

```python
def jaw_tilt(report) -> float:            # deg the jaw turns about Z: 0 = stays parallel
    Ts = report["studies"][0]["series"]["transforms"]["jaw_l"]
    return max(abs(math.degrees(math.atan2(T[1][0], T[0][0]))) for T in Ts)
asm.target("jaws parallel", jaw_tilt, max=0.01)
```

## Clearance semantics

- **joined** = parts of one rigid body, plus the parts a joint/pin actually carries — the two parts
  it names, parts of both bodies within pin_tol of its axis line (shafts, bushings; for a slide only
  parts side by side along its line — not an end stop the line passes through), a part around the
  axis (bearing, eye, socket ring) with the other side's parts filling its bore all round (a pin in
  an eye — not an arm turning inside a housing's bore), parts touching at home (a bushing on its
  rod) — plus declared `joined`/`fasten`/`ball` pairs. All
  ground=True parts form ONE body, but only the carrying parts are exempt: posts, brackets and
  motors elsewhere on the frame keep `clearance`. Joined pairs are never tight, only interference;
  `check_clearance(a, b)` opts a named pair back in. Everything else must keep `clearance` mm or
  it is `tight` (a gap built exactly to `clearance` meets it, as a `clearance` target does).
- Overlap > `interference_tol` mm³ (or mean depth > 0.02 mm) → `interference` (FAIL). An
  `allow_contact` pair may overlap as deep as its `max_depth` (mean depth 2V/A, default 0.1 mm) →
  `contact` (an INFO gives the depth); deeper → `interference`. Touching or a smaller overlap →
  `contact` for joined/allowed/meshing pairs, `tight` for all others.
- Interference is checked between frames too: a hit between two samples is reported
  `between fK–fK+1` with the interpolated drivers — also where the motion reverses between two
  frames (a slider's dead centre), and for joined pairs apart at both frames. Intervals the check gives up on unproven are an `unresolved`
  INFO naming the pairs: there only the frames vouch for the verdict.
- Cost: exact distances run only on the faces the tessellations put within reach; bodies of
  revolution turning about their own axis (pins, shafts, bearings) are measured once per study;
  a part sliding on a prismatic joint stays clear of what surrounds it (a piston and the fins
  around its barrel) at no cost, and a ball or pin with play in its socket/bore is proven clear
  from its sphere/cylinder surfaces;
  an overlap boolean is validated (a coincident-face artefact is dropped, a real overlap kept), so
  nominal fits are cheap and exact. `allow_contact(max_depth=None)` pairs and `virtual` parts are
  never swept. `--verbose` prints where the sweep time went (pairs, exact, booleans, slowest pairs).
- Only the tagged gears/racks of the two coupled bodies (library spur_gear/gear_pair/rack/
  internal_gear) and `mesh()` pairs are meshing (`gear_mesh` INFO, folded into one line); a crank
  fixed to a pinion is checked normally. A gear/rack coupling with nothing engaged at home, or a
  `mesh()` pair whose teeth are farther than `clearance` apart (a planet off its mesh radius), is a
  `gear_mesh` WARN (use `belt`/`couple`, or move the gear into mesh).
- Parts fixed together are checked once at home (`static_interference`).

## Issue codes → what to do

| Code | Sev | Meaning → usual fix |
|---|---|---|
| `invalid_model` | FAIL (INVALID) | script/model error, all listed with did-you-mean → fix every line, rerun |
| `interference` | FAIL | overlap volume, extent, the spot in each part's home geometry (`@ arm (…), post (…)`; one point when they agree), frame, joint values → move/resize/relieve the parts there, change link layers, or limit the study |
| `static_interference` | FAIL/WARN | parts rigidly together overlap at home (WARN when `fix`-attached) → fix the geometry (or `ignore` if intentional, e.g. press fit) |
| `tight_clearance` | WARN | gap < `clearance` + closest-point location (in each part's home geometry) and direction a→b → open the gap along that direction, `allow_contact` if the touch is intended; for a joined pair far from its hinge use `check_clearance` |
| `loop_open` | FAIL | drive sub-range where the loop can't close + the range that closes → change link lengths (Grashof) or restrict the drive; "never closes" = wrong pin point/joint origins |
| `branch_jump` | WARN | passive joint flips assembly mode → keep away from the dead point, more frames, or drive another joint |
| `singular_pose` | WARN | study passes a change/dead point (loop Jacobian singular): the real mechanism may take either branch there, mech kept the previous one → stop the drive short of it, or change link lengths |
| `joint_limit` | WARN | coupled/passive joint leaves its limits (value, frame, driver value) → widen limits or restrict the drive |
| `over_capacity` | FAIL (>cap) / WARN (SF<1.5) | holding load vs actuator → stronger actuator, gearing, shorter arm, counterbalance, lighter parts |
| `target_miss` | target's severity | metric outside [min, max] → change the parameter the metric depends on (`mech sweep`) |
| `underconstrained` | WARN / INFO | study leaves DOF free → drive/couple/pin another joint; INFO = home pose is a toggle point |
| `near_planar` | WARN | loop hinge axes almost (not exactly) parallel → make them exactly parallel |
| `joint_off_part` | WARN | revolute axis > `pin_tol` from (and not around) its parent/child body, or the two bodies meet the axis line only far apart along it (prismatic joints are not checked); a `ball` center off (and not surrounded by) its parts; a `joined`/`fasten` pair apart at home; a `fasten` pair on bodies that move relative to each other → fix origin/axis/center, `fix()` the fastener to its part, or drop the declaration |
| `pin_off_part` (in `invalid_model`) | FAIL | the hinge pin's axis line misses a part, or its two parts meet the line only far apart; a ball pin's point is off a part → fix the point/axis, or name the part that is there (hint) |
| `held_at_home` | INFO | joints no study moves → add a study if their motion matters |
| `subframe_unresolved` | INFO | between-frame intervals the sweep could not prove clear (pairs named; `unresolved` in the summary) → only the frames vouch for them: more `--frames`, or a `shot` there |
| `gear_mesh` | INFO / WARN | meshing pair auto-allowed (expected) / gear or rack coupling whose parts don't engage, or a `mesh()` pair apart → move into mesh, or `belt`/`couple` |
| `contact` | INFO | `allow_contact` pair overlapping within its `max_depth` (mean depth, volume) |

## CLI (`uv run mech …`, exit PASS 0 · WARN 1 · FAIL 2 · INVALID/error 3)

- `run <script> [-p k=v ...] [--study NAME] [--frames N] [--no-export] [--step] [--verbose] [--json]`
  — `-p` values are Python literals (else strings), must follow the script; unknown params list
  the valid ones. Progress on stderr (a terminal: live per-frame `solve`/`clearance` counts; else
  `--verbose`: dots per 10 %), and each study's time split (`solve 2.7 s, clearance 23.7 s, loads
  5.2 s`). `--verbose` also lifts the 15-line / 250-char caps (the summary folds long lists into
  `(+N … — --verbose)`) and adds a `sweep` stats line per study. Exports `output/<name>.mech/`:
  scene.json, report.json (+ `studies[i].sweep_stats`, `mass.parts_com_mm`), **series.json** (every
  study's t, ok, joint values, probe points, load series, residual at full precision — read it for
  per-frame numbers instead of re-deriving them), parts/*.stl.
  `--study`/`--frames` make a **partial run**: header `(partial run: --study tilt, skipped pan)`,
  targets that need a skipped study — or whose miss a skipped study could still repair, which
  includes every callable's miss (it can't say which studies it reads) — are `not evaluated`, Δprev
  compares only studies sampled alike (findings left out are counted: `no change in the shared
  scope (not compared: 1 FAIL, …)`), and nothing is exported (the folder keeps
  the last full run; the report goes to `report.partial.json`). Good for a quick look; the verdict
  needs a full run. `run(build())` in a script writes to `<repo>/output` whatever the working directory.
- `check <script> [-p ...] [--verbose]` — validate + roles (drivers first; passive/fixed counted when
  long) / mobility + home clearance; seconds; no export; the last line is the `mech run` command to copy.
- `sweep <script> k=a:b:step k=v1,v2 ... [--frames N] [--export-best]` — grid ≤ 50 variants; a
  `why` column names each row's first FAIL/WARN (code + subject); SF of an unloaded actuator is `∞`.
- `shot <name> [--issue i] [--study S] [--frame N|home] [--q j:v] [--view V] [--cam az,el] [--zoom F]
  [--section [-]x|y|z[:mm]] [--focus P] [--isolate P] [--hide P] [--explode F] [--axes] [--ghost N]
  [--paths 0|1] [--layout quad] [--param k=v] [-o out.png]` — PNG in `output/<name>.mech/`, named
  after the options (`shot_back_f9_top_z2_axes.png`); no option = the report's targeted view (first
  FAIL/WARN issue, else a ghosted quad). `--frame` applies to `--study` (else the first study);
  `--q` picks the frame nearest its joint values instead (not with `--frame`); `--isolate` keeps the
  others as faint context, `--hide` removes parts; `--focus/--isolate/--hide/--q` repeat or take comma
  lists; bad values fail before a browser starts.
  Hero shot: `--view iso --zoom 1.2 --paths 0`; a pose: `--study dig --frame 40`; inside: `--section y`.
- `list` — exported mechanisms with the status of their last full run.
- Viewer: `cd previewer && npm run dev` → `http://localhost:3000/mech.html?m=<name>`; params
  `study, frame (N|home), q=j:45, hide, isolate, focus=a,b, issue=i, section=x:10, explode=0.5,
  axes=1, view, cam=az,el, zoom=2, ghost=N, paths=0|1, layout=quad, ui=0` (an unknown or malformed
  param is an error with a did-you-mean). The default fit frames the parts visible at the current
  pose (not the motion envelope); ghosts count only with `ghost=N`, long probe paths are cropped
  unless `paths=1`. **View names are world-plane based**: `front` looks along +Y at the XZ plane
  (a vehicle's *side* if it drives along X), `right` looks along −X, `top` down −Z.

## Common mistakes

- Modeling a part at the origin and expecting the joint to place it — parts are drawn **where
  they are at home**; the joint only moves them from there.
- A second joint to close a loop (INVALID "child of 2 joints") → `pin`.
- Loop not closed in the drawing → `loop_open` from frame 0; compute points with
  `circle_intersect` instead of guessing.
- Wrong sign: +θ follows the right-hand rule about `axis`; pick the axis so + is "open/raise".
  Right-hand lead screw, same axis direction: +360° moves the nut −lead (drive negative to lift).
- Prismatic joint drawn away from 0 without `home=` → limits and targets off by the offset.
- Links in the same plane → interference at every crossing; offset them in z (`z=t+0.5`).
- Driving a coupled joint (INVALID) — drive its driver.
- `no gravity load on j_x (axis ∥ g or balanced)` when every axis is parallel to gravity is
  correct (horizontal mechanism), not a bug.
- Seating a servo horn at `frames["shaft"]`: that frame is the mounting face (z=0) — the horn ends
  up buried in the body; use `frames["horn"]` (spline tip, z=14.0 SG90 / 16.4 MG996R).
- Hand-rolled gear placement: use `gear_pair()` (g2 already rotated to mesh) and `gp.ratio`.
- Huge frame counts: 40–75 frames are plenty; `--frames 12` for quick (partial, unexported)
  checks — coarse frames change spans, extremes and loads (and a hit between frames is reported
  as `between f3–f4 j=…`), so confirm with a full run.
- Never import or call `bd_warehouse` (broken with build123d 0.13).
- Workarounds the framework no longer needs (they hide real checks): raising `pin_tol` for a big
  bore, 0.1 mm "safety" gaps on nominal fits, `ignore()` for screws (`fasten`), `allow_contact`
  for pairs a joint carries, `allow_contact(max_depth=0.02)` for a fixed ring gear (`mesh`).
- build123d / numpy gotchas: a clockwise `Polygon(..., align=None)` has a −Z face that won't fuse
  with circles and extrudes the wrong way (use `mech.geom.polygon`); `offset()` of a face returns a
  `Sketch`, not a `Face`; numpy 2 has no `ndarray.ptp()` (use `np.ptp(a)`) and deprecates `np.cross`
  on 2-D vectors (pad to 3-D); a `Cone` whose apex sits on a sphere's centre can silently drop a
  boolean — use frustums, and check `len(shape.solids())` after fusing.
