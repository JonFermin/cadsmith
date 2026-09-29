# mech API cheat-sheet

Full contract: `docs/MECH_SPEC.md`. Templates: `examples/*.py` (all PASS except `hinged_box`,
whose default `gap=0.2` is a deliberate `tight` WARN).

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
| `gear(driver, driven, ratio)` | rev→rev; ratio for co-directional axes (external mesh negative: `gp.ratio`) |
| `screw(driver, driven, lead, *, hand="right")` | rev→prismatic; right hand, co-directional axes: **+360° → −lead** |
| `rack(driver, driven, pitch_radius, *, sign=None)` | rev→prismatic, ratio ±π·r/180 mm/deg (sign from geometry) |
| `couple(driver, driven, ratio=1.0, offset=0.0)` | generic linear coupling, native units |
| `pin(name, a, b, point, axis=None)` | loop closure: `point` stays coincident on a and b; `axis` = hinge pin (axes stay aligned), none = ball |
| `probe(name, part, point)` | track a point (home world) fixed to `part` |
| `actuator(joint, capacity)` | N·m (revolute) / N (prismatic); also the preferred driver of a loop |
| `allow_contact(a, b)` / `ignore(a, b)` | expected contact: touching or overlap reports `contact`, never tight/interference / skip the pair |
| `study(name, drive, *, frames=60, loop="once", duration=3.0)` | `drive = {joint: (start, end) \| [(u, v), ...] \| callable(u) \| number}`, u ∈ [0,1]; `loop="pingpong"` goes there and back |
| `target(label, metric, *, min=None, max=None, study=None, severity="FAIL")` | requirement; `severity="WARN"` for soft goals |

Rules: every part is `ground=True` or the child of exactly one joint; close loops with `pin`
(within `pin_tol` of both parts); any non-coupled, non-fixed joint may be driven, including a
loop joint (the rest of the loop is solved); never drive a coupled joint; don't drive more
joints of a loop than it has DOF (INVALID "overconstrained").

**Default studies** (no `study` declared): each serial joint sweeps its limits over 40 frames
(revolute without limits 0→360°, prismatic without limits: no study); each loop is driven at its
actuator (else a joint with limits, else its first revolute) over 60 frames. Joints no study
moves get one `held_at_home` INFO.

Frame sampling: `once` → u = k/(N−1) (ends included; `frames=73` over 0…360° gives 5° steps);
`pingpong` → N frames there and back (seamless loop in the viewer).

## geom helpers (`from mech.geom import ...`)

- `link(p0, p1, width, thickness, *, z=None, hole=None, normal=(0,0,1)) -> Part` — rounded bar
  between two points, bottom face at offset `z` along `normal`, end holes ø`hole` (default
  0.4·width, 0 = none). Stack links in layers (`z=0`, `t+0.5`, `2t+1`) so they can pass.
- `circle_intersect(c0, r0, c1, r1, side=+1, normal=(0,0,1)) -> (x, y, z)` — closes a linkage in
  the drawing; `side=+1` = left of c0→c1 looking down `normal`. `ValueError` if unreachable.
- Rarely needed: `vec3, unit, rot_about_line(origin, axis, deg), translation(v),
  transform_points(T, pts), to_location(T), from_location(loc)`.

## Parts library (`from mech.parts import ...`)

Each returns a `LibPart(shape, bom, mass_g, frames)`; place with `Pos(...) * Rot(...) * part`
or `part.mate("frame", target_location)`; `frames` are world `Location`s (Z = feature axis) usable
as `at=`. Real standard dimensions, simplified geometry (no threads). `mass_g=None` → material.

| Call | Geometry / frames |
|---|---|
| `nema17(length=40)`, `nema23(length=56)` | face z=0, body −Z, shaft +Z at origin; `shaft`, `hole_1..4`; catalog mass |
| `sg90()`, `mg996r()`, `n20_gearmotor()` | shaft on origin (+Z), servo body toward −X; `shaft`, `hole_*` |
| `bearing("608")` (623–626, 608, 688, 6000–6005, 6200–6202, suffixes ok), `bearing("LM8UU"/"LM10UU")` | centered on origin, axis Z; `center`, `top`, `bottom` |
| `spur_gear(module, teeth, width, *, bore=0, backlash=0.05, pressure_angle=20)` | axis +Z, bottom z=0, tooth on +X; `axis` |
| `gear_pair(m, z1, z2, width, *, backlash=0.05, bore1=0, bore2=0) -> GearPair(g1, g2, ratio, center_distance)` | g1 at origin, g2 at (a, 0, 0) already meshed; `asm.gear(j1, j2, gp.ratio)` with both axes +Z |
| `rack(module, length, width, height)` | along X, teeth +Y, pitch line y=0; `Pos(0, r, 0) * Rot(0, 0, -90) * spur_gear(...)` meshes; `pitch` |
| `gt2_pulley(teeth, width, bore)` | pitch ø 2·teeth/π; `axis`, `belt` |
| `rod(d, length)`, `t8_leadscrew(length)` (`T8_LEAD` = 8) | along +Z from z=0; `bottom`, `top`; steel |
| `t8_nut()` | flange z=0..3.5, body to z=−11.5; `center`, `hole_1..4`; bore ø8 = screw → `allow_contact` |
| `extrusion_2020(length)`, `extrusion_2040(length)` | centered on Z, z=0..length; catalog mass |
| `socket_head_screw(size, length)`, `button_head_screw`, `hex_nut(size)`, `washer(size)` (M2–M8) | screw head underside z=0, shank −Z; `head`, `tip`; nut/washer on z=0 extending +Z |
| `fit(nominal, "H7/g6")` → `{"hole", "shaft", "clearance"}` (min, max) mm | ISO 286; also `tolerance_zone(d, "H7")`, `it_grade` |
| `fdm_hole(d, clearance=0.2)`, `clearance_hole("M3", fit="normal")`, `tap_drill("M3")`, `heat_set_insert("M3")` | hole sizes |

Nominal bores equal nominal shafts (LM8UU on rod(8), 625 on a ø5 shaft): harmless for joined
pairs (joint parent/child, same rigid body); declare `allow_contact` for the others.

## Materials (g/cm³)

PLA 1.24, PETG 1.27, ABS 1.04, ASA 1.07, TPU 1.21, nylon/PA12 1.01, resin 1.15, POM 1.41,
acrylic 1.18, aluminum_6061 2.70 (`aluminum`), steel 7.85, stainless_304 8.00 (`stainless`),
brass 8.50, copper 8.96, plywood 0.68, MDF 0.75, carbon_fiber 1.60, rubber 1.10. Case-insensitive;
`density=` makes a custom material.

## Target metrics (`target(label, metric, min=, max=)`)

Evaluated per study (or the worst over all studies when `study=None`):

| Metric | Value |
|---|---|
| `span:<joint>` / `min:<joint>` / `max:<joint>` | joint range (deg / mm) |
| `min_dist:<pA>,<pB>` / `max_dist:<pA>,<pB>` | distance between two probes over the study |
| `path:<probe>` | path length (mm) |
| `delta:<probe>.x\|y\|z` | max − min along an axis |
| `clearance` | min gap among checked (non-joined, non-allowed) pairs |
| `mass_g` | total mass |
| `load:<joint>` / `sf:<joint>` | max \|holding load\| / capacity ÷ max load |
| `callable(report_dict) -> float` | anything else |

## Clearance semantics

- **joined** = same rigid body, or bodies connected directly by a joint or pin: never `tight`,
  only `interference` when they overlap. Everything else must keep `clearance` mm or it is `tight`.
- Overlap > `interference_tol` mm³ (or mean depth > 0.02 mm) → `interference` (FAIL), unless the
  pair is `allow_contact`-ed (then `contact`). Touching or a smaller overlap → `contact` for
  joined/allowed/meshing pairs, `tight` for all others.
- Gear/rack meshing parts (the engaging parts of the two coupled bodies) are auto-allowed
  (`gear_mesh` INFO); they still fail on real overlap.
- Parts fixed together are checked once at home (`static_interference`).

## Issue codes → what to do

| Code | Sev | Meaning → usual fix |
|---|---|---|
| `invalid_model` | FAIL (INVALID) | script/model error, all listed with did-you-mean → fix every line, rerun |
| `interference` | FAIL | overlap volume, extent, home-world location, frame, joint values → move/resize/relieve the part there, change link layers, or limit the study |
| `static_interference` | FAIL/WARN | parts rigidly together overlap at home (WARN when `fix`-attached) → fix the geometry (or `ignore` if intentional, e.g. press fit) |
| `tight_clearance` | WARN | gap < `clearance` + closest-point location and direction a→b → open the gap along that direction, or `allow_contact` if the touch is intended |
| `loop_open` | FAIL | drive sub-range where the loop can't close + the range that closes → change link lengths (Grashof) or restrict the drive; "never closes" = wrong pin point/joint origins |
| `branch_jump` | WARN | passive joint flips assembly mode → keep away from the dead point, more frames, or drive another joint |
| `joint_limit` | WARN | coupled/passive joint leaves its limits (value, frame, driver value) → widen limits or restrict the drive |
| `over_capacity` | FAIL (>cap) / WARN (SF<1.5) | holding load vs actuator → stronger actuator, gearing, shorter arm, counterbalance, lighter parts |
| `target_miss` | target's severity | metric outside [min, max] → change the parameter the metric depends on (`mech sweep`) |
| `underconstrained` | WARN / INFO | study leaves DOF free → drive/couple/pin another joint; INFO = home pose is a toggle point |
| `near_planar` | WARN | loop hinge axes almost (not exactly) parallel → make them exactly parallel |
| `joint_off_part` | WARN | joint axis > `pin_tol` from its parent/child body → fix origin/axis |
| `held_at_home` | INFO | joints no study moves → add a study if their motion matters |
| `gear_mesh` | INFO | meshing pair auto-allowed (expected) |

## CLI (`uv run mech …`, exit PASS 0 · WARN 1 · FAIL 2 · INVALID/error 3)

- `run <script> [-p k=v ...] [--study NAME] [--frames N] [--no-export] [--step] [--verbose] [--json]`
  — `-p` values are Python literals (else strings), must follow the script; unknown params list
  the valid ones. Exports `output/<name>.mech/` (scene.json, report.json, parts/*.stl).
- `check <script> [-p ...]` — validate + roles/mobility + home clearance; seconds; no export.
- `sweep <script> k=a:b:step k=v1,v2 ... [--frames N] [--export-best]` — grid ≤ 50 variants.
- `shot <name> [--issue i] [--view iso|top|front|right|left|back|bottom] [--ghost N] [--layout quad] [--frame N] [-o out.png]`
  — PNG in `output/<name>.mech/`; default view = the report's targeted view.
- `list` — exported mechanisms with status.
- Viewer: `cd previewer && npm run dev` → `http://localhost:3000/mech.html?m=<name>`; params
  `study, frame, q=j:45, hide, isolate, focus=a,b, issue=i, section=x:10, explode=0.5, axes=1,
  view, cam=az,el, ghost=N, layout=quad, ui=0`.

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
- `load … 0 N·m · SF ∞` when every axis is parallel to gravity is correct (horizontal
  mechanism), not a bug.
- Hand-rolled gear placement: use `gear_pair()` (g2 already rotated to mesh) and `gp.ratio`.
- Huge frame counts: 40–75 frames are plenty; `--frames 12` for quick checks.
- Never import or call `bd_warehouse` (broken with build123d 0.13).
