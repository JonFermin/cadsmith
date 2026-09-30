# `mech` — mechanism analysis framework (spec v2)

**Goal:** Claude writes only a short *model script* (geometry + joints + intent). The framework does
everything else deterministically — kinematics, loop closure, clearance/interference sweeps, mass
properties, gravity holding loads, design-target checks, animation export, screenshots, and a compact
report — so a design iteration costs one `uv run mech run output/x.py` and ~15 lines of text.

Scope v1: kinematics + clearances (+ mass props, quasi-static gravity loads, targets, variant sweeps).
Out of scope: dynamics, FEA, electronics.

Environment: `uv` project at repo root, Python 3.12 (`.python-version`), deps `build123d 0.13`,
`numpy`, `scipy`; dev: `pytest`, `playwright` (optional, for `mech shot`). Run everything with
`uv run ...`; tests `uv run pytest`. Windows 10 host (git-bash available). Never use system Python
(3.14, no OCP wheels). `bd_warehouse` is installed but **broken with build123d 0.13 — do not use it**.

Verified facts about build123d 0.13 / OCP on this host (don't re-discover):
- `Location(np_4x4)` and `Location(list_of_lists)` raise `TypeError`. Use `mech.geom.to_location(T)`.
- `BRepExtrema_DistShapeShape` on Part/Compound does **not** detect containment (a cube buried in a
  cube reports the gap to the outer wall). Use point classification (~0.2 ms) for containment — per
  solid (`mech.geom.inside`): `shape.is_inside(pt)` on a multi-solid compound (library motors,
  bearings, list-of-shapes parts) calls many interior points outside.
- `shape.distance_to(Vector)` works. `Location.__mul__(obj)` defers to `obj.__rmul__` for foreign types.
- `Axis(loc)` gives the location's position + Z direction.
- `export_stl` returns True even when Windows mangles the filename (`:`/reserved names/non-ASCII).
- Exact distance between two meshing involute gears ≈ 0.5 s per call; boolean common ≈ 0.03–0.1 s;
  `BRepExtrema_ShapeProximity` on tessellations ≈ 2.5 ms and honours Locations.
- OCC `MatrixOfInertia` is the proper inertia tensor about the COM (off-diagonals are products with
  the minus sign already applied), in mm⁵ for unit density.

---

## 1. Conventions (non-negotiable)

| Quantity | Unit in API / report | Internal |
|---|---|---|
| length | mm | mm |
| revolute joint value | **degrees** (build123d `Rot()` is degrees too) | radians only inside solver math |
| prismatic joint value | mm | mm |
| mass | g (API/report) | kg in mass-prop math |
| density | g/cm³ | kg/mm³ = ρ·1e-6 |
| torque | N·m | |
| force | N | |
| inertia | kg·mm² | |
| time | s | |

- **Z is up.** Gravity default `(0, 0, -9.80665)` m/s².
- **Parts are modeled in place.** Every part's shape is in *world coordinates at the home pose*
  (where it sits in the drawing). No local frames. Helpers in `mech.geom` (§4) make this easy.
- **`home`** of a joint = the joint value *of the pose you drew*. Limits are in the same coordinate.
- **Joints are declared in world coordinates at home**: `origin` point + `axis` direction.
- A part's **transform** `T` (4×4, mm) maps home-world → current-world. At home every `T = I`.
- **FK (product of exponentials over the tree):** `T_child(q) = T_parent(q) · M_j(q_j − home_j)`;
  `M_j(θ)` = rotation by θ about line (origin, axis) [revolute], translation θ·axis [prismatic], I [fixed].
  Ground parts have `T = I`. Positive rotation = right-hand rule about `axis`.
- **Couplings act on deltas from home:** `q_driven − home_driven = ratio·(q_driver − home_driver) + offset`,
  `offset` default 0 so home is always kinematically consistent.
- **JSON rules:** only Python-native types; `json.dump(..., allow_nan=False)`; floats as
  `float(f"{x:.6g}")`; undefined numbers are `null` (e.g. `sf` when load is 0); matrices via
  `geom.to_json16` (column-major, three.js `Matrix4.fromArray` order).
- **Frame sampling:** `once`: u_k = k/(N−1), t_k = duration·k/(N−1). `pingpong`: u_k = 1 − |1 − 2k/N|,
  t_k = duration·k/N, k = 0..N−1 (seamless wrap). Scene frames are fully expanded; the viewer never mirrors.

---

## 2. The model-script contract (what Claude writes)

Canonical example — **byte-identical to `examples/four_bar.py`** and reproduced in the skill reference:

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

### 2.1 Call table (all frozen signatures; points/axes accept tuple, list, ndarray, `Vector`)

| Call | Meaning |
|---|---|
| `Assembly(name, *, clearance=0.3, interference_tol=0.5, pin_tol=5.0, gravity=(0,0,-9.80665))` | `clearance` = min gap (mm) required between parts that aren't joined; `interference_tol` mm³ |
| `part(name, shape_or_libpart, *, material="PLA", color=None, ground=False, mass_g=None, density=None, opacity=1.0, bom=None) -> str` | shape = build123d Shape/Part/Compound or any object with `.shape` (LibPart; its `.bom`/`.mass_g` used as defaults). Duplicate name → `ValueError` immediately |
| `revolute(name, parent, child, origin=None, axis=None, *, at=None, limits=None, home=0.0) -> str` | hinge. `at=` a `Location` or `Axis` (e.g. `motor.frames["shaft"]`) supplies origin+axis (Z dir) |
| `prismatic(name, parent, child, origin=None, axis=None, *, at=None, limits=None, home=0.0) -> str` | slide along axis |
| `fix(child, parent, name=None) -> str` | rigid attach (name default `fix_<child>`) |
| `gear(driver, driven, ratio, *, name=None)` | revolute→revolute meshing gears; ratio as given assumes both axes point the same way (external mesh: negative) |
| `belt(driver, driven, t_driver, t_driven, *, name=None)` | revolute→revolute belt/chain between parallel axes (within 1e-3 rad): both pulleys turn the same way, ratio = s·t_driver/t_driven (teeth or pitch diameters, > 0), s = sign(axis_driver·axis_driven); never meshing |
| `screw(driver, driven, lead, *, hand="right", name=None)` | revolute→prismatic; ratio = −s·lead/360 for right hand, +s·lead/360 left, s = sign(axis_driver·axis_driven) |
| `rack(driver, driven, pitch_radius, *, sign=None, name=None)` | revolute→prismatic; ratio = sign·π·r/180; sign default from geometry: sign((axis_p × (c − origin_p))·axis_r), c = driven part's bbox center |
| `couple(driver, driven, ratio=1.0, offset=0.0, *, name=None)` | generic linear coupling, native units |
| `pin(name, a, b, point, axis=None)` | loop closure: point (home world) is coincident on parts a and b. With `axis` = hinge (also axes stay aligned); without = ball joint |
| `probe(name, part, point)` | track a point's trajectory |
| `actuator(joint, capacity)` | N·m (revolute) / N (prismatic); also marks the preferred driver of a loop |
| `allow_contact(a, b, *, max_depth=0.1)` / `ignore(a, b)` | expected-touching pair: contact and overlaps of mean depth 2V/A ≤ `max_depth` mm are `contact`, deeper ones still `interference` (`max_depth=None`: any overlap, and no overlap boolean for the pair) / skip pair entirely |
| `check_clearance(a, b)` | hold a *joined* pair (e.g. an arm and the frame it is hinged to) to `clearance` anyway |
| `study(name, drive, *, frames=60, loop="once", duration=3.0)` | drive: `{joint: (start, end) \| [(u, v), ...] \| callable(u) -> value}` with u∈[0,1], values in deg/mm |
| `target(label, metric, *, min=None, max=None, study=None, severity="FAIL")` | design intent, see §2.3 |

Rules:
- `build(**params) -> Assembly` with **every tunable as a keyword default** (enables `-p` and `mech sweep`).
- Every part must be `ground=True` or the child of exactly one joint. Use `pin` (not a second joint) to
  close loops. A pin (its point, or for hinge pins the axis line through the point) must come within
  `Assembly.pin_tol` (default 5 mm — pins pass through holes) of both parts' geometry. Revolute joints
  likewise should touch both parent and child (WARN `joint_off_part` otherwise; a prismatic joint's
  axis line only sets the travel direction, so where it runs is not checked).
- Every part is solid material: a shape with no solid (a Face/Shell/Wire) or zero volume (a boolean
  that removed everything) is INVALID.
- *Joined* pairs are exempt from `clearance` (only overlap counts): parts of one rigid group, and across
  two bodies a joint/pin connects only the parts that carry it (§4.1 `joined_pairs`) — not every part of
  those bodies (all `ground=True` parts form one body, so that would excuse every frame part).
- Any non-coupled, non-fixed joint may appear in a study's `drive` — including a joint on a loop
  (then the *other* loop joints are solved). Driving a coupled joint is an error.

### 2.2 Default studies (when the script declares none, or for joints not covered)
If the script declares **no** study: one study per *mobility group*:
- each non-loop, non-coupled joint: sweep its limits `once` 40 frames; revolute without limits: 0→360.
- each loop: drive the loop joint with an `actuator` (else first-declared loop joint with limits, else
  first-declared revolute loop joint 0→360); 60 frames.
- a driver's range is narrowed to where every joint it drives through couplings stays within *that*
  joint's limits (mapped back through the affine coupling); a driver without limits of its own then
  sweeps the coupled joint's lo → hi (a lead screw turns exactly the nut slide's stroke).
Joints never driven by any study stay at home; `mech` prints one INFO naming them.

### 2.3 Targets
Metric vocabulary (evaluated per study on its *closed* frames; if `study=None`, over all studies —
the worst value counts):
`span:<joint>` (max−min), `min:<joint>`, `max:<joint>`, `min_dist:<probeA>,<probeB>` /
`max_dist:<probeA>,<probeB>` (min/max distance between two probes over the study), `path:<probe>`
(path length mm: chords within each run of closed frames; a `pingpong` study adds the wrap chord back
to frame 0, so it is the full there-and-back cycle), `delta:<probe>.x|y|z` (max−min),
`rot:<part>` (largest rotation of the part away from its home orientation, deg 0…180 — e.g. "the
jaw stays parallel": `rot:jaw_l` max 0.1), `angle:<partA>,<partB>` (largest rotation of B relative
to A, deg — 0 when they keep their relative orientation),
`clearance` (the sweep's signed `min_clearance`, §4.6: min gap among checked non-joined pairs, or
−(mean overlap depth) of the deepest interference of any pair), `mass_g`, `load:<joint>` (max |load|),
`sf:<joint>` (capacity / max|load|), or a callable `f(report_dict) -> float`. A callable receives the
report (targets not yet filled in) whose `studies[i]` entries also carry a per-frame
`"series": {"ok": [bool], "joints": {j: [value]}, "probes": {p: [[x,y,z]]}, "transforms": {part:
[4×4 row-major nested list]}}` (home world → current world, one element per frame).
A revolute loop unknown is re-wrapped by whole turns across each run of open frames (its value
there is a solver guess). For the extent metrics (span, path, delta, rot, angle) studies that do
not move the joint/probe/part are skipped while another study moves it.
Bounds are compared with a float-noise allowance of `1e-9·max(1, |bound|)` (a value that lands on its
bound by construction meets it). Result: `target_miss` issue with the target's severity when value
is outside [min, max]; its message names the violated bound and the signed margin
(`opening: max_dist:pad_l,pad_r 29.2 < min 30 (margin −0.800)`).
In a partial run (`--study` skipped studies) a target of a skipped study, a `study=None` target that
no run study defines (or moves), and a callable that fails on the partial report are *not evaluated*
(`met: null`, no issue, listed as `not evaluated:` on the targets line); a miss on the studies that
ran is a real miss (more studies can only make the worst value worse).

---

## 3. Package layout & ownership

```
mech/
  __init__.py      public API (§3.1)
  geom.py          SHARED helpers — every module uses these, never re-implements them (§4.0)
  assembly.py      dataclasses, Assembly, ModelError, validate()
  materials.py     MATERIALS, Material, get_material()
  massprops.py     MassProps, part_props(), assembly_props()
  kinematics.py    Kinematics (FK, couplings, loop closure, roles, mobility)
  motion.py        sample_drive(), default_studies(), check_study(), run_study()
  clearance.py     ClearanceChecker, PairResult, SweepResult, rigid groups
  statics.py       gravity_loads()
  targets.py       evaluate_targets()
  report.py        Issue, build_report(), format_summary()
  export.py        export_scene()
  runner.py        analyze(), run()  — the ONLY orchestrator
  sweep.py         variant grid runner
  shot.py          headless screenshots (optional playwright)
  cli.py           `mech` console script
  parts/           standard-parts library (§6)
tests/  examples/  previewer/mech.html + previewer/src/mech/*.js
.claude/skills/make-mechanism/SKILL.md + references/mech-api.md
```

### 3.1 Public API (`from mech import *`)
`Assembly, ModelError, Material, MATERIALS, run, analyze, Kinematics, part_props, assembly_props`
(scripts import `build123d` and `mech.geom` / `mech.parts` themselves).

---

## 4. Internal interfaces (match exactly)

### 4.0 geom.py (frozen first; used by everyone)
```python
def vec3(v) -> np.ndarray                       # tuple/list/ndarray/Vector -> float (3,)
def unit(v) -> np.ndarray                        # ValueError on zero
def to_location(T) -> Location                   # SVD-orthonormalize R, assert det>0, gp_Trsf.SetValues(*T[:3,:4].ravel())
def from_location(loc) -> np.ndarray             # 4x4
def rot_about_line(origin, axis, deg) -> np.ndarray   # 4x4
def translation(v) -> np.ndarray                 # 4x4
def transform_points(T, pts) -> np.ndarray
def transform_aabb(bmin, bmax, T) -> tuple[np.ndarray, np.ndarray]
def inside(shape, p, tolerance=1e-6) -> bool     # p in (or on) any solid of shape, classified solid by solid
def to_json16(T) -> list[float]                  # column-major, 6 sig figs
def fnum(x) -> float | None                      # 6 sig figs, None for nan/inf
def slug(name, taken: set[str] | None = None) -> str   # ASCII [a-z0-9_-], Windows-reserved-safe, case-insensitive dedupe
# modeling helpers for scripts
def link(p0, p1, width, thickness, *, z=None, hole=None, normal=(0,0,1)) -> Part
    # rounded bar (slot) between two world points, lying in the plane ⟂ normal, with holes of dia `hole`
    # (default width*0.4, or 0 for none) at both ends; z = bottom-face offset along normal (default: p0's)
def circle_intersect(c0, r0, c1, r1, side=+1, normal=(0,0,1)) -> tuple[float, float, float]
    # planar two-circle intersection; side picks the solution by sign of ((c1-c0) × (p-c0))·normal; ValueError if none
```

### 4.1 assembly.py
```python
class ModelError(Exception):
    errors: list[str]
@dataclass class Part: name; shape; material: Material; color: str; ground: bool; opacity=1.0; mass_g=None; bom=None
                       kind: str|None=None    # LibPart.kind: "gear" | "rack" | "pulley" | None
@dataclass class Joint: name; kind: Literal["revolute","prismatic","fixed"]; parent; child;
                        origin: np.ndarray; axis: np.ndarray; limits: tuple|None=None; home: float=0.0
@dataclass class Coupling: name; driver; driven; ratio: float; offset: float=0.0; kind: str="couple"   # gear|belt|screw|rack|couple
@dataclass class Pin: name; a; b; point: np.ndarray; axis: np.ndarray|None=None
@dataclass class Probe: name; part; point: np.ndarray
@dataclass class Actuator: joint; capacity: float
@dataclass class Study: name; drive: dict; frames: int=60; loop: str="once"; duration: float=3.0
@dataclass class Target: label; metric: str|Callable; min: float|None; max: float|None; study: str|None; severity: str
class Assembly:
    name; clearance; interference_tol; pin_tol; gravity
    parts: dict; joints: dict; couplings: list; pins: list; probes: list; actuators: dict; studies: list
    targets: list; allowed: set[frozenset]; ignored: set[frozenset]; params: dict   # params filled by runner
    allow_depth: dict[frozenset, float|None]      # allow_contact max_depth per allowed pair
    checked: set[frozenset]                       # check_clearance pairs
    # methods per §2.1
    def validate(self) -> list[str]      # NEVER raises; collects ALL errors in one pass
    def parent_joint(self, part) -> Joint | None
    def rigid_groups(self) -> dict[str, int]      # union-find over fixed joints; all ground parts share one group
    def is_joined(self, a, b) -> bool   # same rigid group, OR the pair is in joined_pairs()
    def joined_pairs(self) -> set[frozenset]      # cross-body pairs a joint/pin connects (below)
    def meshing_pairs(self) -> set[frozenset]     # engaged tooth-bearing parts of every gear/rack coupling (below)
```
`joined_pairs()`: for every moving joint (parent, child, origin + axis line) and every pin (a, b, point
[+ hinge axis]) whose two parts lie in different rigid groups, the pairs of those two bodies that carry
it: the two parts it names; every cross pair whose parts *both* come within `pin_tol` of its axis line
(ball pin: its point) — shafts, bearings, bushings, a gear bored on the shaft; and every cross pair that
touches at home (gap ≤ 1e-6 mm: a bushing on its guide rod, a large bearing on its shaft). Minus
`checked`. Everything else on the two bodies — posts, brackets, motors elsewhere on the ground body,
parts fixed to a moving link away from its pivot — is held to `clearance`.
`meshing_pairs()`: for each gear/rack coupling, the two coupled joints' children's rigid groups; per side
the candidates are its parts tagged `kind` gear/rack (library `spur_gear`, `gear_pair`, `rack`), else its
joint child when that engages the other side, else all its parts; meshing pairs = candidate cross pairs
whose home AABBs inflated by `clearance` overlap — none when nothing engages (a `gear_mesh` WARN from
`validate_warnings`, e.g. a belt drive modelled with `gear`: use `belt`, whose couplings never mesh).
`validate()` errors (each with a `difflib.get_close_matches` "did you mean" when a name is unknown):
unknown part/joint names anywhere; floating part; part with two parent joints; joint-tree cycle; a
joint driven by two couplings; coupling cycle; coupling kind mismatch (gear/belt rev→rev; screw/rack
rev→pris); screw or belt axes not parallel; belt teeth ≤ 0; `home` outside `limits`; `pin_off_part`: pin farther than `pin_tol` from either part — for hinge
pins measure `shape.distance_to(Edge.make_line(p − L·n, p + L·n))`, for ball pins `shape.distance_to(Vector(p))` —
naming part + distance; a part that is not a closed solid (no solids, or solid volume ≤ 1e-9 mm³ — with or
without `mass_g`); unknown material; `allow_contact` `max_depth` not None/≥ 0; `check_clearance` of an
unknown part or of a pair that is also `allow_contact`/`ignore`d; a target naming an unknown joint,
probe or part (`rot:<part>`, `angle:<partA>,<partB>`) or a metric not in §2.3. Containment tests (pins, joints,
clearance) classify a point against each solid of a part separately (`geom.inside`): OCC's classifier on a
multi-solid compound misreads interior points. Warnings (non-fatal) via `validate_warnings() -> list[tuple[code, msg]]`: near-planar loop (axes parallel
within 1e-3 rad but not 1e-9) → `near_planar`; revolute joint axis line farther than `pin_tol` from its
parent or child geometry → `joint_off_part` (prismatic joints are not checked); a gear/rack coupling
whose bodies have no meshing pair at home → `gear_mesh`.

### 4.2 materials.py
`Material(name, density)` g/cm³; `MATERIALS` incl. PLA 1.24, PETG 1.27, ABS 1.04, ASA 1.07, TPU 1.21,
nylon/PA12 1.01, resin 1.15, POM 1.41, acrylic 1.18, aluminum_6061 2.70, steel 7.85, stainless_304 8.00,
brass 8.50, copper 8.96, plywood 0.68, MDF 0.75, carbon_fiber 1.60, rubber 1.10.
`get_material(name_or_obj, density=None)` case-insensitive; unknown → `KeyError` listing options.

### 4.3 massprops.py
```python
@dataclass class MassProps: mass_kg; volume_mm3; com: np.ndarray; inertia: np.ndarray  # kg·mm² about COM, world axes
def part_props(part) -> MassProps
def assembly_props(asm, props=None, transforms=None) -> MassProps
```
`part_props`: OCC `VolumeProperties` ×ρ·1e-6; if Σ solid volumes exceeds the fused volume by > 0.1%,
fuse first (`solid_body(shape)`, cached per shape and shared with the clearance checker); `mass_g`
override scales mass and inertia. `assembly_props`:
`I = Σ [R_i I_i R_iᵀ + m_i(‖d_i‖² E − d_i d_iᵀ)]`, `d_i = R_i c_i + t_i − c_total`.

### 4.4 kinematics.py
```python
@dataclass class Pose:
    q: dict[str, float]; transforms: dict[str, np.ndarray]
    residual: float          # max |residual row| in mm after solve
    ok: bool                 # residual <= 1e-4 mm
    limit_violations: list[tuple[str, float, tuple[float, float]]]   # coupled/passive joints only
class Kinematics:
    def __init__(self, asm)           # raises ModelError(asm.validate()) if any errors
    fixed: set[str]; coupled: set[str]          # structural, transitive coupling closure
    loops: list[list[str]]                       # non-fixed joints on each pin's tree path (a→LCA→b)
    L: float                                     # characteristic length = assembly bbox diagonal (mm)
    def fk(self, q: dict[str, float]) -> dict[str, np.ndarray]      # missing joints = home; applies couplings
    def expand(self, q_free: dict[str, float]) -> dict[str, float]  # applies couplings topologically
    def unknowns(self, driven: Iterable[str]) -> list[str]          # non-coupled loop joints not driven
    def mobility(self, driven) -> int            # #unknowns − generic rank of J_p (samples below)
    def overconstrained(self, driven) -> list[str]   # messages if generic rank([J_p J_d]) > rank(J_p) on a loop
    def roles_for(self, studies) -> dict[str, str]   # fixed|coupled|driver (in any study)|passive (on loop)|free
    def solve(self, drive: dict[str, float], guess: dict[str, float] | None = None, *,
              prev: dict[str, float] | None = None, step_scale: float = 1.0) -> Pose
    def jacobians(self, pose, driven) -> tuple[np.ndarray, np.ndarray]   # J_p, J_d of residual wrt unknowns/drivers
    def singular(self, pose, driven) -> bool     # driven loop block of J_p loses its generic rank here (cond > 1e6)
    def point(self, pose, part, p) -> np.ndarray
```
Loop residual per pin: `T_a·p − T_b·p` (3 rows, mm) plus, with axis, `L·(R_a n − R_b n)` (3 rows, mm).
`least_squares(method="trf", x_scale="jac", xtol=ftol=gtol=1e-12)`, unknown revolutes in radians;
couplings re-expanded inside every residual evaluation (a coupling's driver may be a passive joint).
Continuation from `guess` in driver steps ≤ 10° / ≤ 0.05·L (× `step_scale`). Each step tries, in order,
a secant extrapolation of the last two solutions (seeded by `prev`, the pose before `guess`), the
tangent predictor `q_p += −pinv(J_p)·J_d·Δq_d`, and a plain warm start; with a secant it keeps the
first converged solution within half the hinted step of it, else the converged one closest to it — so
a frame that lands exactly on a change point (where both branches meet) stays on the branch the
previous frames were on. After each step, each revolute unknown (that drives no coupling) is wrapped
into the previous step's value ± 180° — which unwraps a passive joint that turns > 180° in one call.
Generic ranks (mobility, overconstrained, singular) = max over home + 3 seeded random poses near home
*on the constraint manifold* (all joints perturbed ±10° / ±0.05·L, then the loop joints re-closed by
least squares; raw perturbations only when home or no perturbation closes): off-manifold samples break
special geometry (a redundant parallel link, a spherical 4R) and report redundant constraints as
independent. Rank by SVD with rtol 1e-9 on J with revolute columns scaled by L.
Never raise from `solve`; non-convergence → `ok=False`.

### 4.5 motion.py
```python
def sample_drive(study) -> tuple[np.ndarray, dict[str, np.ndarray]]   # t (s), per-joint values (§1 sampling)
def default_studies(asm, kin) -> list[Study]                          # §2.2
def check_study(kin, study) -> list[str]   # unknown/coupled/fixed joints in drive; keyframes u∉[0,1];
                                           # driver values outside limits; frames<2; overconstrained drivers;
                                           # a moving drive whose frames all land on one pose (every driver and
                                           # coupled joint back on its frame-0 value, revolutes up to whole
                                           # turns: 2 frames over 0…360°) — "sample one pose — use frames ≥ N+1"
@dataclass class StudyResult:
    study; t: np.ndarray; drive: dict[str, np.ndarray]; poses: list[Pose]
    probes: dict[str, np.ndarray]                   # (frames, 3) world
    branch_jumps: list[tuple[int, str, float]]      # passive jump > 20° / 5 mm between consecutive frames
    singular: list[int] = []                        # frames on a change/dead point (kin.singular)
def run_study(asm, kin, study) -> StudyResult       # frame k warm-starts from k−1, secant from k−2, k−1
```
`branch_jumps`: a passive joint whose change between two closed frames exceeds 20° / 5 mm *and* misses
the trapezoidal tangent prediction by as much — or, next to a `singular` frame (where the tangent is
meaningless and the branches cross, so a switch needs no big change), misses the secant through the
two frames before by as much — and that a fine re-trace of the step (continuation from the previous
frame in 0.2× steps, same secant seed) does not reproduce within 1e-3. Open frames: raw change only.

### 4.6 clearance.py
```python
@dataclass class PairResult:
    a: str; b: str
    distance: float | None      # mm; 0 when touching/overlapping/contained; None for meshing pairs
    volume: float               # overlap mm³ (0 unless distance == 0)
    pa: np.ndarray; pb: np.ndarray                 # closest points (current world); overlap: common-bbox center for both
    status: Literal["interference", "tight", "contact", "ok"]
    joined: bool; allowed: bool; meshing: bool
    extent: np.ndarray | None = None               # overlap common-solid bbox size (home frame of a), mm
    location: np.ndarray | None = None             # overlap/closest-point midpoint in a's HOME frame (via inv(T_a))
    depth: float | None = None                     # mean overlap depth 2V/A_common, mm (None unless overlapping)
    location_b: np.ndarray | None = None           # the same midpoint in b's HOME frame (via inv(T_b))
    at: float | None = None                        # fractional frame when found between two frames
class ClearanceChecker:
    def __init__(self, asm, clearance=None)
    def check_pose(self, transforms, *, include_rigid=True) -> list[PairResult]
    def sweep(self, result) -> "SweepResult"        # skips intra-rigid-group pairs (checked once at home)
@dataclass class SweepResult:
    per_frame: list[list[PairResult]]               # non-"ok" only (incl. sub-frame finds, at their nearest frame)
    worst: dict[frozenset, tuple[int, PairResult]]
    min_clearance: tuple[str, str, float, int] | None   # SIGNED, see below
    stats: dict                                     # pairs_checked, cache_hits, seconds, sub_poses, sub_measured, ...
```
Parts are measured as solid material: `massprops.solid_body` (overlapping solids of a part — a list of
shapes, a library sub-body — fused; faces/shells dropped); containment classifies against each solid.
Status rules: ignored → skipped. overlap > tol (or mean depth `2V/A_common` > 0.02 mm) → **interference**;
for an allowed pair only a mean depth > its `max_depth` (default 0.1 mm; None = never) interferes, any
shallower overlap or touch is **contact**. touching/overlap ≤ tol → **contact** (OK if
joined/allowed/meshing, else **tight**). 0 < d < clearance → **tight** unless joined/allowed/meshing.
Else ok. Intra-rigid-group interference at home → `static_interference` (WARN if the pair is
fix-attached, FAIL otherwise).
`min_clearance` = `(a, b, −depth, frame)` of the deepest interference of *any* pair (joined, allowed and
meshing included) when one occurs in a closed frame or between frames; else `(a, b, gap, frame)` of the
smallest gap among non-joined/allowed/meshing pairs (0 when touching); None when no such pair exists.

Narrowphase: broadphase = transformed AABBs inflated by clearance, then `BRepExtrema_ShapeProximity` on
cached tessellations to prove "farther than τ" (τ = 1e-6 for excused pairs). Excused (joined/allowed)
pairs within reach: only the boolean common — an overlap is measured, anything else counts as touching
(distance 0, no distance query); `allow_contact(max_depth=None)` pairs skip the boolean too. Other pairs:
exact `BRepExtrema_DistShapeShape` (multi-threaded); if distance > 1e-6, containment test per solid on
one vertex each way → distance 0; boolean common (parallel) only when distance ≤ 1e-6. Meshing pairs:
interference-only, prefiltered on the tessellations; boolean only on candidates. Lower bounds for the
minimum-clearance search: max(world-AABB gap, separating-axis gap of the two home boxes along their six
face normals); a pair that can at best tie the current minimum (within 1e-9) is not measured.
Cache key `np.round(inv(T_a) @ T_b, 6).tobytes()`: compute with a at home and b moved by
`to_location(inv(T_a) @ T_b)`; store pa/pb/location in a's home frame, map through T_a on hit.
Between frames (sweep): for every non-joined/allowed/meshing pair, consecutive closed frames k, k+1 with
gap lower bounds g0, g1 are *settled* when the relative motion bound δ (largest chord of either part's
home-box corners in the other's frame × arc factor of the relative rotation × 1.25; ∞ beyond a quarter
turn) gives (g0 + g1 − δ)/2 ≥ need, or when the boxes are separated by ≥ need along the relative rotation
axis at both frames (rotation never moves points along its axis); need = clearance when neither end is
tight, else 0. Unsettled intervals are bisected (≤ 5 levels) at poses solved with the drivers
interpolated (`kin.solve(..., prev=)`); a non-ok find is recorded at the nearest frame with `at` =
k + s, pa/pb on that frame's placement; the search stops at the first interference.
Target: 10 parts × 60 frames < 20 s excluding meshing pairs and allowed pairs in contact (one boolean per
frame each; `max_depth=None` skips it).

### 4.7 statics.py
```python
def gravity_loads(asm, kin, result, props) -> dict[str, dict]
# key = joint: {"unit": "N·m"|"N", "series": [float|None per frame], "max_abs", "frame",
#               "reflected_from": None | driver_name}
```
V(q) = Σ m_i · (−g)·(T_i·com_i) · 1e-3 [J]. Holding load actuator must supply = dV/dq_d (positive →
pushes +q). Total derivative through couplings and loop closure: `dq_p/dq_d = −pinv(J_p)·J_d`, ∂p/∂q by
finite difference of FK (no re-solve) or analytic. Convert: revolute `×180/π` (per-deg → N·m),
prismatic `×1000` (per-mm → N). Frames with `ok=False` or cond(J_p) > 1e8 → None. Reflected loads for
coupled joints in the study, and for *actuated* passive loop joints the study moves (an actuator on the
crank while the study drives the rocker): `load_driver / (dq_c/dq_d in SI)` from the driver the joint
follows most strongly. Actuated joints the study holds still (serial joints it doesn't drive, loop
joints of loops no driver acts on) get their own holding load, computed as one more driver (its own
sensitivity block, so a singular block elsewhere can't blank it). Entry order: study drivers, held
actuators, then reflected joints — so every declared actuator is rated in every study. Frictionless,
backdrivable.

### 4.8 targets.py
`evaluate_targets(asm, report: dict, results) -> list[dict]` → `{"label","metric","value","min","max",
"margin","met","severity","study","worst_study","error"}` — `margin` = signed distance to the nearest
bound (negative outside), `met` null when not evaluated, `error` says why `value` is null or the
target was not evaluated. Shared closed-frame helpers used by report.py too: `closed_runs(res)`,
`joint_series(asm, res, joint)` (closed frames, loop unknowns re-wrapped across open gaps),
`probe_series(res, probe)`, `path_length(res, probe)`.

### 4.9 report.py
```python
@dataclass class Issue: severity: Literal["FAIL","WARN","INFO"]; code: str; message: str
    study: str|None=None; frame: int|None=None; parts: list[str]=field(default_factory=list)
    value: float|None=None; location: list|None=None; extent: list|None=None
# codes: invalid_model, interference, static_interference, tight_clearance, loop_open, branch_jump,
#        singular_pose, joint_limit, over_capacity, underconstrained, near_planar, joint_off_part, target_miss,
#        held_at_home, gear_mesh, contact
def build_report(asm, kin, props: dict[str, MassProps], results: dict[str, StudyResult],
                 sweeps: dict[str, SweepResult], loads: dict[str, dict[str, dict]], home: list[PairResult],
                 *, roles: dict[str, str], viewer_url: str | None, prev: dict | None = None,
                 partial: dict | None = None, out_dir: str | None = None) -> dict
def invalid_report(name, errors: list[str], params: dict, partial: dict | None = None) -> dict   # "INVALID"
def format_summary(report: dict, verbose: bool = False) -> str
```
Dicts are keyed by study name in run order. Issue messages carry actionable detail:
- interference: `0.84 mm³ overlap 0.4×2.1×4.0 mm @ arm1 (29.3, 2.0, 6.7), post (27.2, 2.0, 12.7)` (extent +
  the spot in each part's home geometry — `PairResult.location` / `location_b`; one point when both
  agree, e.g. at home; `Issue.location` is part a's). An `allow_contact` pair deeper than its `max_depth`
  adds `— allowed contact deeper than max_depth 0.1 mm (mean depth 0.282): fix the fit, or raise max_depth /
  max_depth=None if the overlap is intended (belt teeth)`
- tight: gap, closest-point midpoint (each part's home geometry, as above) and direction a→b (a's frame)
- contact (INFO, one per `allow_contact` pair overlapping within its `max_depth`, the deepest over every
  study and the home pose): `rod/bush mean depth 0.0498 mm, 7.90 mm³ @f0 (allowed ≤ 0.1 mm)`
- singular_pose (WARN, one per study with closed frames in `StudyResult.singular`, value = frame count; a
  frame 0 at a singular home pose is left to the `underconstrained` INFO): `study 'turn' passes a
  change/dead point at f12 @ j_crank=180° (2 frames f12, f30) — the loop Jacobian is singular there, so the
  real mechanism can take either assembly branch; mech kept the branch of the frames before`
- loop_open: driver sub-range that fails and the range that closes (`open for j_crank 71.3…90° (f47–f60), max residual 3.2 mm; closes 0…71.3°`;
  several drivers: `open for j_a 10…20°, j_b 30…40° (f3–f5); …`)
- joint_limit: `j_rocker 52.1° > 45 limit @ j_crank=80°`
- the pose of a frame (`f12 j_crank=80.0°`) names every study driver that moves the issue's parts
  (on a part's joint chain, or acting on its loop/coupling; all drivers when the parts are unknown),
  with the coupled outputs they drive in parentheses: `f12 j_pan_motor=−502°, j_tilt=84.0° (j_pan=−167°)`.
- `branch_jump` is only reported between two closed frames (a jump into or out of an open frame is the
  loop failing, already a `loop_open`).
- `held_at_home` is judged structurally (a loop moves when a driver of the studies run acts on it, a
  coupled joint when its coupling root moves), never from sampled values; in a partial run its
  message reads `— not driven by the selected studies`.

Joint ranges and probe stats cover the closed frames only, with the same series as the targets
(`targets.joint_series` / `path_length`), so a ranges line, a probe path and a span target agree.

report.json:
```json
{"name","status":"PASS|WARN|FAIL|INVALID","params":{},
 "partial": null | {"studies":[run],"skipped":[not run],"frames":N|null},
 "parts":8,
 "joints":{"driver":1,"coupled":0,"passive":2,"free":0,"fixed":1},"roles":{"j":"driver"},"mobility":{"study":0},
 "mass":{"total_g","com_mm":[..],"parts":{"base":120.1}},
 "issues":[Issue...],
 "targets":[...],
 "studies":[{"name","frames","joint_ranges":{"j":[min,max]},
             "probes":{"p":{"min":[..],"max":[..],"start":[..],"end":[..],"path_mm"}},
             "min_clearance":{"parts":[a,b],"value","frame","at":{"j":value}[, "subframe"]} | null,
             "loads":{"j":{"unit","max_abs","frame","capacity","sf","reflected_from"[, "why"]}},
             "max_residual"}],
 "clearance":{"required","pairs_checked","allowed_contact"},
 "home_min_clearance":{"parts":[a,b],"value"} | null,   // signed like min_clearance, pairs of different rigid groups
 "viewer_url": "...", "out_dir": null | "<abs export root other than <repo>/output>",
 "delta_prev": {...}}
```
`roles` come from every study of the model (a joint driven only by a skipped study is still a
driver); `mobility` lists the studies run. A load's `why` (only when `max_abs` is null) is
`no closed frame`, `N DOF undetermined` (the study leaves loop DOF free) or `singular at every
closed frame`. `min_clearance.at` = the values of the drivers that move the pair at that frame;
`value` is the sweep's `min_clearance` as reported (the summary marks a negative value, an
interfering pair's overlap depth, as such); `subframe` (e.g. 1.5) is set when the sweep found it between two frames (filed at the nearer
one), and `at` then holds the drivers interpolated to that pose. Issue locations do the same:
`· between f1–f2 j=45.0°`.
Status = worst issue severity (FAIL > WARN > PASS; INFO doesn't count).

**format_summary** (the token budget; ≤ 15 lines default, `verbose` lifts the cap):
fixed line order — header · FAIL · target_miss · WARN · INFO · loops · clearance · loads · targets ·
ranges · Δprev · view. The loops line counts branch jumps and singular frames (`!!` on an open loop, a
branch jump, a singular frame or leftover mobility). A partial run says so in the header: `mech x — PASS (partial run: --study back,
skipped sweep; --frames 12)`. Dedup on code + subject (part pair; the joint of over_capacity /
joint_limit / branch_jump; the pin of loop_open) across studies (keep worst, list study names). 3
significant figures. Over budget: drop from the bottom of each tier, emit
`(+N more: 2 WARN tight, 1 INFO — --verbose)`.
The clearance line is always printed: `clearance min <gap> mm (<a>/<b> @f<k> <every driver>=<value>) ·
required <clearance> · N pairs checked · M allowed-contact pairs` (pairs checked = part pairs that move
relative to each other, not ignored; allowed-contact = `allow_contact` + gear/rack meshing pairs).
Loads: `load <j> max … @f<k> · capacity … → SF …`; joints whose load is exactly 0 (axis ∥ gravity, or
balanced) fold into one `no gravity load on j_a, j_b (axes ∥ g or balanced)` line (rated actuators
only); an undefined load says why (`n/a (no closed frame | N DOF undetermined | singular at every
closed frame)`). Targets: met ones with value and bound, then `not evaluated: <labels>`.
Ranges collapse across studies into one line (closed frames only). Probe Δ = max−min per axis.
Issue locations name every driver's value at the frame, coupled outputs in parentheses
(`f12 j_pan_motor=−502°, j_tilt=84.0° (j_pan=−167°)`).
`Δprev` compares to the previous report (read before overwrite): status, changed build() params,
fixed/new issues, added/removed targets, min clearance, mass, target values — or `Δprev: first run`.
It compares like with like only: studies present in both runs *at the same frame count* (the rest
are listed as `not compared: sweep (not run), turn (72→12 frames)`) plus study-independent issues,
and targets evaluated over the same studies; the status is compared only when both runs sampled
the same studies. Lists longer than 3 end in `(+N more — --verbose)`.
Final line: targeted viewer URL — `...&issue=0&ui=0` when any FAIL/WARN else
`...&ghost=6&layout=quad&ui=0`; plus `shot: uv run mech shot <name>` hint (with `--output-dir <dir>`
when exported outside `<repo>/output`, which the viewer does not serve); a partial run says it was
not exported. Sample — the real first-run output of `uv run mech run examples/four_bar.py` (a test
keeps it so):
```
mech four_bar — PASS   4 parts · 3 joints (1 driver, 2 passive) · 1 loop · 42.4 g · CoG (60.8, 22.5, 3.6)
OK   loop p_B closed (max 6.96e-14 mm) · no branch jumps · mobility 0
clearance min 13.0 mm (frame/coupler @f0 j_crank=0°) · required 0.3 · 6 pairs checked
no gravity load on j_crank (axis ∥ g or balanced)
targets 1/1 · rocker swing span:j_rocker 62.1 ≥ 40
ranges j_crank 0…360° · j_coupler −360…0° · j_rocker −13.2…48.9° · probe mid Δ(71.1, 58.0, 0) path 197 mm
Δprev: first run
view http://localhost:3000/mech.html?m=four_bar&ghost=6&layout=quad&ui=0 · shot: uv run mech shot four_bar
```

### 4.10 export.py
```python
def export_scene(asm, kin, props, results, sweeps, loads, report, out_root: Path, *,
                 roles, tolerance=0.05, step=False) -> Path       # returns out_root/<slug>.mech/
```
Writes `parts/<slug>.stl` (binary, world mm Z-up, `tolerance` + `angular_tolerance=0.2` explicit;
verify file size == 84 + 50·n_tri; any failure raises `ExportError` naming the part and leaves no
`*.tmp.stl` — the CLI prints it as one line, `mech: export failed: …`, exit 3), `report.json`, `scene.json`, optional `assembly.step`,
and removes a stale `report.partial.json`. scene.json/report.json always describe the last *full* run
(every study at its declared frame count): a partial run writes only `report.partial.json`
(`write_report(report, out_root, partial=True)`), and an INVALID full run writes its `report.json` and
calls `clear_scene(out_root, name)` (removes scene.json, part STLs, assembly.step and
report.partial.json), so no viewer or `mech shot` ever shows a model that no longer exists.
```json
{"version": 1, "name", "units": "mm", "up": [0,0,1], "clearance",
 "parts": [{"id","color","opacity","mesh":"parts/<slug>.stl","ground","mass_g","material","bom","bbox":[[..],[..]]}],
 "joints": [{"name","kind","parent","child","origin","axis","limits","home","role"}],
 "pins": [{"name","a","b","point","axis"}], "probes": [{"name","part"}],
 "studies": [{"name","frames","duration","loop","t":[..],"joints":{"j":[..]},
              "transforms":{"part":[[16],..]},          // omitted when T ≈ I (1e-9) in every frame
              "probes":{"p":[[x,y,z],..]},
              "issues":[{"frame","a","b","status","distance","volume","pa","pb"}],
              "loads":{"j":[..]},"residual":[..]}],
 "report": {...}}
```
Viewer uses `parts[].mesh` only (never builds paths from ids). Issue `frame: null` = home pose.

### 4.11 runner.py, sweep.py, shot.py, cli.py
```python
def analyze(asm, *, studies=None, frames=None, export=True, out_root=None, step=False,
            viewer_base="http://localhost:3000", params=None, progress=None) -> dict
            # progress: one stderr line per study, `study pan: 69 frames … 4.2 s` (None = if stderr is a TTY)
def run(asm, **kw) -> dict          # analyze + print(format_summary)
```
`out_root=None` means `<repo>/output` (`runner.OUTPUT_DIR`) whatever the working directory, the
directory the viewer and `mech shot` read. A run with `studies` or `frames` is **partial**: the report
carries `partial` (studies run, skipped, frame override), its Δprev compares against
`report.partial.json` when that holds the same study scope (else `report.json`, compared on the
studies sampled alike), and it never exports a scene or replaces `report.json` (see §4.10).
Order (runner is the only orchestrator): Kinematics(asm) [ModelError → `invalid_report`] →
studies (declared or default) → filter `studies` → `frames` override → `check_study` (errors →
INVALID) → run_study → sweep → gravity_loads → `home = checker.check_pose({p: I})` → roles_for →
build_report (with prev report) → evaluate_targets → export.

CLI (`uv run mech …`; stdout reconfigured to UTF-8; exit INVALID 3, FAIL 2, WARN 1, PASS 0):
- `mech run <script.py> [-p k=v ...] [--study NAME ...] [--frames N] [--no-export] [--step] [--verbose] [--json]`
  imports the script by path (importlib; script dir + repo root on sys.path); `-p` values via
  `ast.literal_eval` (fallback: string) passed as `build(**params)`; unknown param → exit 3 listing valid ones.
  Script exceptions: print only frames from the user script + final exception line (≤ 6 lines); full with `--verbose`.
  `--verbose` also prints the per-study progress on stderr (as does any run on a terminal).
- `mech check <script.py> [-p ...]` — validate + roles/mobility line + home-pose clearance; no studies/export.
- `mech sweep <script.py> k=a:b:step k=v1,v2 ... [--frames N] [--export-best]` — grid product (cap 50
  variants), no export, one row per variant: params | status | #FAIL/#WARN | min clearance | worst SF |
  each target value | why (first FAIL/WARN issue: code + subject). SF of an unloaded rated actuator
  prints `∞` (as in `mech run`), `–` = no value. ≤ ~20 rows; best = status, fewest FAIL/WARN, largest min
  clearance, largest worst SF (a variant without an SF ranks below a measured one). `--export-best` exports the winner as a
  full run (declared frame counts, even with `--frames`) and exits with that run's status.
- `mech shot <name> [--issue i] [--view iso|top|front|right] [--ghost N] [--layout quad] [--frame N] [-o out.png]`
  — if playwright is importable: serve `previewer/dist` + `output/` from a Python `http.server` on a free
  port (runs `npm run build` in previewer if dist is missing), open `mech.html` with the params, wait for
  `window.__mechReady` (or fail fast on `window.__mechError`), screenshot the canvas to
  `output/<slug>.mech/shot_*.png`, print the path. Otherwise print the URL and how to install playwright.
- `mech list` — `output/*.mech` with the status of the last full run (`partial run only` when only
  `--study`/`--frames` runs exist).
- `mech shot` refuses (exit 3) when the last full run was INVALID or when scene.json's embedded report
  differs from report.json (stale scene).

---

## 5. Validation & correctness requirements (issue codes)
- `invalid_model` (FAIL, status INVALID): anything from `validate()` / `check_study()`; no raw tracebacks.
- `underconstrained` WARN: `mobility(driven) > 0` for a study. INFO when home is singular but generic rank is full.
- `overconstrained` drivers → INVALID for that study with a message naming drivers + loop.
- `near_planar` WARN from `validate_warnings()`.
- `joint_limit` WARN: coupled/passive joint outside limits in any frame.
- `loop_open` FAIL: any frame `ok=False`. `branch_jump` WARN. `singular_pose` WARN: closed frames on a
  change/dead point of the driven loops (§4.9).
- `over_capacity` FAIL when max_abs > capacity; WARN when SF < 1.5.
- `static_interference` per §4.6. `gear_mesh` INFO once per auto-allowed meshing pair; WARN for a
  gear/rack coupling with no meshing pair (§4.1). `contact` INFO per `allow_contact` pair overlapping
  within its `max_depth`.
- `held_at_home` INFO: joints never driven by any study (structurally: loop joints of a driven loop and
  coupled joints of a driven chain are driven), or in a partial run by the selected studies.

## 6. Standard parts library `mech.parts`
```python
@dataclass
class LibPart:
    shape: Shape; bom: str; mass_g: float | None = None
    frames: dict[str, Location] = field(default_factory=dict)
    kind: str | None = None     # "gear" (spur_gear, gear_pair) | "rack" | "pulley" (gt2_pulley) | None
    def moved(self, loc: Location) -> "LibPart"        # shape + frames (+ kind)
    def __rmul__(self, loc: Location) -> "LibPart"     # Pos(...) * Rot(...) * nema17()
    def mate(self, frame: str, target: Location) -> "LibPart"   # move so frames[frame] lands on target
```
Real standard dimensions, simplified geometry (no threads). **Never call bd_warehouse.**
- `fits.py`: ISO 286 `fit(nominal, "H7/g6") -> {"hole":(min,max),"shaft":(min,max),"clearance":(min,max)}`
  for 1–120 mm, fits H7/h6, H7/g6, H7/f7, H8/f7, H7/k6, H7/p6, H11/c11 (e.g. 10 H7/g6: hole 10.000–10.015,
  shaft 9.986–9.995). `fdm_hole(d, clearance=0.2)`, `clearance_hole(size, fit="normal")` (ISO 273),
  `tap_drill(size)`, `heat_set_insert(size)`.
- `fasteners.py`: `socket_head_screw(size, length)` (ISO 4762), `button_head_screw`, `hex_nut` (ISO 4032),
  `washer` (ISO 7089), M2–M8. Axis +Z, head underside at z=0, shank toward −Z. Frames `"head"`, `"tip"`.
- `bearings.py`: `bearing(designation)` — 623, 624, 625, 626, 608, 688, 6000–6005, 6200–6202; LM8UU/LM10UU.
  Axis Z, centered at origin. Frame `"center"`.
- `motors.py`: `nema17(length=40)`, `nema23(length=56)`, `sg90()`, `mg996r()`, `n20_gearmotor()`.
  Mounting face at z=0, body toward −Z, shaft +Z. Frames `"shaft"` (+Z at the face) and `"hole_1".."hole_4"`;
  servos also `"horn"` (the spline tip: z=14.0 SG90, 16.4 MG996R).
- `gears.py`: own involute (6–8 samples per flank + root, Polyline → face → extrude; ~0.15 s).
  `spur_gear(module, teeth, width, *, bore=0, backlash=0.05, pressure_angle=20)` axis +Z through origin,
  bottom z=0, a tooth centered on +X; frame `"axis"`.
  `gear_pair(module, z1, z2, width, *, backlash=0.05, bore1=0, bore2=0) -> GearPair(g1, g2, ratio, center_distance)`
  (NamedTuple): g2 at (m(z1+z2)/2, 0, 0) rotated so a tooth space faces −X; ratio = −z1/z2; both axes +Z;
  home-pose common volume must be 0. `rack(module, length, width, height)`, `gt2_pulley(teeth, width, bore)`
  (pitch dia 2·teeth/π).
- `motion_parts.py`: `rod(d, length)`, `t8_leadscrew(length)` (lead 8), `t8_nut()`, `extrusion_2020(length)`,
  `extrusion_2040(length)`.
- `__init__.py` re-exports all + `LibPart`, `GearPair`.

## 7. Viewer (`previewer/mech.html`, `previewer/src/mech/*.js`, three ~0.128, vite port 3000)
- Vite middleware: `/output` serves correct content types (STL `application/octet-stream`, JSON, PNG);
  `/api/mechs` lists `output/*.mech/` with status from report.json; keep the path-traversal guard.
- The viewer must also work as a **static build** (`npm run build` → `previewer/dist`, multi-page: index +
  mech) served by a plain static server with `output/` at `/output/` (used by `mech shot`) — so
  `/api/mechs` is optional (fall back to `?m=` only).
- Z-up (`camera.up = (0,0,1)`), OrbitControls, XY grid, fit-to-bbox.
- Parts: STL at home; per frame `mesh.matrix = T` (matrixAutoUpdate off); missing transform = identity.
- Timeline: study dropdown, play/pause, scrubber, speed; wraps (pingpong frames are pre-expanded).
- Issues: interference red, tight orange at the current frame; segment + spheres pa→pb (see-through,
  a fixed ~5 px on screen in every pane whatever the scene size or zoom, so they never hide the
  overlap they mark; the selected issue's overlap box is drawn at its true extent); issues panel
  from `report.issues` — click → jump to frame, isolate + focus pair; `frame: null` → home pose.
- A missing scene after an INVALID run, or a scene.json whose embedded report differs from
  report.json (stale), is a load error (`window.__mechError`), never an old model shown as current.
- Probe polylines; joint-axis arrows toggle; explode slider; section plane (X/Y/Z + offset);
  part list (visibility, mass); HUD (joint values, loads).
- URL params: `m, study, frame, q=j:45` (nearest frame by joint value), `hide, isolate, focus=a,b`
  (fit camera to those parts' current bbox), `issue=i` (study+frame+isolate+focus from report.issues[i],
  draw pa/pb), `section=x:10, explode=0.5, axes=1, view=iso|top|front|right|left|back|bottom, cam=az,el,
  ghost=N` (N evenly spaced poses as translucent overlays), `layout=quad` (iso/top/front/right in a
  2×2 viewport grid on one canvas), `ui=0` (canvas only, no panels/HUD).
- `window.__mechReady = true` after meshes load and the requested state renders; on any failure
  `window.__mechError = "<message>"`.
- The existing manifest previewer (`index.html`, `src/main.js`) keeps working; add links between the two.

## 8. Examples & tests (analytic truth — these gate "done")
Examples live in `examples/`, each runnable via `uv run mech run examples/x.py`:
- `four_bar.py` (§2, byte-identical). Test: rocker angle vs crank angle equals Freudenstein closed form on
  the drawn branch within 1e-3°; residual < 1e-6 mm; target met; PASS.
- `slider_crank.py`: offset 0, crank r=30 drawn at 0°, rod l=90, slider home = r + l (so x = q_slider).
  Test: slider x = r·cosθ + √(l² − r²sin²θ) within 1e-4 mm (compare probe or q).
- `gear_train.py`: `nema17` + `gear_pair(1, 15, 30, 6)` + output shaft in a 625 bearing. Test: ratio exact
  (output = −½ input), PASS, no interference, `gear_mesh` INFO.
- `leadscrew_stage.py`: vertical T8 screw (revolute driver) + carriage (prismatic, `screw` coupling) on LM8UU
  rods. Tests: 360° → 8 mm (direction per right-hand rule); j_leadrot holding torque =
  m·g·0.008/(2π) N·m within 1%; j_carriage reflected load = m·g N within 1% (m = mass moving with carriage).
- `pendulum_arm.py`: arm along +X, axis (0,−1,0) so +θ raises it, θ from horizontal; τ = +m·g·r·cosθ with
  |τ − τ_exact| ≤ 0.01·m·g·r; actuator capacity check produces over_capacity when capacity is too low.
- `hinged_box.py(gap=…)`: default gap → known `tight` WARN; `gap=-1` variant → interference FAIL with
  overlap volume within 2% of analytic.
- `parallel_gripper.py` (promoted dogfood design, ≤ 60 lines): SG90 + `gear_pair` (ratio −1) + two
  parallelogram jaws. Tests: j_idler = −j_drive; jaws translate (j_jaw = −j_drive, callable
  `jaw_tilt` ≤ 1e-9°); pads meet when drawn closed and are exactly `gap_open` apart at the end of the
  stroke (also for `L=35, open_deg=110`, where the bound is met by construction: no float-noise miss).
- `scissor_lift.py` (promoted dogfood design, ≤ 60 lines): NEMA17 + T8 screw pulling the sliding pivot
  of a two-stage scissor. Tests: j_slide = −(8/360)·j_leadrot; deck travel = 2L(sin θ1 − sin θ0); slide
  stroke = L(cos θ0 − cos θ1); with a dominant payload W the slide force = 2·W·cot θ0 and the screw
  torque = F·lead/(2π) within 1%.
Unit tests: steel-box mass props vs analytic inertia (incl. rotated assembly); ISO fits spot checks;
clearance statuses incl. **containment (4 mm cube inside 20 mm cube → interference 64 mm³)**; cache
correctness (pa/pb after rigid repeat); coupling direction per kind; validation messages with
suggestions; CLI exit codes 0/1/2/3; JSON has no NaN/Infinity; slug on Windows-reserved names.
