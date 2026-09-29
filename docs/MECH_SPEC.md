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
  cube reports the gap to the outer wall). Use `shape.is_inside(pt)` (~0.2 ms) for containment.
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
| `gear(driver, driven, ratio, *, name=None)` | revolute→revolute; ratio as given assumes both axes point the same way (external mesh: negative) |
| `screw(driver, driven, lead, *, hand="right", name=None)` | revolute→prismatic; ratio = −s·lead/360 for right hand, +s·lead/360 left, s = sign(axis_driver·axis_driven) |
| `rack(driver, driven, pitch_radius, *, sign=None, name=None)` | revolute→prismatic; ratio = sign·π·r/180; sign default from geometry: sign((axis_p × (c − origin_p))·axis_r), c = driven part's bbox center |
| `couple(driver, driven, ratio=1.0, offset=0.0, *, name=None)` | generic linear coupling, native units |
| `pin(name, a, b, point, axis=None)` | loop closure: point (home world) is coincident on parts a and b. With `axis` = hinge (also axes stay aligned); without = ball joint |
| `probe(name, part, point)` | track a point's trajectory |
| `actuator(joint, capacity)` | N·m (revolute) / N (prismatic); also marks the preferred driver of a loop |
| `allow_contact(a, b)` / `ignore(a, b)` | expected-touching pair / skip pair entirely |
| `study(name, drive, *, frames=60, loop="once", duration=3.0)` | drive: `{joint: (start, end) \| [(u, v), ...] \| callable(u) -> value}` with u∈[0,1], values in deg/mm |
| `target(label, metric, *, min=None, max=None, study=None, severity="FAIL")` | design intent, see §2.3 |

Rules:
- `build(**params) -> Assembly` with **every tunable as a keyword default** (enables `-p` and `mech sweep`).
- Every part must be `ground=True` or the child of exactly one joint. Use `pin` (not a second joint) to
  close loops. A pin (its point, or for hinge pins the axis line through the point) must come within
  `Assembly.pin_tol` (default 5 mm — pins pass through holes) of both parts' geometry. Joints likewise
  should touch both parent and child (WARN `joint_off_part` otherwise).
- Any non-coupled, non-fixed joint may appear in a study's `drive` — including a joint on a loop
  (then the *other* loop joints are solved). Driving a coupled joint is an error.

### 2.2 Default studies (when the script declares none, or for joints not covered)
If the script declares **no** study: one study per *mobility group*:
- each non-loop, non-coupled joint: sweep its limits `once` 40 frames; revolute without limits: 0→360.
- each loop: drive the loop joint with an `actuator` (else first-declared loop joint with limits, else
  first-declared revolute loop joint 0→360); 60 frames.
Joints never driven by any study stay at home; `mech` prints one INFO naming them.

### 2.3 Targets
Metric vocabulary (evaluated per study; if `study=None`, over all studies — the worst value counts):
`span:<joint>` (max−min), `min:<joint>`, `max:<joint>`, `min_dist:<probeA>,<probeB>` /
`max_dist:<probeA>,<probeB>` (min/max distance between two probes over the study), `path:<probe>` (path length mm), `delta:<probe>.x|y|z` (max−min),
`clearance` (min clearance among checked non-joined pairs), `mass_g`, `load:<joint>` (max |load|),
`sf:<joint>` (capacity / max|load|), or a callable `f(report_dict) -> float`.
Result: `target_miss` issue with the target's severity when value is outside [min, max].

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
@dataclass class Joint: name; kind: Literal["revolute","prismatic","fixed"]; parent; child;
                        origin: np.ndarray; axis: np.ndarray; limits: tuple|None=None; home: float=0.0
@dataclass class Coupling: name; driver; driven; ratio: float; offset: float=0.0; kind: str="couple"   # gear|screw|rack|couple
@dataclass class Pin: name; a; b; point: np.ndarray; axis: np.ndarray|None=None
@dataclass class Probe: name; part; point: np.ndarray
@dataclass class Actuator: joint; capacity: float
@dataclass class Study: name; drive: dict; frames: int=60; loop: str="once"; duration: float=3.0
@dataclass class Target: label; metric: str|Callable; min: float|None; max: float|None; study: str|None; severity: str
class Assembly:
    name; clearance; interference_tol; pin_tol; gravity
    parts: dict; joints: dict; couplings: list; pins: list; probes: list; actuators: dict; studies: list
    targets: list; allowed: set[frozenset]; ignored: set[frozenset]; params: dict   # params filled by runner
    # methods per §2.1
    def validate(self) -> list[str]      # NEVER raises; collects ALL errors in one pass
    def parent_joint(self, part) -> Joint | None
    def rigid_groups(self) -> dict[str, int]      # union-find over fixed joints; all ground parts share one group
    def is_joined(self, a, b) -> bool   # same rigid group, OR groups connected directly by a joint or pin
    def meshing_pairs(self) -> set[frozenset]     # child parts of the two joints of every gear/rack coupling
```
`validate()` errors (each with a `difflib.get_close_matches` "did you mean" when a name is unknown):
unknown part/joint names anywhere; floating part; part with two parent joints; joint-tree cycle; a
joint driven by two couplings; coupling cycle; coupling kind mismatch (gear rev→rev; screw/rack
rev→pris); `home` outside `limits`; `pin_off_part`: pin farther than `pin_tol` from either part — for hinge
pins measure `shape.distance_to(Edge.make_line(p − L·n, p + L·n))`, for ball pins `shape.distance_to(Vector(p))` —
naming part + distance; zero-volume part with `mass_g` set; unknown material. Warnings (non-fatal) via `validate_warnings() -> list[tuple[code, msg]]`: near-planar loop (axes parallel
within 1e-3 rad but not 1e-9) → `near_planar`; joint axis line farther than `pin_tol` from its parent or
child geometry → `joint_off_part`.

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
fuse first; `mass_g` override scales mass and inertia. `assembly_props`:
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
    def mobility(self, driven) -> int            # generic rank of J_p: max over home + 3 small random perturbations
    def overconstrained(self, driven) -> list[str]   # messages if rank([J_p J_d]) > rank(J_p) on a loop
    def roles_for(self, studies) -> dict[str, str]   # fixed|coupled|driver (in any study)|passive (on loop)|free
    def solve(self, drive: dict[str, float], guess: dict[str, float] | None = None) -> Pose
    def jacobians(self, pose, driven) -> tuple[np.ndarray, np.ndarray]   # J_p, J_d of residual wrt unknowns/drivers
    def point(self, pose, part, p) -> np.ndarray
```
Loop residual per pin: `T_a·p − T_b·p` (3 rows, mm) plus, with axis, `L·(R_a n − R_b n)` (3 rows, mm).
`least_squares(method="trf", x_scale="jac", xtol=ftol=gtol=1e-12)`, unknown revolutes in radians;
couplings re-expanded inside every residual evaluation (a coupling's driver may be a passive joint).
Warm start from `guess`; optional predictor `q_p += −pinv(J_p)·J_d·Δq_d`. After solving, wrap each
revolute unknown into `guess ± 180°`. Rank by SVD with rtol 1e-9 on J with revolute columns scaled by L.
Never raise from `solve`; non-convergence → `ok=False`.

### 4.5 motion.py
```python
def sample_drive(study) -> tuple[np.ndarray, dict[str, np.ndarray]]   # t (s), per-joint values (§1 sampling)
def default_studies(asm, kin) -> list[Study]                          # §2.2
def check_study(kin, study) -> list[str]   # unknown/coupled/fixed joints in drive; keyframes u∉[0,1];
                                           # driver values outside limits; frames<2; overconstrained drivers
@dataclass class StudyResult:
    study; t: np.ndarray; drive: dict[str, np.ndarray]; poses: list[Pose]
    probes: dict[str, np.ndarray]                   # (frames, 3) world
    branch_jumps: list[tuple[int, str, float]]      # passive jump > 20° / 5 mm between consecutive frames
def run_study(asm, kin, study) -> StudyResult       # frame k warm-starts from k−1
```

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
    location: np.ndarray | None = None             # overlap/closest-point midpoint in HOME world (via inv(T_a))
class ClearanceChecker:
    def __init__(self, asm, clearance=None)
    def check_pose(self, transforms, *, include_rigid=True) -> list[PairResult]
    def sweep(self, result) -> "SweepResult"        # skips intra-rigid-group pairs (checked once at home)
@dataclass class SweepResult:
    per_frame: list[list[PairResult]]               # non-"ok" only
    worst: dict[frozenset, tuple[int, PairResult]]
    min_clearance: tuple[str, str, float, int] | None   # non-joined, non-allowed, non-meshing pairs
    stats: dict                                     # pairs_checked, cache_hits, seconds
```
Status rules: ignored → skipped. overlap > tol (or mean depth `2V/A_common` > 0.02 mm) → **interference**
unless allowed (allowed → contact). touching/overlap ≤ tol → **contact** (OK if joined/allowed/meshing,
else **tight**). 0 < d < clearance → **tight** unless joined/allowed/meshing. Else ok. Intra-rigid-group
interference at home → `static_interference` (WARN if the pair is fix-attached, FAIL otherwise).

Narrowphase: broadphase = transformed AABBs inflated by clearance. Exact `BRepExtrema_DistShapeShape`;
if distance > 1e-6 and one AABB is inside the other, containment test with `is_inside` on one vertex
each way → distance 0. Boolean common only when distance ≤ 1e-6. Meshing pairs: interference-only,
prefiltered with `BRepExtrema_ShapeProximity` on cached tessellations; boolean only on candidates.
Cache key `np.round(inv(T_a) @ T_b, 6).tobytes()`: compute with a at home and b moved by
`to_location(inv(T_a) @ T_b)`; store pa/pb/location in a's home frame, map through T_a on hit.
Target: 10 parts × 60 frames < 20 s excluding meshing pairs.

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
coupled joints in the study: `load_driver / (dq_c/dq_d in SI)`. Frictionless, backdrivable.

### 4.8 targets.py
`evaluate_targets(asm, report: dict, results) -> list[dict]` → `{"label","metric","value","min","max","met","severity","study"}`.

### 4.9 report.py
```python
@dataclass class Issue: severity: Literal["FAIL","WARN","INFO"]; code: str; message: str
    study: str|None=None; frame: int|None=None; parts: list[str]=field(default_factory=list)
    value: float|None=None; location: list|None=None; extent: list|None=None
# codes: invalid_model, interference, static_interference, tight_clearance, loop_open, branch_jump,
#        joint_limit, over_capacity, underconstrained, near_planar, joint_off_part, target_miss, held_at_home, gear_mesh
def build_report(asm, kin, props: dict[str, MassProps], results: dict[str, StudyResult],
                 sweeps: dict[str, SweepResult], loads: dict[str, dict[str, dict]], home: list[PairResult],
                 *, roles: dict[str, str], viewer_url: str | None, prev: dict | None = None) -> dict
def invalid_report(name, errors: list[str], params: dict) -> dict      # status "INVALID"
def format_summary(report: dict, verbose: bool = False) -> str
```
Dicts are keyed by study name in run order. Issue messages carry actionable detail:
- interference: `0.84 mm³ overlap 0.4×2.1×4.0 mm @ (41.2, 3.0, 10.0)` (extent + home-world location)
- tight: gap, closest-point midpoint (home world) and direction
- loop_open: driver sub-range that fails and the range that closes (`open for j_crank 71.3…90° (f47–f60), max residual 3.2 mm; closes 0…71.3°`)
- joint_limit: `j_rocker 52.1° > 45 limit @ j_crank=80°`

report.json:
```json
{"name","status":"PASS|WARN|FAIL|INVALID","params":{},"parts":8,
 "joints":{"driver":1,"coupled":0,"passive":2,"free":0,"fixed":1},"roles":{"j":"driver"},"mobility":{"study":0},
 "mass":{"total_g","com_mm":[..],"parts":{"base":120.1}},
 "issues":[Issue...],
 "targets":[...],
 "studies":[{"name","frames","joint_ranges":{"j":[min,max]},
             "probes":{"p":{"min":[..],"max":[..],"start":[..],"end":[..],"path_mm"}},
             "min_clearance":{"parts":[a,b],"value","frame"} | null,
             "loads":{"j":{"unit","max_abs","frame","capacity","sf","reflected_from"}},
             "max_residual"}],
 "viewer_url": "..."}
```
Status = worst issue severity (FAIL > WARN > PASS; INFO doesn't count).

**format_summary** (the token budget; ≤ 15 lines default, `verbose` lifts the cap):
fixed line order — header · FAIL · target_miss · WARN · loops/loads · targets · ranges · Δprev · view.
Dedup on code + part pair across studies (keep worst, list study names). 3 significant figures.
Over budget: drop from the bottom of each tier, emit `(+N more: 2 WARN tight, 1 INFO — --verbose)`.
Ranges collapse across studies into one line. Probe Δ = max−min per axis. `Δprev` compares to the
previous `report.json` (read before overwrite): fixed/new issues, min clearance, mass, target values —
or `Δprev: first run`. Final line: targeted viewer URL — `...&issue=0&ui=0` when any FAIL/WARN else
`...&ghost=6&layout=quad&ui=0`; plus `shot: uv run mech shot <name>` hint.
```
mech four_bar — PASS   4 parts · 3 joints (1 driver, 2 passive) · 1 loop · 38.2 g · CoG (48.1, 12.0, 1.9)
OK   loop p_B closed (max 2.1e-9 mm) · no branch jumps · mobility 0
load j_crank max 0.0412 N·m @f18 · capacity 0.5 → SF 12.1
targets 1/1 · rocker swing span:j_rocker 51.3 ≥ 40
ranges j_crank 0…360° · j_rocker −12.0…39.3° · probe mid Δ(62.1, 41.0, 0) path 187 mm
Δprev: first run
view http://localhost:3000/mech.html?m=four_bar&ghost=6&layout=quad&ui=0 · shot: uv run mech shot four_bar
```

### 4.10 export.py
```python
def export_scene(asm, kin, props, results, sweeps, loads, report, out_root: Path, *,
                 roles, tolerance=0.05, step=False) -> Path       # returns out_root/<slug>.mech/
```
Writes `parts/<slug>.stl` (binary, world mm Z-up, `tolerance` + `angular_tolerance=0.2` explicit;
verify file size == 84 + 50·n_tri else raise), `report.json`, `scene.json`, optional `assembly.step`.
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
def analyze(asm, *, studies=None, frames=None, export=True, out_root=Path("output"), step=False,
            viewer_base="http://localhost:3000", params=None) -> dict
def run(asm, **kw) -> dict          # analyze + print(format_summary)
```
Order (runner is the only orchestrator): Kinematics(asm) [ModelError → `invalid_report`] →
studies (declared or default) → filter `studies` → `frames` override → `check_study` (errors →
INVALID) → run_study → sweep → gravity_loads → `home = checker.check_pose({p: I})` → roles_for →
build_report (with prev report) → evaluate_targets → export.

CLI (`uv run mech …`; stdout reconfigured to UTF-8; exit INVALID 3, FAIL 2, WARN 1, PASS 0):
- `mech run <script.py> [-p k=v ...] [--study NAME ...] [--frames N] [--no-export] [--step] [--verbose] [--json]`
  imports the script by path (importlib; script dir + repo root on sys.path); `-p` values via
  `ast.literal_eval` (fallback: string) passed as `build(**params)`; unknown param → exit 3 listing valid ones.
  Script exceptions: print only frames from the user script + final exception line (≤ 6 lines); full with `--verbose`.
- `mech check <script.py> [-p ...]` — validate + roles/mobility line + home-pose clearance; no studies/export.
- `mech sweep <script.py> k=a:b:step k=v1,v2 ... [--frames N] [--export-best]` — grid product (cap 50
  variants), no export, one row per variant: params | status | #FAIL/#WARN | min clearance | worst SF | each target value. ≤ ~20 rows.
- `mech shot <name> [--issue i] [--view iso|top|front|right] [--ghost N] [--layout quad] [--frame N] [-o out.png]`
  — if playwright is importable: serve `previewer/dist` + `output/` from a Python `http.server` on a free
  port (runs `npm run build` in previewer if dist is missing), open `mech.html` with the params, wait for
  `window.__mechReady` (or fail fast on `window.__mechError`), screenshot the canvas to
  `output/<slug>.mech/shot_*.png`, print the path. Otherwise print the URL and how to install playwright.
- `mech list` — `output/*.mech` with status.

---

## 5. Validation & correctness requirements (issue codes)
- `invalid_model` (FAIL, status INVALID): anything from `validate()` / `check_study()`; no raw tracebacks.
- `underconstrained` WARN: `mobility(driven) > 0` for a study. INFO when home is singular but generic rank is full.
- `overconstrained` drivers → INVALID for that study with a message naming drivers + loop.
- `near_planar` WARN from `validate_warnings()`.
- `joint_limit` WARN: coupled/passive joint outside limits in any frame.
- `loop_open` FAIL: any frame `ok=False`. `branch_jump` WARN.
- `over_capacity` FAIL when max_abs > capacity; WARN when SF < 1.5.
- `static_interference` per §4.6. `gear_mesh` INFO once per auto-allowed meshing pair.
- `held_at_home` INFO: joints never driven by any study.

## 6. Standard parts library `mech.parts`
```python
@dataclass
class LibPart:
    shape: Shape; bom: str; mass_g: float | None = None
    frames: dict[str, Location] = field(default_factory=dict)
    def moved(self, loc: Location) -> "LibPart"        # shape + frames
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
  Mounting face at z=0, body toward −Z, shaft +Z. Frames `"shaft"` (+Z at the face) and `"hole_1".."hole_4"`.
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
- Issues: interference red, tight orange at the current frame; segment + spheres pa→pb; issues panel
  from `report.issues` — click → jump to frame, isolate + focus pair; `frame: null` → home pose.
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
Unit tests: steel-box mass props vs analytic inertia (incl. rotated assembly); ISO fits spot checks;
clearance statuses incl. **containment (4 mm cube inside 20 mm cube → interference 64 mm³)**; cache
correctness (pa/pb after rigid repeat); coupling direction per kind; validation messages with
suggestions; CLI exit codes 0/1/2/3; JSON has no NaN/Infinity; slug on Windows-reserved names.
