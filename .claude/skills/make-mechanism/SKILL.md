---
name: make-mechanism
description: Design and verify moving assemblies (linkages, hinges, gear trains, lead screws, servo arms) with the mech framework — kinematics, loop closure, clearance/interference sweeps, mass, gravity holding loads and design targets from one short build123d model script
user_invocable: true
---

# make-mechanism: model → `mech run` → read ~10 lines → iterate

Use this skill when something **moves**: linkages, hinges and lids, sliders, gear trains, lead-screw
stages, servo/stepper arms, or any question about range of motion, collisions during motion,
clearances, holding torque/force, actuator sizing or mass. For a static part, use `/make-model`.

You write only a short **model script** (geometry + joints + intent). `mech` does the rest
deterministically and prints a ≤ 15-line summary. Do not re-derive kinematics or clearances
yourself; do not open STL/JSON outputs; do not screenshot unless a picture is needed.

**First, read `references/mech-api.md`** (call table, conventions, parts library, metric
vocabulary, issue codes with fixes, common mistakes). Templates live in `examples/`.

---

## 1. Clarify (one round max, default aggressively)

Ask only what is missing and would change the design:
- **What moves and what drives it** (motor, servo, hand, spring) and the required motion
  (stroke mm, swing °, turns, output speed ratio).
- **Key dimensions / envelope** and any fixed interfaces (motor model, bearing, rod size).
- **Requirements** worth encoding as targets: minimum travel, clearance, payload mass,
  actuator capacity / safety factor, mass limit.
- Manufacturing if it matters (FDM → 0.2–0.3 mm running clearances; `fit()` for machined).

Skip questions about colors, file formats, or anything with a sane default.

## 2. Write the script from the closest template

| Mechanism | Template |
|---|---|
| planar linkage, loop closed by a pin | `examples/four_bar.py` (canonical) |
| crank → linear slider | `examples/slider_crank.py` |
| motor + gear pair + shaft in a bearing | `examples/gear_train.py` |
| lead screw carriage on linear rods | `examples/leadscrew_stage.py` |
| actuator lifting a payload (loads, SF) | `examples/pendulum_arm.py` |
| lids/doors on hinges, clearance at closure | `examples/hinged_box.py` |

Copy it to `output/<name>.py` and adapt. Rules that matter most (details in the reference):
- `def build(<every tunable as a keyword default>) -> Assembly` and the
  `if __name__ == "__main__": run(build())` footer. Keep it short (~20–50 lines).
- **Model every part in place**, in world mm at the **home pose** you draw. No local frames.
  Use `mech.geom.link` / `circle_intersect` to draw linkages already closed.
- One ground (`ground=True`) body; every other part is the child of exactly **one** joint
  (`revolute`/`prismatic`/`fix`). Close loops with `pin`, never a second joint.
- Choose each joint axis so **+value means the natural direction** (open, raise, extend).
- Put intent in the script: `actuator` (capacity), `study` (what to sweep), `target`
  (requirements), `probe` (points whose path you care about), `allow_contact` for nominal
  line-to-line fits between parts that are not joined.
- Prefer `mech.parts` (motors, bearings, gears, screws, rods, extrusions) with their `frames`
  (`at=motor.frames["shaft"]`) over hand-modeled stand-ins.

## 3. Check, run, read

```bash
uv run mech check output/<name>.py          # validate + roles/mobility + home clearance (fast)
uv run mech run output/<name>.py            # full analysis + export for the viewer
uv run mech run output/<name>.py -p crank=35 gap=0.4   # try params without editing
```

Exit codes: **0 PASS · 1 WARN · 2 FAIL · 3 INVALID/error**. Read the summary top-down:

```
mech pendulum_arm — FAIL   3 parts · 2 joints (1 driver, 1 fixed) · 178 g · CoG (70.8, −11.0, 0)
FAIL over_capacity j_shoulder holding load 0.129 N·m @f18 (j_shoulder=0°) exceeds capacity 0.1 (SF 0.774) — bigger actuator, gearing, or counterbalance
FAIL target_miss servo margin: sf:j_shoulder 0.774 < min 2
OK   open chain (no loops) · mobility 0
load j_shoulder max 0.129 N·m @f18 · capacity 0.1 → SF 0.774
targets 0/1
ranges j_shoulder −90.0…90.0° · probe tip Δ(130, 0, 260) path 408 mm
Δprev: status PASS→FAIL · new over_capacity arm, target_miss servo margin · servo margin 8.36→0.774
view http://localhost:3000/mech.html?m=pendulum_arm&issue=0&ui=0 · shot: uv run mech shot pendulum_arm
```

Line order: header (status, counts, mass, CoG) · FAIL · target_miss · WARN · INFO · loop line
(`OK loop p_B closed (max 7e-14 mm) · no branch jumps · mobility 0`; `!!` = open loop or
leftover mobility) · loads (max holding load, frame, capacity → SF) · targets · ranges (joints,
probe Δ and path) · Δprev · view.

- Issue lines carry the numbers you need (overlap volume/extent/location, gap + direction,
  failing drive sub-range, offending frame and joint values). Act on them directly — see
  "Issue codes" in the reference for the usual fix per code.
- `Δprev` tells you whether the last edit helped. `(+N more … — --verbose)` means lines were
  cut; rerun with `--verbose` only if the hidden ones matter.
- `INVALID` lists every model error in one pass with did-you-mean hints; fix all, rerun.

## 4. Iterate (cap: 6 runs)

Loop: read summary → change **one** thing (a param via `-p`, then bake it into the defaults;
or geometry) → rerun → confirm via `Δprev`. Typical moves:
- interference / tight → move or resize the part at the reported location/direction, add
  clearance, reorder link layers (z), or limit the study range.
- loop_open → the drive range is out of reach: fix link lengths or restrict the drive.
- over_capacity / target_miss on SF → bigger actuator, gearing, shorter arm, lighter payload.

Use `uv run mech sweep output/<name>.py k=a:b:step k2=v1,v2` (inclusive ranges, grid ≤ 50
variants, no export) when the right value isn't obvious: one row per variant (status, F/W
counts, min clearance, worst SF, each target value) plus a `best …` line — far cheaper than
successive runs. `--export-best` exports the winner.

Stop when PASS with targets met (WARNs explained or accepted), or after 6 runs: then report
what still fails, why, and the options — don't keep guessing.

## 5. Pictures only when needed

`uv run mech shot <name>` renders the report's targeted view (first FAIL/WARN issue, else a
ghosted quad view) to `output/<name>.mech/shot_*.png` and prints the path; **read the PNG**.
Options: `--issue i`, `--view iso|top|front|right|left|back|bottom`, `--ghost N`,
`--layout quad`, `--frame N`, `-o file.png`. Use it to present the design, or when geometry
placement is in doubt (a part far from where you expected, a surprising interference). The
first shot after viewer changes rebuilds `previewer/dist` (a few seconds).

Interactive viewer for the user: `cd previewer && npm run dev`, then open the `view` URL.

## 6. Present the result

Keep it short:
1. **Status** and the few numbers that answer the user's question (travel/swing, min
   clearance, peak holding load and SF, mass), each target with its margin.
2. What you changed across iterations and why (one line each), and any accepted WARN.
3. A screenshot if you took one, the viewer URL (needs `npm run dev`), and the files:
   `output/<name>.py` (source of truth) and `output/<name>.mech/` (scene, report, STLs;
   `--step` adds `assembly.step`).
4. Assumptions (materials, clearances, frictionless quasi-static loads) and next options.

Scope: kinematics, clearances, mass properties, quasi-static gravity loads, targets, variant
sweeps. Not dynamics, friction, FEA or electronics; say so if the user needs those.
