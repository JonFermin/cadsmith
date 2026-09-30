# cadsmith

- When the user asks to model or design a physical object, use the /make-model skill (CadQuery/build123d)
- Always save output to ./output/ with a descriptive filename
- After generating the model, also generate a scene manifest JSON at ./output/<name>_manifest.json for the Three.js previewer
- Run the previewer with: cd previewer && npm run dev
- After generating files, the skill captures a screenshot of the preview using Playwright and self-evaluates before presenting to the user
- Screenshots are saved to ./output/<name>_preview.png
- To review and fix an existing model, use the /edit-model skill — it runs the reviewer/fixer agent loop on both the source file and manifest
- To export a model, use the /export-model skill or run directly: `./pipeline/scad-to-godot.sh output/<name>.stl -f glb`
- Pipeline accepts .stl, .step, or .py input and supports stl, obj, glb, gltf output formats
- Use `-d 0.5` to decimate mesh, `-u` for UV maps, `-b` to force Blender processing

## Mechanisms (`mech`)

- When the user asks for something that moves (linkage, hinge, lid, gear train, lead screw, servo/motor arm) or about range of motion, collisions, clearances, holding torque or actuator sizing, use the /make-mechanism skill (read its `references/mech-api.md` first)
- Write the model script to `./output/<name>.py` starting from the closest template in `examples/`; `build()` takes every tunable as a keyword default
- Loop: `uv run mech check output/<name>.py` → `uv run mech run output/<name>.py` → read the ≤15-line summary → edit → rerun (`-p k=v` for quick param changes, `uv run mech sweep` for grids)
- `--study NAME` / `--frames N` runs are partial: not exported, targets needing skipped studies are "not evaluated" — the verdict (and what you report) comes from a full run
- Exit codes: PASS 0, WARN 1, FAIL 2, INVALID 3. Results land in `./output/<name>.mech/` (scene.json, report.json, parts/*.stl)
- `uv run mech shot <name>` only when a picture is needed; read the PNG it prints. Interactive viewer: `cd previewer && npm run dev` → `/mech.html?m=<name>`
- Always `uv run` (Python 3.12 venv), never system Python or pip; never use bd_warehouse. Tests: `uv run pytest`
- `docs/MECH_SPEC.md` is the binding contract for the `mech` package
