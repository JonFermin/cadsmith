# scad-forge

- When the user asks to model or design a physical object, use the /scad-forge skill
- Always save .scad output to ./output/ with a descriptive filename
- After generating the .scad, also generate a scene manifest JSON at ./output/<name>_manifest.json for the Three.js previewer
- Run the previewer with: cd previewer && npm run dev
- After generating files, the skill captures a screenshot of the preview using Playwright and self-evaluates before presenting to the user
- Screenshots are saved to ./output/<name>_preview.png
- To export for Godot: `./pipeline/scad-to-godot.sh output/<name>.scad -f glb`
- Pipeline supports: stl, obj, glb, gltf output formats
- Use `-d 0.5` to decimate mesh, `-u` for UV maps, `-b` to force Blender processing
