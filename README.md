# scad-forge

A Claude Code skill that generates parametric OpenSCAD (.scad) files and interactive Three.js 3D previews from natural language descriptions of physical objects. Describe what you want to make, and scad-forge produces a print-ready .scad file alongside a browser-based 3D preview you can orbit and inspect.

## Install

```bash
git clone git@github.com:JonFermin/scad-forge.git
cd scad-forge
```

The skill auto-loads when Claude Code runs from this directory.

**Portable install:** Copy `.claude/skills/scad-forge/` into `~/.claude/skills/` for global access from any project.

## Usage

From Claude Code:

```
/scad-forge make a phone stand
/scad-forge design a bracket for my shelf
/scad-forge create a cable organizer with 6 slots
```

Or just describe what you want in natural language — the CLAUDE.md hints Claude to use the skill when you mention modeling or designing physical objects.

## Previewer

The repo includes a standalone Three.js previewer for inspecting generated models:

```bash
cd previewer
npm install
npm run dev
```

This opens a browser at http://localhost:3000. Load a `_manifest.json` file via the file picker, drag & drop, or URL parameter.

## How it works

1. You describe a physical object
2. scad-forge asks brief clarifying questions (1 round max, defaults aggressively)
3. Generates a parametric `.scad` file in `./output/`
4. Generates a `_manifest.json` scene description for the previewer
5. Opens the Three.js preview in your browser

The `.scad` file is the source of truth for manufacturing. The Three.js preview is a fast visual approximation — difference() operations show as transparent red wireframes since true CSG isn't available in the browser.

## Project structure

```
scad-forge/
├── .claude/skills/scad-forge/    # The Claude Code skill
│   ├── SKILL.md                  # Skill definition
│   └── references/
│       └── openscad-guide.md     # OpenSCAD cheat sheet
├── previewer/                    # Three.js preview app (Vite)
├── output/                       # Generated .scad and manifest files (gitignored)
├── CLAUDE.md                     # Project-level Claude instructions
└── README.md
```

## License

MIT
