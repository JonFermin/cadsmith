#!/usr/bin/env bash
# scad-to-godot.sh — Export OpenSCAD models for Godot import
#
# Usage:
#   ./pipeline/scad-to-godot.sh <input.scad> [options]
#
# Options:
#   -o, --output-dir DIR     Output directory (default: ./output)
#   -f, --format FORMAT      Export format: stl, obj, glb, gltf (default: glb)
#   -b, --blender            Run Blender cleanup (auto-enabled for glb/gltf)
#   -d, --decimate RATIO     Blender decimate ratio 0.0-1.0 (default: 1.0 = no reduction)
#   -u, --uv                 Generate UV maps in Blender
#   -p, --params "KEY=VAL"   OpenSCAD parameter overrides (repeatable)
#   --no-blender             Skip Blender even for glb/gltf (just convert)
#   -h, --help               Show this help
#
# Prerequisites:
#   - OpenSCAD (openscad CLI)
#   - Blender 3.x+ (blender CLI) — only needed for --blender or glb/gltf output
#
# Examples:
#   ./pipeline/scad-to-godot.sh output/my_model.scad
#   ./pipeline/scad-to-godot.sh output/my_model.scad -f gltf -d 0.5 -u
#   ./pipeline/scad-to-godot.sh output/my_model.scad -f stl
#   ./pipeline/scad-to-godot.sh output/my_model.scad -p 'wall_thickness=3' -p 'height=50'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
OUTPUT_DIR="$PROJECT_DIR/output"
FORMAT="glb"
USE_BLENDER=""
DECIMATE_RATIO="1.0"
GENERATE_UV="false"
NO_BLENDER=""
SCAD_PARAMS=()
INPUT_FILE=""

usage() {
    sed -n '3,17p' "$0" | sed 's/^# \?//'
    exit 0
}

log() { echo "[scad-to-godot] $*"; }
err() { echo "[scad-to-godot] ERROR: $*" >&2; exit 1; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -f|--format)     FORMAT="$2"; shift 2 ;;
        -b|--blender)    USE_BLENDER="true"; shift ;;
        -d|--decimate)   DECIMATE_RATIO="$2"; USE_BLENDER="true"; shift 2 ;;
        -u|--uv)         GENERATE_UV="true"; USE_BLENDER="true"; shift ;;
        -p|--params)     SCAD_PARAMS+=("$2"); shift 2 ;;
        --no-blender)    NO_BLENDER="true"; shift ;;
        -h|--help)       usage ;;
        -*)              err "Unknown option: $1" ;;
        *)               INPUT_FILE="$1"; shift ;;
    esac
done

[[ -z "$INPUT_FILE" ]] && err "No input .scad file specified. Run with --help for usage."
[[ ! -f "$INPUT_FILE" ]] && err "File not found: $INPUT_FILE"
[[ "${INPUT_FILE##*.}" != "scad" ]] && err "Input must be a .scad file"

# Derive base name
BASENAME="$(basename "$INPUT_FILE" .scad)"
mkdir -p "$OUTPUT_DIR"

# Check tools
command -v openscad >/dev/null 2>&1 || err "OpenSCAD not found. Install: https://openscad.org/downloads.html"

# Determine if Blender is needed
NEEDS_BLENDER="false"
if [[ "$FORMAT" == "glb" || "$FORMAT" == "gltf" ]]; then
    NEEDS_BLENDER="true"
fi
if [[ "$USE_BLENDER" == "true" ]]; then
    NEEDS_BLENDER="true"
fi
if [[ "$NO_BLENDER" == "true" ]]; then
    NEEDS_BLENDER="false"
fi

if [[ "$NEEDS_BLENDER" == "true" ]]; then
    command -v blender >/dev/null 2>&1 || err "Blender not found. Install: https://www.blender.org/download/ (needed for $FORMAT export / mesh cleanup)"
fi

# Step 1: OpenSCAD → STL
log "Step 1: Exporting OpenSCAD → STL"

OPENSCAD_ARGS=(-o "$OUTPUT_DIR/${BASENAME}.stl")

# Add parameter overrides
for param in "${SCAD_PARAMS[@]+"${SCAD_PARAMS[@]}"}"; do
    OPENSCAD_ARGS+=(-D "$param")
done

OPENSCAD_ARGS+=("$INPUT_FILE")

log "  Running: openscad ${OPENSCAD_ARGS[*]}"
openscad "${OPENSCAD_ARGS[@]}"

STL_FILE="$OUTPUT_DIR/${BASENAME}.stl"
[[ ! -f "$STL_FILE" ]] && err "OpenSCAD export failed — no STL produced"

STL_SIZE=$(stat -c%s "$STL_FILE" 2>/dev/null || stat -f%z "$STL_FILE" 2>/dev/null)
log "  Exported: ${BASENAME}.stl ($(numfmt --to=iec "$STL_SIZE" 2>/dev/null || echo "${STL_SIZE} bytes"))"

# If only STL requested, we're done
if [[ "$FORMAT" == "stl" && "$NEEDS_BLENDER" == "false" ]]; then
    log "Done! STL ready at: $OUTPUT_DIR/${BASENAME}.stl"
    log "Import into Godot: drag the .stl into your Godot project's res:// folder"
    exit 0
fi

# Step 2 (optional): If OBJ requested without Blender
if [[ "$FORMAT" == "obj" && "$NEEDS_BLENDER" == "false" ]]; then
    log "Step 2: Exporting OpenSCAD → OBJ directly"
    openscad -o "$OUTPUT_DIR/${BASENAME}.obj" "${SCAD_PARAMS[@]+"${SCAD_PARAMS[@]/#/-D }"}" "$INPUT_FILE"
    log "Done! OBJ ready at: $OUTPUT_DIR/${BASENAME}.obj"
    log "Import into Godot: drag the .obj into your Godot project's res:// folder"
    exit 0
fi

# Step 2: Blender processing
if [[ "$NEEDS_BLENDER" == "true" ]]; then
    log "Step 2: Blender mesh processing"
    log "  Decimate ratio: $DECIMATE_RATIO"
    log "  Generate UVs: $GENERATE_UV"
    log "  Output format: $FORMAT"

    BLENDER_OUTPUT="$OUTPUT_DIR/${BASENAME}.${FORMAT}"

    blender --background --python "$SCRIPT_DIR/blender_process.py" -- \
        --input "$STL_FILE" \
        --output "$BLENDER_OUTPUT" \
        --format "$FORMAT" \
        --decimate "$DECIMATE_RATIO" \
        --uv "$GENERATE_UV"

    [[ ! -f "$BLENDER_OUTPUT" ]] && err "Blender export failed — no output produced"

    OUT_SIZE=$(stat -c%s "$BLENDER_OUTPUT" 2>/dev/null || stat -f%z "$BLENDER_OUTPUT" 2>/dev/null)
    log "  Exported: ${BASENAME}.${FORMAT} ($(numfmt --to=iec "$OUT_SIZE" 2>/dev/null || echo "${OUT_SIZE} bytes"))"
fi

# Summary
log ""
log "=== Pipeline Complete ==="
log "  Source:  $INPUT_FILE"
log "  STL:     $OUTPUT_DIR/${BASENAME}.stl"
if [[ "$NEEDS_BLENDER" == "true" ]]; then
    log "  Output:  $OUTPUT_DIR/${BASENAME}.${FORMAT}"
fi
log ""
log "=== Godot Import Instructions ==="
case "$FORMAT" in
    glb|gltf)
        log "  1. Copy ${BASENAME}.${FORMAT} into your Godot project's res:// folder"
        log "  2. Godot auto-imports glTF — the model appears in the FileSystem dock"
        log "  3. Drag it into your scene, or instance it via code:"
        log "     var scene = load(\"res://${BASENAME}.${FORMAT}\")"
        log "     var instance = scene.instantiate()"
        log "     add_child(instance)"
        ;;
    stl)
        log "  1. Copy ${BASENAME}.stl into your Godot project's res:// folder"
        log "  2. Use the Godot STL import plugin, or convert via Blender first"
        log "  Tip: Re-run with -f glb for native Godot support"
        ;;
    obj)
        log "  1. Copy ${BASENAME}.obj into your Godot project's res:// folder"
        log "  2. Godot auto-imports OBJ as a mesh resource"
        log "  3. Create a MeshInstance3D node and assign the mesh"
        ;;
esac
