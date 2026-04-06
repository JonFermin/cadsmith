"""
blender_process.py — Blender headless mesh processing for scad-to-godot pipeline

Called by scad-to-godot.sh via:
    blender --background --python blender_process.py -- [args]

Performs:
    1. Import STL/OBJ
    2. Clean up mesh (remove doubles, recalculate normals)
    3. Optional: decimate to reduce poly count
    4. Optional: generate UV maps (smart project)
    5. Export to glTF/glb, OBJ, or STL
"""

import bpy
import sys
import os
import argparse
import math


def parse_args():
    """Parse arguments after the '--' separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender mesh processor")
    parser.add_argument("--input", required=True, help="Input mesh file (STL/OBJ)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", default="glb", choices=["glb", "gltf", "obj", "stl"],
                        help="Output format")
    parser.add_argument("--decimate", type=float, default=1.0,
                        help="Decimate ratio (0.0-1.0, 1.0 = no decimation)")
    parser.add_argument("--uv", default="false", choices=["true", "false"],
                        help="Generate UV maps")
    return parser.parse_args(argv)


def clear_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(filepath):
    """Import STL or OBJ file."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    # Get the imported object
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj


def cleanup_mesh(obj):
    """Clean up mesh: merge by distance, recalculate normals, remove loose geometry."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bpy.ops.mesh.select_all(action="SELECT")

    # Merge vertices that are very close (remove doubles)
    bpy.ops.mesh.remove_doubles(threshold=0.0001)

    # Recalculate normals to face outward
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # Remove loose vertices/edges
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)

    bpy.ops.object.mode_set(mode="OBJECT")

    vert_count = len(obj.data.vertices)
    face_count = len(obj.data.polygons)
    print(f"[blender] Mesh cleaned: {vert_count} vertices, {face_count} faces")
    return vert_count, face_count


def decimate_mesh(obj, ratio):
    """Apply decimate modifier to reduce polygon count."""
    if ratio >= 1.0:
        return

    print(f"[blender] Decimating mesh to {ratio:.0%} of original")
    before = len(obj.data.polygons)

    mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod.ratio = ratio
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    after = len(obj.data.polygons)
    print(f"[blender] Decimated: {before} -> {after} faces ({after/before:.0%})")


def generate_uv(obj):
    """Generate UV maps using smart UV project."""
    print("[blender] Generating UV maps (Smart UV Project)")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    print("[blender] UV maps generated")


def apply_default_material(obj):
    """Apply a basic material so glTF export includes material data."""
    mat = bpy.data.materials.new(name="ScadForgeMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Neutral gray, slightly rough
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.65, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        bsdf.inputs["Metallic"].default_value = 0.0

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def center_origin(obj):
    """Set origin to geometry center and place at world origin."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)


def export_mesh(filepath, fmt):
    """Export mesh in the specified format."""
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)

    if fmt == "glb":
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            export_format="GLB",
            use_selection=True,
            export_apply=True,
        )
    elif fmt == "gltf":
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            export_format="GLTF_SEPARATE",
            use_selection=True,
            export_apply=True,
        )
    elif fmt == "obj":
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
        )
    elif fmt == "stl":
        bpy.ops.export_mesh.stl(
            filepath=filepath,
            use_selection=True,
        )

    print(f"[blender] Exported: {filepath}")


def main():
    args = parse_args()

    print(f"[blender] Input:  {args.input}")
    print(f"[blender] Output: {args.output}")
    print(f"[blender] Format: {args.format}")

    # Clear default scene
    clear_scene()

    # Import
    obj = import_mesh(args.input)
    print(f"[blender] Imported: {obj.name}")

    # Clean up
    cleanup_mesh(obj)

    # Decimate
    if args.decimate < 1.0:
        decimate_mesh(obj, args.decimate)

    # UV maps
    if args.uv == "true":
        generate_uv(obj)

    # Center at origin
    center_origin(obj)

    # Add default material for glTF
    if args.format in ("glb", "gltf"):
        apply_default_material(obj)

    # Select for export
    obj.select_set(True)

    # Export
    export_mesh(args.output, args.format)

    print("[blender] Done!")


if __name__ == "__main__":
    main()
