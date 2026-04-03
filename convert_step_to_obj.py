r"""Convert STEP/STP files to OBJ using FreeCAD.
Run with: "C:\Program Files\FreeCAD 1.0\bin\FreeCADCmd.exe" convert_step_to_obj.py
"""
import sys
import os
import glob

# FreeCAD imports
import FreeCAD
import Part
import Mesh
import MeshPart

INPUT_DIR = r"C:\Users\harsh\projects\forks\PartField\data\step-200"
OUTPUT_DIR = r"C:\Users\harsh\projects\forks\PartField\data\step-200-obj"

os.makedirs(OUTPUT_DIR, exist_ok=True)

step_files = glob.glob(os.path.join(INPUT_DIR, "*.step")) + \
             glob.glob(os.path.join(INPUT_DIR, "*.STEP")) + \
             glob.glob(os.path.join(INPUT_DIR, "*.stp")) + \
             glob.glob(os.path.join(INPUT_DIR, "*.STP"))

# Deduplicate (case-insensitive glob on Windows may return same files)
seen = set()
unique_files = []
for f in step_files:
    normalized = os.path.normpath(f).lower()
    if normalized not in seen:
        seen.add(normalized)
        unique_files.append(f)

print(f"Found {len(unique_files)} STEP files to convert")

succeeded = 0
failed = 0
errors = []

for i, step_path in enumerate(sorted(unique_files), 1):
    basename = os.path.splitext(os.path.basename(step_path))[0]
    obj_path = os.path.join(OUTPUT_DIR, basename + ".obj")

    if os.path.exists(obj_path):
        print(f"[{i}/{len(unique_files)}] Skipping (already exists): {basename}.obj")
        succeeded += 1
        continue

    print(f"[{i}/{len(unique_files)}] Converting: {os.path.basename(step_path)} -> {basename}.obj")

    try:
        # Load STEP
        shape = Part.Shape()
        shape.read(os.path.abspath(step_path))

        # Tessellate to mesh (0.1mm linear deflection for good quality)
        mesh = MeshPart.meshFromShape(
            Shape=shape,
            LinearDeflection=0.1,
            AngularDeflection=0.5,
            Relative=False
        )

        if mesh.CountFacets == 0:
            raise RuntimeError("Tessellation produced 0 faces")

        print(f"  Vertices: {mesh.CountPoints}, Faces: {mesh.CountFacets}")

        # Export as OBJ
        mesh.write(os.path.abspath(obj_path))
        print(f"  Saved: {obj_path}")
        succeeded += 1

    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append((basename, str(e)))
        failed += 1

print(f"\n{'='*50}")
print(f"Done! Succeeded: {succeeded}/{len(unique_files)}, Failed: {failed}")
if errors:
    print(f"\nFailed files:")
    for name, err in errors:
        print(f"  {name}: {err}")
